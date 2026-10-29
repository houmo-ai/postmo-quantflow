# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Bonsai artifact, numerical and ordinary-Qwen isolation contracts."""
import ast
import importlib.util
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
BONSAI = ROOT / 'xhmodel_merak/xh_llm/models/bonsai'


def _gdn_config(steps=4):
    from xhquant.api import ConfigDict
    return ConfigDict(use_cache=True, split_conv_cache=True, linear_attention_mode='chunk',
                      input_sequence_length=steps, linear_chunk_size=8, batch_size=1,
                      fuse_gdr_ops=False, fuse_gdr_block_recurrent_ops=False)


def _native_gdn():
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
    config = Qwen3_5TextConfig(hidden_size=16, linear_num_key_heads=1, linear_num_value_heads=3,
                              linear_key_head_dim=8, linear_value_head_dim=8, linear_conv_kernel_dim=4)
    model = Qwen3_5GatedDeltaNet(config, 0).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.linspace(-0.1, 0.1, parameter.numel()).reshape_as(parameter))
    return model


@pytest.mark.parametrize('logical_bits', [None, 4])
def test_split_qkv_inherits_integer_payload_and_declared_width(logical_bits):
    from xhmodel_merak.xh_llm.models.bonsai.wrapping import wrap_bonsai_model
    native = _native_gdn()
    codes = (torch.arange(native.in_proj_qkv.weight.numel()) % 3 - 1).to(torch.int8).reshape_as(native.in_proj_qkv.weight)
    native.in_proj_qkv.register_buffer('quant_weight', codes, persistent=False)
    if logical_bits is not None:
        native.in_proj_qkv.logical_bits = logical_bits
    wrapped = wrap_bonsai_model(native, _gdn_config())
    children = (wrapped.in_proj_q, wrapped.in_proj_k, wrapped.in_proj_v)
    torch.testing.assert_close(torch.cat([child.quant_weight for child in children]), codes)
    assert [child.out_features for child in children] == [8, 8, 24]
    assert all(getattr(child, 'logical_bits', None) == logical_bits for child in children)


def test_bonsai_wrapping_does_not_replace_qwen_registry_or_normalization():
    from xhmodel_merak.xh_llm.models.qwen3_5 import _llm_model_impl as qwen
    from xhmodel_merak.xh_llm.models.qwen3_5 import _delta_rule as ordinary_delta
    from xhmodel_merak.xh_llm.register import XHLLM_TRACEABLE_MODULES
    from xhmodel_merak.xh_llm.wrap_model import wrap_llm_model
    qwen.register_wrap_modules()
    registrations = dict(XHLLM_TRACEABLE_MODULES._registry)
    normalization = ordinary_delta.l2norm
    from xhmodel_merak.xh_llm.models.bonsai.wrapping import wrap_bonsai_model
    from xhmodel_merak.xh_llm.models.bonsai.gated_delta_net import BonsaiGatedDeltaNet
    bonsai = wrap_bonsai_model(_native_gdn(), _gdn_config())
    ordinary = wrap_llm_model(_native_gdn(), _gdn_config())
    assert isinstance(bonsai, BonsaiGatedDeltaNet)
    assert isinstance(ordinary, qwen._Qwen3_5GatedDeltaNet)
    assert not isinstance(ordinary, BonsaiGatedDeltaNet)
    assert XHLLM_TRACEABLE_MODULES._registry == registrations
    assert ordinary_delta.l2norm is normalization
    x = torch.tensor([[1e-4, -1e-4]])
    torch.testing.assert_close(ordinary_delta.l2norm(x), F.normalize(x, dim=-1, eps=1e-6))


def test_bonsai_mlp_rotation_uses_original_hf_forward_and_traces(tmp_path):
    from transformers import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5MLP
    from xhmodel_merak.xh_llm.models.bonsai.wrapping import attach_bonsai_activations, wrap_bonsai_model
    model = Qwen3_5MLP(Qwen3_5TextConfig(hidden_size=8, intermediate_size=16), 16)
    forward = type(model).forward
    model = wrap_bonsai_model(model, _gdn_config())
    h4 = torch.tensor([[1., 1., 1., 1.], [1., -1., 1., -1.],
                       [1., 1., -1., -1.], [1., -1., -1., 1.]]) / 2
    signs = {width: (torch.arange(width) % 2 * 2 - 1).float() for width in (8, 16)}
    weights = {width: torch.block_diag(*([h4] * (width // 4))) * signs[width] for width in signs}
    aux = {f'{name}.hadamard_signs': signs[width]
           for name, width in [('gate_proj', 8), ('up_proj', 8), ('down_proj', 16)]}
    aux.update({f'hadamard_matmul.forward.w{width}': value for width, value in weights.items()})
    torch.save(aux, tmp_path / 'phase_a_aux.pt')
    (tmp_path / 'phase_a_metadata.json').write_text(json.dumps({'hadamard': {'block_size': 4}}))
    attach_bonsai_activations(model, tmp_path / 'phase_a_aux.pt')
    assert type(model).forward is forward
    assert model.gate_proj.rotation is model.up_proj.rotation
    x = torch.randn(2, 3, 8, generator=torch.Generator().manual_seed(7))
    rotated = F.linear(x, weights[8])
    intermediate = model.act_fn(F.linear(rotated, model.gate_proj.weight)) * F.linear(rotated, model.up_proj.weight)
    expected = F.linear(F.linear(intermediate, weights[16]), model.down_proj.weight)
    torch.testing.assert_close(model(x), expected)
    torch.testing.assert_close(torch.fx.symbolic_trace(model)(x), expected)


def test_recipe_workflow_and_runtime_are_independent_of_qwen(tmp_path):
    from xhmodel_merak.model_catalog import get_recipe
    from xhmodel_merak.workflows import AutoWorkflow
    from xhmodel_merak.xh_llm.models.bonsai.model import XHBonsaiModel
    from xhmodel_merak.xh_llm.models.bonsai.workflow import BonsaiWorkflow
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import XHQwen3_5Model
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import XHQwen3_5_HMONNXModel
    recipe = get_recipe('bonsai-27b-phase-a')
    assert dict(recipe.source_constraints) == {'text_config.hidden_size': 5120, 'text_config.num_hidden_layers': 64}
    workflow = AutoWorkflow.from_config(str(tmp_path), str(ROOT / recipe.variants[0].workflow_config))
    assert isinstance(workflow, BonsaiWorkflow)
    assert issubclass(XHBonsaiModel, XHQwen3_5Model)
    before = getattr(XHQwen3_5_HMONNXModel, 'LLM_MODEL_CLS', None)
    runtime = XHBonsaiModel.get_hmonnx_inference_cls()
    assert runtime is not XHQwen3_5_HMONNXModel
    assert getattr(XHQwen3_5_HMONNXModel, 'LLM_MODEL_CLS', None) is before


@pytest.mark.parametrize('default_device', [
    'cpu',
    pytest.param('cuda:0', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required')),
])
def test_complete_phase_a_load_wrap_and_frontend_execution(tmp_path, default_device):
    from _unified_fixtures.bonsai import write_checkpoint
    from xhquant.api import FrontendType, to_frontend_graph
    from xhquant.core import CacheTensor
    from xhmodel_merak.xh_llm.models.bonsai.model import XHBonsaiModel, XHBonsaiModelConfig
    from xhmodel_merak.xh_llm.models.bonsai.wrapping import BonsaiLinear

    source = tmp_path / 'source'
    with torch.device(default_device):
        write_checkpoint(source)
        assert torch.empty(0).device == torch.device(default_device)
    fixture_path = Path(write_checkpoint.__code__.co_filename).resolve()
    for filename in ('quant_weight.pt', 'phase_a_aux.pt'):
        saved_locations = set()

        def record_location(storage, location):
            saved_locations.add(location)
            return None  # Preserve torch.load's normal device restoration.

        payload = torch.load(source / filename, weights_only=True, map_location=record_location)
        non_cpu = {name: str(tensor.device) for name, tensor in payload.items() if tensor.device.type != 'cpu'}
        diagnostic = f'{filename}: saved_locations={saved_locations}, non_cpu={non_cpu}, fixture={fixture_path}'
        assert saved_locations == {'cpu'}, diagnostic
        assert not non_cpu, diagnostic
    native = XHBonsaiModel.get_hf_model(source)
    assert native.lm_head.logical_bits == 4
    config = XHBonsaiModelConfig(
        model_name='bonsai_tiny', hf_model=str(source), context_max_length=32,
        prefill_chunk_length=8, max_pe_length=128, quant_scheme={'quant_type': 'w8a8h1_sefp'},
    )
    model = XHBonsaiModel(config)
    model._to_wrap(native)
    wrapped = model._wrap_model
    assert not any(value.is_meta for value in wrapped.parameters())
    assert isinstance(wrapped.lm_head, BonsaiLinear)
    assert wrapped.lm_head.linear.logical_bits == 4
    processor = model.get_data_preprocessor()
    assert processor.input_sequence_length == 8
    assert processor.image_token_id == -1
    generator = torch.Generator().manual_seed(7)
    args = (
        torch.randn(1, 8, 128, generator=generator).to(model._dtype),
        torch.arange(8), torch.arange(8), torch.arange(8),
        torch.tensor([0], dtype=torch.int32), torch.tensor([8], dtype=torch.int32),
        torch.ones(1, 8, dtype=model._dtype),
        [CacheTensor(torch.zeros(1, 1, 32, 64, dtype=model._dtype))],
        [CacheTensor(torch.zeros(1, 1, 32, 64, dtype=model._dtype))],
        [torch.zeros(1, channels, 4, dtype=model._dtype) for channels in (64, 64, 128)],
        [torch.zeros(1, 2, 64, 64, dtype=model._dtype)],
    )
    with torch.no_grad():
        expected = wrapped(*args)
        frontend = to_frontend_graph(wrapped, FrontendType.TorchFX, args).to(dtype=model._dtype)
        flat_args = (*args[:7], *args[7], *args[8], *args[9], *args[10])
        actual = frontend(*flat_args)
    assert expected[0].shape == (1, 1, 512)
    assert torch.isfinite(expected[0]).all()
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-3, atol=1e-3)


def test_public_bonsai_plan_consumes_phase_a_without_quantization(tmp_path, monkeypatch):
    from _unified_cases import CASES
    from _unified_fixtures.bonsai import prepare
    from xhmodel_merak.errors import ModelError
    from xhmodel_merak.model_build.resolver import resolve_build_plan

    case = next(case for case in CASES if case.model_id == 'bonsai-27b-phase-a')
    kwargs, _ = prepare(case, tmp_path, monkeypatch)
    plan = resolve_build_plan(**kwargs)
    assert plan.quant_method is None and plan.quant_checkpoint is None
    assert plan.model_dir == str(tmp_path / 'source')
    assert plan.precision.weight == 'wmix'
    assert dict(plan.deployment_precision) == {'prefill': 'w8a8h1_sefp', 'decode': 'w8a8h1_sefp'}
    config_path = tmp_path / 'source/config.json'
    config = json.loads(config_path.read_text())
    config['text_config']['hidden_size'] = 256
    config_path.write_text(json.dumps(config))
    with pytest.raises(ModelError):
        resolve_build_plan(**kwargs)


def load_file(path, name):
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def functions(path, names, namespace):
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def test_phase_a_loader_keeps_metadata_payload_width_through_quantization(tmp_path):
    from xhquant.quantization.xh2a._xh2a_quant_schema import LinearSchema
    from xhquant.quantization.xh2a.builder import XH2A_QMODULES

    model = nn.Sequential(nn.Linear(128, 4, bias=False, dtype=torch.float16))
    codes = (torch.arange(4 * 128) % 3 - 1).reshape(4, 128).to(torch.int8)
    weight = codes.half() * 0.03125
    checkpoint = tmp_path / 'quant_weight.pt'
    torch.save({'0.weight': weight, '0.quant_weight': codes}, checkpoint)
    (tmp_path / 'phase_a_metadata.json').write_text(json.dumps(dict(
        format='xh2_pq2_via_ssfp4_compat', version=2, ssfp={'man_bit': 4})))
    from xhmodel_merak.xh_llm.models.bonsai.checkpoint import load_phase_a_checkpoint
    torch.save({}, tmp_path / 'phase_a_aux.pt')
    load_phase_a_checkpoint(model, checkpoint, allow_partial=True)
    assert model[0].logical_bits == 4
    qlinear = XH2A_QMODULES[nn.Linear].convert(model[0], LinearSchema())
    qlinear.weight_static_quant()
    assert qlinear.w_cfg.qspec.man_bit == 4
    reconstructed = (qlinear.qweight.float() * qlinear.scale_or_exp).reshape(128, 4).T
    torch.testing.assert_close(reconstructed.half(), weight, rtol=0, atol=0)


def test_hadamard_rotation_is_an_fx_linear_leaf():
    mod = load_file(BONSAI / 'hadamard_activation.py', 'ci_hadamard_activation')
    weight = torch.tensor([[1., 1.], [1., -1.]]) / (2 ** 0.5)
    rotation = mod.HadamardActivation(weight)
    graph = torch.fx.symbolic_trace(rotation)
    assert any(node.op == 'call_module' and node.target == 'proj' for node in graph.graph.nodes)
    x = torch.tensor([[0.2, -0.4]])
    torch.testing.assert_close(graph(x), F.linear(x, weight))
    torch.testing.assert_close(rotation.proj.quant_weight, weight.sign().to(torch.int8))
    from xhquant.quantization.xh2a._xh2a_quant_schema import LinearSchema
    from xhquant.quantization.xh2a.builder import XH2A_QMODULES
    schema = LinearSchema()
    qlinear = XH2A_QMODULES[nn.Linear].convert(rotation.proj, schema)
    qlinear.weight_static_quant()
    assert qlinear.w_cfg.qspec.man_bit == 4


def test_hadamard_aux_preserves_block_size_and_shared_buffers(tmp_path):
    mod = load_file(BONSAI / 'hadamard_activation.py', 'ci_hadamard_aux')
    signs = torch.tensor([1., -1., 1., -1.])
    torch.save({'first.hadamard_signs': signs, 'second.hadamard_signs': signs,
                'hadamard_matmul.forward.w4': torch.eye(4)}, tmp_path / 'phase_a_aux.pt')
    (tmp_path / 'phase_a_metadata.json').write_text(json.dumps({'hadamard': {'block_size': 4}}))
    signed, weights, block = mod.load_phase_a_hadamard_aux(str(tmp_path / 'phase_a_aux.pt'))
    assert block == 4
    root = nn.Module()
    root.first, root.second = nn.Linear(4, 4), nn.Linear(4, 4)
    stats = mod.attach_hadamard_activations(root, signed, weights, block_size=block)
    root.to(dtype=torch.float16)
    bank = root.first.hadamard_activation
    assert bank is root.second.hadamard_activation
    assert stats['attached'] == 2
    assert bank.proj.quant_weight.dtype == torch.int8
    assert 'quant_weight' in bank.proj._buffers
    assert bank.proj.weight.dtype == torch.float16
    torch.testing.assert_close(bank(torch.ones(1, 4)), torch.ones(1, 4, dtype=torch.float16))


@pytest.mark.parametrize('failure', ['signs', 'shape', 'block'])
def test_hadamard_rejects_inconsistent_shared_bank(failure):
    mod = load_file(BONSAI / 'hadamard_activation.py', 'ci_hadamard_invalid')
    root = nn.ModuleDict({'first': nn.Linear(4, 4), 'second': nn.Linear(4, 4)})
    signs = {'first': torch.ones(4), 'second': torch.ones(4)}
    weights, block = {4: torch.eye(4)}, 4
    if failure == 'signs':
        signs['second'] = -signs['first']
    elif failure == 'shape':
        weights[4] = torch.eye(2)
    else:
        block = 0
    with pytest.raises(ValueError):
        mod.attach_hadamard_activations(root, signs, weights, block_size=block)


@pytest.mark.parametrize('failure', ['legacy', 'incomplete', 'missing_weight', 'missing_aux'])
def test_full_export_rejects_incomplete_phase_a(tmp_path, failure):
    model = nn.Module()
    model.model = nn.Module()
    model.model.language_model = nn.Linear(4, 4, bias=False)
    model.lm_head = nn.Linear(4, 8, bias=False)
    checkpoint = tmp_path / 'quant_weight.pt'
    state = model.state_dict()
    meta = dict(format='xh2_pq2_via_ssfp4_compat', version=2, complete_language_model=True,
                float_tensors_converted=True, embedding_inverse_applied=True,
                ssfp={'man_bit': 4}, hadamard={'runtime_activation_transform_required': True})
    if failure == 'legacy':
        meta['version'] = 1
    elif failure == 'incomplete':
        meta['complete_language_model'] = False
    elif failure == 'missing_weight':
        state.pop('model.language_model.weight')
    torch.save(state, checkpoint)
    (tmp_path / 'phase_a_metadata.json').write_text(json.dumps(meta))
    from xhmodel_merak.xh_llm.models.bonsai.checkpoint import load_phase_a_checkpoint
    if failure != 'missing_aux':
        torch.save({}, tmp_path / 'phase_a_aux.pt')
    error = FileNotFoundError if failure == 'missing_aux' else ValueError
    with pytest.raises(error):
        load_phase_a_checkpoint(model, checkpoint)


@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('amplitude', [1.0, 1e-4])
def test_delta_rule_matches_hf_across_chunk_boundary(dtype, amplitude):
    import transformers
    from transformers.utils import is_torchdynamo_exporting
    from xhmodel_merak.xh_llm.models.bonsai import delta_rule as local
    hf = functions(Path(transformers.__file__).parent / 'models/qwen3_5/modeling_qwen3_5.py',
                   ['l2norm', 'torch_chunk_gated_delta_rule', 'torch_recurrent_gated_delta_rule'],
                   {'torch': torch, 'F': F, 'is_torchdynamo_exporting': is_torchdynamo_exporting})
    generator = torch.Generator().manual_seed(7)
    length, heads, dim = 65, 2, 16
    q, k, v = [torch.randn(1, length, heads, dim, generator=generator).to(dtype) for _ in range(3)]
    args = dict(query=q * amplitude, key=k * amplitude, value=v,
                g=-torch.rand(1, length, heads, generator=generator).to(dtype),
                beta=torch.rand(1, length, heads, generator=generator).to(dtype),
                initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True)
    shared = dict(mask_qkv=torch.ones(1, length, 1, 1, dtype=dtype), num_heads=heads,
                  k_head_dim=dim, v_head_dim=dim, batch_size=1, scale=dim ** -0.5)
    chunk = local.torch_chunk_gated_delta_rule(
        **args, **shared, input_sequence_length=length,
        chunk_eye_matrix=torch.eye(64).expand(4, 64, 64),
        chunk_eye_8_batched=torch.eye(8).expand(32, 8, 8), chunk_row_masks=torch.zeros(64, 64))
    recurrent = local.torch_recurrent_gated_delta_rule(**args, **shared, sequence_length=length)
    rtol, atol = (1.4e-1, 3.4e-3) if dtype == torch.float16 else (1e-5, 1e-6)
    for actual, expected in [(chunk, hf['torch_chunk_gated_delta_rule'](**args)),
                             (recurrent, hf['torch_recurrent_gated_delta_rule'](**args))]:
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a.float(), b.float(), rtol=rtol, atol=atol)
