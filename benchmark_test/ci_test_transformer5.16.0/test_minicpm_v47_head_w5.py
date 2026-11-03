"""Consume a real packed W5 head and share its integer payload with DSpark."""

import json
from pathlib import Path

from _minicpm_v47_fixture import minicpmv47_modeling  # noqa: F401

import pytest
import torch
import yaml


@pytest.fixture(params=("fused", "defused"))
def checkpoint(tmp_path, request):
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
    from safetensors.torch import save_file
    from test_minicpm_v_4_7_auto_classes import _tiny_model
    from transformers.models.minicpmv4_7 import modeling_minicpmv4_7

    torch.manual_seed(47)
    host = _tiny_model(modeling_minicpmv4_7, moe=True).half()
    with torch.no_grad():
        for name, value in host.named_parameters():
            value.copy_(torch.ones_like(value) if "norm" in name else torch.randn_like(value) * .02)
    body = {"model." + key: value for key, value in host.model.language_model.state_dict().items()}
    signed = (torch.arange(128 * 128).reshape(128, 128) % 32 - 16).half()
    scales = torch.full((128, 2), 1 / 64, dtype=torch.float16)
    linear = torch.nn.Linear(128, 128, bias=False).half()
    linear.weight.data.copy_(signed / 64)
    head = TorchQuantLinear(bits=5, group_size=64, sym=True, desc_act=False,
                           in_features=128, out_features=128, bias=False)
    head.pack_original(linear, scales, torch.full_like(scales, 16), torch.arange(128, dtype=torch.int32) // 64)
    head.qzero_format(format=2)
    payload = {"lm_head." + key: value for key, value in head.state_dict().items()}
    serialized_body = dict(body)
    if request.param == "defused":
        for name in tuple(body):
            if not name.endswith(".mlp.experts.gate_up_proj"):
                continue
            prefix = name.removesuffix("gate_up_proj")
            gate_up = serialized_body.pop(name)
            down = serialized_body.pop(prefix + "down_proj")
            for expert in range(gate_up.shape[0]):
                gate, up = gate_up[expert].chunk(2, dim=0)
                serialized_body[prefix + f"{expert}.gate_proj.weight"] = gate.contiguous()
                serialized_body[prefix + f"{expert}.up_proj.weight"] = up.contiguous()
                serialized_body[prefix + f"{expert}.down_proj.weight"] = down[expert].contiguous()
    save_file(serialized_body, tmp_path / "model-backbone.safetensors")
    save_file(payload, tmp_path / "model-lm-head.safetensors")
    index = {**dict.fromkeys(serialized_body, "model-backbone.safetensors"),
             **dict.fromkeys(payload, "model-lm-head.safetensors")}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    config = host.config.text_config.to_dict()
    config["quantization_config"] = {
        "quant_method": "gptq", "checkpoint_format": "gptq_v2", "bits": 8,
        "group_size": 64, "sym": True, "desc_act": False, "lm_head": True,
        "dynamic": {"+:^lm_head$": {"bits": 5, "group_size": 64, "sym": True, "desc_act": False},
                    "-:^(?!lm_head$).*": {}},
        "meta": {"lm_head_quantization": {"bits": 5, "storage_bits": 8, "algorithm": "autoround",
                                           "data_type": "int_sym_xh"},
                 "float_qdq_backbone": {"model_family": "minicpm_v_4_7", "artifact_format": "float_causal_bin"}},
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return host, tmp_path, body, signed


def test_packed_head_consumer_preserves_backbone_and_integer_codes(checkpoint):
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.quant_llm import (
        apply_causal_dequant_to_minicpm_host, validate_causal_head_checkpoint,
    )

    host, root, body, signed = checkpoint
    validate_causal_head_checkpoint(root, 5)
    with pytest.raises(ValueError, match="already packed"):
        validate_causal_head_checkpoint(root, 4)
    apply_causal_dequant_to_minicpm_host(host, str(root))
    for key, value in host.model.language_model.state_dict().items():
        torch.testing.assert_close(value, body["model." + key], rtol=0, atol=0)
    torch.testing.assert_close(host.lm_head.quant_weight, signed.to(torch.int16), rtol=0, atol=0)
    torch.testing.assert_close(host.lm_head.weight, signed / 64, rtol=0, atol=0)


def test_target_and_draft_hmonnx_share_calibrated_w5_payload(checkpoint, tmp_path):
    import onnx
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.quant_llm import apply_causal_dequant_to_minicpm_host
    from xhmodel_merak.xh_llm.shared_head import bind_quantized_head, quantized_head
    from xhquant.api import (PrecisionMode, convert_quanted_model_to_hmonnx, ptq_quantize,
                             to_frontend_graph, to_quant_graph)

    host, root, _, signed = checkpoint
    apply_causal_dequant_to_minicpm_host(host, str(root))

    class Head(torch.nn.Module):
        def __init__(self, head):
            super().__init__()
            self.lm_head = head

        def forward(self, hidden):
            return self.lm_head(hidden)

    inputs = [torch.ones(1, 1, 128).half()]
    target = to_quant_graph(to_frontend_graph(Head(host.lm_head), "TorchFX", inputs),
                            "XH2a", {"quant_type": "w5a8h1_sefp"})
    ptq_quantize(target, [inputs], PrecisionMode.ALIGNED, ["cpu"], infer_shape=False)
    target.fixed()
    fixed = quantized_head(target)
    torch.testing.assert_close(fixed.qweight.reshape(128, 128).T, signed.to(torch.int8), rtol=0, atol=0)
    draft_inputs = [inputs[0].repeat(1, 3, 1)]
    draft = to_quant_graph(to_frontend_graph(Head(torch.nn.Linear(128, 128, bias=False).half()),
                                            "TorchFX", draft_inputs), "XH2a", {"quant_type": "w8a8h1_sefp"})
    bind_quantized_head(draft, fixed)
    ptq_quantize(draft, [draft_inputs], PrecisionMode.ALIGNED, ["cpu"], infer_shape=False)
    assert quantized_head(draft).qweight is fixed.qweight
    assert quantized_head(draft).scale_or_exp is fixed.scale_or_exp
    for name, model, values in (("target", target, inputs), ("draft", draft, draft_inputs)):
        output = tmp_path / f"{name}.onnx"
        convert_quanted_model_to_hmonnx(model, values, str(output))
        linear = next(node for node in onnx.load(output).graph.node if node.op_type == "Linear")
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in linear.attribute}
        assert attrs["hmfp_weight_man_bit"] == 5


@pytest.mark.parametrize("dspark", (False, True))
def test_w5_export_configs_reuse_checkpoint_and_keep_dynamic_vision(dspark):
    root = Path(__file__).resolve().parents[2]
    suffix = "_dspark" if dspark else ""
    path = root / ("configs_merak/workflows/xh2a/llm_models/minicpm_v_4_7/35b_a3b/"
                   f"minicpm_v_4_7_moe_xh2a_w4a8_autoround_head_w5{suffix}.yaml")
    config = yaml.safe_load(path.read_text())
    assert config["quant"]["method"] == "autoround"
    assert config["quant"]["head"] == {"bits": 5, "group_size": 64, "algorithm": "inherit"}
    assert config["export"]["model"]["quant_scheme"]["nodes"]["lm_head"]["quant_type"] == "w5a8h1_sefp"
    assert config["export"]["vision"]["patch_token_gears"] == [384, 784, 1536, 2816, 6144]
    model = config["export"]["model"]
    precision = model["flash_attention"]
    assert precision["enable"] is True
    assert set(precision) == {"enable", "q_bits", "k_bits", "v_bits", "p_bits"}
    assert all(precision[name] == 8 for name in ("q_bits", "k_bits", "v_bits", "p_bits"))
    if dspark:
        assert model["dspark_config"]["flash_attention"]["enable"] is False
        assert model["les_prediction_k"] == 0
    matmul = model["quant_scheme"]["ops"]["MatMul"]
    assert matmul["act_scheme"]["bits"] == matmul["act_schema_2"]["bits"] == 8


@pytest.mark.parametrize("bits,head", [(5, None), (5, {"bits": 4}), (8, {"bits": 5})])
def test_quant_head_mismatch_fails_before_loading_weights(bits, head):
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.quant_llm import validate_head_quant_config

    model = {"quant_scheme": {"quant_type": "w8a8h1_sefp", "nodes": {
        "lm_head": {"quant_type": f"w{bits}a8h1_sefp"}}}}
    with pytest.raises(ValueError, match="head"):
        validate_head_quant_config({"head": head} if head else {}, model)


def test_missing_calibrated_checkpoint_reports_resolved_path(tmp_path):
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.quant_llm import prepare_causal_head_checkpoint

    with pytest.raises(FileNotFoundError, match="saved checkpoint") as error:
        prepare_causal_head_checkpoint(tmp_path / "missing", bits=5,
                                       raw_model_dir="unused", quant_cfg={}, device="cpu")
    assert str(tmp_path / "missing" / "config.json") in str(error.value)


@pytest.mark.parametrize("algorithm", ["gptq", "autoround"])
def test_existing_qdq_completes_head_through_producer_and_loads_into_minicpm(tmp_path, algorithm):
    from transformers import Qwen2Tokenizer
    from transformers.models.minicpmv4_7 import modeling_minicpmv4_7
    from test_minicpm_v_4_7_auto_classes import _tiny_model
    from xhmodel_merak.xh_llm.models.minicpm_v_4_7.quant_llm import (
        apply_causal_dequant_to_minicpm_host, prepare_causal_head_checkpoint,
    )

    torch.manual_seed(47)
    torch.set_num_threads(2)
    host = _tiny_model(modeling_minicpmv4_7, moe=True).half()
    with torch.no_grad():
        for name, value in host.named_parameters():
            value.copy_(torch.ones_like(value) if 'norm' in name else torch.randn_like(value) * .02)
    body = {name: value.clone() for name, value in host.model.language_model.state_dict().items()}
    source = tmp_path / 'qdq'
    source.mkdir()
    state = {'model.' + name: value for name, value in body.items()}
    state['lm_head.weight'] = host.lm_head.weight.detach().clone()
    torch.save(state, source / 'pytorch_model.bin')
    (source / 'config.json').write_text(json.dumps(host.config.text_config.to_dict()))
    Qwen2Tokenizer(vocab={'<|endoftext|>': 0, 't': 1, 'Ġ': 2}, merges=[]).save_pretrained(source)
    data = tmp_path / 'calibration.jsonl'
    data.write_text('\n'.join(json.dumps({'text': 't ' * 32}) for _ in range(2)))
    with pytest.warns(UserWarning, match='saving new checkpoint'):
        target = prepare_causal_head_checkpoint(source, bits=5, raw_model_dir=str(source), device='cpu',
            quant_cfg={'head': {'bits': 5, 'algorithm': algorithm, 'iters': 2},
                       'calibration': {'jsonl': str(data), 'nsamples': 2, 'seqlen': 16}})
    assert Path(target) == tmp_path / 'qdq_head_w5'
    data.unlink()
    assert prepare_causal_head_checkpoint(target, bits=5, raw_model_dir=str(source),
                                          device='cpu', quant_cfg={}) == target
    apply_causal_dequant_to_minicpm_host(host, target)
    for name, value in host.model.language_model.state_dict().items():
        torch.testing.assert_close(value, body[name], rtol=0, atol=0)
    assert host.lm_head.quant_weight.min() >= -16
    assert host.lm_head.quant_weight.max() <= 15
