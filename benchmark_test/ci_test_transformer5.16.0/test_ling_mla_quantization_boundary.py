"""Real MLA weight absorption and GPTQ payload-consumption contracts (CPU)."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


class Config(dict):
    __getattr__ = dict.__getitem__


def _mla(gate):
    from xhmodel_merak.xh_llm.models.ling_3_flash._model import _LingMLAAttention

    torch.manual_seed(42)
    module = object.__new__(_LingMLAAttention)
    nn.Module.__init__(module)
    module.config = SimpleNamespace(hidden_size=64)
    module.hidden_size = 64
    module.num_heads = 2
    module.qk_nope_head_dim = 32
    module.qk_rope_head_dim = 32
    module.v_head_dim = 32
    module.kv_lora_rank = 64
    module.q_lora_rank = None
    module.scaling = 0.125
    module.q_proj = nn.Linear(64, 128, bias=False)
    module.kv_b_proj = nn.Linear(64, 128, bias=False)
    module.dense = nn.Linear(64, 64, bias=True)
    module.g_proj = None if gate is None else nn.Linear(64, 2 if gate == "head_wise" else 64, bias=False)
    module.gated_attention_proj_granularity_type = gate
    return module


def _wrap(module):
    module._setup(
        Config(
            use_cache=False,
            batch_size=1,
            input_sequence_length=3,
            enable_rope=False,
            flash_attention={"enable": False},
        )
    )
    return module


@pytest.mark.parametrize("gate", [None, "head_wise", "element_wise"])
def test_absorbed_mla_preserves_float_scores_and_gated_output(gate):
    source = _mla(gate)
    wrapped = _wrap(copy.deepcopy(source))
    hidden = torch.randn(1, 3, 64)
    latent = torch.randn(1, 3, 64)
    queries = source.q_proj(hidden).view(1, 3, 2, 64)[..., :32].transpose(1, 2)
    kv = source.kv_b_proj(latent).view(1, 3, 2, 64).transpose(1, 2)
    scores = queries @ kv[..., :32].transpose(-1, -2)
    absorbed_q = wrapped.q_absorbed_proj(hidden).view(1, 3, 2, 64).transpose(1, 2)
    absorbed_scores = absorbed_q @ latent.unsqueeze(1).transpose(-1, -2)
    torch.testing.assert_close(scores, absorbed_scores, atol=2e-5, rtol=1e-5)
    probabilities = (scores * source.scaling).softmax(-1)
    values = (probabilities @ kv[..., 32:]).transpose(1, 2)
    latent_context = (probabilities @ latent.unsqueeze(1)).transpose(1, 2)
    if gate is not None:
        gates = source.g_proj(hidden).sigmoid().view(1, 3, 2, -1)
        values = values * gates
        if gate == "head_wise":
            latent_context = latent_context * gates
    expected = source.dense(values.reshape(1, 3, 64))
    if gate == "element_wise":
        projected = (
            wrapped.value_up_conv(
                latent_context.permute(0, 2, 3, 1).reshape(1, 128, 3),
            )
            .view(1, 2, 32, 3)
            .permute(0, 3, 1, 2)
        )
        actual = wrapped.dense((projected * gates).reshape(1, 3, 64))
        assert not hasattr(wrapped, "absorbed_dense")
    else:
        actual = wrapped.absorbed_dense(latent_context.reshape(1, 3, 128))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)


def _packed_w8(linear):
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear

    from xhmodel_merak.xh_llm.base_llm_model import gptqmodel_torch_qlinear_converter

    groups = linear.weight.detach().reshape(-1, 64)
    scales = torch.maximum(-groups.min(-1).values / 128, groups.max(-1).values / 127).clamp_min(1e-5)
    scales = scales.reshape(linear.out_features, -1)
    packed = TorchQuantLinear(
        bits=8,
        group_size=64,
        sym=True,
        desc_act=False,
        in_features=linear.in_features,
        out_features=linear.out_features,
        bias=linear.bias is not None,
    )
    packed.pack(linear, scales, torch.full_like(scales, 128), torch.arange(linear.in_features, dtype=torch.int32) // 64)
    packed._init_wf_unsqueeze_buffers()
    packed.dequant_dtype = torch.int16
    gptqmodel_torch_qlinear_converter(packed)
    assert isinstance(packed, nn.Linear)
    assert hasattr(packed, "quant_weight")
    return packed.to(torch.float32)


def test_absorption_creates_new_matrices_without_reusable_integer_payload():
    source = _mla("head_wise")
    quantized = copy.deepcopy(source)
    for name in ("q_proj", "kv_b_proj", "dense"):
        setattr(quantized, name, _packed_w8(getattr(quantized, name)))
    quantized_fused = _wrap(quantized)
    float_fused = _wrap(source)
    for name in ("q_absorbed_proj", "q_rope_proj", "absorbed_dense"):
        layer = getattr(quantized_fused, name)
        assert not hasattr(layer, "quant_weight")
        assert not torch.equal(layer.weight, getattr(float_fused, name).weight)
    assert not hasattr(quantized_fused, "kv_b_proj")


@pytest.mark.parametrize("gate", [None, "head_wise"])
def test_preabsorbed_setup_preserves_linear_and_integer_payload(gate, monkeypatch):
    # Construct the deployed float boundaries, then use the real W8 pack/unpack
    # converter. This tests consumption only; training belongs to GPTQModel CI.
    attention = _wrap(_mla(gate))
    attention.config.ling_mla_layout = "xh2_absorbed_v1"
    expected = {}
    for name in ("q_absorbed_proj", "q_rope_proj", "absorbed_dense"):
        linear = _packed_w8(getattr(attention, name))
        setattr(attention, name, linear)
        expected[name] = (linear, linear.weight.clone(), linear.quant_weight.clone())

    def no_absorption(*args, **kwargs):
        raise AssertionError("Absorbed checkpoint must not multiply weights again")

    monkeypatch.setattr(torch, "einsum", no_absorption)
    _wrap(attention)
    for name, (linear, weight, integers) in expected.items():
        assert getattr(attention, name) is linear
        assert torch.equal(linear.weight, weight)
        assert torch.equal(linear.quant_weight, integers)
    assert attention.attention_scale.item() == attention.scaling


@pytest.mark.parametrize("broken", ["missing", "mixed", "version", "gate"])
def test_absorbed_checkpoint_structure_is_checked(broken):
    attention = _wrap(_mla("head_wise"))
    attention.config.ling_mla_layout = "xh2_absorbed_v1"
    if broken == "missing":
        del attention.q_absorbed_proj
    elif broken == "mixed":
        attention.kv_b_proj = nn.Linear(64, 128)
    elif broken == "version":
        attention.config.ling_mla_layout = "xh2_absorbed_v999"
    else:
        attention.gated_attention_proj_granularity_type = "element_wise"
    with pytest.raises(ValueError):
        _wrap(attention)


def test_absorbed_setup_keeps_two_independent_cache_operators():
    attention = _wrap(_mla("head_wise"))
    attention.config.ling_mla_layout = "xh2_absorbed_v1"
    attention._setup(
        Config(
            use_cache=True,
            kv_cache=SimpleNamespace(cache_axis=2),
            batch_size=1,
            input_sequence_length=3,
            enable_rope=False,
            flash_attention={"enable": False},
        )
    )
    assert attention.k_cache is not attention.v_cache


def test_sefp_equal_latents_need_different_exponent_domains():
    from xhquant.backend.hmops.sefp.sefp import sefp

    # Same latent, different reduction axes. An outlier in one token poisons
    # its channel across tokens for V, but its row across channels for K.
    latent = torch.ones(1, 1, 64, 64, dtype=torch.float16)
    latent[..., 0, 0] = 1024
    k, k_exp = sefp(latent.clone(), dim=-1, manbit=7, nshare=64, assemble=True)
    v, v_exp = sefp(latent.clone(), dim=-2, manbit=7, nshare=64, assemble=True)
    assert not torch.equal(k, v)
    assert k_exp.shape == (1, 1, 64, 1)
    assert v_exp.shape == (1, 1, 1, 64)
    assert k[..., 0, 1].item() != v[..., 0, 1].item()


def test_ssfp_unfold_keeps_absorbed_w8_codes():
    from xhquant.backend.xh2a.functions.te.linear import linear_unfold_weight_xh2a_default

    attention = _wrap(_mla("head_wise"))
    for name in ("q_absorbed_proj", "q_rope_proj", "absorbed_dense"):
        linear = _packed_w8(getattr(attention, name)).half()
        expected = linear.quant_weight.clone()
        linear.weight.quant_val = linear.quant_weight.clone()
        packed = linear_unfold_weight_xh2a_default(linear.weight, mode="ssfp", bit=8, nshare=64)
        integers, scale, bits = packed.unpack(is_scale=True)
        actual = integers.reshape(linear.in_features, linear.out_features).T
        assert bits == 8
        assert torch.equal(actual, expected)
        assert torch.isfinite(scale).all() and (scale > 0).all()


def _verify_ling_fixture(directory):
    manifest = json.loads((directory / "upstream.json").read_text(encoding="utf-8"))
    expected_sources = {"configuration_bailing_moe_v3.py", "modeling_bailing_moe_v3.py"}
    actual_sources = {path.name for path in directory.glob("*.py")}
    if set(manifest["sha256"]) != expected_sources or actual_sources != expected_sources:
        raise ValueError("Ling CI source manifest must cover exactly the two upstream Python files")
    for name, expected in manifest["sha256"].items():
        actual = hashlib.sha256((directory / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Ling CI source checksum mismatch: {name}")
    return directory


def _ling_fixture_dir():
    # Ling is checkpoint-side code, not a Transformers built-in. Keep the real
    # implementation in this CI fixture so neither a model mount nor HF access
    # is required. Verify it before trust_remote_code executes anything.
    return _verify_ling_fixture(Path(__file__).resolve().parent / "fixtures" / "ling3_flash")


@pytest.mark.parametrize("broken", ["checksum", "missing_entry", "extra_code"])
def test_ling_fixture_rejects_unpinned_code(tmp_path, broken):
    directory = tmp_path / "fixture"
    shutil.copytree(_ling_fixture_dir(), directory)
    if broken == "checksum":
        source = directory / "configuration_bailing_moe_v3.py"
        source.write_bytes(source.read_bytes() + b"\n# changed source\n")
    elif broken == "missing_entry":
        path = directory / "upstream.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        del manifest["sha256"]["configuration_bailing_moe_v3.py"]
        path.write_text(json.dumps(manifest), encoding="utf-8")
    else:
        (directory / "unlisted.py").write_text("# not in the manifest\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Ling CI source"):
        _verify_ling_fixture(directory)


def test_tiny_ling_fixture_is_independent_of_model_mount_and_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LING3_FLASH_MODEL_DIR", str(tmp_path / "absent_model"))
    directory = _ling_fixture_dir()
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    assert config["num_hidden_layers"] == config["layer_group_size"] == 1
    assert config["hidden_size"] == 64
    assert config["num_attention_heads"] == 32  # Keep the packed gate width.
    assert config["num_experts"] == 2
    assert config["first_k_dense_replace"] == 0
    assert config["gated_attention_proj_granularity_type"] == "head_wise"
    assert not any(path.suffix in {".safetensors", ".bin", ".pt"} for path in directory.iterdir())


def test_ling_loader_runs_without_fla(tmp_path):
    # A fresh process avoids inheriting FLA or HF remote-code modules from
    # other cases. Only dependency availability is changed, not the loaders.
    script = """
import importlib.abc
import importlib.util
import sys

class NoFLA:
    def __init__(self, finder):
        self.finder = finder

    def __getattr__(self, name):
        # Preserve distribution discovery and cache invalidation.
        return getattr(self.finder, name)

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "fla" or fullname.startswith("fla."):
            return None
        return self.finder.find_spec(fullname, path, target)

# Filter every finder so find_spec returns None, exactly as when uninstalled;
# raising in a finder would incorrectly break optional availability probes.
sys.meta_path[:] = [NoFLA(finder) for finder in sys.meta_path]
assert importlib.util.find_spec("fla") is None
import pytest
with pytest.raises(ModuleNotFoundError):
    importlib.import_module("fla")

status = pytest.main([
    "-c", "benchmark_test/ci_test_transformer5.16.0/pytest.ini",
    "--confcutdir=benchmark_test/ci_test_transformer5.16.0",
    sys.argv[1] + "::test_normal_and_streamed_load_keep_absorbed_payload",
    "-q", "-o", "addopts=", "--tb=short",
])
if status == 0:
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from xhmodel_merak.xh_llm.models.ling_3_flash._compat import load_ling_config

    kda = get_class_from_dynamic_module("modeling_bailing_moe_v3.BailingMoeV3KimiDeltaAttention", sys.argv[2])
    with pytest.raises(ImportError, match="Ling KDA layers require fla-core"):
        kda(load_ling_config(sys.argv[2]), layer_idx=0)
assert not any(name == "fla" or name.startswith("fla.") for name in sys.modules)
assert importlib.util.find_spec("fla") is None
raise SystemExit(status)
"""
    env = os.environ.copy()
    env.pop("LING3_FLASH_MODEL_DIR", None)
    env.update(
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HOME=str(tmp_path / "hf"),
        HF_MODULES_CACHE=str(tmp_path / "hf_modules"),
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(Path(__file__).resolve()), str(_ling_fixture_dir())],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(scope="module")
def absorbed_checkpoint(tmp_path_factory):
    """Real remote-code structure with tiny deterministic weights; no training."""
    from gptqmodel import QuantizeConfig
    from gptqmodel.models.definitions.ling3_flash_mla import absorb_ling_mla, finalize_ling_mla_checkpoint
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
    from gptqmodel.quantization.config import FORMAT
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

    from xhmodel_merak.xh_llm.models.ling_3_flash._compat import load_ling_config, patch_ling_remote_code_compatibility

    code_dir = _ling_fixture_dir()
    config = load_ling_config(str(code_dir))
    torch.manual_seed(31)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, attn_implementation="eager")
    patch_ling_remote_code_compatibility()
    assert len(model.model.layers) == 1
    assert [type(layer.attention).__name__ for layer in model.model.layers] == [
        "BailingMoeV3MultiLatentAttention"
    ]
    assert type(model.model.layers[0].mlp).__name__ == "BailingMoeV3SparseMoeBlock"
    assert len(model.model.layers[0].mlp.experts) == 2
    assert sum(parameter.numel() for parameter in model.parameters()) < 2_000_000
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor([[1, 4, 5]])).logits
    assert logits.shape == (1, 3, 128) and torch.isfinite(logits).all()
    absorb_ling_mla(model)
    model.half()
    attention = model.model.layers[0].attention
    for name in ("q_absorbed_proj", "q_rope_proj", "absorbed_dense"):
        linear = getattr(attention, name)
        groups = linear.weight.float().reshape(-1, 64)
        scales = torch.maximum(-groups.min(-1).values / 128, groups.max(-1).values / 127).clamp_min(1e-5)
        scales = scales.reshape(linear.out_features, -1)
        packed = TorchQuantLinear(
            bits=8,
            group_size=64,
            sym=True,
            desc_act=False,
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=linear.bias is not None,
        )
        packed.pack(
            linear, scales, torch.full_like(scales, 128), torch.arange(linear.in_features, dtype=torch.int32) // 64
        )
        setattr(attention, name, packed)
    model.config.quantization_config = QuantizeConfig(
        bits=8, group_size=64, sym=True, desc_act=False, format=FORMAT.GPTQ_V2
    ).to_dict()
    output = tmp_path_factory.mktemp("ling_absorbed_checkpoint")
    model.save_pretrained(output)
    PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({str(i): i for i in range(128)}, unk_token="3")),
        pad_token="0",
        bos_token="1",
        eos_token="2",
        unk_token="3",
    ).save_pretrained(output)
    finalize_ling_mla_checkpoint(output)
    return output


def test_normal_and_streamed_load_keep_absorbed_payload(absorbed_checkpoint):
    from safetensors.torch import load_file

    from xhmodel_merak.xh_llm.base_llm_model import _dequantize_gptqmodel_hf_model
    from xhmodel_merak.xh_llm.models.ling_3_flash._ling_3_flash_big_export import Ling3FlashBigHFModel
    from xhmodel_merak.xh_llm.models.ling_3_flash._model import _LingMLAAttention
    from xhmodel_merak.xh_llm.models.ling_3_flash.ling_3_flash_model import XHLing3FlashModel

    normal = XHLing3FlashModel.get_hf_model(str(absorbed_checkpoint))
    empty = XHLing3FlashModel.get_empty_hf_model(str(absorbed_checkpoint))
    assert empty.model.layers[0].attention.q_absorbed_proj.weight.is_meta
    Ling3FlashBigHFModel._preprocess_quantized_hf_model(empty, absorbed_checkpoint)
    state = load_file(absorbed_checkpoint / "model.safetensors")
    # Exercise the same selective materialization as the per-layer loader.
    attention = empty.model.layers[0].attention
    prefix = "model.layers.0.attention."
    state = {name[len(prefix) :]: value for name, value in state.items() if name.startswith(prefix)}
    attention.load_state_dict(state, assign=True, strict=True)
    Ling3FlashBigHFModel._prepare_loaded_gptqmodel_modules(attention, empty.config.quantization_config, empty.config)
    _dequantize_gptqmodel_hf_model(attention)
    reference = normal.model.layers[0].attention
    for name in ("q_absorbed_proj", "q_rope_proj", "absorbed_dense"):
        actual, expected = getattr(attention, name), getattr(reference, name)
        assert torch.equal(actual.weight, expected.weight)
        assert torch.equal(actual.quant_weight, expected.quant_weight)
    attention.__class__ = _LingMLAAttention
    _wrap(attention)
    assert not hasattr(attention, "kv_b_proj")
