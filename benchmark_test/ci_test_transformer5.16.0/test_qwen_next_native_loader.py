"""Real tiny Next MoE weights: split checkpoint -> fused HF block, without quantizers."""
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextSparseMoeBlock

from xhmodel_merak.xh_llm.models.qwen3_next._qwen3_next_big_export import Qwen3NextBigHFModel


def _block(device="cpu", dtype=torch.float32):
    config = Qwen3NextConfig(
        hidden_size=8, moe_intermediate_size=4, shared_expert_intermediate_size=8,
        num_experts=12, num_experts_per_tok=2,
    )
    with torch.device(device):
        block = Qwen3NextSparseMoeBlock(config).to(dtype=dtype).eval()
    if device != "meta":
        with torch.no_grad():
            for index, parameter in enumerate(block.parameters()):
                values = torch.arange(parameter.numel()).reshape(parameter.shape)
                parameter.copy_(((values % 29) - 14 + index) / 64)
    return block


def _checkpoint(tmp_path, block, *, split=True, sharded=False, corrupt=None):
    prefix = "model.layers.0.mlp"
    tensors = {f"{prefix}.{name}": value.clone() for name, value in block.state_dict().items()}
    if split:
        gates = tensors.pop(f"{prefix}.experts.gate_up_proj")
        downs = tensors.pop(f"{prefix}.experts.down_proj")
        for index in range(gates.shape[0]):
            gate, up = gates[index].chunk(2, dim=0)
            for name, value in (("gate_proj", gate), ("up_proj", up), ("down_proj", downs[index])):
                tensors[f"{prefix}.experts.{index}.{name}.weight"] = value.clone()
    key = f"{prefix}.experts.10.up_proj.weight"
    if corrupt == "missing":
        tensors.pop(key)
    elif corrupt == "shape":
        tensors[key] = tensors[key][:-1]
    elif corrupt == "partial_fused":
        tensors[f"{prefix}.experts.gate_up_proj"] = block.experts.gate_up_proj.detach().clone()
    names = sorted(tensors, reverse=True)  # Do not depend on insertion/lexical expert ordering.
    chunks = [names[::2], names[1::2]] if sharded else [names]
    weight_map = {}
    for index, chunk in enumerate(chunks):
        filename = f"model-{index}.safetensors" if sharded else "model.safetensors"
        save_file({name: tensors[name] for name in chunk}, str(tmp_path / filename))
        weight_map.update({name: filename for name in chunk})
    if sharded:
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    loader = Qwen3NextBigHFModel.__new__(Qwen3NextBigHFModel)
    loader._hf_model_dir = str(tmp_path)
    loader._weight_mapping = SimpleNamespace(weight_map=loader._read_weight_map())
    return loader, prefix, len(tensors)


@pytest.mark.parametrize("split,sharded", [(True, False), (True, True), (False, True)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_native_expert_loading_matches_real_fused_forward(tmp_path, split, sharded, dtype):
    reference = _block(dtype=dtype)
    loader, prefix, keys = _checkpoint(tmp_path, reference, split=split, sharded=sharded)
    loaded, untouched = _block("meta", dtype), _block("meta", dtype)
    experts_type = type(loaded.experts)
    audit = loader._load_module_from_safetensor(loaded, prefix)
    assert audit["parameter_keys"] == keys
    assert audit["missing_parameter_keys"] == 0
    assert type(loaded.experts) is experts_type
    assert all(parameter.is_meta for parameter in untouched.parameters())
    for name, actual in loaded.state_dict().items():
        torch.testing.assert_close(actual, reference.state_dict()[name], rtol=0, atol=0)
    inputs = torch.linspace(-1, 1, 24, dtype=dtype).reshape(1, 3, 8)
    with torch.no_grad():
        torch.testing.assert_close(loaded(inputs), reference(inputs), rtol=0, atol=0)


@pytest.mark.parametrize("corrupt,match", [
    ("missing", "experts.10.up_proj.weight"),
    ("shape", "Shape mismatch"),
    ("partial_fused", "experts.down_proj"),
])
def test_native_expert_loading_rejects_incomplete_or_wrong_layout(tmp_path, corrupt, match):
    loader, prefix, _ = _checkpoint(tmp_path, _block(), split=True, sharded=True, corrupt=corrupt)
    loaded = _block("meta")
    original_experts = loaded.experts
    with pytest.raises(RuntimeError, match=match):
        loader._load_module_from_safetensor(loaded, prefix)
    assert loaded.experts is original_experts
    assert loaded.experts.gate_up_proj.is_meta
