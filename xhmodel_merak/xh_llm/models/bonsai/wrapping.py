"""Instance-local Bonsai wrappers; never replace Qwen's global registrations."""

import torch.nn as nn

from xhquant.utils.registry import _DMRegistryCls

from ...register import XHLLM_TRACEABLE_MODULES
from ...wrap_model import _wrap_llm_model
from ..qwen3_5 import _llm_model_impl as qwen
from .gated_delta_net import BonsaiGatedDeltaNet
from .hadamard_activation import attach_hadamard_activations, load_phase_a_hadamard_aux


class _NormInInputDtype(nn.Module):
    def __init__(self, norm):
        super().__init__()
        self.norm = norm

    def forward(self, x):
        return self.norm(x).to(x.dtype)


class _SoftmaxInInputDtype(nn.Module):
    def __init__(self, softmax):
        super().__init__()
        self.softmax = softmax

    def forward(self, x, past_seq_length):
        return self.softmax(x, past_seq_length).to(x.dtype)


class BonsaiAttention(qwen._Qwen3_5TextAttention):
    def _setup(self, cfg):
        super()._setup(cfg)
        self.q_norm = _NormInInputDtype(self.q_norm)
        self.k_norm = _NormInInputDtype(self.k_norm)
        self.masked_softmax = _SoftmaxInInputDtype(self.masked_softmax)
        return self


class BonsaiLinear(nn.Module):
    """Trace the rotation and original Linear as separate ordinary operators."""

    def __init__(self, linear, rotation):
        super().__init__()
        self.linear = linear
        self.rotation = rotation

    @property
    def weight(self):
        return self.linear.weight

    def forward(self, x):
        return self.linear(self.rotation(x).to(self.linear.weight.dtype))


def build_bonsai_registry():
    qwen.register_wrap_modules()
    registry = _DMRegistryCls("BonsaiTrace")
    replacements = {
        qwen._Qwen3_5GatedDeltaNet: BonsaiGatedDeltaNet,
        qwen._Qwen3_5TextAttention: BonsaiAttention,
    }
    for hf_type, adapter in XHLLM_TRACEABLE_MODULES._registry.items():
        registry.register_module(
            {hf_type: XHLLM_TRACEABLE_MODULES._key_registry[hf_type]},
            replacements.get(adapter, adapter),
        )
    return registry


def wrap_bonsai_model(model, cfg):
    return _wrap_llm_model(model, cfg, registry=build_bonsai_registry())


def attach_bonsai_activations(model, aux_path):
    signs, forwards, block_size = load_phase_a_hadamard_aux(str(aux_path))
    present = {name: module for name, module in model.named_modules() if isinstance(module, nn.Linear)}
    selected = {}
    for name, tensor in signs.items():
        if name in present:
            selected[name] = tensor
        elif name.endswith(".in_proj_qkv"):
            prefix = name[:-len("in_proj_qkv")]
            for child in ("in_proj_q", "in_proj_k", "in_proj_v"):
                if prefix + child in present:
                    selected[prefix + child] = tensor
        elif not name.endswith(".embed_tokens"):
            raise KeyError(f"Bonsai signed projection is missing: {name}")
    widths = {present[name].in_features for name in selected}
    stats = attach_hadamard_activations(
        model, selected, {width: value for width, value in forwards.items() if width in widths},
        block_size=block_size,
    )
    if stats["missing_weight"]:
        raise ValueError("Bonsai aux lacks a required forward Hadamard bank")
    for name in selected:
        linear = present[name]
        rotation = linear.hadamard_activation
        del linear.hadamard_activation
        prefix, _, leaf = name.rpartition(".")
        parent = model.get_submodule(prefix) if prefix else model
        setattr(parent, leaf, BonsaiLinear(linear, rotation))
    return stats
