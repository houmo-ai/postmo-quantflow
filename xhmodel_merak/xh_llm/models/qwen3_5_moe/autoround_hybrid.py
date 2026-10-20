"""Preserve imported AutoRound payloads around the shared hybrid setup."""

import torch

from ..qwen3_5._hybrid_gated_delta_net import HybridGatedDeltaNetMixin, _split_linear_qkv_tensor
from ..qwen3_5._hybrid_moe import HybridSparseMoeMixin


class AutoRoundGatedDeltaNetMixin(HybridGatedDeltaNetMixin):
    def _setup(self, cfg=None):
        source = getattr(self, "in_proj_qkv", None)
        codes = getattr(source, "quant_weight", None)
        needs_split = (
            cfg.get("split_conv_cache", True)
            and not self._uses_packed_input_projections
            and not hasattr(self, "in_proj_q")
        )
        if codes is not None and needs_split:
            if tuple(codes.shape) != tuple(source.weight.shape):
                raise ValueError(
                    f"in_proj_qkv quant_weight shape {tuple(codes.shape)} does not match "
                    f"weight shape {tuple(source.weight.shape)}"
                )
        payload = None
        if codes is not None and needs_split:
            metadata = {
                name: getattr(source, name)
                for name in ("_autoround_bits", "_autoround_group_size", "_xhquant_weight_origin")
                if hasattr(source, name)
            }
            payload = getattr(source, "quant_scale", None), metadata
        # Do not retain the original floating QKV or its integer storage after
        # shared setup has split and released that projection.
        del source, codes
        result = super()._setup(cfg)
        if payload is not None:
            # Shared setup already splits and clones the exact integer codes.
            scale, metadata = payload
            scales = None if scale is None else _split_linear_qkv_tensor(scale, self.key_dim, self.value_dim, dim=1)
            for index, target in enumerate((self.in_proj_q, self.in_proj_k, self.in_proj_v)):
                target._non_persistent_buffers_set.add("quant_weight")
                if scales is not None:
                    target.register_buffer("quant_scale", scales[index].contiguous(), persistent=False)
                for name, value in metadata.items():
                    setattr(target, name, value)
                target._autoround_split_from = "in_proj_qkv"
        return result


class AutoRoundSparseMoeMixin(HybridSparseMoeMixin):
    def _setup(self, cfg=None):
        experts = self.experts
        result = super()._setup(cfg)
        if hasattr(experts, "gate_up_proj"):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                for suffix in ("quant_weight", "quant_scale"):
                    name = f"{projection}_{suffix}"
                    value = getattr(experts, name, None)
                    if value is not None:
                        setattr(self.moeblock, f"expert_{name}", value)
                        experts._buffers[name] = torch.empty_like(value, device="meta")
            if getattr(experts, "_xhquant_weight_origin", None) == "autoround-gptq-v2":
                self.moeblock._xhquant_weight_origin = "autoround-gptq-v2"
                self.moeblock._autoround_bits = getattr(experts, "_autoround_expert_bits", None)
                self.moeblock._autoround_group_size = getattr(experts, "_autoround_group_size", None)
        return result
