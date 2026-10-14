"""Merak conversion configuration for DeepSeek-V4 Flash."""

from __future__ import annotations

import copy
from xhquant.api import QuantScheme

from ...types import TextLLMModelConfig


def resolve_deepseek_v4_pad_token_id(hf_config) -> int:
    """Resolve the static-padding token from an HF config or config-like object."""

    pad_token_id = getattr(hf_config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(hf_config, "eos_token_id", None)
    if isinstance(pad_token_id, (list, tuple)):
        pad_token_id = pad_token_id[0] if pad_token_id else None
    return 0 if pad_token_id is None else int(pad_token_id)


class XHDeepSeekV4ModelConfig(TextLLMModelConfig):
    """Fixed prefill-256 with decode-1 or DSpark K+1 verification.

    ``max_layers`` optionally selects a diagnostic prefix. Omitting it exports
    the released 43-block network. The cache layout itself is fixed when the
    model is wrapped and therefore cannot be enlarged at runtime.
    """

    def __init__(
        self,
        *,
        model_name: str,
        model_type: str = "DeepseekV4ForCausalLM",
        batch_size: int = 1,
        context_max_length: int = 256 * 1024,
        prefill_chunk_length: int = 256,
        num_logits_to_keep: int = 1,
        use_cache: bool = True,
        enable_auto_offload: bool = False,
        packed_weight_only: bool = True,
        max_layers: int | None = None,
        dspark: dict | None = None,
        graph_geometry: dict | None = None,
        **kwargs,
    ) -> None:
        if model_type != "DeepseekV4ForCausalLM":
            raise ValueError("DeepSeek-V4 Flash architecture must be DeepseekV4ForCausalLM")
        if int(batch_size) != 1:
            raise ValueError("the initial DeepSeek-V4 static runtime supports batch_size=1")
        if int(prefill_chunk_length) != 256:
            raise ValueError("DeepSeek-V4 export currently requires prefill_chunk_length=256")
        if int(context_max_length) <= 0 or int(context_max_length) > 256 * 1024:
            raise ValueError("context_max_length must be in [1, 262144]")
        if int(context_max_length) < 4 * 512:
            raise ValueError("context_max_length must provide at least 512 CSA entries")
        if dspark is not None and not isinstance(dspark, dict):
            raise TypeError("DeepSeek-V4 dspark configuration must be a mapping")
        dspark = copy.deepcopy(dspark or {})
        if dspark:
            num_draft_tokens = kwargs.pop("num_draft_tokens", None)
            if num_draft_tokens is not None:
                dspark["block_size"] = num_draft_tokens
            unknown = set(dspark) - {"model_dir", "block_size", "head_weight_bits", "target_layer_ids"}
            if unknown:
                raise ValueError(f"Unknown DSpark options: {sorted(unknown)}")
            dspark.setdefault("block_size", 5)
            dspark.setdefault("head_weight_bits", 5)
            dspark.setdefault("target_layer_ids", [40, 41, 42])
            if type(dspark["block_size"]) is not int or not 1 <= dspark["block_size"] < prefill_chunk_length:
                raise ValueError("DSpark block_size must be in [1,255]")
            if dspark["head_weight_bits"] not in (4, 5, 8):
                raise ValueError("DSpark shared head supports W4, W5 and W8")
            ids = dspark["target_layer_ids"]
            if not isinstance(ids, (list, tuple)) or len(ids) != 3 or len(set(ids)) != 3 or any(
                type(value) is not int or value < 0 or value >= (max_layers or 43) for value in ids
            ):
                raise ValueError("DSpark requires three distinct target feature layers within max_layers")
            num_logits_to_keep = dspark["block_size"] + 1
        elif int(num_logits_to_keep) != 1:
            raise ValueError("DeepSeek-V4 static export currently returns the last-token logit only")
        if not use_cache:
            raise ValueError("DeepSeek-V4 static export requires cache state")
        if max_layers is not None and not 1 <= int(max_layers) <= 43:
            raise ValueError("max_layers must be in [1, 43]")
        kwargs.pop("enable_prefill_chunk", None)
        kwargs.pop("max_pe_length", None)
        super().__init__(
            model_name=model_name,
            model_type=model_type,
            batch_size=1,
            context_max_length=int(context_max_length),
            prefill_chunk_length=256,
            num_logits_to_keep=int(num_logits_to_keep),
            use_cache=True,
            enable_auto_offload=bool(enable_auto_offload),
            enable_prefill_chunk=True,
            max_pe_length=int(context_max_length),
            max_layers=None if max_layers is None else int(max_layers),
            dspark=dspark,
            **kwargs,
        )
        self.packed_weight_only = bool(packed_weight_only)
        self.dspark = dspark
        allowed_geometry = {"hidden_size", "num_attention_heads", "head_dim", "q_lora_rank",
                            "sliding_window", "hc_mult", "hc_sinkhorn_iters", "index_topk", "index_head_dim"}
        if graph_geometry is not None and (not isinstance(graph_geometry, dict) or
                set(graph_geometry) - allowed_geometry or
                any(type(value) is not int or value <= 0 for value in graph_geometry.values())):
            raise ValueError("DeepSeek-V4 graph_geometry must contain positive declared dimensions")
        self.graph_geometry = copy.deepcopy(graph_geometry or {})
        self.num_draft_tokens = dspark.get("block_size")
        if dspark:
            scheme = copy.deepcopy(self.quant_scheme.to_dict() if hasattr(self.quant_scheme, "to_dict")
                                   else self.quant_scheme) or {"quant_type": "w8a8h1_sefp"}
            scheme.setdefault("nodes", {})["lm_head"] = {
                "quant_type": f"w{dspark['head_weight_bits']}a8h1_sefp"
            }
            self.quant_scheme = QuantScheme(**scheme)


__all__ = ["XHDeepSeekV4ModelConfig", "resolve_deepseek_v4_pad_token_id"]
