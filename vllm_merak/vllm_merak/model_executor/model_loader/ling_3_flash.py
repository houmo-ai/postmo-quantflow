# SPDX-License-Identifier: Apache-2.0
"""Ling 3 Flash loader for asymmetric absorbed-MLA paged K/V caches."""

from typing import Any

from ...constants import HYBRID_CACHE_POLICY_PAGED
from ..register import MERAK_MODEL_REGISTRY
from .qwen3_next import Qwen3NextMerakForCausalLM


@MERAK_MODEL_REGISTRY.register_module("BailingMoeV3ForCausalLM")
class Ling3FlashMerakForCausalLM(Qwen3NextMerakForCausalLM):
    """Ling hybrid runtime with scheduler-owned absorbed-MLA paged state."""

    hybrid_cache_policy = HYBRID_CACHE_POLICY_PAGED

    @classmethod
    def derive_missing_hybrid_layer_types(
        cls,
        hf_model_config: Any,
    ) -> list[str]:
        """Derive the legacy Bailing grouped Full/Linear layer pattern.

        Bailing's source decoder makes every final layer in a complete group a
        Full-Attention layer and treats all trailing partial-group layers as
        Full Attention. The remaining layers use Kimi Delta linear attention.
        """

        num_layers = getattr(hf_model_config, "num_hidden_layers", None)
        group_size = getattr(hf_model_config, "layer_group_size", None)
        if type(num_layers) is not int or num_layers <= 0:
            raise RuntimeError(
                "Ling/Bailing hybrid metadata requires a positive integer "
                f"num_hidden_layers, got {num_layers!r}."
            )
        if type(group_size) is not int or group_size <= 0:
            raise RuntimeError(
                "Ling/Bailing hybrid metadata requires a positive integer "
                f"layer_group_size, got {group_size!r}."
            )

        complete_group_end = num_layers // group_size * group_size
        return [
            (
                "full_attention"
                if (layer_idx + 1) % group_size == 0 or layer_idx >= complete_group_end
                else "linear_attention"
            )
            for layer_idx in range(num_layers)
        ]


__all__ = ["Ling3FlashMerakForCausalLM"]
