"""Macaron HMONNX runtime specialization."""

from ..qwen3_5_moe.qwen3_5_moe_hmonnx_inference import (
    XHQwen3_5MoeHMONNXModel,
)


class XHMacaronV1TallHMONNXModel(XHQwen3_5MoeHMONNXModel):
    """Reuse the Qwen3.5-MoE runtime for the canonical exported graph."""

    def to_fast(self):
        raise RuntimeError(
            "Macaron compiler-bound LoRA uses aligned HMONNX Linear mode; "
            "fast mode is not validated for canonical LoRA initializers"
        )


__all__ = ["XHMacaronV1TallHMONNXModel"]
