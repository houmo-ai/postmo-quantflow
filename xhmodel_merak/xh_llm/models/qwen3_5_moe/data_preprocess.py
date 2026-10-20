"""Device alignment owned by the Qwen MoE checkpoint consumer."""

from ..qwen3_5.data_preprocess import Qwen3_5_DataPreprocess as _Qwen35DataPreprocess


class Qwen3_5_DataPreprocess(_Qwen35DataPreprocess):  # noqa: N801
    def forward(self, data):
        # The HF embedding may stay on CPU while export inputs live on CUDA.
        # Preserve the caller's request and reuse the shared preprocessing math.
        if isinstance(data, dict) and data.get("input_ids") is not None:
            data = {**data, "input_ids": data["input_ids"].to(self.embed_tokens.weight.device)}
        return super().forward(data)


__all__ = ["Qwen3_5_DataPreprocess"]
