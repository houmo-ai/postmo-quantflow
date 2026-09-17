"""WeMM-specific video token and position handling for HMONNX prefill."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from xhmodel_merak.xh_llm.models.qwen3_5.data_preprocess import Qwen3_5_DataPreprocess


class WeMMEmbeddingDataPreprocess(Qwen3_5_DataPreprocess):
    """Add video features while keeping the shared Qwen3.5 processor unchanged."""

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if video_grid_thw is not None:
            # Qwen3.5 places a timestamp and vision marker before each temporal
            # video unit, so RoPE consumes one grid row per unit.
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0).clone()
            video_grid_thw[:, 0] = 1
        return super().get_rope_index(input_ids, inputs_embeds, image_grid_thw, video_grid_thw, attention_mask)

    def forward(self, data: Mapping[str, Any]):
        video_embeds = data.get("video_embeds")
        if video_embeds is None:
            return super().forward(data)

        input_ids = data["input_ids"]
        video_embeds = video_embeds.reshape(-1, video_embeds.shape[-1])
        video_mask = input_ids == self.video_token_id
        video_token_count = int(video_mask.sum().item())
        if video_token_count != video_embeds.shape[0]:
            raise ValueError(
                "Video features and video tokens do not match: "
                f"tokens: {video_token_count}, features {video_embeds.shape[0]}"
            )

        # The shared processor handles image features, padding, masks, RoPE and
        # cache arguments. Replace only the video-token embeddings it returns.
        shared_data = dict(data)
        shared_data.pop("video_embeds")
        outputs = super().forward(shared_data)
        inputs_embeds = outputs[0].clone()
        video_mask = video_mask.to(inputs_embeds.device).unsqueeze(-1).expand(-1, -1, inputs_embeds.shape[-1])
        inputs_embeds[:, : input_ids.shape[1]] = inputs_embeds[:, : input_ids.shape[1]].masked_scatter(
            video_mask,
            video_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
        )
        return (inputs_embeds, *outputs[1:])


__all__ = ["WeMMEmbeddingDataPreprocess"]
