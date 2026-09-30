# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only WeMM-Embedding model."""

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VLVideoInputs
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
    Qwen3_5ProcessingInfo,
)
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLMultiModalProcessor,
)
from vllm.model_executor.models.vision import run_dp_sharded_mrope_vision_model
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)


WEMM_EMBEDDING_TOKEN = "<embedding>"


def _get_wemm_embedding_token_id(tokenizer: Any) -> int:
    token_ids = tokenizer.encode(
        WEMM_EMBEDDING_TOKEN,
        add_special_tokens=False,
    )
    if len(token_ids) != 1:
        raise ValueError(f"Expected {WEMM_EMBEDDING_TOKEN!r} to encode as one token, got {token_ids!r}")
    return int(token_ids[0])


def ensure_wemm_embedding_token(
    tokenizer: Any,
    prompt_token_ids: list[int],
) -> list[int]:
    """Return WeMM input IDs with exactly one final pooling anchor."""

    embedding_token_id = _get_wemm_embedding_token_id(tokenizer)
    without_anchor = [token_id for token_id in prompt_token_ids if token_id != embedding_token_id]
    return [*without_anchor, embedding_token_id]


def prepare_wemm_prompt_token_ids(
    tokenizer: Any,
    prompt: str,
    *,
    max_input_tokens: int | None = None,
) -> list[int]:
    """Tokenize a rendered WeMM prompt and preserve its pooling anchor."""

    if max_input_tokens is not None and max_input_tokens <= 0:
        raise ValueError("max_input_tokens must be positive")

    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    prompt_token_ids = ensure_wemm_embedding_token(tokenizer, prompt_token_ids)
    if max_input_tokens is not None and len(prompt_token_ids) > max_input_tokens:
        prompt_token_ids = [
            *prompt_token_ids[: max_input_tokens - 1],
            prompt_token_ids[-1],
        ]
    return prompt_token_ids


class WeMMProcessingInfo(Qwen3_5ProcessingInfo):
    """Qwen3.5 processing metadata used by WeMM-Embedding."""


class WeMMMultiModalProcessor(Qwen3VLMultiModalProcessor):
    """Apply WeMM's pooling-token contract to multimodal inputs."""

    def _apply_hf_processor_tokens_only(
        self,
        prompt_tokens: list[int],
    ) -> list[int]:
        return ensure_wemm_embedding_token(
            self.info.get_tokenizer(),
            prompt_tokens,
        )

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, Any],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        updates = super()._get_prompt_updates(
            mm_items,
            hf_processor_mm_kwargs,
            out_mm_kwargs,
        )
        hf_config = self.info.get_hf_config()
        vision_start_token_id = int(hf_config.vision_start_token_id)
        vision_end_token_id = int(hf_config.vision_end_token_id)

        wrapped_updates: list[PromptUpdate] = []
        for update in updates:
            if update.modality != "video" or not isinstance(update, PromptReplacement):
                wrapped_updates.append(update)
                continue

            base_replacement = update.replacement

            def replacement(
                item_idx: int,
                *,
                base_replacement: Any = base_replacement,
            ) -> PromptUpdateDetails[list[int]]:
                details = base_replacement(item_idx) if callable(base_replacement) else base_replacement
                if not isinstance(details, PromptUpdateDetails):
                    details = PromptUpdateDetails.from_seq(details)
                if not isinstance(details.full, list):
                    raise TypeError("WeMM video prompt replacement must use token IDs")
                return PromptUpdateDetails(
                    full=[
                        vision_start_token_id,
                        *details.full,
                        vision_end_token_id,
                    ],
                    is_embed=details.is_embed,
                )

            wrapped_updates.append(
                PromptReplacement(
                    modality=update.modality,
                    target=update.target,
                    replacement=replacement,
                )
            )
        return wrapped_updates


def _expand_video_grid_thw(video_grid_thw: torch.Tensor) -> torch.Tensor:
    """Expand each video grid into the frame bundles expected by WeMM.

    The official WeMM processor changes ``[T, H, W]`` into ``T`` rows of
    ``[1, H, W]`` after tokenization. The patch tensor stays concatenated, so
    the number and ordering of patches are unchanged.
    """
    if video_grid_thw.ndim != 2 or video_grid_thw.shape[1] != 3:
        raise ValueError(f"video_grid_thw must have shape [num_videos, 3], got {tuple(video_grid_thw.shape)}")
    temporal = video_grid_thw[:, 0].to(dtype=torch.long)
    if bool(torch.any(temporal < 1)):
        raise ValueError("video_grid_thw temporal dimensions must be positive")
    expanded = torch.repeat_interleave(video_grid_thw, temporal, dim=0)
    expanded[:, 0] = 1
    return expanded


@MULTIMODAL_REGISTRY.register_processor(
    WeMMMultiModalProcessor,
    info=WeMMProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
class WeMMEmbedding(Qwen3_5ForConditionalGeneration):
    """Qwen3.5 backbone with WeMM's frame-bundle video vision path."""

    def _process_video_input(self, video_input: Qwen2_5_VLVideoInputs) -> tuple[torch.Tensor, ...]:
        original_grid_thw = video_input["video_grid_thw"]
        assert original_grid_thw.ndim == 2

        if video_input["type"] == "video_embeds":
            video_embeds = video_input["video_embeds"].type(self.visual.dtype)
        else:
            pixel_values_videos = video_input["pixel_values_videos"].type(self.visual.dtype)
            expanded_grid_thw = _expand_video_grid_thw(original_grid_thw)
            if self.use_data_parallel:
                frame_embeds = run_dp_sharded_mrope_vision_model(
                    self.visual,
                    pixel_values_videos,
                    expanded_grid_thw.tolist(),
                    rope_type="rope_3d",
                )
                temporal_sizes = original_grid_thw[:, 0].tolist()
                grouped_embeds = []
                offset = 0
                for temporal_size in temporal_sizes:
                    next_offset = offset + int(temporal_size)
                    grouped_embeds.append(torch.cat(frame_embeds[offset:next_offset]))
                    offset = next_offset
                return tuple(grouped_embeds)
            video_embeds = self.visual(pixel_values_videos, grid_thw=expanded_grid_thw)

        merge_size = self.visual.spatial_merge_size
        sizes = (original_grid_thw.prod(-1) // merge_size // merge_size).tolist()
        return video_embeds.split(sizes)
