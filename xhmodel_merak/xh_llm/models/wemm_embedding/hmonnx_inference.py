"""HMONNX embedding runtime for WeMM-Embedding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import torch
import torch.nn.functional as F

from xhmodel_merak.xh_llm.infer_mixin import LLMInferenceContextManager
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import XHQwen3_5_HMONNXModel
from xhmodel_merak.xh_llm.models.qwen3_5.visual_token_gears import VISUAL_INPUT_PATCHES

from .data_preprocess import WeMMEmbeddingDataPreprocess
from .processor import XHWeMMEmbeddingProcessor, video_frame_grids


class XHWeMMEmbeddingHMONNXModel(XHQwen3_5_HMONNXModel):  # noqa: N801
    """Pool the last valid hidden state and L2-normalize it at runtime."""

    def _get_data_preprocessor(self) -> WeMMEmbeddingDataPreprocess:
        return WeMMEmbeddingDataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.get_input_sequence_length(),
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            past_conv_caches=self.past_conv_caches,
            past_recurrent_states=self.past_recurrent_states,
            patch_size=self.visual_meta.patch_size,
            image_token_id=self.meta_info.model_config.image_token_id,
            video_token_id=self.meta_info.model_config.video_token_id,
            vision_start_token_id=self.meta_info.model_config.vision_start_token_id,
            vision_end_token_id=self.meta_info.model_config.vision_end_token_id,
            spatial_merge_size=self.meta_info.model_config.spatial_merge_size,
            enable_page_attention=self.enable_page_attention,
        )

    def set_decode(self):
        raise RuntimeError("WeMM-Embedding runtime is prefill-only; Decode is not supported")

    def get_tf_processor(self):
        processor = XHWeMMEmbeddingProcessor.from_pretrained(self.hf_model_dir)
        visual_meta = self.visual_meta
        processor.config.patch_size = visual_meta.patch_size
        processor.config.visual_input_mode = VISUAL_INPUT_PATCHES
        max_patch_capacity = max(int(gear.patch_token_capacity) for gear in visual_meta.gears)
        processor.config.max_pixels = max_patch_capacity * int(visual_meta.patch_size) ** 2
        return processor

    @staticmethod
    def _flatten_visual_features(features: Sequence[torch.Tensor], hidden_size: int) -> torch.Tensor:
        if not features:
            raise ValueError("WeMM visual input produced no features")
        return torch.cat([feature.reshape(-1, hidden_size) for feature in features], dim=0)

    @staticmethod
    def _messages(item: Mapping[str, Any]) -> list[dict[str, Any]]:
        if item.get("messages") is not None:
            return deepcopy(item["messages"])
        content: list[dict[str, Any]] = []
        if item.get("image") is not None:
            image_content = {"type": "image", "image": item["image"]}
            for key in ("min_pixels", "max_pixels", "resized_height", "resized_width"):
                if item.get(key) is not None:
                    image_content[key] = int(item[key])
            content.append(image_content)
        if item.get("video") is not None:
            content.append({"type": "video", "video": item["video"]})
        if item.get("text") is not None:
            content.append({"type": "text", "text": str(item["text"])})
        if not content:
            raise ValueError("WeMM embedding input must contain text, image, or video")
        return [{"role": "user", "content": content}]

    @staticmethod
    def _pool_hidden_states(
        hidden_states: torch.Tensor,
        input_length: int,
        expected_dimension: int,
    ) -> torch.Tensor:
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("WeMM prefill must return a hidden-state tensor")
        if input_length <= 0:
            raise ValueError("WeMM pooling input length must be positive")
        if expected_dimension <= 0:
            raise ValueError("WeMM metadata embedding_dimension must be positive")
        if hidden_states.ndim != 3:
            raise ValueError(
                f"WeMM prefill must return [B, T, H] hidden states, got shape {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[0] <= 0 or hidden_states.shape[-1] <= 0:
            raise ValueError(f"WeMM prefill has an empty batch or embedding dimension: {tuple(hidden_states.shape)}")
        if hidden_states.shape[1] < input_length:
            raise ValueError(
                f"WeMM prefill output length {hidden_states.shape[1]} is shorter "
                f"than the processed input length {input_length}"
            )
        if hidden_states.shape[-1] != expected_dimension:
            raise ValueError(
                f"WeMM prefill output dimension {hidden_states.shape[-1]} does "
                f"not match metadata embedding_dimension {expected_dimension}"
            )
        pooled = hidden_states[:, input_length - 1, :].float()
        return F.normalize(pooled, p=2, dim=-1)

    @staticmethod
    def _validate_input_length(input_length: int, context_length: int) -> None:
        if input_length <= 0:
            raise ValueError("Processed WeMM embedding input must contain at least one token")
        if input_length > context_length:
            raise ValueError(
                f"Processed input length {input_length} exceeds the exported context length {context_length}"
            )

    def _run_multi_chunk_prefill(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        image_embeds: torch.Tensor | None,
        video_embeds: torch.Tensor | None,
        prefill_length: int,
        input_length: int,
    ) -> tuple[torch.Tensor, int]:
        if input_ids.shape[0] != 1:
            raise ValueError("WeMM multi-chunk embedding requires batch size 1")
        if input_length > input_ids.shape[1]:
            raise ValueError(f"Processed input length {input_length} exceeds input_ids width {input_ids.shape[1]}")

        input_ids = input_ids[:, :input_length].to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask[:, :input_length].to(self.device)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(self.device)
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw.to(self.device)

        data_processor = self.get_data_preprocessor()
        inputs_embeds = self.embed_tokens(input_ids)
        if image_embeds is not None:
            image_embeds = image_embeds.reshape(-1, image_embeds.shape[-1])
            image_token_id = int(data_processor.image_token_id)
            image_token_count = int((input_ids == image_token_id).sum().item())
            if image_token_count != image_embeds.shape[0]:
                raise ValueError(
                    "Image features and image tokens do not match: "
                    f"tokens={image_token_count}, features={image_embeds.shape[0]}"
                )
            image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask,
                image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
            )
        if video_embeds is not None:
            video_embeds = video_embeds.reshape(-1, video_embeds.shape[-1])
            video_token_id = int(data_processor.video_token_id)
            video_token_count = int((input_ids == video_token_id).sum().item())
            if video_token_count != video_embeds.shape[0]:
                raise ValueError(
                    "Video features and video tokens do not match: "
                    f"tokens={video_token_count}, features={video_embeds.shape[0]}"
                )
            video_mask = (input_ids == video_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask,
                video_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype),
            )

        position_ids, rope_deltas = data_processor.get_rope_index(
            input_ids,
            inputs_embeds,
            image_grid_thw,
            video_grid_thw,
            attention_mask,
        )
        data_processor.rope_deltas = rope_deltas
        position_ids = position_ids.to(inputs_embeds.device)

        steps = (input_length + prefill_length - 1) // prefill_length
        padding_length = steps * prefill_length - input_length
        if padding_length > 0:
            padding_input_ids = torch.full(
                (1, padding_length),
                int(data_processor.pad_token_id),
                dtype=torch.long,
                device=inputs_embeds.device,
            )
            padding_embeds = self.embed_tokens(padding_input_ids)
            inputs_embeds = torch.cat([inputs_embeds, padding_embeds], dim=1)
            padding_positions = position_ids[:, :, -1:].expand(-1, -1, padding_length)
            position_ids = torch.cat([position_ids, padding_positions], dim=2)

        past_key_caches = data_processor.past_key_caches
        past_value_caches = data_processor.past_value_caches
        past_conv_caches = data_processor.past_conv_caches
        past_recurrent_states = data_processor.past_recurrent_states
        final_hidden_states = None
        final_chunk_length = 0
        running_past_length = 0

        for step_index in range(steps):
            start = step_index * prefill_length
            end = (step_index + 1) * prefill_length
            current_length = min(end, input_length) - start
            linear_attention_mask = (
                torch.cat(
                    [
                        torch.ones(current_length, device=inputs_embeds.device),
                        torch.zeros(prefill_length - current_length, device=inputs_embeds.device),
                    ]
                )
                .unsqueeze(0)
                .to(dtype=torch.float16)
            )
            chunk_output = self.forward(
                inputs_embeds[:, start:end, :],
                position_ids[0, 0, start:end].to(torch.int64),
                position_ids[1, 0, start:end].to(torch.int64),
                position_ids[2, 0, start:end].to(torch.int64),
                torch.tensor([running_past_length], dtype=torch.int32, device=inputs_embeds.device),
                torch.tensor([current_length], dtype=torch.int32, device=inputs_embeds.device),
                linear_attention_mask,
                past_key_caches,
                past_value_caches,
                past_conv_caches,
                past_recurrent_states,
            )
            final_hidden_states = chunk_output[0] if isinstance(chunk_output, (tuple, list)) else chunk_output
            if not isinstance(final_hidden_states, torch.Tensor):
                raise TypeError("WeMM prefill must return a hidden-state tensor")
            final_chunk_length = current_length
            running_past_length += current_length

        if final_hidden_states is None:
            raise RuntimeError("WeMM multi-chunk prefill did not execute any chunk")
        return final_hidden_states, final_chunk_length

    def embed(self, item: Mapping[str, Any], processor: Any = None) -> torch.Tensor:
        if not isinstance(item, Mapping):
            raise TypeError("WeMM embedding input must be a mapping")
        # ``from_pretrained`` selects the runtime device before constructing
        # the standalone token embedding; make the device placement explicit
        # before the input processor performs its embedding lookup.
        self.to(self.device)
        self.embed_tokens.to(self.device)
        processor = processor or self.get_tf_processor()
        model_inputs = processor.apply_chat_template(
            self._messages(item),
            add_generation_prompt=False,
        )
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        image_grid_thw = model_inputs.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(self.device)
        video_grid_thw = model_inputs.get("video_grid_thw")
        if video_grid_thw is not None:
            video_grid_thw = video_grid_thw.to(self.device)
        input_length = int(attention_mask.sum().item()) if attention_mask is not None else int(input_ids.shape[-1])
        prefill_length = int(self.meta_info.model_config.prefill_chunk_length)
        context_length = int(self.meta_info.model_config.context_max_length)
        self._validate_input_length(input_length, context_length)

        image_embeds = None
        video_embeds = None
        pixel_values = model_inputs.get("pixel_values")
        video_values = model_inputs.get("pixel_values_videos")
        if pixel_values is not None and image_grid_thw is None:
            raise ValueError("Image processor did not provide image_grid_thw")
        if video_grid_thw is not None and video_values is None:
            raise ValueError("Video processor did not provide video patch inputs")
        if video_values is not None and video_grid_thw is None:
            raise ValueError("Video processor did not provide video_grid_thw")
        with LLMInferenceContextManager(self, devices=[self.device]):
            self.set_prefill()
            if pixel_values is not None:
                image_embeds = self._flatten_visual_features(
                    self.visual.encode_many(pixel_values, image_grid_thw),
                    int(self.meta_info.embedding_dimension),
                )
            if video_values is not None:
                frame_grids = video_frame_grids(
                    video_values,
                    video_grid_thw,
                    int(self.meta_info.model_config.spatial_merge_size),
                )
                video_embeds = self._flatten_visual_features(
                    self.visual.encode_many(video_values, frame_grids),
                    int(self.meta_info.embedding_dimension),
                )

            data = {
                "input_ids": input_ids,
                "past_seq_length": 0,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
                "image_embeds": image_embeds,
                "video_embeds": video_embeds,
            }
            if input_length <= prefill_length:
                model_output = self.forward(*self.get_data_preprocessor()(data))
                hidden_states = model_output[0] if isinstance(model_output, (tuple, list)) else model_output
                pooled_input_length = input_length
            else:
                hidden_states, pooled_input_length = self._run_multi_chunk_prefill(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    image_embeds=image_embeds,
                    video_embeds=video_embeds,
                    prefill_length=prefill_length,
                    input_length=input_length,
                )
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("WeMM prefill must return a hidden-state tensor")
        expected_dimension = int(self.meta_info.embedding_dimension)
        return self._pool_hidden_states(hidden_states, pooled_input_length, expected_dimension)

    def embed_items(self, items: Sequence[Mapping[str, Any]], processor: Any = None) -> torch.Tensor:
        if not items:
            raise ValueError("At least one WeMM embedding input is required")
        processor = processor or self.get_tf_processor()
        return torch.cat([self.embed(item, processor=processor) for item in items], dim=0)

    def embed_texts(self, texts: Sequence[str], processor: Any = None) -> torch.Tensor:
        if not texts:
            raise ValueError("At least one WeMM text input is required")
        return self.embed_items(
            [{"text": text} for text in texts],
            processor=processor,
        )

    def embed_images(self, images: Sequence[Any], processor: Any = None) -> torch.Tensor:
        if not images:
            raise ValueError("At least one WeMM image input is required")
        return self.embed_items(
            [{"image": image} for image in images],
            processor=processor,
        )


__all__ = ["XHWeMMEmbeddingHMONNXModel"]
