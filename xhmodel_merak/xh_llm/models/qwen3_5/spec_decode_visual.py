# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Request-local visual prefill for the shared dense/MoE speculative runtime."""

from pathlib import Path
from types import SimpleNamespace

import torch

from ...types import CacheList
from .data_preprocess import Qwen3_5_DataPreprocess


class SpecDecodeVisualMixin:
    def configure_multimodal(self, meta_info, model_dir, hf_model_dir):
        # Text requests do not materialize the visual sessions or processor.
        self._multimodal_meta = meta_info
        self._multimodal_model_dir = Path(model_dir)
        self.hf_model_dir = str(hf_model_dir)

    def _ensure_visual(self):
        if getattr(self, "visual", None) is not None:
            return self.visual
        from .qwen3_5_hmonnx_inference import VisualTokenGearHMONNXModel

        config = dict(getattr(self, "_multimodal_meta", {}).get("visual_config") or {})
        if not config.get("gears"):
            raise ValueError("Visual input requires an artifact with visual token gears")
        gears = []
        for raw in config["gears"]:
            gear = dict(raw)
            path = Path(gear["hmonnx"])
            gear["hmonnx"] = str(path if path.is_absolute() else self._multimodal_model_dir / path)
            gears.append(SimpleNamespace(**gear))
        config["gears"] = gears
        self.visual = VisualTokenGearHMONNXModel(SimpleNamespace(**config), device=self.device)
        return self.visual

    def get_tf_processor(self):
        from .qwen3_5_processor import XHQwen3_5Processor
        from .visual_token_gears import VISUAL_INPUT_PATCHES

        visual = self._ensure_visual()
        processor = XHQwen3_5Processor.from_pretrained(self.hf_model_dir)
        processor.config.patch_size = visual.visual_meta.patch_size
        processor.config.visual_input_mode = VISUAL_INPUT_PATCHES
        capacity = max(int(gear.patch_token_capacity) for gear in visual.visual_meta.gears)
        processor.config.max_pixels = capacity * int(visual.visual_meta.patch_size) ** 2
        return processor

    def _prepare_visual_prompt(
        self,
        input_ids,
        *,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        image_embeds=None,
        video_embeds=None,
        inputs_embeds=None,
    ):
        values = (
            pixel_values,
            pixel_values_videos,
            image_grid_thw,
            video_grid_thw,
            image_embeds,
            video_embeds,
            inputs_embeds,
        )
        if all(value is None for value in values):
            return None
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not input_ids.shape[1]:
            raise ValueError("Visual speculative input_ids must be nonempty [1, seq]")
        if self.max_context_tokens is not None and input_ids.shape[1] > self.max_context_tokens:
            raise ValueError("Visual prompt exceeds max_context_tokens; truncation would lose visual spans")
        config = getattr(self, "_multimodal_meta", {}).get("model_config") or {}
        if not config:
            raise ValueError("Visual input requires the artifact's model_config")
        data = dict(
            input_ids=input_ids, past_seq_length=0, image_grid_thw=image_grid_thw, video_grid_thw=video_grid_thw
        )
        for name, pixels, grid, features in (
            ("image", pixel_values, image_grid_thw, image_embeds),
            ("video", pixel_values_videos, video_grid_thw, video_embeds),
        ):
            if pixels is not None and features is not None:
                raise ValueError(f"Provide either {name} pixels or embeddings")
            if pixels is not None:
                if grid is None:
                    raise ValueError(f"{name} pixels require grid_thw")
                visual = self._ensure_visual()
                encoder = visual.encode_many if name == "image" else visual.encode_videos
                rows = encoder(pixels, grid)
                features = torch.cat([row.reshape(-1, row.shape[-1]) for row in rows], dim=0)
            if inputs_embeds is None and (features is None) != (grid is None):
                raise ValueError(f"{name} features and grid_thw must be provided together")
            if features is not None:
                if features.ndim != 2 or features.shape[-1] != self.token_embedding.weight.shape[-1]:
                    raise ValueError(f"{name} embeddings must be [tokens, hidden_size]")
                data[f"{name}_embeds"] = features
        preprocessor = Qwen3_5_DataPreprocess(
            token_embedding=self.token_embedding,
            input_sequence_length=input_ids.shape[1],
            past_key_caches=CacheList(),
            past_value_caches=CacheList(),
            past_conv_caches=CacheList(),
            past_recurrent_states=CacheList(),
            image_token_id=config.get("image_token_id", -1),
            video_token_id=config.get("video_token_id", -1),
            vision_start_token_id=config.get("vision_start_token_id", -1),
            vision_end_token_id=config.get("vision_end_token_id", -1),
            spatial_merge_size=config.get("spatial_merge_size", 2),
        ).to(self.device)
        prepared = preprocessor(data)
        fused = prepared[0]
        if inputs_embeds is not None:
            if (
                image_embeds is not None
                or video_embeds is not None
                or pixel_values is not None
                or pixel_values_videos is not None
            ):
                raise ValueError("inputs_embeds cannot be combined with visual features or pixels")
            if inputs_embeds.shape != fused.shape:
                raise ValueError(f"inputs_embeds must have shape {tuple(fused.shape)}")
            fused = inputs_embeds.to(fused)
        return SimpleNamespace(
            embeddings=fused, positions=torch.stack(prepared[1:4]), rope_delta=preprocessor.rope_deltas
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        tokenizer,
        max_new_tokens=256,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        presence_penalty=0.0,
        stream_output=False,
        return_stats=False,
        *,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        image_embeds=None,
        video_embeds=None,
        inputs_embeds=None,
    ):
        self._visual_prompt = None
        try:
            self._visual_prompt = self._prepare_visual_prompt(
                input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                image_embeds=image_embeds,
                video_embeds=video_embeds,
                inputs_embeds=inputs_embeds,
            )
            return self._generate(
                input_ids,
                tokenizer,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                stream_output=stream_output,
                return_stats=return_stats,
            )
        finally:
            self._visual_prompt = None

    @staticmethod
    def _set_position_feed(feed, positions):
        for name, axis in (
            ("time_position_ids", 0),
            ("hight_position_ids", 1),
            ("height_position_ids", 1),
            ("width_position_ids", 2),
        ):
            if name in feed:
                feed[name] = positions[axis].to(feed[name]).reshape_as(feed[name])

    def _build_prefill_feed(self, chunk_input_ids, valid_len, past_seq_len, cache_state):
        feed = super()._build_prefill_feed(chunk_input_ids, valid_len, past_seq_len, cache_state)
        prompt = getattr(self, "_visual_prompt", None)
        if prompt is None:
            return feed
        end = past_seq_len + valid_len
        # Retain existing padding, masks, physical lengths and cache objects.
        embeddings = feed[self._prefill_inputs_name].clone()
        embeddings[:, :valid_len] = prompt.embeddings[:, past_seq_len:end].to(embeddings)
        feed[self._prefill_inputs_name] = embeddings
        positions = prompt.positions[:, past_seq_len:end]
        pad_len = embeddings.shape[1] - valid_len
        if pad_len:
            padding = torch.arange(1, pad_len + 1, device=positions.device) + positions.max()
            positions = torch.cat((positions, padding.expand(3, -1)), dim=1)
        self._set_position_feed(feed, positions)
        return feed

    def _build_decode_feed(self, token_ids, past_seq_len, cache_state, current_input_length=None):
        feed = super()._build_decode_feed(token_ids, past_seq_len, cache_state, current_input_length)
        prompt = getattr(self, "_visual_prompt", None)
        if prompt is not None:
            length = feed[self._decode_inputs_name].shape[1]
            positions = torch.arange(length, device=self.device) + past_seq_len + prompt.rope_delta.reshape(())
            self._set_position_feed(feed, positions.expand(3, -1))
        return feed

    def _embed_mtp_prompt_tokens(self, token_ids, past_seq_len):
        prompt = getattr(self, "_visual_prompt", None)
        start = past_seq_len + 1  # MTP pairs hidden[i] with embedding[i+1].
        end = start + token_ids.shape[1]
        if prompt is not None and end <= prompt.embeddings.shape[1]:
            return prompt.embeddings[:, start:end].to(device=self.device, dtype=self._dtype)
        return self._embed_token_ids(token_ids)
