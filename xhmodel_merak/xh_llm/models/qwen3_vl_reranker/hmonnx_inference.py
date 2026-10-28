"""Real HMONNX visual + chunked Prefill execution for relevance scores."""

from collections.abc import Sequence

import torch

from xhmodel_merak.xh_llm.infer_mixin import LLMInferenceContextManager
from xhmodel_merak.xh_llm.models.qwen3_vl_embedding.hmonnx_inference import XHQwen3VLEmbeddingHMONNXModel

from .processor import DEFAULT_INSTRUCTION, RerankerProcessor, build_messages, merge_visual_features, prefill_chunks


class RerankerScoringMixin:
    def get_tf_processor(self):
        config = self.reranker_config.visual_config
        return RerankerProcessor(
            self.hf_model_dir,
            width=getattr(config, "max_size_w", None),
            height=getattr(config, "max_size_h", None),
            patch_size=config.patch_size,
            image_token_capacity=getattr(config, "image_token_capacity", None),
            spatial_merge_size=getattr(config, "spatial_merge_size", 2),
            context_length=int(self.reranker_config.context_max_length),
        )

    def _extract_visual_features(self, model_inputs):
        if getattr(self.reranker_config.visual_config, "visual_input_mode", None) == "patches":
            return self._extract_token_visual_features(model_inputs)
        pixels = model_inputs.get("hm_pixel_values")
        if pixels is None:
            return None, None
        pixels = [pixels] if isinstance(pixels, torch.Tensor) else pixels
        groups = []
        for pixel in pixels:
            groups.append(self.visual(pixel.to(device=self.device, dtype=self.dtype)))
            if getattr(self, "enable_golden", False):
                self.visual.update_step()
        return merge_visual_features(groups)

    def _extract_token_visual_features(self, model_inputs):
        grids = model_inputs.get("image_grid_thw")
        if grids is None:
            return None, None
        from xhmodel_merak.xh_llm.models.qwen3_5.visual_token_gears import (
            build_visual_token_gear_inputs,
            pad_flattened_patches,
        )

        groups = []
        sizes = [int(grid.prod()) for grid in grids]
        for pixels, grid in zip(torch.split(model_inputs["pixel_values"], sizes), grids, strict=True):
            if hasattr(self.visual, "encode"):
                groups.append(self.visual.encode(pixels, grid.reshape(1, 3)))
                continue
            config = self.visual.config
            capacity = config.image_token_capacity * config.spatial_merge_size**2
            padded, valid = pad_flattened_patches(pixels, capacity)
            geometry = build_visual_token_gear_inputs(
                grid,
                patch_capacity=capacity,
                num_position_embeddings=self.visual._vision_config.num_position_embeddings,
                spatial_merge_size=config.spatial_merge_size,
                dtype=self.dtype,
                device=self.device,
                rotary_cache_length=config.visual_rope_cache_length,
            )
            args = [padded.to(self.device, self.dtype)] + [
                geometry[name] for name in ("position_ids", "position_weights", "rotary_position_ids", "attention_mask")
            ]
            image, deepstack = self.visual(*args)
            valid //= config.spatial_merge_size**2
            groups.append((image[:, :valid], [feature[:, :valid] for feature in deepstack]))
        return merge_visual_features(groups)

    @torch.inference_mode()
    def score_pair(self, query, document, *, instruction=DEFAULT_INSTRUCTION, processor=None):
        messages = build_messages(query, document, instruction)
        processor = processor or self.get_tf_processor()
        model_inputs = processor.apply_chat_template(messages)
        length = int(model_inputs["input_ids"].shape[1])
        context_length = int(self.reranker_config.context_max_length)
        if length > context_length:
            raise ValueError(f"Processed pair length {length} exceeds exported context capacity {context_length}")
        self.set_prefill()
        with LLMInferenceContextManager(self, devices=[self.device]):
            image_embeds, deepstack = self._extract_visual_features(model_inputs)
            # Compute global media positions before splitting, including images crossing chunks.
            data_processor = self._get_data_preprocessor()
            data_processor.to(self.device, self.dtype)
            data_processor.input_sequence_length = length
            full_inputs = data_processor(
                {
                    "input_ids": model_inputs["input_ids"],
                    "past_seq_length": 0,
                    "image_grid_thw": model_inputs.get("image_grid_thw"),
                    "image_embeds": image_embeds,
                    "deepstack_image_embeds": deepstack,
                }
            )
            for chunk in prefill_chunks(
                full_inputs, chunk_length=int(self.reranker_config.prefill_chunk_length), context_length=context_length
            ):
                logits = self.forward(*chunk)
                if getattr(self, "enable_golden", False):
                    self.prefill_model.update_step()
            if not isinstance(logits, torch.Tensor) or logits.shape != (1, 1, 64):
                raise ValueError("Reranker Prefill must return score_logits with shape [1, 1, 64]")
            score = torch.sigmoid(logits[0, 0, 0].float())
            if not bool(torch.isfinite(score)):
                raise ValueError("Reranker produced a non-finite score")
            return float(score.cpu())

    def rerank(self, query, documents, *, instruction=DEFAULT_INSTRUCTION):
        if not isinstance(documents, Sequence) or isinstance(documents, (str, bytes)) or not documents:
            raise ValueError("documents must be a non-empty sequence")
        processor = self.get_tf_processor()
        return [
            self.score_pair(query, document, instruction=instruction, processor=processor) for document in documents
        ]

    def embed(self, *args, **kwargs):
        raise TypeError("Reranker returns relevance scores; use score_pair() or rerank()")


class XHQwen3VLRerankerHMONNXModel(RerankerScoringMixin, XHQwen3VLEmbeddingHMONNXModel):
    def _build_visual_runtime(self, *, enable_auto_offload, enable_golden):
        if getattr(self.visual_meta, "visual_input_mode", None) == "patches":
            from .visual import RerankerTokenVisualRuntime

            return RerankerTokenVisualRuntime(
                self.visual_meta, device=self._valid_devices[0], enable_golden=enable_golden
            )
        return super()._build_visual_runtime(enable_auto_offload=enable_auto_offload, enable_golden=enable_golden)

    def __init__(self, *args, **kwargs):
        from xhmodel_merak._environment import scoped_environment

        with scoped_environment({"ENABLE_HMINFERENCE_V2": "1"}):
            super().__init__(*args, **kwargs)
        self.enable_golden = bool(kwargs.get("enable_golden", False))

    @property
    def reranker_config(self):
        return self.meta_info.model_config

    def _get_data_preprocessor(self):
        from xhmodel_merak.xh_llm.models.qwen3_vl.data_preprocess import Qwen3VLDataPreprocess

        return Qwen3VLDataPreprocess(
            token_embedding=self.get_input_embeddings(),
            input_sequence_length=self.get_input_sequence_length(),
            image_size_w=getattr(self.visual_meta, "image_size_w", 32),
            image_size_h=getattr(self.visual_meta, "image_size_h", 32),
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            patch_size=self.visual_meta.patch_size,
            image_token_id=self.meta_info.image_token_id,
            video_token_id=self.meta_info.video_token_id,
            vision_start_token_id=self.meta_info.vision_start_token_id,
            vision_end_token_id=self.meta_info.vision_end_token_id,
            spatial_merge_size=self.meta_info.spatial_merge_size,
        )
