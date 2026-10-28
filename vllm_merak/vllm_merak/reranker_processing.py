# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Preserve Reranker scoring anchors across text and multimodal truncation."""

import time
from bisect import bisect_left
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from PIL import Image
from qwen_vl_utils import fetch_image, fetch_video
from vllm.entrypoints.pooling.scoring.io_processor import CrossEncoderIOProcessor, ScoringIOProcessors
from vllm.multimodal.media import ImageMediaIO, MediaConnector, MediaWithBytes, VideoMediaIO
from vllm.pooling_params import PoolingParams

from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.processor import reranker_token_indices


class RerankerVideoIO(VideoMediaIO):
    """Keep vLLM URL access checks while using official Qwen video decoding."""

    def __init__(self, *, fps=1, max_frames=64, total_pixels=7680 * 1024, sequence_metadata=None):
        # JPEG sequences are already sampled frames. Their official default
        # sample_fps is 2; encoded videos use the wrapper's 1 FPS / 64 limit.
        sequence_metadata = sequence_metadata or {}
        unsupported = sequence_metadata.keys() - {
            "fps",
            "frames_indices",
            "total_num_frames",
            "duration",
            "do_sample_frames",
        }
        if unsupported:
            raise ValueError(
                f"Use Reranker mm_processor_kwargs for video sampling, got media_io_kwargs: {sorted(unsupported)}"
            )
        super().__init__(ImageMediaIO(), num_frames=-1, **{"fps": 2, **sequence_metadata})
        self.has_source_indices = bool(sequence_metadata.keys() & {"frames_indices", "total_num_frames", "duration"})
        self.sample_options = {"fps": fps, "max_frames": max_frames}
        self.total_pixels = total_pixels

    def load_file(self, filepath):
        video, metadata = fetch_video(
            {"video": str(filepath), **self.sample_options}, image_patch_size=16, return_video_metadata=True
        )
        metadata = dict(metadata, do_sample_frames=False)
        if hasattr(metadata["frames_indices"], "tolist"):
            metadata["frames_indices"] = metadata["frames_indices"].tolist()
        return video, metadata

    def load_bytes(self, data):
        with NamedTemporaryFile(suffix=".mp4") as temporary:
            temporary.write(data)
            temporary.flush()
            return self.load_file(Path(temporary.name))

    def load_base64(self, media_type, data):
        if media_type.lower() != "video/jpeg":
            return super().load_base64(media_type, data)
        frames, source_metadata = super().load_base64(media_type, data)
        # The official wrapper samples frame lists with integer linspace.
        indices = np.arange(len(frames))
        if len(indices) > self.sample_options["max_frames"]:
            indices = np.linspace(0, len(frames) - 1, self.sample_options["max_frames"], dtype=int)
        video, metadata = fetch_video(
            {
                "video": [Image.fromarray(frames[i]) for i in indices],
                "total_pixels": self.total_pixels,
                "sample_fps": self.kwargs["fps"],
            },
            image_patch_size=16,
            return_video_metadata=True,
        )
        if self.has_source_indices:
            selected = [source_metadata["frames_indices"][i] for i in indices]
            selected.extend([selected[-1]] * (len(video) - len(selected)))
            metadata.update(source_metadata, frames_indices=selected)
        # Raw frame lists use Qwen's synthetic indices, including the padded
        # final position. Already sampled media with source indices retain them.
        metadata["do_sample_frames"] = False
        return video, metadata


def prepare_reranker_score_content(data, connector, *, processor_kwargs=None):
    """Apply the official wrapper's video/image/text ordering and pixel budgets."""
    options = dict(min_pixels=4096, max_pixels=1800 * 1024, total_pixels=7680 * 1024, fps=1, max_frames=64)
    supplied = processor_kwargs or {}
    unexpected = supplied.keys() - options.keys() - {"do_resize", "do_sample_frames"}
    if unexpected:
        raise ValueError(f"Unsupported Reranker preprocessing parameters: {sorted(unexpected)}")
    if supplied.get("do_resize") is True or supplied.get("do_sample_frames") is True:
        raise ValueError("Reranker uses qwen_vl_utils preprocessing; a second HF resize or sampling is unsupported")
    options.update({key: value for key, value in supplied.items() if key in options})
    if any(not isinstance(value, (int, float)) or value <= 0 for value in options.values()):
        raise ValueError("Reranker media budgets and FPS must be positive")
    if int(options["max_frames"]) != options["max_frames"] or options["max_frames"] < 2:
        raise ValueError("Reranker max_frames must be an integer of at least two")
    options["max_frames"] = int(options["max_frames"])
    if isinstance(data, str):
        return data, {}
    texts, images, videos = [], [], []
    for part in data:
        kind = part["type"]
        if kind == "text":
            texts.append(part["text"])
        elif kind == "image_url":
            # Preserve alpha until Qwen's to_rgb applies its white background.
            image = connector.fetch_image(part["image_url"]["url"], image_mode="RGBA")
            if isinstance(image, MediaWithBytes):
                image = image.media
            images.append(
                fetch_image(
                    {"image": image, "min_pixels": options["min_pixels"], "max_pixels": options["max_pixels"]},
                    image_patch_size=16,
                )
            )
        elif kind == "video_url":
            videos.append(
                connector.load_from_url(
                    part["video_url"]["url"],
                    RerankerVideoIO(
                        fps=options["fps"],
                        max_frames=options["max_frames"],
                        total_pixels=options["total_pixels"],
                        sequence_metadata=connector.media_io_kwargs.get("video"),
                    ),
                )
            )
        else:
            raise ValueError(f"Reranker requires raw text/image_url/video_url content, got {kind}")
    media = {}
    if videos:
        media["video"] = videos
    if images:
        media["image"] = images
    prompt = (
        "<|vision_start|><|video_pad|><|vision_end|>" * len(videos)
        + "<|vision_start|><|image_pad|><|vision_end|>" * len(images)
        + "".join(texts)
    )
    return prompt if texts or media else "NULL", media


def truncate_reranker_engine_input(engine_input, tokenizer, *, max_length):
    """Truncate expanded inputs, keeping visual features and offsets aligned."""
    ids = engine_input["prompt_token_ids"]
    # Qwen3-VL's video ranges include ordinary timestamp tokens as well as
    # visual tokens. Keep each range intact so M-RoPE and is_embed still agree.
    protected = [
        index
        for item in engine_input.get("mm_placeholders", {}).get("video", [])
        for index in range(item.offset, item.offset + item.length)
    ]
    indices = reranker_token_indices(ids, tokenizer, max_length=max_length, protected_indices=protected)
    if len(indices) == len(ids):
        return engine_input
    result = dict(engine_input)
    result["prompt_token_ids"] = [ids[index] for index in indices]
    if "mm_placeholders" in result:
        placeholders = {}
        for modality, ranges in result["mm_placeholders"].items():
            relocated = []
            for item in ranges:
                start = bisect_left(indices, item.offset)
                end = bisect_left(indices, item.offset + item.length)
                if indices[start:end] != list(range(item.offset, item.offset + item.length)):
                    raise ValueError("Reranker truncation must preserve every visual placeholder token")
                relocated.append(replace(item, offset=start))
            placeholders[modality] = relocated
        result["mm_placeholders"] = placeholders
    return result


class RerankerCrossEncoderIOProcessor(CrossEncoderIOProcessor):
    """Reuse vLLM scoring, then apply the Reranker-only input contract."""

    def _is_reranker(self):
        architecture = getattr(self.model_config.hf_config, "merak_loader_architecture", None)
        return architecture == "Qwen3VLRerankerMerakForSequenceClassification"

    def get_score_prompt(self, data_1, data_2, encode_kwargs, *args, processor_kwargs=None, **kwargs):
        if not self._is_reranker():
            return super().get_score_prompt(data_1, data_2, encode_kwargs, *args, **kwargs)
        mm_config = self.model_config.multimodal_config
        options = {**(mm_config.mm_processor_kwargs or {}), **(processor_kwargs or {})}
        connector = MediaConnector(
            media_io_kwargs=mm_config.media_io_kwargs,
            allowed_local_media_path=self.model_config.allowed_local_media_path,
            allowed_media_domains=self.model_config.allowed_media_domains,
        )
        query, query_media = prepare_reranker_score_content(data_1, connector, processor_kwargs=options)
        document, document_media = prepare_reranker_score_content(data_2, connector, processor_kwargs=options)
        full_prompt, engine_prompt = super().get_score_prompt(query, document, encode_kwargs, *args, **kwargs)
        media = {
            kind: query_media.get(kind, []) + document_media.get(kind, [])
            for kind in dict.fromkeys([*query_media, *document_media])
        }
        if media:
            engine_prompt["multi_modal_data"] = media
            engine_prompt["mm_processor_kwargs"] = {"do_resize": False, "do_sample_frames": False}
        return full_prompt, engine_prompt

    def _pre_process(
        self,
        scoring_data,
        tok_params,
        pooling_params,
        chat_template=None,
        max_tokens_per_query=0,
        max_tokens_per_doc=0,
        prompt_extras=None,
    ):
        if not self._is_reranker():
            return super()._pre_process(
                scoring_data,
                tok_params,
                pooling_params,
                chat_template,
                max_tokens_per_query,
                max_tokens_per_doc,
                prompt_extras,
            )
        if max_tokens_per_query or max_tokens_per_doc:
            raise ValueError("Reranker truncates the complete pair after media expansion; use truncate_prompt_tokens")
        if tok_params.pad_prompt_tokens is not None:
            raise ValueError("Reranker scoring does not support prompt padding after its template suffix")
        capacity = self.model_config.max_model_len
        if tok_params.max_input_tokens is not None:
            capacity = min(capacity, tok_params.max_input_tokens)
        requested = tok_params.truncate_prompt_tokens
        if requested is not None and requested >= 0:
            capacity = min(capacity, requested)
        # Generic tokenizer truncation would discard the assistant suffix.
        # Expand all media before enforcing the total token budget, so every
        # visual token is counted and its final position can be recomputed.
        unlimited = replace(tok_params, max_total_tokens=None, max_output_tokens=0, truncate_prompt_tokens=None)
        queries, documents = scoring_data.data_1, scoring_data.data_2
        if len(queries) == 1:
            queries = queries * len(documents)
        extras = prompt_extras or {}
        pooling_params = pooling_params or PoolingParams(task="classify")
        engine_inputs = []
        arrival_time = time.time()
        for query, document in zip(queries, documents, strict=True):
            _, prompt = self.get_score_prompt(
                query,
                document,
                unlimited.get_encode_kwargs(),
                chat_template=chat_template,
                chat_template_kwargs=extras.get("chat_template_kwargs"),
                processor_kwargs=extras.get("mm_processor_kwargs"),
            )
            unlimited.apply_post_tokenization(self.tokenizer, prompt)
            if "cache_salt" in extras:
                prompt["cache_salt"] = extras["cache_salt"]
            expanded = self.renderer.process_for_engine(prompt, arrival_time)
            engine_inputs.append(truncate_reranker_engine_input(expanded, self.tokenizer, max_length=capacity))
        return engine_inputs, [pooling_params] * len(engine_inputs)


def register_reranker_scoring_processor():
    """Select this scoped adapter through vLLM's scoring processor mapping."""
    ScoringIOProcessors["cross-encoder"] = RerankerCrossEncoderIOProcessor
