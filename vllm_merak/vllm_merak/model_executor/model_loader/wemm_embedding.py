# SPDX-License-Identifier: Apache-2.0
"""vLLM-Merak pooling loader for WeMM-Embedding HMONNX releases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any, Optional, cast

import torch
from vllm.config import ModelConfig, PoolerConfig, VllmConfig
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaCopySpec,
    MambaStateCopyFunc,
    get_temporal_copy_spec,
)
from vllm.model_executor.layers.pooler import DispatchPooler
from vllm.model_executor.layers.pooler.activations import PoolerNormalize
from vllm.model_executor.layers.pooler.seqwise import SequencePooler
from vllm.model_executor.layers.pooler.seqwise.heads import EmbeddingPoolerHead
from vllm.model_executor.layers.pooler.seqwise.methods import get_seq_pooling_method
from vllm.model_executor.models.interfaces import MultiModalEmbeddings
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.parse import MultiModalDataItems
from vllm.multimodal.processing import (
    PromptReplacement,
    PromptUpdate,
    PromptUpdateDetails,
)

from xhmodel_merak.xh_llm import AutoLLMHONNXModel
from xhmodel_merak.xh_llm.models.wemm_embedding.processor import video_frame_grids

from ...utils import should_preserve_hmonnx_runtime_device_map
from ..register import MERAK_MODEL_REGISTRY
from .merak_for_causal_lm import _resolve_hmfp_block_size
from .qwen3_5 import (
    MerakQwen3_5MultiModalProcessor,
    MerakQwen3_5ProcessingInfo,
    Qwen3VLDummyInputsBuilder,
    Qwen3_5MerakForCausalLM,
)


WEMM_EMBEDDING_TOKEN = "<embedding>"


def _ensure_wemm_embedding_token(
    tokenizer: Any,
    prompt_token_ids: list[int],
) -> list[int]:
    """Return WeMM input IDs ending at its embedding pooling anchor."""

    token_ids = tokenizer.encode(
        WEMM_EMBEDDING_TOKEN,
        add_special_tokens=False,
    )
    if len(token_ids) != 1:
        raise ValueError(
            f"Expected {WEMM_EMBEDDING_TOKEN!r} to encode as one token, "
            f"got {token_ids!r}"
        )
    embedding_token_id = int(token_ids[0])
    if prompt_token_ids and prompt_token_ids[-1] == embedding_token_id:
        return list(prompt_token_ids)
    return [*prompt_token_ids, embedding_token_id]


def _install_wemm_hybrid_cache_page_compatibility() -> None:
    """Install the plugin-wide HMFP/Mamba physical-page compatibility hook."""

    from ...platform import _install_merak_hybrid_cache_page_compatibility

    _install_merak_hybrid_cache_page_compatibility()


class WeMMEmbeddingProcessingInfo(MerakQwen3_5ProcessingInfo):
    """Expose the image and video modalities implemented by the WeMM runtime."""

    def get_supported_mm_limits(self) -> Mapping[str, int | None]:
        return {"image": None, "video": None}

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int]:
        return {
            "image": self.get_max_image_tokens(),
            "video": self.get_max_video_tokens(seq_len, mm_counts),
        }


class WeMMEmbeddingMultiModalProcessor(MerakQwen3_5MultiModalProcessor):
    """Apply WeMM's pooling-token contract to Merak multimodal inputs."""

    def _apply_hf_processor_tokens_only(
        self,
        prompt_tokens: list[int],
    ) -> list[int]:
        return _ensure_wemm_embedding_token(
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
            if update.modality != "video" or not isinstance(
                update, PromptReplacement
            ):
                wrapped_updates.append(update)
                continue

            base_replacement = update.replacement

            def replacement(
                item_idx: int,
                *,
                base_replacement: Any = base_replacement,
            ) -> PromptUpdateDetails[list[int]]:
                details = (
                    base_replacement(item_idx)
                    if callable(base_replacement)
                    else base_replacement
                )
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


def _get_wemm_split_conv_copy_spec(
    state: torch.Tensor,
    block_ids: list[int],
    cur_block_idx: int,
    num_accepted_tokens: int,
) -> MambaCopySpec:
    """Return the source span for a split WeMM convolution state."""

    if num_accepted_tokens != 1:
        raise NotImplementedError(
            "WeMM split convolution state does not support offset copies in "
            "vLLM mamba_cache_mode='align'; speculative decoding must use the "
            "Merak transaction path."
        )
    src_state = state[block_ids[cur_block_idx]]
    return MambaCopySpec(
        start_addr=src_state.data_ptr(),
        num_elements=src_state.numel(),
    )


def _get_wemm_linear_state_copy_funcs() -> tuple[MambaStateCopyFunc, ...]:
    """Copy functions for ``conv_q``, ``conv_k``, ``conv_v``, recurrent."""

    return (
        _get_wemm_split_conv_copy_spec,
        _get_wemm_split_conv_copy_spec,
        _get_wemm_split_conv_copy_spec,
        get_temporal_copy_spec,
    )


def _build_wemm_pooler(pooler_config: PoolerConfig) -> DispatchPooler:
    """Build the FP32 last-token normalization used by native WeMM."""

    return DispatchPooler(
        {
            "embed": SequencePooler(
                pooling=get_seq_pooling_method(
                    pooler_config.get_seq_pooling_type()
                ),
                head=EmbeddingPoolerHead(
                    head_dtype=torch.float32,
                    activation=PoolerNormalize(),
                ),
            )
        }
    )


@MULTIMODAL_REGISTRY.register_processor(
    WeMMEmbeddingMultiModalProcessor,
    info=WeMMEmbeddingProcessingInfo,
    dummy_inputs=Qwen3VLDummyInputsBuilder,
)
@MERAK_MODEL_REGISTRY.register_module("WeMMEmbeddingMerakForEmbedding")
class WeMMEmbeddingMerakForEmbedding(Qwen3_5MerakForCausalLM):
    """Run WeMM's prefill-only hidden-state graph through vLLM pooling."""

    is_pooling_model = True
    default_seq_pooling_type = "LAST"
    supports_multimodal = True
    supports_multimodal_raw_input_only = True
    requires_raw_input_tokens = True
    supports_mrope = True

    @classmethod
    def get_mamba_state_copy_func(cls):
        """Expose vLLM's state-move contract for WeMM's four-state layout."""

        return _get_wemm_linear_state_copy_funcs()

    def _get_block_size(self) -> int:
        finalized_block_size = getattr(self, "_merak_attention_block_size", None)
        if finalized_block_size is None:
            return super()._get_block_size()
        return _resolve_hmfp_block_size(
            SimpleNamespace(block_size=int(finalized_block_size))
        )

    def _load_hmonnx_model(
        self,
        hmonnx_meta_file: str,
        runtime_kwargs: dict[str, Any],
    ) -> Any:
        """Construct ordinary WeMM sessions on the runner's primary GPU.

        HMONNX V2 chooses an automatic node placement when a session is
        constructed without ``device_map``.  Moving that materialized session
        afterward is not guaranteed to move every graph tensor, which can
        leave indices and their source tensors on different CUDA devices.
        Auto-offload owns its multi-device placement and is left unchanged.
        """

        loader_kwargs = dict(runtime_kwargs)
        if not should_preserve_hmonnx_runtime_device_map(self.config):
            loader_kwargs.setdefault("device_map", [self.inference_device])
        return AutoLLMHONNXModel.from_pretrained(
            hmonnx_meta_file,
            **loader_kwargs,
        )

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        kv_cache_dtype: str = "auto",
        *,
        vllm_config: Optional[VllmConfig] = None,
        prefix: str = "",
    ) -> None:
        _install_wemm_hybrid_cache_page_compatibility()
        super().__init__(
            model_config=model_config,
            kv_cache_dtype=kv_cache_dtype,
            vllm_config=vllm_config,
            prefix=prefix,
        )
        if vllm_config is None:
            raise ValueError("WeMM embedding requires vllm_config for pooling")
        if vllm_config.model_config.runner_type != "pooling":
            raise ValueError(
                "WeMM embedding requires runner_type='pooling', got "
                f"{vllm_config.model_config.runner_type!r}"
            )

        metadata = self.inference_model.meta_info
        if getattr(metadata, "embedding_output", None) != "hidden_states":
            raise ValueError(
                "WeMM HMONNX metadata must declare "
                "embedding_output='hidden_states'"
            )
        if getattr(metadata, "embedding_pooling", None) != "last_token":
            raise ValueError(
                "WeMM HMONNX metadata must declare "
                "embedding_pooling='last_token'"
            )
        if getattr(metadata, "embedding_normalize", None) is not True:
            raise ValueError("WeMM HMONNX metadata must enable embedding_normalize")
        embedding_dimension = int(getattr(metadata, "embedding_dimension", 0))
        if embedding_dimension <= 0:
            raise ValueError(
                "WeMM HMONNX metadata must declare a positive "
                "embedding_dimension"
            )

        # The shared request router names its first graph result "logits".  For
        # WeMM that tensor is the exported hidden-state sequence, so use the
        # embedding width as the router's output width.  The pooling runner
        # consumes it directly and never calls compute_logits/sample.
        self.logits_processor = LogitsProcessor(
            embedding_dimension,
            logits_as_input=True,
        )
        pooler_config = vllm_config.model_config.pooler_config
        if pooler_config is None:
            raise ValueError("WeMM embedding requires a resolved pooler_config")
        # Match XHWeMMEmbeddingHMONNXModel._pool_hidden_states exactly: cast
        # the selected last-token state to FP32 before L2 normalization.  Use
        # a model-local head because ModelConfig.head_dtype is read-only and
        # the Merak platform otherwise resolves pooling heads to FP16.
        self.pooler = _build_wemm_pooler(pooler_config)

    def _run_request(self, past_len: int, **kwargs: Any):
        # The generic Merak runner classifies any one-token continuation as
        # decode.  For a pooling request, that continuation can instead be the
        # final input token containing WeMM's embedding anchor.  WeMM has no
        # autoregressive decode path, so route every request chunk through its
        # prefill graph.
        kwargs["execution_phase"] = "prefill"
        return super()._run_request(past_len, **kwargs)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:  # type: ignore[override]
        pixel_values_videos = kwargs.get("pixel_values_videos")
        if pixel_values_videos is None:
            return super().embed_multimodal(**kwargs)
        video_grid_thw = kwargs.get("video_grid_thw")
        if not isinstance(pixel_values_videos, torch.Tensor):
            raise TypeError(
                "WeMM pixel_values_videos must be a tensor, got "
                f"{type(pixel_values_videos).__name__}"
            )
        if not isinstance(video_grid_thw, torch.Tensor):
            raise TypeError(
                "WeMM video_grid_thw must be a tensor, got "
                f"{type(video_grid_thw).__name__}"
            )

        visual = cast(Any, self.inference_model.visual)
        staged_pixels = pixel_values_videos.to(
            device=visual.device,
            dtype=visual.dtype,
        )
        frame_grids = video_frame_grids(
            staged_pixels,
            video_grid_thw.to(device=staged_pixels.device),
            int(self.spatial_merge_size),
        )
        frame_outputs = visual.encode_many(staged_pixels, frame_grids)
        if not isinstance(frame_outputs, (list, tuple)):
            raise TypeError(
                "WeMM video visual encoder must return one tensor per temporal unit"
            )
        temporal_units = [int(value) for value in video_grid_thw[:, 0].tolist()]
        if len(frame_outputs) != sum(temporal_units):
            raise ValueError(
                "WeMM video visual output count does not match video_grid_thw: "
                f"outputs={len(frame_outputs)}, temporal_units={sum(temporal_units)}"
            )

        outputs: list[torch.Tensor] = []
        offset = 0
        for count in temporal_units:
            video_frames = frame_outputs[offset : offset + count]
            outputs.append(
                torch.cat(
                    [self._normalize_visual_embeds(frame) for frame in video_frames],
                    dim=0,
                ).to(
                    device=self._runner_device,
                    dtype=self.input_embeddings.weight.dtype,
                )
            )
            offset += count
        return tuple(outputs)

    def _extract_request_token_embeds(
        self,
        input_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        token_id: int,
    ) -> Optional[torch.Tensor]:
        if input_ids is None or inputs_embeds is None or token_id < 0:
            return None
        flat_input_ids = self._flatten_input_ids(input_ids)
        mask = flat_input_ids == token_id
        if not torch.any(mask):
            return None
        return self._flatten_inputs_embeds(inputs_embeds)[mask]

    def _count_request_video_grids(
        self,
        *,
        num_video_tokens: int,
        video_grid_thw: Optional[torch.Tensor],
        video_grid_offset: int,
    ) -> int:
        """Map Qwen3-VL frame spans back to WeMM video items.

        Qwen3-VL expands one video into a timestamped vision span per temporal
        unit.  Counting ``vision_start`` tokens therefore counts frames, while
        ``video_grid_thw`` retains one row per original video.  Accumulate the
        expected merged-token counts to recover how many grid rows belong to
        this request.
        """

        if num_video_tokens <= 0 or video_grid_thw is None:
            return 0
        if video_grid_thw.ndim != 2 or video_grid_thw.shape[1] != 3:
            raise ValueError(
                "WeMM video_grid_thw must have shape [num_videos, 3], got "
                f"{tuple(video_grid_thw.shape)}"
            )

        merge_area = int(self.spatial_merge_size) ** 2
        covered_tokens = 0
        num_videos = 0
        for grid in video_grid_thw[video_grid_offset:]:
            grid_tokens = int(torch.prod(grid).item())
            if grid_tokens % merge_area:
                raise ValueError(
                    "WeMM video grid is not divisible by the spatial merge "
                    f"area {merge_area}: {grid.tolist()}"
                )
            covered_tokens += grid_tokens // merge_area
            num_videos += 1
            if covered_tokens >= num_video_tokens:
                return num_videos

        raise ValueError(
            "WeMM video grids do not cover the request's video tokens: "
            f"tokens={num_video_tokens}, covered={covered_tokens}"
        )

    def _extract_request_multimodal_kwargs(  # type: ignore[override]
        self,
        *,
        input_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor],
        image_grid_offset: int,
        video_grid_thw: Optional[torch.Tensor],
        video_grid_offset: int,
    ) -> tuple[dict[str, torch.Tensor], int, int]:
        request_kwargs: dict[str, torch.Tensor] = {}
        num_images, _num_video_frame_spans = self._count_multimodal_items(input_ids)
        flat_input_ids = (
            self._flatten_input_ids(input_ids) if input_ids is not None else None
        )
        num_image_tokens = (
            int((flat_input_ids == self.image_token_id).sum().item())
            if flat_input_ids is not None and self.image_token_id >= 0
            else 0
        )
        num_video_tokens = (
            int((flat_input_ids == self.video_token_id).sum().item())
            if flat_input_ids is not None and self.video_token_id >= 0
            else 0
        )
        num_videos = self._count_request_video_grids(
            num_video_tokens=num_video_tokens,
            video_grid_thw=video_grid_thw,
            video_grid_offset=video_grid_offset,
        )

        next_image_offset = image_grid_offset + num_images
        next_video_offset = video_grid_offset + num_videos
        if num_image_tokens:
            image_embeds = self._extract_request_token_embeds(
                input_ids, inputs_embeds, self.image_token_id
            )
            if image_embeds is None:
                raise ValueError("WeMM request is missing merged image embeddings")
            request_kwargs["image_embeds"] = image_embeds
        if num_video_tokens:
            video_embeds = self._extract_request_token_embeds(
                input_ids, inputs_embeds, self.video_token_id
            )
            if video_embeds is None:
                raise ValueError("WeMM request is missing merged video embeddings")
            request_kwargs["video_embeds"] = video_embeds

        if num_images and image_grid_thw is not None:
            request_image_grids = image_grid_thw[image_grid_offset:next_image_offset]
            if request_image_grids.shape[0] != num_images:
                raise ValueError(
                    f"WeMM request expected {num_images} image grids, "
                    f"got {request_image_grids.shape[0]}"
                )
            request_kwargs["image_grid_thw"] = request_image_grids
        elif num_images and image_grid_thw is None:
            next_image_offset = image_grid_offset

        if num_videos and video_grid_thw is not None:
            request_video_grids = video_grid_thw[video_grid_offset:next_video_offset]
            if request_video_grids.shape[0] != num_videos:
                raise ValueError(
                    f"WeMM request expected {num_videos} video grids, "
                    f"got {request_video_grids.shape[0]}"
                )
            request_kwargs["video_grid_thw"] = request_video_grids
        elif num_videos and video_grid_thw is None:
            next_video_offset = video_grid_offset

        return request_kwargs, next_image_offset, next_video_offset


__all__ = [
    "WeMMEmbeddingMerakForEmbedding",
    "WeMMEmbeddingMultiModalProcessor",
    "WeMMEmbeddingProcessingInfo",
    "_build_wemm_pooler",
]
