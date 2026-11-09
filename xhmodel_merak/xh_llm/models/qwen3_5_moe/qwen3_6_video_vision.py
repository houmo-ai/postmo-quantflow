"""Video-frame input for the Qwen3.6-35B-A3B patch-token ViT runtime.

The native Conv3d consumes ``temporal_patch_size`` successive frames per patch
(two for Qwen3.6). Its flattened weights are already preserved by the existing
ViT export. Native vision attention is independent for each temporal group, so
we can run those groups through the existing ``grid_t == 1`` graphs in order.
This module handles the vision tower only; video timestamps and language-model
placeholder/position construction remain the caller's responsibility.
"""

from __future__ import annotations

from typing import Protocol

import torch


class _ImageEncoder(Protocol):
    def encode(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor: ...


class Qwen3_6VideoVision:  # noqa: N801
    """Encode one decoded video using an existing patch-token visual runtime.

    ``visual`` is a ``VisualTokenGearHMONNXModel`` (or an encoder with the same
    ``encode(patches, grid_thw)`` contract). ``video_processor`` is the native
    checkpoint's video processor, not the image processor. Inputs are RGB frames
    in chronological order: a list of PIL images, a NumPy ``[T,H,W,C]`` array,
    or a Torch ``[T,C,H,W]`` tensor.
    """

    def __init__(self, visual: _ImageEncoder, video_processor):
        self.visual = visual
        self.video_processor = video_processor
        self.spatial_merge_size = int(video_processor.merge_size)
        self.temporal_patch_size = int(video_processor.temporal_patch_size)
        if self.spatial_merge_size <= 0 or self.temporal_patch_size <= 0:
            raise ValueError("video processor patch/merge sizes must be positive")

        visual_meta = getattr(visual, "visual_meta", None)
        if visual_meta is not None:
            for field, actual in (
                ("patch_size", int(video_processor.patch_size)),
                ("temporal_patch_size", self.temporal_patch_size),
                ("spatial_merge_size", self.spatial_merge_size),
            ):
                expected = int(getattr(visual_meta, field))
                if actual != expected:
                    raise ValueError(f"video processor {field}={actual} does not match visual runtime {expected}")

    @classmethod
    def from_pretrained(cls, visual: _ImageEncoder, model_dir: str):
        """Load preprocessing parameters from the original local checkpoint."""
        from transformers import AutoVideoProcessor

        return cls(visual, AutoVideoProcessor.from_pretrained(model_dir, local_files_only=True))

    def preprocess(self, frames, **processor_kwargs):
        """Normalize/resize and pack real frames without sampling them again.

        Native preprocessing pads only the final incomplete temporal group by
        repeating the last frame. For example, five frames become groups
        ``[f0,f1], [f2,f3], [f4,f4]`` when temporal_patch_size is two.
        Configure resizing/normalization on the checkpoint's video processor;
        per-call overrides can be passed here as native processor arguments.
        """
        if len(frames) == 0:
            raise ValueError("video frames must not be empty")
        reserved = {
            "videos", "do_sample_frames", "return_tensors", "patch_size",
            "temporal_patch_size", "merge_size", "num_frames", "fps",
        }
        conflicting = reserved.intersection(processor_kwargs)
        if conflicting:
            raise ValueError(f"video frame packing controls cannot be overridden: {sorted(conflicting)}")
        return self.video_processor(
            videos=[frames], do_sample_frames=False, return_tensors="pt", **processor_kwargs
        )

    @torch.no_grad()
    def encode_patches(
        self, pixel_values_videos: torch.Tensor, video_grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Encode native flattened video patches; return ``[video_tokens, D]``.

        ``video_grid_thw`` must describe one video as ``[[T,H,W]]``, where T
        counts temporal groups, not raw frames. Each group must fit a configured
        visual token gear. There is no two-frame limit on the complete video.
        """
        if tuple(video_grid_thw.shape) != (1, 3):
            raise ValueError(f"expected one video_grid_thw with shape [1, 3], got {tuple(video_grid_thw.shape)}")
        if video_grid_thw.dtype not in (torch.int32, torch.int64):
            raise ValueError("video_grid_thw must use an integer dtype")
        t, h, w = (int(value) for value in video_grid_thw[0].tolist())
        if min(t, h, w) <= 0:
            raise ValueError(f"video grid dimensions must be positive, got {(t, h, w)}")
        merge = self.spatial_merge_size
        if h % merge or w % merge:
            raise ValueError(f"video grid {(t, h, w)} must be divisible by spatial_merge_size={merge}")
        expected_patches = t * h * w
        if pixel_values_videos.ndim != 2 or pixel_values_videos.shape[0] != expected_patches:
            raise ValueError(
                f"video grid requires {expected_patches} flattened patches [N, D], "
                f"got {tuple(pixel_values_videos.shape)}"
            )

        group_grid = video_grid_thw.clone()
        group_grid[0, 0] = 1
        group_patch_count = h * w
        group_token_count = group_patch_count // merge**2
        outputs = []
        for patches in pixel_values_videos.split(group_patch_count, dim=0):
            output = self.visual.encode(patches, group_grid)
            if output.ndim == 3 and output.shape[0] == 1:
                output = output[0]
            if output.ndim != 2 or output.shape[0] != group_token_count:
                raise ValueError(
                    f"visual encoder must return {group_token_count} tokens per temporal group, "
                    f"got {tuple(output.shape)}"
                )
            # Sessions may reuse output buffers on the next group invocation.
            outputs.append(output.clone())
        return torch.cat(outputs, dim=0)

    def encode_frames(self, frames, **processor_kwargs) -> torch.Tensor:
        """Preprocess decoded RGB frames and return time-ordered ViT features."""
        inputs = self.preprocess(frames, **processor_kwargs)
        return self.encode_patches(inputs["pixel_values_videos"], inputs["video_grid_thw"])
