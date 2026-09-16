"""Token-capacity helpers for PaddleOCR-VL visual HMONNX gears."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch


VISUAL_INPUT_PATCHES = "patches"
VISUAL_ATTENTION_MASK_FORMAT = "additive_key_padding_bias"
VISUAL_ATTENTION_MASK_SHAPE = "[1,1,1,patch_token_capacity]"
VISUAL_ATTENTION_MASK_OPERATOR = "xhquant.nn.MaskedAdd+Softmax"
VISUAL_POSITION_FORMAT = "bilinear_interpolation_ids_weights"
VISUAL_ROTARY_POSITION_FORMAT = "height_width_2d_ids"


def normalize_image_token_gears(gears: Iterable[int]) -> tuple[int, ...]:
    raw_gears = tuple(gears)
    if any(isinstance(gear, bool) or not isinstance(gear, int) for gear in raw_gears):
        raise TypeError(f"PaddleOCR image token gears must be integers, got {raw_gears}")
    normalized = tuple(sorted(set(raw_gears)))
    if not normalized or normalized[0] <= 0:
        raise ValueError(f"PaddleOCR image token gears must be positive, got {normalized}")
    return normalized


def select_image_token_gear(valid_image_tokens: int, gears: Iterable[int]) -> int:
    valid_image_tokens = int(valid_image_tokens)
    if valid_image_tokens <= 0:
        raise ValueError(f"valid image token count must be positive, got {valid_image_tokens}")
    normalized = normalize_image_token_gears(gears)
    for gear in normalized:
        if valid_image_tokens <= gear:
            return gear
    raise ValueError(
        f"image needs {valid_image_tokens} tokens, which exceeds the largest configured gear {normalized[-1]}"
    )


def patch_token_capacity(image_token_capacity: int, spatial_merge_size: int = 2) -> int:
    image_token_capacity = int(image_token_capacity)
    spatial_merge_size = int(spatial_merge_size)
    if image_token_capacity <= 0:
        raise ValueError(f"image_token_capacity must be positive, got {image_token_capacity}")
    if spatial_merge_size <= 0:
        raise ValueError(f"spatial_merge_size must be positive, got {spatial_merge_size}")
    return image_token_capacity * spatial_merge_size * spatial_merge_size


def factor_image_token_grid(image_token_capacity: int, spatial_merge_size: int = 2) -> torch.Tensor:
    image_token_capacity = int(image_token_capacity)
    merged_h = math.isqrt(image_token_capacity)
    while image_token_capacity % merged_h != 0:
        merged_h -= 1
    merged_w = image_token_capacity // merged_h
    return torch.tensor([[1, merged_h * spatial_merge_size, merged_w * spatial_merge_size]], dtype=torch.long)


def _merge_major_tensor(tensor: torch.Tensor, h: int, w: int, merge_size: int, *, spatial_dims: int) -> torch.Tensor:
    leading = tensor.shape[:-spatial_dims]
    trailing = tensor.shape[-(spatial_dims - 2) :] if spatial_dims > 2 else ()
    tensor = tensor.reshape(*leading, h // merge_size, merge_size, w // merge_size, merge_size, *trailing)
    leading_dims = list(range(len(leading)))
    base = len(leading)
    trailing_dims = list(range(base + 5, base + 5 + len(trailing)))
    tensor = tensor.permute(*leading_dims, base, base + 2, base + 1, base + 3, *trailing_dims)
    return tensor.reshape(*leading, h * w, *trailing)


def pad_patch_images(
    pixel_values: torch.Tensor,
    capacity: int,
    grid_thw: torch.Tensor | None = None,
    spatial_merge_size: int = 2,
) -> tuple[torch.Tensor, int]:
    if pixel_values.ndim == 4:
        pixel_values = pixel_values.unsqueeze(0)
    if pixel_values.ndim != 5 or pixel_values.shape[0] != 1:
        raise ValueError(
            "PaddleOCR visual patches must have shape [N, C, H, W] or [1, N, C, H, W], "
            f"got {tuple(pixel_values.shape)}"
        )
    valid_tokens = int(pixel_values.shape[1])
    capacity = int(capacity)
    if grid_thw is not None:
        grid = torch.as_tensor(grid_thw, dtype=torch.long)
        if grid.numel() != 3:
            raise ValueError(f"PaddleOCR image_grid_thw must contain exactly one grid, got {tuple(grid.shape)}")
        t, h, w = (int(value) for value in grid.reshape(-1).tolist())
        if t != 1 or h * w != valid_tokens:
            raise ValueError(f"pixel_values has {valid_tokens} patches but image_grid_thw declares {[t, h, w]}")
        if h % spatial_merge_size or w % spatial_merge_size:
            raise ValueError(f"grid {[t, h, w]} must be divisible by spatial_merge_size={spatial_merge_size}")
        bsz, _, channels, patch_h, patch_w = pixel_values.shape
        pixel_values = pixel_values.reshape(
            bsz,
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
            channels,
            patch_h,
            patch_w,
        )
        pixel_values = pixel_values.permute(0, 1, 3, 2, 4, 5, 6, 7)
        pixel_values = pixel_values.reshape(bsz, valid_tokens, channels, patch_h, patch_w)
    if valid_tokens > capacity:
        raise ValueError(f"patch token count {valid_tokens} exceeds gear capacity {capacity}")
    if valid_tokens == capacity:
        return pixel_values, valid_tokens
    padding = pixel_values.new_zeros(
        (1, capacity - valid_tokens, pixel_values.shape[2], pixel_values.shape[3], pixel_values.shape[4])
    )
    return torch.cat((pixel_values, padding), dim=1), valid_tokens


def _linear_indices_weights(out_size: int, in_size: int, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # Native PaddleOCR-VL resizes learned positions with align_corners=True.
    # A single output position maps to the first input position.
    scale = (in_size - 1) / (out_size - 1) if out_size > 1 else 0.0
    coords = torch.arange(out_size, dtype=torch.float32, device=device) * scale
    lower = torch.floor(coords)
    upper_weight = coords - lower
    lower_weight = 1.0 - upper_weight
    lower = lower.to(torch.long)
    upper = lower + 1
    lower = lower.clamp(0, in_size - 1)
    upper = upper.clamp(0, in_size - 1)
    return torch.stack((lower, upper), dim=0), torch.stack((lower_weight, upper_weight), dim=0)


def _position_interpolation_inputs(
    *,
    h: int,
    w: int,
    capacity: int,
    num_positions: int,
    dtype: torch.dtype,
    device: torch.device,
    merge_major: bool,
    merge_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    side = math.isqrt(num_positions)
    if side * side != num_positions:
        raise ValueError(f"PaddleOCR position embedding count must be square, got {num_positions}")
    h_ids, h_weights = _linear_indices_weights(h, side, device=device)
    w_ids, w_weights = _linear_indices_weights(w, side, device=device)
    ids = torch.stack(
        (
            h_ids[0, :, None] * side + w_ids[0, None, :],
            h_ids[0, :, None] * side + w_ids[1, None, :],
            h_ids[1, :, None] * side + w_ids[0, None, :],
            h_ids[1, :, None] * side + w_ids[1, None, :],
        ),
        dim=0,
    )
    weights = torch.stack(
        (
            h_weights[0, :, None] * w_weights[0, None, :],
            h_weights[0, :, None] * w_weights[1, None, :],
            h_weights[1, :, None] * w_weights[0, None, :],
            h_weights[1, :, None] * w_weights[1, None, :],
        ),
        dim=0,
    )
    if merge_major:
        ids = _merge_major_tensor(ids, h, w, merge_size, spatial_dims=2)
        weights = _merge_major_tensor(weights, h, w, merge_size, spatial_dims=2)
    ids = ids.reshape(4, -1)
    weights = weights.reshape(4, -1)
    padded_ids = torch.zeros((4, capacity), dtype=torch.long, device=device)
    padded_weights = torch.zeros((4, capacity), dtype=dtype, device=device)
    padded_ids[:, : ids.shape[1]] = ids
    padded_weights[:, : weights.shape[1]] = weights.to(dtype)
    return padded_ids, padded_weights


def _rotary_position_inputs(
    *, h: int, w: int, capacity: int, device: torch.device, merge_major: bool, merge_size: int
) -> torch.Tensor:
    rows = torch.arange(h, device=device).view(h, 1).expand(h, w).reshape(-1)
    cols = torch.arange(w, device=device).view(1, w).expand(h, w).reshape(-1)
    if merge_major:
        rows = _merge_major_tensor(rows.reshape(h, w), h, w, merge_size, spatial_dims=2).reshape(-1)
        cols = _merge_major_tensor(cols.reshape(h, w), h, w, merge_size, spatial_dims=2).reshape(-1)
    rotary_position_ids = torch.zeros((2, capacity), dtype=torch.long, device=device)
    rotary_position_ids[0, : rows.shape[0]] = rows
    rotary_position_ids[1, : cols.shape[0]] = cols
    return rotary_position_ids


def build_visual_token_gear_inputs(
    grid_thw: torch.Tensor,
    *,
    patch_capacity: int,
    num_positions: int,
    spatial_merge_size: int = 2,
    dtype: torch.dtype = torch.float16,
    device: torch.device | str | None = None,
    rotary_cache_length: int | None = None,
    merge_major: bool = True,
) -> dict[str, torch.Tensor | int]:
    grid = torch.as_tensor(grid_thw, dtype=torch.long)
    if grid.numel() != 3:
        raise ValueError(f"one PaddleOCR visual gear invocation handles exactly one image, got {tuple(grid.shape)}")
    t, h, w = (int(value) for value in grid.reshape(-1).tolist())
    if t != 1:
        raise ValueError(f"PaddleOCR visual token gears require grid_t == 1, got {t}")
    if h <= 0 or w <= 0:
        raise ValueError(f"grid height and width must be positive, got {(h, w)}")
    if h % spatial_merge_size or w % spatial_merge_size:
        raise ValueError(f"grid {(t, h, w)} must be divisible by spatial_merge_size={spatial_merge_size}")
    patch_capacity = int(patch_capacity)
    valid_patch_tokens = t * h * w
    merge_unit = int(spatial_merge_size) ** 2
    if patch_capacity % merge_unit:
        raise ValueError(f"patch capacity {patch_capacity} must be divisible by spatial merge unit {merge_unit}")
    if valid_patch_tokens > patch_capacity:
        raise ValueError(f"image grid needs {valid_patch_tokens} patches, which exceeds gear capacity {patch_capacity}")
    if rotary_cache_length is not None and max(h, w) > int(rotary_cache_length):
        raise ValueError(f"image grid {(h, w)} exceeds rotary cache length {rotary_cache_length}")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise ValueError(f"attention mask dtype must be floating point, got {dtype}")

    target_device = torch.device(device) if device is not None else grid.device
    position_ids, position_weights = _position_interpolation_inputs(
        h=h,
        w=w,
        capacity=patch_capacity,
        num_positions=int(num_positions),
        dtype=dtype,
        device=target_device,
        merge_major=merge_major,
        merge_size=int(spatial_merge_size),
    )
    attention_mask = torch.zeros((1, 1, 1, patch_capacity), dtype=dtype, device=target_device)
    attention_mask[..., valid_patch_tokens:] = -torch.finfo(dtype).max
    return {
        "position_ids": position_ids,
        "position_weights": position_weights,
        "rotary_position_ids": _rotary_position_inputs(
            h=h,
            w=w,
            capacity=patch_capacity,
            device=target_device,
            merge_major=merge_major,
            merge_size=int(spatial_merge_size),
        ),
        "attention_mask": attention_mask,
        "valid_patch_tokens": valid_patch_tokens,
        "valid_image_tokens": valid_patch_tokens // merge_unit,
    }


def build_visual_gear_manifest(
    gear_artifacts: Iterable[dict],
    *,
    spatial_merge_size: int = 2,
    rotary_cache_length: int,
) -> dict:
    artifacts = sorted(
        (dict(artifact) for artifact in gear_artifacts),
        key=lambda artifact: int(artifact["image_token_capacity"]),
    )
    gears = normalize_image_token_gears(artifact["image_token_capacity"] for artifact in artifacts)
    if len(gears) != len(artifacts):
        raise ValueError("PaddleOCR visual gear artifacts must have unique image_token_capacity values")
    for gear, artifact in zip(gears, artifacts, strict=True):
        expected_patch_capacity = patch_token_capacity(gear, spatial_merge_size)
        if int(artifact["patch_token_capacity"]) != expected_patch_capacity:
            raise ValueError(
                f"gear {gear} declares patch capacity {artifact['patch_token_capacity']}, "
                f"expected {expected_patch_capacity}"
            )
        if not str(artifact.get("hmonnx", "")).strip():
            raise ValueError(f"gear {gear} is missing its HMONNX path")
    return {
        "schema_version": 2,
        "visual_input_mode": VISUAL_INPUT_PATCHES,
        "token_unit": "post_spatial_merge",
        "routing_policy": "smallest_fit",
        "overflow_policy": "reject",
        "position_format": VISUAL_POSITION_FORMAT,
        "rotary_position_format": VISUAL_ROTARY_POSITION_FORMAT,
        "attention_mask_format": VISUAL_ATTENTION_MASK_FORMAT,
        "attention_mask_shape": VISUAL_ATTENTION_MASK_SHAPE,
        "attention_mask_operator": VISUAL_ATTENTION_MASK_OPERATOR,
        "spatial_merge_size": int(spatial_merge_size),
        "rotary_cache_length": int(rotary_cache_length),
        "gears": artifacts,
    }
