"""MiniCPM vision gear geometry and versioned metadata, without Torch/SDK."""

from collections.abc import Iterable, Mapping
import math


DEFAULT_IMAGE_TOKEN_GEARS = (96, 196, 384, 704, 1536)
DEFAULT_PATCH_TOKEN_GEARS = tuple(gear * 4 for gear in DEFAULT_IMAGE_TOKEN_GEARS)


def image_token_gears_from_patches(patch_gears: Iterable[int], mode: str) -> tuple[int, ...]:
    """Use equal input resolution coverage for 4x and 16x compression."""
    capacities = normalize_image_token_gears(patch_gears)
    if any(capacity % 16 for capacity in capacities):
        raise ValueError("Shared MiniCPM patch_token_gears must be divisible by 16")
    divisor = output_token_divisor(mode)
    return tuple(capacity // divisor for capacity in capacities)


def output_token_divisor(mode: str) -> int:
    if mode not in {"4x", "16x"}:
        raise ValueError(f"Unsupported MiniCPM downsample mode: {mode}")
    return 4 if mode == "4x" else 16


def normalize_image_token_gears(gears: Iterable[int]) -> tuple[int, ...]:
    values = tuple(gears)
    if not values or any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("MiniCPM image_token_gears must contain positive integers")
    return tuple(sorted(set(values)))


def factor_patch_grid(capacity: int, mode: str) -> tuple[int, int]:
    factor = math.isqrt(output_token_divisor(mode))
    if type(capacity) is not int or capacity <= 0 or capacity % (factor * factor):
        raise ValueError(f"Invalid MiniCPM patch capacity {capacity} for {mode}")
    tokens = capacity // (factor * factor)
    height = math.isqrt(tokens)
    while tokens % height:
        height -= 1
    return height * factor, tokens // height * factor


def vision_profile_entries(profile: Mapping, mode: str) -> tuple[Mapping, ...]:
    """Read legacy single graphs or validate a complete smallest-fit gear set."""
    divisor = output_token_divisor(mode)
    if not isinstance(profile, Mapping):
        raise ValueError(f"Missing MiniCPM vision profile {mode}")
    entries = profile.get("gears")
    if entries is None:
        entries = (profile,)
    else:
        if profile.get("schema_version") != 2 or profile.get("downsample_mode") != mode:
            raise ValueError(f"Invalid MiniCPM gear metadata for {mode}")
        if profile.get("token_unit") != "post_spatial_merge":
            raise ValueError("MiniCPM gear token_unit must be post_spatial_merge")
        if profile.get("routing_policy") != "smallest_fit" or profile.get("overflow_policy") != "reject":
            raise ValueError("MiniCPM vision gears require smallest_fit routing and reject overflow")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"MiniCPM vision profile {mode} requires non-empty gears")
        declared = normalize_image_token_gears(profile.get("image_token_gears", ()))
        if tuple(profile["image_token_gears"]) != declared:
            raise ValueError("MiniCPM image_token_gears metadata must be sorted and unique")
        actual = tuple(entry.get("image_token_capacity") for entry in entries if isinstance(entry, Mapping))
        if actual != declared:
            raise ValueError("MiniCPM vision gear entries must match sorted unique image_token_gears")
    for entry in entries:
        if not isinstance(entry, Mapping) or entry.get("downsample_mode") != mode:
            raise ValueError(f"MiniCPM vision gear has mismatched downsample_mode for {mode}")
        capacity = entry.get("token_capacity")
        if type(capacity) is not int or capacity <= 0 or capacity % divisor:
            raise ValueError(f"MiniCPM vision profile {mode} requires aligned positive token_capacity")
        if "image_token_capacity" in entry:
            image_capacity = entry["image_token_capacity"]
            if type(image_capacity) is not int or image_capacity * divisor != capacity:
                raise ValueError("MiniCPM vision image/patch token capacities do not match")
        if not isinstance(entry.get("hmonnx"), str) or not entry["hmonnx"]:
            raise ValueError(f"MiniCPM vision profile {mode} requires hmonnx")
    return tuple(entries)


def select_vision_profile(profile: Mapping, mode: str, grid: tuple[int, int]) -> Mapping:
    """Select the smallest exported graph that holds one image or slice."""
    factor = math.isqrt(output_token_divisor(mode))
    if min(grid) <= 0 or any(value % factor for value in grid):
        raise ValueError(f"Patch grid {grid} must have positive dimensions divisible by {factor} for {mode}")
    count = grid[0] * grid[1]
    entries = vision_profile_entries(profile, mode)
    for entry in entries:
        if count <= entry["token_capacity"]:
            return entry
    raise ValueError(f"Patch grid {grid} needs {count} patch tokens, exceeding capacity {entries[-1]['token_capacity']}")
