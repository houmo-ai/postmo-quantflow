"""Kokoro specialization of the HM model release naming/layout contract."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from .buckets import KokoroBucketRoute


def release_prefix(routes: tuple[KokoroBucketRoute, ...], date: str) -> str:
    if len(date) != 8 or not date.isascii() or not date.isdigit():
        raise ValueError("release_date must be YYYYMMDD")
    datetime.strptime(date, "%Y%m%d")
    if not routes:
        raise ValueError("release must contain at least one route")
    tokens = "-".join(str(route.token_max_length) for route in routes)
    frames = "-".join(str(route.frame_max_length) for route in routes)
    return f"hmquant_xh2_kokoro-82m-v1.1-zh_w16a16_t{tokens}_f{frames}_{date}"


def component_path(root: Path, prefix: str, role: str, key: str, *, reference: bool) -> Path:
    component = f"{role}_{key}"
    directory = root / "reference_onnx" if reference else root / component
    suffix = "_float.onnx" if reference else "_with_act.onnx"
    return directory / f"{prefix}_{component}{suffix}"
