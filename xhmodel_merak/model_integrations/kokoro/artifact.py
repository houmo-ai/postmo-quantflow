"""The registered four-route Kokoro delivery contract, shared by build and load.

No model execution belongs here. The exporter owns graph/asset paths; callers
provide either a confined file resolver (build) or an inventoried one (load).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np


_ROUTES = ((32, 160), (64, 320), (128, 640), (256, 1280))
_ROLES = ("text_duration", "frame_acoustic", "phase_core", "generator_istft")
_PHASE_BOUNDARY = {"execution": "host_onnxruntime_fp32", "input": "f0", "output": "sine"}
FileResolver = Callable[[Any, str], Path]


def load_metadata(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Kokoro metadata must contain a JSON object: {path.name}")
    return data


def validate_runtime_contract(
    meta: Mapping[str, Any], *, components: tuple[str, ...], resolve_file: FileResolver
) -> tuple[Path, Path]:
    """Validate the release and return its packaged config and voice paths."""
    if meta.get("schema_version") != 2 or meta.get("graph_mode") != "bucketed_cpu_phase":
        raise ValueError("unsupported Kokoro export schema; re-export with the CPU phase workflow")
    if meta.get("export_complete") is not True or meta.get("hmonnx_converted") is not True:
        raise ValueError("Kokoro public artifact requires a complete, converted HMONNX export")
    if meta.get("sample_rate") != 24000 or meta.get("phase_boundary") != _PHASE_BOUNDARY:
        raise ValueError("Kokoro requires 24 kHz audio and the complete f0 -> sine Host FP32 boundary")
    if meta.get("target_device") != "XH2a" or meta.get("quant_type") != "w16a16_sefp":
        raise ValueError("Kokoro requires the approved XH2a w16a16_sefp deployment precision")
    seed = meta.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not -(1 << 63) <= seed <= (1 << 64) - 1:
        raise ValueError("Kokoro runtime seed must be a valid 64-bit integer")
    if not isinstance(meta.get("release_prefix"), str) or not meta["release_prefix"]:
        raise ValueError("Kokoro export must identify its release_prefix")

    presets = meta.get("bucket_presets")
    routes = presets.get("routes") if isinstance(presets, Mapping) else None
    if (
        not isinstance(presets, Mapping)
        or presets.get("policy") != "duration_driven_paired_t_f"
        or not isinstance(routes, list)
        or len(routes) != len(_ROUTES)
    ):
        raise ValueError("Kokoro public profile requires all four paired T/F routes")
    expected = {f"t{t:04d}_f{f:04d}": (t, f) for t, f in _ROUTES}
    actual = {}
    for route in routes:
        if not isinstance(route, Mapping) or not isinstance(route.get("key"), str):
            raise ValueError("Kokoro paired routes must contain keyed metadata objects")
        key = route["key"]
        capacities = (route.get("token_max_length"), route.get("frame_max_length"))
        if key in actual or any(type(value) is not int for value in capacities):
            raise ValueError("Kokoro paired route capacities must be unique integers")
        actual[key] = capacities
    if actual != expected:
        raise ValueError("Kokoro paired routes do not match the registered profile")

    declared = meta.get("components")
    if (
        tuple(components) != _ROLES
        or meta.get("component_order") != list(_ROLES)
        or not isinstance(declared, Mapping)
        or set(declared) != set(_ROLES)
    ):
        raise ValueError("Kokoro component set/order does not match the registered recipe")
    for role in _ROLES:
        _validate_buckets(role, declared[role], resolve_file)

    assets = meta.get("runtime_assets")
    voice = assets.get("voice_pack") if isinstance(assets, Mapping) else None
    if not isinstance(voice, Mapping) or voice.get("name") != "zf_001":
        raise ValueError("Kokoro artifact requires the registered zf_001 voice pack")
    config_path = resolve_file(assets.get("config_file"), "runtime_assets.config_file")
    _validate_config(config_path)
    voice_path = resolve_file(voice.get("file"), "runtime_assets.voice_pack.file")
    _validate_voice(voice_path, voice)
    return config_path, voice_path


def _validate_buckets(role: str, component: Any, resolve_file: FileResolver) -> None:
    text = role == "text_duration"
    field = "token_max_length" if text else "frame_max_length"
    expected = {f"t{t:04d}" if text else f"f{f:04d}": t if text else f for t, f in _ROUTES}
    buckets = component.get("buckets") if isinstance(component, Mapping) else None
    if not isinstance(buckets, Mapping) or set(buckets) != set(expected):
        raise ValueError(f"Kokoro {role} bucket metadata must cover all registered routes")
    for key, capacity in expected.items():
        bucket = buckets[key]
        label = f"components.{role}.buckets.{key}"
        if not isinstance(bucket, Mapping) or type(bucket.get(field)) is not int or bucket[field] != capacity:
            raise ValueError(f"Kokoro {label} has invalid capacity metadata")
        if "hmonnx_variants" in bucket:
            raise ValueError("Kokoro native/decomposed variant layouts are retired; re-export the model")
        resolve_file(bucket.get("onnx_file"), f"{label}.onnx_file")
        if role == "phase_core":
            if bucket.get("inputs") != [{"name": "f0", "dtype": "FLOAT", "shape": [1, 2 * capacity]}]:
                raise ValueError("Kokoro phase_core must consume f0 [1,2F] in FP32, not phase increments")
            if bucket.get("outputs") != [{"name": "sine", "dtype": "FLOAT", "shape": [1, 600 * capacity, 9]}]:
                raise ValueError("Kokoro phase_core must produce sine [1,600F,9] in FP32")
            if "hmonnx_file" in bucket:
                raise ValueError("Kokoro phase_core must remain a CPU FP32 ONNX graph")
        else:
            resolve_file(bucket.get("hmonnx_file"), f"{label}.hmonnx_file")
            counts = bucket.get("hmonnx_op_counts")
            if not isinstance(counts, Mapping) or not counts or counts.get("LSTM", 0) != 0:
                raise ValueError(f"Kokoro {label} must use decomposed-only HMONNX")
        _validate_file_references(bucket, resolve_file, label)


def _validate_file_references(value: Any, resolve_file: FileResolver, label: str) -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_file_references(item, resolve_file, f"{label}[{index}]")
    elif isinstance(value, Mapping):
        for key, item in value.items():
            nested = f"{label}.{key}"
            if key in {"file", "onnx_file", "hmonnx_file"}:
                resolve_file(item, nested)
            else:
                _validate_file_references(item, resolve_file, nested)


def _validate_config(path: Path) -> None:
    config = load_metadata(path)
    vocab = config.get("vocab")
    if (
        not isinstance(vocab, Mapping)
        or not vocab
        or any(
            not isinstance(symbol, str)
            or not symbol
            or type(token_id) is not int
            or not 0 <= token_id <= np.iinfo(np.int32).max
            for symbol, token_id in vocab.items()
        )
    ):
        raise ValueError("Kokoro runtime config must contain a usable int32 vocabulary")


def _validate_voice(path: Path, meta: Mapping[str, Any]) -> None:
    if path.suffix != ".npy":
        raise ValueError("Kokoro voice pack must use .npy")
    try:
        voice = np.load(path, allow_pickle=False, mmap_mode="r")
    except Exception as error:
        raise ValueError(f"Kokoro voice pack cannot be loaded with allow_pickle=False: {error}") from error
    if not isinstance(voice, np.ndarray):
        raise ValueError("Kokoro voice pack must contain one NumPy array")
    if voice.size == 0:
        raise ValueError("Kokoro voice pack must be non-empty")
    if not np.issubdtype(voice.dtype, np.number) or np.issubdtype(voice.dtype, np.complexfloating):
        raise ValueError("Kokoro voice pack must contain real numeric values")
    if not np.isfinite(voice).all():
        raise ValueError("Kokoro voice pack contains non-finite values")
    if voice.dtype != np.dtype(np.float32):
        raise ValueError("Kokoro voice pack must use canonical float32 values")
    shape = (1, voice.shape[0]) if voice.ndim == 1 else tuple(voice.shape[-2:])
    if voice.ndim not in {1, 2, 3} or shape != (1, 256):
        raise ValueError(f"Kokoro voice pack cannot produce a [1,256] style: {voice.shape}")
    for field, actual in (
        ("shape", list(voice.shape)),
        ("dtype", str(voice.dtype)),
        ("size_bytes", path.stat().st_size),
    ):
        if meta.get(field) != actual:
            raise ValueError(f"Kokoro voice metadata {field} does not match the voice pack")
