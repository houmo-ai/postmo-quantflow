"""Decoder geometry and runtime files shared by MOSS build and load."""
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

FileResolver = Callable[[Any, str], Path]

REQUIRED_RUNTIME_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "preprocessor_config.json",
    "configuration_moss_transcribe_diarize.py",
    "modeling_moss_transcribe_diarize.py",
    "processing_moss_transcribe_diarize.py",
)


def load_metadata(path: Path) -> dict[str, Any]:
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"MOSS metadata must contain a JSON object: {path.name}")
    return data


def _published_graph(component: Mapping[str, Any], *, phase: str) -> str:
    onnx = component.get("onnx_file")
    hmonnx = component.get("hmonnx_file")
    if isinstance(hmonnx, str) and hmonnx:
        return hmonnx
    if isinstance(onnx, str) and onnx:
        return onnx
    raise ValueError(f"missing published {phase} graph")


def published_runtime_graphs(meta: Mapping[str, Any]) -> tuple[str, ...]:
    """Published graphs are HMONNX; float ONNX is export-time only."""
    graphs = [_published_graph(meta["components"]["encode"], phase="encode")]
    decoder = meta["components"]["prefill_decode"]
    for phase in ("prefill", "decode"):
        graphs.append(_published_graph(decoder[phase], phase=phase))
    graphs.append(meta["weights_file"])
    return tuple(graphs)


def validate_decoder_contract(meta: Mapping[str, Any]) -> None:
    try:
        decoder = meta["components"]["prefill_decode"]
        geometries = []
        for phase in ("prefill", "decode"):
            component = decoder[phase]
            if component["rope_mode"] != "fp32_table_integer_index":
                raise ValueError("dynamic-angle RoPE artifacts must be re-exported")
            sizes = tuple(component[name] for name in (
                "prefill_length", "cache_length", "rope_max_positions",
            ))
            if any(type(size) is not int or size <= 0 for size in sizes):
                raise ValueError("decoder dimensions must be positive integers")
            prefill, cache, rope = sizes
            if not prefill < cache <= rope:
                raise ValueError("require prefill_length < cache_length <= rope_max_positions")
            geometries.append(sizes)
        if geometries[0] != geometries[1]:
            raise ValueError("prefill and decode geometry/rotary tables must agree")
    except (KeyError, TypeError) as error:
        raise ValueError("missing MOSS rotary/geometry contract; re-export this artifact") from error


def validate_runtime_contract(meta: Mapping[str, Any], *, resolve_file: FileResolver) -> None:
    """Shared build/load checks: w8a8 graphs, rotary ABI, and packaged runtime files."""
    if meta.get("quant_type") != "w8a8_sefp":
        raise ValueError("MOSS artifacts must publish w8a8_sefp HMONNX graphs")
    validate_decoder_contract(meta)
    components = meta["components"]
    for relative in published_runtime_graphs(meta):
        resolve_file(relative, "runtime_file")
    resolve_file(meta.get("lm_head_file", "weights.pt"), "lm_head_file")
    for component in (
        components["encode"],
        components["prefill_decode"]["prefill"],
        components["prefill_decode"]["decode"],
    ):
        hmonnx = component.get("hmonnx_file")
        if hmonnx:
            resolve_file(hmonnx, "hmonnx_file")
    for name in REQUIRED_RUNTIME_FILES:
        resolve_file(name, "runtime_resource")
