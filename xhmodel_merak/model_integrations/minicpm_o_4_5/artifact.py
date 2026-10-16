"""The MiniCPM graph/resource contract shared by build and inference."""

from __future__ import annotations

import json
import posixpath
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


def graph_roles(mode: str) -> dict[str, tuple[str, ...]]:
    return {
        "vision": ("main",),
        "audio": ("main",) if mode == "offline" else (f"{mode}_prefill", f"{mode}_decode"),
        "llm": ("prefill", "decode"), "tts": ("prefill", "decode"),
        "speaker": ("campplus", "speech_tokenizer"),
        "token2wav_flow_frontend": ("main",) if mode == "offline" else (
            "stream_flow_prompt_frontend", "stream_flow_frontend",
        ),
        "token2wav_flow_decoder": ("main",) if mode == "offline" else (
            "stream_flow_prompt_estimator_step", "stream_flow_estimator_step",
        ),
        "token2wav_hift": ("part1", "part2") if mode == "offline" else (
            "stream_hift_part1", "stream_hift_part2",
        ),
    }


def load_metadata(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_runtime_contract(
    meta: Mapping[str, Any],
    *,
    resolve_file: Callable[[Any, str], Path],
    deployment_precision: Mapping[str, str],
    variant: str,
) -> None:
    if meta["model_type"] != "MiniCPM-o-4.5" or meta["export_mode"] != variant:
        raise ValueError("MiniCPM-o-4.5 export mode does not match the build manifest")
    if meta["target_device"] != "XH2a":
        raise ValueError("MiniCPM-o-4.5 artifacts must target XH2a")
    components = meta["components"]
    expected_roles = graph_roles(variant)
    if set(components) != set(expected_roles):
        raise ValueError("MiniCPM-o-4.5 requires the complete multimodal component set")
    for name, roles in expected_roles.items():
        component = components[name]
        if component["quant_type"] != deployment_precision[name]:
            raise ValueError(f"{name} graph precision does not match the build manifest")
        if set(component["graphs"]) != set(roles):
            raise ValueError(f"{name} graph roles do not match the {variant} runtime")
        for role in roles:
            resolve_file(component["graphs"][role], f"components.{name}.graphs.{role}")
    for name, field in (("llm", "token_embedding_file"), ("tts", "text_embedding"), ("tts", "code_embedding")):
        resolve_file(components[name][field], f"components.{name}.{field}")
    for name in ("llm", "tts"):
        component = components[name]
        child = load_metadata(resolve_file(component["metadata"], f"components.{name}.metadata"))
        parent = Path(component["metadata"]).parent
        hf_config = posixpath.normpath((parent / child["hf_config"]).as_posix())
        if hf_config != meta["hf_config"]:
            raise ValueError(f"{name} child metadata disagrees with the packaged HF config")
        if name == "llm":
            embedding = posixpath.normpath((parent / child["quant_embedding"]).as_posix())
            if embedding != component["token_embedding_file"]:
                raise ValueError("LLM child metadata disagrees with the packaged embedding")
        for role in ("prefill", "decode"):
            relative = posixpath.normpath((parent / child[f"{role}_hmonnx"]).as_posix())
            resolve_file(relative, f"components.{name}.{role}_hmonnx")
            if relative != component["graphs"][role]:
                raise ValueError(f"{name} child metadata disagrees with the published {role} graph")
    for name in (
        "config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json", "preprocessor_config.json",
        "configuration_minicpmo.py", "modeling_minicpmo.py", "modeling_navit_siglip.py",
        "processing_minicpmo.py", "tokenization_minicpmo_fast.py", "utils.py",
    ):
        resolve_file(f"{meta['hf_config']}/{name}", "hf_config")


def describe_export(meta: Mapping[str, Any]) -> dict[str, Any]:
    entries = {}
    for component, roles in graph_roles(meta["export_mode"]).items():
        for role in roles:
            name = role if component == "llm" else component if role == "main" else f"{component}_{role}"
            entries[name] = {
                "path": meta["components"][component]["graphs"][role],
                "type": role if component in {"llm", "tts"} else "hmonnx",
            }
    entries["tokenizer"] = {"path": meta["hf_config"], "type": "tokenizer"}
    for component, field, name in (
        ("llm", "token_embedding_file", "quant_embedding"),
        ("tts", "text_embedding", "tts_text_embedding"),
        ("tts", "code_embedding", "tts_code_embedding"),
    ):
        entries[name] = {"path": meta["components"][component][field], "type": "embedding"}
    return entries
