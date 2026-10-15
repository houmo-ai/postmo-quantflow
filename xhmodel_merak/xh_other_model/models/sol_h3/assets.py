# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Local-only source inventory. Exported runtime resources use a separate manifest."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourceAssets:
    root: Path
    paths: dict[str, Path]

    @classmethod
    def read(cls, model_dir):
        root = Path(model_dir).expanduser().resolve()
        data = json.loads((root / "checkpoint-paths.json").read_text())
        defaults = {
            "sana_source": "reproduction/Sana",
            "ltx_source": "reproduction/LTX-2",
            "upscaler_source": "reproduction/upscaler-source.py",
            "refiner_prompt_cache": "reproduction/fixed-prompt-int8-a100.pt",
        }
        data = {**defaults, **data}
        names = (
            "h3_model",
            "dense_lora",
            "transformer",
            "refiner_lora",
            "output_video_vae",
            "audio_vae",
            "h3_upscaler_checkpoint",
            "adapter_dir",
            *defaults,
        )
        paths = {}
        for name in names:
            value = data[name]
            if not isinstance(value, str) or not value:
                raise ValueError(f"source inventory requires a local path for {name}")
            path = Path(value).expanduser()
            path = (path if path.is_absolute() else root / path).resolve(strict=True)
            paths[name] = path
        if "ref2va_lora" in data:
            path = Path(data["ref2va_lora"]).expanduser()
            paths["ref2va_lora"] = (path if path.is_absolute() else root / path).resolve()
        for name in (
            "dense_lora",
            "transformer",
            "refiner_lora",
            "output_video_vae",
            "audio_vae",
            "h3_upscaler_checkpoint",
            "upscaler_source",
            "refiner_prompt_cache",
        ):
            if not paths[name].is_file():
                raise ValueError(f"{name} must name a local file")
        for relative in (
            "transformer/config.json",
            "text_encoder/config.json",
            "tokenizer/tokenizer.json",
            "audio_vae/config.json",
        ):
            if not (paths["h3_model"] / relative).is_file():
                raise ValueError(f"H3 source is missing {relative}")
        if not (paths["adapter_dir"] / "model.safetensors").is_file():
            raise ValueError("H3-to-LTX adapter checkpoint is missing")
        return cls(root, paths)

    def __getitem__(self, name):
        return self.paths[name]

    def activate_build_sources(self):
        """Import author libraries from explicit build inputs, without vendoring them."""
        source = self["sana_source"] / "models/minimax_h3/Sol-H3-Spark"
        paths = [source, self["sana_source"]]
        paths.extend(self["ltx_source"] / "packages" / name / "src" for name in ("ltx-core", "ltx-pipelines"))
        for path in paths:
            if not path.is_dir():
                raise ValueError(f"native build source directory is missing: {path}")
        sys.path[:0] = [str(path) for path in paths if str(path) not in sys.path]


def read_prompt_cache(path, *, dimensions=(4096, 2048)):
    import torch

    from .precision import require_finite

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or not payload.get("fingerprint") or not payload.get("prompt"):
        raise ValueError("refiner cache requires the native fixed-prompt recipe and fingerprint")
    contexts = payload["contexts"]
    for role, dim in zip(("video", "audio"), dimensions, strict=True):
        value = contexts[role]
        if value.shape != (1, 1024, dim) or value.dtype != torch.bfloat16:
            raise ValueError(f"native INT8 {role} cache has unexpected shape or dtype")
        require_finite(value, name=f"native {role} prompt cache")
    return payload
