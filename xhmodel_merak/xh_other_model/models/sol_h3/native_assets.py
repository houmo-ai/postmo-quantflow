# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Local native H3 sources and explicitly named, independent adapter branches."""

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NativeSourceAssets:
    root: Path
    paths: dict

    @classmethod
    def read(cls, model_dir):
        root = Path(model_dir).expanduser().resolve(strict=True)
        inventory = root / "checkpoint-paths.json"
        value = json.loads(inventory.read_text())["h3_model"] if inventory.is_file() else "."
        if not isinstance(value, str) or not value:
            raise ValueError("native H3 inventory requires a local h3_model path")
        source = Path(value).expanduser()
        source = (source if source.is_absolute() else root / source).resolve(strict=True)
        for component in ("transformer", "transformer_ref", "text_encoder", "vae", "audio_vae"):
            if not (source / component / "config.json").is_file():
                raise ValueError(f"native H3 source is missing {component}/config.json")
        for component in ("tokenizer", "processor"):
            if not (source / component).is_dir():
                raise ValueError(f"native H3 source is missing {component}")
        return cls(root, {"h3_model": source})

    def __getitem__(self, name):
        return self.paths[name]


def read_native_adapters(paths):
    """Read directory manifests without merging adapter tensors into base weights."""
    from .native_lora import inspect_native_lora
    from .native_schedule import native_schedule

    if paths is None:
        paths = []
    if not isinstance(paths, (list, tuple)):
        raise ValueError("native H3 model.lora.path must be a list of adapter directories")
    adapters = {}
    for path in paths:
        root = Path(path).expanduser().resolve(strict=True)
        manifest = root / "h3_adapter.json"
        if manifest.is_symlink():
            raise ValueError("adapter manifest must not be a symlink")
        data = json.loads(manifest.read_text())
        if not isinstance(data, dict) or set(data) != {"name", "branches", "sampling"}:
            raise ValueError("h3_adapter.json requires exactly name, branches and sampling")
        name, branches, sampling = data["name"], data["branches"], data["sampling"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name) or name == "standard":
            raise ValueError("adapter name must be a lowercase identifier other than standard")
        if name in adapters:
            raise ValueError(f"duplicate native H3 adapter name: {name}")
        if not isinstance(branches, dict) or not branches or not set(branches) <= {"base", "ref"}:
            raise ValueError("native H3 adapter branches must contain base and/or ref")
        if not isinstance(sampling, dict) or set(sampling) != {"steps", "video_shift", "audio_shift"}:
            raise ValueError("native H3 adapter sampling must explicitly specify steps and both shifts")
        schedule = native_schedule(**sampling)
        specs = {}
        for branch, weight in branches.items():
            if not isinstance(weight, str) or not weight:
                raise ValueError("adapter branch weights must be local paths")
            relative = Path(weight)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("adapter weights must be relative files inside their adapter directory")
            target = root / relative
            if any(root.joinpath(*relative.parts[:index]).is_symlink() for index in range(1, len(relative.parts) + 1)):
                raise ValueError("adapter weights must not traverse symlinks")
            target = target.resolve(strict=True)
            if not target.is_relative_to(root) or not target.is_file():
                raise ValueError("adapter weights must be files inside their adapter directory")
            specs[branch] = inspect_native_lora(target, name=name)
        adapters[name] = dict(sampling=dict(sampling), schedule=schedule, branches=specs)
    return adapters
