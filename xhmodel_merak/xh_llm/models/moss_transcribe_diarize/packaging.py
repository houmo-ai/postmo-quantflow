"""Copy Official processor/tokenizer files into an export or build directory."""

from __future__ import annotations

import shutil
from pathlib import Path

OFFICIAL_RUNTIME_FILES = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "vocab.json",
    "merges.txt",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "processor_config.json",
    "configuration_moss_transcribe_diarize.py",
    "modeling_moss_transcribe_diarize.py",
    "processing_moss_transcribe_diarize.py",
)


def copy_official_runtime_files(model_dir: str | Path, destination: str | Path) -> None:
    source = Path(model_dir)
    dest = Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    for name in OFFICIAL_RUNTIME_FILES:
        path = source / name
        if path.is_file():
            shutil.copy2(path, dest / name)
    hf_config = source / "config.json"
    if hf_config.is_file():
        hf_dir = dest / "hf_config"
        hf_dir.mkdir(parents=True, exist_ok=True)
        if not (hf_dir / "config.json").is_file():
            shutil.copy2(hf_config, hf_dir / "config.json")


__all__ = ["OFFICIAL_RUNTIME_FILES", "copy_official_runtime_files"]
