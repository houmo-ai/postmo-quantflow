from __future__ import annotations

from pathlib import Path


TCIM_SUFFIXES = (".hmm", ".hmms")


def resolve_tcim_model_path(model_path: str | Path) -> Path:
    path = Path(model_path).expanduser()
    if path.suffix.lower() not in TCIM_SUFFIXES:
        raise ValueError(f"TCIM metadata must reference an explicit .hmm/.hmms artifact, got: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()
