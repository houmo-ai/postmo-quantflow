from __future__ import annotations

import os
from pathlib import Path


_CONFIG_ROOT_ENV = "XHMODEL_MERAK_CONFIG_ROOT"


def resolve_workflow_config(relative_path: str) -> Path:
    configured_root = os.getenv(_CONFIG_ROOT_ENV)
    if configured_root:
        root = Path(configured_root).expanduser().resolve()
        candidate = root / relative_path
        if not candidate.is_file() and relative_path.startswith("configs_merak/"):
            candidate = root / relative_path.removeprefix("configs_merak/")
        if candidate.is_file():
            return candidate.resolve()
        raise FileNotFoundError(f"Workflow config does not exist under {_CONFIG_ROOT_ENV}: {candidate}")

    for root in Path(__file__).resolve().parents:
        candidate = root / relative_path
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Packaged workflow config does not exist: {relative_path}")


__all__ = ["resolve_workflow_config"]
