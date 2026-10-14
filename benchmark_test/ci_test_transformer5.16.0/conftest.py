"""Pytest bootstrap for the active ModelZoo CI lane."""

from __future__ import annotations

from contextlib import contextmanager
import os
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _prepend_pythonpath(path: Path) -> None:
    text = str(path)
    if path.exists() and text not in sys.path:
        sys.path.insert(0, text)


GPTQMODEL_SOURCE_DIR = os.environ.get("GPTQMODEL_SOURCE_DIR")
if GPTQMODEL_SOURCE_DIR:
    gptqmodel_root = Path(GPTQMODEL_SOURCE_DIR).expanduser().resolve()
    _prepend_pythonpath(gptqmodel_root)
    _prepend_pythonpath(gptqmodel_root / "third_party" / "auto-round")


@contextmanager
def _preserve_initialization_state():
    import torch
    import transformers.modeling_utils as modeling_utils

    initializers = {
        name: getattr(torch.nn.init, name)
        for name in ("kaiming_uniform_", "uniform_", "normal_")
    }
    missing = object()
    init_weights = getattr(modeling_utils, "_init_weights", missing)
    try:
        yield
    finally:
        for name, initializer in initializers.items():
            setattr(torch.nn.init, name, initializer)
        if init_weights is missing:
            if hasattr(modeling_utils, "_init_weights"):
                del modeling_utils._init_weights
        else:
            modeling_utils._init_weights = init_weights


@pytest.fixture(autouse=True)
def restore_model_initialization_state():
    # GPTQModel skips checkpoint initialization by changing process globals.
    # Preserve that optimization inside the test, without leaking to the next
    # native model fixture. This is test isolation, not a production workaround.
    with _preserve_initialization_state():
        yield _preserve_initialization_state


# One scheduler owns all selected per-model integration cases.
from _unified_runner import unified_runs  # noqa: E402, F401
