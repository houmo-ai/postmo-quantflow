# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Pinned native MiniCPM implementation for the Transformers 5.16 CI lane."""

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


@pytest.fixture(scope="module", autouse=True)
def minicpmv47_modeling():
    from transformers import AutoModelForImageTextToText
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    root = Path(__file__).parent / "fixtures/minicpmv4_7"
    manifest = json.loads((root / "source.json").read_text())
    for name, expected in manifest["files"].items():
        data = (root / name).read_bytes()
        assert len(data) == expected["size"]
        assert hashlib.sha256(data).hexdigest() == expected["sha256"]

    package_name = "transformers.models.minicpmv4_7"
    # Keep the reference fixed even if the developer has a newer Transformers.
    # Only this test module sees the extra model; installed packages stay intact.
    with pytest.MonkeyPatch.context() as patch:
        package = ModuleType(package_name)
        package.__path__ = [str(root)]
        patch.setitem(sys.modules, package_name, package)
        for name in ("configuration_minicpmv4_7", "modeling_minicpmv4_7"):
            qualified = f"{package_name}.{name}"
            spec = importlib.util.spec_from_file_location(qualified, root / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            patch.setitem(sys.modules, qualified, module)
            setattr(package, name, module)
            spec.loader.exec_module(module)
        patch.setattr(CONFIG_MAPPING, "_extra_content", dict(CONFIG_MAPPING._extra_content))
        mapping = AutoModelForImageTextToText._model_mapping
        patch.setattr(mapping, "_extra_content", dict(mapping._extra_content))
        yield importlib.import_module(f"{package_name}.modeling_minicpmv4_7")
