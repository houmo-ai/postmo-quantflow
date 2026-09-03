from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from xhmodel_merak import _requirements as requirement_checks
from xhmodel_merak.errors import ModelError


def test_package_metadata_matches_supported_worker_runtime():
    metadata = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8"))
    project = metadata["project"]
    python_range = SpecifierSet(project["requires-python"])
    poetry_python_range = SpecifierSet(metadata["tool"]["poetry"]["dependencies"]["python"])
    transformers = next(
        Requirement(value) for value in project["dependencies"] if Requirement(value).name == "transformers"
    )

    assert Version("3.12") in python_range
    assert Version("3.12") in poetry_python_range
    assert Version("3.13") not in python_range
    assert Version("3.13") not in poetry_python_range
    assert Version("4.57") in transformers.specifier
    assert Version("5.13.9") in transformers.specifier
    assert Version("5.16.1") in transformers.specifier
    assert Version("5.17") not in transformers.specifier


def test_requirement_validator_uses_local_module_version(monkeypatch):
    monkeypatch.setattr(
        requirement_checks,
        "_installed_distribution_version",
        lambda _name: None,
    )
    monkeypatch.setattr(
        requirement_checks.importlib,
        "import_module",
        lambda _name: SimpleNamespace(__version__="5.8.1"),
    )

    requirement_checks.validate_worker_requirements(
        ("gptqmodel>=5.8,<5.9",),
        model="test-model",
        operation="build",
    )


def test_requirement_validator_reports_invalid_local_module_version(monkeypatch):
    monkeypatch.setattr(
        requirement_checks,
        "_installed_distribution_version",
        lambda _name: None,
    )
    monkeypatch.setattr(
        requirement_checks.importlib,
        "import_module",
        lambda _name: SimpleNamespace(__version__="not-pep440"),
    )

    with pytest.raises(ModelError) as caught:
        requirement_checks.validate_worker_requirements(
            ("gptqmodel>=5.8,<5.9",),
            model="test-model",
            operation="build",
        )

    assert caught.value.details["incompatible_requirements"] == [
        {
            "requirement": "gptqmodel>=5.8,<5.9",
            "installed": "not-pep440",
        }
    ]


def test_requirement_validator_handles_a_loaded_module_without_a_spec(monkeypatch):
    def missing_spec(_name):
        raise ValueError

    monkeypatch.setattr(requirement_checks, "_installed_distribution_version", lambda _name: None)
    monkeypatch.setattr(requirement_checks.importlib.util, "find_spec", missing_spec)

    with pytest.raises(ModelError) as caught:
        requirement_checks.validate_worker_requirements(("xhquant",), model="test-model", operation="build")

    assert caught.value.details["incompatible_requirements"] == [{"requirement": "xhquant", "installed": None}]


def test_requirement_validator_checks_registered_python_symbol(monkeypatch):
    monkeypatch.setattr(
        requirement_checks.importlib,
        "import_module",
        lambda _name: SimpleNamespace(recipe=lambda: None),
    )

    requirement_checks.validate_worker_capabilities(
        ("gptqmodel.recipes.test:recipe",),
        model="test-model",
        operation="build",
    )


def test_requirement_validator_reports_missing_python_symbol(monkeypatch):
    monkeypatch.setattr(
        requirement_checks.importlib,
        "import_module",
        lambda _name: SimpleNamespace(),
    )

    with pytest.raises(ModelError) as caught:
        requirement_checks.validate_worker_capabilities(
            ("gptqmodel.recipes.test:recipe",),
            model="test-model",
            operation="build",
        )

    assert caught.value.details["incompatible_capabilities"] == ["gptqmodel.recipes.test:recipe"]


def test_requirement_validator_executes_registered_capability_predicate(monkeypatch):
    monkeypatch.setattr(
        requirement_checks.importlib,
        "import_module",
        lambda _name: SimpleNamespace(available=lambda: True),
    )

    requirement_checks.validate_worker_capabilities(
        ("python-predicate:gptqmodel.recipes.test:available",),
        model="test-model",
        operation="build",
    )


def test_requirement_validator_reports_failed_capability_predicate(monkeypatch):
    monkeypatch.setattr(
        requirement_checks.importlib,
        "import_module",
        lambda _name: SimpleNamespace(available=lambda: False),
    )
    capability = "python-predicate:gptqmodel.recipes.test:available"

    with pytest.raises(ModelError) as caught:
        requirement_checks.validate_worker_capabilities(
            (capability,),
            model="test-model",
            operation="build",
        )

    assert caught.value.details["incompatible_capabilities"] == [capability]
