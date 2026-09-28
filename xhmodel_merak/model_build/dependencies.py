"""Export dependency declarations without installing packages or importing models."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from xhmodel_merak.model_catalog import get_recipe
from xhmodel_merak.model_catalog.dependencies import dependency_profile


def write_dependency_files(root: Path, model_id: str, variant: str) -> dict:
    recipe = get_recipe(model_id)
    selected = recipe.get_variant(variant)
    if selected is None:
        raise ValueError(f"Unknown variant {variant!r} for {model_id!r}")
    profile = dependency_profile(recipe, selected)
    environment_file = root / "environment.json"
    if environment_file.exists() or environment_file.is_symlink():
        raise FileExistsError(environment_file)
    folder = root / "environments"
    folder.mkdir()  # Never overwrite an existing directory or follow a symlink.
    phases = {key: profile[key] for key in ("build", "inference", "checkpoint_build")}
    phases.update({f"quant-{key}": value for key, value in profile["quantization"].items()})
    for name, phase in phases.items():
        if name not in {"build", "inference", "checkpoint_build", "quant-gptq", "quant-autoround", "quant-qat"}:
            raise ValueError(f"Unsupported dependency phase: {name}")
        content = (
            "# Candidate direct requirements; not a transitive lockfile.\n"
            f"# Status: {profile['status']}; see ../environment.json before installation.\n"
            + "\n".join(phase["pip"]) + "\n"
        )
        relative = f"environments/{name}.txt"
        (root / relative).write_text(content, encoding="utf-8")
        phase["requirements_file"] = relative
        phase["sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    record = {
        "schema_version": 1,
        "kind": "declared_requirements",
        "model_id": model_id,
        "variant": variant,
        "path_base": "artifact_root",
        **profile,
    }
    environment_file.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record


def export_model_dependencies(*, model_id: str, output_dir: str, variant: str | None = None) -> dict:
    """Write a new directory of candidate requirements and their validation status.

    No package installation occurs. A blocked/pending profile is not certified
    installable, even though the requirement lines are pip-compatible syntax.
    """
    recipe = get_recipe(model_id)
    selected = recipe.default_variant.name if variant is None else variant
    if recipe.get_variant(selected) is None:
        raise ValueError(f"Unknown variant {selected!r} for {model_id!r}")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    return write_dependency_files(root, model_id, selected)
