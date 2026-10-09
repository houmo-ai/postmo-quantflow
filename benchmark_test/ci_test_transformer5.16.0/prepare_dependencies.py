"""Prepare selected declared packages in an existing CI environment.

This supplements the configured CI environment; it is not a clean-install
validator for a complete blocked dependency profile. No installation occurs
unless --install is explicitly supplied. Versions come only from the recipe.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from xhmodel_merak.model_catalog import get_recipe  # noqa: E402
from xhmodel_merak.model_catalog.dependencies import dependency_profile  # noqa: E402


def select_requirements(model_id, phase, packages, variant=None):
    recipe = get_recipe(model_id)
    selected = recipe.default_variant if variant is None else recipe.get_variant(variant)
    if selected is None:
        raise ValueError(f"Unknown variant {variant!r} for {model_id!r}")
    requested = {canonicalize_name(name) for name in packages}
    if not requested:
        raise ValueError("Select at least one declared package")
    profile = dependency_profile(recipe, selected)
    requirements = [line for line in profile[phase]["pip"]
                    if canonicalize_name(Requirement(line).name) in requested]
    missing = requested - {canonicalize_name(Requirement(line).name) for line in requirements}
    if missing:
        raise ValueError(f"Packages are not declared for {model_id}/{selected.name}/{phase}: {sorted(missing)}")
    return requirements


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--variant")
    parser.add_argument("--phase", choices=("build", "inference", "checkpoint_build"), default="build")
    parser.add_argument("--packages", nargs="+", required=True)
    parser.add_argument("--install", action="store_true", help="Install using this same Python interpreter")
    args = parser.parse_args(argv)
    requirements = select_requirements(args.model_id, args.phase, args.packages, args.variant)
    command = [sys.executable, "-m", "pip", "install", *requirements]
    print(json.dumps(dict(requirements=requirements, command=command), indent=2), flush=True)
    if args.install:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
