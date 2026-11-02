# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Mode views over one shared artifact inventory; no weight copies or sessions."""

from copy import deepcopy

from .capacity import MODES


BRANCH_GRAPHS = ("h3_text_refiner", "h3_denoiser")


def mode_metadata(metadata, mode):
    """Select the native mode ABI while keeping every path relative to one root."""
    if metadata.get("mode") != "all" or mode not in MODES:
        raise ValueError("mode views require a unified bundle and a supported input mode")
    profiles, branches = metadata.get("mode_profiles", {}), metadata.get("branches", {})
    if set(profiles) != set(MODES) or set(branches) != {"base", "ref"}:
        raise ValueError("unified Sol-H3 requires five mode profiles and two H3 branches")
    selected = profiles[mode]
    branch_name = "ref" if mode == "ref2va" else "base"
    if selected.get("branch") != branch_name:
        raise ValueError("Sol-H3 input mode selects the wrong H3 weight branch")
    branch = branches[branch_name]
    aliases = branch.get("graphs", {})
    if set(aliases) != set(BRANCH_GRAPHS):
        raise ValueError("each H3 branch requires its own text refiner and denoiser")
    physical = [name for item in branches.values() for name in item.get("graphs", {}).values()]
    if len(set(physical)) != 4 or any(name not in metadata["graphs"] for name in physical):
        raise ValueError("H3 branch graphs must identify four distinct exported components")
    if len({metadata["graphs"][name]["path"] for name in physical}) != 4:
        raise ValueError("H3 branch graphs cannot alias the same exported weight file")
    resource = selected.get("input_config")
    if resource not in metadata["resources"]:
        raise ValueError("Sol-H3 mode is missing its native input contract")
    result = deepcopy(metadata)
    result.update(
        schema_version=metadata["schema_version"] if metadata.get("schema_version") in (8, 9) else 4,
        mode=mode,
        profile=selected["profile"],
        capacity_plan=deepcopy(selected["capacity_plan"]),
        text_capacity=selected["capacity_plan"]["text_capacity"],
        h3_adapter=deepcopy(branch["h3_adapter"]),
        h3_rope_config=deepcopy(branch["h3_rope_config"]),
        h3_role_layout=deepcopy(branch["h3_role_layout"]),
        folded_h3_projection_parameters=branch["folded_h3_projection_parameters"],
    )
    result["graphs"] = {name: value for name, value in result["graphs"].items() if name not in physical}
    result["graphs"].update({name: deepcopy(metadata["graphs"][target]) for name, target in aliases.items()})
    result["resources"]["input_config"] = metadata["resources"][resource]
    return result
