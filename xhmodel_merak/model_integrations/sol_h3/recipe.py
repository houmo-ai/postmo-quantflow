# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""H3 profile selection over the existing full and LoRA recipe variants."""

from dataclasses import dataclass, replace

from xhmodel_merak.model_catalog.contracts import ModelRecipe, VariantRecipe


@dataclass(frozen=True)
class H3Recipe(ModelRecipe):
    workflows_by_profile: tuple[tuple[str, str], ...] = ()

    def __post_init__(self):
        super().__post_init__()
        if self.workflows_by_profile:
            profiles = [profile for profile, _ in self.workflows_by_profile]
            if len(profiles) != len(set(profiles)) or set(profiles) != set(self.profiles):
                raise ValueError("H3 workflows must map each registered profile exactly once")

    def get_variant(self, name: str) -> VariantRecipe | None:
        variant = super().get_variant(name)
        if variant is not None or not isinstance(name, str):
            return variant
        lora = name.startswith("lora_")
        profile = name.removeprefix("lora_") if lora else name
        workflow = dict(self.workflows_by_profile).get(profile)
        base = super().get_variant("lora" if lora else "full")
        if workflow is None or base is None:
            return None
        return replace(base, workflow_config=workflow)
