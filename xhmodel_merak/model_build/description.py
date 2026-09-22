"""Lightweight descriptions of the build parameters a recipe lets users change."""

from __future__ import annotations

from typing import Any

from xhmodel_merak.model_catalog.contracts import ModelRecipe


def describe_build_parameters(recipe: ModelRecipe) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "model_dir": {"required": True, "description": "Local source weights directory."},
        "output_dir": {
            "default": None,
            "description": (
                "Final artifact directory, without an extra nested release folder. "
                "None creates <release_name>_<YYYYMMDD> in the current directory. "
                "The destination must not exist and its parent must exist."
            ),
        },
        "device": {
            "default": None,
            "description": "None uses XHMODEL_BUILD_DEVICE, falling back to cuda:0.",
        },
    }

    if any(len(variant.export_modes) > 1 for variant in recipe.variants):
        parameters["low_memory"] = {
            "default": None,
            "variants": {
                variant.name: {
                    "default": variant.export_modes[0] == "low_memory",
                    "choices": [mode == "low_memory" for mode in variant.export_modes],
                }
                for variant in recipe.variants
            },
            "description": (
                "None uses the selected variant's default, independent of environment variables. "
                "True uses its layer-wise placeholder exporter. This controls export, not inference."
            ),
        }

    if len(recipe.supported_precisions) > 1:
        parameters["precision"] = {
            "default": {
                "weight": recipe.default_precision.weight,
                "activation": recipe.default_precision.activation,
            },
            "choices": [
                {"weight": weight, "activation": activation} for weight, activation in recipe.supported_precisions
            ],
            "description": (
                "Choose the checkpoint weight/activation pair; graph component choices use deployment_precision."
            ),
        }

    if recipe.deployment_precision_bindings:
        parameters["deployment_precision"] = {
            "default": None,
            "components": {
                component: {
                    "default": recipe.deployment_precision[component],
                    "choices": list(binding.choices),
                    "shared_with": list(binding.components),
                }
                for binding in recipe.deployment_precision_bindings
                for component in binding.components
            },
            "description": (
                "Component to base quant_type; omitted components follow the selected variant/context YAML. "
                "Components sharing one exporter scheme change together. "
                "Packed checkpoint weights and YAML node/operator exceptions are preserved."
            ),
        }

    methods: list[str | None] = list(recipe.quant_methods)
    if any(weight == "w16" for weight, _ in recipe.supported_precisions):
        methods.append(None)
    if len(methods) > 1:
        parameters["quant_method"] = {
            "default": recipe.quant_methods[0] if recipe.default_precision.weight != "w16" else None,
            "choices": methods,
            "description": "Omitting this argument follows precision. None requires W16 and no quant_checkpoint.",
        }
    if recipe.quant_methods:
        parameters["quant_checkpoint"] = {
            "default": None,
            "methods": list(recipe.quant_checkpoint_methods),
            "description": (
                "Reuse existing weights without quantizing again. Omit quant_method to read its producer "
                "from metadata; an explicit method must match."
            ),
        }
        parameters["quant_output_dir"] = {
            "default": None,
            "description": "Keep new quantized weights here; cannot be combined with quant_checkpoint or W16.",
        }

    if recipe.default_context_length is not None:
        parameters["context_length"] = {
            "default": recipe.default_context_length,
            "minimum": 1,
            "maximum": recipe.max_pe_length,
            "description": "Positive integer; must be at least prefill_chunk_length.",
        }
        parameters["prefill_chunk_length"] = {
            "default": recipe.default_prefill_chunk_length,
            "minimum": 1,
            "description": "Positive integer; must not exceed context_length.",
        }

    if len(recipe.profiles) > 1:
        parameters["profile"] = {"default": recipe.default_profile, "choices": list(recipe.profiles)}

    speculative_modes = list(dict.fromkeys(variant.speculative_mode for variant in recipe.variants if variant.speculative_mode))
    if speculative_modes:
        parameters["speculative"] = {
            "default": None,
            "mode": {"choices": speculative_modes},
            "num_draft_tokens": {"default": None, "minimum": 1},
            "description": "None disables speculative decoding. LoRA combinations must be registered by the model recipe.",
        }
    if any(variant.requires_lora for variant in recipe.variants):
        parameters["lora"] = {
            "default": None,
            "paths": {"required": True, "description": "Non-empty list of local LoRA directories."},
            "description": "None disables LoRA. Speculative LoRA requires a registered combination and a shared quant_checkpoint.",
        }
    sources = {variant.name: list(variant.required_sources) for variant in recipe.variants if variant.required_sources}
    if sources:
        parameters["sources"] = {
            "default": None,
            "required_by_variant": sources,
            "description": "Source name to local directory; only the selected variant's names are accepted.",
        }
    if recipe.required_resources:
        parameters["resources"] = {
            "required": True,
            "keys": list(recipe.required_resources),
            "description": "Resource name to local path; all listed names are required. Bundled resources are fixed.",
        }

    # components, vision and attention only accept the registered, fixed values.
    # max_pe_length is fixed by the recipe, not a build_model argument.
    return parameters
