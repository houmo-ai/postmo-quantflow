# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
from xhmodel_merak.model_catalog.contracts import (
    AdapterSpec,
    DeploymentPrecisionBinding,
    ModelRecipe,
    ResolvedPrecision,
)
from xhmodel_merak.model_integrations._dependencies import IMAGE, runtime_dependencies
from xhmodel_merak.model_integrations._shared import AUTOREGRESSIVE_BUILD, variant

PROVIDER_ID = "ocr-rm"
__all__ = ["PROVIDER_ID", "register"]


def register(catalog):
    catalog.register_adapter(
        AdapterSpec(
            name="ocr-rm.inference",
            kind="inference",
            target="xhmodel_merak.model_integrations.ocr_rm.inference:adapter",
            artifact_formats=("autoregressive",),
        ),
        provider=PROVIDER_ID,
    )
    catalog.register_recipe(
        ModelRecipe(
            model_id="ocr-rm-2b-value",
            display_name="OCR RM Qwen3.5 2B Value Head",
            family="ocr-rm",
            config_resolver="xhmodel_merak.model_integrations.ocr_rm.planning:resolve_config",
            export_name="ocr_rm_2b_value",
            task="document-consistency-scoring",
            modalities=("text", "image"),
            variants=(
                variant("full", "configs_merak/workflows/xh2a/llm_models/ocr_rm/ocr_rm_2b_value_xh2a_w8a8.yaml"),
            ),
            default_precision=ResolvedPrecision(weight="w16", activation="a16"),
            supported_precisions=(("w16", "a16"),),
            quant_methods=(),
            deployment_precision={name: "w8a8h1_sefp" for name in ("visual", "prefill", "decode")},
            deployment_precision_bindings=(
                DeploymentPrecisionBinding(
                    ("prefill", "decode"), "export.model.quant_scheme", ("w16a16h1_sefp", "w8a8h1_sefp")
                ),
                DeploymentPrecisionBinding(
                    ("visual",), "export.model.visual_config.quant_scheme", ("w16a16h1_sefp", "w8a8h1_sefp")
                ),
            ),
            components=("visual", "prefill", "decode"),
            golden_components=(),
            build_adapter=AUTOREGRESSIVE_BUILD,
            inference_adapter="ocr-rm.inference",
            artifact_format="autoregressive",
            input_schema={
                "type": "object",
                "properties": {
                    "orig_image": {"type": "string", "minLength": 1},
                    "render_image": {"type": "string", "minLength": 1},
                    "ocr_text": {"type": "string"},
                },
                "required": ["orig_image", "render_image", "ocr_text"],
                "additionalProperties": False,
                "maxMediaInputs": 2,
            },
            output_schema={"type": "object", "properties": {"stats": {"type": "object"}}, "required": ["stats"]},
            default_context_length=8192,
            default_prefill_chunk_length=256,
            max_pe_length=262144,
            default_profile="visualm96-384-1536",
            profiles=("visualm96-384-1536",),
            vision_profile="visualm96-384-1536",
            vision_gears=(96, 384, 1536),
            attention_profiles=("standard",),
            source_constraints=(
                ("model_type", "qwen3_5"),
                ("problem_type", "single_label_classification"),
                ("text_config.hidden_size", 2048),
                ("text_config.num_hidden_layers", 24),
            ),
            dependencies=runtime_dependencies(common=IMAGE),
        ),
        provider=PROVIDER_ID,
    )
