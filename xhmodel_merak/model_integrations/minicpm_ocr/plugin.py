# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
from xhmodel_merak.model_catalog.contracts import (
    AdapterSpec,
    ArtifactFormatSpec,
    DeploymentPrecisionBinding,
    ModelRecipe,
    ResolvedPrecision,
)
from xhmodel_merak.model_integrations._dependencies import IMAGE, runtime_dependencies
from xhmodel_merak.model_integrations._shared import text_output_schema, variant

PROVIDER_ID = "minicpm-ocr"
__all__ = ["PROVIDER_ID", "register"]


def register(catalog):
    catalog.register_artifact_format(ArtifactFormatSpec(name="minicpm_ocr"), provider=PROVIDER_ID)
    catalog.register_adapter(
        AdapterSpec(
            name="minicpm-ocr.build",
            kind="build",
            target="xhmodel_merak.model_integrations.minicpm_ocr.build",
            artifact_formats=("minicpm_ocr",),
        ),
        provider=PROVIDER_ID,
    )
    catalog.register_adapter(
        AdapterSpec(
            name="minicpm-ocr.inference",
            kind="inference",
            target="xhmodel_merak.model_integrations.minicpm_ocr.inference:adapter",
            artifact_formats=("minicpm_ocr",),
        ),
        provider=PROVIDER_ID,
    )
    catalog.register_recipe(
        ModelRecipe(
            model_id="minicpm-ocr-1.0",
            display_name="MiniCPM-OCR 1.0",
            family="minicpm-ocr",
            export_name="minicpm_ocr_1_0",
            task="document-understanding",
            modalities=("text", "image"),
            variants=(
                variant("full", "configs_merak/workflows/xh2a/llm_models/minicpm_ocr/minicpm_ocr_1_0_xh2a_w8a8.yaml"),
            ),
            default_precision=ResolvedPrecision(weight="w16", activation="a16"),
            supported_precisions=(("w16", "a16"),),
            quant_methods=(),
            deployment_precision={name: "w8a8h1_sefp" for name in ("visual", "prefill", "decode")},
            deployment_precision_bindings=(
                DeploymentPrecisionBinding(
                    ("prefill", "decode"), "export.model.quant_scheme", ("w16a16h0_sefp", "w8a8h1_sefp")
                ),
                DeploymentPrecisionBinding(
                    ("visual",), "export.model.visual_config.quant_scheme", ("w16a16h0_sefp", "w8a8h1_sefp")
                ),
            ),
            components=("visual", "prefill", "decode"),
            golden_components=(),
            build_adapter="minicpm-ocr.build",
            inference_adapter="minicpm-ocr.inference",
            artifact_format="minicpm_ocr",
            input_schema={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "minLength": 1},
                    "prompt": {"type": "string", "minLength": 1, "default": "请识别图片中的文字。"},
                    "max_new_tokens": {"type": "integer", "minimum": 1, "maximum": 4096, "default": 512},
                },
                "required": ["image"],
                "additionalProperties": False,
                "maxMediaInputs": 1,
            },
            output_schema=text_output_schema(),
            default_context_length=8192,
            default_prefill_chunk_length=256,
            max_pe_length=262144,
            attention_profiles=("standard",),
            source_constraints=(
                ("model_type", "minicpm_ocr"),
                ("architectures.0", "MiniCPMOcrForConditionalGeneration"),
                ("text_config.hidden_size", 1024),
                ("text_config.num_hidden_layers", 24),
            ),
            dependencies=runtime_dependencies(common=IMAGE),
        ),
        provider=PROVIDER_ID,
    )
