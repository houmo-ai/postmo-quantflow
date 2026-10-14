# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

from _unified_preparation import prepare_context
from _unified_small_model import write_auxiliary, write_native_checkpoint
from _unified_preparation import configure_calibration


def prepare(case, root, monkeypatch):
    catalog, recipe, source, calibration, image, kwargs, inputs, replacements, family = prepare_context(case, root)
    if family in ("glm", "ling"):
        from importlib import import_module
        module = import_module(f"_unified_fixtures.{family}")
        config = module.prepare_source(source, root, calibration, inputs, kwargs, replacements)
    else:
        config = write_native_checkpoint(source, "qwen_moe" if family == "macaron" else family)
    gemma = family.startswith("gemma")
    context = 320 if gemma else 256 if family == "paddleocr" else 64
    prefill = 320 if gemma else 64 if family in ("paddleocr", "ling") else 8
    kwargs.update(context_length=context, prefill_chunk_length=prefill)
    if "calibration" in recipe.required_resources:
        kwargs["resources"] = {"calibration": str(calibration)}
    if family == "paddleocr":
        kwargs["resources"] = {"calibration": str(calibration)}
        inputs["image"] = str(image)
        inputs["prompt"] = "OCR:"
    if case.model_id.startswith("wemm"):
        inputs = {"text": "hello world"}
    if case.variant in ("mtp", "lora_mtp"):
        kwargs["speculative"] = {"mode": "mtp"}
        if gemma:
            from _unified_special_fixtures import write_gemma_assistant

            assistant = root / "assistant"
            replacements.update(write_gemma_assistant(assistant, config, ordered=family in ("gemma_e2b", "gemma_e4b")))
            kwargs["sources"] = {"mtp": str(assistant)}
        else:
            write_auxiliary(source, family + "_mtp")
            replacements.update(
                {
                    "export.model.mtp_config.hidden_size": 64,
                    "export.model.mtp_config.num_key_value_heads": 2,
                    "export.model.mtp_config.head_dim": 32,
                    "export.model.mtp_config.input_sequence_length": 8,
                    "export.model.mtp_config.max_pe_length": 128,
                }
            )
    if case.variant in {"dflash", "dflash2"}:
        draft = root / "draft"
        write_auxiliary(draft, case.variant, target_config=config)
        kwargs.update(speculative={"mode": case.variant}, sources={case.variant: str(draft)})
    if case.variant in ("lora", "lora_mtp") and family != "macaron":
        adapter = root / "lora"
        write_auxiliary(adapter, "qwen_lora")
        kwargs["lora"] = {"paths": [str(adapter)]}
    if family == "macaron" and case.variant == "lora":
        from _unified_special_fixtures import write_macaron_loras

        write_macaron_loras(source)
        kwargs["lora"] = {"paths": [str(source / "loras" / f"L{i}") for i in range(3)]}
    constraints = (("model_type", config["model_type"]),)
    if family in ("minicpm", "ling"):
        constraints = (("architectures.0", config["architectures"][0]),)
    monkeypatch.setitem(
        catalog._recipes,
        case.model_id,
        replace(
            recipe,
            source_constraints=constraints,
            max_pe_length=512 if gemma else max(128, context),
            config_replacements=tuple(replacements.items()),
        ),
    )
    configure_calibration(case, calibration, monkeypatch)
    return kwargs, inputs


def validate_build(case, result, record):
    if case.fixture == "glm":
        from .glm import validate_build as validate_glm
        validate_glm(result, record)
