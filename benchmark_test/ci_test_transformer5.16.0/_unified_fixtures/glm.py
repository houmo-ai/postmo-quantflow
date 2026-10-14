# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
import json


def prepare_source(source, root, calibration, inputs, kwargs, replacements):
    from _glm53_packed_fixture import write_native_glm53

    write_native_glm53(source)
    kwargs["quant_output_dir"] = str(root / "quant")
    config = json.loads((source / "config.json").read_text())
    calibration.write_text(
        "".join(
            json.dumps({"text": " ".join(f"t{2 + (i + row) % 40}" for i in range(128))}) + "\n" for row in range(2)
        )
    )
    inputs["prompt"] = "t2 t3 t4"
    replacements.update(
        {
            "export.model.linear_chunk_size": 8,
            "export.model.placeholder_export_workers": 1,
            "export.model.vision_max_patches": 32,
        }
    )
    return config


def validate_build(result, record):
    from pathlib import Path

    assert result.quant_artifact is not None
    reports = list(Path(result.quant_artifact).rglob("glm53_quantization_recipe.json"))
    assert len(reports) == 1, "GLM checkpoint must preserve real quantization evidence"
    evidence = json.loads(reports[0].read_text())
    provenance = evidence["provenance"]
    assert evidence["method"] == "autoround" and provenance["dry_run"] is False
    assert provenance["iters"] == 4 and provenance["tuning"]["loss_evaluations"] > 0
    assert evidence["calibration"]["actual_nsamples"] == 2
    assert evidence["calibration"]["actual_tokens"] == 128
    assert all(provenance["audit"][key] > 0 for key in ("w4_modules", "w8_modules"))
    record["quantization_evidence"] = evidence
