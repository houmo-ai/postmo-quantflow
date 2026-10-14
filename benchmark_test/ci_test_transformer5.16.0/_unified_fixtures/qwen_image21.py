# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from _unified_preparation import prepare_context


def prepare(case, root, monkeypatch):
    catalog, recipe, source, calibration, image, kwargs, inputs, replacements, family = prepare_context(case, root)
    from _qwen_image21_small_fixture import prepare_qwen_image21

    inputs = prepare_qwen_image21(source, root, catalog, recipe, monkeypatch)
    kwargs["deployment_precision"] = {"image": "w8a8_sefp"}
    return kwargs, inputs


def validate_output(case, output, result, moved, inputs, root, record):
    from xhmodel_merak.model_inference import inference_model

    import numpy as np

    assert (output.height, output.width) == (64, 32)
    assert output.image.shape == (64, 32, 4) and output.image.dtype == np.uint8
    assert result.deployment_precision == {"image": "w8a8_sefp"}
    edited = inference_model(
        artifact_dir=str(moved),
        device="cuda:0",
        inputs={
            **inputs,
            "prompt": "Make it blue.",
            "images": [str(root / "reference.png")],
        },
    )
    assert edited.image.shape == output.image.shape and edited.stats["reference_count"] == 1
    repeated = inference_model(artifact_dir=str(moved), inputs=inputs, device="cuda:0")
    assert np.array_equal(output.image, repeated.image)
    record["repeat_bit_exact"] = True
