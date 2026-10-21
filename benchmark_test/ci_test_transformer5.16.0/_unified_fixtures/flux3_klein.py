# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Tiny native FLUX checkpoint through the shared public integration runner."""

from _unified_preparation import prepare_context


def prepare(case, root, monkeypatch):
    from _flux3_klein_fixture import prepare_source

    _, _, source, _, _, kwargs, _, _, _ = prepare_context(case, root)
    prepare_source(source, monkeypatch)
    kwargs["deployment_precision"] = {"image": "w8a8_sefp"}
    return kwargs, dict(prompt="A red ceramic teapot rests on a table by a window.", steps=2, seed=9)


def validate_output(case, output, result, moved, inputs, root, record):
    import numpy as np
    from PIL import Image

    from xhmodel_merak.model_inference import inference_model

    assert output.image.shape == (32, 64, 3) and output.image.dtype == np.uint8
    assert result.deployment_precision == {"image": "w8a8_sefp"}
    reference = root / "reference.png"
    Image.new("RGB", (59, 43), "blue").save(reference)
    edited = inference_model(artifact_dir=str(moved), device="cuda:0", inputs={**inputs, "images": [str(reference)]})
    assert edited.image.shape == (43, 59, 3)
    repeated = inference_model(artifact_dir=str(moved), device="cuda:0", inputs=inputs)
    np.testing.assert_array_equal(output.image, repeated.image)
    record["repeat_bit_exact"] = True
