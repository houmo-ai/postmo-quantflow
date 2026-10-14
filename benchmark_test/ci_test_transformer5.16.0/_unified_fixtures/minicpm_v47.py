# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from _unified_preparation import prepare_context


def prepare(case, root, monkeypatch):
    catalog, recipe, source, calibration, image, kwargs, inputs, replacements, family = prepare_context(case, root)
    # Packaging-only unified case: stub the heavy vision+LLM converter,
    # keep public build/relocate/validation, and assert unsupported inference.
    from _unified_special_fixtures import configure_minicpm_v47_packaging_stub

    configure_minicpm_v47_packaging_stub(source, monkeypatch, recipe=recipe)
    inputs = {
        "prompt": "describe the image briefly",
        "image": str(image),
        "max_new_tokens": 2,
        "downsample_mode": "4x",
    }
    return kwargs, inputs


def check_inference(moved, inputs, record):
    import pytest
    from xhmodel_merak.model_inference import inference_model

    from xhmodel_merak.errors import ErrorCode, ModelError
    with pytest.raises(ModelError) as exc_info:
        inference_model(artifact_dir=str(moved), inputs=inputs, device='cuda:0')
    error = exc_info.value
    values = error.details.get('values', {})
    assert error.code == ErrorCode.UNSUPPORTED_COMBINATION
    assert values.get('image') == inputs['image']
    assert values.get('downsample_mode') == inputs['downsample_mode']
    record.update(passed=True, stage='complete', inference={'status': 'unsupported_combination', 'code': error.code.value, 'prompt': values.get('prompt'), 'image': values.get('image'), 'downsample_mode': values.get('downsample_mode')})
