# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from _unified_preparation import prepare_context


def prepare(case, root, monkeypatch):
    catalog, recipe, source, calibration, image, kwargs, inputs, replacements, family = prepare_context(case, root)
    # Packaging-only unified case: stub the heavy converter, keep public
    # build/relocate/validation, and assert unsupported unified inference.
    source.mkdir()
    from _unified_special_fixtures import configure_minimax_packaging_stub

    configure_minimax_packaging_stub(source, monkeypatch)
    inputs = {"prompt": "a clay fox walks across wet sand", "image": str(image), "workflow": "fl2va"}
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
    assert values.get('workflow') == 'fl2va'
    assert values.get('image') == inputs['image']
    record.update(passed=True, stage='complete', inference={'status': 'unsupported_combination', 'code': error.code.value, 'workflow': values.get('workflow'), 'prompt': values.get('prompt'), 'image': values.get('image')})
