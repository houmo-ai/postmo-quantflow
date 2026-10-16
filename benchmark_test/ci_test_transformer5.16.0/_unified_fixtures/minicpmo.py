# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import Path

from _unified_preparation import prepare_context


def prepare(case, root, monkeypatch):
    from _unified_special_fixtures import prepare_minicpmo

    catalog, recipe, source, calibration, image, kwargs, inputs, replacements, family = prepare_context(case, root)
    return prepare_minicpmo(catalog, recipe, root, calibration, monkeypatch, case.variant)


def check_inference(moved, inputs, record):
    import torch

    from xhmodel_merak.model_catalog.registry import get_recipe
    from xhmodel_merak.model_inference import inference_model

    torch.manual_seed(4891)
    golden = list(get_recipe("minicpm-o-4.5").golden_components)
    output = inference_model(artifact_dir=str(moved), inputs=inputs, device="cuda:0", golden=golden)
    assert output.model_id == "minicpm-o-4.5"
    value = output.to_dict()
    json.dumps(value, allow_nan=False)
    assert output.num_samples > 0
    meta = json.loads(Path(output.meta_file).read_text())
    expected = {f"{component}/{role}/step_0" for component in golden
                for role in meta["components"][component]["graphs"]}
    assert {path for paths in output.golden_artifacts.values() for path in paths} == expected
    assert all(any((moved / path).rglob("*.npy")) for path in expected)
    record.update(passed=True, stage="complete", inference=value)
