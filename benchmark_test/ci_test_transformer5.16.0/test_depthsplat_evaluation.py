"""Small real numeric and sampling checks; not full-model quality evaluation."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

_PATH = Path(__file__).resolve().parents[2] / "xhmodel_merak/xh_other_model/models/depthsplat/evaluation.py"
_SPEC = importlib.util.spec_from_file_location("depthsplat_evaluation", _PATH)
evaluation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(evaluation)


def test_metrics_known_difference():
    metrics = evaluation.tensor_metrics([3, 4], [6, 8])
    assert metrics["abs_max"] == 4
    assert metrics["abs_mean"] == 3.5
    assert metrics["relative_l2"] == 1
    assert metrics["cosine"] == pytest.approx(1)


@pytest.mark.parametrize(
    "reference,actual,cosine,relative", [([0], [0], 1, 0), ([0], [1], 0, None), ([1], [0], 0, 1), ([1], [-1], -1, 2)]
)
def test_zero_and_opposite_vectors(reference, actual, cosine, relative):
    result = evaluation.tensor_metrics(reference, actual)
    assert result["cosine"] == cosine
    assert result["relative_l2"] == relative


def test_invalid_tensors_and_output_names():
    for reference, actual in [([1], [[1]]), ([], []), ([1j], [1j])]:
        with pytest.raises(ValueError):
            evaluation.tensor_metrics(reference, actual)
    with pytest.raises(ValueError, match="names mismatch"):
        evaluation.compare_outputs({"depth": [1]}, {"gaussian": [1]})
    with pytest.raises(ValueError):
        evaluation.compare_outputs({}, {})
    assert evaluation.compare_outputs({"depth": [1]}, {"depth": [1]})["depth"]["abs_max"] == 0


def test_nonfinite_not_silently_filtered():
    result = evaluation.tensor_metrics([1, np.nan, np.inf], [1, 2, -np.inf])
    assert result["status"] == "nonfinite"
    assert result["reference_nonfinite"] == 2
    assert result["actual_nonfinite"] == 1
    assert all(result[key] is None for key in ("abs_max", "abs_mean", "cosine", "relative_l2"))


@pytest.mark.parametrize("align_corners", [True, False])
def test_real_sampling_rounding_and_pixel_coordinates(align_corners):
    feature = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
    grid = torch.tensor([[[[0.3333, -0.1111], [1.2, 0.0]]]])
    report = evaluation.diagnose_grid_sample(feature, grid, align_corners=align_corners)
    assert report["comparisons"]["feature_roundtrip"]["abs_max"] == 0
    assert report["comparisons"]["grid_roundtrip"]["abs_max"] > 0
    assert report["coordinates"]["outside_normalized_fraction"] == 0.5
    scale = np.array([1.5, 1] if align_corners else [2, 1.5])
    expected = np.max(np.abs((grid.half().float() - grid).numpy()).reshape(-1, 2), axis=0) * scale
    np.testing.assert_allclose(report["coordinates"]["pixel_abs_max_xy"], expected, atol=2e-7)
    assert report["task_accuracy_evaluated"] is False
    assert report["cuda"]["status"] == ("available" if torch.cuda.is_available() else "unavailable")


def test_sampling_rejects_wrong_grid():
    with pytest.raises(ValueError, match="NHW2"):
        evaluation.diagnose_grid_sample(torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2, 3))
