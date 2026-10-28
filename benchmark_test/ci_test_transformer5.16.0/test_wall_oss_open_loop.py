"""LIBERO numeric probes must preserve state semantics and exclude masked DOFs."""

import numpy as np
import pytest


def test_libero_state_probe_uses_right_arm_and_row_major_rotation():
    from examples_merak.vla.wall_oss.open_loop import map_state

    state, mask = map_state([0.1, 0.2, 0.3, np.pi, 0, 0, 0.04, -0.04])
    np.testing.assert_allclose(state[10:13], [0.1, 0.2, 0.3])
    np.testing.assert_allclose(state[13:19], [1, 0, 0, 0, -1, 0], atol=1e-6)
    assert state[19] == pytest.approx(0.08)
    assert np.flatnonzero(mask).tolist() == list(range(10, 20))
    assert np.count_nonzero(state[:10]) == np.count_nonzero(state[20:]) == 0
    rotated, _ = map_state([0, 0, 0, 0, 0, np.pi / 2, 0, 0])
    np.testing.assert_allclose(rotated[13:19], [0, -1, 0, 1, 0, 0], atol=1e-6)
    with pytest.raises(ValueError):
        map_state([0] * 7)


def test_masked_dimensions_cannot_inflate_numeric_agreement():
    from examples_merak.vla.wall_oss.open_loop import numeric_metrics

    expected = np.ones((3, 32, 26))
    actual = expected.copy()
    actual[..., :10] = 999
    mask = np.zeros(26, bool)
    mask[10:20] = True
    result = numeric_metrics(actual, expected, mask)
    assert result["mae"] == result["max_abs"] == result["rmse"] == 0
    assert result["cosine"] == pytest.approx(1)
    actual[..., 10:20] *= 2
    result = numeric_metrics(actual, expected, mask)
    assert result["mae"] == result["nrmse"] == 1


def test_sampling_covers_every_task_without_cross_episode_frames():
    from examples_merak.vla.wall_oss.open_loop import select_frames

    rows = [dict(episode_index=e, task_index=t, length=101) for t in range(40) for e in (2 * t, 2 * t + 1)]
    selection = select_frames(rows, set(range(1, 80, 2)))
    assert len(selection) == 120
    assert {s["task_index"] for s in selection} == set(range(40))
    for t in range(40):
        values = [s for s in selection if s["task_index"] == t]
        assert {s["episode_index"] for s in values} == {2 * t + 1}
        assert [s["frame_index"] for s in values] == [25, 50, 75]


def test_visual_relative_gate_rejects_semantic_feature_changes():
    from examples_merak.vla.wall_oss.open_loop import FLOAT_GATES, validate_visual

    expected = np.random.default_rng(10).normal(size=(64, 16))
    actual = expected.copy()
    actual[0, 0] += 0.003
    assert validate_visual(actual, expected, FLOAT_GATES)["nrmse"] < 0.0001
    with pytest.raises(ValueError, match="visual relative FP32 gate failed"):
        validate_visual(expected[::-1], expected, FLOAT_GATES)
