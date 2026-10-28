"""Expanded BF16 comparisons preserve old probes and allocate work atomically."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest


def test_expansion_preserves_old_pairs_and_balances_tasks():
    from examples_merak.vla.wall_oss.bf16_open_loop import expanded_plan

    episodes = [dict(task_index=t, episode_index=10 * t + e, length=101) for t in range(40) for e in range(10)]
    old = [
        dict(id=3 * t + i, task_index=t, episode_index=10 * t + 4, frame_index=f, seed=10222 + 3 * t + i)
        for t in range(40)
        for i, f in enumerate([25, 50, 75])
    ]
    plan = expanded_plan(episodes, old)
    assert len(plan) == 1920
    assert len({(r["episode_index"], r["frame_index"], r["seed"]) for r in plan}) == 1920
    assert plan[:120] == old
    for t in range(40):
        rows = [r for r in plan if r["task_index"] == t]
        assert len(rows) == 48
        assert len({r["episode_index"] for r in rows}) == 8
        assert len({(r["episode_index"], r["frame_index"]) for r in rows}) == 24


def test_sqlite_claims_are_unique_and_retry_is_recoverable(tmp_path):
    from examples_merak.vla.wall_oss.evaluation_queue import EvaluationQueue

    queue = EvaluationQueue(tmp_path / "queue.sqlite3")
    queue.initialize(12)

    def consume(owner):
        result = []
        while (sample := queue.claim("bf16", owner)) is not None:
            result.append(sample)
            queue.finish("bf16", sample, owner)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(consume, ["a", "b", "c", "d"]))
    assert sorted(i for values in results for i in values) == list(range(12))
    sample = queue.claim("w8", "old")
    with pytest.raises(ValueError, match="claim no longer belongs"):
        queue.finish("w8", sample, "unrelated")
    queue.release_owner("old")
    assert queue.claim("w8", "new") == sample


def test_prediction_reuse_checks_request_contract_and_exact_noise(tmp_path):
    from examples_merak.vla.wall_oss.bf16_open_loop import array_sha, read_record, write_record

    trajectory = np.zeros((11, 1, 32, 26), np.float32)
    actions = np.zeros((1, 32, 26), np.float32)
    pair = dict(request_hash="same-request", noise_sha256=array_sha(trajectory[0]))
    path = tmp_path / "case.npz"
    write_record(path, trajectory, actions, dict(request_hash="same-request", contract="w8-package"))
    result, _, _ = read_record(path, pair, "w8-package")
    np.testing.assert_array_equal(result, trajectory)
    with pytest.raises(ValueError, match="identity mismatch"):
        read_record(path, pair, "different-package")
    with pytest.raises(ValueError, match="different noise"):
        read_record(path, {**pair, "noise_sha256": "different-noise"}, "w8-package")
