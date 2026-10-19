"""Numerical contracts for weight-derived priors and streamed ExFold fitting."""

import pytest
import torch

from xhmodel_merak.xh_llm.models.qwen3_5_moe.safe_calibration_stats import (
    SafePairwiseCalibrator,
    compute_aimer_keep,
)


def pair_oracle(ids, outputs, experts, ridge=1e-3, clip=4.0):
    coeff = torch.eye(experts)
    loss = torch.full((experts, experts), 1e30)
    norm_lists = [[] for _ in range(experts)]
    for token_ids, token_outputs in zip(ids, outputs, strict=True):
        for expert, output in zip(token_ids, token_outputs, strict=True):
            norm_lists[expert].append(output.norm())
    norms = torch.tensor([torch.stack(values).mean() if values else 0 for values in norm_lists])
    median = norms[torch.tensor([bool(values) for values in norm_lists])].median()
    for expert, values in enumerate(norm_lists):
        if not values:
            norms[expert] = median
    for source in range(experts):
        for target in range(experts):
            if source == target:
                continue
            dot, target_sq, source_sq = 0.0, 0.0, 0.0
            for token_ids, token_outputs in zip(ids, outputs, strict=True):
                if source not in token_ids or target not in token_ids:
                    continue
                a = token_outputs[(token_ids == source).nonzero().item()].double()
                b = token_outputs[(token_ids == target).nonzero().item()].double()
                weight = a.norm()
                dot += weight * a.dot(b)
                target_sq += weight * b.square().sum()
                source_sq += weight * a.square().sum()
            if source_sq > 0:
                c = torch.as_tensor(dot / (target_sq + ridge)).clamp(-clip, clip)
                coeff[source, target] = c
                loss[source, target] = (source_sq - 2 * c * dot + c.square() * target_sq).clamp_min(0) / source_sq
    return coeff, loss, norms


def test_weight_priors_match_independent_aimer_formula():
    gate_up = torch.tensor([[[1.0, -2.0], [3.0, 0.0]], [[0.1, 0.0], [0.0, 2.0]], [[2.0, -2.0], [-2.0, 2.0]]])
    down = torch.tensor([[[0.2], [0.1]], [[0.0], [3.0]], [[2.0], [-2.0]]])
    removal = []
    for i in range(3):
        values = torch.cat((gate_up[i].flatten(), down[i].flatten()))
        removal.append(values.abs().mean() / (torch.sqrt(values.square().mean() + 1e-8) + 1e-8))
    removal = torch.stack(removal)
    keep = (removal / (removal.mean() + 1e-8) + 1e-8).reciprocal()
    expected = keep / (keep.mean() + 1e-8)
    actual = compute_aimer_keep(gate_up, down, device="cpu")
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.float32 and actual.device.type == "cpu"
    assert not torch.allclose(actual, torch.ones_like(actual))


def test_directed_pair_fit_matches_independent_oracle_and_unseen_pairs():
    ids = torch.tensor([[0, 1], [1, 2], [0, 1], [1, 2]])
    outputs = torch.tensor(
        [[[2.0, 1.0], [1.0, 0.0]], [[1.0, 2.0], [-1.0, 3.0]], [[3.0, 0.0], [2.0, 1.0]], [[2.0, 2.0], [1.0, 4.0]]]
    )
    calibrator = SafePairwiseCalibrator(4)
    calibrator.update(ids, outputs)
    actual = calibrator.finalize()
    for value, expected in zip(actual, pair_oracle(ids, outputs, 4), strict=True):
        torch.testing.assert_close(value, expected, rtol=2e-6, atol=1e-6)
        assert value.dtype == torch.float32 and value.device.type == "cpu"
    coeff, loss, norms = actual
    assert coeff[0, 1] != coeff[1, 0]
    assert loss[0, 1] != loss[1, 0]
    assert coeff[0, 2] == 0 and loss[0, 2] == pytest.approx(1e30)
    assert norms[3] == norms[:3].median()
    torch.testing.assert_close(coeff.diagonal(), torch.ones(4))
    torch.testing.assert_close(loss.diagonal(), torch.full((4,), 1e30))


def test_streamed_and_single_batch_statistics_agree():
    generator = torch.Generator().manual_seed(4)
    outputs = torch.randn(17, 3, 5, generator=generator)
    ids = torch.stack([torch.randperm(5, generator=generator)[:3] for _ in outputs])
    full, chunks = SafePairwiseCalibrator(6), SafePairwiseCalibrator(6)
    full.update(ids, outputs)
    for begin in range(0, len(ids), 4):
        chunks.update(ids[begin : begin + 4], outputs[begin : begin + 4])
    for a, b in zip(full.finalize(), chunks.finalize(), strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_fit_clips_coefficients_and_preserves_zero_source_as_unobserved():
    calibrator = SafePairwiseCalibrator(3, clip=2)
    calibrator.update(torch.tensor([[0, 1, 2]]), torch.tensor([[[10.0, 0.0], [1.0, 0.0], [0.0, 0.0]]]))
    coeff, loss, norms = calibrator.finalize()
    assert coeff[0, 1] == 2 and loss[0, 1] == pytest.approx(0.64)
    assert coeff[2, 0] == 0 and loss[2, 0] == pytest.approx(1e30)
    assert norms[2] == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_experts": True},
        {"num_experts": 1},
        {"num_experts": 2, "ridge": 0},
        {"num_experts": 2, "clip": float("inf")},
        {"num_experts": 2, "eps": float("nan")},
    ],
)
def test_invalid_calibrator_settings(kwargs):
    with pytest.raises(ValueError):
        SafePairwiseCalibrator(**kwargs)


def test_empty_finalize_is_rejected():
    with pytest.raises(RuntimeError, match="batch"):
        SafePairwiseCalibrator(2).finalize()


@pytest.mark.parametrize(
    "ids,outputs",
    [
        (torch.empty(0, 2, dtype=torch.long), torch.empty(0, 2, 3)),
        (torch.tensor([[0, 0]]), torch.ones(1, 2, 3)),
        (torch.tensor([[-1, 1]]), torch.ones(1, 2, 3)),
        (torch.tensor([[0, 3]]), torch.ones(1, 2, 3)),
        (torch.tensor([[0.0, 1.0]]), torch.ones(1, 2, 3)),
        (torch.tensor([[0, 1]]), torch.ones(1, 2, 3, dtype=torch.long)),
        (torch.tensor([[0, 1]]), torch.full((1, 2, 3), float("nan"))),
        (torch.tensor([[0, 1]]), torch.ones(1, 1, 3)),
    ],
)
def test_invalid_calibration_batches(ids, outputs):
    with pytest.raises(ValueError):
        SafePairwiseCalibrator(3).update(ids, outputs)


@pytest.mark.parametrize(
    "gate_up,down",
    [
        (torch.ones(2, 3, 4), torch.ones(2, 4, 1)),
        (torch.ones(2, 4, 4), torch.ones(3, 4, 2)),
        (torch.full((2, 4, 4), float("inf")), torch.ones(2, 4, 2)),
        (torch.ones(2, 4, 4, dtype=torch.long), torch.ones(2, 4, 2)),
    ],
)
def test_invalid_aimer_weights(gate_up, down):
    with pytest.raises(ValueError):
        compute_aimer_keep(gate_up, down)
