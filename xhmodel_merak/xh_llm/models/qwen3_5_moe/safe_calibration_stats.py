"""Weight priors and streaming statistics for SAFE calibration.

The formulas follow the author's ACEP/ExFold research implementation in
``moe_prune_v2`` and ``moe_dynamic_topk/src/dyace/acep_exfold.py``. This module
implements those statistics locally without importing the research repository
or the upstream ExFold runtime.
"""

import math

import torch


@torch.no_grad()
def compute_aimer_keep(gate_up, down, eps=1e-8, *, device="cpu"):
    """Return inverse normalized AIMER removal scores from fused HF weights."""
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    if (
        gate_up.ndim != 3
        or down.ndim != 3
        or gate_up.shape[0] == 0
        or gate_up.shape[1] == 0
        or gate_up.shape[2] == 0
        or gate_up.shape[1] % 2
        or down.shape != (gate_up.shape[0], gate_up.shape[2], gate_up.shape[1] // 2)
    ):
        raise ValueError("expected gate_up [experts, 2*intermediate, hidden] and down [experts, hidden, intermediate]")
    if not gate_up.is_floating_point() or not down.is_floating_point():
        raise ValueError("AIMER weights must be floating point")
    scores = []
    for expert in range(gate_up.shape[0]):
        branches = (
            gate_up[expert].detach().to(device=device, dtype=torch.float32),
            down[expert].detach().to(device=device, dtype=torch.float32),
        )
        if not all(bool(torch.isfinite(weight).all()) for weight in branches):
            raise ValueError("AIMER weights must be finite")
        count = sum(weight.numel() for weight in branches)
        mean_abs = sum(weight.abs().sum() for weight in branches) / count
        rms = torch.sqrt(sum(weight.square().sum() for weight in branches) / count + eps)
        scores.append((mean_abs / (rms + eps)).cpu())
    removal = torch.stack(scores)
    normalized = removal / (removal.mean() + eps)
    keep = (normalized + eps).reciprocal()
    return (keep / (keep.mean() + eps)).float().cpu()


class SafePairwiseCalibrator:
    """Accumulate directed, source-norm-weighted scalar least squares.

    Only co-routed, unweighted expert outputs enter the statistics. A source
    expert's output is approximated by ``coeff[source, target] * target_output``.
    Storage is O(experts**2), independent of the calibration token count.
    """

    def __init__(self, num_experts, device="cpu", ridge=1e-3, clip=4.0, eps=1e-8):
        if isinstance(num_experts, bool) or not isinstance(num_experts, int) or num_experts <= 1:
            raise ValueError("num_experts must be an integer greater than one")
        if any(not math.isfinite(value) or value <= 0 for value in (ridge, clip, eps)):
            raise ValueError("ridge, clip and eps must be finite and positive")
        self.num_experts = num_experts
        self.device = torch.device(device)
        self.ridge = float(ridge)
        self.clip = float(clip)
        self.eps = float(eps)
        self._pair_sums = torch.zeros((3, num_experts * num_experts), device=self.device, dtype=torch.float64)
        self.device = self._pair_sums.device
        self._norm_sums = torch.zeros(num_experts, device=self.device, dtype=torch.float64)
        self._norm_counts = torch.zeros_like(self._norm_sums)
        self._updates = 0

    @torch.no_grad()
    def update(self, ids, outs):
        if (
            ids.ndim != 2
            or outs.ndim != 3
            or outs.shape[:2] != ids.shape
            or min(outs.shape) < 1
            or ids.dtype not in (torch.int32, torch.int64)
            or not outs.is_floating_point()
        ):
            raise ValueError("expected integer ids [tokens, top_k] and floating outputs [tokens, top_k, hidden]")
        if ids.device != self.device or outs.device != self.device:
            raise ValueError("calibration inputs and accumulator must share a device")
        if not bool(torch.isfinite(outs).all()) or bool(((ids < 0) | (ids >= self.num_experts)).any()):
            raise ValueError("calibration outputs must be finite and expert ids valid")
        ordered = ids.sort(dim=-1).values
        if bool((ordered[:, 1:] == ordered[:, :-1]).any()):
            raise ValueError("routed expert ids must be unique within each token")

        outputs = outs.detach().float()
        norms = torch.linalg.vector_norm(outputs, dim=-1)
        dots = torch.bmm(outputs, outputs.transpose(1, 2))
        sources, targets = ids.long().unsqueeze(2), ids.long().unsqueeze(1)
        distinct = sources != targets
        indices = (sources * self.num_experts + targets)[distinct]
        weight = norms.unsqueeze(2)
        values = (
            dots * weight,
            norms.square().unsqueeze(1) * weight,
            (norms.square().unsqueeze(2) * weight).expand_as(dots),
        )
        for destination, value in zip(self._pair_sums, values, strict=True):
            destination.index_add_(0, indices, value[distinct].double())
        flat_ids = ids.reshape(-1).long()
        self._norm_sums.index_add_(0, flat_ids, norms.reshape(-1).double())
        self._norm_counts.index_add_(0, flat_ids, torch.ones_like(norms.reshape(-1), dtype=torch.float64))
        self._updates += 1

    @torch.no_grad()
    def finalize(self):
        if not self._updates:
            raise RuntimeError("at least one calibration batch is required")
        if not bool(torch.isfinite(self._pair_sums).all()) or not bool(torch.isfinite(self._norm_sums).all()):
            raise RuntimeError("non-finite calibration statistics")
        dot, target_sq, source_sq = self._pair_sums.reshape(3, self.num_experts, self.num_experts)
        observed = source_sq > 0
        coefficient = torch.where(observed, dot / (target_sq + self.ridge), 0.0).clamp(-self.clip, self.clip)
        residual = (source_sq - 2 * coefficient * dot + coefficient.square() * target_sq).clamp_min(0)
        loss = torch.full_like(source_sq, 1e30)
        loss[observed] = residual[observed] / source_sq[observed]
        loss.clamp_max_(1e30)
        loss.fill_diagonal_(1e30)
        coefficient.masked_fill_(loss >= 1e30, 0)
        coefficient.fill_diagonal_(1)
        norms = self._norm_sums / self._norm_counts.clamp_min(1)
        seen = self._norm_counts > 0
        fallback = norms[seen].median()
        norms = torch.where(seen, norms, fallback)
        return tuple(value.float().cpu() for value in (coefficient, loss, norms))
