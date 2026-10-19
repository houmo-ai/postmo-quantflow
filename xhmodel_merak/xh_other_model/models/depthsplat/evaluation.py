"""Numerical comparisons and GridSample isolation; not a task-accuracy gate."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def tensor_metrics(reference, actual):
    """Compare equal-shaped real arrays without hiding NaN/Inf or zero norms.

    Relative L2 is undefined (None) for a zero reference and nonzero actual.
    Nonfinite inputs invalidate all aggregate differences; counts remain useful.
    """
    reference, actual = np.asarray(reference), np.asarray(actual)
    if reference.shape != actual.shape:
        raise ValueError(f"Shape mismatch: {reference.shape} != {actual.shape}")
    if reference.size == 0:
        raise ValueError("Cannot evaluate empty tensors")
    if np.iscomplexobj(reference) or np.iscomplexobj(actual):
        raise ValueError("Expected real tensors")
    reference, actual = reference.astype(np.float64), actual.astype(np.float64)
    result = {
        "shape": list(reference.shape),
        "count": int(reference.size),
        "reference_nonfinite": int((~np.isfinite(reference)).sum()),
        "actual_nonfinite": int((~np.isfinite(actual)).sum()),
        "abs_max": None,
        "abs_mean": None,
        "relative_l2": None,
        "cosine": None,
    }
    if result["reference_nonfinite"] or result["actual_nonfinite"]:
        return {**result, "status": "nonfinite"}
    difference = actual - reference
    reference_norm = np.linalg.norm(reference.ravel())
    actual_norm = np.linalg.norm(actual.ravel())
    difference_norm = np.linalg.norm(difference.ravel())
    relative_l2 = float(difference_norm / reference_norm) if reference_norm else (0.0 if not actual_norm else None)
    cosine = (
        float(np.clip(np.dot((reference / reference_norm).ravel(), (actual / actual_norm).ravel()), -1, 1))
        if reference_norm and actual_norm
        else float(not reference_norm and not actual_norm)
    )
    return {
        **result,
        "status": "finite",
        "abs_max": float(np.abs(difference).max()),
        "abs_mean": float(np.abs(difference).mean()),
        "relative_l2": relative_l2,
        "cosine": cosine,
    }


def compare_outputs(reference: Mapping, actual: Mapping):
    """Compare named outputs, rejecting missing and extra output names."""
    if set(reference) != set(actual):
        raise ValueError(
            f"Output names mismatch: missing={sorted(set(reference) - set(actual))}, extra={sorted(set(actual) - set(reference))}"
        )
    if not reference:
        raise ValueError("No outputs to compare")
    return {name: tensor_metrics(value, actual[name]) for name, value in reference.items()}


def diagnose_grid_sample(feature, grid, *, mode="bilinear", padding_mode="zeros", align_corners=True, cuda_device=None):
    """Isolate FP16 operand rounding from CUDA FP16 sampling on real operands.

    Supports 2D NCHW sampling. CPU FP32 is the common reference; no outputs are
    persisted here. CUDA FP32 is also compared to separate backend differences.
    """
    import torch
    import torch.nn.functional as functional

    def cpu_float(value):
        return torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32)

    feature, grid = cpu_float(feature), cpu_float(grid)
    if feature.ndim != 4 or grid.ndim != 4 or grid.shape[-1] != 2 or feature.shape[0] != grid.shape[0]:
        raise ValueError("Expected NCHW feature and NHW2 grid with equal batch sizes")
    if min(feature.shape) <= 0 or min(grid.shape) <= 0:
        raise ValueError("Sampling operands must be nonempty")
    properties = {"mode": mode, "padding_mode": padding_mode, "align_corners": align_corners}
    feature_half, grid_half = feature.half().float(), grid.half().float()
    reference = functional.grid_sample(feature, grid, **properties)
    comparisons = {}
    rounded_reference = None
    for name, current_feature, current_grid in (
        ("feature_roundtrip", feature_half, grid),
        ("grid_roundtrip", feature, grid_half),
        ("both_roundtrip", feature_half, grid_half),
    ):
        output = functional.grid_sample(current_feature, current_grid, **properties)
        comparisons[name] = tensor_metrics(reference.numpy(), output.numpy())
        if name == "both_roundtrip":
            rounded_reference = output
    height, width = feature.shape[-2:]
    scale = torch.tensor([(width - 1) / 2, (height - 1) / 2] if align_corners else [width / 2, height / 2])
    pixel_reference = (grid + 1) * scale - (0 if align_corners else 0.5)
    pixel_rounded = (grid_half + 1) * scale - (0 if align_corners else 0.5)
    coordinates = {
        "normalized_rounding": tensor_metrics(grid.numpy(), grid_half.numpy()),
        "pixel_rounding": tensor_metrics(pixel_reference.numpy(), pixel_rounded.numpy()),
        "outside_normalized_fraction": float((grid.abs() > 1).any(dim=-1).float().mean()),
        "rounded_outside_normalized_fraction": float((grid_half.abs() > 1).any(dim=-1).float().mean()),
        "pixel_abs_max_xy": (pixel_rounded - pixel_reference).abs().reshape(-1, 2).amax(dim=0).tolist(),
    }
    cuda = {"status": "unavailable", "reason": "CUDA is not available"}
    if torch.cuda.is_available():
        device = torch.device(cuda_device or "cuda")
        if device.type != "cuda":
            raise ValueError("cuda_device must be a CUDA device")
        try:
            output_float = functional.grid_sample(feature.to(device), grid.to(device), **properties)
            output_half = functional.grid_sample(
                feature.to(device).half(), grid.to(device).half(), **properties
            ).float()
            cuda = {
                "status": "available",
                "device": str(device),
                "float32_vs_cpu": tensor_metrics(reference.numpy(), output_float.cpu().numpy()),
                "float16_vs_cpu": tensor_metrics(reference.numpy(), output_half.cpu().numpy()),
                "float16_vs_cuda_float32": tensor_metrics(output_float.cpu().numpy(), output_half.cpu().numpy()),
                "float16_vs_cpu_rounded_operands": tensor_metrics(rounded_reference.numpy(), output_half.cpu().numpy()),
            }
        except RuntimeError as error:
            cuda = {"status": "error", "reason": str(error)}
    return {
        "properties": properties,
        "feature_shape": list(feature.shape),
        "grid_shape": list(grid.shape),
        "feature_rounding": tensor_metrics(feature.numpy(), feature_half.numpy()),
        "coordinates": coordinates,
        "comparisons": comparisons,
        "cuda": cuda,
        "task_accuracy_evaluated": False,
    }
