#!/usr/bin/env python3
"""Verify Phase A SSFP4 (+ Hadamard MatMul) against original Bonsai semantics.

This checks a single signed projection independently of a full model export.

What it checks (default: layer0 mlp.gate_proj):
1) Phase A reconstructed weight == Bonsai affine unpack
2) Exported quant_weight.pt matches conversion
3) Hadamard MatMul bank == FWHT reference
4) Linear forward WITH activation Hadamard ≈ Bonsai original
5) Linear forward WITHOUT activation Hadamard ≠ Bonsai original
6) The traced wrap path (HadamardActivation + Linear) matches Bonsai
   when the aux bank is attached, and is a no-op when it is not

Example:
  /data01/home/xuchen/miniconda3/envs/xhquant/bin/python \
    examples_merak/llm/bonsai/verify_phase_a_vs_bonsai.py \
    --phase-a-dir /tmp/bonsai_pq2_phase_a_smoke2
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bonsai-dir",
        type=str,
        default="/data01/datasets/Ternary-Bonsai-2-27B-mlx-2bit",
    )
    parser.add_argument(
        "--phase-a-dir",
        type=str,
        default="/tmp/bonsai_pq2_phase_a_smoke2",
        help="Directory with quant_weight.pt / phase_a_aux.pt from Phase A convert",
    )
    parser.add_argument(
        "--module",
        type=str,
        default="language_model.model.layers.0.mlp.gate_proj",
        help="Bonsai packed module base name",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--atol-weight", type=float, default=0.0)
    parser.add_argument("--atol-hadamard", type=float, default=1e-5)
    parser.add_argument(
        "--atol-forward",
        type=float,
        default=1e-5,
        help="Base abs tol. Scaled by max(1, K/1024) because the MatMul is fp16.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Optional path to write metrics JSON",
    )
    args = parser.parse_args()

    zoo_root = Path(__file__).resolve().parents[3]
    bonsai = _load_module(
        "verify_bonsai_ssfp4",
        zoo_root / "xhmodel_merak/xh_llm/models/bonsai/bonsai_ssfp4.py",
    )
    hadamard = _load_module(
        "verify_hadamard_matmul",
        zoo_root / "xhmodel_merak/xh_llm/models/bonsai/hadamard_matmul.py",
    )

    from safetensors import safe_open

    bonsai_path = Path(args.bonsai_dir) / "model.safetensors"
    if not bonsai_path.exists():
        raise FileNotFoundError(bonsai_path)

    with safe_open(str(bonsai_path), framework="pt") as f:
        weight_u32 = f.get_tensor(args.module + ".weight")
        scales = f.get_tensor(args.module + ".scales")
        biases = f.get_tensor(args.module + ".biases")
        signs = f.get_tensor(args.module + ".signs")

    converted = bonsai.convert_bonsai_linear_to_ssfp4(weight_u32, scales, biases)
    W = converted["weight"].float()
    Qw = converted["quant_weight"]
    codes = converted["codes"]
    W_affine = bonsai.reconstruct_from_affine(codes, scales, biases)

    phase_a_dir = Path(args.phase_a_dir)
    metadata = json.loads((phase_a_dir / "phase_a_metadata.json").read_text())
    block_size = int(metadata["hadamard"]["block_size"])
    qw_pt = torch.load(phase_a_dir / metadata.get("checkpoint_file", "quant_weight.pt"),
                       map_location="cpu", weights_only=True)
    aux_pt = torch.load(phase_a_dir / metadata.get("aux_file", "phase_a_aux.pt"),
                        map_location="cpu", weights_only=True)

    hf_name = bonsai.bonsai_name_to_hf_name(args.module)
    W_export = qw_pt[f"{hf_name}.weight"].float()
    Qw_export = qw_pt[f"{hf_name}.quant_weight"]
    k_dim = int(W_export.shape[1])
    bank_key = f"hadamard_matmul.forward.w{k_dim}"
    if bank_key not in aux_pt:
        raise KeyError(f"{bank_key} missing from phase_a_aux.pt; have widths for this Linear's K={k_dim}")
    W_h = aux_pt[bank_key].float()

    torch.manual_seed(args.seed)
    x = torch.randn(args.batch, W.shape[1], dtype=torch.float32)
    x_fwht = hadamard.fwht_reference(x, signs.float(), block_size=block_size, inverse=False)
    x_mm = x @ W_h.T

    y_orig = torch.nn.functional.linear(x_fwht, W)
    y_phase_a = torch.nn.functional.linear(x_mm, W_export)
    y_naive = torch.nn.functional.linear(x, W_export)  # HMONNX if Hadamard is not wired

    # Exercise the same module the export graph calls.
    sys.path.insert(0, str(zoo_root))
    from xhmodel_merak.xh_llm.models.bonsai.hadamard_activation import (  # noqa: E402
        HadamardActivation,
        attach_hadamard_activations,
    )

    class _Proj(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(W_export.shape[1], W_export.shape[0], bias=False)
            with torch.no_grad():
                self.proj.weight.copy_(W_export)

        def forward(self, hidden):
            bank = getattr(self.proj, "hadamard_activation", None)
            if bank is not None:
                hidden = bank(hidden)
            return self.proj(hidden)

    proj = _Proj()
    y_unwired = proj(x)
    attach_hadamard_activations(
        proj,
        {"proj": signs.float()},
        {int(W_export.shape[1]): W_h},
        block_size=block_size,
    )
    y_wired = proj(x)
    # HadamardActivation.forward itself must be the MatMul.
    y_bank = HadamardActivation(W_h)(x)
    del proj

    metrics: Dict[str, Any] = {
        "module": args.module,
        "hf_name": hf_name,
        "weight_shape": list(W.shape),
        "codes_unique": sorted(set(codes.unique().tolist())),
        "mant_unique": sorted(set(Qw.unique().tolist())),
        "phaseA_vs_bonsai_affine_max_abs": _max_abs(W, W_affine),
        "export_weight_vs_converted_max_abs": _max_abs(W_export, W),
        "export_quant_weight_vs_converted_max_abs": int(
            (Qw_export.to(torch.int16) - Qw.to(torch.int16)).abs().max().item()
        ),
        "hadamard_bank_vs_fwht_max_abs": _max_abs(x_mm, x_fwht),
        "forward_with_hadamard_vs_bonsai_max_abs": _max_abs(y_phase_a, y_orig),
        "forward_with_hadamard_cosine": _cosine(y_phase_a, y_orig),
        "forward_without_hadamard_vs_bonsai_max_abs": _max_abs(y_naive, y_orig),
        "forward_without_hadamard_cosine": _cosine(y_naive, y_orig),
        "wrap_hadamard_activation_vs_fwht_max_abs": _max_abs(y_bank, x_fwht),
        "wrap_wired_forward_vs_bonsai_max_abs": _max_abs(y_wired, y_orig),
        "wrap_unwired_forward_vs_naive_max_abs": _max_abs(y_unwired, y_naive),
        "hmonnx_activation_hadamard_wired": True,
        "note": (
            "Wrap path applies HadamardActivation (ordinary MatMul) before each "
            "signed Linear when phase_a_aux.pt is loaded at export. Re-export is "
            "required before an existing HMONNX graph picks this up."
        ),
    }

    # fp16 MatMul error grows with K. 5120 stays at the base tol; 17408 needs ~17x.
    forward_tol = float(args.atol_forward) * max(1.0, k_dim / 1024.0)
    metrics["forward_atol"] = forward_tol
    checks = {
        "weight_recon": metrics["phaseA_vs_bonsai_affine_max_abs"] <= args.atol_weight,
        "export_weight": metrics["export_weight_vs_converted_max_abs"] <= args.atol_weight,
        "export_quant": metrics["export_quant_weight_vs_converted_max_abs"] == 0,
        "hadamard": metrics["hadamard_bank_vs_fwht_max_abs"] <= args.atol_hadamard,
        "forward_with_hadamard": metrics["forward_with_hadamard_vs_bonsai_max_abs"]
        <= forward_tol,
        "wrap_activation": metrics["wrap_hadamard_activation_vs_fwht_max_abs"]
        <= args.atol_hadamard,
        "wrap_wired_forward": metrics["wrap_wired_forward_vs_bonsai_max_abs"]
        <= forward_tol,
        "wrap_unwired_is_naive": metrics["wrap_unwired_forward_vs_naive_max_abs"]
        <= args.atol_forward,
        # Explicit negative check: naive path must NOT match original.
        "naive_path_diverges": metrics["forward_without_hadamard_cosine"] < 0.1,
    }
    metrics["checks"] = checks
    metrics["ok"] = all(checks.values())

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")

    if not metrics["ok"]:
        failed = [k for k, v in checks.items() if not v]
        print(f"[FAIL] checks failed: {failed}", file=sys.stderr)
        return 1

    print(
        "[PASS] Phase A weight and wrap HadamardActivation match Bonsai; "
        "the unwired Linear path still diverges."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
