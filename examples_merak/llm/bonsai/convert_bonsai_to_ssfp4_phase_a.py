#!/usr/bin/env python3
"""Convert Ternary-Bonsai MLX 2bit pack → Phase A SSFP4-compatible checkpoint.

Example:
  PYTHONPATH=/path/to/xh2modelzoo \\
  python examples_merak/llm/bonsai/convert_bonsai_to_ssfp4_phase_a.py \\
    --bonsai-dir /data01/datasets/Ternary-Bonsai-2-27B-mlx-2bit \\
    --output-dir work_dirs/bonsai_pq2_phase_a \\
    --include-name language_model.model.layers.0.mlp.gate_proj

Notes:
  - Output is SSFP4 compatibility only (`pq2_weight=false`).
  - Online Hadamard is emitted as MatMul weights, not a special FWHT op.
"""

from __future__ import annotations

import argparse
import json
import sys
import shutil
from pathlib import Path


def _load_local_module(module_name: str, relative_path: str):
    """Load a Phase A helper module by file path.

    Keep offline checkpoint conversion independent of backend initialization.
    """
    import importlib.util

    root = Path(__file__).resolve().parents[3]
    path = root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bonsai-dir",
        type=str,
        required=True,
        help="Path to Bonsai MLX pack directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to write Phase A quant_weight.pt + metadata",
    )
    parser.add_argument(
        "--include-name",
        action="append",
        default=None,
        help=(
            "Optional Bonsai module base name to include "
            "(repeatable). Example: language_model.model.layers.0.mlp.gate_proj"
        ),
    )
    parser.add_argument(
        "--skip-float-tensors",
        action="store_true",
        help="Do not copy non-quant float tensors (norms/A_log/conv/...)",
    )
    parser.add_argument(
        "--build-hadamard-matmul",
        action="store_true",
        default=True,
        help="Also emit Hadamard MatMul weights for unique sign widths (default: true)",
    )
    parser.add_argument(
        "--no-build-hadamard-matmul",
        action="store_false",
        dest="build_hadamard_matmul",
        help="Skip Hadamard MatMul weight bank generation",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="quant_weight.pt",
        help="Checkpoint filename inside output-dir",
    )
    parser.add_argument(
        "--hf-model-dir", default=None,
        help="HF config/tokenizer directory for the converted artifact (no base weights are copied)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bonsai_ssfp4 = _load_local_module(
        "bonsai_phase_a_bonsai_ssfp4",
        "xhmodel_merak/xh_llm/models/bonsai/bonsai_ssfp4.py",
    )
    hadamard_matmul = _load_local_module(
        "bonsai_phase_a_hadamard_matmul",
        "xhmodel_merak/xh_llm/models/bonsai/hadamard_matmul.py",
    )

    state, meta = bonsai_ssfp4.convert_bonsai_pack_to_ssfp4_state_dict(
        args.bonsai_dir,
        include_names=args.include_name,
        include_float_tensors=not args.skip_float_tensors,
    )

    if args.build_hadamard_matmul:
        signs_by_width = hadamard_matmul.collect_signs_by_width_from_state(state)
        block_size = int(meta["hadamard"]["block_size"])
        bank = hadamard_matmul.build_hadamard_matmul_weight_bank(
            signs_by_width,
            block_size=block_size,
            # Embedding inverse rotation was already materialized offline.
            include_inverse=False,
        )
        state.update(bank)
        meta["hadamard"]["matmul_weights"] = sorted(bank.keys())
        meta["hadamard"]["sign_widths"] = sorted(int(w) for w in signs_by_width)
        meta["hadamard"]["implementation"] = "matmul_weight"

    meta["checkpoint_file"] = args.filename
    out = bonsai_ssfp4.save_phase_a_checkpoint(
        state,
        meta,
        args.output_dir,
        filename=args.filename,
        split_aux=True,
    )
    config_source = Path(args.hf_model_dir or args.bonsai_dir)
    config_path = config_source / "config.json"
    if args.hf_model_dir and not config_path.is_file():
        raise FileNotFoundError(config_path)
    if config_path.is_file():
        config = json.loads(config_path.read_text())
        config.pop("quantization_config", None)
        config.pop("quantization", None)
        (Path(args.output_dir) / "config.json").write_text(json.dumps(config, indent=2))
        for filename in (
            "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt",
            "chat_template.jinja", "tokenizer.model",
        ):
            path = config_source / filename
            if path.is_file():
                shutil.copy2(path, Path(args.output_dir) / filename)
    summary = {
        "output": str(out),
        "aux_file": str(Path(args.output_dir) / "phase_a_aux.pt"),
        "num_export_tensors": len(bonsai_ssfp4.split_phase_a_export_tensors(state)[0]),
        "num_aux_tensors": len(bonsai_ssfp4.split_phase_a_export_tensors(state)[1]),
        "num_modules": meta.get("num_modules"),
        "format": meta.get("format"),
        "pq2_weight": meta.get("pq2_weight"),
        "hadamard_matmul_weights": meta.get("hadamard", {}).get("matmul_weights", []),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
