#!/usr/bin/env python3
"""AutoRound W4A8 for MiniMax-H3 Qwen3-VL text encoder.

Quantizes ``MiniMax-H3/text_encoder`` (Qwen3VL) with:
  QuantizationScheme(bits=4, group_size=128, act_bits=8, act_dynamic=True)

Vision / non-text modules are left unquantized (``quant_nontext_module=False``).
Calibration defaults to the local pile-10k JSONL under ``data/calib_data``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_MODEL = Path("/data01/home/chen.xu/xh2_old/MiniMax-H3/text_encoder")
DEFAULT_CALIB = Path("/data01/home/chen.xu/xh2_old/xh2_model_zoo/data/calib_data/NeelNanda-pile-10k.jsonl")
DEFAULT_OUTPUT = Path("/data01/home/chen.xu/xh2_old/xh2_model_zoo/work_dirs/minimax_h3_qwen3vl_autoround_w4a8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=512)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--act-bits", type=int, default=8)
    parser.add_argument("--device-map", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--format",
        default="fake",
        help=("AutoRound save format. W4A8 activation quantization currently forces fake format (default: fake)."),
    )
    parser.add_argument(
        "--low-gpu-mem-usage",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--quant-nontext-module",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also quantize vision / non-text modules (default: false).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing --output directory.",
    )
    parser.add_argument(
        "--fold-output",
        type=Path,
        default=None,
        help=(
            "Optional HF-folded output directory. When set, rename "
            "*.orig_layer.weight keys after fake save for MiniMax export."
        ),
    )
    parser.add_argument(
        "--stage-root",
        type=Path,
        default=None,
        help="Optional MiniMax-H3 staging root pointing text_encoder at --fold-output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.device_map).split(",")[0])

    model = args.model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    if not model.is_dir():
        raise FileNotFoundError(f"text_encoder model dir not found: {model}")
    if not dataset.is_file():
        raise FileNotFoundError(f"calibration jsonl not found: {dataset}")
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"output exists (pass --overwrite): {output}")
    output.mkdir(parents=True, exist_ok=True)

    # AutoRoundMLLM still works for Qwen3-VL; AutoRound may also accept the same kwargs.
    from auto_round import AutoRoundMLLM, QuantizationScheme

    scheme = QuantizationScheme(
        bits=int(args.bits),
        group_size=int(args.group_size),
        act_bits=int(args.act_bits),
        act_dynamic=True,
    )
    provenance = {
        "model": str(model),
        "output": str(output),
        "dataset": str(dataset),
        "scheme": {
            "bits": scheme.bits,
            "group_size": scheme.group_size,
            "act_bits": scheme.act_bits,
            "act_dynamic": scheme.act_dynamic,
            "sym": scheme.sym,
            "data_type": scheme.data_type,
        },
        "nsamples": args.nsamples,
        "seqlen": args.seqlen,
        "iters": args.iters,
        "batch_size": args.batch_size,
        "device_map": args.device_map,
        "quant_nontext_module": args.quant_nontext_module,
        "format": args.format,
        "seed": args.seed,
        "fold_output": str(args.fold_output) if args.fold_output else None,
        "stage_root": str(args.stage_root) if args.stage_root else None,
        "export_quant_type_hint": "w4a8h0_ssfp",
    }
    (output / "minimax_h3_qwen3vl_autoround_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2), flush=True)

    ar = AutoRoundMLLM(
        model=str(model),
        scheme=scheme,
        dataset=str(dataset),
        nsamples=int(args.nsamples),
        seqlen=int(args.seqlen),
        iters=int(args.iters),
        batch_size=int(args.batch_size),
        device_map=args.device_map,
        low_gpu_mem_usage=bool(args.low_gpu_mem_usage),
        quant_nontext_module=bool(args.quant_nontext_module),
        seed=int(args.seed),
    )
    quantized_model, save_folder = ar.quantize_and_save(
        output_dir=str(output),
        format=args.format,
        inplace=True,
    )
    print(
        f"AutoRound W4A8 done: save_folder={save_folder} model_type={type(quantized_model).__name__}",
        flush=True,
    )

    if args.fold_output is not None:
        import importlib.util
        import shutil
        import sys

        helper = Path(__file__).resolve().parent / "fold_autoround_fake_te.py"
        spec = importlib.util.spec_from_file_location("fold_autoround_fake_te", helper)
        if spec is None or spec.loader is None:
            raise ImportError(f"unable to load fold helper: {helper}")
        fold_mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = fold_mod
        spec.loader.exec_module(fold_mod)

        fold_output = args.fold_output.expanduser().resolve()
        if fold_output.exists() and args.overwrite:
            shutil.rmtree(fold_output)
        fake_dir = Path(save_folder[0] if isinstance(save_folder, (list, tuple)) else save_folder)
        fold_info = fold_mod.fold_checkpoint(
            input_dir=fake_dir,
            output_dir=fold_output,
            reference_te=model,
        )
        print(json.dumps({"fold": fold_info}, indent=2), flush=True)
        if args.stage_root is not None:
            stage_info = fold_mod.stage_minimax_tree(
                minimax_root=Path("/data01/home/chen.xu/xh2_old/MiniMax-H3"),
                stage_root=args.stage_root.expanduser().resolve(),
                folded_te=fold_output,
                overwrite=bool(args.overwrite),
            )
            print(json.dumps({"stage": stage_info}, indent=2), flush=True)


if __name__ == "__main__":
    main()
