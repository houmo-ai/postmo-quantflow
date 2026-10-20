"""Build a Qwen3.8 Flash Next package with process-local checkpoint compatibility."""

from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Raw Hugging Face model directory.")
    parser.add_argument("--output-dir", help="Final package directory; must not already exist.")
    parser.add_argument("--device", help="Build execution device, e.g. cuda:0.")
    parser.add_argument("--quant-checkpoint", help="Reuse an existing quantized checkpoint.")
    parser.add_argument("--quant-output-dir", help="Destination for newly quantized weights.")
    parser.add_argument(
        "--quant-method", choices=("rtn", "autoround", "none"),
        help="Default: infer from checkpoint or recipe. 'none' requires W16 precision.",
    )
    parser.add_argument(
        "--precision", type=json.loads,
        help='Checkpoint precision JSON, e.g. {"weight":"w4","activation":"a8"}.',
    )
    parser.add_argument(
        "--deployment-precision", type=json.loads,
        help='Component graph precision JSON, e.g. {"visual":"w8a16h1_sefp"}.',
    )
    parser.add_argument("--context-length", type=int, help="Override context capacity.")
    parser.add_argument("--prefill-chunk-length", type=int, help="Override prefill chunk length.")
    parser.add_argument(
        "--speculative", type=json.loads,
        help='Speculative decoding JSON, e.g. {"mode":"mtp","num_draft_tokens":1}.',
    )
    parser.add_argument("--vision", type=json.loads, help="VisionRequest JSON supported by the public build API.")
    parser.add_argument(
        "--low-memory", action=argparse.BooleanOptionalAction, default=None,
        help="Select a registered export mode; defaults to the recipe mode.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    from xhmodel_merak.xh_llm.models.qwen4_exp.build_entry import build_qwen4_exp_model

    kwargs = vars(args)
    method = kwargs.pop("quant_method")
    if method is not None:
        kwargs["quant_method"] = None if method == "none" else method
    result = build_qwen4_exp_model(**kwargs)
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
