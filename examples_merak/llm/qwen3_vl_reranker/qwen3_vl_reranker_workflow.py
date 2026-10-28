"""Export the real Reranker through the public build interface."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-length", type=int, default=32768)
    parser.add_argument("--weight-bits", type=int, choices=[8, 4], default=8)
    parser.add_argument("--quant-output-dir")
    parser.add_argument("--quant-checkpoint", help="Reuse an existing AutoRound W4 checkpoint")
    parser.add_argument("--prefill-length", type=int, default=256)
    args = parser.parse_args()
    from xhmodel_merak.model_build import build_model

    result = build_model(
        model_id="qwen3-vl-reranker-2b",
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        device=args.device,
        precision={"weight": "w4" if args.weight_bits == 4 else "w16", "activation": "a8"},
        quant_output_dir=args.quant_output_dir,
        quant_checkpoint=args.quant_checkpoint,
        context_length=args.context_length,
        prefill_chunk_length=args.prefill_length,
    )
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
