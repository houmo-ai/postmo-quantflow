"""Build Flash-0731 and its bundled W4/W8 DSpark graphs from an existing checkpoint."""
import argparse

from xhmodel_merak.model_build import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--activation-bits", type=int, choices=(8, 16), default=8)
    args = parser.parse_args()
    quant_type = f"w8a{args.activation_bits}h1_sefp"
    result = build_model(model_id="deepseek-v4-flash-0731", model_dir=args.checkpoint,
                          quant_checkpoint=args.checkpoint, output_dir=args.output_dir,
                          device=args.device, speculative={"mode": "dspark", "num_draft_tokens": 5},
                          deployment_precision={"prefill": quant_type, "decode": quant_type})
    print(result)


if __name__ == "__main__":
    main()
