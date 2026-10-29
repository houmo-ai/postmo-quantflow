"""Build a converted Bonsai Phase A artifact through the unified API."""

import argparse

from xhmodel_merak.model_build import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--converted-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = build_model(
        model_id="bonsai-27b-phase-a", model_dir=args.converted_dir,
        output_dir=args.output_dir, device=args.device,
    )
    print(result)


if __name__ == "__main__":
    main()
