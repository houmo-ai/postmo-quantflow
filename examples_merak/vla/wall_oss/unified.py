"""Build, infer, or collect connected Golden for Wall-OSS-0.5."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--model-dir", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--precision", choices=["w16a16_sefp", "w8a16_sefp", "w8a8_sefp"], default="w16a16_sefp")
    for name in ("infer", "golden"):
        item = sub.add_parser(name)
        item.add_argument("--artifact-dir", required=True)
        item.add_argument("--input-json", required=True)
        if name == "infer":
            item.add_argument("--output", required=True)
    for item in (build, *[sub.choices[name] for name in ("infer", "golden")]):
        item.add_argument("--device", default="cuda:0")
    sub.choices["golden"].add_argument("--output-dir")
    args = parser.parse_args(argv)
    if args.command == "build":
        from xhmodel_merak.model_build import build_model

        result = build_model(
            model_id="wall-oss-0.5",
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            deployment_precision={"policy": args.precision},
            device=args.device,
        )
        print(result.output_dir)
    elif args.command == "infer":
        from xhmodel_merak.model_inference import inference_model

        result = inference_model(
            artifact_dir=args.artifact_dir,
            expected_model_id="wall-oss-0.5",
            device=args.device,
            inputs=json.loads(Path(args.input_json).read_text()),
        )
        Path(args.output).write_text(json.dumps(result.to_dict(), indent=2))
        print(f"actions: {result.horizon} x {result.action_dim}")
    else:
        from xhmodel_merak.xh_other_model.models.wall_oss.golden import dump_golden

        print(
            json.dumps(
                dump_golden(
                    args.artifact_dir,
                    device=args.device,
                    inputs=json.loads(Path(args.input_json).read_text()),
                    output_dir=args.output_dir,
                ),
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
