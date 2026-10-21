# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Build, generate/edit, or collect connected Golden through the public model API."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--model-dir", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--precision", choices=["w16a16_sefp", "w8a16_sefp", "w8a8_sefp"], default="w16a16_sefp")
    build.add_argument("--device", default="cuda:0")
    build.add_argument("--prompt-upsampler", help="Include and build the raw official prompt upsampler checkpoint")
    build.add_argument("--upsampler-bundle", help="Include a previously built prompt component")
    build.add_argument(
        "--image-bundle", help="Reuse a core FLUX bundle; re-export DiT if only reference capacity changes"
    )
    component = sub.add_parser("build-upsampler")
    component.add_argument("--model-dir", required=True)
    component.add_argument("--output-dir", required=True)
    component.add_argument("--device", default="cuda:0")
    component.add_argument("--config", help="Qwen3.5 workflow YAML; exported metadata determines context and gears")
    component.add_argument("--max-new-tokens", type=int, default=3584, help="Caption generation ceiling")
    component_golden = sub.add_parser("golden-upsampler")
    component_golden.add_argument("--artifact-dir", required=True)
    component_golden.add_argument("--image", action="append", required=True)
    component_golden.add_argument("--device", default="cuda:0")
    for name in ("infer", "golden"):
        command = sub.add_parser(name)
        command.add_argument("--artifact-dir", required=True)
        command.add_argument("--prompt", required=True)
        command.add_argument("--image", action="append", default=[])
        command.add_argument("--height", type=int)
        command.add_argument("--width", type=int)
        command.add_argument("--steps", type=int, default=16)
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--prompt-upsampling", action="store_true")
        if name == "infer":
            command.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "golden-upsampler":
        from xhmodel_merak.xh_other_model.models.flux3_klein.upsampler_golden import dump_component_golden

        print(dump_component_golden(args.artifact_dir, args.image, device=args.device))
        return
    if args.command == "build-upsampler":
        from xhmodel_merak.xh_other_model.models.flux3_klein.upsampler_build import build_component

        result = build_component(
            args.model_dir,
            args.output_dir,
            device=args.device,
            config_path=args.config,
            max_new_tokens=args.max_new_tokens,
        )
        print(json.dumps(result, indent=2))
        return
    if args.command == "build":
        from xhmodel_merak.model_build import build_model

        result = build_model(
            model_id="flux-3-image-klein",
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            device=args.device,
            deployment_precision={"image": args.precision},
            sources={
                key: getattr(args, key)
                for key in ("prompt_upsampler", "upsampler_bundle", "image_bundle")
                if getattr(args, key) is not None
            },
        )
        print(json.dumps({"output_dir": result.output_dir, "loadable": result.loadable}, indent=2))
        return
    values = dict(
        prompt=args.prompt,
        images=args.image,
        steps=args.steps,
        seed=args.seed,
        prompt_upsampling=args.prompt_upsampling,
    )
    for key in ("height", "width"):
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    if args.command == "golden":
        from xhmodel_merak.xh_other_model.models.flux3_klein.golden import dump_golden

        print(dump_golden(args.artifact_dir, device=args.device, inputs=values))
        return
    from PIL import Image

    from xhmodel_merak.model_inference import inference_model

    result = inference_model(
        artifact_dir=args.artifact_dir, expected_model_id="flux-3-image-klein", inputs=values, device=args.device
    )
    Image.fromarray(result.image).save(args.output)
    print(
        json.dumps(
            {
                "width": result.width,
                "height": result.height,
                "seed": result.seed,
                "latency_s": result.latency_s,
                "stats": result.stats,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
