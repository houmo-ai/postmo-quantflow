"""Build or run Qwen-Image-2.1 through the public Merak interfaces."""

import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", choices=("qwen-image-2.1", "qwen-image-2.1-turbo"), default="qwen-image-2.1")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--model-dir", help="Build a new artifact from this Diffusers checkpoint first")
    parser.add_argument(
        "--quant-type",
        choices=("w16a16_sefp", "w8a16_sefp", "w8a8_sefp"),
        help="HMONNX export precision when building; defaults to w16a16_sefp",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--image", action="append", default=[], help="Reference image fitted to the published capacity")
    parser.add_argument("--height", type=int, help="Output height, a multiple of 32; defaults to package metadata")
    parser.add_argument("--width", type=int, help="Output width, a multiple of 32; defaults to package metadata")
    parser.add_argument("--steps", type=int, help="Sampling steps when the checkpoint has no fixed sigma grid")
    parser.add_argument("--sigmas", type=float, nargs="+", help="Override the checkpoint sampling grid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-prompt", help="CFG negative prompt; an empty string is allowed")
    parser.add_argument(
        "--guidance-scale", type=float, default=1.0, help="CFG scale; values above 1 require --negative-prompt"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    if args.quant_type is not None and not args.model_dir:
        parser.error("--quant-type requires --model-dir; an existing package already fixes its precision")
    if args.guidance_scale > 1 and args.negative_prompt is None:
        parser.error("--guidance-scale above 1 requires --negative-prompt")

    from PIL import Image

    from xhmodel_merak.model_build import build_model
    from xhmodel_merak.model_inference import inference_model

    artifact_dir = str(Path(args.artifact_dir).expanduser().resolve())
    if args.model_dir:
        build = build_model(
            model_id=args.model_id,
            model_dir=args.model_dir,
            output_dir=artifact_dir,
            device=args.device,
            deployment_precision={"image": args.quant_type or "w16a16_sefp"},
        )
        artifact_dir = build.output_dir
    inputs = dict(
        prompt=args.prompt, images=args.image, seed=args.seed, guidance_scale=args.guidance_scale
    )
    if args.negative_prompt is not None:
        inputs["negative_prompt"] = args.negative_prompt
    for name in ("height", "width", "steps", "sigmas"):
        if getattr(args, name) is not None:
            inputs[name] = getattr(args, name)
    result = inference_model(
        artifact_dir=artifact_dir,
        expected_model_id=args.model_id,
        device=args.device,
        inputs=inputs,
    )
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.image).save(output)
    print(f"Saved {result.width}x{result.height} RGBA to {args.output}; generation {result.latency_s:.3f}s")


if __name__ == "__main__":
    main()
