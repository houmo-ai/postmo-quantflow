# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Execute packaged expansion, generation/editing and direct-caption parity."""

import argparse
import json
from pathlib import Path


def main():
    import torch
    from PIL import Image

    from xhmodel_merak.xh_other_model.models.flux3_klein.bundle import load_runtime, read_metadata

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument(
        "--reference", action="append", required=True, help="Ordered reference; repeat for multiple images"
    )
    parser.add_argument("--reference-counts", help="Comma-separated counts to validate, e.g. 1,2,4,8")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    meta = read_metadata(args.artifact_dir)
    if not meta.get("prompt_upsampler"):
        raise ValueError("validation requires a packaged prompt upsampler")
    references = []
    for path in args.reference:
        with Image.open(path) as source:
            references.append(source.convert("RGB"))
    counts = [len(references)] if args.reference_counts is None else [int(x) for x in args.reference_counts.split(",")]
    if not counts or any(count <= 0 or count > len(references) for count in counts):
        raise ValueError("reference counts must fit the supplied ordered references")
    runtime = load_runtime(args.artifact_dir, device=args.device)
    cases = []
    try:
        requests = [
            ("generate", "A red ceramic teapot on a wooden table beside a window.", []),
            *[
                (
                    "edit" if count == 1 else f"edit_{count}",
                    "Change the teapot to blue and keep everything else unchanged."
                    if count == 1
                    else "Create one coherent photograph combining the reference objects in attachment "
                    "order, preserving their colors and identities.",
                    references[:count],
                )
                for count in counts
            ],
        ]
        for name, prompt, images in requests:
            image, latent = runtime.generate(
                prompt, images=images, height=1024, width=1024, seed=42, steps=args.steps, prompt_upsampling=True
            )
            image.save(output / f"hmonnx_{name}.png")
            caption, expansion = runtime.last_caption, runtime.last_upsampler_stats
            geometry = runtime.last_geometry
            (output / f"hmonnx_{name}.txt").write_text(caption + "\n")
            direct, direct_latent = runtime.generate(
                caption, images=images, height=1024, width=1024, seed=42, steps=args.steps, prompt_upsampling=False
            )
            if not torch.equal(latent, direct_latent) or image.tobytes() != direct.tobytes():
                raise AssertionError("expanded caption and direct caption paths diverged")
            if not torch.isfinite(latent).all() or runtime.last_upsampler_stats is not None:
                raise AssertionError("nonfinite latent or stale expansion state")
            cases.append(
                dict(
                    name=name,
                    reference_count=len(images),
                    caption=caption,
                    expansion=expansion,
                    geometry=geometry,
                    direct_caption_bit_exact=True,
                    finite=True,
                )
            )
            print(json.dumps(cases[-1]), flush=True)
        # A prior multi-image request must not affect the next request's autoregressive state.
        name, prompt, images = requests[-1]
        if runtime.upsampler(prompt, images, (1024, 1024)) != cases[-1]["caption"]:
            raise AssertionError("prompt state leaked across requests")
    finally:
        runtime.upsampler.close()
        del runtime
    report = dict(
        backend="hmonnx_v2",
        steps=args.steps,
        seed=42,
        cases=cases,
        repeated_expansion_identical=True,
        graphs={**meta["graphs"], **{f"upsampler_{k}": v for k, v in meta["prompt_upsampler"]["graphs"].items()}},
    )
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
