# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Validate adaptive reference budgets through packaged HMONNX and the public API."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from xhmodel_merak.model_inference import inference_model
from xhmodel_merak.xh_other_model.models.flux3_klein.bundle import load_runtime
from xhmodel_merak.xh_other_model.models.flux3_klein.dit_graph import DIT_INPUT_NAMES
from xhmodel_merak.xh_other_model.models.flux3_klein.processor import prepare_references

from .validate_dual import PROMPT


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dual-baseline-dir", help="Previous validate_dual deployed output for the first two images")
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if not 3 <= len(args.image) <= 10 or args.steps <= 0:
        parser.error("requires 3 to 10 --image references and positive steps")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    images = []
    for path in args.image:
        with Image.open(path) as image:
            images.append(image.copy())
    runtime = load_runtime(args.artifact_dir, device=args.device)
    capacity = runtime.capacity
    if len(images) > capacity["max_references"]:
        raise ValueError("reference count exceeds the published capacity")
    options = dict(height=args.height, width=args.width, steps=args.steps, seed=args.seed)
    report = dict(artifact_dir=str(Path(args.artifact_dir).resolve()), capacity=capacity, cases=[])
    counts = sorted({2, 3, min(4, len(images)), len(images)})
    for count in counts:
        selected = images[:count]
        image, latent = runtime.generate(args.prompt, images=selected, **options)
        geometry = runtime.last_geometry
        assert len(geometry["references_hw"]) == count
        tokens = sum(h * w // 256 for h, w in geometry["references_hw"])
        assert tokens <= capacity["reference_total_capacity"]
        image.save(output / f"references_{count}.png")
        report["cases"].append(dict(count=count, geometry=geometry, reference_tokens=tokens))
        if count == 2 and args.dual_baseline_dir:
            baseline = Path(args.dual_baseline_dir)
            torch.testing.assert_close(
                latent.cpu(), torch.load(baseline / "hmonnx_dual_latent.pt", weights_only=True), atol=0, rtol=0
            )
            with Image.open(baseline / "hmonnx_dual.png") as previous:
                np.testing.assert_array_equal(np.asarray(image), np.asarray(previous))
            report["dual_baseline_identical"] = True
        if count == len(images):
            final_image, final_latent = image, latent.cpu()
        print(f"adaptive {count} references: {tokens} tokens, {args.steps} steps", flush=True)

    prepared, _ = prepare_references(
        images,
        multi_max_pixels=min(capacity["multi_reference_max_pixels"], capacity["reference_capacity"] * 256),
        max_hw=capacity["canvas_size"],
        total_max_tokens=capacity["reference_total_capacity"],
    )
    for index, image in enumerate(prepared):
        image.save(output / f"prepared_{index}.png")
    # These are connected top-level I/O, not a new per-operator Golden capture.
    session = runtime.sessions["denoise"]
    captured = []

    def capture(*values, deployed_session=session):
        result = deployed_session(*values)
        if not captured:
            ports = output / "dit_io"
            ports.mkdir()
            for name, value in zip(DIT_INPUT_NAMES, values, strict=True):
                np.save(ports / f"{name}.npy", value.cpu().numpy())
            actual = result[0] if isinstance(result, (tuple, list)) else result
            np.save(ports / "output.npy", actual.cpu().numpy())
            captured.append(True)
        return result

    # Change modes, then return: a caller may also preprocess explicitly.
    runtime.generate(args.prompt, images=[], **{**options, "steps": 1, "height": 512, "width": 512})
    runtime.sessions["denoise"] = capture
    image, repeated = runtime.generate(args.prompt, images=prepared, **options)
    torch.testing.assert_close(repeated.cpu(), final_latent, atol=0, rtol=0)
    np.testing.assert_array_equal(np.asarray(image), np.asarray(final_image))
    report["explicit_preprocessing_and_repeat_identical"] = True
    del runtime, session, capture
    torch.cuda.empty_cache()
    public = inference_model(
        artifact_dir=args.artifact_dir,
        expected_model_id="flux-3-image-klein",
        inputs=dict(prompt=args.prompt, images=args.image, **options),
        device=args.device,
    )
    np.testing.assert_array_equal(public.image, np.asarray(final_image))
    assert public.stats["geometry"] == report["cases"][-1]["geometry"]
    report["public_inference_identical"] = True
    report["complete"] = True
    (output / "multi_reference_report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
