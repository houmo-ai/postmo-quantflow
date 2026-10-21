# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Compare two-reference HMONNX editing against the native BF16 trajectory."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from xhmodel_merak.xh_other_model.models.flux3_klein.capacity import validate_capacity, validate_request
from xhmodel_merak.xh_other_model.models.flux3_klein.dit_graph import image_ids
from xhmodel_merak.xh_other_model.models.flux3_klein.processor import prepare_references

from .validate import check_trajectory, metrics


PROMPT = (
    "Create a realistic photograph showing the red ceramic teapot from the first reference image on the left "
    "and the blue ceramic teapot from the second reference image on the right, together on a wooden table "
    "beside a window. Preserve each teapot's distinct color, shape, spout and handle. Soft natural daylight "
    "illuminates both teapots, with gentle shadows and a neutral background."
)


def references(paths):
    result = []
    for path in paths:
        with Image.open(path) as image:
            result.append(image.copy())
    return result


@torch.inference_mode()
def native(args, output):
    from xhmodel_merak.xh_other_model.models.flux3_klein.native_dit import Packed
    from xhmodel_merak.xh_other_model.models.flux3_klein.native_text_encoder import TextEncoder
    from xhmodel_merak.xh_other_model.models.flux3_klein.source import load_native, read_source

    capacity = validate_capacity({})
    prepared, _ = prepare_references(
        references(args.image), multi_max_pixels=capacity["multi_reference_max_pixels"], max_hw=capacity["canvas_size"]
    )
    shapes = [(image.height, image.width) for image in prepared]
    validate_request(capacity, (args.height, args.width), shapes, 80)
    config, _ = read_source(args.model_dir)
    encoder = TextEncoder(str(Path(args.model_dir) / "text_encoder"), device=args.device)
    text, text_ids = encoder.encode(args.prompt)
    validate_request(capacity, (args.height, args.width), shapes, text.shape[0])
    del encoder
    torch.cuda.empty_cache()
    vae = load_native(args.model_dir, "vae", device=args.device)
    latents = []
    for image in prepared:
        pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().to(args.device) / 127.5 - 1
        latent, _ = vae.encode(pixels)
        latents.append(latent.to(torch.bfloat16))
    refs = torch.cat(latents)
    ids = image_ids(args.height, args.width, device=args.device)
    noise = torch.randn((ids.shape[0], 128), generator=torch.Generator().manual_seed(args.seed))
    data = dict(image=noise.to(torch.bfloat16).float().to(args.device), image_cond=refs, text=text)
    packed = Packed(
        data,
        dict(
            image=ids,
            image_cond=torch.cat(
                [image_ids(*shape, reference=i, device=args.device) for i, shape in enumerate(shapes)]
            ),
            text=text_ids,
        ),
        {key: torch.zeros(value.shape[0], device=args.device, dtype=torch.bfloat16) for key, value in data.items()},
        {key: [(0, value.shape[0])] for key, value in data.items()},
    )
    dit = load_native(args.model_dir, "dit", device=args.device, dtype=torch.bfloat16)
    time = torch.linspace(1, 0, args.steps + 1)
    alpha = config["sampler"]["alpha"]
    schedule = alpha * time / (1 + (alpha - 1) * time)
    frames = []
    for index in range(args.steps):
        current, following = schedule[index].item(), schedule[index + 1].item()
        packed.timesteps["image"].fill_(current)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            velocity = dit(packed, timesteps_ref=schedule[index + 1 : index + 2].to(args.device))
        packed.data["image"] = packed.data["image"] + (following - current) * velocity
        if not bool(torch.isfinite(packed.data["image"]).all()):
            raise FloatingPointError(f"native dual reference step {index}")
        frames.append(packed.data["image"].cpu().clone())
        print(f"native dual {index + 1}/{args.steps}", flush=True)
    decode_ids = ids.clone()
    decode_ids[:, 1:3] *= 2
    pixels = vae.decode(packed.data["image"], decode_ids)
    array = ((pixels.float().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(output / "native_dual.png")
    state = dict(
        prompt=args.prompt,
        images=[str(Path(path).resolve()) for path in args.image],
        height=args.height,
        width=args.width,
        steps=args.steps,
        seed=args.seed,
        references_hw=shapes,
        text_tokens=text.shape[0],
        latents=frames,
    )
    torch.save(state, output / "native_dual.pt")
    (output / "native_dual_report.json").write_text(
        json.dumps({k: v for k, v in state.items() if k != "latents"}, indent=2) + "\n"
    )


@torch.inference_mode()
def deployed(args, output):
    from xhmodel_merak.model_inference import inference_model
    from xhmodel_merak.xh_other_model.models.flux3_klein.bundle import load_runtime, read_metadata

    baseline = Path(args.baseline_dir or output)
    state = torch.load(baseline / "native_dual.pt", weights_only=True)
    options = {key: state[key] for key in ("height", "width", "steps", "seed")}
    images = references(state["images"])
    runtime = load_runtime(args.artifact_dir, device=args.device)
    frames = []

    def observe(index, time, latent):
        frames.append(latent[0].cpu())
        print(f"hmonnx dual {index + 1}/{state['steps']}", flush=True)

    image, final = runtime.generate(state["prompt"], images=images, callback=observe, **options)
    comparisons = [metrics(a, b) for a, b in zip(frames, state["latents"], strict=True)]
    check_trajectory(comparisons)
    assert runtime.last_geometry["references_hw"] == [list(shape) for shape in state["references_hw"]]
    image.save(output / "hmonnx_dual.png")
    torch.save(final.cpu(), output / "hmonnx_dual_latent.pt")
    expected = np.asarray(Image.open(baseline / "native_dual.png")).astype(np.float64)
    mse = float(np.square(np.asarray(image).astype(np.float64) - expected).mean())
    report = dict(
        artifact_dir=str(Path(args.artifact_dir).resolve()),
        quant_type=read_metadata(args.artifact_dir)["quant_type"],
        geometry=runtime.last_geometry,
        steps=comparisons,
        rgb_psnr=math.inf if mse == 0 else 10 * math.log10(255**2 / mse),
    )
    # Exercise both reference positions, then switch request modes and return.
    one_step = {**options, "steps": 1}
    _, ordered = runtime.generate(state["prompt"], images=images, **one_step)
    _, swapped = runtime.generate(state["prompt"], images=images[::-1], **one_step)
    assert not torch.equal(ordered, swapped), "reference order did not affect the real DiT"
    resized = [images[0], images[1].resize((1376, 768), Image.Resampling.LANCZOS)]
    runtime.generate(state["prompt"], images=resized, **one_step)
    report["mixed_reference_geometry"] = runtime.last_geometry
    assert len({tuple(shape) for shape in runtime.last_geometry["references_hw"]}) == 2
    assert all(h * w <= 1024**2 for h, w in runtime.last_geometry["references_hw"])
    for changed in (images[:1], []):
        runtime.generate(state["prompt"], images=changed, **{**one_step, "height": 512, "width": 512})
    _, repeated = runtime.generate(state["prompt"], images=images, **options)
    torch.testing.assert_close(repeated, final, atol=0, rtol=0)
    report.update(reference_order_affects_output=True, repeated_after_mode_changes_identical=True)
    del runtime
    torch.cuda.empty_cache()
    public = inference_model(
        artifact_dir=args.artifact_dir,
        expected_model_id="flux-3-image-klein",
        inputs=dict(prompt=state["prompt"], images=state["images"], **options),
        device=args.device,
    )
    np.testing.assert_array_equal(public.image, np.asarray(image))
    report.update(public_inference_identical=True, trajectory_parity_passed=True)
    (output / "dual_reference_report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("native", "deployed"))
    parser.add_argument("--model-dir", default="/data01/datasets/flux-3-image-klein")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--baseline-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--prompt", default=PROMPT)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.mode == "native" and (len(args.image) != 2 or args.steps <= 0):
        parser.error("native requires exactly two --image references and positive steps")
    if args.mode == "deployed" and not args.artifact_dir:
        parser.error("deployed requires --artifact-dir")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (native if args.mode == "native" else deployed)(args, output)


if __name__ == "__main__":
    main()
