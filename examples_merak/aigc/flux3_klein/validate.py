# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Record native BF16 trajectories, then compare deployed generation and editing."""

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from xhmodel_merak.xh_other_model.models.flux3_klein.dit_graph import image_ids
from xhmodel_merak.xh_other_model.models.flux3_klein.native_dit import Packed
from xhmodel_merak.xh_other_model.models.flux3_klein.native_text_encoder import TextEncoder
from xhmodel_merak.xh_other_model.models.flux3_klein.source import load_native, read_source


CASES = (
    (
        "generate",
        (
            "A realistic photograph of a red ceramic teapot with a round glossy body, a curved spout and a loop "
            "handle, resting on a wooden table beside a window. Soft natural daylight illuminates the scene, "
            "with a neutral background and gentle shadows."
        ),
        1024,
        1024,
    ),
    (
        "edit",
        (
            "A realistic photograph of the same ceramic teapot as in the reference image, now with a blue glossy "
            "glaze covering its body, curved spout and loop handle. It rests on the same wooden table beside a "
            "window, preserving the scene and soft natural daylight."
        ),
        768,
        1376,
    ),
)


def metrics(actual, expected):
    a, b = actual.detach().double().cpu().flatten(), expected.detach().double().cpu().flatten()
    return dict(
        max_abs=float((a - b).abs().max()),
        rmse=float((a - b).square().mean().sqrt()),
        cosine=float(torch.nn.functional.cosine_similarity(a[None], b[None])),
    )


def check_trajectory(comparisons):
    """Reject broken graph/sampling contracts before declaring runtime parity.

    These broad numerical gates catch saturation and state/ordering mistakes;
    the report retains all errors for assessment of the particular precision.
    """
    first, last = comparisons[0], comparisons[-1]
    if first["cosine"] < 0.999 or first["rmse"] > 0.01 or last["cosine"] < 0.8 or last["rmse"] > 1.0:
        raise AssertionError(f"deployed trajectory failed parity: first={first}, last={last}")


@torch.inference_mode()
def native_run(source_dir, output, device, steps, seed):
    config, _ = read_source(source_dir)
    encoder = TextEncoder(str(Path(source_dir) / "text_encoder"), device=device)
    vae = load_native(source_dir, "vae", device=device, dtype=torch.float32)
    dit = load_native(source_dir, "dit", device=device, dtype=torch.bfloat16)
    report = dict(
        backend="official_native_bf16", model_dir=str(Path(source_dir).resolve()), steps=steps, seed=seed, cases=[]
    )
    reference = None
    for name, prompt, height, width in CASES:
        text, text_ids = encoder.encode(prompt)
        refs, ref_ids = torch.empty(0, 128, device=device, dtype=torch.bfloat16), torch.empty(0, 4, device=device)
        if reference is not None:
            pixel = torch.from_numpy(np.asarray(reference).copy()).permute(2, 0, 1).float().to(device) / 127.5 - 1
            refs, _ = vae.encode(pixel)
            refs, ref_ids = (
                refs.to(torch.bfloat16),
                image_ids(reference.height, reference.width, reference=0, device=device),
            )
        noise = (
            torch.randn((height * width // 256, 128), generator=torch.Generator().manual_seed(seed))
            .to(torch.bfloat16)
            .float()
            .to(device)
        )
        ids = image_ids(height, width, device=device)
        data = dict(image=noise, image_cond=refs, text=text)
        packed = Packed(
            data,
            dict(image=ids, image_cond=ref_ids, text=text_ids),
            {key: torch.zeros(value.shape[0], device=device, dtype=torch.bfloat16) for key, value in data.items()},
            {key: [(0, value.shape[0])] for key, value in data.items()},
        )
        raw = torch.linspace(1.0, 0.0, steps + 1)
        alpha = config["sampler"]["alpha"]
        schedule = alpha * raw / (1 + (alpha - 1) * raw)
        latents = []
        for index in range(steps):
            current, following = schedule[index].item(), schedule[index + 1].item()
            packed.timesteps["image"].fill_(current)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                velocity = dit(packed, timesteps_ref=schedule[index + 1 : index + 2].to(device))
            packed.data["image"] = packed.data["image"] + (following - current) * velocity
            if not bool(torch.isfinite(packed.data["image"]).all()):
                raise FloatingPointError(f"native {name} step {index}")
            latents.append(packed.data["image"].cpu().clone())
            print(f"native {name} {index + 1}/{steps}", flush=True)
        # AutoEncoder geometry uses unscaled integer ids.
        decode_ids = ids.clone()
        decode_ids[:, 1:3] *= 2
        pixels = vae.decode(packed.data["image"], decode_ids)
        array = ((pixels.float().clamp(-1, 1) + 1) * 127.5).byte().permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(array)
        image.save(output / f"native_{name}.png")
        torch.save(
            dict(latents=latents, prompt=prompt, height=height, width=width, seed=seed, steps=steps),
            output / f"native_{name}.pt",
        )
        report["cases"].append(dict(name=name, height=height, width=width, text_tokens=text.shape[0], finite=True))
        if name == "generate":
            reference = image
    (output / "native_report.json").write_text(json.dumps(report, indent=2) + "\n")
    del encoder, vae, dit
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def deployed_run(artifact, output, device):
    from xhmodel_merak.xh_other_model.models.flux3_klein.bundle import load_runtime

    runtime = load_runtime(artifact, device=device)
    reference = Image.open(output / "native_generate.png").convert("RGB")
    report = dict(backend="hmonnx_v2", artifact_dir=str(Path(artifact).resolve()), cases=[])
    for name, _, _, _ in CASES:
        native = torch.load(output / f"native_{name}.pt", weights_only=True)
        frames = []

        def observe(index, time, value, frames=frames, name=name, steps=native["steps"]):
            frames.append(value.cpu())
            print(f"hmonnx {name} {index + 1}/{steps}", flush=True)

        image, final = runtime.generate(
            native["prompt"],
            height=native["height"],
            width=native["width"],
            steps=native["steps"],
            seed=native["seed"],
            images=[reference] if name == "edit" else [],
            callback=observe,
        )
        comparisons = [metrics(a, b) for a, b in zip(frames, native["latents"], strict=True)]
        expected = np.asarray(Image.open(output / f"native_{name}.png")).astype(np.float64)
        actual = np.asarray(image).astype(np.float64)
        mse = float(np.square(actual - expected).mean())
        image.save(output / f"hmonnx_{name}.png")
        torch.save(final.cpu(), output / f"hmonnx_{name}_latent.pt")
        report["cases"].append(
            dict(
                name=name,
                height=image.height,
                width=image.width,
                rgb_psnr=math.inf if mse == 0 else 10 * math.log10(255**2 / mse),
                steps=comparisons,
                geometry=runtime.last_geometry,
            )
        )
        (output / "deployment_report.json").write_text(json.dumps(report, indent=2) + "\n")
        check_trajectory(comparisons)
    del runtime
    gc.collect()
    torch.cuda.empty_cache()
    verify_requests(artifact, output, device)


@torch.inference_mode()
def verify_requests(artifact, output, device):
    """Resume request-isolation/public checks using already saved paired results."""
    from xhmodel_merak.model_inference import inference_model
    from xhmodel_merak.xh_other_model.models.flux3_klein.bundle import load_runtime

    report = json.loads((output / "deployment_report.json").read_text())
    for case in report["cases"]:
        check_trajectory(case["steps"])
    first = torch.load(output / "native_generate.pt", weights_only=True)
    public = inference_model(
        artifact_dir=artifact,
        expected_model_id="flux-3-image-klein",
        device=device,
        inputs=dict(
            prompt=first["prompt"],
            height=first["height"],
            width=first["width"],
            steps=first["steps"],
            seed=first["seed"],
        ),
    )
    np.testing.assert_array_equal(public.image, np.asarray(Image.open(output / "hmonnx_generate.png")))
    report.update(public_inference_identical=True, trajectory_parity_passed=True)
    (output / "deployment_report.json").write_text(json.dumps(report, indent=2) + "\n")
    del public
    gc.collect()
    torch.cuda.empty_cache()
    runtime = load_runtime(artifact, device=device)
    edited = torch.load(output / "native_edit.pt", weights_only=True)
    reference = Image.open(output / "native_generate.png").convert("RGB")
    runtime.generate(edited["prompt"], images=[reference], height=edited["height"], width=edited["width"], steps=1)
    runtime.generate(first["prompt"], height=512, width=512, steps=1, seed=first["seed"])
    _, repeat = runtime.generate(
        first["prompt"], height=first["height"], width=first["width"], steps=first["steps"], seed=first["seed"]
    )
    torch.testing.assert_close(
        repeat.cpu(), torch.load(output / "hmonnx_generate_latent.pt", weights_only=True), atol=0, rtol=0
    )
    report["repeated_request_identical"] = True
    (output / "deployment_report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["native", "deployed", "checks"])
    parser.add_argument("--model-dir", default="/data01/datasets/flux-3-image-klein")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "native":
        native_run(args.model_dir, output, args.device, args.steps, args.seed)
    elif args.artifact_dir:
        operation = verify_requests if args.mode == "checks" else deployed_run
        operation(args.artifact_dir, output, args.device)
    else:
        parser.error("deployed mode requires --artifact-dir")


if __name__ == "__main__":
    main()
