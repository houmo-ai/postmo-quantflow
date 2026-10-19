# Copyright (c) 2024 Haofei Xu
# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: MIT
# Gaussian adapter orchestration adapted from DepthSplat; see licenses/DepthSplat/LICENSE.
"""Real scene preparation and streamed rendering using an external CUDA renderer."""

import json
from pathlib import Path

import numpy as np

SOURCE_PROFILES = {"2v": "2v_256x256", "6v": "6v_512x960"}
PROFILES = {"2v": (2, 256, 256, 3.0), "6v": (6, 512, 960, 0.1)}


def load_scene(dataset_dir, scene):
    import torch

    root = Path(dataset_dir)
    paths = [root] if root.is_file() else sorted(root.glob("*.torch"))
    if not paths and (root / "test").is_dir():
        paths = sorted((root / "test").glob("*.torch"))
    for path in paths:
        for example in torch.load(path, map_location="cpu", weights_only=True):
            if example["key"] == scene:
                return example
    raise ValueError(f"Scene {scene!r} not found in {root}")


def prepare_sample(dataset_dir, scene, profile, source_dir, output, context_indices=None, target_indices=None):
    from .source import prepare_views

    example = load_scene(dataset_dir, scene)
    if context_indices is None:
        name = "evaluation_index_re10k.json" if profile == "2v" else "re10k_ctx_6v_video.json"
        context_indices = json.loads((Path(source_dir) / "assets" / name).read_text())[scene]["context"]
    if len(context_indices) != PROFILES[profile][0] or len(set(context_indices)) != len(context_indices):
        raise ValueError(f"{profile} requires distinct context frames")
    target_indices = list(range(len(example["images"]))) if target_indices is None else target_indices
    if (
        not target_indices
        or min(context_indices + target_indices) < 0
        or max(context_indices + target_indices) >= len(example["images"])
    ):
        raise ValueError("Frame indices are empty or outside the scene")
    values = prepare_views(example, context_indices, SOURCE_PROFILES[profile], source_dir)
    # Targets are decoded lazily during rendering, keeping full trajectories small.
    values["target_index"] = np.asarray(target_indices, dtype=np.int64)[None]
    values["scene"] = np.asarray(scene)
    values["profile"] = np.asarray(profile)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **values)
    return {
        "scene": scene,
        "profile": profile,
        "context_indices": context_indices,
        "target_frames": len(target_indices),
        "output": str(output),
    }


def _source_imports(source_dir):
    import sys
    import types

    root = Path(source_dir).resolve()
    if not (root / "src/model/decoder/cuda_splatting.py").is_file():
        raise ValueError("--source-dir must point to the DepthSplat source checkout")
    sys.path.insert(0, str(root))
    # Avoid importing unrelated encoder/decoder registries and their optional dependencies.
    for name, suffix in (("src.model.encoder", "src/model/encoder"), ("src.model.decoder", "src/model/decoder")):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(root / suffix)]
            sys.modules[name] = package


def raw_to_gaussians(raw, context, profile, source_dir):
    """Run the original Gaussian adapter on CPU FP32 without loading an encoder."""
    import torch

    _source_imports(source_dir)
    from src.geometry.projection import sample_image_grid
    from src.model.encoder.common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg

    views, height, width, scale_max = PROFILES[profile]
    images = torch.from_numpy(context.get("image", context.get("images")))
    if images.dtype != torch.float32 or tuple(images.shape) != (1, views, 3, height, width):
        raise ValueError("Context images must retain the original profile's FP32 data")
    cameras = [torch.from_numpy(context[name]) for name in ("intrinsics", "extrinsics")]
    if any(x.dtype != torch.float32 or not torch.isfinite(x).all() for x in [images] + cameras):
        raise ValueError("Context cameras and images must be finite original FP32 tensors")
    if raw["depths"].shape != (1, views, height * width, 1, 1) or raw["raw_gaussians"].shape != (
        views,
        37,
        height,
        width,
    ):
        raise ValueError("Raw output shapes do not match the selected profile")
    if any(not np.isfinite(value).all() for value in raw.values()):
        raise ValueError("Raw outputs contain nonfinite values")
    depths, values = (torch.from_numpy(raw[name]).float() for name in ("depths", "raw_gaussians"))
    intrinsics, extrinsics = cameras
    adapter = GaussianAdapter(GaussianAdapterCfg(1e-10, scale_max, 2)).cpu().eval()
    with torch.inference_mode():
        values = (
            values.reshape(1, views, 37, height, width).permute(0, 1, 3, 4, 2).reshape(1, views, height * width, 37)
        )
        opacity = values[..., :1].sigmoid().unsqueeze(-1)
        xy, _ = sample_image_grid((height, width))
        xy = xy.reshape(height * width, 1, 2) + (values[..., 1:3].sigmoid().unsqueeze(-2) - 0.5) / torch.tensor(
            [width, height]
        )
        result = adapter(
            extrinsics[:, :, None, None, None],
            intrinsics[:, :, None, None, None],
            xy.unsqueeze(-2),
            depths,
            opacity,
            values[..., 3:].unsqueeze(-2).unsqueeze(-2),
            (height, width),
            input_images=images,
        )
    return {
        name: getattr(result, name).reshape((1, -1) + tail).numpy()
        for name, tail in (("means", (3,)), ("covariances", (3, 3)), ("harmonics", (3, 9)), ("opacities", ()))
    }


def render_video(
    *,
    dataset_dir,
    scene,
    profile,
    source_dir,
    context_path,
    output_dir,
    gaussians_path=None,
    raw_path=None,
    device="cuda:0",
    max_frames=None,
    chunk_size=10,
    fps=30,
    baseline_metrics=None,
):
    import subprocess
    import time

    import imageio_ffmpeg
    import torch
    from PIL import Image
    from skimage.metrics import structural_similarity

    from .source import prepare_views

    if (gaussians_path is None) == (raw_path is None):
        raise ValueError("Provide exactly one Gaussian NPZ or raw-output NPZ")
    if chunk_size < 1 or fps < 1 or (max_frames is not None and max_frames < 1):
        raise ValueError("Chunk size, frame limit and fps must be positive")
    if torch.device(device).type != "cuda":
        raise ValueError("The external Gaussian renderer requires a CUDA device")
    _source_imports(source_dir)
    from src.model.decoder.cuda_splatting import render_cuda

    start = time.perf_counter()
    with np.load(context_path) as archive:
        context = dict(archive)
    if "scene" in context and str(context["scene"]) != scene:
        raise ValueError("Context scene differs from requested scene")
    if "profile" in context and str(context["profile"]) != profile:
        raise ValueError("Context profile differs from requested profile")
    with np.load(raw_path or gaussians_path) as archive:
        values = dict(archive)
    gaussians = raw_to_gaussians(values, context, profile, source_dir) if raw_path else values
    count = PROFILES[profile][0] * PROFILES[profile][1] * PROFILES[profile][2]
    tails = {"means": (3,), "covariances": (3, 3), "harmonics": (3, 9), "opacities": ()}
    for name, tail in tails.items():
        value = gaussians[name]
        if value.shape != (1, count) + tail or not np.isfinite(value).all():
            raise ValueError(f"Invalid {name} Gaussian tensor for {profile}")
    tensors = {name: torch.from_numpy(gaussians[name]).float().to(device) for name in tails}
    example = load_scene(dataset_dir, scene)
    ids = context.get("target_index", np.arange(len(example["images"]))[None]).reshape(-1).tolist()
    ids = ids[:max_frames]
    if not ids or min(ids) < 0 or max(ids) >= len(example["images"]):
        raise ValueError("Invalid target frame indices")
    baseline = None
    if baseline_metrics:
        entries = json.loads(Path(baseline_metrics).read_text())
        baseline = {row["target_index"]: dict(row) for row in entries}
        if len(baseline) != len(entries) or not set(ids).issubset(baseline):
            raise ValueError("Baseline must have unique target indices and cover every rendered frame")
        for row in baseline.values():
            # Earlier render reports recorded PSNR only; recover its equivalent pixel MSE.
            row.setdefault("mse", float(10 ** (-row["psnr_db"] / 10)))
            if not np.isfinite([row["psnr_db"], row["ssim"], row["mse"]]).all() or row["mse"] <= 0:
                raise ValueError("Baseline metrics must be finite and baseline MSE positive")
    context_ids = context["index"].reshape(-1).tolist()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _, height, width, _ = PROFILES[profile]
    video = output / "trajectory.mp4"
    rows = []
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with open(output / "ffmpeg.log", "w") as log:
        writer = subprocess.Popen(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                str(video),
            ],
            stdin=subprocess.PIPE,
            stderr=log,
        )
        try:
            for begin in range(0, len(ids), chunk_size):
                selected = ids[begin : begin + chunk_size]
                target = prepare_views(example, selected, SOURCE_PROFILES[profile], source_dir)
                n = len(selected)
                camera = [
                    torch.as_tensor(target[name][0], device=device)
                    for name in ("extrinsics", "intrinsics", "near", "far")
                ]
                with torch.inference_mode():
                    prediction = (
                        render_cuda(
                            *camera,
                            (height, width),
                            torch.zeros((n, 3), device=device),
                            tensors["means"].repeat(n, 1, 1),
                            tensors["covariances"].repeat(n, 1, 1, 1),
                            tensors["harmonics"].repeat(n, 1, 1, 1),
                            tensors["opacities"].repeat(n, 1),
                        )
                        .cpu()
                        .numpy()
                    )
                if not np.isfinite(prediction).all():
                    raise ValueError("Renderer produced nonfinite pixels")
                for j, index in enumerate(selected):
                    gt = np.asarray(target["image"][0, j])
                    mse = float(np.mean((np.clip(prediction[j], 0, 1) - gt) ** 2))
                    rows.append(
                        {
                            "target_index": index,
                            "is_context": index in context_ids,
                            "mse": mse,
                            "psnr_db": float(-10 * np.log10(max(mse, np.finfo(np.float32).tiny))),
                            "ssim": float(
                                structural_similarity(
                                    gt,
                                    prediction[j],
                                    win_size=11,
                                    gaussian_weights=True,
                                    channel_axis=0,
                                    data_range=1.0,
                                )
                            ),
                        }
                    )
                    rgb = (np.clip(prediction[j].transpose(1, 2, 0), 0, 1) * 255).round().astype("uint8")
                    writer.stdin.write(rgb.tobytes())
                    if len(rows) in {1, len(ids) // 2 + 1, len(ids)}:
                        Image.fromarray(rgb).save(output / f"frame-{index:04d}.png")
                print(f"Rendered {len(rows)}/{len(ids)}", flush=True)
        finally:
            writer.stdin.close()
            status = writer.wait()
        if status:
            raise RuntimeError(f"ffmpeg failed ({status}); see {output / 'ffmpeg.log'}")
    frames, seconds = imageio_ffmpeg.count_frames_and_secs(str(video))
    if frames != len(rows):
        raise RuntimeError(f"Video frame count {frames} differs from rendered count {len(rows)}")
    report = {
        "scene": scene,
        "profile": profile,
        "context_indices": context_ids,
        "frame_count": len(rows),
        "resolution": [height, width],
        "fps": fps,
        "video_seconds": seconds,
        "wall_seconds": time.perf_counter() - start,
        "aggregates": {},
    }
    for split in ("all", "noncontext"):
        selected = [row for row in rows if split == "all" or not row["is_context"]]
        aggregate = {"frames": len(selected)}
        if selected:
            aggregate.update(
                {name: float(np.mean([row[name] for row in selected])) for name in ("psnr_db", "ssim", "mse")}
            )
            if baseline is not None:
                reference = [baseline[row["target_index"]] for row in selected]
                aggregate["baseline"] = {
                    name: float(np.mean([row[name] for row in reference])) for name in ("psnr_db", "ssim", "mse")
                }
                aggregate["difference"] = {
                    name: aggregate[name] - aggregate["baseline"][name] for name in ("psnr_db", "ssim", "mse")
                }
                aggregate["mse_ratio_of_means_increase_pct"] = 100 * (
                    aggregate["mse"] / aggregate["baseline"]["mse"] - 1
                )
                aggregate["mean_frame_mse_increase_pct"] = float(
                    np.mean([100 * (row["mse"] / ref["mse"] - 1) for row, ref in zip(selected, reference)])
                )
        report["aggregates"][split] = aggregate
    (output / "metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
