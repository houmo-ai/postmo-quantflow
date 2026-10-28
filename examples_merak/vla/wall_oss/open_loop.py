"""Paired LIBERO numeric evaluation of the current, unfine-tuned Wall policy."""

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np


DATASET_REVISION = "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4"
UPSTREAM_REVISION = "ce2a1ae31d9ee1028b97ca40e3f66034cf8c0e5f"
FLOAT_GATES = dict(
    visual_atol=0.002,
    visual_rtol=0.0005,
    visual_nrmse_max=0.0001,
    visual_max_abs=0.05,
    visual_cosine_min=0.99999999,
    trajectory_atol=0.0002,
    trajectory_rtol=0.002,
)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def episode_split(episodes, tasks, train_ratio=0.95, seed=10222):
    """Keep the original task-stratified episode selection self-contained."""
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between zero and one")
    by_task, mapping = defaultdict(list), {}
    for ep, task in zip(episodes, tasks, strict=True):
        ep, task = int(ep), int(task)
        if ep in mapping and mapping[ep] != task:
            raise ValueError("episode belongs to multiple tasks")
        mapping[ep] = task
    for ep, task in mapping.items():
        by_task[task].append(ep)
    train, validation = [], []
    for task in sorted(by_task):
        values = np.array(sorted(by_task[task]))
        if len(values) < 2:
            raise ValueError("each task needs at least two episodes")
        rng = np.random.default_rng(np.random.SeedSequence([seed, task]))
        values = rng.permutation(values).tolist()
        count = min(len(values) - 1, max(1, int(len(values) * train_ratio)))
        train.extend(values[:count])
        validation.extend(values[count:])
    return dict(train=sorted(train), validation=sorted(validation), seed=seed, unit="episode", stratification="task")


def map_state(values):
    """Numerical probe: XYZ, rotation-vector -> first two matrix rows, width.

    This preserves the pretrained right-arm slots. It does not establish the
    calibration or controller mapping required to execute LIBERO actions.
    """
    from scipy.spatial.transform import Rotation

    values = np.asarray(values, np.float32)
    if values.shape != (8,) or not np.isfinite(values).all():
        raise ValueError("requires finite LIBERO 8-D state")
    state = np.zeros(26, np.float32)
    state[10:13] = values[:3]
    state[13:19] = Rotation.from_rotvec(values[3:6]).as_matrix()[:2].reshape(6)
    state[19] = values[6] - values[7]
    mask = np.zeros(26, np.int32)
    mask[10:20] = 1
    return state, mask


def select_frames(episodes, allowed):
    selected = []
    for task in sorted({row["task_index"] for row in episodes}):
        eligible = [r for r in episodes if r["task_index"] == task and r["episode_index"] in allowed]
        if not eligible:
            raise ValueError(f"no held-out episode for task {task}")
        episode = min(eligible, key=lambda row: row["episode_index"])
        for fraction in (0.25, 0.5, 0.75):
            selected.append(
                dict(
                    task_index=task,
                    episode_index=episode["episode_index"],
                    frame_index=int((episode["length"] - 1) * fraction),
                )
            )
    return selected


def numeric_metrics(actual, expected, mask):
    a, b = (
        np.asarray(actual, np.float64)[..., np.asarray(mask, bool)],
        np.asarray(expected, np.float64)[..., np.asarray(mask, bool)],
    )
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid numeric comparison")
    error = a - b
    rmse = np.sqrt(np.mean(error**2))
    return dict(
        mae=float(np.mean(np.abs(error))),
        max_abs=float(np.max(np.abs(error))),
        rmse=float(rmse),
        nrmse=float(rmse / max(1e-12, np.sqrt(np.mean(b**2)))),
        cosine=float(np.dot(a.ravel(), b.ravel()) / max(1e-12, np.linalg.norm(a) * np.linalg.norm(b))),
    )


def decode_frame(path, timestamp):
    import av

    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.codec_context.thread_count = 2
        container.seek(max(0, int(timestamp / stream.time_base)), stream=stream, backward=True)
        for frame in container.decode(stream):
            if frame.time is not None and frame.time >= timestamp - 0.05:
                if abs(frame.time - timestamp) > 0.051:
                    raise ValueError(f"video frame timestamp mismatch: {frame.time} vs {timestamp}")
                return frame.to_image().convert("RGB"), float(frame.time)
    raise ValueError(f"frame unavailable: {path} at {timestamp}")


def validate_visual(actual, expected, gates):
    result = numeric_metrics(actual, expected, np.ones(actual.shape[-1], bool))
    if (
        result["nrmse"] > gates["visual_nrmse_max"]
        or result["max_abs"] > gates["visual_max_abs"]
        or result["cosine"] < gates["visual_cosine_min"]
    ):
        raise ValueError(f"visual relative FP32 gate failed: {result}")
    return result


def identity(root):
    root = Path(root)
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


def prepare(args):
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq

    run, data = args.run_dir.resolve(), args.data_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    (run / "images").mkdir()
    (run / "samples").mkdir()
    table = ds.dataset([str(p) for p in sorted((data / "data").glob("*/*.parquet"))]).to_table(
        columns=["episode_index", "task_index", "frame_index", "index", "timestamp", "observation.state"]
    )
    ep, task = table["episode_index"].to_numpy(), table["task_index"].to_numpy()
    split = episode_split(ep, task)
    metadata = pq.read_table(sorted((data / "meta/episodes").glob("*/*.parquet"))).to_pylist()
    mapping = {int(e): int(t) for e, t in zip(ep, task, strict=True)}
    for row in metadata:
        row["task_index"] = mapping[row["episode_index"]]
    episode_info = {r["episode_index"]: r for r in metadata}
    selection = select_frames(metadata, set(split["validation"]))
    if len(selection) != 120 or len({r["task_index"] for r in selection}) != 40:
        raise ValueError("requires all 40 LIBERO tasks and 120 numeric probes")
    tasks = {r["task_index"]: r["__index_level_0__"] for r in pq.read_table(data / "meta/tasks.parquet").to_pylist()}
    source_videos, samples = {}, []
    frame_indices = table["frame_index"].to_numpy()
    for number, sample in enumerate(selection):
        row_index = np.flatnonzero((ep == sample["episode_index"]) & (frame_indices == sample["frame_index"]))
        if len(row_index) != 1:
            raise ValueError("sample must identify exactly one observation")
        row = table.slice(int(row_index[0]), 1).to_pylist()[0]
        state, mask = map_state(row["observation.state"])
        images, image_sources = [], []
        for camera in ("observation.images.image", "observation.images.image2"):
            meta = episode_info[sample["episode_index"]]
            stem = f"videos/{camera}"
            video = (
                data / stem / f"chunk-{meta[stem + '/chunk_index']:03d}" / f"file-{meta[stem + '/file_index']:03d}.mp4"
            )
            timestamp = meta[stem + "/from_timestamp"] + row["timestamp"]
            image, decoded = decode_frame(video, timestamp)
            if image.size != (256, 256):
                raise ValueError("pinned LIBERO source images must be 256x256")
            output = run / "images" / f"{number:03d}_{camera.rsplit('.', 1)[1]}.png"
            image.save(output)
            images.append(str(output))
            source_videos.setdefault(str(video.relative_to(data)), sha(video))
            image_sources.append(
                dict(
                    camera=camera,
                    video=str(video.relative_to(data)),
                    timestamp=timestamp,
                    decoded=decoded,
                    png_sha256=sha(output),
                )
            )
        values = dict(
            id=number,
            **sample,
            dataset_index=row["index"],
            instruction=tasks[sample["task_index"]],
            images=images,
            state=state.tolist(),
            source_state=row["observation.state"],
            state_mask=mask.tolist(),
            dof_mask=mask.tolist(),
            norm_key="x2_normal",
            seed=10222 + number,
            steps=10,
            sources=image_sources,
        )
        write_json(run / "samples" / f"{number:03d}.json", values)
        samples.append(values)
        print(f"prepared {number + 1}/120", flush=True)
    manifest = dict(
        schema_version=1,
        dataset_id="lerobot/libero",
        dataset_revision=DATASET_REVISION,
        official_revision=UPSTREAM_REVISION,
        official_source=str(args.official_source.resolve()),
        data_dir=str(data),
        model_dir=str(args.model_dir.resolve()),
        base_weights_sha256=sha(args.model_dir / "model.safetensors"),
        artifact_dir=str(args.artifact_dir.resolve()),
        artifact_files=identity(args.artifact_dir),
        info_sha256=sha(data / "meta/info.json"),
        source_videos=source_videos,
        samples=samples,
        protocol=dict(
            purpose="numeric consistency only; no demonstration-label or task-success score",
            sample_count=120,
            tasks=40,
            episode_split=split,
            horizon=32,
            dimensions=26,
            long_edge=448,
            steps=10,
            norm_key="x2_normal",
            active_dimensions=list(range(10, 20)),
            state_mapping="xyz -> 10:13; rotvec -> first two rotation-matrix rows -> 13:19; qpos[0]-qpos[1] -> 19",
            camera_order=["image", "image2"],
            image_rotation="use stored dataset orientation; no extra flip",
            float_gate=FLOAT_GATES,
            quant_gate=dict(
                global_cosine_min=0.995, global_nrmse_max=0.1, global_rmse_max=0.05, sample_cosine_min=0.98
            ),
        ),
    )
    manifest["protocol_hash"] = json_hash(manifest)
    write_json(run / "manifest.json", manifest)


def input_hash(data):
    import torch

    h = hashlib.sha256()
    for tensor in [data["ids"], data["grid"], *data["patches"], *data["rope"]]:
        tensor = tensor.contiguous().cpu()
        h.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
        h.update(tensor.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def worker(args):
    import importlib.metadata

    import torch
    from PIL import Image

    from xhmodel_merak.xh_other_model.models.wall_oss.processor import WallProcessor

    run = args.run_dir.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    if json_hash({k: v for k, v in manifest.items() if k != "protocol_hash"}) != manifest["protocol_hash"]:
        raise ValueError("run manifest changed after protocol registration")
    dependencies = {
        name: importlib.metadata.version(name) for name in ("torch", "transformers", "qwen-vl-utils", "numpy")
    }
    code = {
        str(p): sha(p)
        for p in [Path(__file__), Path(__file__).with_name("float_policy.py"), Path(__file__).with_name("validate.py")]
    }
    model_package = Path(__file__).parents[3] / "xhmodel_merak/xh_other_model/models/wall_oss"
    code.update({str(p): sha(p) for p in sorted(model_package.glob("*.py"))})
    from xhquant.lib import _hsum_impl

    backend = dict(module=_hsum_impl.__name__, file=_hsum_impl.__file__, sha256=sha(_hsum_impl.__file__))
    code_hash = json_hash(dict(code=code, dependencies=dependencies, backend=backend))
    resource = Path(manifest["artifact_dir"])
    processor = WallProcessor(resource)
    selected = [s for s in manifest["samples"] if args.ids is None or s["id"] in args.ids]
    out = run / args.stage
    out.mkdir(exist_ok=True)
    if args.stage == "official":
        from .validate import load_official_model

        model, native = load_official_model(
            SimpleNamespace(
                model_dir=manifest["model_dir"],
                official_source=manifest["official_source"],
                device=args.device,
                dtype="float32",
            )
        )
    elif args.stage == "float":
        from .float_policy import FloatPolicy

        model = FloatPolicy(manifest["model_dir"], resource, args.device)
    else:
        from xhmodel_merak.xh_other_model.models.wall_oss.runtime import WallRuntime

        model = WallRuntime(resource, device=args.device)
        if model.meta["quant_type"] != "w8a8_sefp":
            raise ValueError("this run requires the existing W8A8 artifact")
    started, count = time.monotonic(), 0
    for values in selected:
        path = out / f"{values['id']:03d}.pt"
        if path.exists():
            old = torch.load(path, weights_only=True)
            if old.get("protocol_hash") == manifest["protocol_hash"] and old.get("code_hash") == code_hash:
                count += 1
                continue
        images = [Image.open(p).convert("RGB") for p in values["images"]]
        data = processor.prepare(
            values["instruction"], images, values["state"], values["norm_key"], values["state_mask"]
        )
        fingerprint = input_hash(data)
        torch.cuda.synchronize()
        tick = time.monotonic()
        if args.stage == "official":
            result = official_predict(model, native, processor, data, values, args.device)
        else:
            expected = torch.load(run / "official" / path.name, weights_only=True)
            if expected["input_hash"] != fingerprint:
                raise ValueError("effective processor tensors differ from official reference environment")
            if args.stage == "float":
                result = model.predict(data, expected["noise"], values)
                gates = manifest["protocol"]["float_gate"]
                result["visual_elementwise_close"] = True
                try:
                    torch.testing.assert_close(
                        result["vision"],
                        expected["vision_per_image"],
                        atol=gates["visual_atol"],
                        rtol=gates["visual_rtol"],
                    )
                except AssertionError:
                    result["visual_elementwise_close"] = False
                    from xhmodel_merak.xh_other_model.models.wall_oss.source import load_visual

                    native_visual = load_visual(manifest["model_dir"]).to(args.device)
                    with torch.inference_mode():
                        native_features = torch.cat(
                            [
                                native_visual(
                                    p[0, : int(g.prod())].to(args.device), g[None].to(args.device)
                                ).pooler_output
                                for p, g in zip(data["patches"], data["grid"], strict=True)
                            ]
                        ).cpu()
                    result["native_vision"] = native_features
                    torch.save(result, out / f"{values['id']:03d}.diagnostic.pt")
                    visual_mask = np.ones(result["vision"].shape[-1], bool)
                    write_json(
                        out / f"{values['id']:03d}.diagnostic.json",
                        dict(
                            packed_vs_official=numeric_metrics(result["vision"], expected["vision"], visual_mask),
                            native_vs_official=numeric_metrics(native_features, expected["vision"], visual_mask),
                            packed_vs_native=numeric_metrics(result["vision"], native_features, visual_mask),
                            native_image_batching=numeric_metrics(
                                expected["vision_per_image"], expected["vision"], visual_mask
                            ),
                            packed_vs_official_per_image=numeric_metrics(
                                result["vision"], expected["vision_per_image"], visual_mask
                            ),
                            trajectory=numeric_metrics(
                                result["trajectory"], expected["trajectory"], np.asarray(values["dof_mask"], bool)
                            ),
                        ),
                    )
                result["visual_metrics"] = validate_visual(result["vision"], expected["vision_per_image"], gates)
                torch.testing.assert_close(
                    result["trajectory"],
                    expected["trajectory"],
                    atol=gates["trajectory_atol"],
                    rtol=gates["trajectory_rtol"],
                )
            else:
                # A case is admitted only after its native-to-target float gate.
                reference = torch.load(run / "float" / path.name, weights_only=True)
                if reference["protocol_hash"] != manifest["protocol_hash"] or not reference["float_gate_passed"]:
                    raise ValueError("target float gate missing")
                call = {
                    k: values[k]
                    for k in ("instruction", "state", "norm_key", "state_mask", "dof_mask", "seed", "steps")
                }
                actions, stats = model.predict(**call, images=images, noise=expected["noise"])
                torch.testing.assert_close(model.last_trace[0], expected["noise"], rtol=0, atol=0)
                if stats["image_grids"] != data["grid"].tolist() or stats["prefix_tokens"] != data["ids"].shape[1]:
                    raise ValueError("HMONNX effective input geometry differs from the validated float policy")
                result = dict(
                    trajectory=torch.stack(model.last_trace), actions=torch.from_numpy(actions)[None], stats=stats
                )
                if values["id"] in (0, 60, 119):
                    repeated, _ = model.predict(**call, images=images, noise=expected["noise"])
                    np.testing.assert_array_equal(actions, repeated)
                    result["repeat_exact"] = True
        torch.cuda.synchronize()
        result.update(
            protocol_hash=manifest["protocol_hash"],
            code_hash=code_hash,
            dependencies=dependencies,
            backend=backend,
            input_hash=fingerprint,
            seconds=time.monotonic() - tick,
            float_gate_passed=args.stage == "float",
        )
        if not torch.isfinite(result["trajectory"]).all():
            raise ValueError("nonfinite action trajectory")
        temporary = path.with_suffix(".tmp")
        torch.save(result, temporary)
        temporary.replace(path)
        count += 1
        write_json(
            run / f"{args.stage}_status.json",
            dict(stage=args.stage, complete=count, selected=len(selected), elapsed_seconds=time.monotonic() - started),
        )
        print(
            f"{args.stage}: {count}/{len(selected)}, sample={values['id']}, seconds={result['seconds']:.3f}", flush=True
        )
    if args.stage == "w8" and (out / "000.pt").exists():
        first = manifest["samples"][0]
        saved = torch.load(out / "000.pt", weights_only=True)
        oracle = torch.load(run / "official/000.pt", weights_only=True)
        call = {k: first[k] for k in ("instruction", "state", "norm_key", "state_mask", "dof_mask", "seed", "steps")}
        repeated, _ = model.predict(
            **call, images=[Image.open(p).convert("RGB") for p in first["images"]], noise=oracle["noise"]
        )
        np.testing.assert_array_equal(repeated, saved["actions"][0].numpy())
        write_json(
            run / "repeat_check.json", dict(exact=True, protocol_hash=manifest["protocol_hash"], selected=len(selected))
        )


def official_predict(model, native, processor, data, values, device, capture_visual_control=True):
    import torch

    # Independently encode the same rendered prompt with the native processor.
    from PIL import Image

    from xhmodel_merak.xh_other_model.models.wall_oss.processor import resize_image

    encoded = native(
        text=[data["prompt"]],
        images=[resize_image(Image.open(p)) for p in values["images"]],
        return_tensors="pt",
        do_resize=False,
    )
    torch.testing.assert_close(encoded["input_ids"], data["ids"], rtol=0, atol=0)
    torch.testing.assert_close(encoded["image_grid_thw"], data["grid"], rtol=0, atol=0)
    pixels = torch.cat([x[0, : int(g.prod())] for x, g in zip(data["patches"], data["grid"], strict=True)])
    torch.testing.assert_close(encoded["pixel_values"], pixels, rtol=0, atol=0)
    ids = torch.cat((data["ids"], torch.full((1, 32), native.tokenizer.convert_tokens_to_ids("<|action|>"))), 1).to(
        device
    )
    grid = data["grid"].to(device)
    position, _ = model.get_rope_index(ids, image_grid_thw=grid)
    from xhmodel_merak.xh_other_model.models.wall_oss.processor import language_rope

    for actual, expected in zip(language_rope(position.cpu()), data["rope"], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    captured = {}
    original = model._finalize_flow_action_output

    def finish(trajectory, context):
        captured.update(
            trajectory=torch.cat((context["noise"][None].cpu(), trajectory.cpu()), 0),
            noise=context["noise"].cpu(),
            embeddings=context["inputs_embeds"][:, : data["ids"].shape[1]].float().cpu(),
        )
        return original(trajectory, context)

    hook = model.visual.register_forward_hook(lambda _, __, output: captured.update(vision=output.float().cpu()))
    model._finalize_flow_action_output = finish
    torch.cuda.manual_seed(values["seed"])
    try:
        with torch.inference_mode():
            prediction = model.generate_flow_action(
                input_ids=ids,
                action_horizon=32,
                action_dim=26,
                num_inference_timesteps=values["steps"],
                attention_mask=torch.ones_like(ids),
                position_ids=position,
                moe_token_types=ids == native.tokenizer.convert_tokens_to_ids("<|action|>"),
                pixel_values=pixels.to(device),
                image_grid_thw=grid,
                dataset_names=[values["norm_key"]],
                dof_mask=torch.tensor(values["dof_mask"], device=device, dtype=torch.float32)[None, None].expand(
                    1, 32, 26
                ),
            )
        captured["actions"] = prediction["predict_action"].float().cpu()
    finally:
        hook.remove()
        model._finalize_flow_action_output = original
    # The exported visual graph processes one image per call. Retain a native
    # control with that same batching, as well as the unmodified two-image
    # native policy above: FP32 GEMM/SDPA accumulation changes with batch shape.
    if not capture_visual_control:
        return captured
    with torch.inference_mode():
        captured["vision_per_image"] = torch.cat(
            [
                model.visual(
                    p[0, : int(g.prod())].to(device=device, dtype=model.visual.dtype), grid_thw=g[None].to(device)
                )
                .float()
                .cpu()
                for p, g in zip(data["patches"], data["grid"], strict=True)
            ]
        )
    return captured


def summarize(args):
    import torch

    run = args.run_dir.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    selected = [s for s in manifest["samples"] if args.ids is None or s["id"] in args.ids]
    records, actuals, expected, official, latencies, cohorts = [], [], [], [], {}, {}
    for sample in selected:
        results = {
            stage: torch.load(run / stage / f"{sample['id']:03d}.pt", weights_only=True)
            for stage in ("official", "float", "w8")
        }
        if any(x["protocol_hash"] != manifest["protocol_hash"] for x in results.values()):
            raise ValueError("cannot combine results from different protocols")
        for stage, result in results.items():
            cohorts.setdefault(stage, set()).add(result["code_hash"])
            latencies.setdefault(stage, []).append(result["seconds"])
        a, b, c = (results[stage]["trajectory"].numpy() for stage in ("w8", "float", "official"))
        mask = np.asarray(sample["dof_mask"], bool)
        metrics = numeric_metrics(a[-1], b[-1], mask)
        visual_mask = np.ones(results["float"]["vision"].shape[-1], bool)
        records.append(
            dict(
                id=sample["id"],
                task=sample["task_index"],
                episode=sample["episode_index"],
                frame=sample["frame_index"],
                final=metrics,
                trajectory=numeric_metrics(a[1:], b[1:], mask),
                float_visual=numeric_metrics(
                    results["float"]["vision"], results["official"]["vision_per_image"], visual_mask
                ),
                native_image_batching=numeric_metrics(
                    results["official"]["vision_per_image"], results["official"]["vision"], visual_mask
                ),
                visual_elementwise_close=results["float"]["visual_elementwise_close"],
            )
        )
        actuals.append(a)
        expected.append(b)
        official.append(c)
    if any(len(hashes) != 1 for hashes in cohorts.values()):
        raise ValueError("mixed code/dependency cohorts; rerun the affected stage to refresh cached cases")
    a, b, c = map(np.stack, (actuals, expected, official))
    mask = np.zeros(26, bool)
    mask[10:20] = True
    final = numeric_metrics(a[:, -1], b[:, -1], mask)
    gates = manifest["protocol"]["quant_gate"]
    worst_cosine = min(r["final"]["cosine"] for r in records)
    passed = (
        final["cosine"] >= gates["global_cosine_min"]
        and final["nrmse"] <= gates["global_nrmse_max"]
        and final["rmse"] <= gates["global_rmse_max"]
        and worst_cosine >= gates["sample_cosine_min"]
    )
    report = dict(
        complete=True,
        passed=passed,
        samples=len(selected),
        tasks=len({s["task_index"] for s in selected}),
        protocol_hash=manifest["protocol_hash"],
        scope=manifest["protocol"]["purpose"],
        float_vs_official=numeric_metrics(b[:, 1:], c[:, 1:], mask),
        w8_vs_float_final=final,
        w8_vs_float_trajectory=numeric_metrics(a[:, 1:], b[:, 1:], mask),
        w8_vs_official_final=numeric_metrics(a[:, -1], c[:, -1], mask),
        worst_sample_cosine=worst_cosine,
        gates=gates,
        latency_seconds={
            stage: dict(mean=float(np.mean(v)), median=float(np.median(v))) for stage, v in latencies.items()
        },
        per_dimension_mae=np.abs(a[:, -1] - b[:, -1]).mean(axis=(0, 1, 2)).tolist(),
        sample_rmse_p95=float(np.percentile([r["final"]["rmse"] for r in records], 95)),
        sample_rmse_max=float(max(r["final"]["rmse"] for r in records)),
        visual_elementwise_diagnostic_failures=sum(not r["visual_elementwise_close"] for r in records),
        groups={
            name: numeric_metrics(a[:, -1], b[:, -1], np.isin(np.arange(26), dimensions))
            for name, dimensions in dict(position=range(10, 13), rotation_6d=range(13, 19), gripper=[19]).items()
        },
        visual_control=(
            "FP32 visual uses official per-image execution; actions use the unmodified joint-image native policy"
        ),
        cohorts={stage: sorted(v) for stage, v in cohorts.items()},
        cases=records,
    )
    name = "report.json" if args.ids is None else "pilot_report.json"
    write_json(run / name, report)
    if args.ids is None:
        np.savez_compressed(
            run / "predictions.npz",
            w8=a,
            target_float=b,
            official=c,
            active_mask=mask,
            sample_ids=[s["id"] for s in selected],
        )
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2), flush=True)
    if not passed:
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "official", "float", "w8", "summarize"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--official-source", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ids", type=lambda value: set(map(int, value.split(","))))
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args)
    else:
        import torch

        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.stage == "summarize":
            summarize(args)
        else:
            worker(args)


if __name__ == "__main__":
    main()
