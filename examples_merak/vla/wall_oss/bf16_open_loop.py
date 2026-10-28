"""Expanded, resumable official BF16 versus existing W8A8 LIBERO evaluation."""

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .evaluation_queue import EvaluationQueue
from .open_loop import decode_frame, identity, input_hash, json_hash, map_state, numeric_metrics, sha, write_json


def expanded_plan(episodes, old):
    """Keep the original 120 pairs, then cover 8 episodes/task and 2 seeds."""
    plan = [dict(row) for row in old]
    for row in old:
        plan.append({**row, "id": len(plan), "seed": row["seed"] + 1000000})
    old_keys = {(r["episode_index"], r["frame_index"]) for r in old}
    for task in sorted({r["task_index"] for r in episodes}):
        anchor = next(r["episode_index"] for r in old if r["task_index"] == task)
        others = sorted(
            [r for r in episodes if r["task_index"] == task and r["episode_index"] != anchor],
            key=lambda r: r["episode_index"],
        )
        if len(others) < 7:
            raise ValueError("requires at least 8 episodes per task")
        chosen = [next(r for r in episodes if r["episode_index"] == anchor)]
        chosen += [others[int(i)] for i in np.linspace(0, len(others) - 1, 7, dtype=int)]
        for episode in chosen:
            for fraction in (0.25, 0.5, 0.75):
                frame = int((episode["length"] - 1) * fraction)
                if (episode["episode_index"], frame) in old_keys:
                    continue
                for _ in range(2):
                    index = len(plan)
                    plan.append(
                        dict(
                            id=index,
                            task_index=task,
                            episode_index=episode["episode_index"],
                            frame_index=frame,
                            seed=20222 + index,
                        )
                    )
    return plan


def array_sha(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str((array.shape, str(array.dtype))).encode() + array.tobytes()).hexdigest()


def write_record(path, trajectory, actions, metadata):
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, trajectory=trajectory, actions=actions, metadata=json.dumps(metadata))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_record(path, pair, contract):
    with np.load(path, allow_pickle=False) as archive:
        trajectory, actions = archive["trajectory"], archive["actions"]
        metadata = json.loads(str(archive["metadata"].item()))
    if metadata["request_hash"] != pair["request_hash"] or metadata["contract"] != contract:
        raise ValueError("cached prediction identity mismatch")
    if (
        trajectory.shape != (11, 1, 32, 26)
        or actions.shape != (1, 32, 26)
        or not np.isfinite(trajectory).all()
        or not np.isfinite(actions).all()
    ):
        raise ValueError("invalid cached prediction tensors")
    if array_sha(trajectory[0]) != pair["noise_sha256"]:
        raise ValueError("cached prediction used a different noise tensor")
    return trajectory, actions, metadata


def prepare(args):
    import pyarrow.dataset as ds
    import pyarrow.parquet as pq
    import torch

    from xhmodel_merak.xh_other_model.models.wall_oss.processor import WallProcessor

    old_root = args.previous.resolve()
    old = json.loads((old_root / "manifest.json").read_text())
    if old["dataset_revision"] != "a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4":
        raise ValueError("unexpected dataset revision")
    if (
        identity(old["artifact_dir"]) != old["artifact_files"]
        or sha(Path(old["model_dir"]) / "model.safetensors") != old["base_weights_sha256"]
    ):
        raise ValueError("model artifacts changed; old predictions cannot be reused")
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=False)
    for directory in ("images", "noise", "bf16", "w8", "logs", "engines", "benchmarks"):
        (run / directory).mkdir()
    data = Path(old["data_dir"])
    table = ds.dataset([str(p) for p in sorted((data / "data").glob("*/*.parquet"))]).to_table(
        columns=["episode_index", "task_index", "frame_index", "index", "timestamp", "observation.state"]
    )
    ep = table["episode_index"].to_numpy()
    task = table["task_index"].to_numpy()
    frames = table["frame_index"].to_numpy()
    metadata = pq.read_table(sorted((data / "meta/episodes").glob("*/*.parquet"))).to_pylist()
    tasks = {r["task_index"]: r["__index_level_0__"] for r in pq.read_table(data / "meta/tasks.parquet").to_pylist()}
    mapping = {int(e): int(t) for e, t in zip(ep, task, strict=True)}
    for row in metadata:
        row["task_index"] = mapping[row["episode_index"]]
    meta_by_episode = {r["episode_index"]: r for r in metadata}
    plan = expanded_plan(metadata, old["samples"])
    if len(plan) != 1920:
        raise ValueError("expected 1920 paired samples")
    processor = WallProcessor(old["artifact_dir"])
    old_by_key = {(s["episode_index"], s["frame_index"]): s for s in old["samples"]}
    observations = []
    by_key = {}
    source_videos = {}
    for item in plan:
        key = (item["episode_index"], item["frame_index"])
        if key in by_key:
            continue
        number = len(observations)
        if key in old_by_key:
            value = dict(old_by_key[key])
            value["observation_id"] = number
            for image, source in zip(value["images"], value["sources"], strict=True):
                if sha(image) != source["png_sha256"]:
                    raise ValueError("legacy PNG changed")
        else:
            index = np.flatnonzero((ep == key[0]) & (frames == key[1]))
            if len(index) != 1:
                raise ValueError("ambiguous observation")
            row = table.slice(int(index[0]), 1).to_pylist()[0]
            state, mask = map_state(row["observation.state"])
            value = dict(
                observation_id=number,
                task_index=item["task_index"],
                episode_index=key[0],
                frame_index=key[1],
                dataset_index=row["index"],
                instruction=tasks[item["task_index"]],
                state=state.tolist(),
                source_state=row["observation.state"],
                state_mask=mask.tolist(),
                dof_mask=mask.tolist(),
                norm_key="x2_normal",
                steps=10,
                images=[],
                sources=[],
            )
            meta = meta_by_episode[key[0]]
            for camera in ("observation.images.image", "observation.images.image2"):
                stem = f"videos/{camera}"
                video = (
                    data
                    / stem
                    / f"chunk-{meta[stem + '/chunk_index']:03d}"
                    / f"file-{meta[stem + '/file_index']:03d}.mp4"
                )
                timestamp = meta[stem + "/from_timestamp"] + row["timestamp"]
                image, decoded = decode_frame(video, timestamp)
                path = run / "images" / f"{number:04d}_{camera.rsplit('.', 1)[1]}.png"
                image.save(path)
                value["images"].append(str(path))
                value["sources"].append(
                    dict(
                        camera=camera,
                        video=str(video.relative_to(data)),
                        timestamp=timestamp,
                        decoded=decoded,
                        png_sha256=sha(path),
                    )
                )
                if str(video.relative_to(data)) not in source_videos:
                    source_videos[str(video.relative_to(data))] = sha(video)
        from PIL import Image

        prepared = processor.prepare(
            value["instruction"],
            [Image.open(p).convert("RGB") for p in value["images"]],
            value["state"],
            value["norm_key"],
            value["state_mask"],
        )
        value["input_hash"] = input_hash(prepared)
        observations.append(value)
        by_key[key] = number
        if number % 50 == 0:
            print(f"observations {number + 1}/960", flush=True)
    runtime_code = {
        name: sha(Path(__file__).parents[3] / "xhmodel_merak/xh_other_model/models/wall_oss" / name)
        for name in ("runtime.py", "processor.py")
    }
    common = dict(
        dataset_revision=old["dataset_revision"],
        checkpoint=old["base_weights_sha256"],
        artifacts=json_hash(old["artifact_files"]),
        runtime_code=runtime_code,
        protocol="pretrained H32 448 x2_normal right10; exact fixed FP32 noise; Euler10",
    )
    contracts = {
        "w8": json_hash({**common, "backend": "existing W8A8 SEFP HMONNX"}),
        "bf16": json_hash(
            {**common, "backend": "official Wall-X selected BF16/FP32 SDPA", "revision": old["official_revision"]}
        ),
    }
    pairs = []
    for item in plan:
        index = item["id"]
        obs = observations[by_key[(item["episode_index"], item["frame_index"])]]
        if index < 120:
            reference = torch.load(old_root / "official" / f"{index:03d}.pt", weights_only=True)
            noise = reference["noise"].numpy()
        else:
            noise = torch.randn((1, 32, 26), generator=torch.Generator().manual_seed(item["seed"])).numpy()
        noise_path = run / "noise" / f"{index:04d}.npy"
        np.save(noise_path, noise)
        request = dict(
            input_hash=obs["input_hash"],
            noise=array_sha(noise),
            steps=10,
            mask=obs["dof_mask"],
            state_mask=obs["state_mask"],
            norm_key=obs["norm_key"],
        )
        pair = dict(
            id=index,
            observation_id=obs["observation_id"],
            seed=item["seed"],
            noise=str(noise_path),
            noise_sha256=array_sha(noise),
            request_hash=json_hash(request),
        )
        pairs.append(pair)
        if index < 120:
            old_path = old_root / "w8" / f"{index:03d}.pt"
            saved = torch.load(old_path, weights_only=True)
            if saved["protocol_hash"] != old["protocol_hash"] or saved["input_hash"] != obs["input_hash"]:
                raise ValueError("legacy W8 input identity differs")
            torch.testing.assert_close(saved["trajectory"][0], torch.from_numpy(noise), rtol=0, atol=0)
            meta = dict(
                request_hash=pair["request_hash"],
                contract=contracts["w8"],
                origin="reused",
                source_file=str(old_path),
                source_sha256=sha(old_path),
                producer_commit="6e26479bce0336748aab286175d8495faa5c8df0",
                seconds=saved["seconds"],
            )
            write_record(run / "w8" / f"{index:04d}.npz", saved["trajectory"].numpy(), saved["actions"].numpy(), meta)
    manifest = dict(
        schema_version=1,
        previous=str(old_root),
        dataset_revision=old["dataset_revision"],
        official_revision=old["official_revision"],
        official_source=old["official_source"],
        model_dir=old["model_dir"],
        base_weights_sha256=old["base_weights_sha256"],
        artifact_dir=old["artifact_dir"],
        artifact_files=old["artifact_files"],
        source_videos=source_videos,
        contracts=contracts,
        observations=observations,
        pairs=pairs,
        scope=dict(
            tasks=40,
            episodes_per_task=8,
            frames_per_episode=3,
            seeds_per_frame=2,
            observations=960,
            pairs=1920,
            reused_w8=120,
            new_w8=1800,
            new_bf16=1920,
            episode_selection=(
                "old anchor plus seven evenly spaced episodes across each task; "
                "numeric probes, not task-score test split"
            ),
        ),
    )
    manifest["manifest_hash"] = json_hash(manifest)
    write_json(run / "manifest.json", manifest)
    queue = EvaluationQueue(run / "queue.sqlite3")
    queue.initialize(len(pairs))
    for i in range(120):
        read_record(run / "w8" / f"{i:04d}.npz", pairs[i], contracts["w8"])
        queue.mark_reused("w8", i)
    print(json.dumps(manifest["scope"], indent=2), flush=True)


def predict_native(model, native, processor, data, values, noise):
    """Supply exact external noise through the native context preparation hook."""
    import torch

    from .open_loop import official_predict

    original = model._prepare_flow_action_inputs

    def prepare_context(**kwargs):
        context = original(**kwargs)
        fixed = noise.to(context["inputs_embeds"].device, dtype=torch.float32).clone()
        context["noise"], context["noisy_action"] = fixed, fixed.clone()
        first = context["times"][0].unsqueeze(0).repeat(fixed.shape[0])
        embedding, condition = model.action_preprocessor.step(
            timestep=first, noisy_action=fixed, dof_mask=context["dof_mask"]
        )
        context["inputs_embeds"][context["flow_action_mask"]] = embedding.reshape(
            -1, context["inputs_embeds"].shape[-1]
        ).to(context["inputs_embeds"].dtype)
        context["adarms_cond"] = condition
        return context

    model._prepare_flow_action_inputs = prepare_context
    try:
        return official_predict(model, native, processor, data, values, "cuda:0", capture_visual_control=False)
    finally:
        model._prepare_flow_action_inputs = original


def worker(args):
    import torch
    from PIL import Image

    from xhmodel_merak.xh_other_model.models.wall_oss.processor import WallProcessor

    run = args.run_dir.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    if json_hash({k: v for k, v in manifest.items() if k != "manifest_hash"}) != manifest["manifest_hash"]:
        raise ValueError("manifest changed")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    processor = WallProcessor(manifest["artifact_dir"])
    if args.backend == "bf16":
        from .validate import load_official_model

        model, native = load_official_model(
            SimpleNamespace(
                model_dir=manifest["model_dir"],
                official_source=manifest["official_source"],
                device="cuda:0",
                dtype="bfloat16",
            )
        )
        counts = {}
        for parameter in model.parameters():
            counts[str(parameter.dtype)] = counts.get(str(parameter.dtype), 0) + parameter.numel()
        if (
            model.model.embed_tokens.weight.dtype != torch.bfloat16
            or model.action_preprocessor.w1.weight.dtype != torch.float32
        ):
            raise ValueError("official mixed precision policy not active")
        write_json(
            run / "engines" / f"{args.owner}_dtype.json",
            dict(
                parameter_elements=counts,
                embedding=str(model.model.embed_tokens.weight.dtype),
                visual=str(model.visual.dtype),
                action_projection=str(model.action_preprocessor.w1.weight.dtype),
            ),
        )
    else:
        from xhmodel_merak.xh_other_model.models.wall_oss.runtime import WallRuntime

        model = WallRuntime(manifest["artifact_dir"], device="cuda:0")
        if model.meta["quant_type"] != "w8a8_sefp":
            raise ValueError("expected existing W8A8")
    queue = EvaluationQueue(run / "queue.sqlite3")
    benchmarks = []
    started = None
    if args.bench:
        bench = run / "benchmarks" / args.bench
        bench.mkdir(exist_ok=True)
        write_json(bench / f"{args.owner}_ready.json", dict(pid=os.getpid()))
        deadline = time.time() + 300
        while not (bench / "start.json").exists():
            if time.time() > deadline:
                raise TimeoutError("benchmark start barrier")
            time.sleep(0.1)
        identifiers = list(map(int, args.ids.split(",")))
        iterator = iter(identifiers)
        started = time.monotonic()
    while True:
        index = next(iterator, None) if args.bench else queue.claim(args.backend, args.owner)
        if index is None:
            break
        pair = manifest["pairs"][index]
        observation = manifest["observations"][pair["observation_id"]]
        path = run / args.backend / f"{index:04d}.npz"
        try:
            if not args.bench and path.exists():
                read_record(path, pair, manifest["contracts"][args.backend])
                queue.finish(args.backend, index, args.owner)
                continue
            images = [Image.open(p).convert("RGB") for p in observation["images"]]
            for image, source in zip(observation["images"], observation["sources"], strict=True):
                if sha(image) != source["png_sha256"]:
                    raise ValueError("input PNG changed")
            data = processor.prepare(
                observation["instruction"],
                images,
                observation["state"],
                observation["norm_key"],
                observation["state_mask"],
            )
            if input_hash(data) != observation["input_hash"]:
                raise ValueError("effective inputs differ")
            noise = np.load(pair["noise"], allow_pickle=False)
            if array_sha(noise) != pair["noise_sha256"]:
                raise ValueError("fixed noise changed")
            values = {**observation, "seed": pair["seed"]}
            torch.cuda.synchronize()
            tick = time.monotonic()
            if args.backend == "bf16":
                output = predict_native(model, native, processor, data, values, torch.from_numpy(noise))
                trajectory = output["trajectory"].numpy()
                actions = output["actions"].numpy()
            else:
                call = {
                    k: values[k]
                    for k in ("instruction", "state", "norm_key", "state_mask", "dof_mask", "seed", "steps")
                }
                action, _ = model.predict(**call, images=images, noise=noise)
                trajectory = torch.stack(model.last_trace).numpy()
                actions = action[None]
            torch.cuda.synchronize()
            seconds = time.monotonic() - tick
            if not np.array_equal(trajectory[0], noise) or not np.isfinite(trajectory).all():
                raise ValueError("invalid noise/trajectory")
            record = dict(
                request_hash=pair["request_hash"],
                contract=manifest["contracts"][args.backend],
                origin="new",
                owner=args.owner,
                seconds=seconds,
                engine_code_sha256=sha(__file__),
                parameter_dtype="official_selected_bf16_fp32" if args.backend == "bf16" else "w8a8_sefp",
            )
            if path.exists():
                previous, _, _ = read_record(path, pair, manifest["contracts"][args.backend])
                if args.bench:
                    np.testing.assert_array_equal(trajectory, previous)
            else:
                write_record(path, trajectory, actions, record)
            if args.bench:
                benchmarks.append(dict(id=index, seconds=seconds))
            else:
                queue.finish(args.backend, index, args.owner)
            write_json(
                run / "engines" / f"{args.owner}_heartbeat.json",
                dict(time=time.time(), sample=index, seconds=seconds, pid=os.getpid()),
            )
            print(f"{args.owner} {index} {seconds:.3f}s", flush=True)
        except BaseException as error:
            if not args.bench:
                queue.fail(args.backend, index, args.owner, error)
            raise
    if args.bench:
        write_json(
            bench / f"{args.owner}_result.json",
            dict(
                owner=args.owner,
                backend=args.backend,
                samples=len(benchmarks),
                wall_seconds=time.monotonic() - started,
                cases=benchmarks,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            ),
        )


def spawn(run, backend, gpu, owner, *, bench=None, ids=None):
    project = Path(__file__).resolve().parents[3]
    python = project / "work_dirs" / ("reference_env" if backend == "bf16" else "wall_env") / "bin/python"
    command = [
        str(python),
        "-m",
        "examples_merak.vla.wall_oss.bf16_open_loop",
        "worker",
        "--run-dir",
        str(run),
        "--backend",
        backend,
        "--owner",
        owner,
    ]
    if bench:
        command += ["--bench", bench, "--ids", ",".join(map(str, ids))]
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "PYTHONPATH": str(project),
        "OMP_NUM_THREADS": "2",
        "OPENBLAS_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "TOKENIZERS_PARALLELISM": "false",
    }
    with (run / "logs" / f"{owner}.log").open("ab", buffering=0) as output:
        return subprocess.Popen(
            command,
            cwd=project,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def benchmark(run, gpus):
    identifiers = [0, 87, 119, 240, 360, 480, 600, 720, 840, 960, 1080, 1200, 1320, 1440, 1680, 1919]
    results = {}
    layout = {}
    for backend in ("bf16", "w8"):
        results[backend] = {}
        for slots in (1, 2):
            name = f"{backend}_g{'-'.join(map(str, gpus))}_slots{slots}"
            directory = run / "benchmarks" / name
            directory.mkdir(exist_ok=True)
            if not (directory / "aggregate.json").exists():
                for stale in directory.glob("*.json"):
                    stale.unlink()
                children = {}
                for gpu_index, gpu in enumerate(gpus):
                    for slot in range(slots):
                        owner = f"bench_{backend}_g{gpu}_s{slot}_n{slots}"
                        children[owner] = spawn(
                            run,
                            backend,
                            gpu,
                            owner,
                            bench=name,
                            ids=identifiers[gpu_index * 8 : (gpu_index + 1) * 8][slot::slots],
                        )
                write_json(run / "workers.json", [dict(pid=p.pid, owner=o) for o, p in children.items()])
                deadline = time.time() + 600
                while not all((directory / f"{owner}_ready.json").exists() for owner in children):
                    if time.time() > deadline or any(p.poll() is not None for p in children.values()):
                        for p in children.values():
                            if p.poll() is None:
                                p.terminate()
                        raise RuntimeError(f"benchmark model initialization failed: {name}")
                    time.sleep(1)
                write_json(directory / "start.json", dict(time=time.time()))
                while any(p.poll() is None for p in children.values()):
                    time.sleep(1)
                if any(p.returncode for p in children.values()):
                    raise RuntimeError(f"benchmark inference failed: {name}")
                records = [json.loads((directory / f"{owner}_result.json").read_text()) for owner in children]
                aggregate = {}
                for gpu in gpus:
                    rows = [r for r in records if f"_g{gpu}_" in r["owner"]]
                    wall = max(r["wall_seconds"] for r in rows)
                    aggregate[str(gpu)] = dict(
                        slots=slots,
                        samples=8,
                        seconds=wall,
                        samples_per_second=8 / wall,
                        peak_reserved_bytes=sum(r["peak_reserved_bytes"] for r in rows),
                    )
                write_json(directory / "aggregate.json", aggregate)
            results[backend][str(slots)] = json.loads((directory / "aggregate.json").read_text())
        layout[backend] = {}
        for gpu in map(str, gpus):
            one = results[backend]["1"][gpu]["samples_per_second"]
            two = results[backend]["2"][gpu]["samples_per_second"]
            layout[backend][gpu] = 2 if two > one * 1.05 else 1
        write_json(run / "benchmark.json", dict(gpus=gpus, results=results, layout=layout))
    return layout


def synchronize_cached(run, manifest, queue):
    for backend in ("bf16", "w8"):
        for pair in manifest["pairs"]:
            path = run / backend / f"{pair['id']:04d}.npz"
            if path.exists():
                read_record(path, pair, manifest["contracts"][backend])
                queue.mark_reused(backend, pair["id"])


def phase(run, backend, layout, queue):
    import signal

    children = {}
    restarts = {}

    def launch(gpu, slot):
        key = f"{gpu}_{slot}"
        attempt = restarts.get(key, 0)
        if attempt >= 3:
            raise RuntimeError(f"repeated worker failure: {backend} {key}")
        owner = f"{backend}_g{gpu}_s{slot}_r{attempt}"
        restarts[key] = attempt + 1
        children[key] = dict(
            process=spawn(run, backend, gpu, owner), owner=owner, gpu=gpu, slot=slot, started=time.time()
        )

    for gpu, slots in layout.items():
        for slot in range(slots):
            launch(gpu, slot)
    while True:
        counts = queue.counts(backend)
        write_json(
            run / "status.json",
            dict(
                phase=backend,
                time=time.time(),
                counts={stage: queue.counts(stage) for stage in ("bf16", "w8")},
                workers=[
                    dict(pid=c["process"].pid, owner=c["owner"], gpu=c["gpu"], slot=c["slot"])
                    for c in children.values()
                ],
            ),
        )
        write_json(run / "workers.json", [dict(pid=c["process"].pid, owner=c["owner"]) for c in children.values()])
        if counts.get("failed", 0):
            raise RuntimeError(f"samples exhausted retries: {counts}")
        if counts.get("done", 0) == 1920:
            break
        for child in list(children.values()):
            process = child["process"]
            heartbeat = run / "engines" / f"{child['owner']}_heartbeat.json"
            recent = json.loads(heartbeat.read_text())["time"] if heartbeat.exists() else child["started"]
            if process.poll() is None and time.time() - recent > 900:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=20)
            if process.poll() is not None:
                queue.release_owner(child["owner"])
                if queue.counts(backend).get("pending", 0):
                    launch(child["gpu"], child["slot"])
        time.sleep(5)
    for child in children.values():
        child["process"].wait(timeout=120)


def run_all(args):
    import signal

    run = args.run_dir.resolve()
    lock = (run / "supervisor.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run / "workers.json").exists():
        for row in json.loads((run / "workers.json").read_text()):
            try:
                command = Path(f"/proc/{row['pid']}/cmdline").read_bytes()
                if str(run).encode() in command and b"bf16_open_loop" in command:
                    os.killpg(row["pid"], signal.SIGTERM)
            except (FileNotFoundError, ProcessLookupError):
                pass
    queue = EvaluationQueue(run / "queue.sqlite3")
    queue.recover()
    manifest = json.loads((run / "manifest.json").read_text())
    started = time.time()
    try:
        layout = benchmark(run, args.gpus)
        synchronize_cached(run, manifest, queue)
        bench = json.loads((run / "benchmark.json").read_text())
        eta = {}
        for backend in ("bf16", "w8"):
            rate = sum(
                bench["results"][backend][str(slots)][gpu]["samples_per_second"]
                for gpu, slots in layout[backend].items()
            )
            remaining = 1920 - queue.counts(backend).get("done", 0)
            eta[backend] = dict(remaining=remaining, samples_per_second=rate, estimated_seconds=remaining / rate)
        write_json(
            run / "eta.json",
            dict(
                at=time.time(),
                layout=layout,
                stages=eta,
                estimated_inference_seconds=sum(v["estimated_seconds"] for v in eta.values()),
            ),
        )
        for backend in ("bf16", "w8"):
            phase(run, backend, layout[backend], queue)
        merge(args)
        write_json(
            run / "status.json",
            dict(
                phase="complete",
                time=time.time(),
                counts={s: queue.counts(s) for s in ("bf16", "w8")},
                elapsed_seconds=time.time() - started,
            ),
        )
    except BaseException as error:
        write_json(run / "failure.json", dict(type=type(error).__name__, message=str(error), time=time.time()))
        if (run / "workers.json").exists():
            for row in json.loads((run / "workers.json").read_text()):
                try:
                    command = Path(f"/proc/{row['pid']}/cmdline").read_bytes()
                    if str(run).encode() in command and b"bf16_open_loop" in command:
                        os.killpg(row["pid"], signal.SIGTERM)
                except (FileNotFoundError, ProcessLookupError):
                    pass
        raise


def merge(args):
    import csv

    run = args.run_dir.resolve()
    manifest = json.loads((run / "manifest.json").read_text())
    records = []
    references = []
    quantized = []
    origins = {}
    for pair in manifest["pairs"]:
        bf16, _, bm = read_record(run / "bf16" / f"{pair['id']:04d}.npz", pair, manifest["contracts"]["bf16"])
        w8, _, wm = read_record(run / "w8" / f"{pair['id']:04d}.npz", pair, manifest["contracts"]["w8"])
        observation = manifest["observations"][pair["observation_id"]]
        mask = np.asarray(observation["dof_mask"], bool)
        records.append(
            dict(
                id=pair["id"],
                observation_id=pair["observation_id"],
                task=observation["task_index"],
                episode=observation["episode_index"],
                frame=observation["frame_index"],
                seed=pair["seed"],
                origin=wm["origin"],
                final=numeric_metrics(w8[-1], bf16[-1], mask),
                trajectory=numeric_metrics(w8[1:], bf16[1:], mask),
            )
        )
        references.append(bf16)
        quantized.append(w8)
        origins[wm["origin"]] = origins.get(wm["origin"], 0) + 1
    b, a = np.stack(references), np.stack(quantized)
    mask = np.isin(np.arange(26), range(10, 20))

    def summarize_rows(rows):
        ids = [r["id"] for r in rows]
        return dict(
            samples=len(ids),
            final=numeric_metrics(a[ids, -1], b[ids, -1], mask),
            trajectory=numeric_metrics(a[ids, 1:], b[ids, 1:], mask),
            rmse_p50=float(np.median([r["final"]["rmse"] for r in rows])),
            rmse_p95=float(np.percentile([r["final"]["rmse"] for r in rows], 95)),
            rmse_p99=float(np.percentile([r["final"]["rmse"] for r in rows], 99)),
            worst_cosine=min(r["final"]["cosine"] for r in rows),
        )

    report = dict(
        complete=True,
        baseline="official downloaded checkpoint, native selected BF16/FP32 policy; SDPA",
        scope=manifest["scope"],
        manifest_hash=manifest["manifest_hash"],
        origins=origins,
        overall=summarize_rows(records),
        legacy_120=summarize_rows(records[:120]),
        new_1800=summarize_rows(records[120:]),
        per_task={str(t): summarize_rows([r for r in records if r["task"] == t]) for t in range(40)},
        groups={
            name: numeric_metrics(a[:, -1], b[:, -1], np.isin(np.arange(26), dims))
            for name, dims in dict(position=range(10, 13), rotation=range(13, 19), gripper=[19]).items()
        },
        distribution=dict(
            cosine_below_098=sum(r["final"]["cosine"] < 0.98 for r in records),
            rmse_above_005=sum(r["final"]["rmse"] > 0.05 for r in records),
        ),
        threshold_policy="Descriptive report; no user-approved acceptance threshold and no official LIBERO score",
        worst_cases=sorted(records, key=lambda r: r["final"]["cosine"])[:30],
    )
    write_json(run / "report.json", report)
    with (run / "cases.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(
            [
                "id",
                "observation_id",
                "task",
                "episode",
                "frame",
                "seed",
                "origin",
                "mae",
                "rmse",
                "nrmse",
                "cosine",
                "max_abs",
            ]
        )
        for row in records:
            writer.writerow(
                [row[k] for k in ("id", "observation_id", "task", "episode", "frame", "seed", "origin")]
                + [row["final"][k] for k in ("mae", "rmse", "nrmse", "cosine", "max_abs")]
            )
    np.savez_compressed(run / "predictions.npz", bf16=b, w8=a, active_mask=mask)
    text = [
        "# Wall-OSS-0.5 BF16 / W8A8 LIBERO 开环数值报告",
        "",
        f"配对测试 {len(records)} 组，960 个观测，40 个任务、320 个 episode；每个观测 2 个固定噪声。",
        "",
        "基线使用官方代码与官方 BF16/FP32 混合精度，不使用 FP32 升精度结果作为主基线。"
        "原有 120 个 W8A8 预测通过模型、输入和噪声校验后复用。",
        "",
        "## 整体最终动作误差",
        "",
        "```json",
        json.dumps(report["overall"], ensure_ascii=False, indent=2),
        "```",
        "",
        "原有 120 组与新增 1800 组分开统计，详见 report.json；cases.csv 包含全部样本，predictions.npz 包含逐步轨迹。",
        "",
        "该报告不设官方合格门限，也不是闭环成功率。"
        "448/H32/x2_normal/右臂状态映射沿用既有通用模型数值探针协议，未验证 LIBERO 控制器标定。",
    ]
    (run / "report.md").write_text("\n".join(text) + "\n")
    print(json.dumps(report["overall"], indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "worker", "run", "merge"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--backend", choices=["bf16", "w8"])
    parser.add_argument("--owner", default="manual")
    parser.add_argument("--bench")
    parser.add_argument("--ids")
    parser.add_argument("--detach", action="store_true")
    parser.add_argument("--gpus", type=lambda value: list(map(int, value.split(","))), default=[6, 7])
    args = parser.parse_args()
    if args.stage == "prepare":
        prepare(args)
    elif args.stage == "worker":
        worker(args)
    elif args.stage == "merge":
        merge(args)
    elif args.detach:
        with (args.run_dir / "supervisor.log").open("ab", buffering=0) as output:
            child = subprocess.Popen(
                [
                    "nohup",
                    sys.executable,
                    "-m",
                    "examples_merak.vla.wall_oss.bf16_open_loop",
                    "run",
                    "--run-dir",
                    str(args.run_dir.resolve()),
                    "--gpus",
                    ",".join(map(str, args.gpus)),
                ],
                stdin=subprocess.DEVNULL,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(json.dumps(dict(supervisor_pid=child.pid)), flush=True)
    else:
        run_all(args)


if __name__ == "__main__":
    main()
