"""Golden collection reuses the demo's exact HMONNX -> CPU -> HMONNX chain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch

from .bucketed_runtime import KokoroBucketedRuntime
from .buckets import frame_bucket_key, token_bucket_key
from .host import DEFAULT_INPUT_IDS, load_voice_style
from .independent_split import PHASE_CORE_ROLE, TEXT_DURATION_ROLE
from .runtime import HmonnxRunner, OrtRunner


class GoldenHmonnxRunner(HmonnxRunner):
    def __init__(self, model: Path, device: str) -> None:
        from xhquant.api import HMONNXGoldenInference

        proto = onnx.load(str(model), load_external_data=False)
        self.input_names = [value.name for value in proto.graph.input]
        self.output_names = [value.name for value in proto.graph.output]
        self.device = torch.device(device)
        self.session = HMONNXGoldenInference(str(model))
        self.session.to(self.device)
        self.session.legacy_mode = False
        self.session.save_golden = True
        self.session.golden_dir = str(model.parent)
        self.session.step = 0


class GoldenCpuRunner(OrtRunner):
    def __init__(self, model: Path) -> None:
        super().__init__(model)
        self.model = model

    def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        outputs = super().run(feed)
        step = self.model.parent / "step_0"
        step.mkdir()
        (step / self.model.name).symlink_to(Path("..") / self.model.name)
        prefix = self.model.stem.removesuffix("_with_act")
        for kind, values in (("input", feed), ("output", outputs)):
            for name, value in values.items():
                np.save(step / f"{prefix}_{name}_{kind}.npy", value, allow_pickle=False)
        return outputs


def dump_all_routes(work_dir: str | Path, *, device: str, input_messages: Any = None) -> str:
    root = Path(work_dir).resolve()
    runtime = KokoroBucketedRuntime.from_export(root, backend="hmonnx", device=device)
    if not runtime.meta["hmonnx_converted"]:
        raise ValueError("golden requires HMONNX export")
    # Fail before creating any files if an earlier collection already exists.
    for role in runtime.meta["component_order"]:
        for bucket in runtime.meta["components"][role]["buckets"].values():
            graph = root / bucket["onnx_file" if role == PHASE_CORE_ROLE else "hmonnx_file"]
            if (graph.parent / "step_0").exists():
                raise FileExistsError(f"golden already exists: {graph.parent / 'step_0'}")
    message = {} if input_messages is None else input_messages
    if not isinstance(message, dict) or set(message) - {"input_ids", "speed"}:
        raise ValueError("golden input_messages must be {input_ids: [...], speed: 1.0} or None")
    tokens = np.asarray(message.get("input_ids", DEFAULT_INPUT_IDS), dtype=np.int32).reshape(-1)
    routes = runtime.available_routes()
    if tokens.size < 3 or tokens.size > min(int(route["token_max_length"]) for route in routes):
        raise ValueError("golden tokens must fit every exported route and contain two boundary tokens")
    style = load_voice_style(root / runtime.meta["runtime_assets"]["voice_pack"]["file"], tokens.size - 2).numpy()
    speed = float(message.get("speed", 1.0))
    collected: dict[str, Any] = {}

    def factory(role: str, key: str, path: Path):
        collected[f"{role}_{key}"] = {
            "model_file": str(path.relative_to(root)),
            "golden_dir": str((path.parent / "step_0").relative_to(root)),
            "execution": "cpu_fp32" if role == PHASE_CORE_ROLE else "hmonnx",
            "per_operator": role != PHASE_CORE_ROLE,
        }
        return GoldenCpuRunner(path) if role == PHASE_CORE_ROLE else GoldenHmonnxRunner(path, device)

    runtime.runner_factory = factory
    manifest = {
        "complete": False,
        "release_prefix": runtime.meta["release_prefix"],
        "input_ids": tokens.tolist(),
        "speed": speed,
        "feed_source": "connected_hmonnx_outputs",
        "routes": [],
        "graphs": collected,
    }
    destination = root / "golden_meta_info.json"
    for route in routes:
        print(f"Collecting golden {route['key']}", flush=True)
        try:
            waveform, synthesis = runtime.synthesize(
                tokens,
                style,
                speed=speed,
                token_bucket=int(route["token_max_length"]),
                frame_bucket=int(route["frame_max_length"]),
            )
            if not np.isfinite(waveform).all():
                raise RuntimeError(f"non-finite golden waveform for {route['key']}")
            tk = token_bucket_key(int(route["token_max_length"]))
            fk = frame_bucket_key(int(route["frame_max_length"]))
            expected = {
                f"{role}_{tk if role == TEXT_DURATION_ROLE else fk}" for role in runtime.meta["component_order"]
            }
            if not expected.issubset(collected):
                raise RuntimeError(f"golden missed components: {expected.difference(collected)}")
            manifest["routes"].append(synthesis)
            np.save(root / f"golden_{route['key']}_waveform.npy", waveform, allow_pickle=False)
        finally:
            runtime.clear_runner_cache()
            destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    manifest["complete"] = True
    destination.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(destination)
