from __future__ import annotations

import gc
import json
import shutil
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from ...workflows.base import BaseOtherModelWorkflow
from ...workflows.result import ExportResult, QuantResult
from .assets import resolve_model_assets, verify_release_assets
from .bucketed_runtime import GRAPH_MODE, SCHEMA_VERSION
from .buckets import frame_bucket_key, normalize_bucket_routes, token_bucket_key
from .graph import artifact_metadata, load_official_model
from .host import SAMPLE_RATE, default_static_sample
from .independent_split import (
    FRAME_ACOUSTIC_ROLE,
    GENERATOR_ISTFT_ROLE,
    PHASE_CORE_ROLE,
    TEXT_DURATION_ROLE,
    export_frame_bucket,
    export_text_bucket,
    prepare_independent_model,
    run_duration_alignment_host,
)
from .quant_export import _convert_hmonnx, _export_runtime_assets, _hmonnx_artifact_metadata
from .release import component_path, release_prefix


NPU_ROLES = (TEXT_DURATION_ROLE, FRAME_ACOUSTIC_ROLE, GENERATOR_ISTFT_ROLE)
COMPONENT_ORDER = (TEXT_DURATION_ROLE, FRAME_ACOUSTIC_ROLE, PHASE_CORE_ROLE, GENERATOR_ISTFT_ROLE)


def _validate_export_config(config: Mapping[str, Any]) -> None:
    obsolete = {
        "graph_mode",
        "lstm_variants",
        "decompose_lstm",
        "allow_decomposed_failure",
        "phase_on_npu",
        "text_max_length",
        "frame_max_length",
        "lstm_chunk_length",
        "validate_onnx",
        "simplify",
        "torch_onnx_internal_optimize",
        "stft_pad_mode",
        "stft_phase_mode",
        "f0_norm_mode",
    }.intersection(config)
    if obsolete:
        raise ValueError(f"obsolete Kokoro export options: {sorted(obsolete)}")
    if config["target_device"] != "XH2a":
        raise ValueError("this Kokoro release recipe is validated only for XH2a")
    if int(config.get("opset", 17)) != 17:
        raise ValueError("this Kokoro recipe requires opset 17")
    components = config.get("components", {})
    if set(components) != set(NPU_ROLES):
        raise ValueError(f"export.components must contain exactly {NPU_ROLES}")
    if any(components[role].get("quant_type") != "w16a16_sefp" for role in NPU_ROLES):
        raise ValueError("the approved Kokoro recipe requires w16a16_sefp for all NPU graphs")


class KokoroWorkflow(BaseOtherModelWorkflow):
    """Export static wrappers; inference/accuracy checks belong to the demo."""

    def quant(self, output_dir: str, device: str, config_overrides: Mapping[str, Any] | None = None) -> QuantResult:
        config = self.workflow_config.with_overrides(config_overrides)
        if config.quant is not None:
            raise NotImplementedError("Kokoro performs PTQ during HMONNX export; quant must be null")
        return QuantResult(raw_model_dir=self.model_dir, skipped=True)

    def export(
        self, quant_result: QuantResult, output_dir: str, device: str, config_overrides: Mapping[str, Any] | None = None
    ) -> ExportResult:
        config = self.workflow_config.with_overrides(config_overrides)
        export_cfg = config.build_export_dict()
        _validate_export_config(export_cfg)
        routes = normalize_bucket_routes(
            export_cfg.get("token_buckets"),
            export_cfg.get("audio_seconds_buckets"),
        )
        prefix = release_prefix(routes, str(export_cfg.get("release_date", datetime.now().strftime("%Y%m%d"))))
        root = Path(output_dir).expanduser().resolve() / prefix
        if root.exists():
            raise FileExistsError(f"release already exists: {root}; choose a new output directory")
        assets = resolve_model_assets(self._resolve_export_model_dir(quant_result))
        identity = verify_release_assets(assets)
        root.mkdir(parents=True)
        config_file = config.dump(str(root / "workflow.yaml"))
        runtime_assets = _export_runtime_assets(assets.voice, root)
        hf_dir = root / "hf_config"
        hf_dir.mkdir()
        for source in assets.config.parent.rglob("*"):
            if source.is_file() and source.suffix.lower() in {".json", ".txt", ".yaml", ".yml", ".jinja", ".tiktoken"}:
                destination = hf_dir / source.relative_to(assets.config.parent)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        runtime_assets["config_file"] = str((hf_dir / assets.config.name).relative_to(root))
        example = Path(__file__).resolve().parents[4] / "examples_merak/audio/kokoro"
        shutil.copy2(example / "bucketed_inference_demo.py", root / f"{prefix}_hmonnx.py")
        shutil.copy2(example / "README.md", root / "README.md")
        meta: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "graph_mode": GRAPH_MODE,
            "release_prefix": prefix,
            "export_complete": False,
            "target_device": "XH2a",
            "quant_type": "w16a16_sefp",
            "seed": int(export_cfg.get("seed", 1234)),
            "sample_rate": SAMPLE_RATE,
            "hmonnx_converted": bool(export_cfg.get("convert_hmonnx", True)),
            "asset_identity": identity,
            "runtime_assets": runtime_assets,
            "bucket_presets": {"policy": "duration_driven_paired_t_f", "routes": [route.as_dict() for route in routes]},
            "phase_boundary": {"execution": "host_onnxruntime_fp32", "input": "f0", "output": "sine"},
            "component_order": list(COMPONENT_ORDER),
            "components": {role: {"buckets": {}} for role in COMPONENT_ORDER},
        }
        model = load_official_model(assets)
        meta["wrappers"] = prepare_independent_model(model)
        if meta["hmonnx_converted"] and str(device).startswith("cuda"):
            torch.cuda.set_device(torch.device(device))
        for route in routes:
            print(f"Exporting {route.key}", flush=True)
            tk, fk = token_bucket_key(route.token_max_length), frame_bucket_key(route.frame_max_length)
            sample = default_static_sample(assets.voice, route.token_max_length)
            text = export_text_bucket(
                model=model,
                sample=sample,
                text_max_length=route.token_max_length,
                opset=17,
                output_path=component_path(root, prefix, TEXT_DURATION_ROLE, tk, reference=True),
            )
            self._publish_graph(
                meta,
                root,
                TEXT_DURATION_ROLE,
                tk,
                text.artifact,
                text.feed,
                {"token_max_length": route.token_max_length},
            )
            frame_inputs, _ = run_duration_alignment_host(
                text.outputs,
                speed=sample.speed,
                valid_len=sample.valid_len,
                frame_max_length=route.frame_max_length,
            )
            frame = export_frame_bucket(
                model=model,
                frame_inputs=frame_inputs,
                style=sample.style,
                output_paths={
                    role: component_path(root, prefix, role, fk, reference=role != PHASE_CORE_ROLE)
                    for role in COMPONENT_ORDER[1:]
                },
                frame_max_length=route.frame_max_length,
                seed=meta["seed"],
                opset=17,
                stft_pad_mode="length_aware_reflect",
                stft_phase_mode="cordic",
                f0_norm_mode="adain",
            )
            for role in COMPONENT_ORDER[1:]:
                self._publish_graph(
                    meta,
                    root,
                    role,
                    fk,
                    frame.graphs[role],
                    frame.feeds[role],
                    {"frame_max_length": route.frame_max_length, "audio_seconds": route.audio_seconds},
                )
            del text, frame, frame_inputs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _write_metadata(root, meta)
        meta["export_complete"] = True
        _write_metadata(root, meta)
        return ExportResult(work_dir=str(root), config_file=config_file, meta=meta)

    def _publish_graph(self, meta, root, role, key, artifact, feed, capacity):
        entry = {**artifact_metadata(artifact, root), **capacity}
        if role != PHASE_CORE_ROLE and meta["hmonnx_converted"]:
            path = component_path(root, meta["release_prefix"], role, key, reference=False)
            # Export under the base stem so xhquant chooses the release external_data
            # name. Only the ONNX filename gets the conventional _with_act suffix.
            # Never rename external_data or patch its location after conversion.
            export_path = path.with_name(path.name.removesuffix("_with_act.onnx") + ".onnx")
            _convert_hmonnx(artifact, feed, export_path, target="XH2a", quant_type="w16a16_sefp", debug=self.debug)
            export_path.rename(path)
            entry.update(_hmonnx_artifact_metadata(path, root))
        meta["components"][role]["buckets"][key] = entry

    def dump_golden(self, export_result: ExportResult, device: str, input_messages: Any = None) -> str:
        from .golden import dump_all_routes

        return dump_all_routes(export_result.work_dir, device=device, input_messages=input_messages)


def _write_metadata(root: Path, metadata: Mapping[str, Any]) -> None:
    (root / "export_meta_info.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
