# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
import time
from dataclasses import dataclass

from xhmodel_merak.model_artifact.files import artifact_file
from xhmodel_merak.model_inference.contracts import InferenceResult
from xhmodel_merak.model_inference.inputs import validate_inputs

from .artifact import validate_precision, validate_runtime


@dataclass(frozen=True, kw_only=True)
class VideoInferenceResult(InferenceResult):
    output_path: str
    width: int
    height: int
    frames: int
    fps: int
    seed: int


class _SolH3Adapter:
    def validate_artifact(self, *, recipe, root, manifest, meta_file, **_):
        if meta_file != root / "runtime_meta_info.json":
            raise ValueError("Sol-H3 requires runtime_meta_info.json")
        meta, _ = validate_runtime(
            root, recipe.components, resolve_file=lambda path, field: artifact_file(root, manifest, path, field)
        )
        precision = manifest.get("parameters", {}).get("precision", {})
        validate_precision(
            meta,
            deployment=precision.get("deployment"),
            checkpoint_weight=precision.get("checkpoint", {}).get("weight"),
        )
        if manifest.get("parameters", {}).get("profile") != meta["profile"]:
            raise ValueError("Sol-H3 graph canvas disagrees with the Manifest")

    def run(self, *, recipe, root, manifest, meta_file, inputs, device, components, selected_golden, golden_artifacts):
        from xhmodel_merak.xh_other_model.models.sol_h3.runtime import SolH3Runtime

        values = validate_inputs(recipe, inputs)
        validate_runtime(root, components, resolve_file=lambda path, field: artifact_file(root, manifest, path, field))
        started = time.perf_counter()
        runtime = SolH3Runtime(root, device=device)
        try:
            result = runtime.generate(**values)
        finally:
            runtime.close()
        return VideoInferenceResult(
            model_id=recipe.model_id,
            task=recipe.task,
            output_type="video",
            artifact_dir=str(root),
            meta_file=str(meta_file),
            latency_s=time.perf_counter() - started,
            components=components,
            golden_components=selected_golden,
            golden_artifacts=golden_artifacts,
            output_path=str(result["output_path"]),
            width=runtime.meta["refiner_size"][1],
            height=runtime.meta["refiner_size"][0],
            frames=runtime.meta["refiner_frames"],
            fps=runtime.meta["fps"],
            seed=values["seed"],
            stats=result["stats"],
        )


adapter = _SolH3Adapter()
