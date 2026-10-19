"""Unified API binding for the prepared DepthSplat Gaussian encoder."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from xhmodel_merak.model_artifact.files import artifact_file
from xhmodel_merak.model_inference.contracts import InferenceResult
from xhmodel_merak.model_inference.inputs import validate_inputs
from .build import load_runtime_metadata


def _load(root, manifest, meta_file, components):
    checked = artifact_file(root, manifest, meta_file.relative_to(root).as_posix(), "metadata_file")
    return load_runtime_metadata(checked, lambda path, field: artifact_file(root, manifest, path, field), components)


class _DepthSplatAdapter:
    def validate_artifact(self, *, recipe, root, manifest, meta_file, **_: Any) -> None:
        _load(root, manifest, meta_file, recipe.components)

    def run(
        self, *, recipe, root, manifest, meta_file, inputs, device, components, selected_golden, golden_artifacts
    ) -> InferenceResult:
        normalized = validate_inputs(recipe, inputs)
        meta = _load(root, manifest, meta_file, components)
        from xhmodel_merak.xh_other_model.models.depthsplat.runtime import run_artifact

        started = time.perf_counter()
        outputs = run_artifact(root, meta, Path(normalized["sample"]), device)
        return InferenceResult(
            model_id=recipe.model_id,
            task=recipe.task,
            output_type="gaussian-scene",
            artifact_dir=str(root),
            meta_file=str(meta_file),
            latency_s=time.perf_counter() - started,
            components=components,
            golden_components=selected_golden,
            golden_artifacts=golden_artifacts,
            stats={"outputs": outputs},
        )


adapter = _DepthSplatAdapter()
