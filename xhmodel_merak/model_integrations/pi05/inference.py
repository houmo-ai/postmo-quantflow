from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from xhmodel_merak.model_catalog.contracts import ModelRecipe
from xhmodel_merak.model_inference.contracts import ActionInferenceResult
from xhmodel_merak.model_inference.inputs import require_local_file, validate_inputs


class _PI05Adapter:
    def run(
        self,
        *,
        recipe: ModelRecipe,
        root: Path,
        manifest: dict[str, Any],
        meta_file: Path,
        inputs: dict[str, Any],
        device: str,
        components: tuple[str, ...],
        selected_golden: tuple[str, ...],
        golden_artifacts: dict[str, tuple[str, ...]],
    ) -> ActionInferenceResult:
        normalized = validate_inputs(recipe, inputs)
        image = require_local_file(recipe, "observation.images.image", normalized["observation.images.image"])
        image2 = require_local_file(recipe, "observation.images.image2", normalized["observation.images.image2"])
        from xhmodel_merak.xh_other_model.models.pi05.compact_runtime import PI05CompactRuntime

        runtime = PI05CompactRuntime(root, device=device, manifest=manifest)
        started = time.perf_counter()
        actions, stats = runtime.predict(
            task=normalized["task"],
            image=image,
            image2=image2,
            state=normalized["observation.state"],
            seed=int(normalized["seed"]),
        )
        if not np.isfinite(actions).all():
            raise ValueError("PI0.5 runtime produced non-finite actions")
        latency_s = time.perf_counter() - started
        action_rows = tuple(tuple(float(value) for value in row) for row in actions.tolist())
        return ActionInferenceResult(
            model_id=recipe.model_id,
            task=recipe.task,
            output_type="actions",
            artifact_dir=str(root),
            meta_file=str(meta_file),
            latency_s=latency_s,
            components=components,
            golden_components=selected_golden,
            golden_artifacts=golden_artifacts,
            actions=action_rows,
            horizon=len(action_rows),
            action_dim=len(action_rows[0]) if action_rows else 0,
            seed=int(normalized["seed"]),
            stats=stats,
        )


adapter = _PI05Adapter()


__all__ = ["adapter"]
