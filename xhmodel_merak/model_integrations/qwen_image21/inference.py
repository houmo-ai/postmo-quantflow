"""Bind the public interface to the deployed image runtime."""

import time
from dataclasses import dataclass
from typing import Any

from xhmodel_merak.model_artifact.files import artifact_file
from xhmodel_merak.model_inference.contracts import InferenceResult
from xhmodel_merak.model_inference.inputs import require_local_file, validate_inputs

from .artifact import validate_runtime


@dataclass(frozen=True, kw_only=True)
class ImageInferenceResult(InferenceResult):
    image: Any
    width: int
    height: int
    seed: int


class _ImageAdapter:
    def validate_artifact(self, *, recipe, root, manifest, meta_file, **_):
        if meta_file != root / "runtime_meta_info.json":
            raise ValueError("Qwen-Image requires runtime_meta_info.json")
        meta, _ = validate_runtime(
            root, recipe.components, resolve_file=lambda path, field: artifact_file(root, manifest, path, field)
        )
        if meta.get("model_id", "qwen-image-2.1") != recipe.model_id:
            raise ValueError("runtime model_id disagrees with the requested model identity")
        if manifest.get("parameters", {}).get("precision", {}).get("deployment") != {"image": meta["quant_type"]}:
            raise ValueError("runtime quant_type disagrees with Manifest deployment precision")

    def run(self, *, recipe, root, manifest, meta_file, inputs, device, components, selected_golden, golden_artifacts):
        import numpy as np
        from PIL import Image

        from xhmodel_merak.xh_other_model.models.qwen_image21.bundle import load_runtime
        from xhmodel_merak.xh_other_model.models.qwen_image21.sampling import validate_sigmas

        values = validate_inputs(recipe, inputs)
        if "sigmas" in values:
            values["sigmas"] = validate_sigmas(values["sigmas"])
        meta, _ = validate_runtime(
            root, components, resolve_file=lambda path, field: artifact_file(root, manifest, path, field)
        )
        images = []
        for path in values.pop("images"):
            with Image.open(require_local_file(recipe, "images", path)) as image:
                images.append(image.convert("RGBA"))
        if len(images) > meta["deployment_config"]["max_references"]:
            raise ValueError("reference count exceeds the published capacity")
        if values["guidance_scale"] > 1 and "negative_prompt" not in values:
            raise ValueError("guidance_scale > 1 requires negative_prompt")
        runtime = load_runtime(root, device=device)
        started = time.perf_counter()
        try:
            image, _ = runtime.generate(**values, images=images)
            latency = time.perf_counter() - started
            actual_steps = runtime.num_inference_steps
        finally:
            del runtime
        return ImageInferenceResult(
            model_id=recipe.model_id,
            task=recipe.task,
            output_type="image",
            artifact_dir=str(root),
            meta_file=str(meta_file),
            latency_s=latency,
            components=components,
            golden_components=selected_golden,
            golden_artifacts=golden_artifacts,
            image=np.asarray(image).copy(),
            width=image.width,
            height=image.height,
            seed=values["seed"],
            stats={"steps": actual_steps, "reference_count": len(images)},
        )


adapter = _ImageAdapter()
