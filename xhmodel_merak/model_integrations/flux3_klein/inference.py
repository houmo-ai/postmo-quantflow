"""Bind the public interface to the deployed image runtime."""

import time
from dataclasses import dataclass
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError
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
            raise ValueError("FLUX 3 requires runtime_meta_info.json")
        meta, _ = validate_runtime(
            root, recipe.components, resolve_file=lambda path, field: artifact_file(root, manifest, path, field)
        )
        if meta.get("model_id") != recipe.model_id:
            raise ValueError("runtime model_id disagrees with the requested model identity")
        if manifest.get("parameters", {}).get("precision", {}).get("deployment") != {"image": meta["quant_type"]}:
            raise ValueError("runtime quant_type disagrees with Manifest deployment precision")

    def run(self, *, recipe, root, manifest, meta_file, inputs, device, components, selected_golden, golden_artifacts):
        import numpy as np
        from PIL import Image

        from xhmodel_merak.xh_other_model.models.flux3_klein.bundle import load_runtime

        values = validate_inputs(recipe, inputs)
        meta, _ = validate_runtime(
            root, components, resolve_file=lambda path, field: artifact_file(root, manifest, path, field)
        )
        if values["prompt_upsampling"] and not meta.get("prompt_upsampler"):
            raise ModelError(
                ErrorCode.UNSUPPORTED_COMBINATION,
                "this bundle does not include a prompt upsampler",
                model=recipe.model_id,
            )
        images = []
        for path in values.pop("images"):
            with Image.open(require_local_file(recipe, "images", path)) as image:
                images.append(image.copy())
        if len(images) > meta["capacity"]["max_references"]:
            raise ValueError("reference count exceeds the published capacity")
        runtime = load_runtime(root, device=device)
        started = time.perf_counter()
        try:
            image, _ = runtime.generate(**values, images=images)
            latency = time.perf_counter() - started
            actual_steps = runtime.num_inference_steps
            caption = runtime.last_caption
            upsampler_stats = runtime.last_upsampler_stats
            geometry = runtime.last_geometry
        finally:
            if runtime.upsampler is not None:
                runtime.upsampler.close()
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
            stats={
                "steps": actual_steps,
                "reference_count": len(images),
                "geometry": geometry,
                "caption": caption,
                "prompt_upsampling": values["prompt_upsampling"],
                "upsampler": upsampler_stats,
            },
        )


adapter = _ImageAdapter()
