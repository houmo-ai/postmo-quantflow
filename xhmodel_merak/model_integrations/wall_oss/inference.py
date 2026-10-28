"""Public action-prediction adapter; all learned computation uses HMONNX."""

import time

from xhmodel_merak.model_artifact.files import artifact_file
from xhmodel_merak.model_inference.contracts import ActionInferenceResult
from xhmodel_merak.model_inference.inputs import require_local_file, validate_inputs


class WallAdapter:
    def validate_artifact(self, *, root, manifest, meta_file, **_):
        from xhmodel_merak.xh_other_model.models.wall_oss.build import export_entries
        from xhmodel_merak.xh_other_model.models.wall_oss.runtime import read_metadata

        meta = read_metadata(root)
        if meta_file != root / "runtime_meta_info.json":
            raise ValueError("Wall requires runtime metadata")
        for key, entry in export_entries(root, meta).items():
            if entry["type"] != "tokenizer":
                artifact_file(root, manifest, entry["path"], key)
        if manifest.get("parameters", {}).get("precision", {}).get("deployment") != {"policy": meta["quant_type"]}:
            raise ValueError("Manifest and runtime precision disagree")

    def run(self, *, recipe, root, manifest, meta_file, inputs, device, components, selected_golden, golden_artifacts):
        from PIL import Image

        from xhmodel_merak.xh_other_model.models.wall_oss.runtime import WallRuntime

        values = validate_inputs(recipe, inputs)
        images = []
        for path in values.pop("images"):
            with Image.open(require_local_file(recipe, "images", path)) as image:
                images.append(image.convert("RGB"))
        runtime = WallRuntime(root, device=device)
        start = time.perf_counter()
        actions, stats = runtime.predict(**values, images=images)
        return ActionInferenceResult(
            model_id=recipe.model_id,
            task=recipe.task,
            output_type="actions",
            artifact_dir=str(root),
            meta_file=str(meta_file),
            latency_s=time.perf_counter() - start,
            components=components,
            golden_components=selected_golden,
            golden_artifacts=golden_artifacts,
            stats=stats,
            actions=tuple(tuple(float(x) for x in row) for row in actions),
            horizon=32,
            action_dim=26,
            seed=values["seed"],
        )


adapter = WallAdapter()
