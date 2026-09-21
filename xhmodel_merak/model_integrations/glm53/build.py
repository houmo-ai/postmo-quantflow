"""Reuse text artifact packaging; retain GLM's separately streamed vision graphs."""

import json
from pathlib import Path

from xhmodel_merak.model_build.artifacts import flatten_single_export_root
from xhmodel_merak.model_integrations.autoregressive import build as text_build

from .artifact import audit_glm53_hmonnx, validate_glm53_bundle


def normalize_export(stage_path, export_result, plan):
    flatten_single_export_root(stage_path)
    from .shared_weights import compact_glm53_language_weights

    compact_glm53_language_weights(stage_path)
    return None


def package_runtime(stage_path, plan):
    text_build.package_runtime(stage_path, plan)


def sanitize_artifact(stage_path, plan):
    path = Path(stage_path) / "golden_meta_info.json"
    meta = json.loads(path.read_text())
    vision = meta["glm53_vision"]
    for component in vision["components"]:
        component["hmonnx"] = (Path(vision["directory"]) / component["hmonnx"]).as_posix()
    vision["directory"] = "."
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    # The shared sanitizer now retains only referenced runtime files. Resolve
    # component paths against the bundle before it computes that file closure.
    text_build.sanitize_artifact(stage_path, plan)


def validate_artifact(stage_path, plan):
    result = text_build.validate_artifact(stage_path, plan)
    metadata = json.loads((Path(stage_path) / result["metadata_file"]).read_text())
    validate_glm53_bundle(stage_path, metadata)
    result["precision_audit"] = audit_glm53_hmonnx(stage_path, metadata)
    vision = metadata["glm53_vision"]
    for component in vision["components"]:
        result["export_entries"][f"visual_{component['name']}"] = {
            "path": (Path(vision["directory"]) / component["hmonnx"]).as_posix(),
            "type": "hmonnx",
        }
    return result
