# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
import json

from xhmodel_merak.model_artifact.files import confined_file
from xhmodel_merak.model_build.packaging import validate_onnx_external_locations
from xhmodel_merak.model_build.runtime_assets import quant_runtime_asset_paths


def normalize_export(stage_path, export_result, plan):
    text = json.loads((stage_path / "golden_meta_info.json").read_text())
    return text["model_config"]["model_name"]


def package_runtime(stage_path, plan):
    pass  # Workflow includes native OCR tokenizer and processor resources.


def sanitize_artifact(stage_path, plan):
    # Native source paths are provenance only; runtime exclusively resolves
    # graph and HF resources relative to the metadata file in the package.
    child_file = stage_path / "golden_meta_info.json"
    child = json.loads(child_file.read_text())
    child["model_config"]["hf_model"] = child["hf_config"]
    child_file.write_text(json.dumps(child, indent=2) + "\n")
    entries = describe_artifact(
        stage_path, lambda value, field: confined_file(stage_path, value, field), plan.components
    )
    keep = {
        stage_path / name for name in ("runtime_meta_info.json", "golden_meta_info.json", "effective_workflow.yaml")
    }
    graphs = []
    for entry in entries.values():
        candidate = stage_path / entry["path"]
        if entry["type"] == "tokenizer":
            keep.update(quant_runtime_asset_paths(candidate))
        else:
            keep.add(candidate)
            if candidate.suffix == ".onnx":
                graphs.append(candidate)
    keep.update(validate_onnx_external_locations(stage_path, graphs=graphs))
    keep = {path.resolve() for path in keep}
    for path in sorted(stage_path.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file() and path.resolve() not in keep:
            path.unlink()
        elif path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def describe_artifact(root, resolve_file, components):
    meta = json.loads(resolve_file("runtime_meta_info.json", "metadata").read_text())
    if meta.get("schema_version") != 1 or meta.get("model_type") != "minicpm_ocr":
        raise ValueError("Unsupported MiniCPM-OCR runtime metadata")
    if tuple(meta.get("components", ())) != tuple(components) or meta.get("downsample_mode") != "4x":
        raise ValueError("MiniCPM-OCR component/profile mismatch")
    text_file = resolve_file(meta["text_metadata"], "text_metadata")
    text = json.loads(text_file.read_text())
    entries = {
        "visual": {"path": meta["visual"]["hmonnx"], "type": "hmonnx"},
        "prefill": {"path": text["prefill_hmonnx"], "type": "prefill"},
        "decode": {"path": text["decode_hmonnx"], "type": "decode"},
        "quant_embedding": {"path": text["quant_embedding"], "type": "embedding"},
        "tokenizer": {"path": meta["hf_config"], "type": "tokenizer"},
    }
    for name, item in entries.items():
        value = item["path"] + "/config.json" if item["type"] == "tokenizer" else item["path"]
        resolve_file(value, name)
    return entries


def validate_artifact(stage_path, plan):
    entries = describe_artifact(
        stage_path, lambda value, field: confined_file(stage_path, value, field), plan.components
    )
    return {
        "metadata_file": "runtime_meta_info.json",
        "export_entries": entries,
        "runtime_kind": "minicpm_ocr",
        "components": list(plan.components),
        "requires_source": False,
        "loadable": True,
        "quantized": True,
    }
