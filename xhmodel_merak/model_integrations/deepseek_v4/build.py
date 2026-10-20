"""Publish the complete target/DSpark component closure."""
import json
import hashlib
from pathlib import Path

from xhmodel_merak.model_integrations.autoregressive.build import (
    normalize_export as normalize_autoregressive_export,
    package_runtime as package_autoregressive_runtime, sanitize_artifact,
    validate_artifact as validate_autoregressive_artifact,
)
from xhmodel_merak.model_artifact.files import confined_file


def normalize_export(stage_path, export_result, plan):
    """Share only target prefill/decode weights; DSpark components stay independent."""
    from xhmodel_merak.model_integrations.shared_weights import compact_language_weights

    result = normalize_autoregressive_export(stage_path, export_result, plan)
    compact_language_weights(stage_path)
    return result


def inspect_auxiliary_checkpoint_weights(root, plan):
    """Verify the producer's independently indexed native draft component."""
    from gptqmodel.utils.deepseek_v4_dspark import inspect_deepseek_v4_dspark
    from xhmodel_merak.model_artifact.files import file_sha256

    component = Path(root) / "dspark"
    if not component.exists():
        return {}
    inventory = inspect_deepseek_v4_dspark(root)
    manifest_path = component / "dspark_meta_info.json"
    metadata = json.loads(manifest_path.read_text())
    if inventory is None or metadata.get("format_version") != 1 or metadata.get("num_draft_layers") != 3:
        raise ValueError("Invalid producer-preserved DSpark component")
    if metadata["tensor_count"] != len(inventory["weight_map"]):
        raise ValueError("DSpark tensor inventory differs from its preservation manifest")
    paths = [confined_file(component, name, "dspark.weight")
             for name in sorted(set(inventory["weight_map"].values()))]
    return {"dspark": {"weight_paths": paths, "index_path": component / "model.safetensors.index.json",
        "evidence": {"manifest_sha256": file_sha256(manifest_path), "num_draft_layers": 3,
                     "weights": [{"file": path.relative_to(root).as_posix(), "size": path.stat().st_size,
                                  "sha256": file_sha256(path)} for path in paths]}}}


def package_runtime(stage_path, plan):
    package_autoregressive_runtime(stage_path, plan)
    from transformers import AutoTokenizer
    from xhmodel_merak.xh_llm.models.deepseek_v4.chat import configure_flash_chat_tokenizer

    root = Path(stage_path) / "hf_config"
    tokenizer = configure_flash_chat_tokenizer(
        AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False))
    config_path = root / "tokenizer_config.json"
    config = json.loads(config_path.read_text())
    config["chat_template"] = tokenizer.chat_template
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    meta_path = Path(stage_path) / "golden_meta_info.json"
    if meta_path.is_file():
        metadata = json.loads(meta_path.read_text())
        scheme = metadata.get("model_config", {}).get("quant_scheme")
        if isinstance(scheme, dict) and "nodes_cfg" in scheme:
            scheme.setdefault("nodes", {}).update(scheme.pop("nodes_cfg"))
        # Published-name normalization rewrites graph external-data names.
        # Refresh the exporter checksums to describe these final graph bytes.
        for role in ("prefill", "decode"):
            key = f"{role}_hmonnx"
            graph = confined_file(Path(stage_path), metadata[key], key)
            metadata[f"{key}_md5"] = hashlib.md5(graph.read_bytes()).hexdigest()
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


def validate_dspark(root, metadata, plan):
    if "dspark_draft_decode" not in plan.components:
        return
    component = metadata.get("dspark")
    expected = {"dspark_context", "dspark_context_decode", "dspark_draft_decode", "dspark_markov"}
    if not isinstance(component, dict) or set(component.get("components", {})) != expected:
        raise ValueError("DSpark Artifact requires all four draft graphs")
    if component["spec"].get("max_context_length") != metadata["model_config"]["context_max_length"]:
        raise ValueError("DSpark RoPE context length does not match the target")
    if tuple(component["spec"]["target_layer_ids"]) != tuple(
        metadata["deepseek_v4_auxiliary_layer_ids"]
    ):
        raise ValueError("DSpark feature layers do not match the exported target")
    if component.get("weight_precision") != {
        "routed_experts": 4, "shared_experts": 8, "dense": 8, "group_size": 64,
        "shared_lm_head": component.get("spec", {}).get("head_weight_bits"),
    } or component["spec"]["head_weight_bits"] not in (4, 5, 8):
        raise ValueError("DSpark Artifact has an incompatible W4/W8/shared-head contract")
    if metadata["model_config"]["num_logits_to_keep"] != component["spec"]["block_size"] + 1:
        raise ValueError("DSpark target verification width does not match the draft")
    for entry in component["components"].values():
        if entry.get("type") != "hmonnx" or not confined_file(Path(root), entry["path"], "dspark.graph").is_file():
            raise ValueError("DSpark Artifact has an invalid component graph")


def validate_artifact(stage_path, plan):
    result = validate_autoregressive_artifact(stage_path, plan, speculative_validator=validate_dspark)
    metadata = json.loads((Path(stage_path) / result["metadata_file"]).read_text())
    if metadata.get("dspark"):
        result["export_entries"].update(metadata["dspark"]["components"])
    return result


__all__ = ["normalize_export", "package_runtime", "sanitize_artifact", "validate_artifact", "validate_dspark"]
