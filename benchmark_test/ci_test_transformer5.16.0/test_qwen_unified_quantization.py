"""Public recipe head ownership, frozen checkpoint reuse, and LoRA composition."""
import json

import pytest

from xhmodel_merak.model_build.resolver import resolve_build_plan
from xhmodel_merak.model_build.workflow import materialize_workflow_config
from xhmodel_merak.model_catalog import get_recipe
from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig


def source_for(tmp_path, model_id):
    recipe = get_recipe(model_id)
    data = {}
    for dotted, value in recipe.source_constraints:
        cursor = data
        parts = dotted.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps(data))
    return source


def effective(plan, tmp_path):
    path = tmp_path / "effective.yaml"
    materialize_workflow_config(plan, path)
    return WorkflowConfig.from_file(str(path)).with_overrides(plan.config_overrides)


@pytest.mark.parametrize("model_id", ["qwen3.5-9b", "qwen3.6-27b", "qwen3.8-27b", "qwen3.6-35b-a3b", "qwen3.5-122b-a10b"])
def test_default_target_is_identical_with_or_without_mtp(tmp_path, model_id):
    source = source_for(tmp_path, model_id)
    kwargs = dict(model_id=model_id, model_dir=str(source), output_dir=str(tmp_path / "out"))
    base = effective(resolve_build_plan(**kwargs), tmp_path)
    mtp = effective(resolve_build_plan(**kwargs, speculative={"mode": "mtp"}), tmp_path)
    assert base.quant["head"] == mtp.quant["head"] == {"bits": 5}
    assert base.export["model"]["quant_scheme"] == mtp.export["model"]["quant_scheme"]
    assert mtp.quant["mtp"] == {("expert_bits" if "a3b" in model_id or "a10b" in model_id else "block_bits"): 4,
                                "group_size": 64, "algorithm": "inherit"}
    assert "mtp" not in base.quant


@pytest.mark.parametrize("sidecar", [False, True])
def test_mixed_checkpoint_and_lora_mtp_are_reused_without_search(tmp_path, sidecar):
    source = source_for(tmp_path, "qwen3.6-35b-a3b")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = json.loads((source / "config.json").read_text())
    config.update(quantization_config={"bits": 4, "provider": "auto-round", "quant_method": "gptq",
                  "meta": {"auto_mixed_precision": {"module_bits": {}}, "lm_head_quantization": {"bits": 5}}},
                  mtp_quantization={"scope": "routed_experts", "expert_bits": 4})
    if sidecar:
        (checkpoint / "quantize_config.json").write_text(json.dumps(config.pop("quantization_config")))
    (checkpoint / "config.json").write_text(json.dumps(config))
    lora = tmp_path / "lora"
    lora.mkdir()
    (lora / "adapter_config.json").write_text('{}')
    kwargs = dict(model_id="qwen3.6-35b-a3b", model_dir=str(source), output_dir=str(tmp_path / "out"),
                  speculative={"mode": "mtp"}, lora={"paths": [str(lora)]})
    with pytest.raises(ValueError, match="quant_checkpoint"):
        resolve_build_plan(**kwargs)
    plan = resolve_build_plan(**kwargs, quant_checkpoint=str(checkpoint))
    cfg = effective(plan, tmp_path)
    assert plan.variant == "lora_mtp" and not plan.low_memory
    assert plan.precision.weight == "wmix"
    assert plan.field_sources["precision"] == "quant_checkpoint"
    assert cfg.quant["head"]["bits"] == 5
    assert cfg.export["model"]["lora"]["path"] == [str(lora)]
    assert cfg.export["model"]["lora"]["w_schema"]["bits"] == 8
    assert cfg.export["model"]["lora"]["a_schema"]["bits"] == 16
    # The executor's public checkpoint branch must not call workflow.quant.
    assert plan.quant_checkpoint == str(checkpoint.resolve())

    # Each LoRA view has a self-contained manifest. Publishing it must not
    # rewrite embeddings already hashed for the containing artifact.
    from xhmodel_merak.model_build.manifest import write_child_manifests
    from xhmodel_merak.model_artifact import MANIFEST_NAME
    from xhmodel_merak.model_artifact.validation import read_manifest, validate_manifest_files
    from xhmodel_merak.model_artifact.files import file_sha256
    root = tmp_path / "package"
    child = root / "lora" / "adapter"
    child.mkdir(parents=True)
    (child / "golden_meta_info.json").write_text(json.dumps({"model_config": {"lora": {"path": ["source://lora/0"]}}}))
    (child / "quant_embedding.pt").write_bytes(b"previously normalized embedding")
    (child / "quant_embedding.bin").write_bytes(b"runtime embedding")
    embedding = {"pt": "lora/adapter/quant_embedding.pt", "bin": "lora/adapter/quant_embedding.bin",
                 "pt_sha256": file_sha256(child / "quant_embedding.pt"),
                 "bin_sha256": file_sha256(child / "quant_embedding.bin")}
    metadata = {"children": [{"path": "lora/adapter", "metadata": {
        "metadata_file": "golden_meta_info.json",
        "export_entries": {"quant_embedding": {"path": "quant_embedding.pt", "type": "embedding"}},
    }}]}
    write_child_manifests(root, plan, tmp_path / "effective.yaml", str(checkpoint), {"checked": True},
                          metadata, {"model": "source-fingerprint"}, [embedding])
    _, published = read_manifest(str(child))
    validate_manifest_files(child, published["artifact"])
    assert published["recipe"]["variant"] == "lora_mtp"
    assert published["quantization"]["checkpoint"]["source"] == "reused"
    assert published["artifact"]["embedding"][0]["pt"] == "quant_embedding.pt"
    assert file_sha256(child / "quant_embedding.pt") == embedding["pt_sha256"]
    assert (child / MANIFEST_NAME).is_file()
    assert published["artifact"]["export_meta_info"] == "export_meta_info.json"
    entries = json.loads((child / "export_meta_info.json").read_text())
    assert entries["quant_embedding_bin"] == {"path": "quant_embedding.bin", "type": "embedding"}
    from xhmodel_merak.model_artifact.export_info import publish_export_info, validate_export_info
    validate_export_info(child, entries, inventoried={item["path"] for item in published["artifact"]["files"]})
    (root / "golden_meta_info.json").write_text('{}')
    parent = {"metadata_file": "golden_meta_info.json",
              "export_entries": {"lora": {"adapter": {"quant_embedding": {
                  "path": embedding["pt"], "type": "embedding"}}}},
              "export_json_roots": {"adapter": "lora/adapter"}}
    publish_export_info(root, parent, embedding_metadata=[embedding])
    parent_entries = json.loads((root / "export_meta_info.json").read_text())
    assert parent_entries["lora"]["adapter"]["export_meta_info"] == {
        "path": "lora/adapter/export_meta_info.json", "type": "json"}


def test_lora_runtime_closure_keeps_draft_and_child_config(tmp_path):
    from xhmodel_merak.model_integrations.autoregressive.build import _retain_runtime_files
    child = tmp_path / "lora" / "a"
    (child / "mtp_draft_decode").mkdir(parents=True)
    (child / "mtp_draft_decode" / "weights.bin").write_bytes(b'packed head')
    (child / "merak_config.json").write_text('{}')
    (child / "golden_meta_info.json").write_text('{}')
    (tmp_path / "unused.log").write_text('temporary')
    _retain_runtime_files(tmp_path, {"lora_adapters": [{"meta_file": "lora/a/golden_meta_info.json"}]})
    assert (child / "mtp_draft_decode" / "weights.bin").read_bytes() == b'packed head'
    assert (child / "merak_config.json").exists()
    assert not (tmp_path / "unused.log").exists()
