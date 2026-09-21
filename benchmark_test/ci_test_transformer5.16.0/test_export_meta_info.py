"""Small real artifact contracts; no checkpoint export or numerical inference."""

import json
from types import SimpleNamespace

import pytest

from xhmodel_merak.errors import ModelError
from xhmodel_merak.model_artifact.export_info import publish_export_info, validate_export_info
from xhmodel_merak.model_integrations.autoregressive.export_info import describe_export
from xhmodel_merak.model_integrations.autoregressive.build import _retain_runtime_files
from xhmodel_merak.model_integrations.kokoro.build import describe_export as describe_kokoro
from xhmodel_merak.model_integrations.pi05.build import describe_export as describe_pi05


def _file(root, name, contents="{}"):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    return path


def _ar_meta(root):
    from test_model_build_executor import _write_minimal_onnx

    meta = {
        "prefill_hmonnx": "prefill/model.onnx",
        "decode_hmonnx": "decode/model.onnx",
        "hf_config": "hf_config",
        "quant_embedding": "quant_embedding.pt",
        "visual_config": {
            "hmonnx": "visual_m256/model.onnx",
            "gears": [
                {"hmonnx": "visual_m128/model.onnx"},
                {"hmonnx": "visual_m256/model.onnx"},
            ],
        },
    }
    for name in ("prefill", "decode", "visual_m128", "visual_m256"):
        path = _file(root, f"{name}/model.onnx")
        _write_minimal_onnx(path)
    _file(root, "hf_config/config.json")
    _file(root, "hf_config/tokenizer.json")
    _file(root, "quant_embedding.pt", "embedding fixture")
    return meta


def test_autoregressive_gears_lora_retain_and_relocate(tmp_path):
    root = tmp_path / "stage"
    meta = _ar_meta(root)
    adapters = []
    for name in ("checkpoint1", "checkpoint2"):
        child = _ar_meta(root / name)
        _file(root, f"{name}/golden_meta_info.json", json.dumps(child))
        adapters.append({"name": name, "meta_file": f"{name}/golden_meta_info.json"})
    meta["lora_adapters"] = adapters
    _file(root, "golden_meta_info.json", json.dumps(meta))
    _file(root, "unused.onnx", "not a graph")
    _retain_runtime_files(root, meta)
    assert not (root / "unused.onnx").exists()
    entries = describe_export(root, meta)
    assert set(entries) == {"prefill", "decode", "tokenizer", "quant_embedding", "visual_m128", "visual_m256", "lora"}
    assert entries["visual_m256"]["type"] == "hmonnx"
    assert entries["lora"]["checkpoint2"]["quant_embedding"] == {
        "path": "checkpoint2/quant_embedding.pt",
        "type": "embedding",
    }
    metadata = {
        "metadata_file": "golden_meta_info.json",
        "export_entries": entries,
        "export_json_roots": {name: name for name in ("checkpoint1", "checkpoint2")},
    }
    publish_export_info(root, metadata)
    assert metadata["metadata_file"] == "golden_meta_info.json"
    assert "export_json_roots" not in metadata
    assert entries["golden_meta_info"] == {"path": "golden_meta_info.json", "type": "json"}
    assert entries["hf_config__config"] == {"path": "hf_config/config.json", "type": "json"}
    for name in ("checkpoint1", "checkpoint2"):
        assert entries["lora"][name]["golden_meta_info"] == {
            "path": f"{name}/golden_meta_info.json",
            "type": "json",
        }
        assert entries["lora"][name]["hf_config__config"]["path"] == f"{name}/hf_config/config.json"
        assert f"{name}__golden_meta_info" not in entries
    moved = tmp_path / "moved"
    root.rename(moved)
    validate_export_info(moved, json.loads((moved / "export_meta_info.json").read_text()))


def test_kokoro_mixed_host_and_npu_inventory(tmp_path):
    from _kokoro_fixture import write_release

    meta = write_release(tmp_path / "release")
    root = tmp_path / "release"
    entries = describe_kokoro(meta)
    assert len(entries) == 17
    assert sum(item["type"] == "onnx" for item in entries.values()) == 4
    assert sum(item["type"] == "hmonnx" for item in entries.values()) == 12
    original = (root / "export_meta_info.json").read_bytes()
    metadata = {"metadata_file": "export_meta_info.json", "export_entries": entries}
    publish_export_info(root, metadata)
    assert metadata["metadata_file"] == "runtime_meta_info.json"
    assert (root / "runtime_meta_info.json").read_bytes() == original
    assert entries["runtime_meta_info"] == {"path": "runtime_meta_info.json", "type": "json"}
    assert "export_meta_info" not in entries
    assert json.loads((root / "export_meta_info.json").read_text()) == entries


def test_pi05_uses_component_metadata_and_actual_graph_roles(tmp_path):
    meta = {
        "components": {
            "vision": {"component_dir": "visual", "hmonnx_file": "vision.onnx"},
            "gemma": {"component_dir": "gemma"},
            "expert": {"component_dir": "expert"},
            "other": {
                "component_dir": "action",
                "graphs": [
                    {"name": name, "hmonnx_file": f"{name}.onnx"}
                    for name in ("action_in_proj", "action_out_proj", "time_mlp")
                ],
            },
        },
        "runtime": {"tokenizer": "hf_config", "token_embedding": "quant_embedding.pt"},
    }
    nested = {name: {"prefill_onnx_file": "p.onnx", "decode_onnx_file": "d.onnx"} for name in ("gemma", "expert")}
    entries = describe_pi05(meta, nested)
    for entry in entries.values():
        _file(tmp_path, entry["path"] + "/config.json" if entry["type"] == "tokenizer" else entry["path"])
    validate_export_info(tmp_path, entries)
    assert entries["expert_decode"] == {"path": "expert/d.onnx", "type": "decode"}
    assert entries["action_out_proj"]["type"] == "hmonnx"


def test_moss_packages_hf_resources_and_publishes_encode(tmp_path):
    from test_moss_transcribe_diarize_runtime import _rotary_artifact_meta
    from xhmodel_merak.model_integrations.moss_transcribe_diarize import build
    from xhmodel_merak.model_integrations.moss_transcribe_diarize.artifact import (
        REQUIRED_RUNTIME_FILES,
        validate_runtime_contract,
    )
    from xhmodel_merak.model_artifact.files import confined_file

    source, root = tmp_path / "source", tmp_path / "stage"
    for name in REQUIRED_RUNTIME_FILES:
        _file(source, name)
    _file(root, "graph.onnx")
    _file(root, "weights.pt")
    _file(root, "export_meta_info.json", json.dumps(_rotary_artifact_meta()))
    plan = SimpleNamespace(model_dir=str(source), model_id="moss", components=("encode", "prefill", "decode"))
    build.package_runtime(root, plan)
    metadata = build.validate_artifact(root, plan)
    assert metadata["export_entries"]["encode"]["type"] == "hmonnx"
    assert metadata["export_entries"]["tokenizer"]["path"] == "hf_config"
    publish_export_info(root, metadata)
    for name in REQUIRED_RUNTIME_FILES:
        assert (root / "hf_config" / name).is_file()
        assert not (root / name).exists()
        assert (source / name).is_file()
    validate_runtime_contract(
        json.loads((root / metadata["metadata_file"]).read_text()),
        resolve_file=lambda value, field: confined_file(root.resolve(), value, field),
    )


def test_speculative_graph_aliases_have_one_entry_each(tmp_path):
    meta = _ar_meta(tmp_path)
    prefill, decode = "mtp_draft_prefill/model.onnx", "mtp_draft_decode/model.onnx"
    _file(tmp_path, prefill)
    _file(tmp_path, decode)
    meta["spec_decode"] = {
        "mode": "mtp",
        "draft_prefill_onnx": prefill,
        "draft_decode_onnx": decode,
        "draft": {"prefill_hmonnx": prefill, "decode_hmonnx": decode},
    }
    entries = describe_export(tmp_path, meta)
    assert entries["mtp_draft_prefill"] == {"path": prefill, "type": "prefill"}
    assert entries["mtp_draft_decode"] == {"path": decode, "type": "decode"}
    assert len(entries) == 8
    meta["spec_decode"] = {"mode": "mtp", "draft_onnx": decode}
    entries = describe_export(tmp_path, meta)
    assert entries["mtp_draft_prefill"] == {"path": decode, "type": "prefill"}
    assert entries["mtp_draft_decode"] == {"path": decode, "type": "decode"}


@pytest.mark.parametrize(
    "path",
    ["../outside.onnx", "/tmp/file.onnx", "C:graph.onnx", "a\\b.onnx", "a/../b.onnx", "", "a//b.onnx", "./model.onnx"],
)
def test_reject_nonportable_paths(tmp_path, path):
    with pytest.raises(ModelError):
        validate_export_info(tmp_path, {"encode": {"path": path, "type": "hmonnx"}})


@pytest.mark.parametrize(
    "entry",
    [
        {"path": "model.onnx", "type": "encode"},
        {"path": "model.onnx"},
        {"path": "model.onnx", "type": "onnx", "shape": []},
    ],
)
def test_reject_invalid_schema(tmp_path, entry):
    _file(tmp_path, "model.onnx")
    with pytest.raises(ModelError):
        validate_export_info(tmp_path, {"encode": entry})


def test_missing_symlink_uninventoried_and_collision(tmp_path):
    entries = {"encode": {"path": "model.onnx", "type": "hmonnx"}}
    with pytest.raises(ModelError):
        validate_export_info(tmp_path, entries)
    target = _file(tmp_path, "real.onnx")
    (tmp_path / "model.onnx").symlink_to(target)
    with pytest.raises(ModelError):
        validate_export_info(tmp_path, entries)
    (tmp_path / "model.onnx").unlink()
    _file(tmp_path, "model.onnx")
    with pytest.raises(ModelError):
        validate_export_info(tmp_path, entries, inventoried=set())
    _file(tmp_path, "export_meta_info.json", "original")
    _file(tmp_path, "runtime_meta_info.json", "existing")
    with pytest.raises(ModelError, match="collision"):
        publish_export_info(tmp_path, {"metadata_file": "export_meta_info.json", "export_entries": entries})
    assert (tmp_path / "export_meta_info.json").read_text() == "original"


def test_json_inventory_includes_nested_metadata_and_configuration(tmp_path):
    names = (
        "golden_meta_info.json",
        "merak_config.json",
        "visual_gears.json",
        "Gemma2B/export_meta_info.json",
        "runtime/policy/config.json",
        "hf_config/config.json",
    )
    for name in names:
        _file(tmp_path, name)
    _file(tmp_path, "model_build_manifest.json")
    _file(tmp_path, "effective_workflow.yaml")
    info = {"tokenizer": {"path": "hf_config", "type": "tokenizer"}}
    publish_export_info(tmp_path, {"metadata_file": "golden_meta_info.json", "export_entries": info})
    assert {entry["path"] for entry in info.values() if entry["type"] == "json"} == set(names)
    validate_export_info(tmp_path, info, inventoried=set(names))
    with pytest.raises(ModelError, match="outside the Manifest"):
        validate_export_info(tmp_path, info, inventoried=set(names) - {"merak_config.json"})
    (tmp_path / "merak_config.json").unlink()
    with pytest.raises(ModelError, match="missing"):
        validate_export_info(tmp_path, info)


@pytest.mark.parametrize("path,content", [("bad.json", "not json"), ("bad.txt", "{}")])
def test_json_entries_reject_invalid_format(tmp_path, path, content):
    _file(tmp_path, path, content)
    with pytest.raises(ModelError, match="JSON"):
        validate_export_info(tmp_path, {"metadata": {"path": path, "type": "json"}})


def test_json_names_cannot_overwrite_component_entries(tmp_path):
    _file(tmp_path, "runtime_meta_info.json")
    _file(tmp_path, "model.onnx")
    with pytest.raises(ModelError, match="collision"):
        publish_export_info(
            tmp_path,
            {
                "metadata_file": "runtime_meta_info.json",
                "export_entries": {"runtime_meta_info": {"path": "model.onnx", "type": "hmonnx"}},
            },
        )
    assert not (tmp_path / "export_meta_info.json").exists()


def test_generated_embedding_bins_cover_base_vla_and_lora(tmp_path):
    import torch
    from xhmodel_merak.model_build.embeddings import write_embedding_bins

    paths = ("quant_embedding.pt", "Gemma2B/token_embedding.pt", "checkpoint1/quant_embedding.pt")
    tensor = torch.tensor([[1.0, -2.0], [3.0, 0.5]])
    for value in paths:
        path = tmp_path / value
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"weight": tensor}, path)
    metadata = write_embedding_bins(tmp_path)
    info = {
        "quant_embedding": {"path": paths[0], "type": "embedding"},
        "vla_embedding": {"path": paths[1], "type": "embedding"},
        "lora": {"checkpoint1": {"quant_embedding": {"path": paths[2], "type": "embedding"}}},
    }
    _file(tmp_path, "runtime_meta_info.json")
    publish_export_info(
        tmp_path, {"metadata_file": "runtime_meta_info.json", "export_entries": info}, embedding_metadata=metadata
    )
    published = json.loads((tmp_path / "export_meta_info.json").read_text())
    for entries, name, value in (
        (published, "quant_embedding", paths[0]),
        (published, "vla_embedding", paths[1]),
        (published["lora"]["checkpoint1"], "quant_embedding", paths[2]),
    ):
        binary = value.removesuffix(".pt") + ".bin"
        assert entries[name] == {"path": value, "type": "embedding"}
        assert entries[name + "_bin"] == {"path": binary, "type": "embedding"}
        assert (tmp_path / binary).read_bytes() == tensor.numpy().astype("<f2").tobytes()
    validate_export_info(tmp_path, published)


@pytest.mark.parametrize("mode", ["not_generated", "missing", "collision"])
def test_embedding_bin_inventory_only_accepts_generated_existing_files(tmp_path, mode):
    _file(tmp_path, "quant_embedding.pt")
    _file(tmp_path, "runtime_meta_info.json")
    info = {"quant_embedding": {"path": "quant_embedding.pt", "type": "embedding"}}
    metadata = [{"pt": "quant_embedding.pt", "bin": "quant_embedding.bin"}]
    if mode != "missing":
        _file(tmp_path, "quant_embedding.bin")
    if mode == "collision":
        info["quant_embedding_bin"] = {"path": "quant_embedding.pt", "type": "embedding"}
    descriptor = {"metadata_file": "runtime_meta_info.json", "export_entries": info}
    if mode == "not_generated":
        publish_export_info(tmp_path, descriptor)
        assert "quant_embedding_bin" not in json.loads((tmp_path / "export_meta_info.json").read_text())
    else:
        with pytest.raises(ModelError):
            publish_export_info(tmp_path, descriptor, embedding_metadata=metadata)
        assert not (tmp_path / "export_meta_info.json").exists()
