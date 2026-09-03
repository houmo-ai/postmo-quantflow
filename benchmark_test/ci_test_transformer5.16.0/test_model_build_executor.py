import hashlib
import json
import os
import socket
import sys
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import yaml
from safetensors.torch import save_file as save_safetensors

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_build.checkpoint import inspect_quant_artifact, verify_quant_model_identity
from xhmodel_merak.model_build.contracts import BuildResult, ResolvedPrecision
from xhmodel_merak.model_build.embeddings import write_embedding_bins
from xhmodel_merak.model_build.executor import execute_build
from xhmodel_merak.model_build.fingerprints import verify_resolved_source_bindings, verify_resource_fingerprints
from xhmodel_merak.model_build.manifest import file_manifest
from xhmodel_merak.model_build.packaging import validate_onnx_external_locations
from xhmodel_merak.model_build.resolver import _resource_sha256, resolve_build_plan
from xhmodel_merak.model_build.workflow import resolve_quant_result
from xhmodel_merak.model_build.workspace import (
    acquire_output_lock,
    clear_owned_directory,
    publish,
    release_output_lock,
    remove_owned_directory,
    rename_noreplace,
    reserve_owned_directory,
)
from xhmodel_merak.model_catalog import get_recipe, resolve_workflow_config
from xhmodel_merak.xh_llm.workflows.base import BaseLLMWorkflow
from xhmodel_merak.xh_llm.workflows.result import QuantResult


@pytest.fixture(autouse=True)
def _skip_worker_dependency_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "xhmodel_merak.model_build.executor.validate_worker_requirements",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "xhmodel_merak.model_build.executor.validate_worker_capabilities",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "xhmodel_merak.model_inference.api.validate_worker_requirements",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "xhmodel_merak.model_inference.api.validate_worker_capabilities",
        lambda *_args, **_kwargs: None,
    )


def _write_minimal_onnx(path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    input_value = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])
    output_value = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "test_graph",
        [input_value],
        [output_value],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)


def _checkpoint_tensors(
    num_hidden_layers: int,
    *,
    hidden_size: int = 4096,
    value: float = 1.0,
) -> dict[str, torch.Tensor]:
    tensors = {
        "model.embed_tokens.weight": torch.full((2, hidden_size), value, dtype=torch.float16),
        "model.norm.weight": torch.full((hidden_size,), value, dtype=torch.float16),
    }
    for layer in range(num_hidden_layers):
        tensors[f"model.layers.{layer}.self_attn.qweight"] = torch.full((2, 2), value, dtype=torch.float16)
        tensors[f"model.layers.{layer}.mlp.down_proj.qweight"] = torch.full((2, 2), value, dtype=torch.float16)
        tensors[f"model.layers.{layer}.input_layernorm.weight"] = torch.full((hidden_size,), value, dtype=torch.float16)
    return tensors


class FakeWorkflow:
    def __init__(self, model_dir: str, config_path: str, *, export_gears=True):
        self.model_dir = model_dir
        self.config_path = config_path
        self.export_gears = export_gears
        self.quant_calls = []

    def quant(self, output_dir, device, config_overrides=None):
        self.quant_calls.append((output_dir, device, config_overrides))
        return SimpleNamespace(raw_model_dir=self.model_dir, skipped=True, quanted_model_dir=None)

    def export(self, quant_result, output_dir, device, config_overrides=None):
        output = Path(output_dir)
        output.mkdir(exist_ok=True)
        config_file = output / "workflow_override.yaml"
        config_file.write_text(Path(self.config_path).read_text(encoding="utf-8"), encoding="utf-8")

        legacy_root = output / "hmquant_qwen3_5_9b_448x448_20260825"
        legacy_root.mkdir()
        (legacy_root / "hf_config").mkdir()
        hf_config = json.loads((Path(self.model_dir) / "config.json").read_text(encoding="utf-8"))
        hf_config["_name_or_path"] = self.model_dir
        (legacy_root / "hf_config" / "config.json").write_text(json.dumps(hf_config), encoding="utf-8")
        (legacy_root / "hf_config" / "tokenizer_config.json").write_text(
            json.dumps({"asset_source": "raw-model"}),
            encoding="utf-8",
        )
        torch.save({"weight": torch.zeros((2, 4096), dtype=torch.float16)}, legacy_root / "quant_embedding.pt")
        _write_minimal_onnx(legacy_root / "prefill.hmonnx")
        _write_minimal_onnx(legacy_root / "decode.hmonnx")
        visual_gears = (96, 196, 384, 704, 1536)
        visual_config = (
            {
                "visual_input_mode": "patches",
                "spatial_merge_size": 2,
                "image_token_capacity": 1536,
                "image_token_gears": list(visual_gears),
                "routing_policy": "smallest_fit",
                "gear_manifest": "visual_gears.json",
                "hmonnx": f"visual_m1536/{legacy_root.name}_visual_m1536_with_act.onnx",
                "gears": [
                    {
                        "image_token_capacity": gear,
                        "patch_token_capacity": gear * 4,
                        "hmonnx": f"visual_m{gear}/{legacy_root.name}_visual_m{gear}_with_act.onnx",
                    }
                    for gear in visual_gears
                ],
            }
            if self.export_gears
            else {"image_size_w": 448}
        )
        for gear in visual_gears:
            (legacy_root / f"visual_m{gear}").mkdir()
            _write_minimal_onnx(legacy_root / f"visual_m{gear}/{legacy_root.name}_visual_m{gear}_with_act.onnx")
        if self.export_gears:
            (legacy_root / "visual_gears.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "visual_input_mode": "patches",
                        "token_unit": "post_spatial_merge",
                        "routing_policy": "smallest_fit",
                        "overflow_policy": "reject",
                        "gears": visual_config["gears"],
                    }
                ),
                encoding="utf-8",
            )
        (legacy_root / "golden_meta_info.json").write_text(
            json.dumps(
                {
                    "model_config": {
                        "hf_model": self.model_dir,
                        "context_max_length": 2048,
                        "prefill_chunk_length": 256,
                        "max_pe_length": 262144,
                    },
                    "hf_config": "hf_config",
                    "quant_embedding": "quant_embedding.pt",
                    "prefill_hmonnx": "prefill.hmonnx",
                    "decode_hmonnx": "decode.hmonnx",
                    "visual_config": visual_config,
                }
            ),
            encoding="utf-8",
        )
        (legacy_root / "merak_config.json").write_text(
            json.dumps({"model_type": "merak_llm"}),
            encoding="utf-8",
        )
        (legacy_root / "hmquant_qwen3_5_9b_448x448_20260825_weights.bin").write_bytes(b"weights")
        return SimpleNamespace(work_dir=str(output), config_file=str(config_file), meta={})


def _base_export_workflow(monkeypatch: pytest.MonkeyPatch, model_dir: Path) -> BaseLLMWorkflow:
    class FakeWorkflowConfig:
        name = "production_contract"

        def with_overrides(self, _overrides):
            return self

        def dump(self, path):
            Path(path).write_text("export: {}\n", encoding="utf-8")

    class FakeConfig:
        def __init__(self, value):
            self.model = value["model"]
            self.pretty_text = repr(value)

    class FakeModelConfig:
        @staticmethod
        def to_json_string():
            return "{}"

    class FakeModel:
        def export_hmonnx(self, output_dir):
            (Path(output_dir) / "exported.hmonnx").write_bytes(b"graph")
            return {"exported": True}

    api_module = ModuleType("xhquant.api")
    api_module.Config = FakeConfig
    api_module.get_xhquant_logger = lambda: SimpleNamespace(info=lambda *_args, **_kwargs: None)
    api_module.set_random_seed = lambda *_args, **_kwargs: None
    api_module.xhquant_init = lambda *_args, **_kwargs: None
    utils_module = ModuleType("xhquant.utils")
    utils_module.MemoryTracker = lambda *_args, **_kwargs: nullcontext()
    utils_module.TimeProfiler = lambda *_args, **_kwargs: nullcontext()
    monkeypatch.setitem(sys.modules, "xhquant.api", api_module)
    monkeypatch.setitem(sys.modules, "xhquant.utils", utils_module)

    import xhmodel_merak.xh_llm as xh_llm

    monkeypatch.setattr(
        xh_llm,
        "AutoLLMConfig",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: FakeModelConfig()),
    )
    monkeypatch.setattr(
        xh_llm,
        "AutoLLMModel",
        SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: FakeModel()),
    )

    workflow = BaseLLMWorkflow.__new__(BaseLLMWorkflow)
    workflow.workflow_config = FakeWorkflowConfig()
    workflow.model_dir = str(model_dir)
    workflow.seed = 1024
    workflow.debug = False
    workflow._build_export_config = lambda *_args: {"model": {}}
    return workflow


def test_base_llm_export_accepts_only_an_executor_reserved_empty_directory(monkeypatch, tmp_path):
    workflow = _base_export_workflow(monkeypatch, tmp_path / "model")
    quant_result = QuantResult(raw_model_dir=workflow.model_dir, skipped=True)
    stage = tmp_path / "stage"
    identity = reserve_owned_directory(stage, SimpleNamespace(model_id="contract-test"), "stage_dir")
    try:
        result = workflow.export(quant_result, str(stage), "cpu")
        assert Path(result.config_file).is_file()
        assert (stage / "exported.hmonnx").is_file()
    finally:
        remove_owned_directory(stage, identity)

    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "sentinel").write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not an empty directory"):
        workflow.export(quant_result, str(occupied), "cpu")

    regular_file = tmp_path / "regular-file"
    regular_file.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not an empty directory"):
        workflow.export(quant_result, str(regular_file), "cpu")


@pytest.fixture
def source_and_quant(tmp_path):
    source = tmp_path / "source"
    quant = tmp_path / "quant"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {
                    "hidden_size": 4096,
                    "num_hidden_layers": 32,
                    "rope_theta": 1_000_000.0,
                },
            }
        ),
        encoding="utf-8",
    )
    quant.mkdir()
    (quant / "quantization_config.json").write_text(
        json.dumps({"bits": 4, "provider": "auto-round", "quant_method": "gptq"}),
        encoding="utf-8",
    )
    quant_config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    quant_config["_name_or_path"] = str(source)
    quant_config["transformers_version"] = "quantizer-version"
    quant_config["quantization_config"] = {"group_size": 128}
    (quant / "config.json").write_text(json.dumps(quant_config), encoding="utf-8")
    (quant / "tokenizer.json").write_text(json.dumps({"version": "1.0"}), encoding="utf-8")
    (quant / "tokenizer_config.json").write_text(
        json.dumps({"asset_source": "quant-checkpoint"}),
        encoding="utf-8",
    )
    (quant / "processor_config.json").write_text(
        json.dumps({"processor_class": "TestProcessor"}),
        encoding="utf-8",
    )
    (quant / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    save_safetensors(_checkpoint_tensors(32), quant / "model.safetensors")
    return source, quant


def _plan(source: Path, quant: Path, output: Path):
    return resolve_build_plan(
        model_id="qwen3.5-9b",
        model_dir=str(source),
        output_dir=str(output),
        quant_checkpoint=str(quant),
    )


def test_executor_materializes_dynamic_vit_flattens_legacy_root_and_writes_manifest(
    monkeypatch,
    source_and_quant,
    tmp_path,
):
    source, quant = source_and_quant
    output = tmp_path / "artifact"
    captured = {}

    def create_workflow(model_dir, config_path):
        captured["effective"] = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        return FakeWorkflow(model_dir, config_path)

    monkeypatch.setattr("xhmodel_merak.model_build.workflow.create_workflow", create_workflow)
    monkeypatch.setattr(
        "xhmodel_merak.model_integrations.autoregressive.build._validate_tokenizer_snapshot",
        lambda *_args, **_kwargs: None,
    )

    plan = _plan(source, quant, output)
    result = execute_build(plan)

    visual = captured["effective"]["export"]["model"]["visual_config"]
    model_config = captured["effective"]["export"]["model"]
    assert visual["visual_input_mode"] == "patches"
    assert visual["image_token_gears"] == [96, 196, 384, 704, 1536]
    assert model_config["fuse_gdr_ops"] is True
    assert model_config["fuse_gdr_block_recurrent_ops"] is True
    assert result.output_dir == str(output.resolve())
    assert not any(path.is_dir() for path in output.glob("hmquant_*"))
    assert (output / "prefill.hmonnx").is_file()
    assert (output / f"{plan.artifact_name}_weights.bin").read_bytes() == b"weights"
    assert not any("448x448" in path.name or "20260825" in path.name for path in output.rglob("*"))
    assert (output / "effective_workflow.yaml").is_file()
    assert not (output / "workflow_override.yaml").exists()
    assert not (output / "recipe_identity.json").exists()
    assert json.loads((output / "hf_config/tokenizer_config.json").read_text(encoding="utf-8")) == {
        "asset_source": "quant-checkpoint"
    }
    assert (output / "hf_config/tokenizer.json").is_file()
    assert (output / "hf_config/processor_config.json").is_file()
    assert (output / "hf_config/chat_template.jinja").is_file()

    manifest = json.loads((output / "model_build_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert manifest["model_id"] == result.model_id
    assert manifest["format"] == {"name": "autoregressive", "version": 1}
    assert "runtime" not in manifest
    assert "model_name" not in result.to_dict()
    assert "artifact_name" not in result.to_dict()
    assert manifest["export"]["artifact_name"] == plan.artifact_name
    assert manifest["parameters"]["vision_profile"] == "visualm96-196-384-704-1536"
    assert manifest["artifact"]["requires_source"] is False
    assert manifest["artifact"]["loadable"] is True
    assert result.loadable is True
    assert manifest["quantization"]["checkpoint"]["source"] == "reused"
    assert manifest["quantization"]["checkpoint"]["verification"]["weight_bits"] == 4
    assert manifest["quantization"]["checkpoint"]["verification"]["verification"] == "checkpoint_verified"
    assert manifest["artifact"]["dynamic_vision"]["routing_policy"] == "smallest_fit"
    assert manifest["artifact"]["dynamic_vision"]["gear_manifest"] == "visual_gears.json"
    meta = json.loads((output / "golden_meta_info.json").read_text())
    gear_manifest = json.loads((output / "visual_gears.json").read_text())
    for gear in manifest["artifact"]["dynamic_vision"]["gears"]:
        capacity = gear["image_token_capacity"]
        assert gear["hmonnx"] == f"visual_m{capacity}/{plan.artifact_name}_visual_m{capacity}_with_act.onnx"
        assert (output / gear["hmonnx"]).is_file()
    assert gear_manifest["gears"] == meta["visual_config"]["gears"]
    assert not (output / "visual").exists()
    assert not (output / "visual_meta_info.json").exists()
    assert [item["image_token_capacity"] for item in manifest["artifact"]["dynamic_vision"]["gears"]] == [
        96,
        196,
        384,
        704,
        1536,
    ]
    embedding = torch.load(output / "quant_embedding.pt", map_location="cpu", weights_only=True)["weight"]
    assert embedding.dtype == torch.float16
    assert embedding.is_contiguous()
    assert (output / "quant_embedding.bin").read_bytes() == embedding.numpy().astype("<f2").tobytes()
    assert "model_build_manifest.json" not in {item["path"] for item in manifest["artifact"]["files"]}
    published_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output.rglob("*")
        if path.is_file() and path.suffix in {".json", ".yaml"}
    )
    assert str(source.resolve()) not in published_text
    assert str(quant.resolve()) not in published_text

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        lambda *_args, **_kwargs: SimpleNamespace(
            output_text="ok",
            output_tokens=1,
            latency_s=0.1,
            tokens_per_second=10.0,
            spec_decode_mode=None,
            stats={},
        ),
    )
    from xhmodel_merak.model_inference import inference_model

    inference_result = inference_model(inputs={"prompt": "hello"}, artifact_dir=str(output))
    assert inference_result.output_text == "ok"


def test_manifest_uses_registry_logical_workflow_under_a_stripped_config_root(
    monkeypatch,
    source_and_quant,
    tmp_path,
):
    source, quant = source_and_quant
    logical_config = get_recipe("qwen3.5-9b").default_variant.workflow_config
    source_config = resolve_workflow_config(logical_config)
    config_root = tmp_path / "config-root"
    stripped_config = config_root / logical_config.removeprefix("configs_merak/")
    stripped_config.parent.mkdir(parents=True)
    stripped_config.write_bytes(source_config.read_bytes())
    monkeypatch.setenv("XHMODEL_MERAK_CONFIG_ROOT", str(config_root))
    plan = _plan(source, quant, tmp_path / "artifact")
    assert Path(plan.config_path) == stripped_config.resolve()
    assert plan.workflow_config == logical_config
    monkeypatch.setattr(
        "xhmodel_merak.model_build.workflow.create_workflow",
        lambda model_dir, config_path: FakeWorkflow(model_dir, config_path),
    )
    monkeypatch.setattr(
        "xhmodel_merak.model_integrations.autoregressive.build._validate_tokenizer_snapshot",
        lambda *_args, **_kwargs: None,
    )

    result = execute_build(plan)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["recipe"]["source_config"] == logical_config
    assert not (Path(result.output_dir) / "recipe_identity.json").exists()


def test_executor_removes_logs_raw_workflows_and_provenance_paths(
    monkeypatch,
    source_and_quant,
    tmp_path,
):
    source, quant = source_and_quant

    class LeakyWorkflow(FakeWorkflow):
        def export(self, *args, **kwargs):
            result = super().export(*args, **kwargs)
            export_root = next(Path(result.work_dir).glob("hmquant_*"))
            (export_root / "duplicate_workflow.yaml").write_text(
                Path(result.config_file).read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (export_root / "export_hmonnx.log").write_text(
                f"loading {source.resolve()} from {quant.resolve()}\n",
                encoding="utf-8",
            )
            (export_root / "qwen35_autoround_recipe_provenance.json").write_text(
                json.dumps({"command": ["--model", str(source.resolve())]}),
                encoding="utf-8",
            )
            (export_root / "runtime_notes.json").write_text(
                json.dumps({"message": f"model={source.resolve()}/config.json"}),
                encoding="utf-8",
            )
            return result

    monkeypatch.setattr(
        "xhmodel_merak.model_build.workflow.create_workflow",
        lambda model_dir, config_path: LeakyWorkflow(model_dir, config_path),
    )
    monkeypatch.setattr(
        "xhmodel_merak.model_integrations.autoregressive.build._validate_tokenizer_snapshot",
        lambda *_args, **_kwargs: None,
    )
    output = tmp_path / "artifact"

    execute_build(_plan(source, quant, output))

    assert list(output.rglob("*.log")) == []
    assert not any("provenance" in path.name.lower() for path in output.rglob("*"))
    assert [path.relative_to(output).as_posix() for path in output.rglob("*.yaml")] == ["effective_workflow.yaml"]
    notes = json.loads((output / "runtime_notes.json").read_text(encoding="utf-8"))
    assert notes["message"] == "model=source://model/config.json"
    published_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in output.rglob("*")
        if path.is_file() and path.suffix.lower() in {".json", ".jinja", ".yaml"}
    )
    assert str(source.resolve()) not in published_text
    assert str(quant.resolve()) not in published_text


def test_executor_rejects_invalid_utf8_published_text_with_a_source_path(
    monkeypatch,
    source_and_quant,
    tmp_path,
):
    source, quant = source_and_quant

    class InvalidTextWorkflow(FakeWorkflow):
        def export(self, *args, **kwargs):
            result = super().export(*args, **kwargs)
            export_root = next(Path(result.work_dir).glob("hmquant_*"))
            (export_root / "runtime_notes.txt").write_bytes(str(source.resolve()).encode("utf-8") + b"\xff")
            return result

    monkeypatch.setattr(
        "xhmodel_merak.model_build.workflow.create_workflow",
        lambda model_dir, config_path: InvalidTextWorkflow(model_dir, config_path),
    )
    monkeypatch.setattr(
        "xhmodel_merak.model_integrations.autoregressive.build._validate_tokenizer_snapshot",
        lambda *_args, **_kwargs: None,
    )
    output = tmp_path / "artifact"

    with pytest.raises(ModelError) as exc_info:
        execute_build(_plan(source, quant, output))

    assert exc_info.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED
    assert exc_info.value.details["file"] == "runtime_notes.txt"
    assert exc_info.value.details["reason"].startswith("UnicodeDecodeError:")
    assert not output.exists()


def test_executor_rejects_an_existing_output_directory(
    monkeypatch,
    source_and_quant,
    tmp_path,
):
    source, quant = source_and_quant
    output = tmp_path / "artifact"
    output.mkdir()
    sentinel = output / "unrelated.txt"
    sentinel.write_text("keep", encoding="utf-8")
    monkeypatch.setattr(
        "xhmodel_merak.model_build.workflow.create_workflow",
        lambda model_dir, config_path: FakeWorkflow(model_dir, config_path),
    )

    with pytest.raises(ModelError, match="must not already exist"):
        execute_build(_plan(source, quant, output))

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (output / "model_build_manifest.json").exists()


def test_executor_rejects_static_vit_artifacts_without_publishing_partial_output(
    monkeypatch,
    source_and_quant,
    tmp_path,
):
    source, quant = source_and_quant
    output = tmp_path / "artifact"
    monkeypatch.setattr(
        "xhmodel_merak.model_build.workflow.create_workflow",
        lambda model_dir, config_path: FakeWorkflow(model_dir, config_path, export_gears=False),
    )
    monkeypatch.setattr(
        "xhmodel_merak.model_integrations.autoregressive.build._validate_tokenizer_snapshot",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ModelError) as exc_info:
        execute_build(_plan(source, quant, output))

    assert exc_info.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED
    assert not output.exists()


def test_executor_rejects_quant_checkpoint_precision_mismatch(monkeypatch, source_and_quant, tmp_path):
    source, quant = source_and_quant
    (quant / "quantization_config.json").write_text(json.dumps({"bits": 8}), encoding="utf-8")
    output = tmp_path / "artifact"
    monkeypatch.setattr(
        "xhmodel_merak.model_build.workflow.create_workflow",
        lambda model_dir, config_path: FakeWorkflow(model_dir, config_path),
    )

    with pytest.raises(ModelError) as exc_info:
        execute_build(_plan(source, quant, output))

    assert exc_info.value.code == ErrorCode.QUANT_ARTIFACT_MISMATCH
    assert not output.exists()


def test_quant_inspection_verifies_all_mixed_precision_bit_widths(source_and_quant, tmp_path):
    source, quant = source_and_quant
    output = tmp_path / "artifact"
    mixed_plan = replace(
        _plan(source, quant, output),
        precision=ResolvedPrecision(
            weight="wmix",
            activation="amix",
            components={
                "attention": {"weight": "w8", "activation": "a8"},
                "experts": {"weight": "w4", "activation": "a8"},
            },
        ),
    )
    (quant / "quantization_config.json").write_text(
        json.dumps(
            {
                "bits": 8,
                "provider": "auto-round",
                "dynamic": {"experts": {"bits": 4}},
            }
        ),
        encoding="utf-8",
    )

    inspection = inspect_quant_artifact(str(quant), mixed_plan)

    assert inspection["weight_bits_present"] == [4, 8]
    assert inspection["region_weight_bits"] == {
        "attention": {"weight_bits": 8, "evidence": "default_with_explicit_overrides"},
        "experts": {"weight_bits": 4, "evidence": "explicit"},
    }
    assert inspection["weights"][0]["file"] == "model.safetensors"
    assert len(inspection["weights"][0]["sha256"]) == 64


def test_build_result_deep_copies_nested_precision_components():
    precision = ResolvedPrecision(
        weight="wmix",
        activation="amix",
        components={"attention": {"weight": "w8", "activation": "a8"}},
    )
    result = BuildResult(
        output_dir="artifact",
        model_id="test-model",
        task="text-generation",
        profile=None,
        manifest_file="manifest.json",
        effective_workflow_file="workflow.yaml",
        quant_artifact=None,
        precision=precision,
        deployment_precision={},
        quant_method=None,
        context_length=1,
        prefill_chunk_length=1,
        max_pe_length=1,
        vision_profile=None,
        vision_gears=(),
        components=(),
        loadable=True,
    )

    result.precision.components["attention"]["weight"] = "changed"

    assert precision.components["attention"]["weight"] == "w8"


def test_quant_inspection_rejects_incomplete_mixed_precision_metadata(source_and_quant, tmp_path):
    source, quant = source_and_quant
    mixed_plan = replace(
        _plan(source, quant, tmp_path / "artifact"),
        precision=ResolvedPrecision(
            weight="wmix",
            activation="amix",
            components={
                "attention": {"weight": "w8", "activation": "a8"},
                "experts": {"weight": "w4", "activation": "a8"},
            },
        ),
    )
    (quant / "quantization_config.json").write_text(
        json.dumps({"bits": 8, "provider": "auto-round"}),
        encoding="utf-8",
    )

    with pytest.raises(ModelError) as exc_info:
        inspect_quant_artifact(str(quant), mixed_plan)

    assert exc_info.value.code == ErrorCode.QUANT_ARTIFACT_MISMATCH


def test_quant_inspection_rejects_swapped_mixed_precision_regions(source_and_quant, tmp_path):
    source, quant = source_and_quant
    mixed_plan = replace(
        _plan(source, quant, tmp_path / "artifact"),
        precision=ResolvedPrecision(
            weight="wmix",
            activation="amix",
            components={
                "attention": {"weight": "w8", "activation": "a8"},
                "experts": {"weight": "w4", "activation": "a8"},
            },
        ),
    )
    (quant / "quantization_config.json").write_text(
        json.dumps(
            {
                "bits": 8,
                "provider": "auto-round",
                "dynamic": {
                    "attention": {"bits": 4},
                    "experts": {"bits": 8},
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ModelError) as exc_info:
        inspect_quant_artifact(str(quant), mixed_plan)

    assert exc_info.value.code == ErrorCode.QUANT_ARTIFACT_MISMATCH
    assert {item["component"] for item in exc_info.value.details["invalid_regions"]} == {
        "attention",
        "experts",
    }


def test_quant_inspection_requires_metadata_provider_and_weights(source_and_quant, tmp_path):
    source, quant = source_and_quant
    plan = _plan(source, quant, tmp_path / "artifact")

    model_config = (quant / "config.json").read_text(encoding="utf-8")
    (quant / "quantization_config.json").unlink()
    (quant / "config.json").unlink()
    with pytest.raises(ModelError, match="no supported quantization metadata"):
        inspect_quant_artifact(str(quant), plan)

    (quant / "config.json").write_text(model_config, encoding="utf-8")
    (quant / "quantization_config.json").write_text(json.dumps({"bits": 4}), encoding="utf-8")
    with pytest.raises(ModelError, match="AutoRound provider"):
        inspect_quant_artifact(str(quant), plan)

    (quant / "quantization_config.json").write_text(
        json.dumps({"bits": 4, "provider": "auto-round"}),
        encoding="utf-8",
    )
    (quant / "model.safetensors").unlink()
    with pytest.raises(ModelError, match="no checkpoint weight files"):
        inspect_quant_artifact(str(quant), plan)


def test_quant_inspection_rejects_weight_index_path_escape(source_and_quant, tmp_path):
    source, quant = source_and_quant
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"outside")
    (quant / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"weight": "../outside.safetensors"}}),
        encoding="utf-8",
    )

    with pytest.raises(ModelError) as exc_info:
        inspect_quant_artifact(str(quant), _plan(source, quant, tmp_path / "artifact"))

    assert exc_info.value.code == ErrorCode.QUANT_ARTIFACT_MISMATCH


def _install_fake_onnx(monkeypatch, location: str, **values: str) -> None:
    external_data = [SimpleNamespace(key="location", value=location)]
    external_data.extend(SimpleNamespace(key=key, value=value) for key, value in values.items())
    graph = SimpleNamespace(
        initializer=[SimpleNamespace(external_data=external_data)],
        sparse_initializer=[],
        node=[],
    )
    monkeypatch.setitem(
        sys.modules,
        "onnx",
        SimpleNamespace(
            checker=SimpleNamespace(check_model=lambda *_args, **_kwargs: None),
            load_model=lambda *_args, **_kwargs: SimpleNamespace(graph=graph),
        ),
    )


def test_onnx_external_data_must_stay_inside_artifact(monkeypatch, tmp_path):
    stage = tmp_path / "artifact"
    stage.mkdir()
    (stage / "model.onnx").write_bytes(b"onnx")
    outside = tmp_path / "weights.bin"
    outside.write_bytes(b"weights")
    _install_fake_onnx(monkeypatch, "../weights.bin")

    with pytest.raises(ModelError) as exc_info:
        validate_onnx_external_locations(stage)

    assert exc_info.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED
    assert exc_info.value.details["invalid_external_data"][0]["reason"].startswith("location must")


@pytest.mark.parametrize("location", ["./weights.bin", "nested//weights.bin", "C:weights.bin"])
def test_onnx_external_data_rejects_noncanonical_raw_locations(monkeypatch, tmp_path, location):
    stage = tmp_path / "artifact"
    stage.mkdir()
    (stage / "model.onnx").write_bytes(b"onnx")
    _install_fake_onnx(monkeypatch, location)

    with pytest.raises(ModelError) as exc_info:
        validate_onnx_external_locations(stage)

    assert exc_info.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED
    invalid = exc_info.value.details["invalid_external_data"][0]
    assert invalid == {
        "onnx": "model.onnx",
        "location": location,
        "reason": "location must be a portable relative Artifact path",
    }


def test_onnx_external_data_rejects_symlinks(monkeypatch, tmp_path):
    stage = tmp_path / "artifact"
    stage.mkdir()
    (stage / "model.onnx").write_bytes(b"onnx")
    outside = tmp_path / "weights.bin"
    outside.write_bytes(b"weights")
    (stage / "weights.bin").symlink_to(outside)
    _install_fake_onnx(monkeypatch, "weights.bin")

    with pytest.raises(ModelError) as exc_info:
        validate_onnx_external_locations(stage)

    assert exc_info.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED
    assert exc_info.value.details["symlinks"] == ["weights.bin"]


def test_onnx_external_data_accepts_a_regular_local_file(monkeypatch, tmp_path):
    stage = tmp_path / "artifact"
    stage.mkdir()
    (stage / "model.onnx").write_bytes(b"onnx")
    (stage / "weights.bin").write_bytes(b"weights")
    _install_fake_onnx(monkeypatch, "weights.bin")

    validate_onnx_external_locations(stage)


def test_onnx_external_data_accepts_a_nested_posix_path(monkeypatch, tmp_path):
    stage = tmp_path / "artifact"
    nested = stage / "nested"
    nested.mkdir(parents=True)
    (stage / "model.onnx").write_bytes(b"onnx")
    (nested / "weights.bin").write_bytes(b"weights")
    _install_fake_onnx(monkeypatch, "nested/weights.bin")

    validate_onnx_external_locations(stage)


def test_onnx_external_data_rejects_out_of_range_slices(monkeypatch, tmp_path):
    stage = tmp_path / "artifact"
    stage.mkdir()
    (stage / "model.onnx").write_bytes(b"onnx")
    (stage / "weights.bin").write_bytes(b"weights")
    _install_fake_onnx(monkeypatch, "weights.bin", offset="5", length="8")

    with pytest.raises(ModelError) as exc_info:
        validate_onnx_external_locations(stage)

    assert exc_info.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED
    assert "exceed" in exc_info.value.details["invalid_external_data"][0]["reason"]


def test_onnx_validation_rejects_malformed_graph_bytes(tmp_path):
    stage = tmp_path / "artifact"
    stage.mkdir()
    (stage / "golden_meta_info.json").write_text(
        json.dumps({"prefill_hmonnx": "model.HMONNX"}),
        encoding="utf-8",
    )
    (stage / "model.HMONNX").write_bytes(b"not-an-onnx-model")

    with pytest.raises(ModelError, match="not a valid ONNX/HMONNX"):
        validate_onnx_external_locations(stage)


def test_manifest_and_publish_reject_artifact_symlinks(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (stage / "linked.bin").symlink_to(outside)

    with pytest.raises(ModelError) as manifest_error:
        file_manifest(stage)
    assert manifest_error.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED

    output = tmp_path / "output"
    with pytest.raises(ModelError) as publish_error:
        publish(stage, output)
    assert publish_error.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED
    assert not output.exists()


@pytest.mark.parametrize("node_kind", ["fifo", "socket"])
def test_artifact_trees_reject_non_regular_files(tmp_path, node_kind):
    stage = tmp_path / "stage"
    stage.mkdir()
    special = stage / node_kind
    bound_socket = None
    if node_kind == "fifo":
        os.mkfifo(special)
    else:
        bound_socket = socket.socket(socket.AF_UNIX)
        bound_socket.bind(str(special))
    try:
        with pytest.raises(ModelError, match="only regular files and directories") as exc_info:
            file_manifest(stage)
    finally:
        if bound_socket is not None:
            bound_socket.close()

    assert exc_info.value.details["special_nodes"] == [node_kind]


def test_generated_quant_is_verified_before_export_and_cleaned_on_failure(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {"hidden_size": 4096, "num_hidden_layers": 32},
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "artifact"

    class InvalidGeneratedQuant:
        def __init__(self):
            self.quant_root = None
            self.export_called = False

        def quant(self, output_dir, **_kwargs):
            self.quant_root = Path(output_dir)
            checkpoint = self.quant_root / "checkpoint"
            checkpoint.mkdir(parents=True)
            (checkpoint / "quantization_config.json").write_text(
                json.dumps({"bits": 8, "provider": "auto-round"}),
                encoding="utf-8",
            )
            save_safetensors(
                _checkpoint_tensors(32),
                checkpoint / "model.safetensors",
            )
            return SimpleNamespace(quanted_model_dir=str(checkpoint), skipped=False)

        def export(self, **_kwargs):
            self.export_called = True
            pytest.fail("export must not consume an unverified Quant Artifact")

    workflow = InvalidGeneratedQuant()
    monkeypatch.setattr("xhmodel_merak.model_build.workflow.create_workflow", lambda *_args: workflow)
    plan = resolve_build_plan(model_id="qwen3.5-9b", model_dir=str(source), output_dir=str(output))

    with pytest.raises(ModelError) as exc_info:
        execute_build(plan)

    assert exc_info.value.code == ErrorCode.QUANT_ARTIFACT_MISMATCH
    assert workflow.export_called is False
    assert workflow.quant_root is not None and not workflow.quant_root.exists()
    assert not output.exists()


def test_generated_quant_must_stay_under_reserved_root(monkeypatch, source_and_quant, tmp_path):
    source, external = source_and_quant
    output = tmp_path / "artifact"

    class EscapingGeneratedQuant:
        def __init__(self):
            self.quant_root = None
            self.export_called = False

        def quant(self, output_dir, **_kwargs):
            self.quant_root = Path(output_dir)
            self.quant_root.mkdir(exist_ok=True)
            return SimpleNamespace(quanted_model_dir=str(external), skipped=False)

        def export(self, **_kwargs):
            self.export_called = True
            pytest.fail("export must not consume a Quant Artifact outside quant_output_dir")

    workflow = EscapingGeneratedQuant()
    monkeypatch.setattr("xhmodel_merak.model_build.workflow.create_workflow", lambda *_args: workflow)
    plan = resolve_build_plan(model_id="qwen3.5-9b", model_dir=str(source), output_dir=str(output))

    with pytest.raises(ModelError, match="outside quant_output_dir"):
        execute_build(plan)

    assert workflow.export_called is False
    assert workflow.quant_root is not None and not workflow.quant_root.exists()
    assert external.exists()


@pytest.mark.parametrize(
    "quant_result",
    [
        SimpleNamespace(skipped=False, quanted_model_dir=None),
        SimpleNamespace(quanted_model_dir=None),
        SimpleNamespace(skipped=True, quanted_model_dir="unexpected-checkpoint"),
    ],
)
def test_no_quant_path_requires_an_explicitly_skipped_result_without_checkpoint(
    quant_result,
    source_and_quant,
    tmp_path,
):
    source, quant = source_and_quant
    output = tmp_path / "artifact"
    plan = replace(_plan(source, quant, output), quant_method=None, quant_checkpoint=None)
    workflow = SimpleNamespace(quant=lambda **_kwargs: quant_result)

    with pytest.raises(ModelError, match="explicitly skipped") as exc_info:
        resolve_quant_result(workflow, plan, output)

    assert exc_info.value.code == ErrorCode.QUANT_ARTIFACT_MISMATCH
    assert not list(tmp_path.glob(".artifact.quant-skip-*"))


def test_resource_fingerprint_changes_and_empty_embeddings_are_rejected(tmp_path):
    resource = tmp_path / "resource"
    resource.mkdir()
    (resource / "tokenizer.model").write_bytes(b"before")
    expected = _resource_sha256(resource)
    plan = SimpleNamespace(
        model_id="pi0.5-libero",
        resources={"tokenizer": SimpleNamespace(path=str(resource), sha256=expected)},
    )
    (resource / "tokenizer.model").write_bytes(b"after")

    with pytest.raises(ModelError, match="changed"):
        verify_resource_fingerprints(plan)

    stage = tmp_path / "embedding-artifact"
    stage.mkdir()
    torch.save({"weight": torch.empty((0, 4), dtype=torch.float16)}, stage / "quant_embedding.pt")
    with pytest.raises(ModelError, match="dimensions must be positive"):
        write_embedding_bins(stage)


def test_embedding_hidden_dimension_uses_nested_text_config(tmp_path):
    stage = tmp_path / "artifact"
    (stage / "hf_config").mkdir(parents=True)
    (stage / "hf_config" / "config.json").write_text(
        json.dumps(
            {
                "vision_config": {"hidden_size": 8},
                "model_config": {"text_config": {"hidden_size": 4}},
            }
        ),
        encoding="utf-8",
    )
    torch.save({"weight": torch.ones((2, 4))}, stage / "quant_embedding.pt")

    metadata = write_embedding_bins(stage)

    assert metadata[0]["shape"] == [2, 4]


def test_embedding_hidden_dimension_rejects_mismatch_and_accepts_pi_component_metadata(tmp_path):
    invalid = tmp_path / "invalid"
    (invalid / "hf_config").mkdir(parents=True)
    (invalid / "hf_config" / "config.json").write_text(
        json.dumps({"text_config": {"hidden_size": 4}}),
        encoding="utf-8",
    )
    torch.save({"weight": torch.ones((2, 3))}, invalid / "quant_embedding.pt")

    with pytest.raises(ModelError, match="hidden dimension 3.*hidden_size=4"):
        write_embedding_bins(invalid)

    pi_stage = tmp_path / "pi-artifact"
    component = pi_stage / "Gemma2B"
    component.mkdir(parents=True)
    (component / "export_meta_info.json").write_text(json.dumps({"hidden_size": 4}), encoding="utf-8")
    torch.save({"weight": torch.ones((2, 4))}, component / "token_embedding.pt")

    assert write_embedding_bins(pi_stage)[0]["shape"] == [2, 4]


def test_executor_rejects_a_symlinked_stage_root_without_touching_its_target(
    monkeypatch,
    source_and_quant,
    tmp_path,
):
    source, quant = source_and_quant
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    class SymlinkStageWorkflow(FakeWorkflow):
        def export(self, quant_result, output_dir, device, config_overrides=None):
            stage = Path(output_dir)
            stage.rmdir()
            stage.symlink_to(outside, target_is_directory=True)
            return SimpleNamespace(work_dir=str(stage), config_file=str(sentinel), meta={})

    monkeypatch.setattr(
        "xhmodel_merak.model_build.workflow.create_workflow",
        lambda model_dir, config_path: SymlinkStageWorkflow(model_dir, config_path),
    )
    output = tmp_path / "artifact"

    with pytest.raises(ModelError, match="root must be a regular directory"):
        execute_build(_plan(source, quant, output))

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not output.exists()


def test_executor_rechecks_a_quant_checkpoint_after_export(monkeypatch, source_and_quant, tmp_path):
    source, quant = source_and_quant

    class MutatingQuantWorkflow(FakeWorkflow):
        def export(self, *args, **kwargs):
            save_safetensors(
                _checkpoint_tensors(32, value=0.0),
                quant / "model.safetensors",
            )
            return super().export(*args, **kwargs)

    monkeypatch.setattr(
        "xhmodel_merak.model_build.workflow.create_workflow",
        lambda model_dir, config_path: MutatingQuantWorkflow(model_dir, config_path),
    )
    monkeypatch.setattr(
        "xhmodel_merak.model_integrations.autoregressive.build._validate_tokenizer_snapshot",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ModelError, match="changed after its pre-export verification"):
        execute_build(_plan(source, quant, tmp_path / "artifact"))


def test_executor_rejects_source_weight_mutation_during_export(monkeypatch, source_and_quant, tmp_path):
    source, quant = source_and_quant
    weight = source / "model.safetensors"
    weight.write_bytes(b"before")

    class MutatingSourceWorkflow(FakeWorkflow):
        def export(self, *args, **kwargs):
            weight.write_bytes(b"after")
            return super().export(*args, **kwargs)

    monkeypatch.setattr(
        "xhmodel_merak.model_build.workflow.create_workflow",
        lambda model_dir, config_path: MutatingSourceWorkflow(model_dir, config_path),
    )
    monkeypatch.setattr(
        "xhmodel_merak.model_integrations.autoregressive.build._validate_tokenizer_snapshot",
        lambda *_args, **_kwargs: None,
    )

    with pytest.raises(ModelError, match="Build sources changed"):
        execute_build(_plan(source, quant, tmp_path / "artifact"))


def test_executor_rejects_model_directory_replacement_after_plan_resolution(
    monkeypatch,
    source_and_quant,
    tmp_path,
):
    source, quant = source_and_quant
    plan = _plan(source, quant, tmp_path / "artifact")
    source.rename(tmp_path / "resolved-qwen9")
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {"hidden_size": 2560, "num_hidden_layers": 32},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "xhmodel_merak.model_build.workflow.create_workflow",
        lambda *_args, **_kwargs: pytest.fail("a replacement source must not reach workflow execution"),
    )

    with pytest.raises(ModelError, match="identity or config changed") as exc_info:
        execute_build(plan)

    assert exc_info.value.details["changed_sources"][0]["source"] == "model"
    assert not Path(plan.output_dir).exists()


def test_additional_source_config_is_bound_at_plan_resolution(source_and_quant, tmp_path):
    source, quant = source_and_quant
    assistant = tmp_path / "assistant"
    assistant.mkdir()
    assistant_config = {
        "hidden_size": 4096,
        "num_target_layers": 32,
        "dflash_config": {"mask_token_id": 248070, "target_layer_ids": [1, 9, 17, 25]},
    }
    (assistant / "config.json").write_text(json.dumps(assistant_config), encoding="utf-8")
    plan = resolve_build_plan(
        model_id="qwen3.5-9b",
        model_dir=str(source),
        output_dir=str(tmp_path / "artifact"),
        quant_checkpoint=str(quant),
        speculative={"mode": "dflash"},
        sources={"dflash": str(assistant)},
    )
    assistant_config["dflash_config"]["mask_token_id"] = 248071
    (assistant / "config.json").write_text(json.dumps(assistant_config), encoding="utf-8")

    with pytest.raises(ModelError, match="identity or config changed") as exc_info:
        verify_resolved_source_bindings(plan)

    assert exc_info.value.details["changed_sources"][0]["source"] == "dflash"


def test_explicit_quant_output_cleanup_preserves_a_replacement_directory(monkeypatch, tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {"hidden_size": 4096, "num_hidden_layers": 32},
            }
        ),
        encoding="utf-8",
    )
    quant_output = tmp_path / "quant-output"

    class ReplacingQuantWorkflow:
        def quant(self, output_dir, **_kwargs):
            root = Path(output_dir)
            root.rmdir()
            root.mkdir()
            (root / "foreign.txt").write_text("keep", encoding="utf-8")
            raise RuntimeError("quant failed after replacement")

    monkeypatch.setattr(
        "xhmodel_merak.model_build.workflow.create_workflow",
        lambda *_args: ReplacingQuantWorkflow(),
    )
    plan = resolve_build_plan(
        model_id="qwen3.5-9b",
        model_dir=str(source),
        output_dir=str(tmp_path / "artifact"),
        quant_output_dir=str(quant_output),
    )

    with pytest.raises(ModelError, match="quant failed after replacement"):
        execute_build(plan)

    assert (quant_output / "foreign.txt").read_text(encoding="utf-8") == "keep"


def test_publish_rejects_a_replaced_stage_and_cleanup_preserves_it(tmp_path):
    stage = tmp_path / "stage"
    plan = SimpleNamespace(model_id="test-model")
    identity = reserve_owned_directory(stage, plan, "stage_dir")
    stage.rmdir()
    stage.mkdir()
    sentinel = stage / "foreign.txt"
    sentinel.write_text("keep", encoding="utf-8")

    try:
        with pytest.raises(ModelError, match="replaced before publication"):
            publish(stage, tmp_path / "output", identity)
    finally:
        remove_owned_directory(stage, identity)

    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert not (tmp_path / "output").exists()


def test_owned_cleanup_does_not_traverse_a_directory_swapped_after_quarantine(monkeypatch, tmp_path):
    owned = tmp_path / "owned"
    plan = SimpleNamespace(model_id="test-model")
    identity = reserve_owned_directory(owned, plan, "owned_dir")
    (owned / "payload.bin").write_bytes(b"owned")
    replacement = {}

    def swap_before_clear(descriptor):
        quarantine = next(tmp_path.glob(".owned.cleanup-*"))
        quarantine.rename(tmp_path / "displaced-owned")
        quarantine.mkdir()
        (quarantine / "foreign.txt").write_text("keep", encoding="utf-8")
        replacement["path"] = quarantine
        clear_owned_directory(descriptor)

    monkeypatch.setattr("xhmodel_merak.model_build.workspace.clear_owned_directory", swap_before_clear)

    remove_owned_directory(owned, identity)

    assert (replacement["path"] / "foreign.txt").read_text(encoding="utf-8") == "keep"


def test_executor_rejects_a_replaced_recipe_before_publish(monkeypatch, source_and_quant, tmp_path):
    source, quant = source_and_quant
    replacement = {}

    def create_workflow(model_dir, config_path):
        config = Path(config_path)
        contents = config.read_bytes()
        recipe = config.parent
        recipe.rename(tmp_path / "displaced-recipe")
        recipe.mkdir()
        config.write_bytes(contents)
        sentinel = recipe / "foreign.txt"
        sentinel.write_text("keep", encoding="utf-8")
        replacement["sentinel"] = sentinel
        return FakeWorkflow(model_dir, config_path)

    monkeypatch.setattr("xhmodel_merak.model_build.workflow.create_workflow", create_workflow)
    monkeypatch.setattr(
        "xhmodel_merak.model_integrations.autoregressive.build._validate_tokenizer_snapshot",
        lambda *_args, **_kwargs: None,
    )
    output = tmp_path / "artifact"

    with pytest.raises(ModelError, match="recipe_dir was replaced"):
        execute_build(_plan(source, quant, output))

    assert replacement["sentinel"].read_text(encoding="utf-8") == "keep"
    assert not output.exists()


def test_quant_provider_matching_is_exact(source_and_quant):
    source, quant = source_and_quant
    plan = _plan(source, quant, quant.parent / "unused")
    metadata = quant / "quantization_config.json"
    metadata.write_text(
        json.dumps({"bits": 4, "provider": "definitely-not-autoround", "quant_method": "gptq"}),
        encoding="utf-8",
    )

    with pytest.raises(ModelError, match="AutoRound provider"):
        inspect_quant_artifact(str(quant), plan)


@pytest.mark.parametrize(
    ("name", "payload"),
    [
        ("model.safetensors", b"not-safetensors"),
        ("pytorch_model.bin", b"PK\x03\x04truncated"),
        ("model.gguf", b"GGUF"),
    ],
)
def test_quant_inspection_rejects_malformed_weight_containers(source_and_quant, name, payload):
    source, quant = source_and_quant
    plan = _plan(source, quant, quant.parent / "unused")
    (quant / "model.safetensors").unlink(missing_ok=True)
    (quant / name).write_bytes(payload)

    with pytest.raises(ModelError, match="weight container is invalid"):
        inspect_quant_artifact(str(quant), plan)


def test_quant_inspection_rejects_a_single_scalar_safetensors_checkpoint(source_and_quant):
    source, quant = source_and_quant
    save_safetensors({"unrelated": torch.ones((1,), dtype=torch.float16)}, quant / "model.safetensors")

    with pytest.raises(ModelError, match="complete model checkpoint"):
        inspect_quant_artifact(str(quant), _plan(source, quant, quant.parent / "unused"))


@pytest.mark.parametrize("missing", ["embedding", "layer"])
def test_quant_inspection_requires_embedding_and_complete_source_layer_coverage(source_and_quant, missing):
    source, quant = source_and_quant
    tensors = _checkpoint_tensors(32)
    if missing == "embedding":
        del tensors["model.embed_tokens.weight"]
    else:
        for name in [name for name in tensors if name.startswith("model.layers.31.")]:
            del tensors[name]
    save_safetensors(tensors, quant / "model.safetensors")

    with pytest.raises(ModelError, match="complete model checkpoint") as exc_info:
        inspect_quant_artifact(str(quant), _plan(source, quant, quant.parent / "unused"))

    if missing == "embedding":
        assert exc_info.value.details["has_embedding"] is False
    else:
        assert exc_info.value.details["missing_layers"] == [31]


def test_quant_inspection_rejects_wrong_embedding_and_norm_shapes(source_and_quant):
    source, quant = source_and_quant
    tensors = _checkpoint_tensors(32)
    tensors["model.embed_tokens.weight"] = torch.ones((2, 8), dtype=torch.float16)
    tensors["model.layers.7.input_layernorm.weight"] = torch.ones((8,), dtype=torch.float16)
    save_safetensors(tensors, quant / "model.safetensors")

    with pytest.raises(ModelError, match="verifiably complete") as exc_info:
        inspect_quant_artifact(str(quant), _plan(source, quant, quant.parent / "unused"))

    mismatched_names = {item["tensor"] for item in exc_info.value.details["shape_mismatches"]}
    assert mismatched_names == {
        "model.embed_tokens.weight",
        "model.layers.7.input_layernorm.weight",
    }


def test_quant_inspection_requires_attention_mlp_and_norm_groups_in_every_layer(source_and_quant):
    source, quant = source_and_quant
    tensors = _checkpoint_tensors(32)
    del tensors["model.layers.9.mlp.down_proj.qweight"]
    save_safetensors(tensors, quant / "model.safetensors")

    with pytest.raises(ModelError, match="verifiably complete") as exc_info:
        inspect_quant_artifact(str(quant), _plan(source, quant, quant.parent / "unused"))

    assert exc_info.value.details["incomplete_layer_groups"] == {9: ["mlp_or_expert"]}


def test_quant_inspection_uses_only_shards_referenced_by_the_weight_index(source_and_quant):
    source, quant = source_and_quant
    (quant / "model.safetensors").unlink()
    save_safetensors(
        {"model.embed_tokens.weight": torch.ones((2, 2), dtype=torch.float16)},
        quant / "model-00001-of-00002.safetensors",
    )
    layer_tensors = _checkpoint_tensors(32)
    del layer_tensors["model.embed_tokens.weight"]
    save_safetensors(layer_tensors, quant / "model-00002-of-00002.safetensors")
    (quant / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ModelError, match="not referenced by its index") as exc_info:
        inspect_quant_artifact(str(quant), _plan(source, quant, quant.parent / "unused"))

    assert exc_info.value.details["unreferenced"] == ["model-00002-of-00002.safetensors"]


def test_quant_inspection_rejects_an_index_key_absent_from_its_declared_shard(source_and_quant):
    source, quant = source_and_quant
    tensors = _checkpoint_tensors(32)
    save_safetensors(tensors, quant / "model.safetensors")
    weight_map = {name: "model.safetensors" for name in tensors}
    weight_map["model.layers.0.self_attn.missing_qweight"] = "model.safetensors"
    (quant / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map}),
        encoding="utf-8",
    )

    with pytest.raises(ModelError, match="does not match the tensors") as exc_info:
        inspect_quant_artifact(str(quant), _plan(source, quant, quant.parent / "unused"))

    assert exc_info.value.details["missing_or_misplaced"][0]["tensor"].endswith("missing_qweight")


def test_quant_inspection_rejects_multiple_indexes_that_combine_incomplete_models(source_and_quant):
    source, quant = source_and_quant
    (quant / "model.safetensors").unlink()
    save_safetensors(
        {"model.embed_tokens.weight": torch.ones((2, 2), dtype=torch.float16)},
        quant / "model-primary.safetensors",
    )
    layer_tensors = _checkpoint_tensors(32)
    del layer_tensors["model.embed_tokens.weight"]
    save_safetensors(layer_tensors, quant / "adapter-layers.safetensors")
    (quant / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.embed_tokens.weight": "model-primary.safetensors"}}),
        encoding="utf-8",
    )
    (quant / "adapter_model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "adapter-layers.safetensors" for name in layer_tensors}}),
        encoding="utf-8",
    )

    with pytest.raises(ModelError, match="at most one authoritative weight index") as exc_info:
        inspect_quant_artifact(str(quant), _plan(source, quant, quant.parent / "unused"))

    assert exc_info.value.details["indexes"] == [
        "adapter_model.safetensors.index.json",
        "model.safetensors.index.json",
    ]


def test_quant_inspection_rejects_ambiguous_mixed_weight_formats(source_and_quant):
    source, quant = source_and_quant
    save_safetensors(
        {"model.embed_tokens.weight": torch.ones((2, 2), dtype=torch.float16)},
        quant / "model.safetensors",
    )
    torch.save(
        {"model.layers.0.self_attn.qweight": torch.ones((2, 2), dtype=torch.float16)},
        quant / "pytorch_model.bin",
    )

    with pytest.raises(ModelError, match="multiple weight containers requires one authoritative index"):
        inspect_quant_artifact(str(quant), _plan(source, quant, quant.parent / "unused"))


def test_quant_inspection_rejects_two_unrelated_safetensors_tensors(source_and_quant):
    source, quant = source_and_quant
    save_safetensors(
        {
            "unrelated.first": torch.ones((2, 2), dtype=torch.float16),
            "unrelated.second": torch.ones((2, 2), dtype=torch.float16),
        },
        quant / "model.safetensors",
    )

    with pytest.raises(ModelError, match="complete model checkpoint") as exc_info:
        inspect_quant_artifact(str(quant), _plan(source, quant, quant.parent / "unused"))

    assert exc_info.value.details["missing_layers"] == list(range(32))
    assert exc_info.value.details["has_embedding"] is False


def test_quant_inspection_rejects_a_structurally_valid_but_partial_torch_zip_checkpoint(source_and_quant):
    source, quant = source_and_quant
    (quant / "model.safetensors").unlink()
    torch.save(
        {
            "model.embed_tokens.weight": torch.ones((2, 2), dtype=torch.float16),
            "model.layers.0.self_attn.qweight": torch.ones((2, 2), dtype=torch.float16),
        },
        quant / "pytorch_model.bin",
    )

    with pytest.raises(ModelError, match="verifiably complete"):
        inspect_quant_artifact(str(quant), _plan(source, quant, quant.parent / "unused"))


def test_quant_inspection_verifies_tensor_names_and_shapes_in_a_complete_torch_zip(source_and_quant):
    source, quant = source_and_quant
    (quant / "model.safetensors").unlink()
    torch.save(_checkpoint_tensors(32), quant / "pytorch_model.bin")

    inspection = inspect_quant_artifact(str(quant), _plan(source, quant, quant.parent / "unused"))

    assert inspection["weights"][0]["file"] == "pytorch_model.bin"


def test_quant_inspection_rejects_checkpoint_source_model_identity_mismatch(source_and_quant):
    source, quant = source_and_quant
    (quant / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {"hidden_size": 2048, "num_hidden_layers": 32},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ModelError, match="does not match the source model config") as exc_info:
        inspect_quant_artifact(str(quant), _plan(source, quant, quant.parent / "unused"))

    assert exc_info.value.details["identity_mismatches"]["hidden_size"] == {
        "expected": 4096,
        "actual": 2048,
    }


def test_quant_checkpoint_allows_quantization_provenance_but_rejects_rope_theta_mismatch(source_and_quant):
    source, quant = source_and_quant
    plan = _plan(source, quant, quant.parent / "unused")

    inspection = inspect_quant_artifact(str(quant), plan)
    assert {item["file"] for item in inspection["runtime_assets"]} >= {
        "config.json",
        "processor_config.json",
        "tokenizer.json",
    }

    checkpoint_config = json.loads((quant / "config.json").read_text(encoding="utf-8"))
    checkpoint_config["text_config"]["rope_theta"] = 10_000.0
    (quant / "config.json").write_text(json.dumps(checkpoint_config), encoding="utf-8")

    with pytest.raises(ModelError, match="does not match the source model config") as exc_info:
        inspect_quant_artifact(str(quant), plan)

    assert exc_info.value.details["identity_mismatches"]["rope_theta"] == {
        "expected": 1_000_000.0,
        "actual": 10_000.0,
    }


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("num_experts", 128),
        ("router_aux_loss_coef", 0.0),
    ],
)
def test_quant_checkpoint_config_must_satisfy_installed_recipe_source_constraints(
    tmp_path,
    field,
    replacement,
):
    source = tmp_path / "source"
    source.mkdir()
    source_config = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "hidden_size": 2048,
            "num_hidden_layers": 40,
            "num_experts": 256,
            "router_aux_loss_coef": 0.001,
        },
    }
    (source / "config.json").write_text(json.dumps(source_config), encoding="utf-8")
    plan = resolve_build_plan(
        model_id="qwen3.6-35b-a3b",
        model_dir=str(source),
        output_dir=str(tmp_path / "artifact"),
    )
    checkpoint_config = json.loads(json.dumps(source_config))
    checkpoint_config["text_config"][field] = replacement
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config_path = checkpoint / "config.json"
    config_path.write_text(json.dumps(checkpoint_config), encoding="utf-8")

    with pytest.raises(ModelError, match="installed Registry source constraints") as exc_info:
        verify_quant_model_identity(checkpoint, [(config_path, checkpoint_config)], plan)

    assert exc_info.value.details["source_constraint_mismatches"] == [
        {
            "field": f"text_config.{field}",
            "expected": source_config["text_config"][field],
            "actual": replacement,
        }
    ]


def test_quant_inspection_requires_a_checkpoint_model_config(source_and_quant):
    source, quant = source_and_quant
    (quant / "config.json").unlink()

    with pytest.raises(ModelError, match="one loadable model config.json"):
        inspect_quant_artifact(str(quant), _plan(source, quant, quant.parent / "unused"))


def test_materialization_rejects_a_changed_registry_workflow(source_and_quant, tmp_path):
    source, quant = source_and_quant
    plan = _plan(source, quant, tmp_path / "artifact")
    config = tmp_path / "workflow.yaml"
    original = Path(plan.config_path).read_bytes()
    config.write_bytes(original)
    pinned = hashlib.sha256(original).hexdigest()
    changed_plan = replace(plan, config_path=str(config), source_config_sha256=pinned)
    config.write_bytes(original + b"\n# changed\n")

    with pytest.raises(ModelError, match="changed after BuildPlan resolution"):
        execute_build(changed_plan)


def test_output_lock_keeps_a_stable_inode_and_excludes_a_second_builder(tmp_path):
    output = tmp_path / "artifact"
    lock_path = tmp_path / ".artifact.xhbuild.lock"
    lock_path.write_text("pid=dead\n", encoding="ascii")
    initial_identity = (lock_path.stat().st_dev, lock_path.stat().st_ino)

    lock = acquire_output_lock(output)
    with pytest.raises(ModelError, match="already publishing"):
        acquire_output_lock(output)
    release_output_lock(lock)

    assert lock_path.is_file()
    assert (lock_path.stat().st_dev, lock_path.stat().st_ino) == initial_identity
    next_lock = acquire_output_lock(output)
    release_output_lock(next_lock)
    assert (lock_path.stat().st_dev, lock_path.stat().st_ino) == initial_identity


def test_output_lock_acquire_closes_and_unlocks_when_owner_write_fails(monkeypatch, tmp_path):
    import xhmodel_merak.model_build.workspace as executor_module

    output = tmp_path / "artifact"
    descriptors = []
    real_write = executor_module.os.write

    def fail_write(descriptor, _payload):
        descriptors.append(descriptor)
        raise OSError("reported owner-write failure")

    monkeypatch.setattr(executor_module.os, "write", fail_write)
    with pytest.raises(ModelError, match="already publishing"):
        acquire_output_lock(output)

    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])

    monkeypatch.setattr(executor_module.os, "write", real_write)
    lock = acquire_output_lock(output)
    release_output_lock(lock)


def test_output_lock_release_closes_descriptor_when_unlock_raises(monkeypatch, tmp_path):
    import xhmodel_merak.model_build.workspace as executor_module

    output = tmp_path / "artifact"
    lock = acquire_output_lock(output)
    descriptor = lock[1]
    real_flock = executor_module.fcntl.flock

    def fail_unlock(fd, operation):
        if operation == executor_module.fcntl.LOCK_UN:
            raise OSError("reported unlock failure")
        return real_flock(fd, operation)

    monkeypatch.setattr(executor_module.fcntl, "flock", fail_unlock)
    with pytest.raises(OSError, match="unlock failure"):
        release_output_lock(lock)
    with pytest.raises(OSError):
        os.fstat(descriptor)

    monkeypatch.setattr(executor_module.fcntl, "flock", real_flock)
    next_lock = acquire_output_lock(output)
    release_output_lock(next_lock)


def test_atomic_publish_never_replaces_an_existing_empty_directory(tmp_path):
    stage = tmp_path / "stage"
    stage.mkdir()
    (stage / "payload").write_bytes(b"payload")
    output = tmp_path / "output"
    output.mkdir()

    with pytest.raises(ModelError, match="will not be overwritten"):
        rename_noreplace(stage, output)

    assert stage.is_dir()
    assert list(output.iterdir()) == []


def test_onnx_external_data_checks_function_local_tensors(monkeypatch, tmp_path):
    stage = tmp_path / "artifact"
    stage.mkdir()
    (stage / "model.onnx").write_bytes(b"onnx")
    (stage / "weights.bin").write_bytes(b"weights")
    external_data = [
        SimpleNamespace(key="location", value="weights.bin"),
        SimpleNamespace(key="offset", value="5"),
        SimpleNamespace(key="length", value="8"),
    ]
    tensor = SimpleNamespace(external_data=external_data)
    attribute = SimpleNamespace(
        HasField=lambda name: name == "t",
        t=tensor,
        tensors=[],
        graphs=[],
    )
    empty_graph = SimpleNamespace(initializer=[], sparse_initializer=[], node=[])
    model = SimpleNamespace(
        graph=empty_graph,
        functions=[SimpleNamespace(node=[SimpleNamespace(attribute=[attribute])])],
        training_info=[],
    )
    monkeypatch.setitem(
        sys.modules,
        "onnx",
        SimpleNamespace(
            checker=SimpleNamespace(check_model=lambda *_args, **_kwargs: None),
            load_model=lambda *_args, **_kwargs: model,
        ),
    )

    with pytest.raises(ModelError) as exc_info:
        validate_onnx_external_locations(stage)

    assert "exceed" in exc_info.value.details["invalid_external_data"][0]["reason"]
