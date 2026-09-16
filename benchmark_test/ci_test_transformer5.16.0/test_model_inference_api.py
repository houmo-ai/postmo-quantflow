import hashlib
import json
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from xhmodel_merak._requirements import validate_worker_capabilities
from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_artifact import validation as artifact_validation
from xhmodel_merak.model_catalog import get_recipe, resolve_workflow_config
from xhmodel_merak.model_catalog.contracts import ArtifactFormatSpec
from xhmodel_merak.model_catalog.identity import build_artifact_name
from xhmodel_merak.model_catalog.registry import get_artifact_format_spec
from xhmodel_merak.model_inference import api as inference_api
from xhmodel_merak.model_inference import inference_model


ARTIFACT_NAME = "hmquant_xh2_qwen3_5_4b_w4a8_256_2k_mpe256k_visualm96-196-384-704-1536"


@pytest.fixture(autouse=True)
def _skip_worker_dependency_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "xhmodel_merak.model_inference.api.validate_worker_requirements",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "xhmodel_merak.model_inference.api.validate_worker_capabilities",
        lambda *_args, **_kwargs: None,
    )


@pytest.fixture
def artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "artifact"
    root.mkdir()
    hf_config = root / "hf_config"
    hf_config.mkdir()
    (hf_config / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {"hidden_size": 2560, "num_hidden_layers": 32},
            }
        ),
        encoding="utf-8",
    )
    meta = root / "golden_meta_info.json"
    meta.write_text(
        json.dumps(
            {
                "model_config": {"model_type": "Qwen3_5ForConditionalGeneration"},
                "hf_config": "hf_config",
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 3,
        "model_id": "qwen3.5-4b",
        "export": {"artifact_name": ARTIFACT_NAME},
        "recipe": {"variant": "full"},
        "parameters": {"components": ["visual", "prefill", "decode"]},
        "artifact": {
            "metadata_file": meta.name,
            "loadable": True,
            "files": [{"path": meta.name, "size": meta.stat().st_size}],
        },
    }
    (root / "model_build_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _complete_manifest(root)
    return root


def _runtime_result():
    return SimpleNamespace(
        output_text="ok",
        output_tokens=1,
        latency_s=0.25,
        tokens_per_second=4.0,
        spec_decode_mode=None,
        stats={},
    )


def _write_visual_metadata(root, gears):
    import onnx

    entries = []
    for capacity in gears:
        path = root / f"visual_m{capacity}" / "visual_with_act.onnx"
        path.parent.mkdir(exist_ok=True)
        tensor = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1])
        graph = onnx.helper.make_graph([], "visual_path_fixture", [tensor], [tensor])
        onnx.save(onnx.helper.make_model(graph), path)
        entries.append(
            {
                "image_token_capacity": capacity,
                "patch_token_capacity": capacity * 4,
                "hmonnx": path.relative_to(root).as_posix(),
            }
        )
    manifest = {
        "schema_version": 1,
        "visual_input_mode": "patches",
        "token_unit": "post_spatial_merge",
        "routing_policy": "smallest_fit",
        "overflow_policy": "reject",
        "gears": entries,
    }
    (root / "visual_gears.json").write_text(json.dumps(manifest))
    return {
        "visual_input_mode": "patches",
        "spatial_merge_size": 2,
        "image_token_capacity": gears[-1],
        "image_token_gears": list(gears),
        "routing_policy": "smallest_fit",
        "gear_manifest": "visual_gears.json",
        "hmonnx": entries[-1]["hmonnx"],
        "gears": entries,
    }


def _complete_manifest(artifact: Path) -> dict:
    manifest_path = artifact / "model_build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    recipe = get_recipe(manifest["model_id"])
    variant = recipe.get_variant(manifest["recipe"]["variant"])
    assert variant is not None
    config_path = resolve_workflow_config(variant.workflow_config)
    manifest["schema_version"] = 3
    manifest["export"]["model_name"] = recipe.export_name
    manifest["export"]["artifact_name"] = build_artifact_name(
        recipe,
        variant,
        recipe.export_name,
        recipe.default_precision,
        recipe.default_prefill_chunk_length,
        recipe.default_context_length,
    )
    manifest["task"] = recipe.task
    manifest["modalities"] = list(recipe.modalities)
    manifest["recipe"].update(
        {
            "version": recipe.recipe_version,
            "source_config": variant.workflow_config,
            "source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        }
    )
    effective_workflow = artifact / "effective_workflow.yaml"
    export_model = {
        "model_name": recipe.export_name,
        "context_max_length": recipe.default_context_length,
        "prefill_chunk_length": recipe.default_prefill_chunk_length,
        "max_pe_length": recipe.max_pe_length,
    }
    # Keep this minimal artifact fixture faithful to the installed graph schemes.
    for binding in recipe.deployment_precision_bindings:
        current = export_model
        parts = binding.config_path.removeprefix("export.model.").split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = {"quant_type": recipe.deployment_precision[binding.components[0]]}
    effective_workflow.write_text(
        yaml.safe_dump(
            {"export": {"model": export_model}},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    manifest["recipe"].update(
        {
            "effective_workflow": effective_workflow.name,
            "effective_workflow_sha256": hashlib.sha256(effective_workflow.read_bytes()).hexdigest(),
        }
    )
    manifest["artifact"]["components"] = list(recipe.components + variant.extra_components)
    manifest["parameters"]["components"] = list(recipe.components + variant.extra_components)
    manifest["parameters"].update(
        {
            "precision": {
                "checkpoint": recipe.default_precision.to_dict(),
                "deployment": dict(recipe.deployment_precision),
            },
            "quant_method": recipe.quant_methods[0] if recipe.quant_methods else None,
            "profile": recipe.default_profile,
            "context_length": recipe.default_context_length,
            "prefill_chunk_length": recipe.default_prefill_chunk_length,
            "max_pe_length": recipe.max_pe_length,
            "vision_profile": recipe.vision_profile,
            "vision_gears": list(recipe.vision_gears),
            "golden_components": list(recipe.golden_components),
        }
    )
    metadata_path = artifact / manifest["artifact"].get(
        "metadata_file",
        manifest["artifact"].get("golden_meta_info"),
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if recipe.vision_gears and "visual_config" not in metadata:
        metadata["visual_config"] = _write_visual_metadata(artifact, recipe.vision_gears)
    metadata.setdefault("model_config", {}).update(
        {
            "context_max_length": manifest["parameters"]["context_length"],
            "prefill_chunk_length": manifest["parameters"]["prefill_chunk_length"],
            "max_pe_length": manifest["parameters"]["max_pe_length"],
        }
    )
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    manifest["format"] = {
        "name": recipe.artifact_format,
        "version": get_artifact_format_spec(recipe.artifact_format).version,
    }
    inventoried_paths = {
        str(entry["path"])
        for entry in manifest["artifact"]["files"]
        if isinstance(entry, dict) and isinstance(entry.get("path"), str)
    }
    inventoried_paths.update({"hf_config/config.json", effective_workflow.name, effective_workflow.name})
    if recipe.vision_gears:
        inventoried_paths.add("visual_gears.json")
        inventoried_paths.update(entry["hmonnx"] for entry in metadata["visual_config"]["gears"])
    manifest["artifact"]["files"] = [
        {
            "path": relative,
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for relative in sorted(inventoried_paths)
        if (path := artifact / relative).is_file()
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _refresh_inventoried_file(artifact: Path, manifest: dict, relative: str) -> None:
    path = artifact / relative
    for entry in manifest["artifact"]["files"]:
        if entry["path"] == relative:
            entry["size"] = path.stat().st_size
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            break
    else:
        raise AssertionError(f"missing inventory entry for {relative}")
    if relative == manifest["recipe"]["effective_workflow"]:
        manifest["recipe"]["effective_workflow_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()


def _write_minimal_fixed_artifact(tmp_path: Path, model: str) -> tuple[Path, str]:
    recipe = get_recipe(model)
    variant = recipe.default_variant
    assert recipe.fixed_artifact_name is not None
    root = tmp_path / recipe.export_name
    root.mkdir()
    metadata = root / "metadata.json"
    metadata.write_text("{}", encoding="utf-8")
    effective_workflow = root / "effective_workflow.yaml"
    effective_workflow.write_text(
        yaml.safe_dump({"export": {"model": {"type": "fixed-runtime"}}}, sort_keys=False),
        encoding="utf-8",
    )
    config_path = resolve_workflow_config(variant.workflow_config)
    components = list(recipe.components + variant.extra_components)
    manifest = {
        "schema_version": 3,
        "model_id": recipe.model_id,
        "export": {"model_name": recipe.export_name, "artifact_name": recipe.fixed_artifact_name},
        "task": recipe.task,
        "modalities": list(recipe.modalities),
        "recipe": {
            "version": recipe.recipe_version,
            "variant": variant.name,
            "source_config": variant.workflow_config,
            "source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "effective_workflow": effective_workflow.name,
            "effective_workflow_sha256": hashlib.sha256(effective_workflow.read_bytes()).hexdigest(),
        },
        "parameters": {
            "precision": {
                "checkpoint": recipe.default_precision.to_dict(),
                "deployment": dict(recipe.deployment_precision),
            },
            "quant_method": None,
            "profile": recipe.default_profile,
            "context_length": None,
            "prefill_chunk_length": None,
            "max_pe_length": recipe.max_pe_length,
            "vision_profile": recipe.vision_profile,
            "vision_gears": list(recipe.vision_gears),
            "components": components,
            "golden_components": list(recipe.golden_components),
        },
        "artifact": {
            "metadata_file": metadata.name,
            "loadable": True,
            "components": components,
            "files": [],
        },
        "format": {"name": recipe.artifact_format, "version": get_artifact_format_spec(recipe.artifact_format).version},
    }
    manifest["artifact"]["files"] = [
        {
            "path": path.name,
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in (effective_workflow, metadata)
    ]
    (root / "model_build_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, recipe.fixed_artifact_name


def _rewrite_inventory(root: Path) -> None:
    manifest_path = root / "model_build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact"]["files"] = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != manifest_path
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _write_valid_kokoro_preflight_artifact(tmp_path: Path) -> tuple[Path, str]:
    from _kokoro_fixture import write_release

    root, artifact_name = _write_minimal_fixed_artifact(tmp_path, "kokoro-82m-v1.1-zh")
    metadata = write_release(root)
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    _rewrite_inventory(root)
    return root, artifact_name


def test_inference_model_validates_manifest_and_runs_registered_adapter(monkeypatch, artifact):
    captured = {}

    def quick_test(path, **kwargs):
        captured["path"] = path
        captured.update(kwargs)
        return _runtime_result()

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        quick_test,
    )

    result = inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))

    assert result.output_text == "ok"
    assert result.components == ("visual", "prefill", "decode")
    assert captured["path"] == str(artifact / "golden_meta_info.json")
    assert captured["prompt"] == "hello"
    assert "input_text" not in captured
    assert captured["device"] == "cuda:0"
    assert captured["device_map"] == ["cuda:0"]
    assert captured["image_path"] is None
    assert captured["stream_output"] is False


def test_inference_model_accepts_explicit_device_without_forwarding_it_to_schema(monkeypatch, artifact):
    captured = {}

    def quick_test(path, **kwargs):
        captured.update(kwargs)
        return _runtime_result()

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        quick_test,
    )

    inference_model(inputs={"prompt": "hello"}, device="cuda:6", artifact_dir=str(artifact))

    assert captured["device"] == "cuda:6"
    assert captured["device_map"] == ["cuda:6"]


def test_inference_model_scopes_golden_to_selected_components(monkeypatch, artifact):
    image = artifact.parent / "image.png"
    image.touch()

    def quick_test(path, **kwargs):
        assert kwargs["golden_components"] == ("visual", "decode")
        for capacity in (96, 196, 384, 704, 1536):
            step = artifact / f"visual_m{capacity}" / "step_0"
            step.mkdir()
            (step / "output.npy").write_bytes(b"visual")
        (artifact / "decode" / "step_0").mkdir(parents=True)
        (artifact / "decode" / "step_0" / "output.npy").write_bytes(b"decode")
        return _runtime_result()

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        quick_test,
    )

    result = inference_model(
        inputs={"prompt": "hello", "image": str(image)},
        artifact_dir=str(artifact),
        golden=["decode", "visual", "decode"],
    )

    assert result.golden_components == ("visual", "decode")
    assert result.golden_artifacts == {
        "visual": tuple(sorted(f"visual_m{capacity}/step_0" for capacity in (96, 196, 384, 704, 1536))),
        "decode": ("decode/step_0",),
    }


@pytest.mark.parametrize("defect", ["manifest_graph", "default_graph"])
def test_inference_model_rechecks_visual_contract_even_with_valid_checksums(monkeypatch, artifact, defect):
    manifest_path = artifact / "model_build_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if defect == "manifest_graph":
        relative = "visual_gears.json"
        payload = json.loads((artifact / relative).read_text())
        payload["gears"][0]["hmonnx"] = payload["gears"][1]["hmonnx"]
    else:
        relative = "golden_meta_info.json"
        payload = json.loads((artifact / relative).read_text())
        payload["visual_config"]["hmonnx"] = payload["visual_config"]["gears"][0]["hmonnx"]
    (artifact / relative).write_text(json.dumps(payload))
    _refresh_inventoried_file(artifact, manifest, relative)
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        lambda *_args, **_kwargs: pytest.fail("invalid visual metadata must fail before opening the runtime"),
    )
    with pytest.raises(ModelError, match="dynamic vision Artifact is invalid"):
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))


def test_inference_model_rejects_expected_model_id_mismatch(artifact):
    with pytest.raises(ModelError) as exc_info:
        inference_model(expected_model_id="qwen3.5-9b", inputs={"prompt": "hello"}, artifact_dir=str(artifact))

    assert exc_info.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED


def test_inference_model_rejects_unknown_inputs_before_runtime(monkeypatch, artifact):
    called = False

    def quick_test(path, **kwargs):
        nonlocal called
        called = True
        return _runtime_result()

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        quick_test,
    )

    with pytest.raises(ModelError) as exc_info:
        inference_model(
            inputs={"prompt": "hello", "temperature": 0.8},
            artifact_dir=str(artifact),
        )

    assert exc_info.value.code == ErrorCode.INVALID_PARAMETER
    assert called is False


def test_inference_model_rejects_unregistered_or_unloadable_golden(artifact):
    with pytest.raises(ModelError) as exc_info:
        inference_model(
            inputs={"prompt": "hello"},
            artifact_dir=str(artifact),
            golden=["mtp"],
        )
    assert exc_info.value.code == ErrorCode.GOLDEN_NOT_SUPPORTED

    manifest_path = artifact / "model_build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact"]["loadable"] = False
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelError) as exc_info:
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))
    assert exc_info.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED


def test_inference_model_omits_device_map_for_speculative_runtime(monkeypatch, artifact):
    manifest_path = artifact / "model_build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_id"] = "qwen3.5-9b"
    manifest["recipe"]["variant"] = "mtp"
    manifest["parameters"]["components"].append("mtp")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    hf_config = artifact / "hf_config" / "config.json"
    hf_config.write_text(
        json.dumps({"model_type": "qwen3_5", "text_config": {"hidden_size": 4096, "num_hidden_layers": 32}}),
        encoding="utf-8",
    )
    _complete_manifest(artifact)

    def quick_test(path, **kwargs):
        assert "device_map" not in kwargs
        return SimpleNamespace(**{**_runtime_result().__dict__, "spec_decode_mode": "mtp"})

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        quick_test,
    )

    result = inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))

    assert result.spec_decode_mode == "mtp"
    assert result.components[-1] == "mtp"


def test_inference_model_rejects_unregistered_manifest_variant(monkeypatch, artifact):
    manifest_path = artifact / "model_build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recipe"]["variant"] = "mtp"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        lambda *args, **kwargs: pytest.fail("runtime must not load an unregistered variant"),
    )

    with pytest.raises(ModelError) as exc_info:
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))

    assert exc_info.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED
    assert "variant" in str(exc_info.value).lower()


def test_inference_model_requires_image_for_visual_golden(monkeypatch, artifact):
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        lambda *args, **kwargs: pytest.fail("runtime must not run without a visual input"),
    )

    with pytest.raises(ModelError) as exc_info:
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact), golden=["visual"])

    assert exc_info.value.code == ErrorCode.GOLDEN_NOT_SUPPORTED


def test_inference_model_rejects_decode_golden_without_a_decode_step(monkeypatch, artifact):
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        lambda *args, **kwargs: pytest.fail("runtime must not load for an impossible decode Golden request"),
    )

    with pytest.raises(ModelError) as exc_info:
        inference_model(
            inputs={"prompt": "hello", "max_new_tokens": 1},
            artifact_dir=str(artifact),
            golden=["decode"],
        )

    assert exc_info.value.code == ErrorCode.UNSUPPORTED_COMBINATION
    assert "max_new_tokens >= 2" in str(exc_info.value)


def test_inference_model_verifies_file_checksums_before_runtime(monkeypatch, artifact):
    manifest_path = artifact / "model_build_manifest.json"
    manifest = _complete_manifest(artifact)
    meta = artifact / "golden_meta_info.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    meta.write_bytes(b"x" * meta.stat().st_size)
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        lambda *args, **kwargs: pytest.fail("runtime must not load a corrupted Artifact"),
    )

    with pytest.raises(ModelError) as exc_info:
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))

    assert exc_info.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED
    assert "checksum" in str(exc_info.value).lower()


@pytest.mark.parametrize("during_runtime", [False, True])
@pytest.mark.parametrize("damage", ["missing", "size", "checksum"])
def test_extra_runtime_files_do_not_hide_inventoried_file_damage(monkeypatch, artifact, during_runtime, damage):
    weight = artifact / "weights.bin"
    weight.write_bytes(b"weights")
    _rewrite_inventory(artifact)
    (artifact / "runtime.tmp").write_text("allowed extra file")
    runtime_called = False

    def corrupt():
        if damage == "missing":
            weight.unlink()
        elif damage == "size":
            weight.write_bytes(b"truncated")
        else:
            weight.write_bytes(b"changed")

    def quick_test(*_args, **_kwargs):
        nonlocal runtime_called
        runtime_called = True
        corrupt()
        return _runtime_result()

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx", quick_test,
    )
    if not during_runtime:
        corrupt()
    with pytest.raises(ModelError) as exc_info:
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))
    assert exc_info.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED
    assert runtime_called is during_runtime
    assert exc_info.value.details["path"] in ("weights.bin", str(weight))


def test_inference_model_validates_artifact_before_worker_preflight(monkeypatch, artifact):
    manifest_path = artifact / "model_build_manifest.json"
    manifest = _complete_manifest(artifact)
    manifest["parameters"]["max_pe_length"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    preflight_called = False

    def worker_preflight(*_args, **_kwargs):
        nonlocal preflight_called
        preflight_called = True
        pytest.fail("worker state must not be checked before Artifact provenance")

    monkeypatch.setattr(
        "xhmodel_merak.model_inference.api.validate_worker_requirements",
        worker_preflight,
    )

    with pytest.raises(ModelError) as exc_info:
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))

    assert exc_info.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED
    assert preflight_called is False


@pytest.mark.parametrize("cache_setting", [None, "0", "1"])
def test_inference_model_leaves_sdk_cache_policy_unchanged(monkeypatch, artifact, cache_setting):
    _complete_manifest(artifact)
    monkeypatch.delenv("XHQUANT_DISABLE_INITIALIZER_DIGEST_CACHE", raising=False)
    if cache_setting is not None:
        monkeypatch.setenv("XHQUANT_DISABLE_INITIALIZER_DIGEST_CACHE", cache_setting)

    def quick_test(*_args, **_kwargs):
        import os

        assert os.environ.get("XHQUANT_DISABLE_INITIALIZER_DIGEST_CACHE") == cache_setting
        return _runtime_result()

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        quick_test,
    )

    result = inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))

    assert result.output_text == "ok"

    import os

    assert os.environ.get("XHQUANT_DISABLE_INITIALIZER_DIGEST_CACHE") == cache_setting


@pytest.mark.parametrize("multi_graph", [False, True])
def test_inference_model_real_sdk_deduplicates_with_runtime_cache(monkeypatch, artifact, multi_graph):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from xhquant.xhonnxruntime import llm_hmonnx_loader as loader

    monkeypatch.delenv("XHQUANT_DISABLE_INITIALIZER_DIGEST_CACHE", raising=False)

    # Real SDK graph/weight loading with eight-byte external weights. Only the
    # expensive generation loop is replaced; cache naming, IO and sharing run.
    names = ("prefill", "decode", "draft") if multi_graph else ("prefill", "decode")
    paths = {}
    for name in names:
        path = artifact / name / "model.onnx"
        path.parent.mkdir()
        graph = helper.make_graph(
            [helper.make_node("Add", ["input", "weight"], ["output"])],
            name,
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2])],
            [numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32), name="weight")],
        )
        onnx.save_model(
            helper.make_model(graph),
            path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.bin",
            size_threshold=0,
        )
        paths[name] = path
    _rewrite_inventory(artifact)
    before = (artifact / "model_build_manifest.json").read_bytes()
    monkeypatch.delattr(loader, "initializer_digest_cache_disable_available", raising=False)
    monkeypatch.setattr(inference_api, "validate_worker_capabilities", validate_worker_capabilities)

    def quick_test(*_args, **_kwargs):
        graphs = {name: loader.load_onnx_graph_only(str(path)) for name, path in paths.items()}
        if multi_graph:
            pool = loader.load_multi_graph_initializer_pool(graphs)
            weights = [pool.graphs[name]["weight"] for name in names]
        else:
            pool = loader.load_shared_initializer_pool(graphs["prefill"], graphs["decode"])
            weights = [pool.prefill["weight"], pool.decode["weight"]]
        assert all(weight is weights[0] for weight in weights)
        return _runtime_result()

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        quick_test,
    )
    assert inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact)).output_text == "ok"
    caches = list(artifact.glob(".xhquant_*_hmonnx_sha256_*.json"))
    assert caches
    assert inference_model(inputs={"prompt": "again"}, artifact_dir=str(artifact)).output_text == "ok"
    assert all(path.is_file() for path in caches)
    assert (artifact / "model_build_manifest.json").read_bytes() == before


def test_inference_leaves_digest_cache_setting_unchanged_after_runtime_failure(monkeypatch, artifact):
    _complete_manifest(artifact)
    monkeypatch.setenv("XHQUANT_DISABLE_INITIALIZER_DIGEST_CACHE", "0")

    def fail(*_args, **_kwargs):
        assert os.environ["XHQUANT_DISABLE_INITIALIZER_DIGEST_CACHE"] == "0"
        raise RuntimeError("runtime failed")

    monkeypatch.setattr("xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx", fail)
    with pytest.raises(ModelError, match="runtime failed"):
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))
    assert os.environ["XHQUANT_DISABLE_INITIALIZER_DIGEST_CACHE"] == "0"


@pytest.mark.parametrize("model", ["pi0.5-libero", "kokoro-82m-v1.1-zh"])
def test_runtime_cache_is_allowed_for_all_artifact_formats(tmp_path, model):
    artifact, _ = _write_minimal_fixed_artifact(tmp_path, model)
    cache = artifact / f".xhquant_llm_hmonnx_sha256_{'a' * 64}.json"
    cache.write_text("{}")
    # The shared inventory contract does not depend on the task or SDK format.
    manifest = json.loads((artifact / "model_build_manifest.json").read_text())
    artifact_validation.validate_manifest_files(artifact, manifest["artifact"])
    assert cache.read_text() == "{}"


@pytest.mark.parametrize("node_kind", ["symlink", "fifo"])
def test_inference_model_cache_names_do_not_exempt_non_regular_files(monkeypatch, artifact, tmp_path, node_kind):
    cache = artifact / f".xhquant_llm_hmonnx_sha256_{'a' * 64}.json"
    if node_kind == "symlink":
        target = tmp_path / "outside.json"
        target.write_text("{}")
        cache.symlink_to(target)
    else:
        os.mkfifo(cache)
    with pytest.raises(ModelError, match="file inventory") as exc_info:
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))
    field = "symlinks" if node_kind == "symlink" else "non_regular_nodes"
    assert exc_info.value.details[field] == [cache.name]


def test_inference_model_still_checks_inventoried_cache_hashes(monkeypatch, artifact):
    cache = artifact / f".xhquant_llm_hmonnx_sha256_{'a' * 64}.json"
    cache.write_text("{}")
    _rewrite_inventory(artifact)

    def quick_test(*_args, **_kwargs):
        cache.write_text("[]")  # Same size, different digest; must not be exempt.
        return _runtime_result()

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        quick_test,
    )
    with pytest.raises(ModelError, match="checksum"):
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))


@pytest.mark.parametrize(
    "relative",
    [
        f"hf_config/.xhquant_llm_hmonnx_sha256_{'a' * 64}.json",
        f"decode/.xhquant_llm_hmonnx_sha256_{'a' * 64}.json",
        f".xhquant_llm_hmonnx_sha256_{'A' * 64}.json",
        f".xhquant_llm_hmonnx_sha256_{'a' * 63}.json",
        f".xhquant_llm_hmonnx_sha256_{'a' * 65}.json",
        f".xhquant_llm_hmonnx_sha256_{'a' * 64}.json.tmp",
        f".xhquant_unknown_hmonnx_sha256_{'a' * 64}.json",
        "runtime/session.log",
        "runtime/loading.tmp",
    ],
)
def test_inference_model_accepts_extra_regular_files(monkeypatch, artifact, relative):
    cache = artifact / relative
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text("{}")
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        lambda *_args, **_kwargs: _runtime_result(),
    )
    assert inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact)).output_text == "ok"
    assert cache.is_file()


@pytest.mark.parametrize("model", ["pi0.5-libero", "kokoro-82m-v1.1-zh"])
def test_inference_model_runs_fixed_recipe_with_none_sequence_lengths(monkeypatch, tmp_path, model):
    artifact, artifact_name = _write_minimal_fixed_artifact(tmp_path, model)
    calls = []
    fake_adapter = SimpleNamespace(
        run=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(output_type="ok"),
    )
    monkeypatch.setattr("xhmodel_merak.model_inference.api.load_adapter", lambda _kind, _name: fake_adapter)

    inputs = (
        {"text": "hello"}
        if model.startswith("kokoro")
        else {"instruction": "pick up", "state": [0.0] * 8, "images": ["unused"]}
    )
    monkeypatch.setattr(inference_api, "validate_inputs", lambda _recipe, value: dict(value))
    result = inference_model(inputs=inputs, artifact_dir=str(artifact))

    assert result.output_type == "ok"
    assert calls[0]["recipe"].model_id == model


def test_inference_model_validates_kokoro_contract_before_worker_preflight(monkeypatch, tmp_path):
    artifact, artifact_name = _write_minimal_fixed_artifact(tmp_path, "kokoro-82m-v1.1-zh")
    worker_preflight_called = False

    def worker_preflight(*_args, **_kwargs):
        nonlocal worker_preflight_called
        worker_preflight_called = True
        pytest.fail("worker dependencies must not mask a malformed Kokoro Artifact")

    monkeypatch.setattr(
        "xhmodel_merak.model_inference.api.validate_worker_requirements",
        worker_preflight,
    )

    with pytest.raises(ModelError, match="adapter contract") as exc_info:
        inference_model(inputs={"text": "hello"}, artifact_dir=str(artifact))

    assert exc_info.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED
    assert worker_preflight_called is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("graph_mode", "bucketed_precision_split"),
        ("schema_version", 1),
        ("export_complete", False),
        ("phase_boundary", {"execution": "host_onnxruntime_fp32", "input": "phase_increments", "output": "sine"}),
    ],
)
def test_inference_model_rejects_retired_kokoro_release_before_worker_preflight(
    monkeypatch,
    tmp_path,
    field,
    value,
):
    artifact, _ = _write_valid_kokoro_preflight_artifact(tmp_path)
    path = artifact / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata[field] = value
    path.write_text(json.dumps(metadata), encoding="utf-8")
    _rewrite_inventory(artifact)
    monkeypatch.setattr(
        "xhmodel_merak.model_inference.api.validate_worker_requirements",
        lambda *_args, **_kwargs: pytest.fail("invalid release must fail before worker loading"),
    )
    with pytest.raises(ModelError, match="adapter contract") as exc_info:
        inference_model(inputs={"text": "你好"}, artifact_dir=str(artifact))
    assert exc_info.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED


@pytest.mark.parametrize(
    "corruption",
    [
        "missing_seed",
        "missing_routes",
        "missing_components",
        "missing_voice",
        "invalid_vocab",
        "invalid_voice_dtype",
    ],
)
def test_inference_model_validates_full_kokoro_runtime_contract_before_worker_preflight(
    monkeypatch,
    tmp_path,
    corruption,
):
    artifact, artifact_name = _write_valid_kokoro_preflight_artifact(tmp_path)
    metadata_path = artifact / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if corruption == "missing_seed":
        metadata.pop("seed")
    elif corruption == "missing_routes":
        metadata.pop("bucket_presets")
    elif corruption == "missing_components":
        metadata["components"] = {}
    elif corruption == "missing_voice":
        metadata["runtime_assets"] = {}
    elif corruption == "invalid_vocab":
        config_path = artifact / "hf_config" / "config.json"
        config_path.write_text(json.dumps({"vocab": {}}), encoding="utf-8")
    else:
        voice_path = artifact / "assets" / "voices" / "zf_001.npy"
        np.save(voice_path, np.ones((1, 256), dtype=np.float64), allow_pickle=False)
        voice_metadata = metadata["runtime_assets"]["voice_pack"]
        voice_metadata["dtype"] = "float64"
        voice_metadata["size_bytes"] = voice_path.stat().st_size
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _rewrite_inventory(artifact)
    worker_preflight_called = False

    def worker_preflight(*_args, **_kwargs):
        nonlocal worker_preflight_called
        worker_preflight_called = True
        pytest.fail("worker dependencies must not mask an invalid Kokoro runtime contract")

    monkeypatch.setattr(
        "xhmodel_merak.model_inference.api.validate_worker_requirements",
        worker_preflight,
    )

    with pytest.raises(ModelError, match="adapter contract") as exc_info:
        inference_model(inputs={"text": "hello"}, artifact_dir=str(artifact))

    assert exc_info.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED
    assert worker_preflight_called is False


def test_inference_model_allows_uninventoried_cache_and_runtime_golden(monkeypatch, artifact):
    _complete_manifest(artifact)
    cache = artifact / ".xhquant_loader_sha256_cache.json"
    cache.write_text("{}", encoding="utf-8")
    golden = artifact / "decode" / "step_0" / "output.npy"

    def quick_test(*_args, **_kwargs):
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_bytes(b"runtime-output")
        return _runtime_result()

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        quick_test,
    )

    assert inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact)).output_text == "ok"
    result = inference_model(
        inputs={"prompt": "hello"},
        artifact_dir=str(artifact),
        golden=["decode"],
    )
    assert result.output_text == "ok"
    assert cache.read_text() == "{}"


@pytest.mark.parametrize(
    "cache_name",
    [
        ".xhquant_llm_hmonnx_sha256_prefill-decode.json",
        ".xhquant_multi_hmonnx_sha256_all-graphs.json",
    ],
)
def test_inference_model_never_removes_digest_cache_from_stale_golden_path(
    monkeypatch,
    artifact,
    cache_name,
):
    _complete_manifest(artifact)
    cache = artifact / "decode" / "step_0" / cache_name
    cache.parent.mkdir(parents=True)
    cache.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        lambda *_args, **_kwargs: _runtime_result(),
    )

    assert inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact)).output_text == "ok"
    assert cache.is_file()


@pytest.mark.parametrize(
    "component_dir", ["Decode", "Vision", "Visual_m96", "visual_m0", "visual_m096", "visual_m96_extra"]
)
def test_inference_model_stale_golden_component_matching_is_case_sensitive(
    monkeypatch,
    artifact,
    component_dir,
):
    _complete_manifest(artifact)
    output = artifact / component_dir / "step_0" / "output.npy"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"stale")
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        lambda *_args, **_kwargs: _runtime_result(),
    )

    assert inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact)).output_text == "ok"
    assert output.is_file()


@pytest.mark.parametrize(
    "component_dir,component", [("decode", "decode"), ("visual_m96", "visual"), ("visual_m1536", "visual")]
)
def test_inference_model_allows_repeated_golden_runs_without_reusing_prior_output(
    monkeypatch, artifact, component_dir, component
):
    _complete_manifest(artifact)
    golden = artifact / component_dir / "step_0" / "output.npy"
    run_count = 0
    image = artifact.parent / "image.png"
    image.touch()
    inputs = {"prompt": "hello", "image": str(image)}

    def quick_test(*_args, **_kwargs):
        nonlocal run_count
        run_count += 1
        assert not golden.exists()
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_bytes(f"run-{run_count}".encode())
        return _runtime_result()

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        quick_test,
    )

    first = inference_model(
        inputs=inputs,
        artifact_dir=str(artifact),
        golden=[component],
    )
    assert golden.read_bytes() == b"run-1"

    second = inference_model(
        inputs=inputs,
        artifact_dir=str(artifact),
        golden=[component],
    )

    assert run_count == 2
    assert first.golden_artifacts == second.golden_artifacts == {component: (f"{component_dir}/step_0",)}
    assert golden.read_bytes() == b"run-2"


@pytest.mark.parametrize("node_kind", ["symlink", "fifo", "socket"])
def test_inference_model_rejects_unsafe_nodes_in_stale_golden_paths(
    monkeypatch,
    artifact,
    node_kind,
):
    _complete_manifest(artifact)
    output = artifact / "decode" / "step_0" / "output.npy"
    output.parent.mkdir(parents=True)
    bound_socket = None
    if node_kind == "symlink":
        target = artifact.parent / "outside-output.npy"
        target.write_bytes(b"outside")
        output.symlink_to(target)
    elif node_kind == "fifo":
        os.mkfifo(output)
    else:
        bound_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        bound_socket.bind(str(output))

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        lambda *_args, **_kwargs: pytest.fail("runtime must not load an Artifact containing an unsafe node"),
    )
    try:
        with pytest.raises(ModelError, match="file inventory") as exc_info:
            inference_model(
                inputs={"prompt": "hello"},
                artifact_dir=str(artifact),
                golden=["decode"],
            )
        assert exc_info.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED
        detail = "symlinks" if node_kind == "symlink" else "non_regular_nodes"
        assert exc_info.value.details[detail] == ["decode/step_0/output.npy"]
        assert output.is_symlink() if node_kind == "symlink" else output.exists()
    finally:
        if bound_socket is not None:
            bound_socket.close()


def test_inference_model_never_removes_inventoried_golden_shaped_files(monkeypatch, artifact):
    manifest_path = artifact / "model_build_manifest.json"
    manifest = _complete_manifest(artifact)
    output = artifact / "decode" / "step_0" / "output.npy"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"inventoried")
    manifest["artifact"]["files"].append(
        {
            "path": output.relative_to(artifact).as_posix(),
            "size": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def quick_test(*_args, **_kwargs):
        assert output.read_bytes() == b"inventoried"
        return _runtime_result()

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        quick_test,
    )

    inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))

    assert output.read_bytes() == b"inventoried"


def test_inference_model_allows_selected_inventoried_golden_to_change(monkeypatch, artifact):
    manifest_path = artifact / "model_build_manifest.json"
    manifest = _complete_manifest(artifact)
    output = artifact / "decode" / "step_0" / "output.npy"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"build-golden")
    manifest["artifact"]["files"].append(
        {
            "path": output.relative_to(artifact).as_posix(),
            "size": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    def quick_test(*_args, **kwargs):
        assert kwargs["golden_components"] == ("decode",)
        assert output.read_bytes() == b"build-golden"
        output.write_bytes(b"runtime-gold")
        return _runtime_result()

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        quick_test,
    )

    result = inference_model(
        inputs={"prompt": "hello"},
        artifact_dir=str(artifact),
        golden=["decode"],
    )

    assert output.read_bytes() == b"runtime-gold"
    assert result.golden_artifacts == {"decode": ("decode/step_0",)}


def test_inference_model_reports_only_selected_golden_with_extra_runtime_files(monkeypatch, artifact):
    _complete_manifest(artifact)

    def quick_test(*_args, **_kwargs):
        for component in ("decode", "prefill"):
            output = artifact / component / "step_0" / "output.npy"
            output.parent.mkdir(parents=True)
            output.write_bytes(component.encode())
        return _runtime_result()

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        quick_test,
    )

    result = inference_model(
        inputs={"prompt": "hello"}, artifact_dir=str(artifact), golden=["decode"],
    )
    assert result.golden_artifacts == {"decode": ("decode/step_0",)}
    assert (artifact / "prefill/step_0/output.npy").read_bytes() == b"prefill"


def test_inference_model_allows_runtime_digest_cache_alongside_golden(monkeypatch, artifact):
    _complete_manifest(artifact)
    output = artifact / "decode" / "step_0" / "output.npy"
    cache = output.parent / ".xhquant_multi_hmonnx_sha256_runtime.json"

    def quick_test(*_args, **_kwargs):
        output.parent.mkdir(parents=True)
        output.write_bytes(b"runtime-output")
        cache.write_text("{}", encoding="utf-8")
        return _runtime_result()

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        quick_test,
    )

    result = inference_model(
        inputs={"prompt": "hello"}, artifact_dir=str(artifact), golden=["decode"],
    )
    assert result.golden_artifacts == {"decode": ("decode/step_0",)}
    assert cache.is_file()


@pytest.mark.parametrize(
    "component,extra_dir,extra_step",
    [
        ("decode", "Decode", "step_0"),
        ("visual", "Vision", "step_0"),
        ("visual", "Visual_m96", "step_0"),
        ("visual", "visual_m096", "step_0"),
        ("decode", "decode", "step_tmp"),
        ("decode", "decode", "step_"),
        ("decode", "decode", "step_1.tmp"),
    ],
)
def test_inference_model_does_not_report_extra_directory_as_golden(
    monkeypatch, artifact, component, extra_dir, extra_step,
):
    _complete_manifest(artifact)
    output = artifact / extra_dir / extra_step / "output.npy"
    selected_dir = "visual_m96" if component == "visual" else "decode"
    image = artifact.parent / "image.png"
    image.touch()

    def quick_test(*_args, **_kwargs):
        output.parent.mkdir(parents=True)
        output.write_bytes(b"runtime-output")
        selected = artifact / selected_dir / "step_0" / "output.npy"
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.write_bytes(b"selected-output")
        return _runtime_result()

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        quick_test,
    )

    result = inference_model(
        inputs={"prompt": "hello", "image": str(image)}, artifact_dir=str(artifact), golden=[component],
    )
    assert result.output_text == "ok"
    assert result.golden_artifacts == {component: (f"{selected_dir}/step_0",)}
    assert output.read_bytes() == b"runtime-output"


def test_inference_model_accepts_exact_lowercase_vision_alias_for_visual_golden(monkeypatch, artifact):
    _complete_manifest(artifact)
    image = artifact.parent / "image.png"
    image.write_bytes(b"image")
    output = artifact / "vision" / "step_0" / "output.npy"

    def quick_test(*_args, **_kwargs):
        output.parent.mkdir(parents=True)
        output.write_bytes(b"runtime-output")
        return _runtime_result()

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        quick_test,
    )

    result = inference_model(
        inputs={"prompt": "hello", "image": str(image)},
        artifact_dir=str(artifact),
        golden=["visual"],
    )

    assert result.golden_artifacts == {"visual": ("vision/step_0",)}


def test_inference_model_uses_root_manifest_with_extra_nested_manifest(monkeypatch, artifact):
    _complete_manifest(artifact)
    nested_manifest = artifact / "nested" / "model_build_manifest.json"
    nested_manifest.parent.mkdir()
    nested_manifest.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        lambda *_args, **_kwargs: _runtime_result(),
    )
    assert inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact)).output_text == "ok"
    assert nested_manifest.read_text() == "{}"


@pytest.mark.parametrize("schema_version", [None, 1, 2, True, "3"])
def test_inference_model_rejects_legacy_schemas_even_with_old_opt_in(monkeypatch, artifact, schema_version):
    monkeypatch.setenv("XHMODEL_ALLOW_LEGACY_MANIFEST", "1")
    manifest_path = artifact / "model_build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = schema_version
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelError, match="rebuild"):
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))


def test_inference_model_rejects_noncanonical_artifact_name(monkeypatch, artifact):
    manifest_path = artifact / "model_build_manifest.json"
    manifest = _complete_manifest(artifact)
    manifest["export"]["artifact_name"] = "hmquant_alias"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        lambda *_args, **_kwargs: pytest.fail("runtime must not load a renamed Artifact"),
    )

    with pytest.raises(ModelError, match="installed model Recipe"):
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))


def test_inference_model_revalidates_inventory_after_runtime(monkeypatch, artifact):
    _complete_manifest(artifact)
    meta = artifact / "golden_meta_info.json"
    golden = artifact / "decode" / "step_0" / "output.npy"

    def quick_test(*_args, **_kwargs):
        meta.write_bytes(b"x" * meta.stat().st_size)
        golden.parent.mkdir(parents=True)
        golden.write_bytes(b"runtime-output")
        return _runtime_result()

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        quick_test,
    )

    with pytest.raises(ModelError, match="checksum does not match"):
        inference_model(
            inputs={"prompt": "hello"},
            artifact_dir=str(artifact),
            golden=["decode"],
        )


def test_inference_model_reports_only_new_or_changed_exact_component_golden(monkeypatch, artifact):
    spoof = artifact / "notdecode" / "step_0" / "output.npy"
    spoof.parent.mkdir(parents=True)
    spoof.write_bytes(b"spoof")
    _rewrite_inventory(artifact)
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        lambda *_args, **_kwargs: _runtime_result(),
    )
    with pytest.raises(ModelError, match="produced no artifacts"):
        inference_model(
            inputs={"prompt": "hello"},
            artifact_dir=str(artifact),
            golden=["decode"],
        )

    existing = artifact / "decode" / "step_0" / "output.npy"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"old")

    def update_golden(*_args, **_kwargs):
        existing.write_bytes(b"new-output")
        return _runtime_result()

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        update_golden,
    )
    result = inference_model(
        inputs={"prompt": "hello"},
        artifact_dir=str(artifact),
        golden=["decode"],
    )

    assert result.golden_artifacts == {"decode": ("decode/step_0",)}


def test_inference_model_rejects_recipe_relabeling_before_runtime(monkeypatch, artifact):
    manifest_path = artifact / "model_build_manifest.json"
    manifest = _complete_manifest(artifact)
    manifest["model_id"] = "qwen3.5-9b"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        lambda *_args, **_kwargs: pytest.fail("runtime must not load a relabeled Artifact"),
    )

    with pytest.raises(ModelError, match="installed model Recipe"):
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))


def test_inference_model_rejects_coordinated_relabeling_from_local_hf_config(monkeypatch, artifact):
    from xhmodel_merak.model_build.resolver import _build_artifact_name

    manifest_path = artifact / "model_build_manifest.json"
    manifest = _complete_manifest(artifact)
    recipe = get_recipe("qwen3.5-9b")
    variant = recipe.default_variant
    config_path = resolve_workflow_config(variant.workflow_config)
    relabeled_name = _build_artifact_name(
        recipe,
        variant,
        recipe.export_name,
        recipe.default_precision,
        recipe.default_prefill_chunk_length,
        recipe.default_context_length,
    )
    manifest.update(
        {
            "model_id": recipe.model_id,
            "export": {"model_name": recipe.export_name, "artifact_name": relabeled_name},
            "task": recipe.task,
            "modalities": list(recipe.modalities),
        }
    )
    manifest["recipe"].update(
        {
            "version": recipe.recipe_version,
            "variant": variant.name,
            "source_config": variant.workflow_config,
            "source_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        }
    )
    manifest["artifact"]["components"] = list(recipe.components)
    manifest["parameters"].update(
        {
            "components": list(recipe.components),
            "precision": {"checkpoint": recipe.default_precision.to_dict()},
            "context_length": recipe.default_context_length,
            "prefill_chunk_length": recipe.default_prefill_chunk_length,
            "max_pe_length": recipe.max_pe_length,
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        lambda *_args, **_kwargs: pytest.fail("runtime must not load a relabeled Artifact"),
    )

    with pytest.raises(ModelError, match="installed model Recipe"):
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))


def test_inference_model_requires_effective_workflow_binding(artifact):
    manifest_path = artifact / "model_build_manifest.json"
    manifest = _complete_manifest(artifact)
    del manifest["recipe"]["effective_workflow_sha256"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelError, match="installed model Recipe"):
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))


def test_inference_model_safely_parses_inventoried_effective_workflow(artifact):
    manifest_path = artifact / "model_build_manifest.json"
    manifest = _complete_manifest(artifact)
    marker = artifact.parent / "unsafe-yaml-loaded"
    effective = artifact / "effective_workflow.yaml"
    command = json.dumps(f"touch {marker}")
    effective.write_text(f"!!python/object/apply:os.system [{command}]\n", encoding="utf-8")
    _refresh_inventoried_file(artifact, manifest, effective.name)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelError, match="safely parse"):
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))

    assert not marker.exists()


@pytest.mark.parametrize(
    ("workflow_field", "parameter_field"),
    [
        ("context_max_length", "context_length"),
        ("prefill_chunk_length", "prefill_chunk_length"),
        ("max_pe_length", "max_pe_length"),
    ],
)
def test_inference_model_binds_effective_workflow_lengths_to_manifest_parameters(
    artifact,
    workflow_field,
    parameter_field,
):
    manifest_path = artifact / "model_build_manifest.json"
    manifest = _complete_manifest(artifact)
    effective = artifact / "effective_workflow.yaml"
    effective_data = yaml.safe_load(effective.read_text(encoding="utf-8"))
    expected = manifest["parameters"][parameter_field]
    effective_data["export"]["model"][workflow_field] = expected + 1
    effective.write_text(yaml.safe_dump(effective_data, sort_keys=False), encoding="utf-8")
    _refresh_inventoried_file(artifact, manifest, effective.name)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelError, match="installed model Recipe") as exc_info:
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))

    mismatch = exc_info.value.details["mismatches"][f"effective_workflow.export.model.{workflow_field}"]
    assert mismatch == {"expected": expected, "actual": expected + 1}


@pytest.mark.parametrize(
    ("metadata_field", "parameter_field"),
    [
        ("context_max_length", "context_length"),
        ("prefill_chunk_length", "prefill_chunk_length"),
        ("max_pe_length", "max_pe_length"),
    ],
)
def test_inference_model_binds_runtime_metadata_lengths_to_manifest_parameters(
    artifact,
    metadata_field,
    parameter_field,
):
    manifest_path = artifact / "model_build_manifest.json"
    manifest = _complete_manifest(artifact)
    metadata_path = artifact / "golden_meta_info.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = manifest["parameters"][parameter_field]
    metadata["model_config"][metadata_field] = expected + 1
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    _refresh_inventoried_file(artifact, manifest, metadata_path.name)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelError, match="metadata lengths") as exc_info:
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))

    assert exc_info.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED
    assert exc_info.value.details["mismatches"][metadata_field] == {
        "expected": expected,
        "actual": expected + 1,
    }


def test_inference_model_binds_effective_workflow_model_name_across_qwen_relabel(artifact):
    from xhmodel_merak.model_build.resolver import _build_artifact_name

    manifest_path = artifact / "model_build_manifest.json"
    hf_config_path = artifact / "hf_config" / "config.json"
    hf_config_path.write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {"hidden_size": 5120, "num_hidden_layers": 64},
            }
        ),
        encoding="utf-8",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    qwen36 = get_recipe("qwen3.6-27b")
    manifest["model_id"] = qwen36.model_id
    manifest["export"]["artifact_name"] = _build_artifact_name(
        qwen36,
        qwen36.default_variant,
        qwen36.export_name,
        qwen36.default_precision,
        qwen36.default_prefill_chunk_length,
        qwen36.default_context_length,
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest = _complete_manifest(artifact)

    qwen38 = get_recipe("qwen3.8-27b")
    variant = qwen38.default_variant
    source_config = resolve_workflow_config(variant.workflow_config)
    relabeled_name = _build_artifact_name(
        qwen38,
        variant,
        qwen38.export_name,
        qwen38.default_precision,
        qwen38.default_prefill_chunk_length,
        qwen38.default_context_length,
    )
    manifest.update(
        {
            "model_id": qwen38.model_id,
            "export": {"model_name": qwen38.export_name, "artifact_name": relabeled_name},
            "task": qwen38.task,
            "modalities": list(qwen38.modalities),
        }
    )
    manifest["recipe"].update(
        {
            "version": qwen38.recipe_version,
            "variant": variant.name,
            "source_config": variant.workflow_config,
            "source_config_sha256": hashlib.sha256(source_config.read_bytes()).hexdigest(),
        }
    )
    manifest["parameters"].update(
        {
            "precision": {
                "checkpoint": qwen38.default_precision.to_dict(),
                "deployment": dict(qwen38.deployment_precision),
            },
            "components": list(qwen38.components),
            "golden_components": list(qwen38.golden_components),
            "max_pe_length": qwen38.max_pe_length,
            "vision_profile": qwen38.vision_profile,
            "vision_gears": list(qwen38.vision_gears),
        }
    )
    manifest["artifact"]["components"] = list(qwen38.components)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelError, match="installed model Recipe") as exc_info:
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))

    mismatch = exc_info.value.details["mismatches"]["effective_workflow.export.model.model_name"]
    assert mismatch == {"expected": "qwen3_8_27b", "actual": "qwen3_6_27b"}


def test_recipe_contract_accepts_registered_context_workflow(tmp_path):
    from xhmodel_merak.model_artifact.validation import validate_recipe_contract
    from xhmodel_merak.model_build.resolver import _build_artifact_name, _select_context_workflow

    recipe = get_recipe("qwen3.8-27b")
    variant = recipe.get_variant("dflash")
    assert variant is not None
    context_length = 262144
    workflow_config = _select_context_workflow(recipe, variant, context_length)
    config_path = resolve_workflow_config(workflow_config)
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    from xhmodel_merak.model_catalog.precision import deployment_precision_from_workflow

    effective_data = yaml.safe_load(config_path.read_text())
    effective_data["export"]["model"].update(
        {
            "model_name": recipe.export_name,
            "context_max_length": context_length,
            "prefill_chunk_length": recipe.default_prefill_chunk_length,
            "max_pe_length": recipe.max_pe_length,
        }
    )
    deployment = deployment_precision_from_workflow(recipe, effective_data)

    effective_workflow = tmp_path / "effective_workflow.yaml"
    effective_workflow.write_text(
        yaml.safe_dump(
            effective_data,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    components = list(recipe.components + variant.extra_components)
    manifest = {
        "schema_version": 3,
        "model_id": recipe.model_id,
        "export": {
            "model_name": recipe.export_name,
            "artifact_name": _build_artifact_name(
                recipe,
                variant,
                recipe.export_name,
                recipe.default_precision,
                recipe.default_prefill_chunk_length,
                context_length,
            ),
        },
        "task": recipe.task,
        "modalities": list(recipe.modalities),
        "recipe": {
            "version": recipe.recipe_version,
            "variant": variant.name,
            "source_config": workflow_config,
            "source_config_sha256": config_sha256,
            "effective_workflow": effective_workflow.name,
            "effective_workflow_sha256": hashlib.sha256(effective_workflow.read_bytes()).hexdigest(),
        },
        "parameters": {
            "precision": {
                "checkpoint": recipe.default_precision.to_dict(),
                "deployment": dict(recipe.deployment_precision),
            },
            "quant_method": recipe.quant_methods[0],
            "profile": recipe.default_profile,
            "context_length": context_length,
            "prefill_chunk_length": recipe.default_prefill_chunk_length,
            "max_pe_length": recipe.max_pe_length,
            "vision_profile": recipe.vision_profile,
            "vision_gears": list(recipe.vision_gears),
            "components": components,
            "golden_components": list(recipe.golden_components),
        },
        "artifact": {
            "components": components,
            "files": [
                {"path": effective_workflow.name},
            ],
        },
        "format": {"name": recipe.artifact_format, "version": get_artifact_format_spec(recipe.artifact_format).version},
    }

    manifest["parameters"]["precision"]["deployment"] = deployment
    validate_recipe_contract(tmp_path, manifest, recipe, variant)


@pytest.mark.parametrize("tamper", [None, "manifest", "workflow", "unsupported"])
def test_artifact_precision_is_bound_to_effective_workflow(artifact, tamper):
    from xhmodel_merak.model_artifact.validation import validate_recipe_contract

    manifest = _complete_manifest(artifact)
    recipe = get_recipe(manifest["model_id"])
    workflow_path = artifact / manifest["recipe"]["effective_workflow"]
    workflow = yaml.safe_load(workflow_path.read_text())
    workflow["export"]["model"]["visual_config"]["quant_scheme"]["quant_type"] = "w8a16h1_sefp"
    manifest["parameters"]["precision"]["deployment"]["visual"] = "w8a16h1_sefp"
    if tamper == "manifest":
        manifest["parameters"]["precision"]["deployment"]["visual"] = "w8a8h1_sefp"
    elif tamper == "workflow":
        workflow["export"]["model"]["visual_config"]["quant_scheme"]["quant_type"] = "w8a8h1_sefp"
    elif tamper == "unsupported":
        workflow["export"]["model"]["visual_config"]["quant_scheme"]["quant_type"] = "fp32"
        manifest["parameters"]["precision"]["deployment"]["visual"] = "fp32"
    workflow_path.write_text(yaml.safe_dump(workflow))
    manifest["recipe"]["effective_workflow_sha256"] = hashlib.sha256(workflow_path.read_bytes()).hexdigest()
    if tamper is None:
        validate_recipe_contract(artifact, manifest, recipe, recipe.default_variant)
    else:
        with pytest.raises(ModelError) as caught:
            validate_recipe_contract(artifact, manifest, recipe, recipe.default_variant)
        assert "parameters.precision.deployment" in caught.value.details["mismatches"]


def test_inference_model_rejects_removed_model_name_override(monkeypatch, artifact):
    from xhmodel_merak.model_build.resolver import _build_artifact_name

    manifest_path = artifact / "model_build_manifest.json"
    manifest = _complete_manifest(artifact)
    recipe = get_recipe(manifest["model_id"])
    variant = recipe.default_variant
    custom_model_name = "qwen_custom"
    manifest["export"]["model_name"] = custom_model_name
    manifest["export"]["artifact_name"] = _build_artifact_name(
        recipe,
        variant,
        custom_model_name,
        recipe.default_precision,
        recipe.default_prefill_chunk_length,
        recipe.default_context_length,
    )
    effective = artifact / "effective_workflow.yaml"
    effective_data = yaml.safe_load(effective.read_text(encoding="utf-8"))
    effective_data["export"]["model"]["model_name"] = custom_model_name
    effective.write_text(yaml.safe_dump(effective_data, sort_keys=False), encoding="utf-8")
    _refresh_inventoried_file(artifact, manifest, effective.name)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        lambda *_args, **_kwargs: _runtime_result(),
    )

    with pytest.raises(ModelError, match="installed model Recipe"):
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))


@pytest.mark.parametrize(
    ("path", "value", "mismatch_field"),
    [
        (("parameters", "components"), ["visual", "decode", "prefill"], "parameters.components"),
        (("parameters", "max_pe_length"), 131072, "parameters.max_pe_length"),
        (
            ("parameters", "precision", "deployment"),
            {"visual": "w16a16_sefp", "prefill": "w8a8h1_sefp", "decode": "w8a8h1_sefp"},
            "parameters.precision.deployment",
        ),
        (("parameters", "golden_components"), ["decode"], "parameters.golden_components"),
        (("parameters", "vision_profile"), "other", "parameters.vision_profile"),
        (("parameters", "vision_gears"), [96], "parameters.vision_gears"),
    ],
)
def test_inference_model_rejects_duplicated_provenance_mismatch(
    artifact,
    path,
    value,
    mismatch_field,
):
    manifest_path = artifact / "model_build_manifest.json"
    manifest = _complete_manifest(artifact)
    target = manifest
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelError, match="installed model Recipe") as exc_info:
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))

    assert mismatch_field in exc_info.value.details["mismatches"]


def test_inference_model_requires_integer_lengths_for_autoregressive_v2(artifact):
    manifest_path = artifact / "model_build_manifest.json"
    manifest = _complete_manifest(artifact)
    manifest["parameters"]["context_length"] = None
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ModelError, match="installed model Recipe") as exc_info:
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))

    assert "parameters.sequence_lengths" in exc_info.value.details["mismatches"]


def test_inference_model_rejects_absolute_runtime_metadata_paths(monkeypatch, artifact, tmp_path):
    outside = tmp_path / "outside.onnx"
    outside.write_bytes(b"outside")
    meta = artifact / "golden_meta_info.json"
    meta.write_text(json.dumps({"prefill_hmonnx": str(outside)}), encoding="utf-8")
    _complete_manifest(artifact)
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        lambda *_args, **_kwargs: pytest.fail("runtime must not read external metadata paths"),
    )

    with pytest.raises(ModelError, match="inside the Artifact root"):
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))


def test_inference_model_rejects_malformed_manifest_contracts(artifact):
    manifest_path = artifact / "model_build_manifest.json"
    original = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest = dict(original)
    manifest["schema_version"] = "2"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelError, match="schema_version"):
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))

    manifest = json.loads(json.dumps(original))
    manifest["parameters"]["components"] = ["decode"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelError, match="installed model Recipe"):
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))

    manifest = json.loads(json.dumps(original))
    manifest["artifact"]["metadata_file"] = r"C:\outside.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelError, match="inside the Artifact root"):
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))


def test_inference_model_requires_size_and_inventories_nested_golden(artifact):
    manifest_path = artifact / "model_build_manifest.json"
    manifest = _complete_manifest(artifact)
    del manifest["artifact"]["files"][0]["size"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelError, match="valid size"):
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))

    extra = artifact / "unlisted-golden.bin"
    extra.write_bytes(b"golden")
    manifest = _complete_manifest(artifact)
    manifest["artifact"]["golden_artifacts"] = {"decode": [extra.name]}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelError, match="file inventory"):
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))


def test_inference_model_rejects_manifest_symlink_and_string_golden(artifact, tmp_path):
    with pytest.raises(ModelError) as golden_error:
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact), golden="decode")
    assert golden_error.value.code == ErrorCode.INVALID_PARAMETER

    manifest_path = artifact / "model_build_manifest.json"
    outside = tmp_path / "outside-manifest.json"
    manifest_path.replace(outside)
    manifest_path.symlink_to(outside)
    with pytest.raises(ModelError) as symlink_error:
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(artifact))
    assert symlink_error.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED


def test_inference_model_rejects_an_artifact_root_symlink(artifact, tmp_path):
    linked_root = tmp_path / "linked-artifact"
    linked_root.symlink_to(artifact, target_is_directory=True)

    with pytest.raises(ModelError, match="Artifact root"):
        inference_model(inputs={"prompt": "hello"}, artifact_dir=str(linked_root))


@pytest.mark.parametrize(
    "validator_module",
    (SimpleNamespace(), SimpleNamespace(validate_artifact=None)),
    ids=("missing-attribute", "not-callable"),
)
def test_registered_format_validator_must_be_available_and_callable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    validator_module: SimpleNamespace,
) -> None:
    spec = ArtifactFormatSpec(
        name="autoregressive",
        validator="xhmodel_merak.missing_format:validate_artifact",
    )
    monkeypatch.setattr(artifact_validation, "get_artifact_format_spec", lambda _name: spec)
    monkeypatch.setattr(artifact_validation.importlib, "import_module", lambda _name: validator_module)

    with pytest.raises(ModelError, match="registered format contract") as exc_info:
        artifact_validation.validate_format_artifact(
            get_recipe("ling-3.0-flash"),
            tmp_path,
            {},
            tmp_path / "metadata.json",
        )

    assert exc_info.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED


def test_artifact_can_be_relocated_without_source_config_or_adapter_identity(monkeypatch, artifact):
    from dataclasses import replace

    from xhmodel_merak.model_catalog import resources

    original_recipe = get_recipe("qwen3.5-4b")
    renamed_recipe = replace(
        original_recipe,
        recipe_version="new-build-recipe",
        build_adapter="renamed.build",
        inference_adapter="renamed.inference",
    )
    monkeypatch.setattr(inference_api, "get_recipe", lambda _model_id: renamed_recipe)
    monkeypatch.setattr(
        resources, "resolve_workflow_config", lambda *_args: pytest.fail("inference must not reopen build-source YAML")
    )
    selected = []
    from xhmodel_merak.model_integrations.qwen.inference import adapter

    def installed_adapter(kind, name):
        selected.append((kind, name))
        assert (kind, name) == ("inference", "renamed.inference")
        return adapter

    monkeypatch.setattr(inference_api, "load_adapter", installed_adapter)
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation.quick_test_hmonnx",
        lambda *_args, **_kwargs: _runtime_result(),
    )
    manifest_path = artifact / "model_build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["recipe"]["source_config"] = "removed-build-source/workflow.yaml"
    manifest["recipe"]["source_config_sha256"] = "0" * 64
    # Unknown historical executable metadata must never control loading.
    manifest["runtime"] = {"inference_adapter": "os:system"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    destination = artifact.parent / "renamed-release"
    artifact.rename(destination)

    result = inference_model(
        artifact_dir=str(destination),
        inputs={"prompt": "hello"},
        expected_model_id=original_recipe.model_id,
        device="cpu",
    )
    assert result.model_id == original_recipe.model_id
    assert result.artifact_dir == str(destination)
    assert selected == [("inference", "renamed.inference")]
    assert "model_name" not in result.to_dict()
    assert not (destination / "recipe_identity.json").exists()


@pytest.mark.parametrize("field,value", [("name", "unknown"), ("version", 2), ("version", True)])
def test_artifact_format_version_is_the_runtime_contract(artifact, field, value):
    manifest_path = artifact / "model_build_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["format"][field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ModelError, match="installed model Recipe") as caught:
        inference_model(artifact_dir=str(artifact), inputs={"prompt": "hello"})
    assert f"format.{field}" in caught.value.details["mismatches"]


def test_expected_model_mismatch_does_not_clean_or_load_artifact(monkeypatch, artifact):
    output = artifact / "decode" / "step_0" / "output.npy"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"prior-golden")
    monkeypatch.setattr(inference_api, "load_adapter", lambda *_args: pytest.fail("no runtime import"))
    with pytest.raises(ModelError, match="expected model"):
        inference_model(artifact_dir=str(artifact), inputs={"prompt": "hello"}, expected_model_id="qwen3.5-9b")
    assert output.read_bytes() == b"prior-golden"


def test_inference_requires_explicit_artifact_directory_and_no_name_alias(artifact):
    with pytest.raises(TypeError, match="artifact_dir"):
        inference_model(inputs={"prompt": "hello"})
    with pytest.raises(TypeError):
        inference_model(ARTIFACT_NAME, {"prompt": "hello"}, artifact_dir=str(artifact))
    with pytest.raises(ModelError, match="allowed fields"):
        inference_model(artifact_dir=str(artifact), inputs={"prompt": "hello", "device": "cpu"})


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:6"])
def test_device_selection_is_explicit_without_silent_fallback(device):
    assert inference_api._resolve_device(device) == device
