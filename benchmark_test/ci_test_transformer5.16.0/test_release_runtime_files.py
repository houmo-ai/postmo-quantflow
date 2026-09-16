"""Delivery contains runtime dependencies, not the export/checkpoint workspace."""

import json

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from xhmodel_merak.errors import ModelError
from xhmodel_merak.model_build.runtime_assets import quant_runtime_asset_paths
from xhmodel_merak.model_integrations.autoregressive.build import _retain_runtime_files


def test_hf_assets_follow_config_and_local_code_dependencies(tmp_path):
    files = {
        "config.json": json.dumps({"auto_map": {"AutoConfig": "configuration_local.Config"}}),
        "tokenizer_config.json": json.dumps({"vocab_file": "assets/custom.vocab"}),
        "processor_config.json": "{}",
        "generation_config.json": "{}",
        "quantization_config.json": "{}",
        "tokenizer.json": "{}",
        "configuration_local.py": "from .helpers import VALUE\nclass Config: pass\n",
        "helpers.py": "VALUE = 1\n",
        "assets/custom.vocab": "hello\n",
        "chat_templates/default.jinja": "{{ messages }}",
        "eval_results.json": "{}",
        "mmlu_pro_results.json": "{}",
        "training_args.json": "{}",
        "debug.py": "raise RuntimeError('must not import checkpoint code')",
        "tokenizer_debug.json": "{}",
        "model.safetensors.index.json": "{}",
        "model.safetensors": "weights",
        "README.md": "not needed at runtime",
    }
    for name, content in files.items():
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    selected = {path.relative_to(tmp_path).as_posix() for path in quant_runtime_asset_paths(tmp_path)}
    assert selected == {
        "config.json",
        "tokenizer_config.json",
        "processor_config.json",
        "generation_config.json",
        "quantization_config.json",
        "tokenizer.json",
        "configuration_local.py",
        "helpers.py",
        "assets/custom.vocab",
        "chat_templates/default.jinja",
    }
    assert all((tmp_path / name).is_file() for name in files)  # Source is never cleaned.


@pytest.mark.parametrize("spec", [None, "mtp", "dflash", "dflash2"])
def test_release_retains_graph_weight_closure_and_flat_draft_directories(tmp_path, spec):
    root = tmp_path / "stage"
    root.mkdir()
    meta = {"hf_config": "hf_config", "quant_embedding": "quant_embedding.pt"}
    for name in ("effective_workflow.yaml", "merak_config.json", "quant_embedding.pt"):
        (root / name).write_text("{}")
    (root / "hf_config").mkdir()
    (root / "hf_config/config.json").write_text("{}")
    (root / "hf_config/tokenizer.json").write_text("{}")
    (root / "hf_config/eval_results.json").write_text("{}")
    components = ["prefill", "decode", "visual_m96"]
    if spec:
        components += [f"{spec}_draft_prefill", f"{spec}_draft_decode"]
    expected = {
        "effective_workflow.yaml",
        "merak_config.json",
        "golden_meta_info.json",
        "quant_embedding.pt",
        "hf_config/config.json",
        "hf_config/tokenizer.json",
    }
    if spec:
        draft_config = root / "draft_config"
        draft_config.mkdir()
        (draft_config / "config.json").write_text("{}")
        (draft_config / "eval_results.json").write_text("{}")
        meta["spec_decode"] = {"draft": {"hf_config": "draft_config"}}
        expected.add("draft_config/config.json")
    for component in components:
        directory = root / component
        directory.mkdir()
        graph_path = directory / "release_with_act.onnx"
        graph = helper.make_graph(
            [helper.make_node("Add", ["x", "weight"], ["y"])],
            component,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
            [numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32), "weight")],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        onnx.save_model(
            model,
            graph_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="release_external_data",
            size_threshold=0,
        )
        # Nested contracts, not filenames, identify the draft graphs.
        meta[component] = {"hmonnx": graph_path.relative_to(root).as_posix()}
        expected.update({f"{component}/release_with_act.onnx", f"{component}/release_external_data"})
    (root / "golden_meta_info.json").write_text(json.dumps(meta))
    for name in ("layers/layer_0.onnx", "decode/trace.json", "debug.log", "unused_external_data"):
        target = root / name
        target.parent.mkdir(exist_ok=True)
        target.write_text("temporary")
    _retain_runtime_files(root, meta)
    assert {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()} == expected
    assert not (root / "layers").exists()
    for component in components:
        reloaded = onnx.load(root / component / "release_with_act.onnx")
        np.testing.assert_array_equal(numpy_helper.to_array(reloaded.graph.initializer[0]), [1.0, 2.0])


def test_release_does_not_remove_files_if_required_external_weights_are_missing(tmp_path):
    graph = helper.make_graph([], "empty", [], [])
    tensor = numpy_helper.from_array(np.array([1.0], dtype=np.float32), "w")
    graph.initializer.append(tensor)
    onnx.save_model(
        helper.make_model(graph),
        tmp_path / "model.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="missing_external_data",
        size_threshold=0,
    )
    (tmp_path / "missing_external_data").unlink()
    sentinel = tmp_path / "temporary.txt"
    sentinel.write_text("still owned by stage")
    with pytest.raises(ModelError):
        _retain_runtime_files(tmp_path, {"hmonnx": "model.onnx"})
    assert sentinel.read_text() == "still owned by stage"


def test_sanitize_does_not_parse_unneeded_evaluation_reports(tmp_path):
    from xhmodel_merak.model_integrations.autoregressive.build import sanitize_artifact

    config = tmp_path / "hf_config"
    config.mkdir()
    (config / "config.json").write_text("{}")
    (config / "tokenizer.json").write_text("{}")
    (config / "eval_results.json").write_text("[1, 2, 3]")
    (config / "mmlu_pro_results.json").write_text("not even JSON")
    (tmp_path / "golden_meta_info.json").write_text(json.dumps({"hf_config": "hf_config"}))
    sanitize_artifact(tmp_path, None)
    assert {path.name for path in config.iterdir()} == {"config.json", "tokenizer.json"}


@pytest.mark.parametrize("value", ["missing.vocab", "../outside.vocab"])
def test_explicit_runtime_dependency_must_exist_and_stay_inside_source(tmp_path, value):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"vocab_file": value}))
    with pytest.raises(ValueError, match="runtime asset|Runtime asset"):
        quant_runtime_asset_paths(tmp_path)
