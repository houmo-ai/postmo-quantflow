"""Exercise Qwen4 weight publication with executable graphs and Host PLE assets."""

import copy
import hashlib
import json
import os
import shutil
import tarfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from onnx.reference import ReferenceEvaluator


def _write_graph(root, phase, bias, *, ple=False, packed_value=0):
    directory = root / phase
    directory.mkdir(parents=True)
    weight = np.arange(4, dtype=np.float32).reshape(2, 2)
    tensors = [
        numpy_helper.from_array(weight, f"{phase}_weight"),
        numpy_helper.from_array(np.array(bias, np.float32), "bias"),
    ]
    if phase == "decode":
        tensors.extend(
            (
                numpy_helper.from_array(weight.flatten(), "other_shape"),
                numpy_helper.from_array(weight.view(np.int32), "other_dtype"),
            )
        )
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])]
    if ple:
        inputs.append(helper.make_tensor_value_info("ple_embeddings_0", TensorProto.FLOAT16, [1, 1, 1, 64]))
    graph = helper.make_graph(
        [
            helper.make_node("MatMul", ["x", f"{phase}_weight"], ["product"]),
            helper.make_node("Add", ["product", "bias"], ["sum"]),
            helper.make_node("Constant", [], ["constant"], value=numpy_helper.from_array(np.ones(2, np.float32))),
            helper.make_node("Add", ["sum", "constant"], ["y"]),
        ],
        phase,
        inputs,
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
        tensors,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    if ple:
        np.full((8, 1, 32), packed_value, np.int8).tofile(directory / "ple.packed.bin")
        np.full((8, 1, 1), -10, np.int16).tofile(directory / "ple.exponent.bin")
        manifest = {
            "version": 1,
            "layers": [
                {
                    "layer_idx": 0,
                    "input_name": "ple_embeddings_0",
                    "embedding_dim": 64,
                    "output_dtype": "float16",
                    "ngram_size": 2,
                    "heads_per_ngram": 1,
                    "eos_token_id": 0,
                    "multipliers": [1, 3],
                    "vocab_sizes": [8],
                    "offsets": [0],
                    "packed": {"path": "ple.packed.bin", "shape": [8, 1, 32], "dtype": "int8"},
                    "exponent": {"path": "ple.exponent.bin", "shape": [8, 1, 1], "dtype": "int16"},
                }
            ],
        }
        item = model.metadata_props.add()
        item.key, item.value = "qwen4_exp_ple", json.dumps(manifest)
    path = directory / "model.onnx"
    onnx.save_model(
        model,
        path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="original.bin",
        size_threshold=0,
        convert_attribute=True,
    )
    return path.relative_to(root).as_posix()


def _bundle(root, *, mtp=False, different_ple=False):
    graphs = {
        phase: _write_graph(root, phase, bias, ple=True, packed_value=int(different_ple and phase == "decode"))
        for phase, bias in (("prefill", [1, 2]), ("decode", [3, 4]))
    }
    metadata = {f"{phase}_{kind}": path for phase, path in graphs.items() for kind in ("hmonnx", "onnx")}
    metadata["qwen4_exp_ple"] = {}
    for phase, path in graphs.items():
        model = onnx.load(root / path, load_external_data=False)
        manifest = json.loads(next(item.value for item in model.metadata_props if item.key == "qwen4_exp_ple"))
        for spec in manifest["layers"]:
            for kind in ("packed", "exponent"):
                spec[kind]["path"] = (Path(phase) / spec[kind]["path"]).as_posix()
        metadata["qwen4_exp_ple"][phase] = manifest
    if mtp:
        draft = {phase: _write_graph(root, phase, [7, 8]) for phase in ("mtp_prefill", "mtp_decode")}
        metadata["spec_decode"] = {
            "mode": "mtp",
            "runtime_contract_version": 5,
            "num_draft_tokens": 4,
            "verify_length": 5,
            "target_verify_graph": "decode",
            "target_verify_padding": 5,
            "target": {"verify_hmonnx": graphs["decode"], "verify_chunk_length": 5},
            "draft_prefill_onnx": draft["mtp_prefill"],
            "draft_decode_onnx": draft["mtp_decode"],
            "draft_onnx": draft["mtp_decode"],
            "draft": {"prefill_hmonnx": draft["mtp_prefill"], "decode_hmonnx": draft["mtp_decode"]},
        }
    (root / "golden_meta_info.json").write_text(json.dumps(metadata))
    return graphs


def _evaluate(path):
    model = onnx.load(path)
    feed = {"x": np.array([[2, 3]], np.float32)}
    if any(value.name == "ple_embeddings_0" for value in model.graph.input):
        feed["ple_embeddings_0"] = np.zeros((1, 1, 1, 64), np.float16)
    return ReferenceEvaluator(model).run(None, feed)[0]


def _payloads(path):
    model = onnx.load(path)
    tensors = [("initializer", tensor.name, tensor) for tensor in model.graph.initializer]
    tensors.extend(
        (str(index), attr.name, attr.t)
        for index, node in enumerate(model.graph.node)
        for attr in node.attribute
        if attr.type == attr.TENSOR
    )
    return [
        (owner, name, tensor.data_type, tuple(tensor.dims), numpy_helper.to_array(tensor).tobytes())
        for owner, name, tensor in tensors
    ]


def _structure(path):
    model = onnx.load(path, load_external_data=False)
    graph = copy.deepcopy(model.graph)
    tensors = list(graph.initializer)
    tensors.extend(attr.t for node in graph.node for attr in node.attribute if attr.type == attr.TENSOR)
    for tensor in tensors:
        tensor.ClearField("external_data")
        tensor.ClearField("raw_data")
        tensor.ClearField("data_location")
    return graph.SerializeToString()


def _ple_values(path):
    from xhmodel_merak.xh_llm.models.qwen4_exp.host_ple_inputs import HostPLEInputs

    host = HostPLEInputs.from_graph(path)
    return host.prepare(
        torch.tensor([[2]], dtype=torch.int32),
        {"ple_ngram_history_0": torch.tensor([[1]], dtype=torch.int32)},
        1,
        device="cpu",
    )[0]


def _file_snapshot(root):
    return {
        path.relative_to(root).as_posix(): (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def _assert_published(root, graphs, expected):
    from xhmodel_merak.xh_llm.models.qwen4_exp.ple_export import read_qwen4_exp_ple_specs

    metadata = json.loads((root / "golden_meta_info.json").read_text())
    assert metadata["shared_external_data"]["files"] == {phase: f"{phase}/weights.bin" for phase in graphs}
    assert (root / "prefill/weights.bin").samefile(root / "decode/weights.bin")
    for phase, path in graphs.items():
        graph_path = root / path
        assert metadata[f"{phase}_hmonnx"] == metadata[f"{phase}_onnx"] == path
        assert metadata[f"{phase}_hmonnx_md5"] == hashlib.md5(graph_path.read_bytes()).hexdigest()
        onnx.checker.check_model(graph_path)
        np.testing.assert_array_equal(_evaluate(graph_path), expected[phase]["output"])
        assert _payloads(graph_path) == expected[phase]["payloads"]
        assert _structure(graph_path) == expected[phase]["structure"]
        torch.testing.assert_close(_ple_values(graph_path), expected[phase]["ple"], atol=0, rtol=0)
        assert not (root / phase / "original.bin").exists()
        local = read_qwen4_exp_ple_specs(graph_path)
        rebased = copy.deepcopy(local)
        for spec in rebased:
            for kind in ("packed", "exponent"):
                spec[kind]["path"] = (Path(phase) / spec[kind]["path"]).as_posix()
        assert metadata["qwen4_exp_ple"][phase] == {"version": 1, "layers": rebased}
    spans = {}
    for phase, relative in graphs.items():
        model = onnx.load(root / relative, load_external_data=False)
        spans[phase] = {
            tensor.name: {entry.key: entry.value for entry in tensor.external_data}
            for tensor in model.graph.initializer
        }
    assert spans["prefill"]["prefill_weight"]["offset"] == spans["decode"]["decode_weight"]["offset"]
    assert spans["prefill"]["bias"]["offset"] != spans["decode"]["bias"]["offset"]
    assert spans["decode"]["decode_weight"]["offset"] != spans["decode"]["other_shape"]["offset"]
    assert spans["decode"]["decode_weight"]["offset"] != spans["decode"]["other_dtype"]["offset"]
    return metadata


def test_published_weights_and_ple_remain_shared_after_tar_move(tmp_path):
    from xhmodel_merak.model_integrations.qwen4_exp.build import normalize_export

    root = tmp_path / "artifact"
    graphs = _bundle(root)
    expected = {
        phase: {
            "output": _evaluate(root / path),
            "payloads": _payloads(root / path),
            "structure": _structure(root / path),
            "ple": _ple_values(root / path),
        }
        for phase, path in graphs.items()
    }
    normalize_export(root, SimpleNamespace(work_dir=root), None)
    _assert_published(root, graphs, expected)
    for kind in ("packed", "exponent"):
        assert (root / "prefill" / f"ple.{kind}.bin").samefile(root / "decode" / f"ple.{kind}.bin")
    archive = tmp_path / "model.tar"
    with tarfile.open(archive, "w") as out:
        out.add(root, arcname="model")
    shutil.rmtree(root)
    with tarfile.open(archive) as source:
        source.extractall(tmp_path / "relocated", filter="data")
    moved = tmp_path / "relocated/model"
    _assert_published(moved, graphs, expected)
    for kind in ("packed", "exponent"):
        assert (moved / "prefill" / f"ple.{kind}.bin").samefile(moved / "decode" / f"ple.{kind}.bin")


def test_wide_verify_and_drafts_survive_idempotent_publication(tmp_path):
    from xhmodel_merak.xh_llm.models.qwen4_exp.shared_weights import publish_qwen4_exp_shared_weights

    graphs = _bundle(tmp_path, mtp=True)
    metadata_path = tmp_path / "golden_meta_info.json"
    original_meta = json.loads(metadata_path.read_text())
    draft_snapshot = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for phase in ("mtp_prefill", "mtp_decode")
        for path in (tmp_path / phase).iterdir()
    }
    private = tmp_path / ".qwen4_fixed_head.pt"
    private.write_bytes(b"private fixed head state")
    manifest = {"status": "complete", "publication_marker": {"identity": "preserve"}, "graphs": {}}
    for phase, relative in graphs.items():
        manifest["graphs"][phase] = {
            "hmonnx": str(tmp_path / relative),
            "resources": {},
        }
    manifest_path = tmp_path / "export_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    publish_qwen4_exp_shared_weights(tmp_path)
    updated = json.loads(metadata_path.read_text())
    assert updated["spec_decode"] == original_meta["spec_decode"]
    assert updated["spec_decode"]["target"]["verify_hmonnx"] == updated["decode_hmonnx"]
    for phase, relative in graphs.items():
        assert updated[f"{phase}_hmonnx"] == relative
        assert updated[f"{phase}_hmonnx_md5"] == hashlib.md5((tmp_path / relative).read_bytes()).hexdigest()
    refreshed = json.loads(manifest_path.read_text())
    assert refreshed["publication_marker"] == manifest["publication_marker"]
    for phase, relative in graphs.items():
        assert refreshed["graphs"][phase]["resources"] == {
            name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
            for name in (relative, f"{phase}/weights.bin", f"{phase}/ple.packed.bin", f"{phase}/ple.exponent.bin")
        }
    assert {relative: (tmp_path / relative).read_bytes() for relative in draft_snapshot} == draft_snapshot
    assert private.read_bytes() == b"private fixed head state"
    published = _file_snapshot(tmp_path)
    publish_qwen4_exp_shared_weights(tmp_path)
    assert _file_snapshot(tmp_path) == published
    # Resume can reconstruct runtime metadata without the publication marker.
    updated.pop("shared_external_data")
    metadata_path.write_text(json.dumps(updated))
    graph_resources = {path: state for path, state in _file_snapshot(tmp_path).items() if path != metadata_path.name}
    publish_qwen4_exp_shared_weights(tmp_path)
    assert json.loads(metadata_path.read_text())["shared_external_data"]["files"] == {
        phase: f"{phase}/weights.bin" for phase in graphs
    }
    current_resources = {path: state for path, state in _file_snapshot(tmp_path).items() if path != metadata_path.name}
    assert current_resources == graph_resources


def test_different_ple_payloads_are_not_merged(tmp_path):
    from xhmodel_merak.xh_llm.models.qwen4_exp.shared_weights import publish_qwen4_exp_shared_weights

    graphs = _bundle(tmp_path, different_ple=True)
    payloads = {phase: _ple_values(tmp_path / relative) for phase, relative in graphs.items()}
    original_files = {
        (phase, kind): (tmp_path / phase / f"ple.{kind}.bin").read_bytes()
        for phase in graphs
        for kind in ("packed", "exponent")
    }
    publish_qwen4_exp_shared_weights(tmp_path)
    assert (tmp_path / "prefill/weights.bin").samefile(tmp_path / "decode/weights.bin")
    assert not (tmp_path / "prefill/ple.packed.bin").samefile(tmp_path / "decode/ple.packed.bin")
    assert (tmp_path / "prefill/ple.exponent.bin").samefile(tmp_path / "decode/ple.exponent.bin")
    for (phase, kind), expected in original_files.items():
        assert (tmp_path / phase / f"ple.{kind}.bin").read_bytes() == expected
    for phase, relative in graphs.items():
        torch.testing.assert_close(_ple_values(tmp_path / relative), payloads[phase], rtol=0, atol=0)


@pytest.mark.parametrize("failure_point", ["weights", "ple_after_first_table"])
def test_failed_hardlink_preserves_usable_original_package(tmp_path, monkeypatch, failure_point):
    from xhmodel_merak.xh_llm.models.qwen4_exp.shared_weights import publish_qwen4_exp_shared_weights

    graphs = _bundle(tmp_path)
    manifest = {
        "status": "complete",
        "graphs": {
            phase: {
                "hmonnx": str(tmp_path / relative),
                "resources": {},
            }
            for phase, relative in graphs.items()
        },
    }
    (tmp_path / "export_manifest.json").write_text(json.dumps(manifest))
    expected = {phase: _evaluate(tmp_path / path) for phase, path in graphs.items()}
    originals = {path: state[0] for path, state in _file_snapshot(tmp_path).items()}
    original_link = os.link
    failed = False

    def fail_link(source, target, *args, **kwargs):
        nonlocal failed
        target = Path(target)
        at_weights = failure_point == "weights" and target.name == "weights.bin"
        at_second_ple = (
            failure_point == "ple_after_first_table"
            and target.name.startswith(".ple.exponent.bin.")
            and target.suffix == ".link"
        )
        if at_second_ple:
            # Both canonical graphs and the first PLE link have already changed.
            for relative in graphs.values():
                path = tmp_path / relative
                assert path.read_bytes() != originals[relative]
                graph = onnx.load(path, load_external_data=False)
                assert all(
                    {item.key: item.value for item in tensor.external_data}["location"] == "weights.bin"
                    for tensor in graph.graph.initializer
                )
            assert (tmp_path / "prefill/ple.packed.bin").samefile(tmp_path / "decode/ple.packed.bin")
        if at_weights or at_second_ple:
            failed = True
            raise OSError("injected shared weight hardlink failure")
        return original_link(source, target, *args, **kwargs)

    monkeypatch.setattr(os, "link", fail_link)
    with pytest.raises(OSError, match="injected shared weight hardlink failure"):
        publish_qwen4_exp_shared_weights(tmp_path)
    assert failed
    assert set(_file_snapshot(tmp_path)) == set(originals)
    for relative, data in originals.items():
        assert (tmp_path / relative).read_bytes() == data
    for phase, path in graphs.items():
        onnx.checker.check_model(tmp_path / path)
        np.testing.assert_array_equal(_evaluate(tmp_path / path), expected[phase])


@pytest.mark.parametrize("inline_weights", [False, True])
def test_root_ple_resources_and_completed_records_survive_weight_sharing(tmp_path, inline_weights):
    from xhmodel_merak.xh_llm.models.qwen4_exp.shared_weights import publish_qwen4_exp_shared_weights

    graphs = _bundle(tmp_path)
    directory = tmp_path / "ple"
    directory.mkdir()
    model = onnx.load(tmp_path / graphs["prefill"], load_external_data=False)
    declaration = json.loads(next(item.value for item in model.metadata_props if item.key == "qwen4_exp_ple"))
    spec = declaration["layers"][0]
    for kind in ("packed", "exponent"):
        destination = directory / f"table.{kind}.bin"
        shutil.copyfile(tmp_path / "prefill" / spec[kind]["path"], destination)
        spec[kind]["path"] = destination.relative_to(tmp_path).as_posix()
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(declaration))
    for relative in graphs.values():
        path = tmp_path / relative
        model = onnx.load(path, load_external_data=inline_weights)
        del model.metadata_props[:]
        onnx.save(model, path)
        for payload in path.parent.glob("ple.*.bin"):
            payload.unlink()
    metadata_path = tmp_path / "golden_meta_info.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["qwen4_exp_ple"] = {"manifest": "ple/manifest.json"}
    metadata_path.write_text(json.dumps(metadata))
    baseline = {
        name: (_structure(tmp_path / path), _payloads(tmp_path / path), _evaluate(tmp_path / path))
        for name, path in graphs.items()
    }
    ple_bytes = {path.name: path.read_bytes() for path in directory.iterdir()}
    records = {name: {"hmonnx": str(tmp_path / path), "resources": {}} for name, path in graphs.items()}
    completion = tmp_path / "export_manifest.json"
    completion.write_text(json.dumps({"status": "complete", "graphs": records}))
    publish_qwen4_exp_shared_weights(tmp_path)
    assert (tmp_path / "prefill/weights.bin").samefile(tmp_path / "decode/weights.bin")
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == ple_bytes
    records = json.loads(completion.read_text())["graphs"]
    for name, relative in graphs.items():
        path = tmp_path / relative
        assert _structure(path) == baseline[name][0]
        assert _payloads(path) == baseline[name][1]
        np.testing.assert_array_equal(_evaluate(path), baseline[name][2])
        assert records[name]["resources"] == {
            resource: hashlib.sha256((tmp_path / resource).read_bytes()).hexdigest()
            for resource in (
                relative,
                f"{name}/weights.bin",
                "ple/manifest.json",
                "ple/table.packed.bin",
                "ple/table.exponent.bin",
            )
        }
        assert f"{name}/weights.bin" in records[name]["resources"]
        assert {"ple/manifest.json", "ple/table.packed.bin", "ple/table.exponent.bin"} <= set(
            records[name]["resources"]
        )
    snapshot = _file_snapshot(tmp_path)
    publish_qwen4_exp_shared_weights(tmp_path)
    assert _file_snapshot(tmp_path) == snapshot
