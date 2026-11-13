"""Serving exports produce runtime packages without invoking chat or Golden."""

import json
from types import SimpleNamespace

import pytest
import torch

from xhmodel_merak.xh_llm.models.qwen4_exp import package_export


@pytest.mark.parametrize(
    "config",
    [
        {"batch_size": 2},
        {"context_max_length": 0},
        {"context_max_length": 4, "prefill_chunk_length": 8},
        {"spec_decode_mode": "mtp", "context_max_length": 1, "prefill_chunk_length": 1},
        {"spec_decode_mode": "mtp", "context_max_length": 3, "prefill_chunk_length": 1},
        *({"spec_decode_mode": "mtp", "verify_length": value} for value in (1, 0, -1, True, 4.0, "4", None)),
    ],
)
def test_invalid_serving_dimensions_fail_before_loading(tmp_path, monkeypatch, config):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid input loaded model weights")

    monkeypatch.setattr(package_export, "load_qwen4_exp_model_for_export", unexpected)
    with pytest.raises(ValueError):
        package_export._export_qwen4_exp_package(
            model_dir=tmp_path, output_dir=tmp_path / "out", device="cpu", model_config=config
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "2", None, 8])
def test_mtp_rejects_invalid_draft_budget_before_loading_weights(tmp_path, monkeypatch, count):
    def unexpected_load(*args, **kwargs):
        pytest.fail("unsupported MTP budget must fail before loading the checkpoint")

    monkeypatch.setattr(package_export, "load_qwen4_exp_model_for_export", unexpected_load)
    with pytest.raises(ValueError, match="draft|verify"):
        package_export._export_qwen4_exp_package(
            model_dir=tmp_path / "source",
            output_dir=tmp_path / "output",
            device="cpu",
            model_config={
                "spec_decode_mode": "mtp",
                "num_draft_tokens": count,
                "context_max_length": 8,
                "prefill_chunk_length": 4,
            },
        )


@pytest.mark.parametrize("precision", ["sefp4", "fp16"])
def test_ple_precision_config_is_explicit_and_defaults_to_sefp4(precision):
    from xhmodel_merak.xh_llm.models.qwen4_exp.configuration_qwen4_exp import XHQwen4ExpModelConfig

    assert XHQwen4ExpModelConfig(model_name="qwen4").ple_quant_type == "sefp4"
    assert XHQwen4ExpModelConfig(model_name="qwen4", ple_quant_type=precision).ple_quant_type == precision
    with pytest.raises(ValueError, match="PLE"):
        XHQwen4ExpModelConfig(model_name="qwen4", ple_quant_type="int4")


def test_resume_refuses_changing_ple_precision(tmp_path):
    import json

    output = tmp_path / "output"
    output.mkdir()
    (output / "export_manifest.json").write_text(json.dumps({"ple_quant_type": "fp16"}))
    with pytest.raises(ValueError, match="resume PLE"):
        package_export._export_qwen4_exp_package(
            model_dir=tmp_path / "source",
            output_dir=output,
            model_config={},
            device="cpu",
            low_memory=False,
            resume=True,
        )


@pytest.mark.parametrize("length,hc_width,valid", [(2, 8, True), (1, 8, False), (2, 4, False)])
def test_resume_validates_actual_graph_sequence_and_unmixed_hc(tmp_path, length, hc_width, valid):
    import onnx
    from onnx import TensorProto, helper

    path = tmp_path / "model.onnx"
    graph = helper.make_graph(
        [helper.make_node("Concat", ["inputs_embeds", "inputs_embeds"], ["target_hidden_states"], axis=2)],
        "hc_contract",
        [helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT16, [1, 2, 4])],
        [helper.make_tensor_value_info("target_hidden_states", TensorProto.FLOAT16, [1, 2, 8])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    kwargs = {"expected_attention_layers": 0, "expected_sequence_length": length, "expected_hc_width": hc_width}
    if valid:
        assert package_export._validate_graph(path, **kwargs) == {"Concat": 1}
    else:
        with pytest.raises(ValueError, match="incompatible"):
            package_export._validate_graph(path, **kwargs)


@pytest.mark.parametrize(
    "verify_width,operators,valid,fused",
    [
        (2, ["GDRRecurrentScan"] * 2, True, True),
        (5, ["GDRRecurrentScan"] * 5, True, True),
        (5, ["Identity"], True, False),
        (5, ["GDRRecurrentScan"], False, True),
        (5, ["GDRChunkScan"], False, True),
        (5, ["GDRChunkScan"], False, False),
        (5, ["GDRRecurrentScan"] * 5 + ["GDRBlockTriInverse"], False, True),
    ],
)
def test_verify_resume_requires_one_recurrent_update_per_static_token(tmp_path, operators, valid, fused, verify_width):
    import onnx
    from onnx import TensorProto, helper

    from xhmodel_merak.xh_llm.models.qwen4_exp.package_export import _validate_graph

    nodes = []
    previous = "inputs_embeds"
    for index, operator in enumerate(operators):
        output = f"out_{index}"
        nodes.append(helper.make_node(operator, [previous], [output], domain="test.gdr"))
        previous = output
    graph = helper.make_graph(
        nodes,
        "verify_contract",
        [helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT, [1, verify_width, 4])],
        [helper.make_tensor_value_info(previous, TensorProto.FLOAT, [1, verify_width, 4])],
    )
    graph.node.append(helper.make_node("Identity", [previous], ["present_delta_recurrent_state_0"]))
    graph.output.append(
        helper.make_tensor_value_info(
            "present_delta_recurrent_state_0",
            TensorProto.FLOAT,
            [1, verify_width, 4],
        )
    )
    path = tmp_path / "verify.onnx"
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("test.gdr", 1)]),
        path,
    )
    if valid:
        counts = _validate_graph(
            path,
            expected_attention_layers=0,
            expected_sequence_length=verify_width,
            expected_recurrent_layers=1,
            require_fused_recurrent=fused,
        )
        assert counts.get("GDRRecurrentScan", 0) == (verify_width if fused else 0)
        del graph.output[-1]
        onnx.save(
            helper.make_model(graph, opset_imports=[helper.make_opsetid("test.gdr", 1), helper.make_opsetid("", 17)]),
            path,
        )
        with pytest.raises(RuntimeError, match="return all 1 recurrent states"):
            _validate_graph(
                path,
                expected_attention_layers=0,
                expected_sequence_length=verify_width,
                expected_recurrent_layers=1,
                require_fused_recurrent=fused,
            )
    else:
        with pytest.raises(RuntimeError, match="recurrent token updates"):
            _validate_graph(
                path,
                expected_attention_layers=0,
                expected_sequence_length=verify_width,
                expected_recurrent_layers=1,
                require_fused_recurrent=fused,
            )


@pytest.mark.parametrize("fused", [None, False, True])
def test_prefill_only_resume_rejects_old_chunk_verify_in_final_validation(tmp_path, monkeypatch, fused):
    import json

    import onnx
    from onnx import TensorProto, helper

    source = tmp_path / "source"
    source.mkdir()
    target = tmp_path / "existing"
    target.mkdir()
    graphs = {}
    for phase, width in (("prefill", 4), ("decode", 2)):
        nodes = [helper.make_node("Identity", ["inputs_embeds"], ["target_hidden_states"])]
        if phase == "decode":
            nodes.append(helper.make_node("GDRChunkScan", ["inputs_embeds"], ["chunk"], domain="test.gdr"))
        graph = helper.make_graph(
            nodes,
            phase,
            [helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT16, [1, width, 4])],
            [helper.make_tensor_value_info("target_hidden_states", TensorProto.FLOAT16, [1, width, 4])],
        )
        path = target / f"{phase}.onnx"
        onnx.save(
            helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("test.gdr", 1)]),
            path,
        )
        graphs[phase] = {"hmonnx": str(path)}
    manifest_path = target / "export_manifest.json"
    previous = {
        "status": "partial",
        "source_model": str(source),
        "context_length": 8,
        "prefill_chunk_length": 4,
        "ple_quant_type": "sefp4",
        "cache_abi": "llm_cache_inplace_v1",
        "graphs": graphs,
    }
    manifest_path.write_text(json.dumps(previous))
    model = torch.nn.Module()
    monkeypatch.setattr(package_export, "load_qwen4_exp_model_for_export", lambda *args, **kwargs: model)
    monkeypatch.setattr(package_export, "extract_qwen4_exp_text_model", lambda model: model)
    monkeypatch.setattr(
        package_export,
        "extract_qwen4_exp_text_config",
        lambda model: SimpleNamespace(
            layer_types=["linear_attention"],
            hidden_size=4,
            hc_count=1,
        ),
    )

    def unexpected(*args, **kwargs):
        pytest.fail("Resume must validate existing graphs before exporting drafts or completing metadata")

    monkeypatch.setattr(package_export, "export_qwen4_exp_hmonnx", unexpected)
    monkeypatch.setattr(package_export, "build_qwen4_exp_serving_metadata", unexpected)
    (source / "config.json").write_text("{}")
    # Isolate the existing-graph audit; fixed-head recovery has a separate
    # actual two-process regression.
    monkeypatch.setattr(package_export, "_load_fixed_head_checkpoint", lambda *args: object())
    monkeypatch.setattr(package_export, "_fixed_head_bindings", lambda *args: {})
    config = {
        "spec_decode_mode": "mtp",
        "num_draft_tokens": 1,
        "context_max_length": 8,
        "prefill_chunk_length": 4,
        "verify_length": 2,
    }
    if fused is not None:
        config["use_fused_gdr"] = fused
    with pytest.raises(RuntimeError, match="recurrent token updates"):
        package_export._export_qwen4_exp_package(
            model_dir=source,
            output_dir=target,
            device="cpu",
            model_config=config,
            phase="prefill",
            resume=True,
            low_memory=False,
        )
    assert json.loads(manifest_path.read_text())["status"] == "partial"
    assert not (target / "golden_meta_info.json").exists()


@pytest.mark.parametrize("kind", ["key", "value", "indexer"])
def test_resume_rejects_redundant_public_inplace_kv_outputs(tmp_path, kind):
    import onnx
    from onnx import TensorProto, helper

    output = "present_qsa_indexer_key_cache_0" if kind == "indexer" else f"present_past_{kind}_cache_0"
    path = tmp_path / "old_package_export.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["cache"], [output])],
        "old_cache_output",
        [helper.make_tensor_value_info("cache", TensorProto.FLOAT16, [1, 1, 8, 8])],
        [helper.make_tensor_value_info(output, TensorProto.FLOAT16, [1, 1, 8, 8])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    with pytest.raises(RuntimeError, match="in-place K/V cache outputs; re-export"):
        package_export._validate_graph(path, expected_attention_layers=0, cache_abi="llm_cache_inplace_v1")


@pytest.mark.parametrize("previous_length,new_length", [(2, 4), (4, 2)])
def test_resume_refuses_changing_verify_length_before_loading(tmp_path, monkeypatch, previous_length, new_length):
    import json

    output = tmp_path / "output"
    output.mkdir()
    (output / "export_manifest.json").write_text(json.dumps({"verify_length": previous_length}))

    def unexpected(*args, **kwargs):
        pytest.fail("Resume width mismatch must fail before loading weights")

    monkeypatch.setattr(package_export, "load_qwen4_exp_model_for_export", unexpected)
    with pytest.raises(ValueError, match="resume verify length"):
        package_export._export_qwen4_exp_package(
            model_dir=tmp_path / "source",
            output_dir=output,
            device="cpu",
            resume=True,
            model_config={"spec_decode_mode": "mtp", "num_draft_tokens": 1, "verify_length": new_length},
        )


def test_resume_source_bindings_hash_selected_tensor_bytes_with_sharded_index(tmp_path):
    import json

    from safetensors.torch import save_file

    (tmp_path / "config.json").write_text("{}")
    tensors = {
        "lm_head.weight": torch.ones(8, 4),
        "mtp.fixture.weight": torch.ones(4),
        "model.layers.0.weight": torch.ones(4, 4),
    }
    shard = tmp_path / "model-00001-of-00001.safetensors"

    def save():
        save_file(tensors, shard)

    save()
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard.name for name in tensors}})
    )
    before = package_export._fixed_head_bindings(tmp_path, {})
    tensors["model.layers.0.weight"].add_(1)
    save()
    assert package_export._fixed_head_bindings(tmp_path, {}) == before
    tensors["lm_head.weight"].add_(1)
    save()
    head_changed = package_export._fixed_head_bindings(tmp_path, {})
    assert head_changed["source_head_tensors"] != before["source_head_tensors"]
    assert head_changed["mtp_source_tensors"] == before["mtp_source_tensors"]
    tensors["mtp.fixture.weight"].add_(1)
    save()
    assert package_export._fixed_head_bindings(tmp_path, {})["mtp_source_tensors"] != before["mtp_source_tensors"]


def test_resume_draft_resources_refuse_external_path_escape(tmp_path):
    import onnx
    from onnx import TensorProto, helper

    tensor = helper.make_tensor("weight", TensorProto.FLOAT, [1], [1.0])
    tensor.ClearField("float_data")
    tensor.data_location = TensorProto.EXTERNAL
    tensor.external_data.add(key="location", value="../outside")
    graph = helper.make_graph([], "unsafe", [], [], [tensor])
    path = tmp_path / "model.onnx"
    onnx.save(helper.make_model(graph), path)
    with pytest.raises(Exception, match="inside the Artifact root"):
        package_export._draft_resource_fingerprints(tmp_path, path)


def test_resume_source_bindings_support_hf_snapshot_symlinks(tmp_path):
    from safetensors.torch import save_file

    blob = tmp_path / "blobs" / "weights"
    blob.parent.mkdir()
    save_file({"lm_head.weight": torch.ones(8, 4), "mtp.fixture.weight": torch.ones(4)}, blob)
    source = tmp_path / "snapshot"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").symlink_to(blob)
    bindings = package_export._fixed_head_bindings(source, {})
    assert bindings["source_head_tensors"]["lm_head.weight"]["shape"] == [8, 4]
    assert bindings["mtp_source_tensors"]["mtp.fixture.weight"]["shape"] == [4]


def test_draft_resource_fingerprint_tracks_real_external_weights(tmp_path):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    path = tmp_path / 'draft.onnx'
    graph = helper.make_graph(
        [helper.make_node('Add', ['x', 'weight'], ['y'])], 'external_draft',
        [helper.make_tensor_value_info('x', TensorProto.FLOAT16, [8])],
        [helper.make_tensor_value_info('y', TensorProto.FLOAT16, [8])],
        [numpy_helper.from_array(np.arange(8, dtype=np.float16), name='weight')],
    )
    onnx.save_model(
        helper.make_model(graph), path, save_as_external_data=True,
        all_tensors_to_one_file=True, location='draft.weights', size_threshold=0,
    )
    before = package_export._draft_resource_fingerprints(tmp_path, path)
    sidecar = tmp_path / 'draft.weights'
    changed = bytearray(sidecar.read_bytes())
    changed[0] ^= 1
    sidecar.write_bytes(changed)
    after = package_export._draft_resource_fingerprints(tmp_path, path)
    assert before['draft.onnx'] == after['draft.onnx']
    assert before['draft.weights'] != after['draft.weights']


@pytest.mark.parametrize("options", [{"low_memory": "yes"}, {"low_memory": True, "ple_quant_type": "fp16"}])
def test_invalid_streamed_options_fail_before_output_or_loading(tmp_path, options):
    options = dict(options)
    low_memory = options.pop("low_memory")
    with pytest.raises(ValueError):
        package_export._export_qwen4_exp_package(
            model_dir=tmp_path / "missing",
            output_dir=tmp_path / "output",
            device="cpu",
            low_memory=low_memory,
            model_config=options,
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    "settings",
    [
        {"num_draft_tokens": 0},
        {"num_draft_tokens": True},
        {"num_draft_tokens": 4, "verify_length": 4},
        {"num_draft_tokens": 8},
    ],
)
def test_invalid_mtp_candidate_capacity_fails_before_loading_or_output(tmp_path, settings):
    with pytest.raises(ValueError):
        package_export._export_qwen4_exp_package(
            model_dir=tmp_path / "missing",
            output_dir=tmp_path / "output",
            device="cpu",
            model_config={"spec_decode_mode": "mtp", "context_max_length": 8, "prefill_chunk_length": 3, **settings},
        )
    assert not (tmp_path / "output").exists()


def test_resume_rejects_export_mode_change_before_loading(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "export_manifest.json").write_text(json.dumps({"export_mode": "standard"}))
    with pytest.raises(ValueError, match="export mode"):
        package_export._export_qwen4_exp_package(
            model_dir=tmp_path / "missing",
            output_dir=output,
            device="cpu",
            low_memory=True,
            resume=True,
            model_config={},
        )
