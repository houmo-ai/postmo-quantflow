"""Serving exports produce runtime packages without invoking chat or Golden."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from xhmodel_merak.xh_llm.models.qwen4_exp import serving_workflow as serving


@pytest.mark.parametrize("verify_length", [None, 2, 4])
@pytest.mark.parametrize("mtp", [False, True])
def test_export_complete_package_in_reserved_empty_stage(tmp_path, monkeypatch, mtp, verify_length):
    resolved_verify_length = 4 if verify_length is None else verify_length
    verify_phase = f"decode_m{resolved_verify_length}"
    import onnx
    from onnx import TensorProto, helper

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "tokenizer.json").write_text("{}")
    target = tmp_path / "stage"
    target.mkdir()
    text_config = SimpleNamespace(
        layer_types=["full_attention"],
        num_key_value_heads=1,
        head_dim=4,
        num_hidden_layers=1,
        hidden_size=4,
        eos_token_id=0,
        pad_token_id=0,
        hc_count=1,
        num_attention_heads=1,
        indexer_head_dim=4,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=2,
        ple_conv_kernel_size=2,
        ngram_size=2,
        ple_layer_ids=[],
        ple_hidden_size=4,
    )
    model = torch.nn.Module()
    model.embed_tokens = torch.nn.Embedding(8, 4)
    monkeypatch.setattr(serving, "load_qwen4_exp_model_for_export", lambda *a, **kw: model)
    monkeypatch.setattr(serving, "extract_qwen4_exp_text_model", lambda model: model)
    monkeypatch.setattr(serving, "extract_qwen4_exp_text_config", lambda model: text_config)
    monkeypatch.setattr(serving, "build_qwen4_exp_serving_sample_inputs", lambda *a, **kw: (None, None, ()))
    phases = []
    quant_scheme = {
        "quant_type": "w16a16h1_sefp",
        "nodes": {"lm_head": {"quant_type": "w8a16h1_sefp"}},
        "ops": {
            "GPTQPackedLinear": {"quant_type": "w8a8h1_sefp"},
            "GPTQPackedMoeBlock": {"quant_type": "w4a8h0_ssfp"},
            "MoeBlock": {"quant_type": "w4a8h0_ssfp"},
        },
    }

    def export(model, directory, **kwargs):
        phases.append(kwargs["input_sequence_length"])
        assert kwargs["ple_quant_type"] == "sefp4"
        assert kwargs["quant_config"] == quant_scheme
        assert kwargs["linear_attention_mode"] == ("recurrent" if directory.name == f"{verify_phase}_raw" else None)
        if mtp:
            assert kwargs["return_hidden_states"] is True
        directory.mkdir()
        path = directory / "model.onnx"
        graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"], name="identity")], "export_fixture",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 4])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT16, [1, 4])],
        )
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
        return SimpleNamespace(hmonnx_path=str(path))

    def convert(source, destination, **kwargs):
        destination.parent.mkdir()
        destination.write_bytes(Path(source).read_bytes())

    monkeypatch.setattr(serving, "export_qwen4_exp_hmonnx", export)
    monkeypatch.setattr(serving, "convert_qwen4_exp_kv_to_page_cache", convert)
    monkeypatch.setattr(serving, "convert_qwen4_exp_kv_to_llm_cache", convert)
    monkeypatch.setattr(serving, "_validate_page_graph", lambda *a, **kw: {"PageKVCache": 1, "FlashAttention": 1})
    draft_lengths = []
    if mtp:
        from xhmodel_merak.xh_llm.models.qwen4_exp import mtp_export

        def export_draft(source, directory, **kwargs):
            draft_lengths.append(kwargs["input_sequence_length"])
            directory.mkdir()
            path = directory / "model.onnx"
            path.write_bytes((target / "prefill_raw/model.onnx").read_bytes())
            return {"onnx_path": str(path), "input_sequence_length": kwargs["input_sequence_length"],
                    "cache_contract": {}, "hidden_size": 4, "hc_count": 1}

        monkeypatch.setattr(mtp_export, "export_qwen4_exp_mtp", export_draft)
    result = serving.export_qwen4_exp_serving_package(
        model_dir=source,
        output_dir=target,
        device="cpu",
        low_memory=False,
        model_config={
            "context_max_length": 8,
            "prefill_chunk_length": 4,
            "quant_scheme": quant_scheme,
            "spec_decode_mode": "mtp" if mtp else None,
            **({"verify_length": verify_length} if verify_length is not None else {}),
        },
    )
    assert phases == ([4, 1, resolved_verify_length] if mtp else [4, 1])
    if mtp:
        import json

        assert draft_lengths == [4, 1]
        metadata = json.loads((target / "golden_meta_info.json").read_text())
        spec = metadata["spec_decode"]
        assert spec["runtime_contract_version"] == 3
        assert spec["target"] == {"verify_hmonnx": f"{verify_phase}/qwen4_exp_{verify_phase}_llm_cache.onnx",
                                  "verify_chunk_length": resolved_verify_length}
        assert spec["draft"]["prefill_hmonnx"] == "mtp_prefill/model.onnx"
        assert spec["draft"]["decode_hmonnx"] == "mtp_decode/model.onnx"
        assert spec["draft"]["prefill_chunk_length"] == 4
        assert spec["draft"]["decode_chunk_length"] == 1
        assert metadata[f"{verify_phase}_hmonnx"] == spec["target"]["verify_hmonnx"]
        assert spec["verify_length"] == spec["target_verify_padding"] == resolved_verify_length
        assert spec["target_verify_graph"] == verify_phase
        assert spec["num_draft_tokens"] == 1
        assert result["verify_length"] == resolved_verify_length
    assert result["status"] == "complete"
    assert result["ple_quant_type"] == "sefp4"
    assert (target / "golden_meta_info.json").is_file()
    assert (target / "merak_config.json").is_file()
    assert (target / "hf_config" / "tokenizer.json").is_file()
    assert torch.load(target / "quant_embedding.pt", weights_only=True)["weight"].shape == (8, 4)
    assert not (target / "golden").exists()
    assert not (target / "qwen4_exp_hmonnx_conversation.json").exists()
    with pytest.raises(FileExistsError):
        serving.export_qwen4_exp_serving_package(model_dir=source, output_dir=target, device="cpu", model_config={})


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

    monkeypatch.setattr(serving, "load_qwen4_exp_model_for_export", unexpected)
    with pytest.raises(ValueError):
        serving.export_qwen4_exp_serving_package(
            model_dir=tmp_path, output_dir=tmp_path / "out", device="cpu", model_config=config
        )
    assert not (tmp_path / "out").exists()


def test_workflow_routes_serving_without_legacy_decode_validation(tmp_path, monkeypatch):
    import json

    from xhmodel_merak.xh_llm.models.qwen4_exp.workflow import Qwen4ExpWorkflow
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    config = SimpleNamespace(
        name="qwen-serving",
        export={"model": {"export_mode": "serving", "device_map": "cpu"}},
    )
    config.with_overrides = lambda overrides: config
    config.dump = lambda path: Path(path).write_text("export: {}\n")
    runner = Qwen4ExpWorkflow.__new__(Qwen4ExpWorkflow)
    runner.workflow_config = config
    runner.model_dir = str(tmp_path / "source")
    observed = {}

    def export(**kwargs):
        observed.update(kwargs)
        target = Path(kwargs["output_dir"])
        target.mkdir()
        (target / "golden_meta_info.json").write_text(json.dumps({"model_config": {"model_type": "qwen4_exp"}}))
        return {"status": "complete"}

    monkeypatch.setattr(serving, "export_qwen4_exp_serving_package", export)
    result = runner.export(
        QuantResult(raw_model_dir=runner.model_dir, skipped=True),
        str(tmp_path / "export"),
        "cpu",
    )
    assert observed["device_map"] == "cpu"
    assert result.meta["model_config"]["model_type"] == "qwen4_exp"
    assert Path(result.config_file).is_file()


def test_mtp_rejects_unsupported_draft_budget_before_loading_weights(tmp_path, monkeypatch):
    def unexpected_load(*args, **kwargs):
        pytest.fail("unsupported MTP budget must fail before loading the checkpoint")

    monkeypatch.setattr(serving, "load_qwen4_exp_model_for_export", unexpected_load)
    with pytest.raises(ValueError, match="exactly one draft token"):
        serving.export_qwen4_exp_serving_package(
            model_dir=tmp_path / "source", output_dir=tmp_path / "output", device="cpu",
            model_config={"spec_decode_mode": "mtp", "num_draft_tokens": 2},
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
        serving.export_qwen4_exp_serving_package(
            model_dir=tmp_path / "source", output_dir=output,
            model_config={}, device="cpu", low_memory=False, resume=True,
        )


@pytest.mark.parametrize("length,hc_width,valid", [(2, 8, True), (1, 8, False), (2, 4, False)])
def test_resume_validates_actual_graph_sequence_and_unmixed_hc(tmp_path, length, hc_width, valid):
    import onnx
    from onnx import TensorProto, helper

    path = tmp_path / "model.onnx"
    graph = helper.make_graph(
        [helper.make_node("Concat", ["inputs_embeds", "inputs_embeds"], ["target_hidden_states"], axis=2)],
        "hc_contract", [helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT16, [1, 2, 4])],
        [helper.make_tensor_value_info("target_hidden_states", TensorProto.FLOAT16, [1, 2, 8])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    kwargs = {"expected_attention_layers": 0, "expected_sequence_length": length, "expected_hc_width": hc_width}
    if valid:
        assert serving._validate_page_graph(path, **kwargs) == {"Concat": 1}
    else:
        with pytest.raises(ValueError, match="incompatible"):
            serving._validate_page_graph(path, **kwargs)


@pytest.mark.parametrize('verify_width', [2, 4])
@pytest.mark.parametrize('operators,valid,fused', [
    (['GDRRecurrentScan', 'GDRRecurrentScan'], True, True),
    (['Identity'], True, False),
    (['GDRRecurrentScan'], False, True),
    (['GDRChunkScan'], False, True),
    (['GDRChunkScan'], False, False),
    (['GDRRecurrentScan', 'GDRRecurrentScan', 'GDRBlockTriInverse'], False, True),
])
def test_verify_resume_requires_one_recurrent_update_per_static_token(tmp_path, operators, valid, fused, verify_width):
    import onnx
    from onnx import TensorProto, helper

    from xhmodel_merak.xh_llm.models.qwen4_exp.serving_workflow import _validate_page_graph

    if operators == ["GDRRecurrentScan", "GDRRecurrentScan"]:
        operators = ["GDRRecurrentScan"] * verify_width
    nodes = []
    previous = 'inputs_embeds'
    for index, operator in enumerate(operators):
        output = f'out_{index}'
        nodes.append(helper.make_node(operator, [previous], [output], domain='test.gdr'))
        previous = output
    graph = helper.make_graph(
        nodes, 'verify_contract',
        [helper.make_tensor_value_info('inputs_embeds', TensorProto.FLOAT, [1, verify_width, 4])],
        [helper.make_tensor_value_info(previous, TensorProto.FLOAT, [1, verify_width, 4])],
    )
    graph.node.append(helper.make_node('Identity', [previous], ['present_delta_recurrent_state_0']))
    graph.output.append(helper.make_tensor_value_info(
        'present_delta_recurrent_state_0', TensorProto.FLOAT, [1, verify_width, 4],
    ))
    path = tmp_path / 'verify.onnx'
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17), helper.make_opsetid('test.gdr', 1)]), path,
    )
    if valid:
        counts = _validate_page_graph(path, expected_attention_layers=0, expected_sequence_length=verify_width,
                                      expected_recurrent_layers=1, require_fused_recurrent=fused)
        assert counts.get('GDRRecurrentScan', 0) == (verify_width if fused else 0)
        del graph.output[-1]
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid('test.gdr', 1),
                                                        helper.make_opsetid('', 17)]), path)
        with pytest.raises(RuntimeError, match='return all 1 recurrent states'):
            _validate_page_graph(path, expected_attention_layers=0, expected_sequence_length=verify_width,
                                 expected_recurrent_layers=1, require_fused_recurrent=fused)
    else:
        with pytest.raises(RuntimeError, match='recurrent token updates'):
            _validate_page_graph(path, expected_attention_layers=0, expected_sequence_length=verify_width,
                                 expected_recurrent_layers=1, require_fused_recurrent=fused)


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
    for phase, width in (("prefill", 4), ("decode", 1), ("decode_m2", 2)):
        nodes = [helper.make_node("Identity", ["inputs_embeds"], ["target_hidden_states"])]
        if phase == "decode_m2":
            nodes.append(helper.make_node("GDRChunkScan", ["inputs_embeds"], ["chunk"], domain="test.gdr"))
        graph = helper.make_graph(
            nodes, phase, [helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT16, [1, width, 4])],
            [helper.make_tensor_value_info("target_hidden_states", TensorProto.FLOAT16, [1, width, 4])],
        )
        path = target / f"{phase}.onnx"
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17),
                                                        helper.make_opsetid("test.gdr", 1)]), path)
        graphs[phase] = {"page_hmonnx": str(path)}
    manifest_path = target / "export_manifest.json"
    previous = {"status": "partial", "source_model": str(source), "context_length": 8,
                "prefill_chunk_length": 4, "ple_quant_type": "sefp4",
                "cache_abi": "llm_cache_inplace_v1", "graphs": graphs}
    manifest_path.write_text(json.dumps(previous))
    model = torch.nn.Module()
    monkeypatch.setattr(serving, "load_qwen4_exp_model_for_export", lambda *args, **kwargs: model)
    monkeypatch.setattr(serving, "extract_qwen4_exp_text_model", lambda model: model)
    monkeypatch.setattr(serving, "extract_qwen4_exp_text_config", lambda model: SimpleNamespace(
        layer_types=["linear_attention"], hidden_size=4, hc_count=1,
    ))
    def unexpected(*args, **kwargs):
        pytest.fail("Resume must validate existing graphs before exporting drafts or completing metadata")
    monkeypatch.setattr(serving, "export_qwen4_exp_hmonnx", unexpected)
    monkeypatch.setattr(serving, "build_qwen4_exp_serving_metadata", unexpected)
    config = {"spec_decode_mode": "mtp", "context_max_length": 8, "prefill_chunk_length": 4, "verify_length": 2}
    if fused is not None:
        config["use_fused_gdr"] = fused
    with pytest.raises(RuntimeError, match="recurrent token updates"):
        serving.export_qwen4_exp_serving_package(
            model_dir=source, output_dir=target, device="cpu", model_config=config,
            phase="prefill", resume=True, low_memory=False,
        )
    assert json.loads(manifest_path.read_text())["status"] == "partial"
    assert not (target / "golden_meta_info.json").exists()


@pytest.mark.parametrize("kind", ["key", "value", "indexer"])
def test_resume_rejects_redundant_public_inplace_kv_outputs(tmp_path, kind):
    import onnx
    from onnx import TensorProto, helper

    output = "present_qsa_indexer_key_cache_0" if kind == "indexer" else f"present_past_{kind}_cache_0"
    path = tmp_path / "old_serving.onnx"
    graph = helper.make_graph(
        [helper.make_node("Identity", ["cache"], [output])], "old_cache_output",
        [helper.make_tensor_value_info("cache", TensorProto.FLOAT16, [1, 1, 8, 8])],
        [helper.make_tensor_value_info(output, TensorProto.FLOAT16, [1, 1, 8, 8])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
    with pytest.raises(RuntimeError, match="in-place K/V cache outputs; re-export"):
        serving._validate_page_graph(path, expected_attention_layers=0, cache_abi="llm_cache_inplace_v1")


@pytest.mark.parametrize("previous_length,new_length", [(2, 4), (4, 2)])
def test_resume_refuses_changing_verify_length_before_loading(tmp_path, monkeypatch, previous_length, new_length):
    import json

    output = tmp_path / "output"
    output.mkdir()
    (output / "export_manifest.json").write_text(json.dumps({"verify_length": previous_length}))

    def unexpected(*args, **kwargs):
        pytest.fail("Resume width mismatch must fail before loading weights")

    monkeypatch.setattr(serving, "load_qwen4_exp_model_for_export", unexpected)
    with pytest.raises(ValueError, match="resume verify length"):
        serving.export_qwen4_exp_serving_package(
            model_dir=tmp_path / "source", output_dir=output, device="cpu", resume=True,
            model_config={"spec_decode_mode": "mtp", "verify_length": new_length},
        )


@pytest.mark.parametrize("explicit,environment,expected_mode", [
    (None, None, "low_memory"),
    (None, "1", "low_memory"),
    (None, "0", "standard"),
    (None, "false", "standard"),
    (False, None, "standard"),
    (False, "1", "standard"),
    (True, "0", "low_memory"),
])
def test_direct_serving_mode_respects_explicit_choice_and_environment(
    tmp_path, monkeypatch, explicit, environment, expected_mode,
):
    import os

    from xhmodel_merak.xh_llm.models.qwen4_exp import streaming_loader

    variable = "HUGE_MODEL_EXPORT_ENABLED"
    if environment is None:
        monkeypatch.delenv(variable, raising=False)
    else:
        monkeypatch.setenv(variable, environment)

    def standard(*args, **kwargs):
        raise RuntimeError("selected standard")

    def streamed(*args, **kwargs):
        raise RuntimeError("selected low_memory")

    monkeypatch.setattr(serving, "load_qwen4_exp_model_for_export", standard)
    monkeypatch.setattr(streaming_loader, "Qwen4ExpStreamingLoader", streamed)
    with pytest.raises(RuntimeError, match=f"^selected {expected_mode}$"):
        serving.export_qwen4_exp_serving_package(
            model_dir=tmp_path / "source", output_dir=tmp_path / "output", model_config={},
            device="cpu", low_memory=explicit,
        )
    assert os.environ.get(variable) == environment
