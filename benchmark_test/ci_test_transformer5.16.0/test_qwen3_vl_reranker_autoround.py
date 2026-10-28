"""Real AutoRound W4 production, export, relocation and scoring."""

from _unified_cases import cases_for


def test_public_autoround_w4_build_relocate_golden_and_inference(tmp_path, monkeypatch):
    import json
    import shutil

    import onnx
    from _unified_fixtures.qwen3_vl_reranker import prepare
    from _unified_preparation import configure_calibration
    from safetensors import safe_open

    from xhmodel_merak.model_build import build_model
    from xhmodel_merak.model_inference import inference_model
    from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.workflow import dump_reranker_golden

    case = cases_for("test_unified_model_qwen3_vl_reranker_2b.py")[0]
    kwargs, inputs = prepare(case, tmp_path, monkeypatch)
    configure_calibration(case, tmp_path / "calibration.jsonl", monkeypatch)
    kwargs.update(precision={"weight": "w4", "activation": "a8"}, quant_output_dir=str(tmp_path / "quant"))
    result = build_model(**kwargs)
    checkpoints = list((tmp_path / "quant").rglob("*.safetensors"))
    packed = []
    for path in checkpoints:
        with safe_open(path, framework="pt") as weights:
            packed.extend(name for name in weights.keys() if name.endswith("qweight"))
    assert len(packed) == 3 * 7
    moved = tmp_path / "moved_w4"
    shutil.move(kwargs["output_dir"], moved)
    shutil.rmtree(kwargs["model_dir"])
    shutil.rmtree(tmp_path / "quant")
    meta_file = moved / "golden_meta_info.json"
    meta = json.loads(meta_file.read_text())
    assert meta["model_config"]["quant_scheme"]["quant_type"] == "w8a8h1_sefp"
    graph = onnx.load(moved / meta["prefill_hmonnx"], load_external_data=False)
    assert sum(node.op_type == "FlashAttention" for node in graph.graph.node) == 3
    linears = [node for node in graph.graph.node if node.op_type == "Linear"]
    attrs = [{attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute} for node in linears]
    assert sum(attr.get("hmfp_weight_man_bit") == 4 for attr in attrs) == 21
    assert all(
        attr["mode"] == b"ssfp" and attr["hmfp_weight_hidden_bit"] == 0
        for attr in attrs
        if attr.get("hmfp_weight_man_bit") == 4
    )
    assert all(attr["hmfp_weight_nshare"] == 64 for attr in attrs)
    output = inference_model(artifact_dir=str(moved), inputs=inputs, device="cuda:0")
    repeated = inference_model(artifact_dir=str(moved), inputs=inputs, device="cuda:0")
    assert output.scores == repeated.scores and len(output.scores) == 2
    golden = dump_reranker_golden(meta_file, "cuda:0", {"query": inputs["query"], "documents": inputs["documents"][:1]})
    assert len(golden["golden_dirs"]) == 3
    synthetic = dump_reranker_golden(meta_file, "cuda:0", golden_name="golden_synthetic")
    assert len(synthetic["golden_dirs"]) == 3
    assert result.loadable
