"""Streamed serving publication and real target/draft SDK execution."""

import json

import onnx
import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

from xhmodel_merak.xh_llm.models.qwen4_exp.serving_workflow import export_qwen4_exp_serving_package


def test_streamed_package_exports_and_executes_target_and_draft_graphs(tmp_path, monkeypatch):
    from xhmodel_merak.exec_backends import HMONNXInferenceConfig
    from xhmodel_merak.xh_llm.models.qwen4_exp.host_ple_runtime import (
        Qwen4ExpHostPLEInferenceV2,
        enable_qwen4_exp_inplace_kv,
    )
    from xhmodel_merak.xh_llm.models.qwen4_exp.mtp_model import Qwen4ExpMTPModel
    from xhmodel_merak.xh_llm.models.qwen4_exp.mtp_runtime import MTPGraphRuntime

    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "0")
    monkeypatch.delenv("HUGE_MODEL_EXPORT_ENABLED", raising=False)
    torch.manual_seed(103)
    config = Qwen4ExpTextConfig(
        max_position_embeddings=16,
        vocab_size=32, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, linear_key_head_dim=4, linear_value_head_dim=4,
        linear_num_key_heads=2, linear_num_value_heads=2, moe_intermediate_size=8,
        shared_expert_intermediate_size=8, num_experts=2, num_experts_per_tok=1,
        hc_count=2, hc_lowrank=4, layer_types=["linear_attention", "qwen_sparse_attention"],
        ple_layer_ids=[], indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8,
        indexer_budget=4, indexer_compress_ratio=4,
        rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.5}, eos_token_id=0,
        mtp={"num_hidden_layers": 1, "layer_types": ["full_attention"]},
    )
    native = Qwen4ExpForCausalLM(config).half().eval()
    checkpoint = tmp_path / "source"
    native.save_pretrained(checkpoint)
    assistant = Qwen4ExpMTPModel(config).half().eval()
    save_file({f"mtp.{name}": value.contiguous() for name, value in assistant.state_dict().items()},
              checkpoint / "mtp.safetensors")
    output = tmp_path / "artifact"
    report = export_qwen4_exp_serving_package(
        model_dir=checkpoint, output_dir=output, device="cpu",
        model_config={
            "context_max_length": 8, "prefill_chunk_length": 3, "spec_decode_mode": "mtp",
            "quant_scheme": {"quant_type": "w8a8h1_sefp", "ops": {"MoeBlock": {"quant_type": "w4a8h0_ssfp"}}},
            "use_fused_gdr": True, "cache_abi": "llm_cache_inplace_v1",
        },
    )
    assert report["status"] == "complete" and report["export_mode"] == "low_memory"
    meta = json.loads((output / "golden_meta_info.json").read_text())
    assert meta["spec_decode"]["verify_length"] == 4
    assert meta["spec_decode"]["target_verify_graph"] == "decode_m4"
    assert meta["spec_decode"]["num_draft_tokens"] == 1
    paths = [meta["prefill_hmonnx"], meta["decode_hmonnx"], meta["decode_m4_hmonnx"],
             meta["spec_decode"]["draft_prefill_onnx"], meta["spec_decode"]["draft_decode_onnx"]]
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "1")
    head_payloads = []
    for relative in paths:
        path = output / relative
        onnx.checker.check_model(str(path))
        graph = onnx.load(path, load_external_data=False)
        if relative == meta["decode_m4_hmonnx"]:
            inputs = {value.name: value for value in graph.graph.input}
            assert inputs["inputs_embeds"].type.tensor_type.shape.dim[1].dim_value == 4
            assert sum(node.op_type == "GDRRecurrentScan" for node in graph.graph.node) == 4
            assert not any(node.op_type == "GDRChunkScan" for node in graph.graph.node)
        assert not any(node.op_type == "PlaceHolder" for node in graph.graph.node)
        head, = [node for node in graph.graph.node if node.op_type == "Linear"
                 and any("lm_head.qweight" in name for name in node.input)]
        initializers = {value.name: value for value in graph.graph.initializer}
        from onnx import numpy_helper

        head_payloads.append(tuple(numpy_helper.to_array(initializers[name], base_dir=str(path.parent)).tobytes()
                                   for name in head.input[1:]))
        for node in graph.graph.node:
            if node.op_type == "MoeBlock":
                attributes = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
                assert attributes["hmfp_weight_man_bit"] == 4
        assert not any(value.name.startswith(("present_past_key_cache_", "present_past_value_cache_",
                                              "present_qsa_indexer_key_cache_")) for value in graph.graph.output)
        session = Qwen4ExpHostPLEInferenceV2(str(path), HMONNXInferenceConfig(exec_devices=["cpu"]))
        cache_names = {name for name in session.get_input_names()
                       if name.startswith(("past_key_cache_", "past_value_cache_", "qsa_indexer_key_cache_"))}
        assert any(name.startswith("qsa_indexer_key_cache_") for name in cache_names)
        paired = {name for name in cache_names if name.startswith(("past_key_cache_", "past_value_cache_"))}
        enable_qwen4_exp_inplace_kv(session, paired)
        feeds = {}
        for name in session.get_input_names():
            info = session.get_input(name)
            shape = tuple(info.shape)
            value = torch.zeros(shape, dtype=info.dtype)
            if info.name == "current_input_length":
                value.fill_(1)
            if info.name in cache_names or info.name.startswith("delta_recurrent_state_"):
                from xhquant.core.cache_tensor import CacheTensor

                value = CacheTensor(value)
            feeds[info.name] = value
        raw = session.run(feeds)
        outputs = dict(zip((value.name for value in graph.graph.output),
                           raw if isinstance(raw, (tuple, list)) else (raw,), strict=True))
        assert outputs["logits"].shape[-1] == config.vocab_size
        assert all(torch.isfinite(value).all() for value in outputs.values())
    assert all(value == head_payloads[0] for value in head_payloads)
    draft = meta["spec_decode"]["draft"]
    runtime = MTPGraphRuntime(
        str(output / draft["decode_hmonnx"]), cache_contract=draft["cache_contract"],
        context_length=8, device="cpu", cache_abi="llm_cache_inplace_v1",
        prefill_path=str(output / draft["prefill_hmonnx"]), prefill_chunk_length=3,
    )
    assert runtime.prefill_session is not None
    assert torch.load(output / "quant_embedding.pt", weights_only=True)["weight"].shape == (32, 16)


@pytest.mark.parametrize("options", [{"low_memory": "yes"}, {"low_memory": True, "ple_quant_type": "fp16"}])
def test_invalid_streamed_options_fail_before_output_or_loading(tmp_path, options):
    options = dict(options)
    low_memory = options.pop("low_memory")
    with pytest.raises(ValueError):
        export_qwen4_exp_serving_package(
            model_dir=tmp_path / "missing", output_dir=tmp_path / "output", device="cpu",
            low_memory=low_memory, model_config=options,
        )
    assert not (tmp_path / "output").exists()


def test_resume_rejects_export_mode_change_before_loading(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "export_manifest.json").write_text(json.dumps({"export_mode": "standard"}))
    with pytest.raises(ValueError, match="export mode"):
        export_qwen4_exp_serving_package(
            model_dir=tmp_path / "missing", output_dir=output, device="cpu",
            low_memory=True, resume=True, model_config={},
        )
