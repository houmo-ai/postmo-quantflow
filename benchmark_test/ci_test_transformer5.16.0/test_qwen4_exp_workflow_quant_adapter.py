from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from xhmodel_merak.xh_llm.models.qwen4_exp.workflow import Qwen4ExpWorkflow
from xhmodel_merak.xh_llm.workflows.auto import AutoLLMWorkflow
from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig
from xhmodel_merak.xh_llm.workflows.result import QuantResult


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    REPO_ROOT
    / "configs_merak/workflows/xh2a/llm_models/qwen4_exp/flash_next/"
    / "qwen3_8_flash_next_hf_autoround_xh2a_256k.yaml"
)


def test_qwen4_exp_config_routes_through_auto_workflow(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    workflow = AutoLLMWorkflow.from_config(str(model_dir), str(CONFIG))

    assert isinstance(workflow, Qwen4ExpWorkflow)


def test_qwen4_exp_workflow_invokes_autoround_recipe(monkeypatch, tmp_path):
    captured = {}
    recipe = types.ModuleType("gptqmodel.recipes.qwen4_exp")

    def fake_quantize_qwen4_exp_autoround(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(output_dir=tmp_path / "quantized")

    recipe.quantize_qwen4_exp = fake_quantize_qwen4_exp_autoround
    monkeypatch.setitem(sys.modules, "gptqmodel.recipes.qwen4_exp", recipe)

    workflow = Qwen4ExpWorkflow.__new__(Qwen4ExpWorkflow)
    workflow.model_dir = "/data01/datasets/Qwen3.8-Flash-Next"
    config = WorkflowConfig.from_file(str(CONFIG))
    config.quant.update(method="autoround", iters=200, disable_opt_rtn=None, vision_quarot="mean_preserving")
    config.quant["runtime"]["dry_run"] = True
    config.export["model"]["model_config"] = {
        "model_type": "qwen4_exp",
        "text_config": {
            "model_type": "qwen4_exp_text",
            "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "qwen_sparse_attention"],
            "hc_count": 2,
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "ple_layer_ids": [1],
            "ngram_size": 2,
            "heads_per_ngram": 2,
            "ngram_vocab_size_base": 32,
            "ple_embed_dim": 32,
            "ple_conv_kernel_size": 4,
            "indexer_budget": 4,
            "indexer_compress_ratio": 2,
            "indexer_head_dim": 8,
            "indexer_kv_heads": 1,
            "indexer_n_heads": 2,
        },
    }
    result = workflow._quant_autoround_api(
        str(tmp_path),
        "cuda:0",
        config.quant,
        config.export["model"],
    )

    assert result.quanted_model_dir == str(tmp_path / "quantized")
    assert captured["method"] == "autoround"
    assert captured["model_dir"] == workflow.model_dir
    assert captured["bits"] == 8
    assert captured["attention_bits"] == 8
    assert captured["linear_attention_bits"] == 8
    assert captured["shared_expert_bits"] == 8
    assert captured["expert_bits"] == 4
    assert captured["group_size"] == 64
    assert captured["vision_quarot"] == "mean_preserving"
    assert captured["language_quarot"] is False
    assert captured["moe_routing"] == "bypass"
    assert captured["auto_forward_data_parallel"] is False
    assert captured["moe_routing_batch_size"] == 128
    assert captured["wait_for_submodule_finalizers"] is True
    assert captured["gc_mode"] == "on_stage_end"
    assert captured["dry_run"] is True
    assert captured["skip_eval"] is True
    assert captured["calibration_jsonl"].endswith("data/calib_data/NeelNanda-pile-10k.jsonl")


def test_qwen4_exp_workflow_rejects_language_rotation():
    config = WorkflowConfig.from_file(str(CONFIG))
    config.quant["rotation"] = "hadamard"

    with pytest.raises(ValueError, match="language/global QuaRot"):
        Qwen4ExpWorkflow._validate_qwen4_exp_quant_config(config.quant)


def test_qwen4_exp_workflow_accepts_partial_quantization_limit():
    config = WorkflowConfig.from_file(str(CONFIG))
    config.quant["max_quant_layers"] = 1

    Qwen4ExpWorkflow._validate_qwen4_exp_quant_config(config.quant)


def _tiny_stateful_model():
    from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

    config = Qwen4ExpTextConfig(
        max_position_embeddings=16,
        vocab_size=32, hidden_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_key_head_dim=4, linear_value_head_dim=4,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_conv_kernel_dim=4,
        moe_intermediate_size=8, shared_expert_intermediate_size=8,
        num_experts=2, num_experts_per_tok=1, hc_count=2, hc_lowrank=4,
        layer_types=["linear_attention", "full_attention"],
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8,
        indexer_budget=4, indexer_compress_ratio=4,
        ple_layer_ids=[1], ple_embed_dim=64, ple_conv_kernel_size=4,
        ngram_size=3, heads_per_ngram=1, ngram_vocab_size_base=5,
        make_ngram_vocab_size_divisible_by=8, split_ngram_parts=1, eos_token_id=0,
        rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.5},
    )
    config._attn_implementation = "eager"
    return Qwen4ExpForCausalLM(config).eval()


def _run_wrapper(model, samples, capacity):
    from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import (
        Qwen4ExpExportableAttention,
        build_qwen4_exp_text_export_wrapper,
    )
    from xhmodel_merak.xh_llm.models.qwen4_exp.qsa_host_controls import build_qsa_host_controls
    from xhquant.core.cache_tensor import CacheTensor

    wrapper = build_qwen4_exp_text_export_wrapper(
        model, input_sequence_length=samples[0].shape[1], cache_length=capacity,
    ).eval()
    cloned = [value.detach().clone() for value in samples]
    for index, spec in enumerate(wrapper.cache_contract.tensors, start=10):
        if spec.state_kind in {"kv_key", "kv_value", "qsa_indexer_key"}:
            cloned[index] = CacheTensor(cloned[index])
    qsa = next(module for module in wrapper.modules() if isinstance(module, Qwen4ExpExportableAttention))
    cloned.extend(build_qsa_host_controls(
        int(samples[1].item()), int(samples[2].item()), samples[0].shape[1], capacity,
        qsa.packed_qsa.candidate_topk, device=samples[0].device, dtype=samples[0].dtype,
    ))
    with torch.no_grad():
        return wrapper(*cloned)


def test_decode_sample_retains_real_gdr_qsa_ple_prefill_states():
    import copy

    from xhmodel_merak.xh_llm.models.qwen4_exp.hmonnx_export import (
        build_qwen4_exp_decode_sample_inputs,
        build_qwen4_exp_serving_sample_inputs,
    )

    torch.manual_seed(31)
    model = _tiny_stateful_model()
    reference = copy.deepcopy(model)
    prompt, token = torch.tensor([[2, 7, 11, 8, 9]]), torch.tensor([[13]])
    _, config, decode = build_qwen4_exp_decode_sample_inputs(
        model, prefill_ids=prompt, decode_ids=token, context_length=8,
    )
    from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import Qwen4ExpCacheContract

    contract = Qwen4ExpCacheContract.from_text_config(config)
    warm = dict(zip(contract.expected_keys(), decode[10:], strict=True))
    assert int(decode[1].item()) == prompt.shape[1]
    for spec in contract.tensors:
        assert torch.count_nonzero(warm[spec.name]) > 0, spec.name
        if "cache_length" in spec.shape:
            assert warm[spec.name].shape[spec.shape.index("cache_length")] == 8
    actual = _run_wrapper(model, decode, 8)
    ids = torch.cat((prompt, token), dim=-1)
    _, _, initial = build_qwen4_exp_serving_sample_inputs(reference, input_sequence_length=6, context_length=8)
    whole = (reference.model.embed_tokens(ids), *initial[1:3], ids.int(), *initial[4:])
    expected = _run_wrapper(reference, whole, 8)
    torch.testing.assert_close(actual[0], expected[0][:, -1:], rtol=2e-4, atol=2e-5)
    for observed, target in zip(actual[1:], expected[1:], strict=True):
        torch.testing.assert_close(observed, target, rtol=2e-4, atol=2e-5)
    _, _, cold = build_qwen4_exp_decode_sample_inputs(model, decode_ids=token, context_length=8)
    cleared = _run_wrapper(model, (*decode[:10], *cold[10:]), 8)
    assert not torch.allclose(actual[0], cleared[0], rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("capacity", [0, 5])
def test_decode_sample_rejects_invalid_capacity_before_prefill(capacity):
    from xhmodel_merak.xh_llm.models.qwen4_exp.hmonnx_export import build_qwen4_exp_decode_sample_inputs

    with pytest.raises(ValueError, match="context length|sample exceeds"):
        build_qwen4_exp_decode_sample_inputs(
            _tiny_stateful_model(), prefill_ids=torch.tensor([[2, 7, 11, 8, 9]]), context_length=capacity,
        )


def test_qwen4_exp_workflow_exports_hmonnx_and_golden(monkeypatch, tmp_path):
    import json

    from xhmodel_merak.xh_llm.models.qwen4_exp import hmonnx_export

    torch.manual_seed(47)
    model = _tiny_stateful_model()
    model_dir, output_dir = tmp_path / "model", tmp_path / "export"
    model_dir.mkdir()
    # Checkpoint/tokenizer IO and SDK publication are outside this workflow
    # contract. Prompt execution, state construction and logits remain real.
    monkeypatch.setattr(hmonnx_export, "load_qwen4_exp_model_for_export", lambda *args, **kwargs: model)
    observed = {}

    def publish(model, directory, **kwargs):
        assert kwargs["cache_length"] == 8
        assert kwargs["sample_inputs"][3].shape[-1] == 1
        observed.update(kwargs)
        # Exercise the frontend's in-place-input boundary: Golden must retain
        # the original prompt state even when capture writes to sample caches.
        observed["prefill_state"] = tuple(value.detach().clone() for value in kwargs["sample_inputs"][4:])
        for value in kwargs["sample_inputs"][4:]:
            value.zero_()
        return SimpleNamespace(
            hmonnx_path=str(directory / "model.onnx"),
            preflight_path=str(directory / "preflight.json"), io_manifest_path=str(directory / "io.json"),
        )

    monkeypatch.setattr(hmonnx_export, "export_qwen4_exp_hmonnx", publish)

    def run_graph(path, sample_inputs, golden_dir, **kwargs):
        golden_dir.mkdir(parents=True)
        for actual, saved in zip(sample_inputs[4:], observed["prefill_state"], strict=True):
            torch.testing.assert_close(actual, saved, rtol=0, atol=0)
        outputs = _run_wrapper(model, sample_inputs, observed["cache_length"])
        observed["logits"] = outputs[0]
        return {"logits": outputs[0]}

    monkeypatch.setattr(hmonnx_export, "run_qwen4_exp_hmonnx_golden", run_graph)
    workflow = Qwen4ExpWorkflow.__new__(Qwen4ExpWorkflow)
    workflow.model_dir = str(model_dir)
    workflow.workflow_config = WorkflowConfig.from_file(str(CONFIG))
    workflow.workflow_config.export["model"]["context_max_length"] = 8
    workflow.debug, workflow.seed = False, 42
    workflow._build_validation_decode = lambda *args: (
        SimpleNamespace(decode=lambda ids, **kwargs: "测试回复"),
        "rendered prompt", torch.tensor([[2, 7, 11, 8, 9]]),
    )
    head_calls = []
    hook = model.lm_head.register_forward_hook(lambda module, args, output: head_calls.append(tuple(output.shape)))
    try:
        result = workflow.export(QuantResult(raw_model_dir=str(model_dir), skipped=True), str(output_dir), "cpu")
    finally:
        hook.remove()
    assert head_calls == [(1, 5, 32), (1, 1, 32)]
    report = json.loads(Path(result.meta["conversation_report"]).read_text())
    assert report["prefill_state_source"] == "native_explicit_cache_wrapper"
    assert report["prefill_state_tensor_count"] == len(observed["prefill_state"])
    assert report["cache_length"] == 8
    assert report["generated_token_ids"] == [
        observed["sample_inputs"][3].item(), observed["logits"][:, -1].argmax(dim=-1).item(),
    ]
    assert result.meta["golden_outputs"] == ["logits"]
    assert Path(result.meta["golden_dir"]).is_dir()
    assert (output_dir / "golden_meta_info.json").is_file()
    assert workflow.dump_golden(result, "cpu", "你好") == result.meta["conversation_report"]



@pytest.mark.parametrize("algorithm, method", [
    ("gptqmodel", "rtn"), ("autoround", "autoround"), ("auto_round", "autoround"),
])
def test_qwen4_workflow_dispatches_default_method_without_shared_changes(monkeypatch, tmp_path, algorithm, method):
    from test_qwen4_exp_quant_adapter import _config

    captured = {}
    recipe = types.ModuleType("gptqmodel.recipes.qwen4_exp")

    def quantize(**kwargs):
        captured.update(kwargs)
        return {"output_dir": str(tmp_path / "quantized")}

    recipe.quantize_qwen4_exp = quantize
    monkeypatch.setitem(sys.modules, "gptqmodel.recipes.qwen4_exp", recipe)
    workflow = Qwen4ExpWorkflow.__new__(Qwen4ExpWorkflow)
    workflow.model_dir = str(tmp_path / "native")
    workflow.workflow_config = WorkflowConfig.from_file(str(CONFIG))
    workflow.workflow_config.quant.update(algorithm=algorithm, vision_quarot=None)
    for key in ("method", "iters", "disable_opt_rtn"):
        workflow.workflow_config.quant.pop(key, None)
    workflow.workflow_config.export["model"]["model_config"] = _config()
    workflow.seed = 42
    result = workflow.quant(str(tmp_path / "quantized"), "cpu")
    assert result.quanted_model_dir == str(tmp_path / "quantized")
    assert captured["method"] == method
    assert captured["iters"] == (0 if method == "rtn" else 200)
    if method == "rtn":
        assert captured["disable_opt_rtn"] is True
    else:
        assert "disable_opt_rtn" not in captured
    assert "vision_quarot" not in captured
    assert captured["language_quarot"] is False
