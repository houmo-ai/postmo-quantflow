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


def test_static_prefill_decode_retains_real_gdr_qsa_ple_states():
    import copy

    from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import Qwen4ExpCacheContract
    from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import (
        Qwen4ExpExportableAttention,
        build_qwen4_exp_text_export_wrapper,
    )
    from xhmodel_merak.xh_llm.models.qwen4_exp.static_runtime import Qwen4ExpCacheMixin, Qwen4ExpDataPreprocess

    torch.manual_seed(31)
    native = _tiny_stateful_model()
    decode_source, whole_source = copy.deepcopy(native), copy.deepcopy(native)
    contract = Qwen4ExpCacheContract.from_text_config(native.config)
    cache = Qwen4ExpCacheMixin(contract, batch_size=1, context_length=8, dtype=torch.float32)
    prompt, token = torch.tensor([[2, 7, 11, 8, 9]]), torch.tensor([[13]])

    def stage(source, width, state, ids, past):
        embedding = source.model.embed_tokens
        wrapper = build_qwen4_exp_text_export_wrapper(source, input_sequence_length=width, cache_length=8).eval()
        qsa = next(m for m in wrapper.modules() if isinstance(m, Qwen4ExpExportableAttention))
        processor = Qwen4ExpDataPreprocess(token_embedding=embedding, input_sequence_length=width,
                                         cache_mixin=state, pad_token_id=0, qsa_topk=qsa.packed_qsa.candidate_topk)
        with torch.no_grad():
            return wrapper(*processor({"input_ids": ids, "past_seq_length": past}))

    prefill = stage(native, 5, cache, prompt, 0)
    names = ["logits", *(f"present_{spec.name}" for spec in contract.tensors
                         if spec.state_kind not in {"kv_key", "kv_value", "qsa_indexer_key"})]
    cache.commit(prefill, names)
    for spec in contract.tensors:
        assert torch.count_nonzero(cache.tensors[spec.name]) > 0, spec.name
        if "cache_length" in spec.shape:
            assert cache.tensors[spec.name].shape[spec.shape.index("cache_length")] == 8
    actual = stage(decode_source, 1, cache, token, 5)
    cold = Qwen4ExpCacheMixin(contract, batch_size=1, context_length=8, dtype=torch.float32)
    expected = stage(whole_source, 6, cold, torch.cat((prompt, token), dim=-1), 0)
    torch.testing.assert_close(actual[0], expected[0][:, -1:], rtol=2e-4, atol=2e-5)
    for observed, target in zip(actual[1:], expected[1:], strict=True):
        torch.testing.assert_close(observed, target, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("capacity", [0, 5])
def test_static_decode_rejects_exhausted_context_before_cache_allocation(capacity):
    from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import Qwen4ExpCacheContract
    from xhmodel_merak.xh_llm.models.qwen4_exp.static_runtime import Qwen4ExpCacheMixin, Qwen4ExpDataPreprocess
    native = _tiny_stateful_model()
    state = Qwen4ExpCacheMixin(Qwen4ExpCacheContract.from_text_config(native.config),
                               batch_size=1, context_length=capacity, dtype=torch.float32)
    processor = Qwen4ExpDataPreprocess(token_embedding=native.model.embed_tokens, input_sequence_length=1,
                                      cache_mixin=state, pad_token_id=0)
    with pytest.raises(ValueError, match="context capacity"):
        processor({"input_ids": torch.tensor([[13]]), "past_seq_length": 5})
    assert not state._cache_initialized


@pytest.mark.parametrize("spec_decode_mode", [None, "mtp"])
def test_workflow_uses_real_model_entry_and_root_metadata(monkeypatch, tmp_path, spec_decode_mode):
    import json

    from xhmodel_merak.xh_llm.models.qwen4_exp.configuration_qwen4_exp import XHQwen4ExpModelConfig
    from xhmodel_merak.xh_llm.models.qwen4_exp.qwen4_exp_model import XHQwen4ExpModel

    native = _tiny_stateful_model()
    source, checkpoint, output = (tmp_path / name for name in ("source", "checkpoint", "export"))
    source.mkdir()
    checkpoint.mkdir()
    native.config.save_pretrained(source)
    native.config.save_pretrained(checkpoint)
    workflow = Qwen4ExpWorkflow.from_config(str(source), str(CONFIG))
    workflow.workflow_config.export["model"].update(
        context_max_length=8, prefill_chunk_length=4, visual_config=None, spec_decode_mode=spec_decode_mode,
    )
    observed = []

    def publish(model, directory):
        # Expensive SDK publication is the injection boundary. Shared workflow
        # must resolve and instantiate the actual Qwen4 model/config itself,
        # choosing the supplied checkpoint rather than the raw source.
        assert isinstance(model, XHQwen4ExpModel)
        assert isinstance(model.config, XHQwen4ExpModelConfig)
        assert Path(model.hf_model_dir) == checkpoint
        assert model.config.context_max_length == 8
        assert model.config.prefill_chunk_length == 4
        assert model.config.ple_quant_type == "sefp4"
        assert model.config.spec_decode_mode == spec_decode_mode
        assert model.config.mtp_source_dir == str(source)
        observed.append(model)
        metadata = {"model_config": model.config.to_dict(), "qwen4_exp_serving": {}}
        Path(directory, "golden_meta_info.json").write_text(json.dumps(metadata))
        Path(directory, "merak_config.json").write_text(json.dumps({"disable_prefix_caching": True}))
        return metadata

    monkeypatch.setattr(XHQwen4ExpModel, "export_hmonnx", publish)
    result = workflow.export(
        QuantResult(raw_model_dir=str(source), quanted_model_dir=str(checkpoint)), str(output), "cpu",
    )
    assert len(observed) == 1
    assert Path(result.work_dir) == output
    assert Path(result.config_file).is_file()
    assert json.loads((output / "golden_meta_info.json").read_text()) == result.meta
    assert json.loads((output / "merak_config.json").read_text())["disable_prefix_caching"] is True
    assert not list(output.glob("hmquant*"))
    assert not (output / "golden").exists()


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
