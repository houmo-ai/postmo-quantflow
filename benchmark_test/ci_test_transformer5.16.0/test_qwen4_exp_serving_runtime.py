from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from xhmodel_merak.xh_llm.hmonnx.text_llm_hmonnx_model import TextLLMHMONNXModel
from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
    QWEN_SPARSE_ATTENTION,
    Qwen4ExpCacheContract,
)
from xhmodel_merak.xh_llm.models.qwen4_exp.qwen4_exp_hmonnx_inference import (
    Qwen4ExpServingDataPreprocess,
    Qwen4ExpServingState,
    XHQwen4ExpHMONNXModel,
)
from xhmodel_merak.xh_llm.models.qwen4_exp.qwen4_exp_model import XHQwen4ExpModel
from xhmodel_merak.xh_llm.models.qwen4_exp.static_runtime import Qwen4ExpCacheMixin, Qwen4ExpDataPreprocess
from xhquant.core import CacheTensor


def _text_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="qwen4_exp_text",
        num_hidden_layers=3,
        layer_types=[LINEAR_ATTENTION, QWEN_SPARSE_ATTENTION, FULL_ATTENTION],
        hidden_size=16,
        hc_count=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        indexer_head_dim=4,
        indexer_budget=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        ple_conv_kernel_size=3,
        ngram_size=3,
        ple_layer_ids=[1],
        eos_token_id=151643,
    )


@pytest.mark.parametrize(
    "invalid",
    [
        {"batch_size": 2},
        {"context_max_length": 3},
        {"prefill_chunk_length": 0},
        {"prefill_chunk_length": 9},
        {"context_max_length": 8.5},
    ],
)
def test_static_model_configuration_rejects_invalid_geometry_before_checkpoint_load(invalid):
    from xhmodel_merak.xh_llm.models.qwen4_exp.configuration_qwen4_exp import XHQwen4ExpModelConfig

    values = {"context_max_length": 8, "prefill_chunk_length": 4, **invalid}
    with pytest.raises(ValueError, match="Qwen4 static export requires"):
        XHQwen4ExpModelConfig(hf_model="/missing/checkpoint", model_name="qwen4_tiny", **values)


def test_static_cache_mixin_accepts_model_lifecycle_arguments_and_allocates_contract_state() -> None:
    contract = Qwen4ExpCacheContract.from_text_config(_text_config())
    cache = Qwen4ExpCacheMixin(
        contract,
        batch_size=1,
        context_length=8,
        dtype=torch.float16,
        device="cpu",
        pad_token_id=151643,
        use_cache=True,
    )
    assert cache._cache_initialized is False
    values = cache.inputs_in_contract_order()
    assert cache._cache_initialized is True
    assert tuple(value.shape for value in values) == tuple(
        tuple(
            int(dim) if isinstance(dim, int) else {"batch": 1, "cache_length": 8, "qsa_pool_length": 2}[dim]
            for dim in spec.shape
        )
        for spec in contract.tensors
    )
    assert cache.tensors["ple_ngram_history_0"].tolist() == [[151643, 151643]]
    cache.clear_kv_cache()
    assert cache._cache_initialized is False
    assert cache.tensors == {}


def test_static_ple_history_uses_eos_independently_of_padding():
    cache = Qwen4ExpCacheMixin(
        Qwen4ExpCacheContract.from_text_config(_text_config()),
        batch_size=1,
        context_length=8,
        dtype=torch.float16,
        pad_token_id=0,
        ple_eos_token_id=151643,
    )
    cache.prepare_kv_cache()
    histories = [cache.tensors[spec.name] for spec in cache.contract.tensors if spec.state_kind == "ple_ngram_history"]
    assert histories
    assert all(torch.all(value == 151643).item() for value in histories)


def test_static_cache_dtype_survives_reset_and_keeps_cache_lists():
    contract = Qwen4ExpCacheContract.from_text_config(_text_config())
    cache = Qwen4ExpCacheMixin(contract, batch_size=1, context_length=8, dtype=torch.float16)
    keys, values = cache.past_key_caches, cache.past_value_caches
    cache.inputs_in_contract_order()
    cache.to(dtype=torch.float32)
    cache.clear_kv_cache()
    cache.inputs_in_contract_order()
    assert cache.past_key_caches is keys and cache.past_value_caches is values
    assert all(value.dtype == torch.float32 for value in (*keys, *values))
    assert cache.tensors["ple_ngram_history_0"].dtype == torch.int32
    assert cache.kvcache_config.cache_torch_dtype == torch.float32


@pytest.mark.parametrize("allocate_first", [False, True])
def test_static_cache_migration_before_allocation_and_after_clear(allocate_first):
    cache = Qwen4ExpCacheMixin(
        Qwen4ExpCacheContract.from_text_config(_text_config()),
        batch_size=1,
        context_length=8,
        dtype=torch.float16,
    )
    if allocate_first:
        cache.prepare_kv_cache()
        cache.clear_kv_cache()
    cache.to(device="cpu", dtype=torch.float32)
    assert not cache.tensors
    assert cache.kvcache_config.cache_torch_dtype == torch.float32
    cache.prepare_kv_cache()
    assert cache.tensors["key_cache_1"].dtype == torch.float32
    assert cache.tensors["ple_ngram_history_0"].dtype == torch.int32


@pytest.mark.parametrize("stage_width,past_length", [(4, 0), (4, 4), (1, 0), (1, 4)])
def test_static_data_preprocess_emits_fixed_qwen4_wrapper_abi(stage_width, past_length) -> None:
    contract = Qwen4ExpCacheContract.from_text_config(_text_config())
    cache = Qwen4ExpCacheMixin(contract, batch_size=1, context_length=8, dtype=torch.float16, device="cpu")
    embedding = torch.nn.Embedding(32, 16)
    processor = Qwen4ExpDataPreprocess(
        token_embedding=embedding,
        input_sequence_length=stage_width,
        cache_mixin=cache,
        pad_token_id=0,
    )
    ids = [1, 2] if stage_width == 4 else [1]
    padded = ids + [0] * (stage_width - len(ids))
    values = processor({"input_ids": torch.tensor([ids]), "past_seq_length": past_length})
    assert len(values) == 11
    assert len(values[10]) == len(contract.tensors)
    assert values[0].shape == (1, stage_width, 16)
    assert values[0].dtype == embedding.weight.dtype
    torch.testing.assert_close(values[0], embedding.weight[torch.tensor([padded])], rtol=0, atol=0)
    assert values[1].shape == values[2].shape == (1,)
    assert values[1].tolist() == [past_length]
    assert values[2].tolist() == [len(ids)]
    assert values[3].shape == (1, stage_width)
    assert values[3].tolist() == [padded]
    assert all(value.dtype == torch.int32 for value in values[1:10])
    for position in values[4:7]:
        assert position.shape == (1, stage_width)
        assert position.tolist() == [list(range(past_length, past_length + stage_width))]
    for position in values[7:10]:
        assert position.shape == (1, 1)
        assert position.tolist() == [[past_length // 4 * 4]]
    for spec, value in zip(contract.tensors, values[10], strict=True):
        assert value is cache.tensors[spec.name]


@pytest.mark.parametrize(
    "defect,message",
    [
        ("count", "output count"),
        ("undeclared_recurrent_omission", "output count"),
        ("unknown", "Unknown Qwen4-Exp cache output"),
        ("duplicate", "must be unique"),
        ("dtype", "incompatible dtype"),
        ("shape", "shape changed"),
    ],
)
def test_static_cache_commit_rejects_invalid_outputs_without_partial_updates(defect, message) -> None:
    contract = Qwen4ExpCacheContract.from_text_config(_text_config())
    cache = Qwen4ExpCacheMixin(contract, batch_size=1, context_length=8, dtype=torch.float16)
    initial = cache.inputs_in_contract_order()
    snapshots = tuple(value.detach().clone() for value in initial)
    names = ["logits", *(f"present_{spec.name}" for spec in contract.tensors)]
    values = [torch.zeros(1, 1, 32), *(torch.full_like(value, 3) for value in initial)]
    # Invalidate the final update, after several valid updates have been parsed.
    if defect == "count":
        values.pop()
    elif defect == "undeclared_recurrent_omission":
        # The declared ABI still includes this output. Explicit omission of
        # both its name and value is valid and has a separate regression below.
        values.pop(
            1 + next(index for index, spec in enumerate(contract.tensors) if spec.state_kind == "delta_recurrent")
        )
    elif defect == "unknown":
        names[-1] = "present_unknown_state"
    elif defect == "duplicate":
        names[-1] = names[-2]
    elif defect == "dtype":
        values[-1] = values[-1].to(torch.float32)
    else:
        values[-1] = values[-1][..., :1]
    with pytest.raises(ValueError, match=message):
        cache.commit(tuple(values), names)
    for original, snapshot, spec in zip(initial, snapshots, contract.tensors, strict=True):
        assert cache.tensors[spec.name] is original
        torch.testing.assert_close(cache.tensors[spec.name], snapshot, rtol=0, atol=0)


def test_static_cache_commit_uses_explicit_names_for_fused_recurrent_omission() -> None:
    contract = Qwen4ExpCacheContract.from_text_config(_text_config())
    cache = Qwen4ExpCacheMixin(contract, batch_size=1, context_length=8, dtype=torch.float16)
    initial = cache.inputs_in_contract_order()
    recurrent = cache.tensors["delta_recurrent_state_0"]
    recurrent.fill_(7)
    selected = [
        (spec, value)
        for spec, value in zip(contract.tensors, initial, strict=True)
        if spec.state_kind != "delta_recurrent"
    ]
    names = ["logits", *(f"present_{spec.name}" for spec, _ in selected)]
    logits = torch.zeros(1, 1, 32)
    values = (logits, *(torch.full_like(value, 3) for _, value in selected))
    assert cache.commit(values, names) is logits
    assert cache.tensors["delta_recurrent_state_0"] is recurrent
    assert recurrent.flatten().tolist() == [7] * recurrent.numel()
    for spec, previous in selected:
        torch.testing.assert_close(cache.tensors[spec.name], torch.full_like(previous, 3))
        if isinstance(previous, CacheTensor):
            assert cache.tensors[spec.name] is previous


def _static_tiny_model(tmp_path, *, use_fused_gdr=False):
    from test_qwen4_exp_workflow_quant_adapter import _tiny_stateful_model

    from xhmodel_merak.xh_llm.models.qwen4_exp.configuration_qwen4_exp import XHQwen4ExpModelConfig

    native = _tiny_stateful_model().half()
    # These wrapper tests supply native explicitly; disk loading is exercised
    # separately with a complete conditional-generation checkpoint below.
    native.config.save_pretrained(tmp_path)
    model = XHQwen4ExpModel(
        XHQwen4ExpModelConfig(
            hf_model=str(tmp_path),
            model_name="qwen4_tiny",
            context_max_length=8,
            prefill_chunk_length=4,
            visual_config=None,
            use_fused_gdr=use_fused_gdr,
        )
    )
    model.to_wrap(native)
    return model


def test_static_model_loads_complete_conditional_checkpoint_before_wrapping(tmp_path):
    from test_qwen4_exp_workflow_quant_adapter import _tiny_stateful_model
    from transformers import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration

    from xhmodel_merak.xh_llm.models.qwen4_exp.configuration_qwen4_exp import XHQwen4ExpModelConfig

    config = Qwen4ExpConfig(
        text_config=_tiny_stateful_model().config.to_dict(),
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 16,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "num_position_embeddings": 16,
        },
    )
    native = Qwen4ExpForConditionalGeneration(config).half().eval()
    native.save_pretrained(tmp_path)
    model = XHQwen4ExpModel(
        XHQwen4ExpModelConfig(
            hf_model=str(tmp_path),
            model_name="qwen4_tiny",
            context_max_length=8,
            prefill_chunk_length=4,
            visual_config=None,
        )
    )
    loaded = model.get_native_model()
    assert type(loaded) is type(native)
    expected, actual = native.state_dict(), loaded.state_dict()
    assert expected.keys() == actual.keys()
    for name, tensor in expected.items():
        assert actual[name].shape == tensor.shape
        assert actual[name].dtype == tensor.dtype
        torch.testing.assert_close(actual[name], tensor, rtol=0, atol=0)
    model.to_wrap(loaded)
    model.set_decode()
    inputs = model.get_data_preprocessor()({"input_ids": torch.tensor([[7]]), "past_seq_length": 0})
    with torch.inference_mode():
        logits = model(*inputs)
    assert logits.shape == (1, 1, 32)
    assert torch.isfinite(logits).all()


def test_static_fused_gdr_declares_phase_outputs_and_keeps_recurrent_storage(tmp_path):
    model = _static_tiny_model(tmp_path, use_fused_gdr=True)
    cache = model.get_kvcache_mixin()
    cache.prepare_kv_cache()
    recurrent = cache.tensors["delta_recurrent_state_0"]
    assert isinstance(recurrent, CacheTensor)
    for stage in ("prefill", "decode"):
        getattr(model, "set_" + stage)()
        names = model.get_export_cfg()["output_names"]
        assert ("present_delta_recurrent_state_0" in names) == (stage == "decode")
        inputs = model.get_data_preprocessor()({"input_ids": torch.tensor([[7]]), "past_seq_length": 0})
        with torch.inference_mode():
            logits = model(*inputs)
        assert logits.shape == (1, 4 if stage == "prefill" else 1, 32)
        assert torch.isfinite(logits).all()
        assert cache.tensors["delta_recurrent_state_0"] is recurrent
        assert recurrent.count_nonzero().item() > 0


def test_static_qsa_controls_survive_wrapper_release(tmp_path) -> None:
    from xhmodel_merak.xh_llm.models.qwen4_exp.qsa_host_controls import QSA_HOST_CONTROL_NAMES

    model = _static_tiny_model(tmp_path)
    model.get_kvcache_mixin().prepare_kv_cache()
    data = {"input_ids": torch.tensor([[2, 7]]), "past_seq_length": 0}
    before = model.get_data_preprocessor()(data)
    assert tuple(model.get_export_cfg()["input_names"][-5:]) == QSA_HOST_CONTROL_NAMES
    model.release_wraped_model()
    model._data_processor = None
    after = model.get_data_preprocessor()(data)
    assert len(before) == len(after)
    assert 10 + len(after[10]) == len(model.get_export_cfg()["input_names"])
    for actual, expected in zip(after[10][-5:], before[10][-5:], strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_static_wrap_dispatches_decode_with_one_token_and_commits_state(tmp_path) -> None:
    model = _static_tiny_model(tmp_path)
    cache = model.get_kvcache_mixin()
    cache.prepare_kv_cache()
    model.set_decode()
    values = model.get_data_preprocessor()({"input_ids": torch.tensor([[7]]), "past_seq_length": 0})
    assert values[0].shape[1] == 1
    with torch.inference_mode():
        logits = model.forward(*values)
    assert logits.shape == (1, 1, 32)
    assert torch.isfinite(logits).all()
    assert cache.tensors["ple_ngram_history_0"].tolist() == [[0, 7]]
    assert cache.tensors["delta_conv_state_0"].count_nonzero().item() > 0


@pytest.mark.parametrize("frontend", [False, True])
def test_static_model_dtype_migration_keeps_wrapper_cache_and_metadata_consistent(tmp_path, frontend):
    model = _static_tiny_model(tmp_path)
    if frontend:
        model.to_fronted()
    cache = model.get_kvcache_mixin()
    cache.prepare_kv_cache()
    model.to(dtype=torch.float32)
    assert model.embed_tokens.weight.dtype == torch.float32
    if frontend:
        for graph in model._frontend_model.values():
            assert all(value.dtype == torch.float32 for value in graph.parameters() if value.is_floating_point())
    for phase in ("prefill", "decode"):
        cache.prepare_kv_cache()
        getattr(model, "set_" + phase)()
        with torch.inference_mode():
            logits = model(*model.get_data_preprocessor()(model.get_dummy_inputs()))
        assert logits.dtype == torch.float32
        assert torch.isfinite(logits).all()
        assert cache.kvcache_config.cache_torch_dtype == torch.float32
        for spec in cache.contract.tensors:
            expected = torch.float32 if spec.dtype == "floating" else torch.int32
            assert cache.tensors[spec.name].dtype == expected
        cache.clear_kv_cache()


def test_static_model_captures_real_prefill_and_decode_frontends(tmp_path) -> None:
    from xhmodel_merak.xh_llm.models.qwen4_exp.qsa_host_controls import QSA_HOST_CONTROL_NAMES

    model = _static_tiny_model(tmp_path)
    assert model.frontend_type == "TorchFX"
    reference = {}
    reference_states = {}
    for stage in ("prefill", "decode"):
        getattr(model, "set_" + stage)()
        with torch.inference_mode():
            reference[stage] = model(*model.get_data_preprocessor()(model.get_dummy_inputs())).clone()
        reference_states[stage] = {
            name: value.detach().clone() for name, value in model.get_kvcache_mixin().tensors.items()
        }
        model.get_kvcache_mixin().clear_kv_cache()
    model.set_prefill()
    model.to_fronted()
    assert model._wrap_model is None
    assert model._decode_wrap_model is None
    for stage, length in (("prefill", 4), ("decode", 1)):
        getattr(model, "set_" + stage)()
        graph = model._frontend_model[stage]
        placeholders = [node for node in graph.graph.nodes if node.op == "placeholder"]
        embedding_info = placeholders[0].meta.get("val", placeholders[0].meta.get("tensor_meta"))
        assert tuple(embedding_info.shape) == (1, length, 16)
        model.get_kvcache_mixin().prepare_kv_cache()
        values = model.get_data_preprocessor()(model.get_dummy_inputs())
        assert values[0].shape == (1, length, 16)
        assert 10 + len(values[10]) == len(model.get_export_cfg()["input_names"])
        assert tuple(model.get_export_cfg()["input_names"][-5:]) == QSA_HOST_CONTROL_NAMES
        assert values[10][-2].shape == (1, length)
        assert values[10][-1].shape[:2] == (1, length)
        with torch.inference_mode():
            actual = model(*values)
        torch.testing.assert_close(actual, reference[stage], rtol=2e-3, atol=2e-3)
        for name, expected in reference_states[stage].items():
            torch.testing.assert_close(model.get_kvcache_mixin().tensors[name], expected, rtol=2e-3, atol=2e-3)
        model.get_kvcache_mixin().clear_kv_cache()


def test_serving_state_keeps_only_non_page_kv_tensors_and_resets() -> None:
    contract = Qwen4ExpCacheContract.from_text_config(_text_config())
    state = Qwen4ExpServingState(
        contract,
        batch_size=1,
        context_length=9,
        dtype=torch.float16,
        device="cpu",
        ple_eos_token_id=151643,
    )

    assert "key_cache_1" not in state.tensors
    assert "value_cache_1" not in state.tensors
    assert "qsa_indexer_key_cache_1" in state.tensors
    assert isinstance(state.tensors["qsa_indexer_key_cache_1"], CacheTensor)
    assert state.tensors["qsa_indexer_key_cache_1"].shape == (1, 1, 2, 4)
    assert state.tensors["qsa_indexer_key_tail_1"].shape == (1, 4, 4)
    assert state.group_start_position.shape == (3, 1, 1)
    state.group_start_position.fill_(9)
    state.tensors["qsa_indexer_key_cache_1"].fill_(3)
    state.reset()
    assert state.group_start_position.count_nonzero().item() == 0
    assert state.tensors["qsa_indexer_key_cache_1"].sum().item() == 0
    assert state.tensors["ple_ngram_history_0"].tolist() == [[151643, 151643]]


def test_explicit_cache_state_owns_full_capacity_kv_and_preserves_storage():
    from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import (
        qwen4_exp_cache_input_specs,
        resolve_qwen4_exp_cache_abi,
    )

    contract = Qwen4ExpCacheContract.from_text_config(_text_config())
    state = Qwen4ExpServingState(
        contract, batch_size=1, context_length=9, dtype=torch.float16, device="cpu", own_kv=True
    )
    specs = qwen4_exp_cache_input_specs(contract)
    assert specs["past_key_cache_0"].name == "key_cache_1"
    assert specs["past_key_cache_1"].name == "key_cache_2"
    values = state.inputs_in_contract_order(past_key_caches=None, past_value_caches=None, enable_page_attention=False)
    for spec, value in zip(contract.tensors, values, strict=True):
        assert value is state.tensors[spec.name]
        if spec.state_kind in {"kv_key", "kv_value"}:
            assert isinstance(value, CacheTensor)
            assert value.shape == (1, 2, 9, 4)
    owner = state.tensors["key_cache_1"]
    owner[:, :, :2].fill_(3)
    state.commit_outputs((torch.zeros(1), owner[:, :, :2]), ("logits", "present_past_key_cache_0"))
    assert state.tensors["key_cache_1"] is owner
    assert owner[:, :, :2].sum() == 48
    assert resolve_qwen4_exp_cache_abi() == "page_kv_v1"
    assert resolve_qwen4_exp_cache_abi("llm_cache_inplace_v1") == "llm_cache_inplace_v1"
    import pytest

    with pytest.raises(ValueError, match="Unsupported Qwen4 cache ABI"):
        resolve_qwen4_exp_cache_abi("unknown")


def test_runtime_exposes_vllm_adapter_reset_contract() -> None:
    contract = Qwen4ExpCacheContract.from_text_config(_text_config())
    runtime = XHQwen4ExpHMONNXModel.__new__(XHQwen4ExpHMONNXModel)
    runtime._qwen4_state = Qwen4ExpServingState(
        contract,
        batch_size=1,
        context_length=9,
        dtype=torch.float16,
        device="cpu",
    )
    runtime._qwen4_state.tensors["qsa_indexer_key_cache_1"].fill_(5)

    runtime.reset_internal_state()

    assert runtime._qwen4_state.tensors["qsa_indexer_key_cache_1"].sum().item() == 0
    assert callable(runtime.reset_qwen4_state)


def test_recurrent_cache_survives_fused_updates_and_unfused_outputs() -> None:
    state = Qwen4ExpServingState(
        Qwen4ExpCacheContract.from_text_config(_text_config()),
        batch_size=1,
        context_length=9,
        dtype=torch.float16,
        device="cpu",
    )
    recurrent = state.tensors["delta_recurrent_state_0"]
    assert isinstance(recurrent, CacheTensor)
    inputs = state.inputs_in_contract_order(past_key_caches=None, past_value_caches=None, enable_page_attention=True)
    assert any(value is recurrent for value in inputs)
    logits = torch.zeros(1, 1, 8)
    # Fused scan publishes its final state in place, even without a state output.
    recurrent.fill_(2)
    state.commit_outputs((logits,), ("logits",))
    assert state.tensors["delta_recurrent_state_0"] is recurrent
    assert recurrent.sum().item() == recurrent.numel() * 2
    # Unfused serving graphs publish a plain output tensor for the same state.
    updated = torch.full(recurrent.shape, 3, dtype=recurrent.dtype)
    assert not isinstance(updated, CacheTensor)
    state.commit_outputs((logits, updated), ("logits", "present_delta_recurrent_state_0"))
    assert state.tensors["delta_recurrent_state_0"] is recurrent
    torch.testing.assert_close(recurrent, updated)
    state.to(dtype=torch.float32)
    assert state.tensors["delta_recurrent_state_0"] is recurrent
    assert isinstance(recurrent, CacheTensor) and recurrent.dtype == torch.float32
    state.reset()
    assert isinstance(state.tensors["delta_recurrent_state_0"], CacheTensor)
    assert state.tensors["delta_recurrent_state_0"].count_nonzero().item() == 0


def test_serving_preprocessor_preserves_valid_length_for_padded_prefill() -> None:
    contract = Qwen4ExpCacheContract.from_text_config(_text_config())
    state = Qwen4ExpServingState(
        contract,
        batch_size=1,
        context_length=256,
        dtype=torch.float16,
        device="cpu",
    )
    processor = Qwen4ExpServingDataPreprocess(
        token_embedding=torch.nn.Embedding(32, 16),
        input_sequence_length=64,
        state=state,
        past_key_caches=None,
        past_value_caches=None,
        enable_page_attention=True,
        pad_token_id=0,
    )

    net_inputs = processor(
        {
            "input_ids": torch.ones((1, 3), dtype=torch.long),
            "past_seq_length": 128,
            "current_input_length": 3,
        }
    )

    assert net_inputs[1].tolist() == [128]
    assert net_inputs[2].tolist() == [3]
    assert net_inputs[0].shape == (1, 64, 16)
    assert net_inputs[3].shape == (1, 64)
    assert net_inputs[3][0, :3].tolist() == [1, 1, 1]
    assert net_inputs[3][0, 3:].tolist() == [0] * 61


def test_serving_preprocessor_prepares_host_qsa_controls_for_padded_prefill() -> None:
    contract = Qwen4ExpCacheContract.from_text_config(_text_config())
    state = Qwen4ExpServingState(contract, batch_size=1, context_length=64, dtype=torch.float16, device="cpu")
    processor = Qwen4ExpServingDataPreprocess(
        token_embedding=torch.nn.Embedding(32, 16),
        input_sequence_length=4,
        state=state,
        past_key_caches=None,
        past_value_caches=None,
        enable_page_attention=True,
        pad_token_id=0,
        qsa_control_spec=(2, torch.float16),
    )
    values = processor({"input_ids": torch.tensor([[1, 2]]), "past_seq_length": 3})
    offset, new_count, write_start, query_valid, mask = values[-5:]
    assert (offset.item(), new_count.item(), write_start.item()) == (1, 1, 0)
    assert query_valid.tolist() == [[True, True, False, False]]
    assert mask.shape == (1, 4, 1, 11) and mask.dtype == torch.float16
    assert (mask[0, 2, 0] == 0).nonzero().flatten().tolist() == []


def test_qwen38_export_cli_defaults_to_256_token_prefill(monkeypatch) -> None:
    script_path = (
        Path(__file__).resolve().parents[2]
        / "examples_merak/llm/qwen4_exp/debug_scripts/export_qwen4_exp_serving_hmonnx.py"
    )
    spec = importlib.util.spec_from_file_location("qwen38_serving_export_cli", script_path)
    assert spec is not None and spec.loader is not None
    export_qwen38_next_serving_hmonnx = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export_qwen38_next_serving_hmonnx)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_qwen4_exp_serving_hmonnx.py",
            "--model-dir",
            "/tmp/model",
            "--output-dir",
            "/tmp/export",
            "--config",
            "/tmp/config.yaml",
        ],
    )

    args = export_qwen38_next_serving_hmonnx._parse_args()

    assert args.prefill_chunk_length == 256


@pytest.mark.parametrize("verify_length", [2, 4])
def test_verify_dispatches_independent_target_sdk_graph(tmp_path, verify_length):
    import onnx
    from onnx import TensorProto, helper

    from xhmodel_merak.exec_backends import HMONNXInferenceConfig, HMONNXInferenceV2

    path = tmp_path / "verify.onnx"
    graph = helper.make_graph(
        [
            helper.make_node("Add", ["x", "x"], ["logits"], name="verify_logits"),
            helper.make_node("Identity", ["x"], ["target_hidden_states"], name="verify_hidden"),
        ],
        "independent_verify",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, verify_length, 4])],
        [
            helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, verify_length, 4]),
            helper.make_tensor_value_info("target_hidden_states", TensorProto.FLOAT, [1, verify_length, 4]),
        ],
    )
    proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    proto.ir_version = 9
    onnx.save(proto, path)
    session = HMONNXInferenceV2(str(path), session_config=HMONNXInferenceConfig(exec_devices=["cpu"]))
    model = object.__new__(XHQwen4ExpHMONNXModel)
    model.verify_model = SimpleNamespace(hmonnx_session=session, forward=session.forward)
    model.prefill_model = model.decode_model = None
    model._qwen4_verify_length = verify_length
    model._data_processor = None
    model._qwen4_state = Qwen4ExpServingState(
        Qwen4ExpCacheContract.from_text_config(_text_config()),
        batch_size=1,
        context_length=8,
        dtype=torch.float32,
        device="cpu",
    )
    model.set_verify()
    assert model.get_input_sequence_length() == verify_length
    assert model.active_graph() is model.verify_model
    assert model.active_component() == f"decode_m{verify_length}"
    inputs = torch.arange(verify_length * 4, dtype=torch.float32).view(1, verify_length, 4)
    logits = model.forward(inputs)
    torch.testing.assert_close(logits, inputs * 2)
    torch.testing.assert_close(model.last_hidden_states, inputs)


@pytest.mark.parametrize("verify_length", [2, 4])
@pytest.mark.parametrize("declared_role", [True, False])
def test_verify_loader_resolves_metadata_width_and_legacy_role(tmp_path, verify_length, declared_role):
    import onnx
    from onnx import TensorProto, helper

    from xhmodel_merak.exec_backends import get_backend

    paths = {}
    role = f"decode_m{verify_length}"
    for phase, width in (("prefill", 8), ("decode", 1), ("mtp_prefill", 8), ("mtp_decode", 1), (role, verify_length)):
        path = tmp_path / phase / "model.onnx"
        path.parent.mkdir()
        graph = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"], name=f"{phase}_identity")],
            phase,
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, width, 4])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, width, 4])],
        )
        onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)
        paths[phase] = path
    model = object.__new__(XHQwen4ExpHMONNXModel)
    model._backend = get_backend()
    model._qwen4_spec = {
        "mode": "mtp",
        "runtime_contract_version": 3,
        "target": {"verify_hmonnx": f"{role}/model.onnx", "verify_chunk_length": verify_length},
        "draft": {"prefill_hmonnx": "mtp_prefill/model.onnx", "decode_hmonnx": "mtp_decode/model.onnx"},
    }
    if declared_role:
        model._qwen4_spec["target_verify_graph"] = role
    prefill, decode, _ = model._load_hmonnx_graphs(paths["prefill"], paths["decode"])
    assert prefill is not decode
    assert model._qwen4_verify_length == verify_length
    assert model._qwen4_verify_phase == role
    assert set(model.external_ple) == {"prefill", "decode", role}
    assert model._qwen4_verify_graph is not None
    model._qwen4_spec["target_verify_graph"] = "decode_m4" if verify_length == 2 else "decode_m2"
    with pytest.raises(ValueError, match="role disagrees"):
        model._load_hmonnx_graphs(paths["prefill"], paths["decode"])


@pytest.mark.parametrize("cache_op", ["KVcache", "LLMCache"])
def test_explicit_kv_writer_validation_cannot_count_qsa_as_value_cache(tmp_path, cache_op):
    import onnx
    import pytest
    from onnx import TensorProto, helper

    from xhmodel_merak.exec_backends import HMONNXInferenceConfig, MultiHMONNXLoader
    from xhmodel_merak.xh_llm.models.qwen4_exp.host_ple_runtime import (
        Qwen4ExpHostPLEInferenceV2,
        enable_qwen4_exp_inplace_kv,
    )

    def session_for(names):
        inputs = [
            helper.make_tensor_value_info("states", TensorProto.FLOAT16, [1, 1, 1, 8]),
            helper.make_tensor_value_info("past", TensorProto.INT32, [1]),
            helper.make_tensor_value_info("valid", TensorProto.INT32, [1]),
        ]
        nodes, outputs = [], []
        for name in names:
            inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT16, [1, 1, 8, 8]))
            nodes.append(
                helper.make_node(
                    cache_op,
                    ["states", "past", "valid", name],
                    [f"out_{name}"],
                    name=f"writer_{name}",
                    domain="ai.houmo.xh2a",
                    axis=2,
                    attention_max_length=-1,
                )
            )
            outputs.append(helper.make_tensor_value_info(f"out_{name}", TensorProto.FLOAT16, [1, 1, 8, 8]))
        path = tmp_path / f"cache_{len(names)}.onnx"
        onnx.save(
            helper.make_model(
                helper.make_graph(nodes, "cache", inputs, outputs),
                opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("ai.houmo.xh2a", 1)],
            ),
            path,
        )
        loader = MultiHMONNXLoader({"cache": path})
        session = Qwen4ExpHostPLEInferenceV2.from_onnx_graph(
            str(path),
            loader.graphs["cache"],
            HMONNXInferenceConfig(exec_devices=["cpu"]),
        )
        session.initialize()
        return session

    key, value, qsa = "past_key_cache_0", "past_value_cache_0", "qsa_indexer_key_cache_0"
    with pytest.raises(ValueError, match="paired K/V"):
        enable_qwen4_exp_inplace_kv(session_for([key, qsa]), {key, value})
    session = session_for([key, value, qsa])
    enable_qwen4_exp_inplace_kv(session, {key, value})
    feed = {name: CacheTensor(torch.zeros(1, 1, 8, 8, dtype=torch.float16)) for name in (key, value, qsa)}
    feed.update(
        states=torch.ones(1, 1, 1, 8, dtype=torch.float16),
        past=torch.tensor([2], dtype=torch.int32),
        valid=torch.ones(1, dtype=torch.int32),
    )
    session.run(feed)
    for name in (key, value):
        assert feed[name][:, :, 2].sum() == 8
        assert feed[name][:, :, :2].sum() == feed[name][:, :, 3:].sum() == 0


def _runtime_lifecycle_model():
    model = object.__new__(XHQwen4ExpHMONNXModel)
    model.__dict__.update(
        _qwen4_mtp_runtime=None,
        _qwen4_state=Qwen4ExpServingState(
            SimpleNamespace(tensors=()),
            batch_size=1,
            context_length=8,
            dtype=torch.float32,
            device="cpu",
        ),
        vision_encoder=torch.nn.Linear(2, 2),
    )
    return model


def test_device_hook_moves_state_and_visual_encoder_together(monkeypatch):
    model = _runtime_lifecycle_model()
    monkeypatch.setattr(TextLLMHMONNXModel, "_set_device", lambda self, device: self)
    model._set_device("meta")
    assert model._qwen4_state.group_start_position.device.type == "meta"
    assert model.vision_encoder.weight.device.type == "meta"


def test_persistent_mtp_device_rejection_precedes_any_model_move(monkeypatch):
    model = _runtime_lifecycle_model()
    model.__dict__["_qwen4_mtp_runtime"] = SimpleNamespace(device=torch.device("cpu"))
    moved = []
    monkeypatch.setattr(TextLLMHMONNXModel, "_set_device", lambda self, device: moved.append(device))
    with pytest.raises(ValueError, match="fixed|device"):
        model._set_device("meta")
    assert moved == []
    assert model._qwen4_state.group_start_position.device.type == "cpu"
    assert model.vision_encoder.weight.device.type == "cpu"


def test_model_close_releases_its_persistent_draft_once():
    model = _runtime_lifecycle_model()
    closed = []
    model.__dict__["_qwen4_mtp_runtime"] = SimpleNamespace(close=lambda: closed.append(True))
    model.close()
    model.close()
    assert closed == [True]
    assert model._qwen4_mtp_runtime is None
