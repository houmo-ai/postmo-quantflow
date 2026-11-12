"""Small real HC/QSA/MoE assistant and speculative-state regressions."""

import pytest
import torch
from transformers import Qwen4ExpTextConfig

from xhmodel_merak.xh_llm.models.qwen4_exp.mtp_model import (
    Qwen4ExpMTPExportWrapper,
    Qwen4ExpMTPModel,
    mtp_text_config,
)
from xhmodel_merak.xh_llm.models.qwen4_exp.serving_graph import convert_qwen4_exp_kv_to_llm_cache
from xhquant.core.cache_tensor import CacheTensor


@pytest.mark.parametrize(
    "settings,candidates,width",
    [
        ({}, 4, 5),
        ({"num_draft_tokens": 1}, 1, 2),
        ({"num_draft_tokens": 3, "verify_length": 8}, 3, 8),
    ],
)
def test_mtp_candidate_and_verify_export_dimensions(settings, candidates, width):
    from xhmodel_merak.xh_llm.models.qwen4_exp.mtp_contract import (
        mtp_verify_phase,
        resolve_mtp_num_draft_tokens,
        resolve_mtp_verify_length,
    )

    assert resolve_mtp_num_draft_tokens(settings) == candidates
    assert resolve_mtp_verify_length(settings) == width
    # A graph role describes a physical width, including existing width-2/4 graphs.
    assert mtp_verify_phase(2) == "decode_m2"
    assert mtp_verify_phase(4) == "decode_m4"


@pytest.mark.parametrize(
    "settings",
    [{"num_draft_tokens": value} for value in (0, -1, True, 1.5, "4")]
    + [{"verify_length": value} for value in (4, True, 5.5, "5")],
)
def test_mtp_rejects_invalid_candidate_capacity(settings):
    from xhmodel_merak.xh_llm.models.qwen4_exp.mtp_contract import resolve_mtp_verify_length

    with pytest.raises(ValueError):
        resolve_mtp_verify_length(settings)


def test_fixed_head_checkpoint_reuses_quantization_across_static_phases(tmp_path, monkeypatch):
    import onnx
    from onnx import numpy_helper

    from xhmodel_merak.xh_llm.models.qwen4_exp.serving_workflow import (
        _load_fixed_head_checkpoint,
        _save_fixed_head_checkpoint,
    )
    from xhmodel_merak.xh_llm.models.qwen4_exp.static_export import export_qwen4_exp_hmonnx
    from xhquant.quantization.xh2a.qmodules._qlinear_base import QLinearBase

    class HeadModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = torch.nn.Linear(16, 32, bias=False).half()

        def forward(self, inputs_embeds):
            return self.lm_head(inputs_embeds)

    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "0")
    count = []
    original = QLinearBase.weight_static_quant

    def counted(module):
        count.append(module)
        return original(module)

    monkeypatch.setattr(QLinearBase, "weight_static_quant", counted)
    fixed_head, payloads = None, []
    for phase, width in (("prefill", 3), ("decode", 1)):
        model = HeadModel().eval()
        result = export_qwen4_exp_hmonnx(
            model,
            tmp_path / phase,
            sample_inputs=(torch.ones(1, width, 16).half(),),
            export_wrapper=model,
            export_input_names=["inputs_embeds"],
            export_output_names=["logits"],
            quant_config={"quant_type": "w8a8h1_sefp"},
            ple_quant_type="fp16",
            shared_quantized_head=fixed_head,
            capture_quantized_head=True,
        )
        assert "fixed_quantized_head" not in result.to_dict()
        if fixed_head is None:
            bindings = {"source_config_sha256": "test-source", "export_config_sha256": "test-export"}
            record = _save_fixed_head_checkpoint(tmp_path, result.fixed_quantized_head, bindings)
            fixed_head = _load_fixed_head_checkpoint(tmp_path, record, bindings)
            assert torch.equal(fixed_head.qweight, result.fixed_quantized_head.qweight)
            assert torch.equal(fixed_head.scale_or_exp, result.fixed_quantized_head.scale_or_exp)
        graph = onnx.load(result.onnx_path)
        (head,) = [node for node in graph.graph.node if node.op_type == "Linear"]
        initializers = {value.name: value for value in graph.graph.initializer}
        payloads.append(tuple(numpy_helper.to_array(initializers[name]).tobytes() for name in head.input[1:]))
    assert len(count) == 1
    assert payloads[0] == payloads[1]


def config():
    return Qwen4ExpTextConfig(
        max_position_embeddings=16,
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=2,
        num_experts_per_tok=1,
        hc_count=2,
        hc_lowrank=4,
        layer_types=["linear_attention", "full_attention"],
        ple_layer_ids=[],
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=4,
        indexer_compress_ratio=4,
        rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.5},
        mtp={"num_hidden_layers": 1, "layer_types": ["full_attention"]},
    )


def test_mtp_uses_own_qsa_hc_cache_and_no_target_ple():
    c = config()
    c.ple_layer_ids = [2]
    result = mtp_text_config(c)
    assert result.layer_types == ["qwen_sparse_attention"]
    assert result.ple_layer_ids == []
    assert c.ple_layer_ids == [2]
    wrapper = Qwen4ExpMTPExportWrapper(Qwen4ExpMTPModel(c), torch.nn.Linear(16, 32), cache_length=8)
    assert {s.state_kind for s in wrapper.cache_contract.tensors} == {
        "kv_key",
        "kv_value",
        "qsa_indexer_key",
        "qsa_indexer_tail",
        "qsa_key_tail",
        "qsa_value_tail",
    }


def test_multiple_qsa_layers_share_one_mrope_lookup(tmp_path, monkeypatch):
    import onnx
    from transformers import Qwen4ExpForCausalLM

    from xhmodel_merak.xh_llm.models.qwen4_exp.static_export import (
        build_qwen4_exp_serving_sample_inputs,
        export_qwen4_exp_hmonnx,
    )

    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "0")
    c = Qwen4ExpTextConfig(**{**config().to_dict(), "layer_types": ["full_attention"] * 2})
    native = Qwen4ExpForCausalLM(c).half().eval()
    _, _, sample = build_qwen4_exp_serving_sample_inputs(native, input_sequence_length=2, context_length=8)
    result = export_qwen4_exp_hmonnx(
        native,
        tmp_path,
        text_config=c,
        sample_inputs=sample,
        input_sequence_length=2,
        cache_length=8,
        quant_device="cpu",
        quant_config={"quant_type": "w16a16h1_sefp", "preserve_model_dtype": True},
    )
    graph = onnx.load(result.hmonnx_path, load_external_data=False).graph
    ropes = [node for node in graph.node if node.op_type == "Rope"]
    producers = {name: node for node in graph.node for name in node.output}
    lookups = [node for node in graph.node if node.op_type == "Gather" and "rotary_emb" in node.input[0]]
    cos_lookups = {node.name for node in lookups if "cos_cached" in node.input[0]}
    sin_lookups = {node.name for node in lookups if "sin_cached" in node.input[0]}
    assert len(ropes) == 8  # Q, K, index Q and pooled index K in each QSA layer.
    assert len(cos_lookups) == len(sin_lookups) == 3  # One T/H/W lookup per table.
    assert [list(node.input[1:]) for node in ropes[:4]] == [list(node.input[1:]) for node in ropes[4:]]

    def lookup_ancestors(value):
        pending, visited, found = [value], set(), set()
        while pending:
            name = pending.pop()
            if name in visited:
                continue
            visited.add(name)
            node = producers.get(name)
            if node is not None:
                if node.name in cos_lookups | sin_lookups:
                    found.add(node.name)
                else:
                    pending.extend(node.input)
        return found

    for rope in ropes:
        assert lookup_ancestors(rope.input[1]) == cos_lookups
        assert lookup_ancestors(rope.input[2]) == sin_lookups


def test_mtp_grouped_hyperstream_fusion():
    torch.manual_seed(4)
    model = Qwen4ExpMTPModel(config()).eval()
    embedding, hidden = torch.randn(1, 3, 16), torch.randn(1, 3, 32)
    normalized = model.pre_fc_norm_hidden(hidden).reshape(1, 3, 2, 16)
    projected = model.fc_embedding(model.pre_fc_norm_embedding(embedding))
    expected = torch.stack([model.fc_hidden(normalized[:, :, i]) + projected for i in range(2)], 2).flatten(-2)
    torch.testing.assert_close(model.fuse(embedding, hidden), expected)
    altered = hidden.clone()
    altered[..., 16:] *= -1
    assert not torch.equal(model.fuse(embedding, altered), expected)
    from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import Qwen4ExpExportableRMSNorm

    original_weights = (model.pre_fc_norm_embedding.weight, model.pre_fc_norm_hidden.weight)
    Qwen4ExpMTPExportWrapper(model, torch.nn.Linear(16, 32), cache_length=8)
    assert isinstance(model.pre_fc_norm_embedding, Qwen4ExpExportableRMSNorm)
    assert isinstance(model.pre_fc_norm_hidden, Qwen4ExpExportableRMSNorm)
    assert model.pre_fc_norm_embedding.weight is original_weights[0]
    assert model.pre_fc_norm_hidden.weight is original_weights[1]
    assert model.pre_fc_norm_hidden.weight.shape == (32,)
    assert model.pre_fc_norm_hidden.norm.weight.shape == (16,)
    torch.testing.assert_close(model.fuse(embedding, hidden), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "overrides",
    [
        {"mtp_use_dedicated_embeddings": True},
        {"mtp": {"num_hidden_layers": 2, "layer_types": ["full_attention"] * 2}},
        {"mtp": {"num_hidden_layers": 1, "layer_types": ["full_attention"], "mtp_use_hidden_state_from_layer": 4}},
    ],
)
def test_mtp_rejects_unknown_checkpoint_architecture(overrides):
    c = config().to_dict()
    c.update(overrides)
    with pytest.raises(ValueError):
        mtp_text_config(c)


@pytest.mark.parametrize("length", [0, 9])
def test_mtp_export_rejects_invalid_chunk_before_loading_weights(tmp_path, length):
    from xhmodel_merak.xh_llm.models.qwen4_exp.mtp_export import export_qwen4_exp_mtp

    with pytest.raises(ValueError, match="input_sequence_length"):
        export_qwen4_exp_mtp(tmp_path / "absent", tmp_path / "export", cache_length=8, input_sequence_length=length)
    assert not (tmp_path / "export").exists()


@pytest.mark.parametrize("draft,accepted,next_token", [([2, 3], 2, 4), ([2, 9], 1, 3), ([9, 8], 0, 2), ([], 0, 2)])
def test_verify_restores_all_state_on_rejection(draft, accepted, next_token):
    from xhmodel_merak.xh_llm.models.qwen4_exp.mtp_runtime import verify_draft_transaction

    state = {name: torch.zeros(1) for name in ("gdr", "ple", "qsa", "positions", "pagekv")}
    calls = []

    def snapshot():
        return {k: v.clone() for k, v in state.items()}

    def restore(saved):
        for k, v in saved.items():
            state[k].copy_(v)

    def consume(tokens):
        calls.append(tuple(tokens))
        for token in tokens:
            for v in state.values():
                v.mul_(10).add_(token)
        logits = torch.full((1, len(tokens), 16), -100.0)
        for i, t in enumerate(tokens):
            logits[0, i, t + 1] = 1
        hidden = torch.tensor(tokens, dtype=torch.float32).view(1, -1, 1).repeat(1, 1, 4)
        return logits, hidden

    result = verify_draft_transaction(
        pending_token=1, draft_tokens=draft, snapshot=snapshot, restore=restore, verify=consume, replay=consume
    )
    assert result.accepted_draft_tokens == accepted
    assert result.next_token == next_token
    assert result.committed_input_tokens == tuple([1] + draft[:accepted])
    expected = 0
    for t in result.committed_input_tokens:
        expected = expected * 10 + t
    assert all(v.item() == expected for v in state.values())
    assert result.target_hidden[0, 0, 0].item() == result.committed_input_tokens[-1]
    assert len(calls) == (1 if accepted == len(draft) else 2)


def test_verify_failure_restores_target_state():
    from xhmodel_merak.xh_llm.models.qwen4_exp.mtp_runtime import verify_draft_transaction

    state = torch.tensor([7.0])

    def broken(tokens):
        state.add_(10)
        raise RuntimeError("session failed")

    with pytest.raises(RuntimeError, match="session failed"):
        verify_draft_transaction(
            pending_token=1,
            draft_tokens=[2],
            snapshot=lambda: state.clone(),
            restore=state.copy_,
            verify=broken,
            replay=broken,
        )
    assert state.item() == 7


def test_mtp_loads_only_assistant_shard_and_rejects_missing_weight(tmp_path):
    import json

    from safetensors.torch import save_file

    source = Qwen4ExpMTPModel(config()).eval()
    values = {"mtp." + k: v.contiguous() for k, v in source.state_dict().items()}
    save_file(values, str(tmp_path / "assistant.safetensors"))
    mapping = {k: "assistant.safetensors" for k in values}
    mapping["model.language_model.layers.0.attn.weight"] = "absent-target-shard.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": mapping}))
    (tmp_path / "config.json").write_text(json.dumps({"text_config": config().to_dict()}))
    loaded = Qwen4ExpMTPModel.from_pretrained(tmp_path)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)
    assert all(t.device.type == "cpu" for t in loaded.buffers())
    values.pop("mtp.fc_hidden.weight")
    save_file(values, str(tmp_path / "assistant.safetensors"))
    with pytest.raises(RuntimeError, match="fc_hidden.weight"):
        Qwen4ExpMTPModel.from_pretrained(tmp_path)


def test_mtp_qsa_prefill_and_decode_share_pool_and_tail(tmp_path, monkeypatch):
    import copy

    from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import resolve_qwen4_exp_cache_shape
    from xhmodel_merak.xh_llm.models.qwen4_exp.mtp_export import export_qwen4_exp_mtp
    from xhmodel_merak.xh_llm.models.qwen4_exp.mtp_runtime import MTPGraphRuntime
    from xhmodel_merak.xh_llm.models.qwen4_exp.qsa_host_controls import build_qsa_host_controls
    from xhmodel_merak.xh_llm.mrope import prepare_mrope_position_inputs

    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "0")
    torch.manual_seed(108)
    assistant = Qwen4ExpMTPModel(config()).half().eval()
    head = torch.nn.Linear(16, 32, bias=False).half().eval()
    reference = Qwen4ExpMTPExportWrapper(
        copy.deepcopy(assistant),
        copy.deepcopy(head),
        cache_length=8,
    ).eval()
    graphs = {}
    for name, width in (("prefill", 3), ("decode", 1)):
        graphs[name] = export_qwen4_exp_mtp(
            None,
            tmp_path / name,
            assistant=assistant,
            lm_head=head,
            cache_length=8,
            input_sequence_length=width,
            dtype=torch.float16,
            quant_config={"quant_type": "w16a16h1_sefp", "preserve_model_dtype": True},
        )
    import onnx_graphsurgeon as gs

    from xhmodel_merak.exec_backends import MultiHMONNXLoader

    loader = MultiHMONNXLoader({name: data["onnx_path"] for name, data in graphs.items()})
    templates = loader.graphs
    payloads = {
        name: {key: tensor.values for key, tensor in graph.tensors().items() if isinstance(tensor, gs.Constant)}
        for name, graph in templates.items()
    }

    def open_runtime():
        return MTPGraphRuntime(
            graphs["decode"]["onnx_path"],
            onnx_graph=templates["decode"],
            cache_contract=graphs["decode"]["cache_contract"],
            prefill_path=graphs["prefill"]["onnx_path"],
            prefill_graph=templates["prefill"],
            prefill_chunk_length=3,
            context_length=8,
            device="cpu",
        )

    runtime = open_runtime()
    specs = reference.cache_contract.tensors
    state = [
        torch.zeros(
            resolve_qwen4_exp_cache_shape(spec, batch_size=1, context_length=8),
            dtype=torch.int32 if spec.dtype == "integer" else torch.float16,
        )
        for spec in specs
    ]
    for i, spec in enumerate(specs):
        if spec.state_kind in {"kv_key", "kv_value", "qsa_indexer_key"}:
            state[i] = CacheTensor(state[i])
    group_start = torch.zeros(3, 1, 1, dtype=torch.int32)
    embeddings, hidden = torch.randn(1, 4, 16).half(), torch.randn(1, 4, 32).half()
    positions = torch.tensor([[[1, 1, 2, 2]], [[2, 3, 3, 4]], [[4, 5, 6, 7]]], dtype=torch.int32)
    expected = []
    with torch.no_grad():
        for i in range(4):
            current_pos, pool_pos, group_start = prepare_mrope_position_inputs(
                group_start,
                positions[:, :, i : i + 1],
                i,
                1,
                1,
            )
            controls = build_qsa_host_controls(i, 1, 1, 8, 1, device="cpu", dtype=torch.float16)
            output = reference(
                embeddings[:, i : i + 1],
                hidden[:, i : i + 1],
                torch.tensor([i], dtype=torch.int32),
                torch.ones(1, dtype=torch.int32),
                *current_pos.unbind(0),
                *pool_pos.unbind(0),
                (*state, *controls),
            )
            expected.append(output[:2])
            for j, spec in enumerate(specs):
                if spec.state_kind not in {"kv_key", "kv_value", "qsa_indexer_key"}:
                    state[j] = output[2 + j]
        actual = [runtime.prefill(embeddings[:, :2], hidden[:, :2], position_ids=positions[:, :, :2])]
        for i in (2, 3):
            actual.append(
                runtime.step(embeddings[:, i : i + 1], hidden[:, i : i + 1], position_ids=positions[:, :, i : i + 1])
            )
    for kind in range(2):
        torch.testing.assert_close(
            torch.cat([value[kind] for value in actual], dim=1),
            torch.cat([value[kind] for value in expected], dim=1),
            atol=3e-3,
            rtol=2e-2,
        )
    for name in ("qsa_indexer_key_cache_0", "qsa_indexer_key_tail_0", "qsa_key_tail_0", "qsa_value_tail_0"):
        index = [spec.name for spec in specs].index(name)
        torch.testing.assert_close(runtime.states[name], state[index], atol=3e-3, rtol=2e-2)
    assert runtime.states["qsa_indexer_key_cache_0"][:, :, 0].abs().sum() > 0
    runtime.close()
    # A later request must be able to parse both templates again. The SDK
    # consumes Constant shells, while model-owned weight payloads stay shared.
    for name, graph in templates.items():
        for key, payload in payloads[name].items():
            assert payload is not None
            assert graph.tensors()[key].values is payload
    repeated = open_runtime()
    try:
        again = [repeated.prefill(embeddings[:, :2], hidden[:, :2], position_ids=positions[:, :, :2])]
        for i in (2, 3):
            again.append(
                repeated.step(embeddings[:, i : i + 1], hidden[:, i : i + 1], position_ids=positions[:, :, i : i + 1])
            )
        for first, second in zip(actual, again, strict=True):
            for initial, replayed in zip(first, second, strict=True):
                torch.testing.assert_close(initial, replayed, rtol=0, atol=0)
    finally:
        repeated.close()
    for name, graph in templates.items():
        for key, payload in payloads[name].items():
            assert graph.tensors()[key].values is payload


def test_target_verify_exports_padded_tokens_with_unmixed_hc(tmp_path, monkeypatch):
    verify_width = 4
    import onnx
    from transformers import Qwen4ExpForCausalLM

    from xhmodel_merak.xh_llm.models.qwen4_exp.ple_export import read_qwen4_exp_ple_specs
    from xhmodel_merak.xh_llm.models.qwen4_exp.static_export import (
        build_qwen4_exp_serving_sample_inputs,
        export_qwen4_exp_hmonnx,
    )

    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "0")
    torch.manual_seed(39)
    c = Qwen4ExpTextConfig(
        **{
            **config().to_dict(),
            "ple_layer_ids": [1],
            "ple_embed_dim": 64,
            "ple_conv_kernel_size": 2,
            "ngram_size": 2,
            "heads_per_ngram": 1,
            "ngram_vocab_size_base": 5,
            "make_ngram_vocab_size_divisible_by": 8,
            "split_ngram_parts": 1,
            "eos_token_id": 0,
            "indexer_compress_ratio": 4,
        }
    )
    native = Qwen4ExpForCausalLM(c).half().eval()
    _, _, sample = build_qwen4_exp_serving_sample_inputs(
        native,
        input_sequence_length=verify_width,
        context_length=8,
    )
    result = export_qwen4_exp_hmonnx(
        native,
        tmp_path / "raw",
        text_config=c,
        sample_inputs=sample,
        input_sequence_length=verify_width,
        cache_length=8,
        quant_device="cpu",
        quant_config={"quant_type": "w16a16h1_sefp", "preserve_model_dtype": True},
        use_fused_gdr=True,
        linear_attention_mode="recurrent",
        return_hidden_states=True,
    )
    graph_path = convert_qwen4_exp_kv_to_llm_cache(
        result.hmonnx_path,
        tmp_path / f"decode_m{verify_width}/model.onnx",
        context_length=8,
        num_kv_heads=1,
        head_dim=8,
    )
    graph = onnx.load(graph_path, load_external_data=False).graph
    onnx.checker.check_model(str(graph_path))

    def shape(value):
        return [dim.dim_value for dim in value.type.tensor_type.shape.dim]

    inputs = {value.name: shape(value) for value in graph.input}
    outputs = {value.name: shape(value) for value in graph.output}
    assert inputs["inputs_embeds"] == [1, verify_width, 16]
    assert inputs["ple_input_ids"] == [1, verify_width]
    assert inputs["ple_embeddings_0"] == [1, verify_width, 1, 64]
    assert inputs["ple_ngram_history_0"] == [1, 1]
    assert inputs["ple_conv_state_0"] == [1, 32, 2]
    from xhmodel_merak.xh_llm.mrope import MROPE_CURRENT_POSITION_NAMES, MROPE_POOL_POSITION_NAMES

    assert all(inputs[name] == [1, verify_width] for name in MROPE_CURRENT_POSITION_NAMES)
    assert all(inputs[name] == [1, (verify_width + 3) // 4] for name in MROPE_POOL_POSITION_NAMES)
    assert "past_position_ids" not in inputs and "present_past_position_ids" not in outputs
    position_values = set(MROPE_CURRENT_POSITION_NAMES + MROPE_POOL_POSITION_NAMES)
    for node in graph.node:
        if (
            node.op_type in {"Concat", "Cast", "Unsqueeze", "Squeeze", "Identity"}
            and node.input
            and all(name in position_values for name in node.input if name)
        ):
            position_values.update(node.output)
    assert not any(node.op_type == "Gather" and node.input[0] in position_values for node in graph.node)
    assert (
        sum(
            node.op_type == "Gather"
            and "rotary_emb" in node.input[0]
            and ("cos_cached" in node.input[0] or "sin_cached" in node.input[0])
            for node in graph.node
        )
        == 6
    )
    assert inputs["qsa_key_tail_1"] == inputs["qsa_value_tail_1"] == [1, 1, 3, 8]
    assert "qsa_tail_indices" not in inputs
    assert outputs["logits"] == [1, verify_width, 32]
    assert outputs["target_hidden_states"] == [1, verify_width, 32]
    assert outputs["present_qsa_key_tail_1"] == outputs["present_qsa_value_tail_1"] == [1, 1, 3, 8]
    assert sum(node.op_type == "GDRRecurrentScan" for node in graph.node) == verify_width
    assert not any(node.op_type in {"GDRChunkScan", "GatherElements"} for node in graph.node)
    (select,) = [node for node in graph.node if node.op_type == "SparseAttentionSelectKV"]
    assert not any(
        node.op_type == "Gather" and node.input[0] in (select.input[2], select.input[6]) for node in graph.node
    )
    assert (
        sum(node.op_type == "Concat" and any(value in node.input for value in select.output) for node in graph.node)
        == 2
    )
    assert not any(
        name.startswith(("present_past_key_cache_", "present_past_value_cache_", "present_qsa_indexer_key_cache_"))
        for name in outputs
    )
    assert not any(node.op_type in {"PageKVCache", "LLMCache"} for node in graph.node)
    assert sum(node.op_type == "KVcache" for node in graph.node) >= 2
    assert not any(node.op_type == "Qwen4PLESEFP4Gather" for node in graph.node)
    (ple,) = read_qwen4_exp_ple_specs(graph_path)
    assert ple["input_name"] == "ple_embeddings_0"
    assert ple["packed"]["shape"] == [8, 1, 32]

    _, _, decode_sample = build_qwen4_exp_serving_sample_inputs(
        native,
        input_sequence_length=1,
        context_length=8,
    )
    decode_result = export_qwen4_exp_hmonnx(
        native,
        tmp_path / "decode_raw",
        text_config=c,
        sample_inputs=decode_sample,
        input_sequence_length=1,
        cache_length=8,
        quant_device="cpu",
        quant_config={"quant_type": "w16a16h1_sefp", "preserve_model_dtype": True},
        use_fused_gdr=True,
        return_hidden_states=True,
        linear_attention_mode="recurrent",
    )
    decode_path = convert_qwen4_exp_kv_to_llm_cache(
        decode_result.hmonnx_path,
        tmp_path / "decode/model.onnx",
        context_length=8,
        num_kv_heads=1,
        head_dim=8,
    )

    # Exercise two valid positions in the legacy width-four graph, followed by
    # another call, so padding must preserve every HC/QSA/PLE state.
    from xhmodel_merak.exec_backends import HMONNXInferenceConfig, MultiHMONNXLoader
    from xhmodel_merak.xh_llm.models.qwen4_exp.host_ple_inputs import HostPLEInputs
    from xhmodel_merak.xh_llm.models.qwen4_exp.host_ple_runtime import Qwen4ExpHostPLEInferenceV2
    from xhmodel_merak.xh_llm.models.qwen4_exp.qsa_host_controls import (
        QSA_HOST_CONTROL_NAMES,
        build_qsa_host_controls,
        qsa_host_control_spec,
    )
    from xhmodel_merak.xh_llm.models.qwen4_exp.xhquant_extensions import qwen4_exp_xhquant_extension_scope
    from xhmodel_merak.xh_llm.mrope import prepare_mrope_position_inputs

    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "1")
    monkeypatch.setattr(torch._dynamo.config, "disable", True)
    loader = MultiHMONNXLoader({"verify": graph_path, "decode": decode_path})
    with qwen4_exp_xhquant_extension_scope():
        session = Qwen4ExpHostPLEInferenceV2.from_onnx_graph(
            graph_path,
            loader.graphs["verify"],
            HMONNXInferenceConfig(exec_devices=["cpu"]),
        )
        session.initialize()
        decode_session = Qwen4ExpHostPLEInferenceV2.from_onnx_graph(
            decode_path,
            loader.graphs["decode"],
            HMONNXInferenceConfig(exec_devices=["cpu"]),
        )
        decode_session.initialize()
    assert session.enable_inplace_cache_updates() >= 2
    assert decode_session.enable_inplace_cache_updates() >= 2
    host = HostPLEInputs.from_graph(graph_path)
    decode_host = HostPLEInputs.from_graph(decode_path)

    def initialize_feed(runtime, host_ple):
        host_names = {layer["input_name"] for layer in host_ple.manifest["layers"]}
        result = {}
        for name in runtime.get_input_names():
            if name in host_names:
                continue
            info = runtime.get_input(name)
            value = torch.zeros(tuple(info.shape), dtype=info.dtype)
            mutable = name.startswith(
                ("past_key_cache_", "past_value_cache_", "qsa_indexer_key_cache_", "delta_recurrent_state_")
            )
            result[name] = CacheTensor(value) if mutable else value
        return result

    def execute(runtime, host_ple, data, valid_length):
        prepared = host_ple.prepare(data["ple_input_ids"], data, valid_length, device="cpu")
        for layer, value in zip(host_ple.manifest["layers"], prepared, strict=True):
            data[layer["input_name"]] = value
        result = dict(zip(runtime.get_output_names(), runtime.run(data), strict=True))
        for name, value in result.items():
            state_name = name.removeprefix("present_")
            if name.startswith("present_") and state_name in data:
                data[state_name].copy_(value)
        return result

    feed = initialize_feed(session, host)
    decode_feed = initialize_feed(decode_session, decode_host)
    feed["current_input_length"].fill_(2)
    decode_feed["current_input_length"].fill_(1)
    kv_pointers = {
        name: value.data_ptr()
        for name, value in feed.items()
        if name.startswith(("past_key_cache_", "past_value_cache_", "qsa_indexer_key_cache_"))
    }
    outputs = []
    group_start = torch.zeros(3, 1, 1, dtype=torch.int32)
    decode_group_start = group_start.clone()
    for past in (0, 2):
        feed["inputs_embeds"].normal_()
        feed["ple_input_ids"].copy_(torch.arange(past + 1, past + 1 + verify_width, dtype=torch.int32).unsqueeze(0))
        feed["past_seq_length"].fill_(past)
        positions = torch.arange(past, past + 2, dtype=torch.int32).view(1, 1, 2).expand(3, 1, 2)
        current, pooled, group_start = prepare_mrope_position_inputs(
            group_start,
            positions,
            past,
            verify_width,
            2,
        )
        feed.update(zip(MROPE_CURRENT_POSITION_NAMES, current.unbind(0), strict=True))
        feed.update(zip(MROPE_POOL_POSITION_NAMES, pooled.unbind(0), strict=True))
        topk, dtype = qsa_host_control_spec(session)
        controls = build_qsa_host_controls(past, 2, verify_width, 8, topk, device="cpu", dtype=dtype)
        feed.update(zip(QSA_HOST_CONTROL_NAMES, controls, strict=True))
        prepared = host.prepare(feed["ple_input_ids"], feed, 2, device="cpu")
        for layer, value in zip(host.manifest["layers"], prepared, strict=True):
            feed[layer["input_name"]] = value
        previous_key = feed["past_key_cache_0"].clone()
        previous_indexer = {
            name: feed[name].clone() for name in kv_pointers if name.startswith("qsa_indexer_key_cache_")
        }
        assert previous_indexer
        actual = execute(session, host, feed, 2)
        assert all(feed[name].data_ptr() == pointer for name, pointer in kv_pointers.items())
        assert actual["logits"].shape == (1, verify_width, 32)
        assert actual["target_hidden_states"].shape == (1, verify_width, 32)
        assert torch.isfinite(actual["logits"]).all() and torch.isfinite(actual["target_hidden_states"]).all()
        assert feed["past_key_cache_0"][:, :, past : past + 2].abs().sum() > 0
        for name, value in previous_indexer.items():
            if past == 0:
                torch.testing.assert_close(feed[name], value, rtol=0, atol=0)
            else:
                assert feed[name][:, :, :1].abs().sum() > 0
        assert any(name.startswith("present_qsa_indexer_key_tail_") for name in actual)
        if past:
            torch.testing.assert_close(feed["past_key_cache_0"][:, :, :past], previous_key[:, :, :past], rtol=0, atol=0)
        expected = []
        for offset in range(2):
            position = past + offset
            decode_feed["inputs_embeds"].copy_(feed["inputs_embeds"][:, offset : offset + 1])
            decode_feed["ple_input_ids"].copy_(feed["ple_input_ids"][:, offset : offset + 1])
            decode_feed["past_seq_length"].fill_(position)
            step_positions = torch.full((3, 1, 1), position, dtype=torch.int32)
            current, pooled, decode_group_start = prepare_mrope_position_inputs(
                decode_group_start,
                step_positions,
                position,
                1,
                1,
            )
            decode_feed.update(zip(MROPE_CURRENT_POSITION_NAMES, current.unbind(0), strict=True))
            decode_feed.update(zip(MROPE_POOL_POSITION_NAMES, pooled.unbind(0), strict=True))
            decode_feed.update(
                zip(
                    QSA_HOST_CONTROL_NAMES,
                    build_qsa_host_controls(
                        position,
                        1,
                        1,
                        8,
                        topk,
                        device="cpu",
                        dtype=dtype,
                    ),
                    strict=True,
                )
            )
            expected.append(execute(decode_session, decode_host, decode_feed, 1))
        for name in ("logits", "target_hidden_states"):
            torch.testing.assert_close(
                actual[name][:, :2],
                torch.cat([result[name] for result in expected], dim=1),
                rtol=0,
                atol=0,
            )
        for name in decode_feed:
            if name.startswith(
                (
                    "past_key_cache_",
                    "past_value_cache_",
                    "qsa_indexer_key_cache_",
                    "delta_recurrent_state_",
                    "delta_conv_state_",
                    "ple_ngram_history_",
                    "ple_conv_state_",
                    "qsa_indexer_key_tail_",
                    "qsa_key_tail_",
                    "qsa_value_tail_",
                )
            ):
                torch.testing.assert_close(feed[name], decode_feed[name], rtol=0, atol=0)
        torch.testing.assert_close(group_start, decode_group_start, rtol=0, atol=0)
        outputs.append(actual)
    assert not torch.equal(outputs[0]["logits"], outputs[1]["logits"])


@pytest.mark.parametrize("storage", ["numpy", "shared", "lazy"])
def test_mtp_graph_shell_copy_keeps_initializer_storage_without_loading(storage, monkeypatch):
    import numpy as np
    import onnx
    import onnx_graphsurgeon as gs
    from onnx_graphsurgeon.ir.tensor import LazyValues

    from xhmodel_merak.xh_llm.models.qwen4_exp.mtp_runtime import _copy_mtp_graph
    from xhquant.xhonnxruntime.llm_hmonnx_loader import SharedTensor

    array = np.arange(4, dtype=np.float16)
    payload = (
        SharedTensor(torch.from_numpy(array))
        if storage == "shared"
        else LazyValues(onnx.numpy_helper.from_array(array))
        if storage == "lazy"
        else array
    )
    weight = gs.Constant("weight", np.empty(0, dtype=np.uint8))
    weight.values = payload
    value = gs.Variable("input", dtype=np.float16, shape=[4])
    output = gs.Variable("output", dtype=np.float16, shape=[4])
    graph = gs.Graph(
        nodes=[gs.Node("Add", name="add", inputs=[value, weight], outputs=[output])],
        inputs=[value],
        outputs=[output],
        name="draft",
        doc_string="request template",
        opset=18,
    )

    def no_materialization(*args):
        raise AssertionError("copy must not materialize lazy initializer data")

    monkeypatch.setattr(LazyValues, "load", no_materialization)
    cloned = _copy_mtp_graph(graph)
    assert cloned.name == graph.name and cloned.doc_string == graph.doc_string and cloned.opset == graph.opset
    assert cloned.nodes[0] is not graph.nodes[0]
    assert cloned.inputs[0] is not value and cloned.outputs[0] is not output
    assert cloned.inputs[0].name == value.name
    assert cloned.inputs[0].dtype == value.dtype and cloned.inputs[0].shape == value.shape
    assert cloned.inputs[0].type == value.type
    clone_weight = cloned.tensors()["weight"]
    assert clone_weight is not weight and clone_weight._values is payload
    assert cloned.inputs[0].outputs[0] is cloned.nodes[0]
    assert cloned.outputs[0].inputs[0] is cloned.nodes[0]
    assert clone_weight.outputs[0] is cloned.nodes[0]
    # Reproduce the SDK parser's last-use release on its owned shell.
    clone_weight.values = None
    assert weight._values is payload
    assert weight.outputs[0] is graph.nodes[0]
