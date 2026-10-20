from __future__ import annotations

import pytest
import torch

from xhmodel_merak.xh_llm.models.qwen4_exp.selected_kv_attention import Qwen4ExpQSAConfig


def _pack_kv_payload(key, value):
    batch, heads, length, dim = key.shape
    return torch.cat((key.transpose(1, 2).reshape(batch, length, heads * dim),
                      value.transpose(1, 2).reshape(batch, length, heads * dim)), dim=-1).unsqueeze(1)


def _unpack_selected_kv(payload, num_kv_heads, head_dim):
    width = num_kv_heads * head_dim
    key, value = payload.split([width, width], dim=-1)
    shape = (*payload.shape[:3], num_kv_heads, head_dim)
    return key.reshape(shape).transpose(2, 3), value.reshape(shape).transpose(2, 3)


def _initial_tail(cache, past):
    previous = cache[:, :, max(0, past - 3) : past]
    return torch.nn.functional.pad(previous, (0, 0, 3 - previous.shape[2], 0))


def test_shared_mrope_matches_native_at_long_multiaxis_positions():
    from transformers import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRotaryEmbedding

    from xhmodel_merak.xh_llm.models.qwen3_5._llm_model_impl import _Qwen3_5TextModel
    from xhmodel_merak.xh_llm.mrope import StaticMRotaryEmbedding

    config = Qwen4ExpTextConfig(
        hidden_size=256, num_attention_heads=1, head_dim=256, max_position_embeddings=262144,
        rope_parameters={"rope_type": "default", "rope_theta": 10000000.0,
                         "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10]},
    )
    native = Qwen4ExpTextRotaryEmbedding(config)
    shared = StaticMRotaryEmbedding(native.inv_freq, native.attention_scaling, 262144, native.mrope_section)
    positions = torch.tensor([
        [[0, 2048, 2049, 65536, 262143]],
        [[1, 2049, 2048, 100000, 200000]],
        [[2, 2050, 2051, 12345, 130000]],
    ], dtype=torch.int32)
    embedding = torch.zeros(1, 5, 1, dtype=torch.float16)
    axes = tuple(positions.unbind(0))
    for expected, actual in zip(native(embedding, positions), shared(embedding, *axes), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    from_qwen35 = _Qwen3_5TextModel._build_qwen3_5_mrope_position_embeddings(
        None, time_position_ids=positions[0, 0].long(), hight_position_ids=positions[1, 0].long(),
        width_position_ids=positions[2, 0].long(), inv_freq=native.inv_freq,
        attention_scaling=native.attention_scaling, time_mask=shared.time_mask.squeeze(0).half(),
        hight_mask=shared.height_mask.squeeze(0).half(), width_mask=shared.width_mask.squeeze(0).half(),
        use_precomputed_cache=True, cos_cached=shared.cos_cached.half(), sin_cached=shared.sin_cached.half(),
    )
    for expected, actual in zip(shared(embedding, *axes), from_qwen35, strict=True):
        torch.testing.assert_close(actual.squeeze(1), expected, rtol=0, atol=0)
    assert shared(embedding, *axes)[0][0, 1, 0] != shared(embedding, *axes)[0][0, 2, 0]


@pytest.mark.parametrize("past,current", [(0, 3), (1, 3), (3, 2), (4, 3), (7, 2), (8, 3), (11, 2), (14, 2)])
def test_selected_kv_primitive_attention_preserves_partial_blocks_and_padding(past, current):
    import xhquant.nn as xhnn
    from xhmodel_merak.xh_llm.models.qwen4_exp.qsa_host_controls import build_qsa_host_controls
    from xhmodel_merak.xh_llm.models.qwen4_exp.selected_kv_attention import (
        Qwen4ExpSelectedKVAttention,
    )

    torch.manual_seed(53)
    query = torch.randn(1, 6, 3, 8).half()
    key, value = (torch.randn(1, 2, 17, 8).half() for _ in range(2))
    key[:, :, past + current:] = 0
    value[:, :, past + current:] = 0
    index_query = torch.randn(1, 2, 3, 8).half()
    blocks = torch.randn(1, 1, 4, 8).half()
    counts = torch.tensor([[past + i + 1 if i < current else 0 for i in range(3)]], dtype=torch.int32)
    module = Qwen4ExpSelectedKVAttention(
        Qwen4ExpQSAConfig(num_heads=6, num_kv_heads=2, head_dim=8, scale=8**-.5),
        candidate_topk=2, index_head_dim=8, input_length=3,
    )
    assert module.select_kv.pool_size == 4
    assert not any(isinstance(child, xhnn.FlashAttention) for child in module.modules())
    assert isinstance(module.mask_add, xhnn.MaskedAdd)
    observed_masks = []
    module.mask_add.register_forward_pre_hook(lambda _, args: observed_masks.append(args[1].detach().clone()))
    controls = build_qsa_host_controls(past, current, 3, 17, 2, device="cpu", dtype=query.dtype)
    actual, next_key_tail, next_value_tail = module(
        query, index_query, blocks, key, value, key[:, :, past:past + 3], value[:, :, past:past + 3],
        _initial_tail(key, past), _initial_tail(value, past), controls[4],
        torch.tensor([past], dtype=torch.int32), torch.tensor([current], dtype=torch.int32),
    )
    torch.testing.assert_close(next_key_tail, _initial_tail(key, past + current))
    torch.testing.assert_close(next_value_tail, _initial_tail(value, past + current))
    payload = _pack_kv_payload(key, value)
    selected = xhnn.SparseAttentionSelectKV(candidate_topk=2, score_scale=8**-.5,
                                           pool_size=4, compression_ratio=4)(
        index_query, blocks, payload, torch.tensor([past], dtype=torch.int32),
        current_length=torch.tensor([current], dtype=torch.int32))
    expected = torch.zeros_like(actual)
    for position in range(current):
        visible = int(counts[0, position])
        full = min(visible // 4, 2) * 4
        tokens = torch.cat((selected[0, position, :full], payload[0, 0, (visible // 4)*4:visible]))
        k, v = _unpack_selected_kv(tokens[None, None], 2, 8)
        k, v = (t[0, 0].repeat_interleave(3, dim=0) for t in (k, v))
        scores = (query[0, :, position:position+1] @ k.transpose(-1, -2)) * (8**-.5)
        expected[0, :, position] = (scores.softmax(-1) @ v).squeeze(-2)
    torch.testing.assert_close(actual[:, :, :current], expected[:, :, :current], atol=1e-3, rtol=1e-3)
    assert torch.isfinite(actual).all()
    mask = observed_masks[0].reshape(3, -1)
    for position in range(3):
        visible = int(counts[0, position])
        full = min(visible // 4, 2) * 4
        valid = torch.zeros(11, dtype=torch.bool)
        valid[:full] = True
        valid[11 - visible % 4:] = True
        torch.testing.assert_close(mask[position, valid], torch.zeros_like(mask[position, valid]))
        torch.testing.assert_close(
            mask[position, ~valid], torch.full_like(mask[position, ~valid], torch.finfo(query.dtype).min)
        )


@pytest.mark.parametrize("past,current", [(0, 3), (3, 3), (2047, 3), (262141, 3)])
def test_qsa_host_controls_cover_long_context_and_padding(past, current):
    from xhmodel_merak.xh_llm.models.qwen4_exp.qsa_host_controls import build_qsa_host_controls

    _, _, _, query_valid, mask = build_qsa_host_controls(
        past, current, 8, 262144, 512, device="cpu", dtype=torch.float16,
    )
    for position in range(8):
        visible = past + position + 1 if position < current else 0
        blocks, remainder = divmod(visible, 4)
        assert bool(query_valid[0, position]) == (position < current)
        valid = torch.zeros(2051, dtype=torch.bool)
        valid[: min(blocks, 512) * 4] = True
        valid[2051 - remainder:] = True
        assert torch.equal(mask[0, position, 0] == 0, valid)


def test_qsa_streaming_pool_matches_native_across_prefill_and_decode():
    import copy
    from types import SimpleNamespace

    from transformers import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextQSAIndexer, apply_rotary_pos_emb

    from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import Qwen4ExpExportableQSAIndexer
    from xhmodel_merak.xh_llm.models.qwen4_exp.qsa_host_controls import build_qsa_host_controls
    from xhquant.core import CacheTensor

    torch.manual_seed(275)
    config = Qwen4ExpTextConfig(hidden_size=16, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8,
                                indexer_budget=4, indexer_compress_ratio=4,
                                rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.5})
    native = Qwen4ExpTextQSAIndexer(config, layer_idx=0).half().eval()
    indexer = Qwen4ExpExportableQSAIndexer(copy.deepcopy(native), sequence_length=3).eval()
    layer = SimpleNamespace(indexer_keys=CacheTensor(torch.zeros(1, 1, 4, 8).half()),
                            indexer_tail=torch.zeros(1, 4, 8).half())
    angles = torch.randn(1, 17, 2).repeat_interleave(2, dim=-1).half()
    cos, sin = angles.cos(), angles.sin()
    history = []
    past = 0
    with torch.no_grad():
        for width, current in ((3, 3), (4, 2), (1, 1), (4, 4), (3, 3), (4, 4)):
            indexer.set_sequence_length(width)
            hidden = torch.randn(1, width, 16).half()
            history.append(hidden[:, :current])
            controls = build_qsa_host_controls(past, current, width, 17, 1, device="cpu", dtype=hidden.dtype)
            cache = SimpleNamespace(layers=[layer], qsa_host_controls=controls,
                                    current_position_indices=torch.arange(past, past + width).view(1, -1),
                                    current_input_length=torch.tensor([current], dtype=torch.int32))
            group_positions = torch.arange(past // 4, past // 4 + indexer.pool_groups) * 4
            compact_cos = torch.cat((cos[:, past:past + width], cos[:, group_positions]), dim=1)
            compact_sin = torch.cat((sin[:, past:past + width], sin[:, group_positions]), dim=1)
            indexer(hidden, (compact_cos, compact_sin), cache)
            past += current
            all_hidden = torch.cat(history, dim=1)
            raw = native.index_qk_proj(all_hidden)[..., 16:].reshape(1, past, 8)
            complete = past // 4
            if complete:
                grouped = raw[:, :complete * 4].reshape(1, complete, 4, 8)
                reference = native.k_layernorm(grouped.float().mean(dim=2).to(raw.dtype)).unsqueeze(1)
                reference = apply_rotary_pos_emb(reference, cos=cos[:, :complete * 4:4],
                                                 sin=sin[:, :complete * 4:4])
                torch.testing.assert_close(layer.indexer_keys[:, :, :complete], reference, atol=2e-3, rtol=2e-3)
            torch.testing.assert_close(layer.indexer_tail[:, -min(past, 4):], raw[:, -4:], atol=0, rtol=0)


def test_qsa_selected_blocks_match_native_scoring_with_pruning():
    from xhmodel_merak.xh_llm.models.qwen4_exp.qsa_host_controls import build_qsa_host_controls
    from xhmodel_merak.xh_llm.models.qwen4_exp.selected_kv_attention import Qwen4ExpSelectedKVAttention

    torch.manual_seed(298)
    query = torch.randn(1, 4, 3, 8).half()
    index_query = torch.zeros(1, 2, 3, 8).half()
    block_keys = torch.zeros(1, 1, 4, 8).half()
    for position in range(3):
        index_query[0, 0, position, position] = 1
        block_keys[0, 0, position, position] = 3
    key, value = (torch.randn(1, 2, 17, 8).half() for _ in range(2))
    module = Qwen4ExpSelectedKVAttention(
        Qwen4ExpQSAConfig(num_heads=4, num_kv_heads=2, head_dim=8), candidate_topk=1, index_head_dim=8,
        input_length=3,
    )
    controls = build_qsa_host_controls(12, 3, 3, 17, 1, device="cpu", dtype=query.dtype)
    actual, _, _ = module(
        query, index_query, block_keys, key, value, key[:, :, 12:15], value[:, :, 12:15],
        _initial_tail(key, 12), _initial_tail(value, 12), controls[4],
        torch.tensor([12], dtype=torch.int32), torch.tensor([3], dtype=torch.int32),
    )
    expected = torch.zeros_like(actual)
    for position in range(3):
        visible = 13 + position
        scores = torch.relu(index_query[0, :, position].float() @ block_keys[0, 0, :3].float().T).sum(0)
        chosen = scores.argmax().item()
        indices = list(range(chosen * 4, chosen * 4 + 4)) + list(range(12, visible))
        selected_key = key[0, :, indices].repeat_interleave(2, dim=0)
        selected_value = value[0, :, indices].repeat_interleave(2, dim=0)
        logits = (query[0, :, position:position + 1] @ selected_key.transpose(-1, -2)) * (8 ** -0.5)
        expected[0, :, position] = (logits.softmax(-1) @ selected_value).squeeze(-2)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


def test_selected_kv_export_uses_tail_state_and_local_attention_precision(tmp_path):
    import onnx

    import xhquant.nn as xhnn
    from xhmodel_merak.xh_llm.models.qwen4_exp.hmonnx_export import (
        _align_graph_constants_with_consumers,
        _dynamo_export_compatibility_scope,
        _export_qwen4_exp_hmonnx_program,
        _normalize_frontend_gather_overloads,
        sanitize_qwen4_exp_frontend_graph,
    )
    from xhmodel_merak.xh_llm.models.qwen4_exp.qsa_host_controls import build_qsa_host_controls
    from xhmodel_merak.xh_llm.models.qwen4_exp.selected_kv_attention import Qwen4ExpSelectedKVAttention
    from xhmodel_merak.xh_llm.models.qwen4_exp.xhquant_extensions import qwen4_exp_xhquant_extension_scope
    from xhquant.api import (
        ConfigDict,
        FrontendType,
        PrecisionMode,
        ptq_quantize,
        to_export_graph,
        to_frontend_graph,
        to_quant_graph,
    )

    torch.manual_seed(73)
    module = Qwen4ExpSelectedKVAttention(
        Qwen4ExpQSAConfig(num_heads=4, num_kv_heads=2, head_dim=8),
        candidate_topk=2, index_head_dim=8, input_length=2,
    ).eval()
    controls = build_qsa_host_controls(3, 1, 2, 9, 2, device="cpu", dtype=torch.float16)
    args = (torch.randn(1, 4, 2, 8).half(), torch.randn(1, 2, 2, 8).half(),
            torch.randn(1, 1, 2, 8).half(), torch.randn(1, 2, 9, 8).half(),
            torch.randn(1, 2, 9, 8).half(), torch.randn(1, 2, 2, 8).half(),
            torch.randn(1, 2, 2, 8).half(), torch.randn(1, 2, 3, 8).half(),
            torch.randn(1, 2, 3, 8).half(), controls[4],
            torch.tensor([3], dtype=torch.int32), torch.tensor([1], dtype=torch.int32))
    with qwen4_exp_xhquant_extension_scope(), _dynamo_export_compatibility_scope():
        frontend = to_frontend_graph(module, FrontendType.DynamoFX, args, enable_fuse=False)
        _normalize_frontend_gather_overloads(frontend)
        sanitize_qwen4_exp_frontend_graph(frontend)
        assert sum(isinstance(child, xhnn.MaskedAdd) for child in frontend.modules()) == 1
        quant = to_quant_graph(frontend, 'xh2a', ConfigDict({'quant_type': 'w8a8_sefp'}))
        ptq_quantize(quant, [list(args)], PrecisionMode.ALIGNED, ['cpu'], infer_shape=False)
        quant.fixed()
        _align_graph_constants_with_consumers(quant)
        exported = to_export_graph(quant, list(args))
        path = tmp_path / 'selected.onnx'
        _export_qwen4_exp_hmonnx_program(exported, args, path,
                                        input_names=tuple(f'input_{i}' for i in range(len(args))),
                                        output_names=('output', 'next_key_tail', 'next_value_tail'))
    graph = onnx.load(path, load_external_data=False)
    infos = {value.name: value for value in graph.graph.value_info}
    assert not any(node.op_type == 'FlashAttention' for node in graph.graph.node)
    assert not any(node.op_type == 'GatherElements' for node in graph.graph.node)
    assert not any(node.op_type in {'BitShift', 'Clip'} for node in graph.graph.node)
    assert not any(node.op_type == 'Cast' and any(attr.name == 'to' and attr.i == onnx.TensorProto.UINT32
                                               for attr in node.attribute) for node in graph.graph.node)
    assert not any(node.op_type == 'Max' and len(node.input) == 1 for node in graph.graph.node)
    assert not any(node.op_type == 'Cast' and any(attr.name == 'to' and attr.i == onnx.TensorProto.INT64
                                               for attr in node.attribute) for node in graph.graph.node)
    matmuls = [node for node in graph.graph.node if node.op_type == 'MatMul']
    assert len(matmuls) == 2
    for node in matmuls:
        attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        assert attrs['hmfp_act_man_bit'] == attrs['hmfp_weight_man_bit'] == 16
        assert infos[node.output[0]].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    softmax, = [node for node in graph.graph.node if node.op_type == 'Softmax']
    assert all(infos[name].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
               for name in (*softmax.input, *softmax.output))
    assert not any(node.op_type == 'Gather' for node in graph.graph.node)
    selector, = [node for node in graph.graph.node if node.op_type == 'SparseAttentionSelectKV']
    assert len(selector.output) == 2
    producers = {output: node for node in graph.graph.node for output in node.output}

    def cache_source(name):
        while name in producers and producers[name].op_type in {'Cast', 'Identity', 'HMIdentity'}:
            name = producers[name].input[0]
        return name

    assert cache_source(selector.input[2]) == 'input_3'
    assert cache_source(selector.input[6]) == 'input_4'
    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in selector.attribute}
    assert attrs['pool_size'] == attrs['compression_ratio'] == 4
