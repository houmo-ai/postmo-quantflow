"""DeepSeek CSA fused selection, causal Host inputs and real graph export."""

from types import SimpleNamespace

import onnx
import pytest
import torch

from xhmodel_merak.xh_llm.models.deepseek_v4.cache_abi import CSA, DeepSeekV4CacheABI
from xhmodel_merak.xh_llm.models.deepseek_v4.deepseek_v4_hmonnx_inference import XHDeepSeekV4HMONNXModel
from xhmodel_merak.xh_llm.models.deepseek_v4.deepseek_v4_model import XHDeepSeekV4Model
from xhmodel_merak.xh_llm.models.deepseek_v4.host_masks import build_deepseek_v4_attention_masks
from xhmodel_merak.xh_llm.models.deepseek_v4.projection import BoundGroupedOutputProjection
from xhmodel_merak.xh_llm.models.deepseek_v4.runtime import DeepSeekV4CacheMixin, DeepSeekV4DataPreprocess
from xhmodel_merak.xh_llm.models.deepseek_v4.static_attention import StaticCSAAttention
from xhmodel_merak.xh_llm.models.deepseek_v4.static_cache import DeepSeekV4StaticCacheSpec


@pytest.mark.parametrize(
    "past,current,fixed",
    [(0, 7, 256), (3, 1, 1), (4, 1, 1), (78, 7, 256), (2359, 1, 1), (43456, 256, 256), (261888, 256, 256)],
)
def test_host_candidate_lengths_and_selected_mask(past, current, fixed):
    spec = DeepSeekV4StaticCacheSpec()
    masks = build_deepseek_v4_attention_masks(
        spec, input_sequence_length=fixed, past_length=past, current_length=current
    )
    lengths = masks.csa_candidate_lengths
    expected = [(past + i + 1) // 4 if i < current else 0 for i in range(fixed)]
    assert lengths.dtype == torch.int32 and lengths.shape == (1, fixed)
    assert lengths[0].tolist() == expected
    selected_mask = masks.csa_attention_mask[0, 0, :, :512]
    assert (selected_mask == 0).sum(-1).tolist() == [min(n, 512) for n in expected]
    abi = DeepSeekV4CacheABI()
    assert abi.attention_mask_shapes("decode" if fixed == 1 else "prefill")["csa_candidate_lengths"] == (1, fixed)


@pytest.mark.parametrize("version", [None, 1, 3])
def test_runtime_rejects_incompatible_state_abi_before_loading(version):
    with pytest.raises(ValueError, match="state ABI expected 2.*re-export"):
        XHDeepSeekV4HMONNXModel(SimpleNamespace(deepseek_v4_state_abi_version=version))


def test_preprocessor_and_export_input_order():
    spec = DeepSeekV4StaticCacheSpec(
        max_context_length=64,
        prefill_chunk_length=8,
        sliding_window=4,
        latent_head_dim=128,
        index_head_dim=128,
        index_topk=4,
    )
    runtime = DeepSeekV4CacheMixin(DeepSeekV4CacheABI(spec=spec, layer_types=(CSA,)))
    processor = DeepSeekV4DataPreprocess(
        token_embedding=torch.nn.Embedding(16, 128).half(), input_sequence_length=8, cache_mixin=runtime, pad_token_id=0
    )
    inputs = processor({"input_ids": torch.tensor([[1, 2, 3]]), "past_seq_length": 3})
    model = object.__new__(XHDeepSeekV4Model)
    model.layer_types = (CSA,)
    model.cache_abi = runtime.abi
    names = model.get_export_cfg()["input_names"]
    position = names.index("csa_candidate_lengths")
    assert position == 8
    assert inputs[position].dtype == torch.int32
    assert inputs[position].tolist() == [[1, 1, 1, 0, 0, 0, 0, 0]]


def _attention_case(fixed, past, current, device):
    torch.manual_seed(46158)
    spec = DeepSeekV4StaticCacheSpec(
        max_context_length=64,
        prefill_chunk_length=8,
        sliding_window=16,
        latent_head_dim=128,
        index_head_dim=128,
        index_topk=4,
    )
    model = (
        StaticCSAAttention(
            hidden_size=128,
            q_lora_rank=64,
            num_heads=2,
            head_dim=128,
            num_groups=1,
            o_lora_rank=64,
            rope_dim=64,
            input_sequence_length=fixed,
            window_size=16,
            swa_backing_length=spec.swa_physical_length,
            rms_norm_eps=1e-6,
            cache_capacity=16,
            index_heads=2,
            index_head_dim=128,
            index_topk=4,
        )
        .eval()
        .to(device)
    )
    # Use the same bound, nonzero output projection as a loaded checkpoint.
    model.output_projection = BoundGroupedOutputProjection(
        torch.nn.Linear(256, 64, bias=False).half().to(device),
        torch.nn.Linear(64, 128, bias=False).half().to(device),
        num_heads=2,
        head_dim=128,
        num_groups=1,
        rope_dim=64,
    )
    masks = build_deepseek_v4_attention_masks(
        spec, input_sequence_length=fixed, past_length=past, current_length=current, device=device
    )
    states = model.initial_state(1, device=device)
    angles = torch.randn(1, fixed, 32, device=device, dtype=torch.float16)
    compressed_angles = torch.randn(1, (fixed + 3) // 4, 32, device=device, dtype=torch.float16)
    args = (
        torch.randn(1, fixed, 128, device=device, dtype=torch.float16),
        torch.tensor([past], dtype=torch.int32, device=device),
        torch.tensor([current], dtype=torch.int32, device=device),
        masks.csa_candidate_lengths,
        masks.csa_attention_mask,
        torch.zeros(1, 1, spec.swa_physical_length, 128, device=device, dtype=torch.float16),
        torch.randn(1, 1, 16, 128, device=device, dtype=torch.float16),
        torch.randn(1, 1, 16, 128, device=device, dtype=torch.float16),
        torch.tensor([past // 4], dtype=torch.int32, device=device),
        masks.csa_compressor_validity,
        masks.csa_compressor_new_count,
        masks.csa_compressor_offset,
        masks.csa_compressor_phase_indices,
        *states,
        angles.cos(),
        angles.sin(),
        compressed_angles.cos(),
        compressed_angles.sin(),
    )
    return model, args


@pytest.mark.parametrize("fixed,past,current", [(8, 0, 7), (8, 3, 8), (1, 7, 1), (1, 8, 1)])
def test_complete_csa_uses_one_fused_selection(fixed, past, current):
    model, args = _attention_case(fixed, past, current, "cpu")
    output = model(*args)
    assert output.output.shape == (1, fixed, 128) and torch.isfinite(output.output).all()
    assert torch.count_nonzero(output.output[:, current:]) == 0
    graph = torch.export.export(model, args).graph
    assert sum(node.target == torch.ops.xh.SparseAttentionSelectKV.default for node in graph.nodes) == 1
    assert not any(node.target == torch.ops.aten.topk.default for node in graph.nodes)
    assert sum(node.target == torch.ops.xh.LLMCache.default for node in graph.nodes) == 3
    # Check the actual Indexer inputs against an independent causal score/gather reference.
    captured = []
    handle = model.indexer.select_kv.register_forward_hook(
        lambda module, values, result: captured.append((values, result))
    )
    try:
        model(*args)
    finally:
        handle.remove()
    (q, keys, payload, total, weights, lengths), selected = captured[0]
    scores = (torch.relu(q @ keys.transpose(-1, -2)) * weights).sum(1) * (2 * 128) ** -0.5
    scores = scores.masked_fill(torch.arange(16).view(1, 1, -1) >= lengths.unsqueeze(-1), -torch.inf)
    indices = scores.argsort(dim=-1, descending=True, stable=True)[..., :4]
    expected = payload[0, 0][indices]
    keep = torch.arange(4).view(1, 1, -1, 1) < lengths.clamp_max(4).unsqueeze(-1).unsqueeze(-1)
    expected = torch.where(keep, expected, 0)
    torch.testing.assert_close(selected, expected, rtol=0, atol=0)
    assert total.item() == (past + current) // 4


class _CSAExport(torch.nn.Module):
    """Flatten the internal named tuple at the same boundary as the full model."""

    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(
        self,
        hidden,
        past,
        current,
        lengths,
        mask,
        swa_cache,
        main_cache,
        index_cache,
        write_start,
        validity,
        new_count,
        offset,
        phase,
        main_kv,
        main_score,
        index_kv,
        index_score,
        cos,
        sin,
        compressed_cos,
        compressed_sin,
    ):
        output = self.attention(
            hidden,
            past,
            current,
            lengths,
            mask,
            swa_cache,
            main_cache,
            index_cache,
            write_start,
            validity,
            new_count,
            offset,
            phase,
            main_kv,
            main_score,
            index_kv,
            index_score,
            cos,
            sin,
            compressed_cos,
            compressed_sin,
        )
        return (
            output.output,
            output.swa_kv_context,
            output.main_cache,
            output.index_k_cache,
            output.main_next_kv_state,
            output.main_next_score_state,
            output.index_next_kv_state,
            output.index_next_score_state,
        )


@pytest.mark.parametrize("fixed,past,current", [(8, 3, 7), (1, 7, 1)])
@pytest.mark.parametrize("quant_type", ["w8a8h1_sefp", "w8a16h1_sefp"])
def test_csa_hmonnx_export_and_runtime(tmp_path, fixed, past, current, quant_type):
    from xhquant.api import (
        HMONNXInference,
        PrecisionMode,
        ptq_quantize,
        to_export_hmonnx_from_quanted_graph,
        to_frontend_graph,
        to_quant_graph,
    )
    from xhquant.core import CacheTensor, HybridCacheTensor

    assert torch.cuda.is_available(), "CSA HMONNX regression requires a GPU"
    model, args = _attention_case(fixed, past, current, "cuda:0")
    args = list(args)
    args[5] = HybridCacheTensor(args[5])
    args[5].cache_valid_len = past
    args[6:8] = [CacheTensor(cache) for cache in args[6:8]]
    initial_caches = [cache.data.clone() for cache in args[5:8]]
    graph = to_quant_graph(to_frontend_graph(_CSAExport(model), "TorchFX", args), "XH2a", {"quant_type": quant_type})
    ptq_quantize(graph, [args], PrecisionMode.ALIGNED, ["cuda:0"], infer_shape=False)
    expected = [value.dequantize().clone() for value in graph(*args)]
    for cache, initial in zip(args[5:8], initial_caches, strict=True):
        cache.data.copy_(initial)
    args[5].cache_valid_len = past
    path = tmp_path / "csa.onnx"
    to_export_hmonnx_from_quanted_graph(graph, args, str(path))
    exported = onnx.load(path)
    nodes = [node for node in exported.graph.node if node.op_type == "SparseAttentionSelectKV"]
    assert len(nodes) == 1 and nodes[0].domain == "ai.houmo.xh2a"
    assert len(nodes[0].input) == 6 and len(nodes[0].output) == 1
    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in nodes[0].attribute}
    assert attrs["candidate_topk"] == 4 and attrs.get("pool_size", 1) == 1
    assert not any(node.op_type == "TopK" for node in exported.graph.node)
    length_input = exported.graph.input[3]
    assert length_input.type.tensor_type.elem_type == onnx.TensorProto.INT32
    assert [dim.dim_value for dim in length_input.type.tensor_type.shape.dim] == [1, fixed]
    session = HMONNXInference(str(path))
    session.to("cuda:0")
    for cache, initial in zip(args[5:8], initial_caches, strict=True):
        cache.data.copy_(initial)
    args[5].cache_valid_len = past
    actual = session(*args)
    if isinstance(actual, dict):
        actual = tuple(actual.values())
    for value, reference in zip(actual, expected, strict=True):
        assert value.dtype == torch.float16
        torch.testing.assert_close(value.float(), reference.float(), rtol=2e-3, atol=2e-3)
