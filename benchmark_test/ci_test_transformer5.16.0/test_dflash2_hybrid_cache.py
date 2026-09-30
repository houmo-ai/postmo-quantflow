"""Sliding context storage must preserve DFlash2 attention across proposals."""
import torch


def test_compact_attention_matches_full_cache_across_partial_commits(monkeypatch):
    from xhquant.core import CacheTensor, HybridCacheTensor
    from xhquant.core.context import get_current_context
    from xhmodel_merak.xh_llm.models.qwen3_5._dflash_model_impl import DFlashCrossAttention

    monkeypatch.setattr(get_current_context().kvcache, "fast_mode", False)
    torch.manual_seed(531)
    window, chunk, block = 8, 4, 4
    common = dict(hidden_size=64, num_attention_heads=2, num_key_value_heads=1,
                  head_dim=32, rms_norm_eps=1e-6, input_sequence_length=chunk,
                  max_pe_length=128, rope_theta=10000, use_cache=True,
                  query_dependent_mask=True)
    context = DFlashCrossAttention(**common, cache_window=window).eval()
    decode = DFlashCrossAttention(**common).eval()
    decode.load_state_dict(context.state_dict())
    compact = [HybridCacheTensor(torch.zeros(1, 1, 32, 32)) for _ in range(2)]
    full = [CacheTensor(torch.zeros(1, 1, 128, 32)) for _ in range(2)]
    absolute = 0

    def mask(length, capacity):
        q = length + torch.arange(block)[:, None]
        k = torch.arange(capacity)[None, :]
        visible = (k < length + block) & ((q-k).abs() < window)
        return torch.where(visible, 0.0, -65504.0).unsqueeze(0)

    with torch.no_grad():
        # Includes full prefill chunks, partial acceptance, and repeated window shifts.
        for accepted in [4, 4, 1, 2, 4, 1, 3, 4]:
            hidden = torch.randn(1, chunk, 64)
            position, count = torch.tensor([absolute]), torch.tensor([accepted])
            context.build_target_kv(hidden, position, count, *compact)
            decode.build_target_kv(hidden, position, count, *full)
            absolute += accepted
            retained = compact[0].cache_valid_len
            for local, reference in zip(compact, full):
                torch.testing.assert_close(local[:, :, :retained],
                                           reference[:, :, absolute-retained:absolute])
            noise = torch.randn(1, block, 64)
            pos, count = torch.tensor([absolute]), torch.tensor([block])
            actual = decode.forward_decode(
                noise, pos, count, *[CacheTensor(c.data) for c in compact],
                mask(retained, 32), cache_position=torch.tensor([retained]))
            expected = decode.forward_decode(noise, pos, count, *full, mask(absolute, 128))
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            assert all(c.cache_valid_len == retained for c in compact)


def test_context_hmonnx_updates_hybrid_cache(tmp_path):
    import onnx
    from xhquant.api import (to_frontend_graph, to_quant_graph, ptq_quantize,
                             PrecisionMode, convert_quanted_model_to_hmonnx)
    from xhquant.core import HybridCacheTensor
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceV2, HMONNXInferenceConfig
    from xhquant.xhonnxruntime.llm_hmonnx_loader import MultiHMONNXLoader
    from xhmodel_merak.xh_llm.models.qwen3_5._dflash_model_impl import DFlashModel
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_dflash_model import _build_dflash_export_adapter

    torch.set_num_threads(2)
    core = DFlashModel(mode="context", hidden_size=64, num_attention_heads=2,
        num_key_value_heads=1, head_dim=32, intermediate_size=128, num_hidden_layers=1,
        rms_norm_eps=1e-6, num_target_layers=1, target_layer_ids=[0], vocab_size=128,
        input_sequence_length=4, max_pe_length=128, max_sequence_length=32,
        architecture="DFlash2DraftModel", block_size=4, conv_kernel_size=2,
        conv_group_size=32, selector_rank=8, selector_top_k=8, sliding_window=8).half().eval()
    adapter = _build_dflash_export_adapter(core, mode="context", num_hidden_layers=1, is_dflash2=True)
    args = [torch.randn(1, 4, 64).half(), torch.tensor([0]), torch.tensor([4]),
            torch.zeros(1, 1, 32, 32).half(), torch.zeros(1, 1, 32, 32).half()]
    graph = to_quant_graph(to_frontend_graph(adapter, "TorchFX", args), "XH2a",
                           {"quant_type": "w8a8h1_sefp"})
    ptq_quantize(graph, [args], PrecisionMode.ALIGNED, ["cpu"], infer_shape=False)
    graph.fixed()
    path = str(tmp_path / "context.onnx")
    convert_quanted_model_to_hmonnx(graph, args, path)
    model = onnx.load(path, load_external_data=False)
    caches = [node for node in model.graph.node if node.op_type == "KVcache"]
    assert len(caches) == 2
    assert all(next(attr.i for attr in node.attribute if attr.name == "attention_max_length") == 8
               for node in caches)
    loader = MultiHMONNXLoader({"context": path})
    session = HMONNXInferenceV2.from_onnx_graph(path, loader.graphs["context"],
                                              HMONNXInferenceConfig(exec_devices=["cpu"]))
    runtime_args = [value.to(dtype=info.dtype) for value, info in zip(args, session.inputs, strict=True)]
    runtime_args[-2:] = [HybridCacheTensor(value) for value in runtime_args[-2:]]
    for absolute in [0, 4, 8, 12]:
        runtime_args[1].fill_(absolute)
        session(*runtime_args)
        assert all(cache.cache_valid_len == min(absolute, 7) + 4 for cache in runtime_args[-2:])
        assert all(torch.isfinite(cache).all() for cache in runtime_args[-2:])
