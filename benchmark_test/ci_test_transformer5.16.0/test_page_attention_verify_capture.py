"""Preserve MTP per-token kernel choices when selecting a cached graph."""
from types import MethodType, SimpleNamespace

import torch

from xhmodel_merak.xh_llm.hmonnx.base_llm_hmonnx_model import BaseLLMHMONNXModel
from xhquant.core.hmfp_kv_cache import allocate_hmfp_paged_kv_cache
from xhquant.xhonnxruntime.parsers.page_attention import PageAttention


def test_verify_context_separates_boundary_graphs_and_reuses_metadata():
    page = PageAttention(2, 1, -1.0, True, None, 8, 8, 8, 8)
    module = torch.nn.Module()
    module.add_module('attention', page)
    graph = torch.fx.Graph()
    graph.output(graph.call_module('attention'))
    executable = torch.fx.GraphModule(module, graph)
    selected = []
    interpreter = SimpleNamespace(set_capture_cache_key=selected.append)
    session = SimpleNamespace(graph_module=executable, interpreter=interpreter)
    model = SimpleNamespace(_llm_prefill=False, decode_model=SimpleNamespace(hmonnx_session=session))
    model._get_page_attention_modules = MethodType(BaseLLMHMONNXModel._get_page_attention_modules, model)
    cache = allocate_hmfp_paged_kv_cache(num_blocks=64, block_size=64,
        num_kv_heads=1, head_size=64, device='cpu')
    pointers = []
    for length in [827, 1023, 1024, 1025, 1027, 1028, 2048]:
        slots = torch.arange(length-5, length, dtype=torch.int64)
        BaseLLMHMONNXModel.set_page_attention_context(model, [cache],
            torch.arange(64, dtype=torch.int64), slots, 64, effective_kv_length=length)
        context = page._get_context()
        assert torch.equal(context.slot_mapping, slots)
        assert context.effective_kv_length == length
        pointers.append(context.slot_mapping.data_ptr())
    assert len(set(pointers)) == 1
    assert selected[0] == selected[1]
    assert selected[5] == selected[6]
    assert len(set(selected[1:6])) == 5
