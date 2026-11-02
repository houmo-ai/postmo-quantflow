# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Export and replay the compiler's short-KV and FlashAttention boundary."""

import onnx
import pytest
import torch

from xhmodel_merak.xh_other_model.common.attention import CompilerAttention
from xhmodel_merak.xh_other_model.models.sol_h3.export import export_graph


@pytest.mark.parametrize("kv", [511, 512])
def test_attention_export_and_replay_at_compiler_kv_boundary(tmp_path, kv):
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    class Attention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.core = CompilerAttention(num_heads=1, scale=0.25)

        def forward(self, q, k, v, mask):
            return self.core(q, k, v, key_mask=mask)

    q = torch.zeros(1, 1, 2, 64)
    k = torch.zeros(1, 1, kv, 64)
    q[0, 0, :, 0] = torch.tensor([4.0, -4.0])
    k[0, 0, 8:16, 0] = 1.0
    v = torch.full_like(k, 4.0)
    v[:, :, :8] = 0.5
    v[:, :, 8:16] = 1.5
    mask = torch.zeros(1, 1, 1, kv)
    mask[..., 16:] = -65504
    inputs = (q, k, v, mask)
    path = export_graph(
        Attention().eval(), inputs, tmp_path / "attention.onnx", ["q", "k", "v", "mask"], ["output"],
        quant_scheme={"target_device": "XH2a", "quant_type": "w8a8_sefp"},
    )
    nodes = onnx.load(path, load_external_data=False).graph.node
    flashes = [node for node in nodes if node.op_type == "FlashAttention"]
    if kv == 511:
        assert not flashes
        assert sum(node.op_type == "MatMul" for node in nodes) == 2
        assert sum(node.op_type == "Softmax" for node in nodes) == 1
    else:
        assert len(flashes) == 1 and len(flashes[0].input) == 5 and flashes[0].input[4] == "mask"

    runtime = HMONNXInferenceV2(str(path), session_config=HMONNXInferenceConfig(exec_devices=["cuda:0"]))
    payload = [value.cuda().half() for value in inputs]
    for current_mask in (mask, torch.zeros_like(mask)):
        payload[-1] = current_mask.cuda().half()
        actual = runtime(*payload).float().cpu()
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=current_mask, scale=0.25)
        torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.02)
