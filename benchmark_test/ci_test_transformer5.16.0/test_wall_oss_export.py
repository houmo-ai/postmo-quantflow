"""Real three-graph SDK export and GPU inference with windowed vision and routed KV."""

import pytest
import torch
from _wall_oss_small_fixture import make_small_source
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

from xhmodel_merak.xh_other_model.models.wall_oss.build import export_graph
from xhmodel_merak.xh_other_model.models.wall_oss.graph import ActionDenoiser, ExpertStack
from xhmodel_merak.xh_other_model.models.wall_oss.processor import vision_geometry
from xhmodel_merak.xh_other_model.models.wall_oss.vision import PackedVision


@pytest.mark.parametrize("precision", ["w16a16_sefp", "w8a16_sefp", "w8a8_sefp"])
def test_real_three_graph_cascade_and_cache_reset(tmp_path, precision):
    from xhquant.core import CacheTensor
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    torch.manual_seed(4)
    cfg = Qwen2_5_VLVisionConfig(
        depth=2, hidden_size=160, intermediate_size=192, num_heads=2, out_hidden_size=128, fullatt_block_indexes=[1]
    )
    cfg._attn_implementation = "sdpa"
    vision = Qwen2_5_VisionTransformerPretrainedModel(cfg).eval()
    source = make_small_source()
    cache_shape = (1, 1, 512, 64)
    caches = tuple(CacheTensor(torch.zeros(cache_shape)) for _ in range(4))
    patches = torch.randn(1, 128, 1176)
    geometry = vision_geometry(12, 10, capacity=128, head_dim=80)
    prefix_inputs = (
        torch.randn(1, 256, 128),
        torch.ones(1, 1, 256, 64),
        torch.zeros(1, 1, 256, 64),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([35], dtype=torch.int32),
        caches,
    )
    action_inputs = (
        torch.randn(1, 4, 6),
        torch.ones(1, 4, 6),
        torch.randn(1, 64),
        torch.ones(1, 1, 4, 64),
        torch.zeros(1, 1, 4, 64),
        torch.tensor([35], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        caches,
    )
    cache_names = [f"cache_{i}" for i in range(4)]
    graphs = {
        "visual": (
            PackedVision(vision, 128),
            (patches, *geometry),
            ["patches", "order", "reverse", "cos", "sin", "mask", "full_range", "window_range"],
        ),
        "prefill": (
            ExpertStack(source, expert=0, query_capacity=256, cache_capacity=512),
            prefix_inputs,
            ["embeddings", "cos", "sin", "past", "current", *cache_names],
        ),
        "denoise": (
            ActionDenoiser(source, query_capacity=4, cache_capacity=512),
            action_inputs,
            ["action", "dof", "time", "cos", "sin", "past", "current", *cache_names],
        ),
    }
    sessions = {}
    for name, (module, inputs, names) in graphs.items():
        path = tmp_path / name / (name + ".onnx")
        export_graph(module, inputs, path, names, "cuda:0", precision)
        sessions[name] = HMONNXInferenceV2(str(path), session_config=HMONNXInferenceConfig(exec_devices=["cuda:0"]))
        if name != "visual":
            assert sessions[name].enable_inplace_cache_updates() == 4

    def move(x):
        return x.cuda().half() if x.is_floating_point() else x.cuda()

    visual = sessions["visual"](move(patches), *[move(x) for x in geometry]).clone()
    assert torch.isfinite(visual).all()
    assert visual[:, 30:].count_nonzero() == 0
    prefix = prefix_inputs[0].clone()
    prefix[:, :30] = visual[:, :30].float().cpu()

    def run(action):
        cache = tuple(CacheTensor(torch.zeros(cache_shape, device="cuda", dtype=torch.float16)) for _ in range(4))
        sessions["prefill"](move(prefix), *[move(x) for x in prefix_inputs[1:-1]], *cache)
        frozen = tuple(x.data[:, :, :35].clone() for x in cache)
        result = sessions["denoise"](move(action), *[move(x) for x in action_inputs[1:-1]], *cache).clone()
        for old, new in zip(frozen, cache, strict=True):
            torch.testing.assert_close(old, new.data[:, :, :35], rtol=0, atol=0)
        return result

    actual = run(action_inputs[0])
    again = run(action_inputs[0])
    torch.testing.assert_close(actual, again, rtol=0, atol=0)
    with torch.no_grad():
        expected = source.joint(
            prefix[:, :35],
            action_inputs[0],
            action_inputs[1],
            action_inputs[2],
            torch.ones(1, 1, 39, 64),
            torch.zeros(1, 1, 39, 64),
        )
    error = torch.linalg.vector_norm(actual.float().cpu() - expected) / torch.linalg.vector_norm(expected)
    assert error < 0.12, float(error)
