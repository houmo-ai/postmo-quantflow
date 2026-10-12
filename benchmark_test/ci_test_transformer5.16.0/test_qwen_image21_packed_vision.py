"""One packed patch graph preserves native Qwen3-VL geometry and DeepStack."""

import pytest
import torch
from test_qwen_image21_feature_encoder import tiny_config

from xhmodel_merak.xh_other_model.models.qwen_image21.vision_graph import PackedVision, geometry_inputs


@pytest.mark.parametrize("grid", [(2, 2), (2, 4), (4, 2), (4, 6)])
@pytest.mark.parametrize("head_dim", [16, 72])
@pytest.mark.parametrize("native_attention", [False, True])
def test_packed_capacity_matches_native_position_and_vision_features(grid, head_dim, native_attention):
    from xhmodel_merak.xh_llm.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

    torch.manual_seed(5)
    cfg = tiny_config().vision_config
    cfg.hidden_size = cfg.num_heads * head_dim
    cfg._attn_implementation = "sdpa"
    source = Qwen3VLVisionModel(cfg).eval()
    graph = PackedVision(source, 36, use_noncausal_flash_attention=native_attention).eval()
    count = grid[0] * grid[1]
    patches = torch.randn(count, 1536) * 0.2
    thw = torch.tensor([[1, *grid]])
    geometry = geometry_inputs(
        grid, 36, source.num_grid_per_side, cfg.hidden_size // cfg.num_heads, device="cpu", dtype=torch.float32
    )
    padded = torch.nn.functional.pad(patches, (0, 0, 0, 36 - count))[None]
    with torch.inference_mode():
        indices, weights, cos, sin, mask, length = geometry
        position = source.pos_embed(indices.long()) * weights[:, :, None]
        position = sum(position[i] for i in range(4))
        torch.testing.assert_close(position[:count], source.fast_pos_embed_interpolate(thw), atol=0, rtol=0)
        angle = source.rot_pos_emb(thw)
        angle = torch.cat((angle, angle), dim=-1)
        torch.testing.assert_close(cos[0, 0, :count], angle.cos(), atol=0, rtol=0)
        torch.testing.assert_close(sin[0, 0, :count], angle.sin(), atol=0, rtol=0)
        main, deep = source.forward_ori(patches, thw)
        actual = graph(padded, *geometry)
        for value, expected in zip(actual, (main, *deep), strict=True):
            torch.testing.assert_close(value[:, : count // 4], expected[None], atol=2e-5, rtol=3e-5)
            assert torch.count_nonzero(value[:, count // 4 :]) == 0
        dirty = graph(padded + 1000 * (1 - mask), *geometry)
        for value, expected in zip(dirty, actual, strict=True):
            torch.testing.assert_close(value, expected, atol=0, rtol=0)


@pytest.mark.parametrize("grid", [(1, 4), (2, 3), (6, 8), (0, 4)])
def test_packed_geometry_rejects_nonmerged_or_oversized_grids(grid):
    with pytest.raises(ValueError):
        geometry_inputs(grid, 36, 48, 72, device="cpu", dtype=torch.float16)
