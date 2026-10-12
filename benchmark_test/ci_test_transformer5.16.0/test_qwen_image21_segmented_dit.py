"""Complete ordered DiT segments against the native block-causal transformer."""

import pytest
import torch

from xhmodel_merak.xh_other_model.models.qwen_image21.artifacts import DenoiseGear
from xhmodel_merak.xh_other_model.models.qwen_image21.export import make_text_projection
from xhmodel_merak.xh_other_model.models.qwen_image21.full_graph import SegmentDenoiser
from xhmodel_merak.xh_other_model.models.qwen_image21.layout import prepare_layout
from xhmodel_merak.xh_other_model.models.qwen_image21.segment_runtime import SegmentDenoiseRequest


@pytest.mark.parametrize(
    "slots,reference_shapes,roles",
    [
        ([False, False, False], [], ["text"]),
        ([False, True, False, False], [(1, 2, 2)], ["text", "image", "text"]),
        ([False, True, False, True, False], [(1, 2, 2), (1, 2, 2)], ["text", "image", "text", "image", "text"]),
        ([False, True, True, False], [(1, 2, 2), (1, 2, 2)], ["text", "image", "image", "text"]),
    ],
)
@pytest.mark.parametrize("text_capacity", [32, 576, 768])
@pytest.mark.parametrize("native_attention", [False, True])
def test_ordered_segments_match_native_and_overwrite_only_target(
    slots, reference_shapes, roles, text_capacity, native_attention
):
    from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21Transformer2DModel

    torch.manual_seed(17)
    source = QwenImage21Transformer2DModel(
        patch_size=1,
        in_channels=4,
        out_channels=4,
        context_in_dim=96,
        num_layers=3,
        num_attention_heads=2,
        attention_head_dim=64,
        axes_dims_rope=(16, 24, 24),
        causal_condition=True,
    ).eval()
    gear = DenoiseGear(32, 16, 64, 16, text_tokens=text_capacity)
    graphs = {
        "text": SegmentDenoiser(source, gear.text_tokens, text_projection=make_text_projection(source)),
        "image": SegmentDenoiser(source, gear.image_tokens, cache_capacity=gear.cache_tokens,
                                 use_noncausal_flash_attention=native_attention),
    }
    image_slots = torch.tensor([slots])
    features = torch.randn(1, len(slots), 96)
    references = torch.randn(1, 4 * len(reference_shapes), 4)

    def request(target_shape=(1, 2, 2)):
        shapes = [*reference_shapes, target_shape]
        layout = prepare_layout(
            image_slots, shapes, prefix_capacity=gear.prefix_tokens, target_capacity=gear.target_tokens
        )
        rope = source.pos_embed(shapes, layout.compact_image_mask, device="cpu")
        return SegmentDenoiseRequest(
            layout,
            rope,
            features,
            references,
            text=graphs["text"],
            image=graphs["image"],
            gear=gear,
            layers=3,
            heads=2,
            head_dim=64,
        )

    current = request()
    assert [role for _, _, role in current.segments] == roles
    with torch.inference_mode():
        current.initialize_prefix()
        p = current.layout.prefix_tokens
        prefix = [x[:, :, :p].clone() for x in current.caches]
        first_target = None
        for time in (1.0, 0.6, 0.2):
            target = torch.randn(1, 4, 4)
            actual = current(target, time)
            native = source(
                hidden_states=torch.cat((references, target), dim=1),
                encoder_hidden_states=features,
                timestep=torch.tensor([time]),
                img_shapes=[[*reference_shapes, (1, 2, 2)]],
                img_mask=torch.cat((image_slots, torch.ones(1, 1, dtype=torch.bool)), dim=1),
                return_dict=False,
            )[0][:, -4:]
            torch.testing.assert_close(actual, native, atol=2e-6, rtol=2e-5)
            for index, cache in enumerate(current.caches):
                torch.testing.assert_close(cache[:, :, :p], prefix[index], atol=0, rtol=0)
                assert torch.count_nonzero(cache[:, :, p + 4 :]) == 0
            if first_target is None:
                first_target, first_output = target.clone(), actual.clone()
                for cache in current.caches:
                    cache[:, :, p + 4 :] = 1000
                torch.testing.assert_close(current(target, time), actual, atol=0, rtol=0)
                for cache in current.caches:
                    cache[:, :, p + 4 :] = 0
        other = request()
        torch.testing.assert_close(other(first_target, 1.0), first_output, atol=0, rtol=0)
        assert all(a.data_ptr() != b.data_ptr() for a, b in zip(current.caches, other.caches, strict=True))
        large = request((1, 2, 4))
        assert large(torch.randn(1, 8, 4), 0.5).shape == (1, 8, 4)
        torch.testing.assert_close(request()(first_target, 1.0), first_output, atol=0, rtol=0)
