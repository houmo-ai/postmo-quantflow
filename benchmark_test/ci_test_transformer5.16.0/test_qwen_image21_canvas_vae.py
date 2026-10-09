"""Real residual VAE boundaries under one canvas and changing request geometry."""

import pytest
import torch
from torch.nn import functional as F

from xhmodel_merak.xh_other_model.models.qwen_image21.canvas_vae import (
    CanvasVaeDecode,
    CanvasVaeEncode,
    CanvasVaeSession,
    spatial_inputs,
)


@pytest.fixture(scope="module")
def source():
    from diffusers.models.autoencoders.autoencoder_kl_qwenimage21 import AutoencoderKLQwenImage21

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        return AutoencoderKLQwenImage21(
            base_dim=8, decoder_base_dim=8, z_dim=4, num_res_blocks=2, latents_mean=[0.0] * 4, latents_std=[1.0] * 4
        ).eval()


@pytest.mark.parametrize("actual_hw", [(32, 32), (32, 64), (64, 32), (64, 96)])
def test_canvas_encode_decode_match_native_and_exclude_dirty_padding(source, actual_hw):
    torch.manual_seed(11)
    height, width = actual_hw
    canvas = (96, 96)
    image = torch.randn(1, 4, height, width)
    latent = torch.randn(1, 4, height // 16, width // 16)
    masks = spatial_inputs(actual_hw, canvas, dtype=torch.float32)
    encode, decode = CanvasVaeEncode(source).eval(), CanvasVaeDecode(source).eval()
    with torch.inference_mode():
        native_latent = source.encode(image.unsqueeze(2), return_dict=False)[0].mode().squeeze(2)
        native_image = source.decode(latent.unsqueeze(2), return_dict=False)[0].clamp(-1, 1).squeeze(2)
        padded_image = F.pad(image, (0, canvas[1] - width, 0, canvas[0] - height))
        padded_latent = F.pad(latent, (0, (canvas[1] - width) // 16, 0, (canvas[0] - height) // 16))
        encoded, decoded = encode(padded_image, *masks), decode(padded_latent, *masks)
        torch.testing.assert_close(encoded[..., : height // 16, : width // 16], native_latent, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(decoded[..., :height, :width], native_image, atol=1e-5, rtol=3e-5)
        assert torch.count_nonzero(encoded * (1 - masks[4])) == 0
        assert torch.count_nonzero(decoded * (1 - masks[0])) == 0
        torch.testing.assert_close(encode(padded_image + 1000 * (1 - masks[0]), *masks), encoded, atol=0, rtol=0)
        torch.testing.assert_close(decode(padded_latent + 1000 * (1 - masks[4]), *masks), decoded, atol=0, rtol=0)


def test_canvas_session_reuses_graph_across_sizes_without_request_residue(source):
    canvas = (96, 96)
    module = CanvasVaeDecode(source).eval()
    session = CanvasVaeSession(module, canvas, encode=False, token_capacity=36)
    small = torch.randn(1, 4, 2, 4)
    large = torch.randn(1, 4, 4, 6)
    with torch.inference_mode():
        first = session(small)
        assert session(large).shape == (1, 4, 64, 96)
        torch.testing.assert_close(session(small), first, atol=0, rtol=0)
    for actual in ((48, 64), (128, 32), (0, 32)):
        with pytest.raises(ValueError):
            spatial_inputs(actual, canvas)


def test_canvas_attention_permutation_has_row_holes_and_restores_exactly():
    masks, order, restore, length = spatial_inputs((32, 64), (96, 96))[-4:]
    assert length.item() == 8
    assert order[:8].tolist() == [0, 1, 2, 3, 6, 7, 8, 9]
    values = torch.arange(36)
    torch.testing.assert_close(values.index_select(0, order).index_select(0, restore), values)
