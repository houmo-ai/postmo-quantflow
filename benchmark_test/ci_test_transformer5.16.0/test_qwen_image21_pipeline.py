# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Host scheduling against the reusable-output ABI of deployed graph sessions."""

from types import SimpleNamespace

import pytest
import torch

from xhmodel_merak.xh_other_model.models.qwen_image21.pipeline import ImageRuntime


def runtime_fixture(**runtime_options):
    # These deterministic graph ports isolate Host behavior from learned model
    # accuracy. In particular, both requests share one mutable head output,
    # exactly the ownership hazard of an HMONNX session used for CFG.
    shared = torch.empty(1, 4, 4, dtype=torch.float16)
    requests, decoded = [], []

    def encode(prompt, images):
        value = 1 if prompt == "positive" else -1
        return torch.full((1, 1, 4), value, dtype=torch.float16), torch.zeros(1, 1, dtype=torch.bool)

    def new_request(layout, rope, features, references):
        assert references.shape == (1, 0, 4)
        value = features[0, 0, 0].item()

        def request(latents, timestep):
            assert latents.shape == (1, 4, 4) and latents.dtype == torch.float16
            shared.fill_(value)
            return shared

        requests.append(request)
        return request

    def decode(latents):
        decoded.append(latents.clone())
        return torch.nn.functional.interpolate(latents.float(), size=(32, 32)).half()

    denoiser = SimpleNamespace(
        device="cpu",
        meta={"latent_channels": 4},
        gear=SimpleNamespace(prefix_tokens=4, target_tokens=4),
        new_request=new_request,
    )
    runtime = ImageRuntime(
        encode,
        denoiser,
        None,
        decode,
        {"latents_mean": [1, 2, 3, 4], "latents_std": [0.5, 0.25, 0.125, 0.0625]},
        {"use_dynamic_shifting": False},
        lambda shapes, mask, device: None,
        dict(
            output_size=[32, 32],
            reference_size=[32, 32],
            canvas_size=[32, 32],
            target_capacity=4,
            reference_capacity=4,
            visual_patch_capacity=4,
        ),
        **runtime_options,
    )
    return runtime, requests, decoded


def test_guidance_owns_branch_output_and_preserves_native_noise_order():
    runtime, requests, decoded = runtime_fixture()
    image, latent = runtime.generate(
        "positive", height=32, width=32, steps=1, seed=23, negative_prompt="negative", guidance_scale=2
    )
    initial = torch.randn((1, 1, 4, 2, 2), generator=torch.Generator().manual_seed(23))
    expected = (initial.reshape(1, 4, 4).transpose(1, 2) - 3).half()
    torch.testing.assert_close(latent, expected, atol=0, rtol=0)
    restored = expected.float().transpose(1, 2).reshape(1, 4, 2, 2) * runtime.std + runtime.mean
    torch.testing.assert_close(decoded[0], restored.half(), atol=0, rtol=0)
    assert image.mode == "RGBA" and image.size == (32, 32)
    assert len(requests) == 2 and requests[0] is not requests[1]
    _, second = runtime.generate(
        "positive", height=32, width=32, steps=1, seed=23, negative_prompt="negative", guidance_scale=2
    )
    torch.testing.assert_close(second, latent, atol=0, rtol=0)
    assert len(requests) == 4 and requests[2] is not requests[0]


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"steps": 0}, "steps"),
        ({"height": 32.0}, "multiples"),
        ({"guidance_scale": float("nan")}, "guidance_scale"),
        ({"guidance_scale": 2}, "negative prompt"),
        ({"height": 64}, "spatial canvas"),
    ],
)
def test_generation_rejects_unsupported_requests(kwargs, error):
    runtime, requests, _ = runtime_fixture()
    arguments = {"height": 32, "width": 32, **kwargs}
    with pytest.raises(ValueError, match=error):
        runtime.generate("positive", **arguments)
    assert not requests
