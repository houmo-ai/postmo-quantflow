# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Native reference and exported visual features must preserve model numerics."""

import pytest
import torch


def test_native_flow_and_visual_export(tmp_path, monkeypatch):
    from _pi05_small_fixture import write_pi05
    from lerobot.policies import pi_gemma
    from xhmodel_merak.xh_other_model.models.pi05._export_utils import load_pi05_policy, Siglip

    source = tmp_path / 'model'
    write_pi05(source, tmp_path / 'tokenizer', monkeypatch)
    native_forward = pi_gemma.PiGemmaModel.forward
    native_mask = pi_gemma.create_causal_mask
    from examples_merak.vla.pi05.pi05_libero_fp_eval import make_strict_pi05_policy
    from examples_merak.vla.pi05.pi05_droid_hmonnx_validation import load_config
    from lerobot.envs.configs import LiberoEnv

    config = load_config(source, "cpu", 50)
    policy = make_strict_pi05_policy(config, env_cfg=LiberoEnv())
    tower = policy.model.paligemma_with_expert
    image = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        expected = tower.embed_image(image)
        actual = Siglip(tower.paligemma.model)(image)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        batch = {'observation.images.image': image,
                 'observation.images.image2': image.flip(-1),
                 'observation.state': torch.zeros(1, 8),
                 'observation.language.tokens': torch.ones(1, 8, dtype=torch.long),
                 'observation.language.attention_mask': torch.ones(1, 8, dtype=torch.bool)}
        from xhmodel_merak.xh_other_model.models.pi05._lerobot_compat import language_prefix_embeddings

        prefix, _, _ = policy.model.embed_prefix(
            [image], [torch.ones(1, dtype=torch.bool)],
            batch['observation.language.tokens'], batch['observation.language.attention_mask'],
        )
        language = language_prefix_embeddings(policy, batch['observation.language.tokens'])
        torch.testing.assert_close(language, prefix[:, -8:], rtol=0, atol=0)
        actions = policy.predict_action_chunk(batch, noise=torch.zeros(1, 50, 32))
    assert actions.shape == (1, 50, 7)
    assert torch.isfinite(actions).all()
    assert pi_gemma.PiGemmaModel.forward is native_forward
    assert pi_gemma.create_causal_mask is native_mask


@pytest.mark.parametrize('adaptive', [False, True])
def test_mask_adapter_preserves_native_forward(tmp_path, adaptive):
    from lerobot.policies import pi_gemma
    from transformers import GemmaConfig
    from xhmodel_merak.xh_other_model.models.pi05._lerobot_compat import adapt_pi05_reference

    config = GemmaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=16, vocab_size=64)
    config._attn_implementation = 'eager'
    config.adarms_cond_dim = 16 if adaptive else None
    model = adapt_pi05_reference(pi_gemma.PiGemmaModel(config).eval())
    with pytest.raises(ValueError, match='complete 4D'):
        model(inputs_embeds=torch.zeros(1, 4, 32), attention_mask=torch.ones(1, 4))
    mask = torch.zeros(1, 1, 4, 4)
    with torch.no_grad():
        output = model(inputs_embeds=torch.randn(1, 4, 32), attention_mask=mask,
                       adarms_cond=torch.randn(1,16) if adaptive else None)
    assert output.last_hidden_state.shape == (1,4,32)
    assert torch.isfinite(output.last_hidden_state).all()
