# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Native LeRobot Pi0.5 with two Gemma layers and one SigLIP layer.

The factory overrides dimensions only. Native attention, adaptive RMSNorm,
flow matching, state normalization, export and runtime implementations execute.
"""

import json
from collections import ChainMap


def write_pi05(source, tokenizer_dir, monkeypatch):
    import sentencepiece as spm
    import torch
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pi05 import PI05Config, PI05Policy
    from lerobot.policies.pi05 import modeling_pi05
    from safetensors.torch import save_file
    from transformers import LlamaTokenizer

    mapping = modeling_pi05.CONFIG_MAPPING

    def small_paligemma_config():
        config = mapping["paligemma"]()
        config.vision_config.hidden_size = 64
        config.vision_config.num_hidden_layers = 1
        config.vision_config.num_attention_heads = 4
        return config

    monkeypatch.setattr(modeling_pi05, "CONFIG_MAPPING", ChainMap({"paligemma": small_paligemma_config}, mapping))
    monkeypatch.setattr(modeling_pi05, "get_gemma_config", lambda variant: modeling_pi05.GemmaConfig(
        width=64, depth=2, mlp_dim=128, num_heads=4, num_kv_heads=1, head_dim=16,
    ))
    # LeRobot assigns the released projector width after constructing its
    # config. Align that dimension immediately before the native constructor.
    native_paligemma = modeling_pi05.PaliGemmaForConditionalGenerationWithPiGemma

    def small_paligemma(*, config):
        config.vision_config.projection_dim = config.text_config.hidden_size
        return native_paligemma(config=config)

    monkeypatch.setattr(modeling_pi05, "PaliGemmaForConditionalGenerationWithPiGemma", small_paligemma)
    config = PI05Config(
        device="cpu", dtype="float32", num_inference_steps=2,
        input_features={
            "observation.images.image": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
            "observation.images.image2": PolicyFeature(FeatureType.VISUAL, (3, 224, 224)),
            "observation.state": PolicyFeature(FeatureType.STATE, (8,)),
        },
        output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4891)
        policy = PI05Policy(config).eval()
    policy.save_pretrained(source)
    # The compact runtime consumes quantile tensors directly. Keep asymmetric
    # non-identity bounds so normalization and action unnormalization execute.
    for name in ("configuration.json", "policy_preprocessor.json", "policy_postprocessor.json"):
        (source / name).write_text(json.dumps({}))
    save_file({"observation.state.q01": torch.full((8,), -2.),
               "observation.state.q99": torch.full((8,), 3.)},
              source / "policy_preprocessor_step_2_normalizer_processor.safetensors")
    save_file({"action.q01": torch.full((7,), -1.), "action.q99": torch.full((7,), 2.)},
              source / "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    tokenizer_dir.mkdir()
    corpus = tokenizer_dir / "corpus.txt"
    corpus.write_text("pick up the object\nhello world\n" + " ".join(str(i) for i in range(256)))
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=str(tokenizer_dir / "tokenizer"),
                                  vocab_size=64, hard_vocab_limit=False, num_threads=1, shuffle_input_sentence=False)
    tokenizer = LlamaTokenizer(
        vocab_file=str(tokenizer_dir / "tokenizer.model"), pad_token="<unk>"
    )
    tokenizer.save_pretrained(tokenizer_dir)
    # The release contract retains this explicit snapshot file, while newer
    # Transformers also stores these values inside tokenizer_config.json.
    (tokenizer_dir / "special_tokens_map.json").write_text(json.dumps(tokenizer.special_tokens_map))
    corpus.unlink()
    (tokenizer_dir / "tokenizer.vocab").unlink()
