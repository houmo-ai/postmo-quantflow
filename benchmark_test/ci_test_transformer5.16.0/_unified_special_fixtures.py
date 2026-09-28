# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Native hybrid, assistant and multi-adapter fixture factories."""

import copy
import json
from pathlib import Path


def configure_kokoro_minimal(catalog, recipe, monkeypatch):
    """Reduce only CI profile geometry, keeping build/load validation active.

    Like the other native fixtures, this profile is local to the worker. The
    public four-route profile is still checked by test_kokoro_unified_api.
    """
    from dataclasses import replace
    from xhmodel_merak.model_integrations.kokoro import artifact

    profile = "bucketed-t32f160"
    monkeypatch.setattr(artifact, "_ROUTES", ((32, 160),))
    overrides = dict(recipe.config_replacements)
    overrides.update({"export.token_buckets": [32], "export.audio_seconds_buckets": [4]})
    monkeypatch.setitem(catalog._recipes, recipe.model_id, replace(
        recipe, default_profile=profile, profiles=(profile,),
        config_replacements=tuple(overrides.items()),
        fixed_artifact_name="hmquant_xh2_kokoro-82m-v1.1-zh_w16a16_t32_f160",
    ))


def write_ling(source):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
    from xhmodel_merak.xh_llm.models.ling_3_flash._compat import (
        load_ling_config, patch_ling_remote_code_compatibility,
    )

    bundled = Path(__file__).with_name("fixtures") / "ling3_flash"
    source.mkdir()
    for name in ("configuration_bailing_moe_v3.py", "modeling_bailing_moe_v3.py"):
        (source / name).write_bytes((bundled / name).read_bytes())
    config = json.loads((bundled / "config.json").read_text())
    config.update(num_hidden_layers=2, layer_group_size=2)
    (source / "config.json").write_text(json.dumps(config))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4891)
        native = AutoModelForCausalLM.from_config(load_ling_config(str(source)),
                                                 trust_remote_code=True, attn_implementation="eager")
        # The upstream KDA constructor leaves dt_bias as torch.empty because
        # deployed models load it from a checkpoint. This synthetic checkpoint
        # must supply a defined value instead of saving allocator contents.
        with torch.no_grad():
            for name, parameter in native.named_parameters():
                if name.endswith(".attention.dt_bias"):
                    parameter.zero_()
    patch_ling_remote_code_compatibility()
    native.half().save_pretrained(source)
    vocab = {str(i): i for i in range(128)}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="3"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="0", bos_token="1",
                                        eos_token="2", unk_token="3")
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    tokenizer.save_pretrained(source)
    return config


def write_gemma_assistant(source, target, *, ordered):
    import torch
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_mtp_model import _resolve_assistant_text_types

    family = target["model_type"]
    text = copy.deepcopy(target["text_config"])
    text.update(num_kv_shared_layers=2, hidden_size_per_layer_input=0, vocab_size_per_layer_input=0,
                enable_moe_block=False, num_experts=None, top_k_experts=None, moe_intermediate_size=None,
                use_double_wide_mlp=False)
    config_cls, model_cls, _ = _resolve_assistant_text_types({"model_type": family + "_assistant"})
    config = config_cls(**text)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4891)
        model = model_cls(config)
    generator = torch.Generator().manual_seed(4891)
    from safetensors.torch import save_file
    state = {"model." + k: v for k, v in model.state_dict().items()
             if not any(part in k for part in (".k_proj.", ".v_proj.", ".k_norm."))}
    state.update({"pre_projection.weight": torch.randn(64, 128, generator=generator) * .02,
                  "post_projection.weight": torch.randn(64, 64, generator=generator) * .02})
    if ordered:
        state.update({"masked_embedding.centroids.weight": torch.randn(8, 64, generator=generator) * .02,
                      "masked_embedding.token_ordering": torch.randperm(512, generator=generator)})
    source.mkdir()
    save_file({k: (v.to(torch.bfloat16) if v.is_floating_point() else v).contiguous()
               for k, v in state.items()}, source / "model.safetensors")
    raw = {"model_type": family + "_assistant", "text_config": config.to_dict(),
           "backbone_hidden_size": 64, "use_ordered_embeddings": ordered,
           "num_centroids": 8, "centroid_intermediate_top_k": 2}
    (source / "config.json").write_text(json.dumps(raw))
    return {"export.model.mtp_config." + k: v for k, v in {
        "assistant_num_hidden_layers": 2,
        "assistant_layer_pattern": ["sliding_attention", "full_attention"],
        "assistant_hidden_size": 64, "assistant_num_attention_heads": 4,
        "assistant_num_key_value_heads": config.per_layer_config[0].num_key_value_heads,
        "assistant_num_global_key_value_heads": config.to_dict().get("per_layer_config", {}).get("1", {}).get("num_key_value_heads", 0),
        "head_dim": 16,
    }.items()}


def write_macaron_loras(source):
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    config = json.loads((source / "config.json").read_text())["text_config"]
    with safe_open(source / "model.safetensors", framework="pt") as checkpoint:
        shapes = {key: checkpoint.get_slice(key).get_shape() for key in checkpoint.keys()}
    rank = 2
    for index in range(4):
        tensors = {}
        for key, shape in shapes.items():
            if len(shape) != 2 or not key.endswith(".weight") or ".layers." not in key:
                continue
            if not any(part in key for part in (".linear_attn.", ".self_attn.", ".mlp.shared_expert.")):
                continue
            module = key.removeprefix("model.language_model.").removesuffix(".weight")
            prefix = f"base_model.model.model.{module}"
            tensors[prefix + ".lora_A.weight"] = torch.full((rank, shape[1]), .01 * (index + 1))
            tensors[prefix + ".lora_B.weight"] = torch.full((shape[0], rank), .02 * (index + 1))
        for layer in range(config["num_hidden_layers"]):
            e, h, m = config["num_experts"], config["hidden_size"], config["moe_intermediate_size"]
            for suffix, a, b in (("base_layer", (e * rank, h), (2 * m, e * rank)),
                                  ("", (e * rank, m), (h, e * rank))):
                prefix = f"base_model.model.model.layers.{layer}.mlp.experts" + ("." + suffix if suffix else "")
                tensors[prefix + ".lora_A.weight"] = torch.full(a, .01 * (index + 1))
                tensors[prefix + ".lora_B.weight"] = torch.full(b, .02 * (index + 1))
        folder = source / "loras" / f"L{index}"
        folder.mkdir(parents=True)
        (folder / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA", "bias": "none", "r": rank, "lora_alpha": rank * 2,
            "target_parameters": ["experts.gate_up_proj", "experts.down_proj"],
        }))
        save_file(tensors, folder / "adapter_model.safetensors")
