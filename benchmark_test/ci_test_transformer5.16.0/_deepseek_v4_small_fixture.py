# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Real small mixed-attention target, packed RTN weights and native draft."""
import copy
import json
import re
from pathlib import Path


def write_deepseek_v4_checkpoint(root: Path):
    import defuser
    import torch
    from safetensors.torch import save_file
    from transformers import DeepseekV4Config, DeepseekV4ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers
    from gptqmodel import QuantizeConfig
    from gptqmodel.models.definitions.deepseek_v4 import DeepSeekV4QModel
    from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear
    from gptqmodel.utils.deepseek_v4_dspark import preserve_deepseek_v4_dspark

    root.mkdir(parents=True)
    config = DeepseekV4Config(hidden_size=128, moe_intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=128, q_lora_rank=64,
        o_groups=1, o_lora_rank=64, n_routed_experts=2, n_shared_experts=1,
        num_experts_per_tok=1, num_hash_layers=0, sliding_window=16, index_topk=4,
        index_n_heads=4, index_head_dim=128, vocab_size=256, qk_rope_head_dim=64,
        eos_token_id=2, pad_token_id=0, bos_token_id=1,
        max_position_embeddings=2048, dspark_target_layer_ids=[1, 2, 3],
        dspark_noise_token_id=255, dspark_markov_rank=8, dspark_block_size=5,
        layer_types=["sliding_attention", "sliding_attention", "compressed_sparse_attention",
                     "heavily_compressed_attention"], mlp_layer_types=["moe"] * 4)
    torch.manual_seed(523)
    model = DeepseekV4ForCausalLM(config).eval().bfloat16()
    DeepSeekV4QModel.before_model_load(DeepSeekV4QModel, load_quantized_model=False)
    defuser.convert_model(model, cleanup_original=True)
    state = model.state_dict()
    packed_count = 0
    for name, linear in model.named_modules():
        selected = re.search(r"\.self_attn\.(q_a_proj|q_b_proj|kv_proj|o_b_proj)$", name) or (
            ".mlp.experts." in name or ".mlp.shared_experts." in name)
        if not isinstance(linear, torch.nn.Linear) or not selected:
            continue
        bits = 4 if ".mlp.experts." in name else 8
        weight = linear.weight.detach().float()
        scales = weight.reshape(weight.shape[0], -1, 64).abs().amax(-1).clamp_min(1e-5) / ((1 << (bits - 1)) - 1)
        zeros = torch.full_like(scales, 1 << (bits - 1))
        groups = torch.arange(weight.shape[1], dtype=torch.int32) // 64
        packed = TorchQuantLinear(bits=bits, group_size=64, desc_act=False, sym=True,
            in_features=weight.shape[1], out_features=weight.shape[0], bias=False, pack_dtype=torch.int32)
        packed.pack(linear, scales, zeros, groups, workers=1)
        # This is an actual packed quantization, not a renamed floating tensor.
        assert packed.qweight.dtype == torch.int32 and packed.qweight.numel() < weight.numel()
        state.pop(name + ".weight")
        state.update({name + "." + key: value.contiguous() for key, value in packed.state_dict().items()})
        packed_count += 1
    assert packed_count > 20
    save_file({key: value.contiguous() for key, value in state.items()}, root / "model.safetensors")
    config.quantization_config = QuantizeConfig(bits=8, group_size=64, sym=True, desc_act=False,
        offload_to_disk=False,
        dynamic={r".*\.mlp\.experts\.\d+\.(gate_proj|up_proj|down_proj)$":
                 {"bits": 4, "group_size": 64, "sym": True}}).to_dict()
    config.save_pretrained(root)
    vocabulary = {f"t{i}": i for i in range(4, 256)}
    vocabulary.update({"[UNK]": 0, "<｜begin▁of▁sentence｜>": 1, "<｜end▁of▁sentence｜>": 2, "</think>": 3})
    backend = Tokenizer(models.WordLevel(vocabulary, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[UNK]",
        bos_token="<｜begin▁of▁sentence｜>", eos_token="<｜end▁of▁sentence｜>",
        additional_special_tokens=["</think>"])
    tokenizer.save_pretrained(root)

    native = root.parent / "native_draft"
    native.mkdir()
    draft_config = copy.deepcopy(config)
    del draft_config.quantization_config
    draft_config.num_hidden_layers = 3
    draft_config.layer_types = ["sliding_attention"] * 3
    draft_config.mlp_layer_types = ["moe"] * 3
    draft_config.save_pretrained(native)
    torch.manual_seed(524)
    draft = DeepseekV4ForCausalLM(draft_config).eval().bfloat16()
    native_weights = {}
    leaves = {
        "self_attn.q_a_proj.": "attn.wq_a.", "self_attn.q_b_proj.": "attn.wq_b.",
        "self_attn.q_a_norm.": "attn.q_norm.", "self_attn.q_b_norm.": "attn.q_b_norm.",
        "self_attn.kv_proj.": "attn.wkv.", "self_attn.kv_norm.": "attn.kv_norm.",
        "self_attn.o_a_proj.": "attn.wo_a.", "self_attn.o_b_proj.": "attn.wo_b.",
        "self_attn.sinks": "attn.attn_sink", "input_layernorm.": "attn_norm.",
        "post_attention_layernorm.": "ffn_norm.", "attn_hc.fn": "hc_attn_fn",
        "attn_hc.base": "hc_attn_base", "attn_hc.scale": "hc_attn_scale",
        "ffn_hc.fn": "hc_ffn_fn", "ffn_hc.base": "hc_ffn_base", "ffn_hc.scale": "hc_ffn_scale",
        "mlp.gate.weight": "ffn.gate.weight", "mlp.gate.e_score_correction_bias": "ffn.gate.bias",
        "mlp.shared_experts.gate_proj.": "ffn.shared_experts.w1.",
        "mlp.shared_experts.up_proj.": "ffn.shared_experts.w3.",
        "mlp.shared_experts.down_proj.": "ffn.shared_experts.w2.",
    }
    for i, layer in enumerate(draft.model.layers):
        for key, value in layer.state_dict().items():
            prefix = f"mtp.{i}."
            if key == "mlp.experts.gate_up_proj":
                for expert in range(2):
                    native_weights[prefix + f"ffn.experts.{expert}.w1.weight"] = value[expert, :64].contiguous()
                    native_weights[prefix + f"ffn.experts.{expert}.w3.weight"] = value[expert, 64:].contiguous()
            elif key == "mlp.experts.down_proj":
                for expert in range(2):
                    native_weights[prefix + f"ffn.experts.{expert}.w2.weight"] = value[expert].contiguous()
            else:
                original = next(name for name in leaves if key.startswith(name))
                native_weights[prefix + leaves[original] + key[len(original):]] = value.contiguous()
    for key, value in draft.model.hc_head.state_dict().items():
        native_weights["mtp.2." + {"hc_fn": "hc_head_fn", "hc_base": "hc_head_base", "hc_scale": "hc_head_scale"}[key]] = value
    native_weights["mtp.2.norm.weight"] = draft.model.norm.weight.detach()
    native_weights.update({"mtp.0.main_proj.weight": torch.randn(128, 384).bfloat16() * .02,
        "mtp.0.main_norm.weight": torch.ones(128),
        "mtp.2.markov_head.markov_w1.weight": torch.randn(256, 8).bfloat16() * .02,
        "mtp.2.markov_head.markov_w2.weight": torch.randn(256, 8).bfloat16() * .02,
        "mtp.2.confidence_head.proj.weight": torch.randn(1, 136).bfloat16() * .02})
    save_file({key: value.contiguous() for key, value in native_weights.items()}, native / "draft.safetensors")
    (native / "model.safetensors.index.json").write_text(json.dumps({"weight_map": dict.fromkeys(native_weights, "draft.safetensors")}))
    preserve_deepseek_v4_dspark(native, root)
    return config.to_dict()
