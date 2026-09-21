"""Synthetic canonical packed tensors for format consumption tests.

Payloads are constructed by RTN here; provider labels only describe the tested
schema. The producer's separate quant lane exercises actual AutoRound tuning,
save and reload. This fixture provides no algorithm or model-quality evidence.
Here small native GLM KDA+DSA+mHC+MoE+vision blocks retain G64-sized projections.
"""

import torch
from safetensors.torch import save_file
from test_glm53_multimodal import save_tiny_processor, tiny_multimodal_config
from transformers.models.glm5_next import Glm5NextForConditionalGeneration


def write_packed_glm53(source_dir, checkpoint_dir, *, autoround_format=False):
    from defuser.modeling.moe_experts_interface import prepare_model_for_moe_quantization
    from gptqmodel import QuantizeConfig
    from gptqmodel.recipes.glm5_3_flash import EXPERT_PATTERN

    torch.manual_seed(53464)
    config = tiny_multimodal_config()
    config.text_config.update(
        dict(
            hidden_size=128,
            intermediate_size=192,
            q_lora_rank=64,
            kv_lora_rank=64,
            qk_nope_head_dim=64,
            v_head_dim=64,
            qk_head_dim=64,
            linear_head_dim=64,
            moe_intermediate_size=64,
            index_head_dim=16,
        )
    )
    config.vision_config.update(
        dict(hidden_size=64, intermediate_size=128, depth=1, out_hidden_size=128, projection_intermediate_size=128)
    )
    source = Glm5NextForConditionalGeneration(config).eval().half()
    source.save_pretrained(source_dir)
    save_tiny_processor(source_dir)
    prepare_model_for_moe_quantization(source)
    weights = dict(source.state_dict())
    packed_names = []
    for name, module in source.named_modules():
        if not isinstance(module, torch.nn.Linear) or not name.startswith("model.language_model.layers."):
            continue
        if ".mlp." in name:
            selected = name.endswith(("gate_proj", "up_proj", "down_proj"))
        else:
            selected = name.endswith(
                (
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "g_a_proj",
                    "g_b_proj",
                    "o_proj",
                    "q_a_proj",
                    "q_b_proj",
                    "kv_a_proj_with_mqa",
                )
            )
        if not selected:
            continue
        bits = 4 if ".mlp.experts." in name else 8
        factor, zero = 32 // bits, 2 ** (bits - 1)
        original = weights.pop(name + ".weight").T.float()
        blocks = original.reshape(-1, 64, module.out_features)
        scales = (blocks.abs().amax(1) / (zero - 1)).clamp_min(1e-6).half()
        codes = (original / scales.repeat_interleave(64, 0)).round().clamp(-zero, zero - 1).long() + zero
        shifts = torch.arange(factor, dtype=torch.int64) * bits
        weights[name + ".qweight"] = (
            (codes.reshape(-1, factor, module.out_features) << shifts[None, :, None]).sum(1).int()
        )
        # GPTQ v1 stores zero-1; the native AutoRound GPTQModel export uses v2.
        zeros = torch.full(
            (module.in_features // 64, module.out_features // factor, factor),
            zero if autoround_format else zero - 1,
        )
        weights[name + ".qzeros"] = (zeros.long() << shifts).sum(-1).int()
        weights[name + ".scales"] = scales
        weights[name + ".g_idx"] = torch.arange(module.in_features, dtype=torch.int32) // 64
        module.weight.data.copy_(((codes - zero) * scales.repeat_interleave(64, 0)).T.half())
        packed_names.append(name)
    quant = QuantizeConfig(
        bits=8,
        group_size=64,
        sym=True,
        desc_act=False,
        dynamic={EXPERT_PATTERN: {"bits": 4, "group_size": 64, "sym": True}},
    )
    config.quantization_config = quant.to_dict()
    if autoround_format:
        config.quantization_config.update(provider="auto-round", checkpoint_format="gptq_v2")
    checkpoint_dir.mkdir()
    config.save_pretrained(checkpoint_dir)
    save_file(weights, checkpoint_dir / "model.safetensors")
    save_tiny_processor(checkpoint_dir)
    return source, packed_names
