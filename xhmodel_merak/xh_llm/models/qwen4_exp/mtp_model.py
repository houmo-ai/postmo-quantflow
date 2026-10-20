"""Qwen4-Exp MTP assistant with its own HC and sparse-attention state.

Unlike Qwen3.5, Qwen4's assistant consumes the target's *unmixed*
hyper-connection streams. Its checkpoint has separate embedding/hidden
projections and a grouped HC normalization; it does not use a concatenating
``fc`` or the target model's PLE state.

The residual_linear_shared and pre-mixer HC contract is independently
implemented from the checkpoint layout and the vLLM semantic reference:
https://github.com/vllm-project/vllm/blob/main/vllm/models/qwen4_exp/nvidia/mtp.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch import nn
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpPreTrainedModel

from .modeling import Qwen4ExpTextExportWrapper


def mtp_text_config(text_config):
    """Construct the assistant configuration without changing the target."""
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    data = text_config.to_dict() if hasattr(text_config, "to_dict") else copy.deepcopy(text_config)
    mtp = data.get("mtp", {})
    layers = int(mtp.get("num_hidden_layers", data.get("mtp_num_hidden_layers", 0)))
    types = mtp.get("layer_types")
    if layers != 1 or types != ["full_attention"]:
        raise ValueError("Qwen4 MTP requires the checkpoint's single full-attention assistant")
    if data.get("mtp_use_dedicated_embeddings", False):
        raise ValueError("Dedicated Qwen4 MTP embeddings are not supported")
    if mtp.get("mtp_use_hidden_state_from_layer") is not None:
        raise ValueError("Qwen4 MTP requires final target hyper-connection states")
    data.update(num_hidden_layers=layers, layer_types=list(types), ple_layer_ids=[])
    if "rope_theta" in mtp:
        data["rope_parameters"] = dict(data.get("rope_parameters", {}), rope_theta=mtp["rope_theta"])
    return Qwen4ExpTextConfig(**data)


class Qwen4ExpMTPModel(Qwen4ExpPreTrainedModel):
    """Native assistant core retaining grouped HC, MoE and QSA semantics."""

    def __init__(self, text_config):
        super().__init__(mtp_text_config(text_config))
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (
            Qwen4ExpTextDecoderLayer,
            Qwen4ExpTextGatedResidual,
            Qwen4ExpTextRMSNorm,
            Qwen4ExpTextRotaryEmbedding,
        )

        c = self.config
        self.pre_fc_norm_embedding = Qwen4ExpTextRMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        self.pre_fc_norm_hidden = Qwen4ExpTextRMSNorm(
            c.hidden_size * c.hc_count, group_size=c.hidden_size, eps=c.rms_norm_eps
        )
        self.fc_embedding = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
        self.fc_hidden = nn.Linear(c.hidden_size, c.hidden_size, bias=False)
        self.layers = nn.ModuleList([Qwen4ExpTextDecoderLayer(c, 0)])
        self.rotary_emb = Qwen4ExpTextRotaryEmbedding(c)
        self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(c, use_combine=False)
        self.post_init()

    def fuse(self, next_token_embedding, target_hidden):
        c = self.config
        embedding = self.fc_embedding(self.pre_fc_norm_embedding(next_token_embedding))
        hidden = self.pre_fc_norm_hidden(target_hidden).unflatten(-1, (c.hc_count, c.hidden_size))
        return (self.fc_hidden(hidden) + embedding.unsqueeze(-2)).flatten(-2)

    def forward(self, inputs_embeds, attention_mask, position_ids, past_key_values, **kwargs):
        # The outer wrapper supplies current and C4 group-start T/H/W positions.
        c = self.config
        embedding = inputs_embeds[..., : c.hidden_size]
        target_hidden = inputs_embeds[..., c.hidden_size :]
        hidden = self.fuse(embedding, target_hidden)
        rotary = self.rotary_emb(embedding, *position_ids)
        for layer in self.layers:
            hidden = layer(
                hidden,
                position_embeddings=rotary,
                attention_mask=attention_mask["full_attention"],
                conv_mask=None,
                past_key_values=past_key_values,
            )
        mixed = self.hyper_connection_mixer(hidden)
        from transformers.modeling_outputs import BaseModelOutputWithPast

        return BaseModelOutputWithPast(
            last_hidden_state=mixed, hidden_states=(hidden,), past_key_values=past_key_values
        )

    @classmethod
    def from_pretrained(cls, model_dir, *, dtype=torch.float32):
        """Read only MTP tensors; never load the 48-layer target a second time."""
        from safetensors import safe_open

        source = Path(model_dir)
        config = json.loads((source / "config.json").read_text())
        with torch.device("meta"):
            model = cls(config.get("text_config", config))
        index = source / "model.safetensors.index.json"
        if index.exists():
            mapping = json.loads(index.read_text())["weight_map"]
            paths = sorted({source / shard for key, shard in mapping.items() if key.startswith("mtp.")})
        else:
            mapping = None
            paths = sorted(source.glob("*.safetensors"))
        tensors = {}
        for path in paths:
            with safe_open(path, framework="pt") as reader:
                for key in reader.keys():
                    if not key.startswith("mtp.") or (mapping is not None and mapping.get(key) != path.name):
                        continue
                    tensors[key.removeprefix("mtp.")] = reader.get_tensor(key).to(dtype)
        if not tensors:
            raise ValueError(f"No Qwen4 MTP weights in {source}")
        model.load_state_dict(tensors, strict=True, assign=True)
        # RoPE buffers are nonpersistent and thus absent from state_dict.
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRotaryEmbedding

        model.rotary_emb = Qwen4ExpTextRotaryEmbedding(model.config)
        return model.eval()


class Qwen4ExpMTPExportWrapper(nn.Module):
    """Draft ABI: embedding, target HC, lengths, and its own cache.

    Outputs are logits, assistant HC, then cache tensors in contract order.
    The assistant configuration never creates a PLE layer.
    """

    def __init__(self, assistant, lm_head, *, input_sequence_length=1, cache_length=262144):
        super().__init__()
        self.core = Qwen4ExpTextExportWrapper(
            assistant,
            assistant.config,
            input_sequence_length=input_sequence_length,
            cache_length=cache_length,
            return_native_hidden_states=True,
        )
        self.lm_head = lm_head
        self.cache_contract = self.core.cache_contract

    def forward(
        self, next_token_embedding, target_hidden, past_seq_length, current_input_length,
        time_position_ids, height_position_ids, width_position_ids,
        pool_time_position_ids, pool_height_position_ids, pool_width_position_ids, *cache,
    ):
        outputs = self.core(
            torch.cat((next_token_embedding, target_hidden), dim=-1),
            past_seq_length,
            current_input_length,
            torch.zeros(next_token_embedding.shape[:2], dtype=torch.int32, device=next_token_embedding.device),
            time_position_ids, height_position_ids, width_position_ids,
            pool_time_position_ids, pool_height_position_ids, pool_width_position_ids,
            *cache,
        )
        return self.lm_head(outputs[0]), *outputs[1:]
