"""DSpark draft model that adds Markov and confidence heads around Qwen DFlash."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from xhquant import nn as xhnn

from ..qwen3_5._dflash_model_impl import DFlashModel
from ..qwen3_5.qwen3_5_dflash_model import XHQwen3_5DFlashDraftModel

_PARENT_DFLASH_FROM_PRETRAINED = DFlashModel.from_pretrained


from .draft_checkpoint import read_draft_checkpoint
from .dspark_heads import DSparkConfidenceHead, DSparkMarkovHead, greedy_dspark_proposals


class MiniCPMDSparkModel(DFlashModel):
    def __init__(self, *args, dspark_config=None, num_output_draft_tokens=None, compute_confidence=False, confidence_logits=False, **kwargs):
        self.dspark_config = dict(dspark_config or {})
        self.num_output_draft_tokens = num_output_draft_tokens
        self.compute_confidence = bool(compute_confidence)
        self._confidence_enabled = bool(compute_confidence or self.dspark_config.get("enable_confidence_head"))
        self.confidence_logits = bool(confidence_logits)
        self.is_dspark = kwargs.get("architecture") == "Qwen3DSparkModel" or bool(self.dspark_config)
        super().__init__(*args, **kwargs)
        rank = int(self.dspark_config.get("markov_rank", 1))
        self.markov = DSparkMarkovHead(self.vocab_size, rank)
        hidden = self.hidden_size + (rank if self.dspark_config.get("confidence_head_with_markov") else 0)
        self.confidence = DSparkConfidenceHead(hidden) if self._confidence_enabled else None
        self.proposal_argmax = xhnn.ArgMax(axis=-1, keepdim=False)

    def forward_decode(self, noise_embedding, past_seq_length, current_input_length, attn_mask, anchor_token_id, *cache_tensors, cache_position=None):
        past_key_caches, past_value_caches = self._split_cache_tensors(cache_tensors)
        hidden_states = noise_embedding
        if self.input_embedding_scale != 1.0:
            hidden_states = hidden_states * self.input_embedding_scale
        for idx, layer in enumerate(self.layers):
            hidden_states = layer.forward_decode(
                hidden_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                target_key_cache=past_key_caches[idx],
                target_value_cache=past_value_caches[idx],
                attn_mask=attn_mask,
                cache_position=cache_position,
            )
        normalized_hidden = self.norm(hidden_states)
        width = self.block_size if self.num_output_draft_tokens is None else int(self.num_output_draft_tokens)
        logits = self.lm_head(normalized_hidden[:, :width])
        if not self.is_dspark:
            return logits
        hidden = normalized_hidden
        if anchor_token_id is None:
            anchor_token_id = torch.zeros(noise_embedding.shape[0], dtype=torch.long, device=noise_embedding.device)
        tokens, probabilities = greedy_dspark_proposals(
            logits,
            hidden,
            anchor_token_id,
            self.markov,
            self.confidence,
            bool(self.dspark_config.get("confidence_head_with_markov")),
            self.num_output_draft_tokens if self.num_output_draft_tokens is not None else self.block_size,
            self.proposal_argmax,
            raw_scores=self.confidence_logits,
        )
        if not self.compute_confidence:
            return tokens
        return tokens, probabilities

    @staticmethod
    def from_pretrained(dflash_model_dir, target_model_dir, **kwargs):
        raw = read_draft_checkpoint(dflash_model_dir)
        num_output_draft_tokens = kwargs.pop("num_output_draft_tokens", None)
        confidence_logits = kwargs.pop("confidence_logits", None)
        if raw.get("architectures") != ["Qwen3DSparkModel"]:
            return _PARENT_DFLASH_FROM_PRETRAINED(dflash_model_dir, target_model_dir, **kwargs)
        source = Path(dflash_model_dir)
        with tempfile.TemporaryDirectory() as temp:
            rewritten = Path(temp)
            with safe_open(str(source / "model.safetensors"), framework="pt") as handle:
                filtered = {
                    key: handle.get_tensor(key)
                    for key in handle.keys()
                    if not key.startswith(("markov.", "confidence."))
                }
            save_file(filtered, str(rewritten / "model.safetensors"))
            config = json.loads((source / "config.json").read_text(encoding="utf-8"))
            config["dflash_config"] = raw["dflash_config"]
            config["architectures"] = ["DFlashDraftModel"]
            (rewritten / "config.json").write_text(json.dumps(config), encoding="utf-8")
            model = _PARENT_DFLASH_FROM_PRETRAINED(temp, target_model_dir, **kwargs)
        model.__class__ = MiniCPMDSparkModel
        model.dspark_config = raw
        model.is_dspark = True
        model.markov = DSparkMarkovHead(model.vocab_size, int(raw["markov_rank"]))
        hidden = model.hidden_size + (int(raw["markov_rank"]) if raw.get("confidence_head_with_markov") else 0)
        model.confidence = DSparkConfidenceHead(hidden) if raw.get("enable_confidence_head") else None
        with safe_open(str(source / "model.safetensors"), framework="pt") as handle:
            state = {key: handle.get_tensor(key) for key in handle.keys() if key.startswith(("markov.", "confidence."))}
        model.load_state_dict(state, strict=False)
        model.compute_confidence = bool(raw.get("enable_confidence_head"))
        model.confidence_logits = bool(model.compute_confidence if confidence_logits is None else confidence_logits)
        model.num_output_draft_tokens = num_output_draft_tokens
        model.proposal_argmax = xhnn.ArgMax(axis=-1, keepdim=False)
        if model.is_dspark:
            decode = model.forward_decode

            def forward_decode(noise_embedding, past_seq_length, current_input_length, attn_mask, anchor_token_id, *caches, cache_position=None):
                return decode(
                    noise_embedding, past_seq_length, current_input_length, attn_mask, anchor_token_id, *caches, cache_position=cache_position,
                )

            model.forward_decode = forward_decode
        return model


class MiniCPMDSparkDraftModel(XHQwen3_5DFlashDraftModel):
    """DSpark loader and graph ABI, inherited from the unchanged Qwen exporter."""

    def __init__(self, config):
        super().__init__(config)
        if self.config.mode == "decode" and self._is_dspark():
            self.config.input_sequence_length = int(self.config.block_size)

    def _is_dspark(self):
        return getattr(self.config, "architecture", None) == "Qwen3DSparkModel"

    def _to_wrap(self, hf_model):
        if not self._is_dspark():
            return super()._to_wrap(hf_model)
        from .dspark_export import _build_dspark_export_adapter

        core = MiniCPMDSparkModel.from_pretrained(
            self.config.dflash_model_dir,
            self.config.target_model_dir,
            mode=self.config.mode,
            dtype=self.dtype,
            input_sequence_length=self.config.input_sequence_length,
            max_pe_length=self.config.max_pe_length,
            max_sequence_length=self.config.max_sequence_length,
            flash_attention=self.config.flash_attention,
            activation_residual_scale=self.config.activation_residual_scale,
            num_output_draft_tokens=self.config.num_output_draft_tokens,
            confidence_logits=True,
        )
        self._wrap_model = _build_dspark_export_adapter(
            core, mode=self.config.mode, num_hidden_layers=self.config.num_hidden_layers,
            use_flash_attention=self._uses_flash_attention(), is_dspark=True,
        )

    def get_dummy_inputs(self):
        values = super().get_dummy_inputs()
        if self.config.mode != "decode" or not self._is_dspark():
            return values
        return {
            "noise_embedding": values["noise_embedding"],
            "past_seq_length": values["past_seq_length"],
            "current_input_length": values["current_input_length"],
            "anchor_token_id": torch.zeros(self.config.batch_size, dtype=torch.long),
            "attn_mask": torch.zeros(
                self.config.batch_size, self.config.max_sequence_length,
                dtype=values["noise_embedding"].dtype,
            ),
            **{name: value for name, value in values.items()
               if name.startswith(("past_key_", "past_value_"))},
        }

    def get_export_cfg(self):
        cfg = super().get_export_cfg()
        if self.config.mode != "decode" or not self._is_dspark():
            return cfg
        names = ["noise_embedding", "past_seq_length", "current_input_length", "anchor_token_id", "attn_mask"]
        names.extend(name for name in cfg["input_names"] if name.startswith(("past_key_", "past_value_")))
        produced = ["draft_token_ids"]
        confidence = read_draft_checkpoint(self.config.hf_model).get("enable_confidence_head")
        if confidence:
            produced.append("confidence_logits")
        return {"input_names": names, "output_names": produced}


class MiniCPMDSparkDraftExport:
    """Select a derived draft class for the parent's local import during export."""

    def __init__(self, num_output_draft_tokens):
        self.num_output_draft_tokens = num_output_draft_tokens

    def __enter__(self):
        from ..qwen3_5 import qwen3_5_dflash_model

        self.module = qwen3_5_dflash_model
        self.original_class = self.module.XHQwen3_5DFlashDraftModel
        if self.num_output_draft_tokens is not None:
            self.module.XHQwen3_5DFlashDraftModel = MiniCPMDSparkDraftModel
        return self

    def __exit__(self, *_args):
        self.module.XHQwen3_5DFlashDraftModel = self.original_class
