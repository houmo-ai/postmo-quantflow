"""Registered Xing4.0 model and remote-code loading hooks."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any

import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig, PreTrainedModel
from transformers.initialization import no_init_weights

from xhmodel_merak.configuration_utils import BaseAttrDict
from xhmodel_merak.model_build.runtime_assets import quant_runtime_asset_paths

from ...builder import register_llm_model
from ...base_llm_model import XHLLMModelProcessor
from ...text_llm_model import TextLLMModel, TextLLMModelConfig
from .xing4_0_hmonnx_inference import XHXing4_0HMONNXModel


class XHXing4_0ModelConfig(TextLLMModelConfig):
    """Merak export configuration for Xing4.0."""


@register_llm_model("Xing4_0ForCausalLM")
class XHXing4_0Model(TextLLMModel):
    HF_MODEL_CLS = PreTrainedModel
    HF_AUTO_MODEL_CLS = AutoModelForCausalLM
    HMONNXINFERENCE_CLS = XHXing4_0HMONNXModel
    CONFIG_CLS = XHXing4_0ModelConfig
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.xing4_0.workflow:Xing4_0Workflow"

    def __init__(self, config: XHXing4_0ModelConfig):
        super().__init__(config)
        self.wrap_cfg["kv_cache"] = BaseAttrDict(self.kvcache_config.to_dict())

    def get_tokenizer(self, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        return super().get_tokenizer(**kwargs)

    def get_tf_processor(self):
        return XHLLMModelProcessor(self.get_tokenizer())

    @classmethod
    def _load_hf_model(cls, hf_model_dir: str, **kwargs):
        kwargs.setdefault("trust_remote_code", True)
        return super()._load_hf_model(hf_model_dir, **kwargs)

    @classmethod
    def get_empty_hf_model(cls, hf_model_dir: str | Path, **kwargs) -> Any:
        """Reconstruct the saved layout for generation without initializing weights."""
        kwargs.setdefault("trust_remote_code", True)
        kwargs.setdefault("dtype", cls.HF_MODEL_DTYPE)
        include_buffers = kwargs.pop("include_buffers", False)
        config = AutoConfig.from_pretrained(str(hf_model_dir), trust_remote_code=True)
        with no_init_weights(), init_empty_weights(include_buffers=include_buffers):
            hf_model = cls.HF_AUTO_MODEL_CLS.from_config(config, **kwargs)
            if hf_model.can_generate():
                try:
                    hf_model.generation_config = GenerationConfig.from_pretrained(str(hf_model_dir))
                except OSError:
                    pass
        if getattr(config, "mla_weight_layout", "hf_unabsorbed") != "hf_unabsorbed":
            from gptqmodel.models.definitions.xing4_0_mla import validate_xing_mla_model

            validate_xing_mla_model(hf_model)
        return hf_model

    @classmethod
    def _get_hf_model_for_compatible(cls, hf_model_dir=None):
        return cls.get_empty_hf_model(hf_model_dir, trust_remote_code=True)

    def init_wrap_model(self, hf_model: nn.Module) -> Any:
        from ._model import register_wrap_modules as xing4_register_wrap_modules

        xing4_register_wrap_modules(hf_model)
        return super().init_wrap_model(hf_model)

    def _wraped_post(self, hf_model: nn.Module):
        # MLA attention has no ``head_dim``; mirror Glm4MoeLite instead of
        # TextLLMModel's MHA kv-cache setup.
        language_model = self._get_language_model(self._wrap_model)
        self._mla_fused_projection_names = [
            name
            for name, _ in self._wrap_model.named_modules()
            if name.endswith((".self_attn.q_absorbed_proj", ".self_attn.o_proj"))
        ]
        self.embed_tokens = copy.deepcopy(language_model.get_input_embeddings())

        pad_token_id = getattr(language_model.config, "pad_token_id", None)
        if pad_token_id is None:
            eos_token_id = language_model.config.eos_token_id
            pad_token_id = eos_token_id[0] if isinstance(eos_token_id, (list, tuple)) else eos_token_id
        self.pad_token_id = pad_token_id

        if self.use_cache:
            num_decoder_layers = int(language_model.config.num_hidden_layers)
            max_layers = self.config.get_max_decode_layers()
            if max_layers > 0:
                assert max_layers <= num_decoder_layers
                num_decoder_layers = max_layers
            first_attn = language_model.layers[0].self_attn
            qk_rope_head_dim = int(getattr(language_model.config, "qk_rope_head_dim", first_attn.qk_rope_head_dim))
            kv_lora_rank = int(getattr(language_model.config, "kv_lora_rank", first_attn.kv_lora_rank))
            compress_kv_dim = kv_lora_rank + qk_rope_head_dim
            batch_size = self.config.batch_size
            self.kvcache_config.num_layers = num_decoder_layers
            key_cache_shape = [
                batch_size,
                1,
                int(self.config.context_max_length),
                compress_kv_dim,
            ]
            value_cache_shape = [
                batch_size,
                1,
                int(self.config.context_max_length),
                kv_lora_rank,
            ]
            self.kvcache_config.kv_cache_shape = [key_cache_shape, value_cache_shape]
            self.kvcache_config.batch_size = batch_size
            self.wrap_cfg["kv_cache"] = BaseAttrDict(self.kvcache_config.to_dict())

    def get_quant_cfg(self):
        quant_cfg = super().get_quant_cfg()
        if str(quant_cfg.get("quant_type", "")).startswith("w4"):
            # Preserve the producer's W8 absorbed MLA boundaries and payload;
            # routed experts keep the workflow's W4 scheme.
            nodes = quant_cfg.setdefault("nodes_cfg", {})
            for name in self._mla_fused_projection_names:
                alias = name.replace(".", "_")
                override = nodes.get(name, nodes.get(alias, {"quant_type": "w8a8h1_sefp"}))
                for key in (name, alias):
                    nodes.setdefault(key, override)
        return quant_cfg

    def _extra_export_metadata(self, output_dir: str, meta_info):
        """Package declared runtime assets without rewriting producer code."""
        hf_config_dir = Path(output_dir) / meta_info.hf_config
        source_dir = Path(self.config.hf_model)
        assets = quant_runtime_asset_paths(source_dir)
        if not any(source.suffix == ".py" for source in assets):
            raise FileNotFoundError(f"Xing4.0 export requires remote-code files in {source_dir}")
        if not (source_dir / "tokenizer.model").is_file():
            raise FileNotFoundError(f"Xing4.0 export requires tokenizer.model under {source_dir}")
        for source in assets:
            destination = hf_config_dir / source.relative_to(source_dir)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        return meta_info


__all__ = ["XHXing4_0Model", "XHXing4_0ModelConfig"]
