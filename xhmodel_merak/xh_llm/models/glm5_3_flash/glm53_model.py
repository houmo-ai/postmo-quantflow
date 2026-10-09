# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""GLM-5.3 multimodal model with separately streamed vision and text exports."""

import gc
import json
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForImageTextToText
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration

from ....utils import calculate_file_md5
from ...builder import register_llm_model
from ...text_llm_model import TextLLMModel
from ...types import LLMModelState, ModelSwitcher
from .contract import Glm53CacheSpec
from .hf_compatible import build_glm53_hf_compatible
from .modeling import StaticGlm53ForCausalLM
from .runtime import Glm53CacheMixin, Glm53DataPreprocess
from .xh_glm53_config import XHGlm53ModelConfig


@register_llm_model("Glm5NextForConditionalGeneration")
class XHGlm53Model(TextLLMModel):
    CONFIG_CLS = XHGlm53ModelConfig
    HF_MODEL_CLS = Glm5NextForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    HF_MODEL_DTYPE = torch.float16
    transformers_min_version = "5.16.1"
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.glm5_3_flash.workflow:Glm53Workflow"
    BUILD_HF_COMPATIBLE_FUNC = staticmethod(build_glm53_hf_compatible)

    def __init__(self, config):
        import xhquant.nn as xhnn

        if not hasattr(xhnn, "SparseAttentionSelectKV"):
            raise RuntimeError("GLM-5.3 export requires xhquant.nn.SparseAttentionSelectKV; update xhquanttool")
        super().__init__(config)
        hf_config = AutoConfig.from_pretrained(self.hf_model_dir)
        self.cache_spec = Glm53CacheSpec.from_config(hf_config, config.context_max_length, config.max_layers)
        self.kvcache_config.num_layers = len(self.cache_spec.layer_types)
        self._kvcache_mixin = Glm53CacheMixin(self.cache_spec, self.kvcache_config)
        self.pad_token_id = hf_config.get_text_config().pad_token_id
        self._decode_wrap_model = None

    def get_native_model(self):
        raise RuntimeError("GLM-5.3 loads checkpoint parts through export_hmonnx(); full-model loading is unsupported")

    def get_quant_cfg(self):
        config = super().get_quant_cfg()
        nodes = config.setdefault("nodes_cfg", {})
        for index, kind in enumerate(self.cache_spec.layer_types):
            if kind == "linear_attention":
                for name in ("forget_gate_f_a_proj", "forget_gate_f_b_proj", "b_proj"):
                    nodes[f"layers_{index}_attention_{name}"] = {"quant_type": "w16a16h0_sefp"}
            else:
                for name in ("k_projection", "v_projection"):
                    # MatMul treats both operands as inputs. A w8a16 quant_type
                    # sets the unused weight schema, so configure its RHS
                    # input explicitly for these learned constant factors.
                    nodes[f"layers_{index}_attention_{name}"] = {
                        "act_schema": {"fp_mode": "sefp", "man_bit": 16},
                        "act_schema_2": {"fp_mode": "sefp", "man_bit": 8, "hidden_bit": False, "nshare": 64},
                    }
        return config

    def _set_device(self, device):
        super()._set_device(device)
        decode = getattr(self, "_decode_wrap_model", None)
        if decode is not None:
            decode.to(device=device)

    def _set_dtype(self, dtype):
        super()._set_dtype(dtype)
        if dtype is not None:
            if self.embed_tokens is not None:
                self.embed_tokens.to(dtype=self.dtype)
            decode = getattr(self, "_decode_wrap_model", None)
            if decode is not None:
                decode.to(dtype=self.dtype)
            self._kvcache_mixin.to(dtype=self.dtype)

    def get_dummy_inputs(self):
        length = self.config.prefill_chunk_length if self.is_prefill() else 1
        return {
            "input_ids": torch.zeros(1, length, dtype=torch.long),
            "past_seq_length": 0
            if self.is_prefill()
            else min(self.config.prefill_chunk_length, self.config.context_max_length - 1),
        }

    def init_wrap_model(self, source):
        text = source.language_model
        common = dict(
            text_model=text, lm_head=source.lm_head, spec=self.cache_spec, chunk_size=self.config.linear_chunk_size
        )
        self._wrap_model = (
            StaticGlm53ForCausalLM(**common, input_length=self.config.prefill_chunk_length).to(self.dtype).eval()
        )
        self._decode_wrap_model = StaticGlm53ForCausalLM(**common, input_length=1).to(self.dtype).eval()
        self.embed_tokens = text.embed_tokens
        return self._wrap_model

    def _to_fronted(self, wrap_model):
        self.set_prefill()
        prefill = super()._to_fronted(wrap_model)
        self.set_decode()
        decode = super()._to_fronted(self._decode_wrap_model)
        self._decode_wrap_model = None
        self.set_prefill()
        result = ModelSwitcher({"prefill": prefill, "decode": decode})
        result.set_activate_model("prefill")
        return result

    def _to_quanted(self, frontend_model, state, **kwargs):
        kwargs.setdefault("infer_shape", False)
        result = {}
        for stage in ("prefill", "decode"):
            getattr(self, "set_" + stage)()
            result[stage] = super()._to_quanted(frontend_model[stage], state, **kwargs)
            result[stage].cpu()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.set_prefill()
        models = ModelSwitcher(result)
        models.set_activate_model("prefill")
        return models

    def _select_stage(self, stage):
        if self._state == LLMModelState.FRONTED:
            self._frontend_model.set_activate_model(stage)
        elif self._state in (LLMModelState.QUANTED_ALIGNED, LLMModelState.QUANTED_FAST, LLMModelState.QUANTED_DISABLE):
            self._quanted_model.set_activate_model(stage)
        self._data_processor = None

    def set_prefill(self):
        self._select_stage("prefill")
        super().set_prefill()

    def set_decode(self):
        self._select_stage("decode")
        super().set_decode()

    def _get_data_preprocessor(self):
        return Glm53DataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.wrap_cfg.input_sequence_length,
            cache_mixin=self._kvcache_mixin,
            pad_token_id=self.pad_token_id,
        )

    def get_export_cfg(self):
        return {"input_names": self.cache_spec.input_names(), "output_names": self.cache_spec.output_names()}

    def _extra_export_metadata(self, output_dir, meta_info):
        meta_info.glm53_cache_spec = asdict(self.cache_spec)
        meta_info.glm53_state_abi_version = 3
        meta_info.glm53_export_mode = "streamed_ffn"
        meta_info.glm53_dsa_selection = "vp_pool"
        meta_info.supported_modalities = ["text"]
        if getattr(self, "_vision_metadata", None) is not None:
            meta_info.glm53_vision = self._vision_metadata
            meta_info.supported_modalities = ["text", "image", "video"]
        return meta_info

    def export_hmonnx(self, output_dir):
        """Export only via bounded FFN loading and prefill/decode subgraph inlining."""
        import copy

        from .big_export import Glm53BigHFModel
        from .checkpoint import empty_glm53_source

        if self._state != LLMModelState.NONE:
            raise RuntimeError("Streamed GLM-5.3 export requires a fresh model instance")
        if self.config.export_vision:
            from .vision_export import export_glm53_vision

            self._vision_metadata = export_glm53_vision(
                self.hf_model_dir,
                Path(output_dir) / "visual",
                self.config.vision_max_patches,
                device="cuda" if torch.cuda.is_available() else "cpu",
                chip_arch=self.config.chip_arch,
            )
        source = empty_glm53_source(self.hf_model_dir, self.config.max_layers)
        template = copy.deepcopy(source)
        helper = Glm53BigHFModel(self.hf_model_dir, source, Glm53BigHFModel.PLACEHOLDER_TYPES)
        self.to_wrap(source)
        helper.register_layer_as_placeholder(self._wrap_model)
        helper.register_layer_as_placeholder(self._decode_wrap_model)
        self.to_fronted()
        info = self.get_export_info(output_dir)
        self.set_prefill()
        prefill_cfg = copy.deepcopy(self.get_wrap_cfg())
        helper.register_layer_as_place_holder(self._frontend_model.prefill)
        self.set_decode()
        decode_cfg = copy.deepcopy(self.get_wrap_cfg())
        helper.register_layer_as_place_holder(self._frontend_model.decode)
        directory = Path(info.exported_dir)
        if getattr(self, "_vision_metadata", None) is not None:
            directory.mkdir(parents=True, exist_ok=True)
            (Path(output_dir) / "visual").rename(directory / "visual")
        helper.export_prefill_decode_placeholder_layers(
            self._frontend_model.prefill,
            self._frontend_model.decode,
            template,
            self.config.chip_arch,
            prefill_cfg,
            decode_cfg,
            self.get_quant_cfg(),
            directory / "prefill" / "placeholders",
            directory / "decode" / "placeholders",
            placeholder_export_workers=self.config.placeholder_export_workers,
            empty_hf_model_factory=empty_glm53_source,
        )
        self.set_prefill()
        self.to_quanted_aligned()
        self._quanted_model.prefill.fixed()
        self._quanted_model.decode.fixed()
        self._export_hmonnx(info)
        for stage in ("prefill", "decode"):
            path = str(directory / getattr(info.meta, stage + "_hmonnx"))
            helper.replace_hmonnx_placeholders_with_subgraphs(path)
            setattr(info.meta, stage + "_hmonnx_md5", calculate_file_md5(path))
        Path(info.exported_dir, "golden_meta_info.json").write_text(json.dumps(info.meta.to_dict(), indent=2) + "\n")
        (directory / "merak_config.json").write_text(
            json.dumps(
                {
                    "architectures": ["MerakForCausalLM"],
                    "config_format": "merak_llm",
                    "load_format": "merak_llm",
                    "model_type": "merak_llm",
                    "xh_model": {"model_type": "hmonnx", "meta_info": "golden_meta_info.json"},
                    "enable_page_attention": False,
                },
                indent=2,
            )
            + "\n"
        )
        return info.meta

    def forward(self, *args, **kwargs):
        if self._state == LLMModelState.WRAP and self.is_decode():
            return self._kvcache_mixin.commit(self._decode_wrap_model(*args, **kwargs))
        return self._kvcache_mixin.commit(super().forward(*args, **kwargs))
