"""Qwen4-Exp model entry and static Merak lifecycle."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForImageTextToText
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration

from ...builder import register_llm_model
from ...types import ExportData, LLMModelState, ModelSwitcher, VLLMModelMeta
from ...vision_llm_model import VisionLLMModel
from .configuration_qwen4_exp import XHQwen4ExpModelConfig
from .qwen4_exp_hmonnx_inference import XHQwen4ExpHMONNXModel
from .static_runtime import Qwen4ExpCacheMixin, Qwen4ExpDataPreprocess


@register_llm_model("Qwen4ExpForConditionalGeneration")
@register_llm_model("qwen4_exp", master=False)
class XHQwen4ExpModel(VisionLLMModel):
    """Qwen4-Exp static model entry.

    The model owns the wrapper, cache contract, frontend stages and public
    export entry. The workflow remains responsible for quantization recipes.
    """

    CONFIG_CLS = XHQwen4ExpModelConfig
    META_CLS = VLLMModelMeta
    HMONNXINFERENCE_CLS = XHQwen4ExpHMONNXModel
    HF_MODEL_CLS = Qwen4ExpForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    HF_MODEL_DTYPE = torch.float16
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.qwen4_exp.workflow:Qwen4ExpWorkflow"
    transformers_min_version = "5.16.0"

    def __init__(self, config):
        super().__init__(config)
        from .cache_contract import Qwen4ExpCacheContract
        if not config.hf_model:
            raise ValueError("Qwen4-Exp lifecycle requires config.hf_model")
        hf_config = AutoConfig.from_pretrained(config.hf_model, trust_remote_code=False)
        text_config = getattr(hf_config, "text_config", hf_config)
        self._text_config = text_config
        visual_config = getattr(config, "visual_config", None)
        if visual_config is not None and getattr(visual_config, "enable", False):
            from .vision import XHQwen4ExpVisionModel

            self.visual = XHQwen4ExpVisionModel(visual_config)
            self.visual.config.model_name = f"{config.model_name}_visual"
        self.pad_token_id = getattr(text_config, "pad_token_id", None)
        if self.pad_token_id is None:
            self.pad_token_id = getattr(text_config, "eos_token_id", 0)
        if isinstance(self.pad_token_id, (list, tuple)):
            self.pad_token_id = self.pad_token_id[0]
        ple_eos = getattr(text_config, "eos_token_id", 0)
        if isinstance(ple_eos, (list, tuple)):
            ple_eos = ple_eos[0]
        self.qwen4_cache_contract = Qwen4ExpCacheContract.from_text_config(text_config)
        self._qwen4_cache = Qwen4ExpCacheMixin(
            self.qwen4_cache_contract,
            batch_size=int(config.batch_size),
            context_length=int(config.context_max_length),
            dtype=self.dtype,
            device="cpu",
            pad_token_id=int(self.pad_token_id or 0),
            ple_eos_token_id=int(ple_eos or 0),
            use_cache=bool(config.use_cache),
        )
        self._kvcache_config = self._qwen4_cache.kvcache_config
        self._kvcache_mixin = self._qwen4_cache
        self._decode_wrap_model = None
        self._llm_prefill = True

    def get_native_model(self):
        from .checkpoint import load_qwen4_exp_model_for_export

        return load_qwen4_exp_model_for_export(
            self.hf_model_dir,
            device="cpu",
            device_map=getattr(self.config, "device_map", None),
            dtype=self.dtype,
        )

    def _set_device(self, device):
        super()._set_device(device)
        for stages in (self._frontend_model, self._quanted_model):
            if isinstance(stages, ModelSwitcher):
                stages.to(device=device)
        decode = getattr(self, "_decode_wrap_model", None)
        if decode is not None:
            decode.to(device=device)

    def _set_dtype(self, dtype):
        super()._set_dtype(dtype)
        if dtype is not None:
            if self.embed_tokens is not None:
                self.embed_tokens.to(dtype=self.dtype)
            for stages in (self._frontend_model, self._quanted_model):
                if isinstance(stages, ModelSwitcher):
                    stages.to(dtype=self.dtype)
        decode = getattr(self, "_decode_wrap_model", None)
        if decode is not None and dtype is not None:
            decode.to(dtype=dtype)
        cache = getattr(self, "_qwen4_cache", None)
        if cache is not None and dtype is not None:
            cache.to(dtype=dtype)

    def get_dummy_inputs(self):
        length = int(self.config.prefill_chunk_length) if self.is_prefill() else 1
        return {"input_ids": torch.zeros((1, length), dtype=torch.long), "past_seq_length": 0}

    def init_wrap_model(self, source):
        from ...big_hf_model_helper import _copy_module_shared_params
        from .checkpoint import _adapt_gptq_linears
        from .modeling import (
            Qwen4ExpExportableGatedDeltaNet,
            adapt_qwen4_exp_rms_norms,
            build_qwen4_exp_text_export_wrapper,
            extract_qwen4_exp_text_config,
            extract_qwen4_exp_text_model,
        )
        _adapt_gptq_linears(source)

        # Qwen4 wrapper construction replaces native submodules in place.  Keep
        # prefill/decode structurally independent while sharing parameters and
        # buffers, matching the public Qwen3.5/GLM5.3 lifecycle without making
        # a second full-weight copy.
        decode_source = _copy_module_shared_params(source)
        text_model = extract_qwen4_exp_text_model(source)
        self.embed_tokens = text_model.embed_tokens
        self._wrap_model = build_qwen4_exp_text_export_wrapper(
            source,
            text_config=extract_qwen4_exp_text_config(source),
            batch_size=int(self.config.batch_size),
            input_sequence_length=int(self.config.prefill_chunk_length),
            cache_length=int(self.config.context_max_length),
        ).to(dtype=self.dtype).eval()
        self._decode_wrap_model = build_qwen4_exp_text_export_wrapper(
            decode_source,
            text_config=extract_qwen4_exp_text_config(decode_source),
            batch_size=int(self.config.batch_size),
            input_sequence_length=1,
            cache_length=int(self.config.context_max_length),
        ).to(dtype=self.dtype).eval()
        self._inplace_recurrent_names = {}
        for stage, wrapper in (("prefill", self._wrap_model), ("decode", self._decode_wrap_model)):
            adapt_qwen4_exp_rms_norms(wrapper)
            if getattr(self.config, "use_fused_gdr", False):
                for module in tuple(wrapper.modules()):
                    if isinstance(module, Qwen4ExpExportableGatedDeltaNet):
                        module.enable_fused_gdr()
            self._inplace_recurrent_names[stage] = {
                spec.name for spec in self.qwen4_cache_contract.tensors
                if spec.state_kind == "delta_recurrent"
                and wrapper._linear_attention_uses_gdr_cache_state(spec.layer_idx)
            }
        self._qwen4_cache.inplace_state_names = set().union(*self._inplace_recurrent_names.values())
        self._qsa_topk = None
        self._qsa_host_control_names = ()
        if any(getattr(module, "packed_qsa", None) is not None for module in self._wrap_model.modules()):
            from .qsa_host_controls import QSA_HOST_CONTROL_NAMES

            self._qsa_host_control_names = tuple(QSA_HOST_CONTROL_NAMES)
            self._qsa_topk = next(m.packed_qsa.candidate_topk for m in self._wrap_model.modules()
                                  if getattr(m, "packed_qsa", None) is not None)
        return self._wrap_model

    def _to_fronted(self, wrap_model):
        from .xhquant_extensions import qwen4_exp_xhquant_extension_scope

        with qwen4_exp_xhquant_extension_scope():
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
        from .xhquant_extensions import qwen4_exp_xhquant_extension_scope

        kwargs.setdefault("infer_shape", False)
        result = {}
        with qwen4_exp_xhquant_extension_scope():
            for stage in ("prefill", "decode"):
                getattr(self, "set_" + stage)()
                result[stage] = super()._to_quanted(frontend_model[stage], state, **kwargs)
                result[stage].cpu()
        self.set_prefill()
        models = ModelSwitcher(result)
        models.set_activate_model("prefill")
        return models

    def _select_stage(self, stage: str) -> None:
        if self._state == LLMModelState.FRONTED and self._frontend_model is not None:
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
        return Qwen4ExpDataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=int(self.wrap_cfg.input_sequence_length),
            cache_mixin=self._qwen4_cache,
            pad_token_id=self._qwen4_cache.pad_token_id,
            qsa_topk=self._qsa_topk,
        )

    def _get_wrapper_cfg(self):
        names = self.qwen4_cache_contract.expected_keys()
        inplace = self._inplace_recurrent_names["prefill" if self.is_prefill() else "decode"] | {
            spec.name for spec in self.qwen4_cache_contract.tensors
            if spec.state_kind in {"kv_key", "kv_value", "qsa_indexer_key"}
        }
        host_controls = tuple(getattr(self, "_qsa_host_control_names", ()))
        return {
            "input_names": [
                "inputs_embeds", "past_seq_length", "current_input_length", "ple_input_ids",
                "time_position_ids", "height_position_ids", "width_position_ids",
                "pool_time_position_ids", "pool_height_position_ids", "pool_width_position_ids",
                *names,
                *host_controls,
            ],
            "output_names": ["logits", *(f"present_{name}" for name in names if name not in inplace)],
        }

    def get_export_cfg(self):
        from .cache_contract import qwen4_exp_cache_input_specs

        cfg = self._get_wrapper_cfg()
        name_map = {spec.name: name for name, spec in qwen4_exp_cache_input_specs(self.qwen4_cache_contract).items()}
        cfg["input_names"] = [name_map.get(name, name) for name in cfg["input_names"]]
        cfg["output_names"] = [
            "present_" + name_map.get(name[8:], name[8:]) if name.startswith("present_") else name
            for name in cfg["output_names"]
        ]
        return cfg

    @staticmethod
    def xh1_hmonnx_compatible(input_names):
        return list(input_names)

    def get_export_info(self, output_dir):
        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        exported = ExportData()
        exported.exported_dir = str(root)
        exported.model_name = self.config.model_name
        exported.meta = self.META_CLS()
        return exported

    def forward(self, *args, **kwargs):
        if self._state == LLMModelState.WRAP and self.is_decode():
            outputs = self._decode_wrap_model(*args, **kwargs)
        else:
            outputs = super().forward(*args, **kwargs)
        if isinstance(outputs, (tuple, list)):
            return self._qwen4_cache.commit(outputs, self._get_wrapper_cfg()["output_names"])
        return outputs

    def export_hmonnx(self, output_dir: str, *, phase="both", resume=False, low_memory=None):
        from .package_export import _export_qwen4_exp_package

        self.work_dir = str(output_dir)
        model_config = self.config.to_dict()
        model_config["quant_scheme"] = self.config._export_quant_config
        manifest = _export_qwen4_exp_package(
            model_dir=self.hf_model_dir, output_dir=output_dir, model_config=model_config,
            device=self.config.get("quant_device") or ("cuda" if torch.cuda.is_available() else "cpu"),
            device_map=self.config.get("device_map"), model_name=self.config.model_name,
            mtp_source_dir=self.config.get("mtp_source_dir"), phase=phase, resume=resume, low_memory=low_memory,
        )
        if manifest["status"] != "complete":
            return None
        from .shared_weights import publish_qwen4_exp_shared_weights

        publish_qwen4_exp_shared_weights(output_dir)
        return self.META_CLS.from_dict(json.loads(Path(manifest["meta"]).read_text()))


__all__ = ["XHQwen4ExpModel"]
