"""MiniCPM-V-4.7 text-only adapters for the Merak Qwen3.5 Dense/MoE stacks."""

from __future__ import annotations

import copy
import json
import os
from contextlib import contextmanager
from pathlib import Path

import torch.nn as nn
from accelerate import init_empty_weights

from xhmodel_merak._environment import scoped_environment
from xhmodel_merak.utils import calculate_file_md5
from xhmodel_merak.xh_llm.base_llm_model import BaseLLMModel
from xhmodel_merak.xh_llm.llm_data_processor import BaseLLMInputProcessor
from xhmodel_merak.xh_llm.utils import is_huge_model_export_enabled
from xhmodel_merak.xh_llm.minicpm_v_series.text_export import (
    export_minicpm_qwen35_text_hmonnx,
)
from xhmodel_merak.xh_llm.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
from xhmodel_merak.xh_llm.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import XHQwen3_5Model
from xhmodel_merak.xh_llm.models.qwen3_5_moe.qwen3_5_moe_model import XHQwen3_5MoeModel
from xhquant.utils import log_function_call

from .auto_classes import register_minicpmv47_auto_classes
from .dspark_config import MiniCPMV47DSparkConfig, MiniCPMV47ModelConfig, MiniCPMV47MoeModelConfig
from .positions import MiniCPMV47DataPreprocess

# MiniCPM keeps Qwen ABI placeholders for vision_start/end even though its
# multimodal chat template uses dedicated image/video pad tokens.
_DEFAULT_VISION_START_TOKEN_ID = 248053
_DEFAULT_VISION_END_TOKEN_ID = 248054


@contextmanager
def _draft_export_environment(config):
    # Torch 2.8's strict Dynamo guards scale poorly with GDN verification.
    # Retain standard ONNX optimization so exported slicing/tuple operations
    # are normalized for the runtime, and honor explicit caller overrides.
    flags = ("XHQUANT_STRICT_EXPORT",)
    values = (
        {name: os.environ.get(name, "0") for name in flags} if getattr(config.dflash_config, "is_dspark", False) else {}
    )
    with scoped_environment(values):
        yield


def _untie_lm_head_if_needed(text_model) -> None:
    input_weight = text_model.model.language_model.get_input_embeddings().weight
    if text_model.lm_head.weight is input_weight:
        text_model.lm_head.weight = nn.Parameter(
            input_weight.detach().clone(),
            requires_grad=input_weight.requires_grad,
        )
    text_model.config.text_config.tie_word_embeddings = False


def _apply_minicpm_text_abi(hf_model) -> None:
    hf_model.config.vision_start_token_id = (
        getattr(hf_model.config, "vision_start_token_id", None) or _DEFAULT_VISION_START_TOKEN_ID
    )
    hf_model.config.vision_end_token_id = (
        getattr(hf_model.config, "vision_end_token_id", None) or _DEFAULT_VISION_END_TOKEN_ID
    )
    if getattr(hf_model.config, "vision_config", None) is not None:
        hf_model.config.vision_config.spatial_merge_size = 2


class _MiniCPMV47TextMixin:
    """Shared MiniCPM positions, cache masking and text export for both stacks."""

    transformers_min_version = "5.18.0.dev0"

    def __init__(self, config):
        prediction_k = config.get("les_prediction_k", 0)
        if type(prediction_k) is not int or prediction_k < 0:
            raise ValueError("les_prediction_k must be a nonnegative integer (0 disables LES)")
        if prediction_k:
            if not isinstance(self, XHQwen3_5MoeModel):
                raise ValueError("LES requires MiniCPM MoE; the 1B dense model has no experts")
            if config.spec_decode_mode:
                raise ValueError("LES currently requires single-token ordinary decode; DSpark verify is multi-token")
            if config.dynamic_prune["mode"] != "off":
                raise ValueError("LES requires original Top-K routing, without ACEP pruning")
        draft = getattr(config, "dflash_config", None)
        with MiniCPMV47DSparkConfig._prepared_flat_checkpoint(getattr(draft, "hf_model", None)):
            super().__init__(config)
        self.wrap_cfg["mask_causal_values"] = True

    def _wraped_post(self, hf_model):
        # These fields only describe the text-graph preprocessing ABI. The
        # MiniCPM runtime supplies canvas positions separately and passes
        # image_grid_thw=None because its crops do not use Qwen vision grids.
        _apply_minicpm_text_abi(hf_model)
        return super()._wraped_post(hf_model)

    def _to_wrap(self, hf_model):
        from .causal_values import enable_minicpm_causal_values

        # Native 5.18 configs keep this in rope_parameters; the shared Qwen
        # wrapper still consumes the legacy attribute on the language model.
        text_config = self._get_language_model(hf_model).config
        if not hasattr(text_config, "partial_rotary_factor"):
            text_config.partial_rotary_factor = (text_config.rope_parameters or {}).get("partial_rotary_factor", 1.0)
        super()._to_wrap(hf_model)
        language_model = self._get_language_model(self._wrap_model)
        for index in self.full_attention_layer_indices:
            attention = language_model.layers[index].self_attn
            enable_minicpm_causal_values(attention)
            attention._update_cfg(self.wrap_cfg)

    def _get_data_preprocessor(self) -> BaseLLMInputProcessor:
        return MiniCPMV47DataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.wrap_cfg.input_sequence_length,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            past_conv_caches=self.past_conv_caches,
            past_recurrent_states=self.past_recurrent_states,
            patch_size=14,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
            vision_end_token_id=self.config.vision_end_token_id,
            spatial_merge_size=2,
        )

    def get_export_info(self, output_dir):
        info = BaseLLMModel.get_export_info(self, output_dir)
        self._minicpm_export_info = info
        return info

    def _export_target_verify_variant(self, exported_info, wrap_template, verify_length, adapter=None):
        from xhquant.api import to_export_graph, to_export_hmonnx_v2

        from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import _copy_model_shared_params
        from xhmodel_merak.xh_llm.shared_head import quantized_head
        from xhmodel_merak.xh_llm.types import LLMModelState
        from xhmodel_merak.xh_llm.utils import unfold_args

        canonical_state = self._state
        canonical_wrap_model = self._wrap_model
        canonical_frontend_model = self._frontend_model
        canonical_quanted_model = self._quanted_model
        canonical_llm_prefill = self._llm_prefill
        canonical_data_processor = self._data_processor
        canonical_rope_deltas = getattr(canonical_data_processor, "rope_deltas", None)
        canonical_wrap_cfg = copy.deepcopy(self.wrap_cfg)
        variant_wrap_model = _copy_model_shared_params(wrap_template)
        try:
            self._state = LLMModelState.WRAP
            self._wrap_model = variant_wrap_model
            self._frontend_model = None
            self._quanted_model = None
            self._data_processor = None
            self._llm_prefill = True
            self._restore_prefill_wrap_cfg()
            if adapter is not None:
                raise NotImplementedError(
                    "Confidence truncation with LoRA requires length-specific LoRA target verify graphs"
                )
            self.set_decode()
            self.wrap_cfg["verify_output_intermediates"] = True
            self.wrap_cfg["num_logits_to_keep"] = 0
            self.set_input_sequence_length(verify_length)
            if canonical_rope_deltas is None:
                raise RuntimeError("Target verify export requires rope_deltas from the completed target prefill trace")
            variant_data_processor = self.get_data_preprocessor()
            variant_data_processor.rope_deltas = canonical_rope_deltas
            # Only trace this decode length. The Qwen pair exporter resets to
            # prefill and rebuilds the canonical verify length from config.
            frontend_model = BaseLLMModel._to_fronted(self, variant_wrap_model)
            self._frontend_model = frontend_model
            self.release_wraped_model()
            self._state = LLMModelState.FRONTED
            self._shared_target_head = None
            self.bind_target_head(quantized_head(canonical_quanted_model.decode))
            try:
                quanted_model = BaseLLMModel._to_quanted(self, frontend_model, LLMModelState.QUANTED_ALIGNED)
            finally:
                if hasattr(self, "_shared_target_head"):
                    del self._shared_target_head
            quanted_model.fixed()
            self._quanted_model = quanted_model
            self._state = LLMModelState.QUANTED_ALIGNED
            with self.get_kvcache_mixin().kv_cache_scope(device="meta"):
                data_processor = self.get_data_preprocessor()
                inputs = unfold_args(data_processor(self.get_dummy_inputs()))
                exported_model = to_export_graph(quanted_model, inputs)
                decode_dir = Path(exported_info.exported_dir) / f"decode_{verify_length}"
                decode_dir.mkdir(parents=True, exist_ok=True)
                verify_file = decode_dir / f"{exported_info.model_name}_verify_{verify_length}.onnx"
                export_cfg = copy.deepcopy(self.get_export_cfg())
                if "input_names" in export_cfg or hasattr(export_cfg, "input_names"):
                    export_cfg["input_names"] = self.xh1_hmonnx_compatible(export_cfg["input_names"])
                return to_export_hmonnx_v2(
                    exported_model, inputs, str(verify_file), export_cfg, normalize_onnx_name=True,
                )
        finally:
            self._wrap_model = canonical_wrap_model
            self._frontend_model = canonical_frontend_model
            self._quanted_model = canonical_quanted_model
            self._state = canonical_state
            self._llm_prefill = canonical_llm_prefill
            self._data_processor = canonical_data_processor
            self.wrap_cfg.clear()
            self.wrap_cfg.update(canonical_wrap_cfg)

    def _finish_dspark_export(self, meta_info, wrap_template):
        draft = self.config.dflash_config
        if not getattr(draft, "is_dspark", False):
            return meta_info
        if is_huge_model_export_enabled() and draft.enable_confidence_truncation:
            raise NotImplementedError("DSpark confidence truncation requires regular target export")
        section = meta_info.spec_decode
        section["mode"] = "dspark"
        section["draft"].update(
            architecture="Qwen3DSparkModel",
            abi="minicpm_dspark_v2",
            block_size=int(draft.block_size),
            proposal_start=0,
            num_output_draft_tokens=int(self.config.num_draft_tokens),
            decode_outputs=["draft_token_ids"] + (["confidence_logits"] if draft.enable_confidence_truncation else []),
        )
        section.update(
            enable_confidence_truncation=bool(draft.enable_confidence_truncation),
            confidence_transform="sigmoid",
            confidence_aggregation="prefix_product",
        )
        if draft.confidence_threshold is not None:
            section["confidence_threshold"] = float(draft.confidence_threshold)
        exported_info = self._minicpm_export_info
        output_dir = Path(exported_info.exported_dir)
        if draft.enable_confidence_truncation:
            if wrap_template is None:
                raise RuntimeError("Confidence truncation requires a retained target wrap template")
            max_length = int(self.config.num_draft_tokens) + 1
            graphs = {str(max_length): meta_info.decode_hmonnx}
            digests = {str(max_length): meta_info.decode_hmonnx_md5}
            for length in range(1, max_length):
                exported = self._export_target_verify_variant(exported_info, wrap_template, length)
                graphs[str(length)] = str(Path(exported).relative_to(output_dir))
                digests[str(length)] = calculate_file_md5(exported)
            meta_info.verify_graphs = graphs
            meta_info.verify_graph_md5 = digests
            section["verify_graphs"] = dict(graphs)
            section["verify_graph_md5"] = dict(digests)
        (output_dir / "golden_meta_info.json").write_text(json.dumps(meta_info.to_dict(), indent=4), encoding="utf-8")
        return meta_info

    @log_function_call()
    def export_hmonnx(self, output_dir: str):
        """Export Qwen3.5 Prefill/Decode without its unrelated Vision graph."""

        if self.config.spec_decode_mode:
            draft = getattr(self.config, "dflash_config", None)
            dspark = bool(getattr(draft, "is_dspark", False))
            with _draft_export_environment(self.config):
                if dspark and draft.enable_confidence_truncation and is_huge_model_export_enabled():
                    raise NotImplementedError("DSpark confidence truncation requires regular target export")
                saved_length = None
                if dspark:
                    saved_length = draft.input_sequence_length
                    draft.input_sequence_length = int(draft.block_size)
                    draft.num_output_draft_tokens = int(self.config.num_draft_tokens)
                try:
                    from ..qwen3_5.qwen3_5_llm_model import _copy_model_shared_params
                    from .dspark_config import MiniCPMV47DSparkConfig
                    from .dspark_model import MiniCPMDSparkDraftExport

                    wrap_template = None
                    if dspark and draft.enable_confidence_truncation:
                        self.to_wrap()
                        wrap_template = _copy_model_shared_params(self._wrap_model)
                    with MiniCPMV47DSparkConfig._prepared_flat_checkpoint(getattr(draft, "hf_model", None) if dspark else None):
                        with MiniCPMDSparkDraftExport(int(self.config.num_draft_tokens) if dspark else None):
                            meta = XHQwen3_5Model.export_hmonnx(self, output_dir)
                finally:
                    if saved_length is not None:
                        draft.input_sequence_length = saved_length
                if dspark:
                    self._finish_dspark_export(meta, wrap_template)
                return meta
        return export_minicpm_qwen35_text_hmonnx(self, output_dir)


class MiniCPMV47TextModel(_MiniCPMV47TextMixin, XHQwen3_5Model):
    """Export the embedded Qwen3.5 Dense backend without Qwen's Vision graph."""

    CONFIG_CLS = MiniCPMV47ModelConfig

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs):
        """Load MiniCPM, then expose its text branch through a Qwen3.5 shell.

        ``quant_weight`` optionally points at a GPTQ-dequantized CausalLM
        ``pytorch_model.bin`` (``model.*`` + ``lm_head.weight``). It is applied
        onto the MiniCPM host before the Qwen3.5 shell wrap, matching the
        MiniCPM-V-4.5 pattern.
        """

        from .quant_llm import apply_causal_dequant_to_minicpm_host

        register_minicpmv47_auto_classes()
        kwargs.setdefault("trust_remote_code", True)
        full_model = super().get_hf_model(
            hf_model_dir,
            quant_weight=None,
            **kwargs,
        )
        if quant_weight:
            apply_causal_dequant_to_minicpm_host(full_model, quant_weight)
        qwen_config = Qwen3_5Config(
            text_config=full_model.config.text_config.to_dict(),
            image_token_id=full_model.config.image_token_id,
            video_token_id=full_model.config.video_token_id,
            vision_start_token_id=_DEFAULT_VISION_START_TOKEN_ID,
            vision_end_token_id=_DEFAULT_VISION_END_TOKEN_ID,
            tie_word_embeddings=False,
        )
        with init_empty_weights():
            text_model = Qwen3_5ForConditionalGeneration(qwen_config)

        text_model.model.language_model = full_model.model.language_model
        text_model.lm_head = full_model.lm_head
        text_model.generation_config = full_model.generation_config
        text_model.name_or_path = str(hf_model_dir)
        _untie_lm_head_if_needed(text_model)
        del full_model
        return text_model.eval()


class MiniCPMV47MoeTextModel(_MiniCPMV47TextMixin, XHQwen3_5MoeModel):
    """Export the embedded Qwen3.5 MoE backend without Qwen's Vision graph."""

    CONFIG_CLS = MiniCPMV47MoeModelConfig

    def _to_fronted(self, wrap_model):
        frontend = super()._to_fronted(wrap_model)
        if self.config.get("les_prediction_k", 0):
            from .les_graph import configure_les_decode_graph

            self._les_contract = configure_les_decode_graph(frontend.decode, self.config.les_prediction_k)
        return frontend

    def get_export_cfg(self):
        cfg = super().get_export_cfg()
        if self.config.get("les_prediction_k", 0) and self.is_decode():
            cfg["output_names"].extend(entry["name"] for entry in self._les_contract["outputs"])
        return cfg

    def get_export_info(self, output_dir):
        data = super().get_export_info(output_dir)
        if self.config.get("les_prediction_k", 0):
            from .les import validate_prediction_k
            from .les_graph import les_contract

            text = self._get_language_model(self._wrap_model).config
            validate_prediction_k(self.config.les_prediction_k, text.num_experts)
            data.meta.les = les_contract(len(self.layer_types), self.config.les_prediction_k, text.num_experts_per_tok)
        return data

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs):
        """Load MiniCPM MoE, then expose its text branch through a MoE shell.

        ``XHQwen3_5MoeModel.get_hf_model`` is inherited from Dense Qwen3.5 and
        applies ``qwen3_5_patch``. MiniCPM MoE modules are not Dense Qwen
        classes, so skip that path and load through ``XHBaseModel`` instead.

        ``quant_weight`` optionally points at a GPTQ/AutoRound-dequantized
        CausalLM ``pytorch_model.bin`` and is applied onto the MiniCPM host
        before the MoE shell wrap.
        """

        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
            Qwen3_5MoeConfig,
        )
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeForConditionalGeneration,
        )

        from .quant_llm import apply_causal_dequant_to_minicpm_host

        register_minicpmv47_auto_classes()
        kwargs.setdefault("trust_remote_code", True)
        full_model = super(XHQwen3_5Model, cls).get_hf_model(
            hf_model_dir,
            quant_weight=None,
            **kwargs,
        )
        if quant_weight:
            apply_causal_dequant_to_minicpm_host(full_model, quant_weight)
        qwen_config = Qwen3_5MoeConfig(
            text_config=full_model.config.text_config.to_dict(),
            image_token_id=full_model.config.image_token_id,
            video_token_id=full_model.config.video_token_id,
            vision_start_token_id=_DEFAULT_VISION_START_TOKEN_ID,
            vision_end_token_id=_DEFAULT_VISION_END_TOKEN_ID,
            tie_word_embeddings=False,
        )
        with init_empty_weights():
            text_model = Qwen3_5MoeForConditionalGeneration(qwen_config)

        text_model.model.language_model = full_model.model.language_model
        text_model.lm_head = full_model.lm_head
        text_model.generation_config = full_model.generation_config
        text_model.name_or_path = str(hf_model_dir)
        _untie_lm_head_if_needed(text_model)
        del full_model
        return text_model.eval()


__all__ = ["MiniCPMV47TextModel", "MiniCPMV47MoeTextModel"]
