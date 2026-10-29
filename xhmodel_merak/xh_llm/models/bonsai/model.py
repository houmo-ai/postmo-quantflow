"""A Bonsai-only specialization of the existing Merak Qwen3.5 model."""

from accelerate import init_empty_weights

from ...builder import register_llm_model
from ..qwen3_5.data_preprocess import Qwen3_5_DataPreprocess
from ..qwen3_5.qwen3_5_llm_model import XHQwen3_5Model
from ..qwen3_5.xh_qwen3_5_config import XHQwen3_5ModelConfig
from .checkpoint import checkpoint_paths, load_phase_a_checkpoint
from .runtime import XHBonsaiHMONNXModel
from .wrapping import attach_bonsai_activations, wrap_bonsai_model


class XHBonsaiModelConfig(XHQwen3_5ModelConfig):
    def __init__(self, *, model_type="BonsaiForConditionalGeneration", **kwargs):
        if kwargs.get("spec_decode_mode") is not None or kwargs.get("lora") is not None:
            raise ValueError("Bonsai Phase A currently supports the standard text-only export")
        if kwargs.get("visual_config") is not None:
            raise ValueError("Bonsai Phase A does not export a visual tower")
        if kwargs.get("fuse_gdr_ops") or kwargs.get("fuse_gdr_block_recurrent_ops"):
            raise ValueError("Bonsai Phase A uses its FP32 unfused delta-rule reference")
        kwargs.setdefault("fuse_gdr_ops", False)
        kwargs.setdefault("fuse_gdr_block_recurrent_ops", False)
        super().__init__(model_type=model_type, **kwargs)


@register_llm_model("BonsaiForConditionalGeneration")
class XHBonsaiModel(XHQwen3_5Model):
    CONFIG_CLS = XHBonsaiModelConfig
    HMONNXINFERENCE_CLS = XHBonsaiHMONNXModel
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.bonsai.workflow:BonsaiWorkflow"

    def _get_data_preprocessor(self):
        return Qwen3_5_DataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.wrap_cfg.input_sequence_length,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            past_conv_caches=self.past_conv_caches,
            past_recurrent_states=self.past_recurrent_states,
        )

    @classmethod
    def get_hf_model(cls, hf_model_dir, quant_weight=None, **kwargs):
        checkpoint, _, _ = checkpoint_paths(hf_model_dir, quant_weight)
        config = cls.HF_MODEL_CLS.config_class.from_pretrained(hf_model_dir)
        with init_empty_weights(include_buffers=False):
            model = cls.HF_MODEL_CLS(config)
        load_phase_a_checkpoint(model, checkpoint)
        return model.eval()

    def init_wrap_model(self, hf_model=None):
        if hf_model is None:
            hf_model = self.get_native_model()
        checkpoint, aux, _ = checkpoint_paths(self.hf_model_dir)
        if not getattr(hf_model, "_bonsai_phase_a_loaded", False):
            load_phase_a_checkpoint(hf_model, checkpoint)
        hf_model.to(dtype=self._dtype)
        self._wraped_pre(hf_model)
        self._wrap_model = wrap_bonsai_model(hf_model, self.get_wrap_cfg())
        self._wraped_post(hf_model)
        attach_bonsai_activations(self._wrap_model, aux)
        self._trim_cpu_allocator()
        return self._wrap_model
