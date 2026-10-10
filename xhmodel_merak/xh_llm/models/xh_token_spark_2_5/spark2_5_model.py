import json
from pathlib import Path
from typing import Any

from transformers import AutoModelForCausalLM, GenerationConfig

from xhmodel_merak.xh_llm.builder import register_llm_model
from xhmodel_merak.xh_llm.llm_data_processor import (
    BaseLLMInputProcessor,
    SlidingWindowInputProcessorConfig,
    SlidingWindowLLMInputProcessor,
)
from xhmodel_merak.xh_llm.text_llm_hf_compatible import build_text_llm_with_hf_compatible_v1
from xhmodel_merak.xh_llm.text_llm_model import TextLLMModel, TextLLMModelConfig

from .spark2_5_hmonnx_inference import XHSpark2_5HMONNXModel
from .spark_2_5 import Spark2_5Config, Spark2_5ForCausalLM, register_spark2_5


register_spark2_5()


class XHSpark2_5ModelConfig(TextLLMModelConfig):  # noqa: N801 - Spark 2_5 public name
    def __init__(self, *, enable_page_attention=False, **kwargs):
        if enable_page_attention:
            raise ValueError("Spark sliding-window processing does not support PageAttention")
        super().__init__(enable_page_attention=False, **kwargs)


@register_llm_model("Spark2_5ForCausalLM")
class XHSpark2_5Model(TextLLMModel):  # noqa: N801 - Spark 2_5 public name
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.xh_token_spark_2_5.workflow:Spark2_5Workflow"
    transformers_min_version = "4.58.0"
    transformers_max_version = "5.16.1"
    HF_MODEL_CLS = Spark2_5ForCausalLM
    HF_AUTO_MODEL_CLS = AutoModelForCausalLM
    HMONNXINFERENCE_CLS = XHSpark2_5HMONNXModel
    CONFIG_CLS = XHSpark2_5ModelConfig
    BUILD_HF_COMPATIBLE_FUNC = staticmethod(build_text_llm_with_hf_compatible_v1)

    def __init__(self, config: XHSpark2_5ModelConfig):
        super().__init__(config)
        self.sliding_window: int = -1
        self.wrap_cfg["flash_attention"] = self.config.flash_attention

    @classmethod
    def _local_checkpoint_config(cls, hf_model_dir: str) -> Spark2_5Config | None:
        config_path = Path(hf_model_dir) / "config.json"
        if config_path.is_file():
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if config.get("model_type") != Spark2_5Config.model_type:
                raise ValueError(
                    f"Expected Spark checkpoint model_type={Spark2_5Config.model_type!r}, "
                    f"got {config.get('model_type')!r}: {config_path}"
                )
            # Use the locally registered implementation for Merak wrapping.
            config.pop("auto_map", None)
            return Spark2_5Config.from_dict(config)
        return None

    @classmethod
    def _load_hf_config(cls, hf_model_dir: str, **kwargs: Any) -> Any:
        config = cls._local_checkpoint_config(hf_model_dir)
        if config is not None:
            return config
        config = super()._load_hf_config(hf_model_dir, **kwargs)
        if config.model_type != Spark2_5Config.model_type:
            raise ValueError(
                f"Expected Spark checkpoint model_type={Spark2_5Config.model_type!r}, "
                f"got {config.model_type!r}: {hf_model_dir}"
            )
        return config

    @classmethod
    def _load_hf_model(cls, hf_model_dir: str, **kwargs: Any) -> Any:
        config = cls._load_hf_config(hf_model_dir, **kwargs)
        if "config" not in kwargs:
            kwargs["config"] = config
        return super()._load_hf_model(hf_model_dir, **kwargs)

    @classmethod
    def _load_gptqmodel(cls, hf_model_dir: str, **kwargs: Any) -> Any:
        cls._load_hf_config(hf_model_dir, **kwargs)
        from .gptqmodel_wrap import register_spark2_5_gptqmodel

        register_spark2_5_gptqmodel()
        return super()._load_gptqmodel(hf_model_dir, **kwargs)

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs: Any) -> Any:
        # Preserve each projection's W4/W8 payload and logical bits for SDK PTQ.
        kwargs.setdefault("gptq_weight_mode", "packed")
        return super().get_hf_model(hf_model_dir, quant_weight=quant_weight, **kwargs)

    @classmethod
    def get_empty_hf_model(cls, hf_model_dir: str, **kwargs: Any) -> Any:
        config = cls._load_hf_config(hf_model_dir, **kwargs)
        from accelerate import init_empty_weights

        include_buffers = kwargs.pop("include_buffers", False)
        kwargs.setdefault("dtype", cls.HF_MODEL_DTYPE)
        kwargs.setdefault("trust_remote_code", True)
        with init_empty_weights(include_buffers=include_buffers):
            model = cls.HF_AUTO_MODEL_CLS.from_config(config, **kwargs)
        try:
            model.generation_config = GenerationConfig.from_pretrained(hf_model_dir)
        except OSError:
            pass  # Keep the generation defaults derived from the model config.
        return model

    def init_wrap_model(self, hf_model: Any) -> Any:
        from ._model import register_wrap_modules

        self.sliding_window = hf_model.config.sliding_window
        self.config.sliding_window = self.sliding_window  # type: ignore
        register_wrap_modules()
        return super().init_wrap_model(hf_model)

    @classmethod
    def _postprocess_gptqmodel_structure(cls, native_hf_model: Any, **kwargs: Any) -> Any:
        native_hf_model = cls.untied_weights(native_hf_model)
        native_hf_model.config.tie_word_embeddings = False
        return native_hf_model

    def _wraped_post(self, hf_model: Any) -> None:
        # TextLLMModel already handles the standard Spark2_5 embedding and cache layout.
        super()._wraped_post(hf_model)

    def _get_hf_config_files(self, hf_model_path: Path) -> list[Path]:
        return [
            *hf_model_path.glob("*.json"),
            *hf_model_path.glob("*.jinja"),
            *hf_model_path.glob("*.model"),
        ]

    def _get_data_preprocessor(self) -> BaseLLMInputProcessor:
        assert self.sliding_window > 0
        preprocessor = SlidingWindowLLMInputProcessor(
            SlidingWindowInputProcessorConfig(
                embed_tokens=self.embed_tokens,
                input_sequence_length=self.wrap_cfg.input_sequence_length,
                past_key_caches=self.past_key_caches,
                past_value_caches=self.past_value_caches,
                pad_token_id=self.pad_token_id,
                sliding_window=self.sliding_window,
            )
        )
        return preprocessor

    def get_export_cfg(self) -> dict[str, list[str]]:
        export_cfg = super().get_export_cfg()
        export_cfg["input_names"].insert(3, "sliding_attention_mask")
        return export_cfg
