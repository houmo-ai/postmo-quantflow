from transformers import AutoConfig, AutoModelForImageTextToText

from . import hmonnx_validation
from .configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from .modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from .qwen3_5_hmonnx_inference import XHQwen3_5_HMONNXModel
from .qwen3_5_llm_model import XHQwen3_5Model
from .qwen3_5_onnx_model import Qwen3_5ONNXModel
from .qwen3_5_spec_decode_onnx_model import Qwen3_5SpecDecodeONNXModel
from .qwen3_5_vision_model import XHQwen3_5VisionModel
from .xh_qwen3_5_config import XHQwen3_5_VisualConfig, XHQwen3_5ModelConfig


def _register_transformers_once(register_fn) -> None:
    """Register vendored Qwen3.5 classes without replacing native support."""
    try:
        register_fn()
    except ValueError as exc:
        message = str(exc).lower()
        if "already" not in message and "exist" not in message:
            raise


_register_transformers_once(lambda: AutoConfig.register("qwen3_5", Qwen3_5Config))
_register_transformers_once(lambda: AutoConfig.register("qwen3_5_text", Qwen3_5TextConfig))
_register_transformers_once(
    lambda: AutoModelForImageTextToText.register(Qwen3_5Config, Qwen3_5ForConditionalGeneration)
)


__all__ = [
    "Qwen3_5Config",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5TextConfig",
    "Qwen3_5ONNXModel",
    "Qwen3_5SpecDecodeONNXModel",
    "XHQwen3_5Model",
    "XHQwen3_5ModelConfig",
    "XHQwen3_5_HMONNXModel",
    "XHQwen3_5VisionModel",
    "XHQwen3_5_VisualConfig",
    "hmonnx_validation",
]
