"""Xing4.0 runtime using the text-LLM ABI and exported independent MLA caches."""

from ...hmonnx import TextLLMHMONNXModel
from ...base_llm_model import XHLLMModelProcessor


class XHXing4_0HMONNXModel(TextLLMHMONNXModel):
    """Use the standard text-LLM HMONNX ABI for Xing4.0."""

    def get_tf_processor(self):
        return XHLLMModelProcessor(self.get_tokenizer())


__all__ = ["XHXing4_0HMONNXModel"]
