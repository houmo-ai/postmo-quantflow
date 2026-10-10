from xhmodel_merak.xh_llm.hmonnx import TextLLMHMONNXModel
from xhmodel_merak.xh_llm.llm_data_processor import (
    BaseLLMInputProcessor,
    SlidingWindowInputProcessorConfig,
    SlidingWindowLLMInputProcessor,
)
from xhmodel_merak.xh_llm.types import LLMModelMeta


class XHSpark2_5HMONNXModel(TextLLMHMONNXModel):  # noqa: N801 - Spark 2_5 public name
    def __init__(self, meta_info: LLMModelMeta, **kwargs):
        if kwargs.get("enable_page_attention"):
            raise ValueError("Spark sliding-window processing does not support PageAttention")
        super().__init__(meta_info, **kwargs)
        self.sliding_window = meta_info.model_config.sliding_window  # type: ignore

    def _get_data_preprocessor(self) -> BaseLLMInputProcessor:
        assert self.sliding_window > 0
        preprocessor = SlidingWindowLLMInputProcessor(
            SlidingWindowInputProcessorConfig(
                embed_tokens=self.get_input_embeddings(),
                input_sequence_length=self.get_input_sequence_length(),
                past_key_caches=self.past_key_caches,
                past_value_caches=self.past_value_caches,
                enable_page_attention=self.enable_page_attention,
                pad_token_id=self.pad_token_id,
                sliding_window=self.sliding_window,
            )
        )
        return preprocessor


__all__ = [
    "XHSpark2_5HMONNXModel",
]
