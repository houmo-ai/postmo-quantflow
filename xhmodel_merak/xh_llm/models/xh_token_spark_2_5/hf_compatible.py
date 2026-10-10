"""Transformers generation compatibility for the Spark2_5 XH wrapper.

The generic xhmodel-merak compatibility layer treats ``past_key_values=None``
as a request to disable its externally managed KV tensors.  Spark2_5 uses
preallocated XHQuant cache tensors instead, so an empty Transformers cache is
still needed on the first prefill call to keep those tensors in the data path.
"""

from typing import Any

import torch

from xhmodel_merak.xh_llm.text_llm_hf_compatible import TextLLMHFCompatible
from xhquant.utils.registry import _DMRegistryCls


class _Spark2_5HFCompatible_(TextLLMHFCompatible):  # noqa: N801
    def _setup(self, text_llm_model):
        m = super()._setup(text_llm_model)
        if text_llm_model is not None:
            if hasattr(m, "model"):
                del m.model
            if hasattr(m, "lm_head"):
                del m.lm_head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return m


def build_spark_tokens_hf_compatible(hf_model: Any, text_llm_model: Any) -> Any:
    LLM_COMPATIBLE_MODULES = _DMRegistryCls("XHCompatible")  # noqa: N806
    hf_model_cls = type(hf_model)
    if hf_model_cls not in LLM_COMPATIBLE_MODULES:
        LLM_COMPATIBLE_MODULES.register_module(
            {
                hf_model_cls: hf_model_cls.__name__,
            },
            _Spark2_5HFCompatible_,
        )
    return LLM_COMPATIBLE_MODULES.convert(hf_model, text_llm_model=text_llm_model)
