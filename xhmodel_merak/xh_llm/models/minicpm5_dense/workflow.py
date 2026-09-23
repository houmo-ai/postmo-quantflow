from collections.abc import Mapping
from typing import Any

from ..qwen3.workflow import XHQwen3HMONNXWorkflow


class XHMiniCPM5Workflow(XHQwen3HMONNXWorkflow):
    expected_model_config_cls_name = "XHMiniCPM5ModelConfig"
    expected_model_cls_name = "XHMiniCPM5Model"

    def build_input_message(self, input_messages: Any) -> list[dict[str, Any]]:
        if isinstance(input_messages, str):
            prompt = input_messages
        elif isinstance(input_messages, Mapping):
            prompt = input_messages.get("text")
        else:
            prompt = None
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("MiniCPM5 input text must be a non-empty string")
        return [{"role": "user", "content": prompt}]


__all__ = ["XHMiniCPM5Workflow"]
