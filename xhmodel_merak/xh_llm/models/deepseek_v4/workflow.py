"""Workflow integration for low-memory DeepSeek-V4 Flash export."""

from __future__ import annotations

import argparse
import gc
import tempfile
from pathlib import Path
from collections.abc import Mapping
from typing import Any

from ...utils import configure_huge_model_export
from ...workflows.base import BaseLLMWorkflow
from ...workflows.result import ExportResult, QuantResult


class DeepSeekV4Workflow(BaseLLMWorkflow):
    """Export an existing GPTQModel/AutoRound checkpoint through workflow YAML."""

    expected_model_config_cls_name = "XHDeepSeekV4ModelConfig"
    expected_model_cls_name = "XHDeepSeekV4Model"

    def get_model(self, quant_result=None, config_overrides=None, work_dir=None):
        model = super().get_model(quant_result, config_overrides, work_dir)
        if hasattr(self, "_dspark_export_device"):
            model._dspark_export_device = self._dspark_export_device
        return model

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        runtime = workflow_config.data.get("runtime", {})
        if not isinstance(runtime, Mapping):
            raise TypeError("DeepSeek-V4 workflow runtime config must be a mapping")
        low_memory = runtime.get("low_memory", True)
        if not isinstance(low_memory, bool):
            raise TypeError("DeepSeek-V4 workflow runtime.low_memory must be a boolean")
        configure_huge_model_export(low_memory)
        self._dspark_export_device = device
        dspark = workflow_config.export["model"].get("dspark")
        if dspark and not dspark.get("model_dir"):
            # Native FP4/FP8 decoding belongs to the producer. Keep the large
            # BF16 preparation outside the published stage and remove it once
            # the W4/W8 runtime graphs have been exported.
            from gptqmodel.utils.deepseek_v4_dspark import prepare_deepseek_v4_dspark_bf16

            parent = Path(output_dir).resolve().parent
            parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="dspark-preparation-", dir=parent) as temporary:
                prepared = Path(temporary) / "bf16"
                prepare_deepseek_v4_dspark_bf16(self._resolve_export_model_dir(quant_result), prepared, device=device)
                overrides = dict(config_overrides or {})
                overrides["export.model.dspark.model_dir"] = str(prepared)
                return super().export(quant_result=quant_result, output_dir=output_dir,
                                      device=device, config_overrides=overrides)
        return super().export(
            quant_result=quant_result,
            output_dir=output_dir,
            device=device,
            config_overrides=config_overrides,
        )

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any = None,
        *,
        device_map: list[str] | None = None,
    ) -> str:
        """Dump one prefill/decode step using the exported runtime package."""

        import torch

        from examples_merak.llm.deepseekv4.deepseek_v4_xh_hmonnx_generate import run

        prompt = "17 multiplied by 3 equals what? Answer with only the result."
        if isinstance(input_messages, str):
            prompt = input_messages
        elif isinstance(input_messages, Mapping) and input_messages.get("text"):
            prompt = str(input_messages["text"])

        if device_map is None:
            normalized_device = str(device).strip().lower()
            if normalized_device == "cuda":
                device_map = ["0"]
            elif normalized_device.startswith("cuda:"):
                device_map = [normalized_device.removeprefix("cuda:")]
            else:
                device_map = [token.strip() for token in normalized_device.split(",") if token.strip()]

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        run(
            argparse.Namespace(
                config=export_result.work_dir,
                prompt=prompt,
                context_file=None,
                raw_prompt=False,
                max_new_tokens=2,
                device=",".join(device_map),
                pack_w4=True,
                cuda_graph=False,
                golden=True,
                stream=False,
                load_only=False,
                debug=self.debug,
                log_file=None,
            )
        )
        return self._find_golden_meta_file(export_result)


__all__ = ["DeepSeekV4Workflow"]
