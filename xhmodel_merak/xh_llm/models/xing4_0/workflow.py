"""Merak workflow for Xing4.0-29B-A4B."""

from collections.abc import Mapping
from typing import Any

from xhmodel_merak._environment import scoped_environment

from ...workflows.base import BaseLLMWorkflow
from ...workflows.result import ExportResult, QuantResult
from ...workflows.runtime_config import finalize_merak_runtime_config


class Xing4_0Workflow(BaseLLMWorkflow):
    expected_model_config_cls_name = "XHXing4_0ModelConfig"
    expected_model_cls_name = "XHXing4_0Model"

    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        quant_cfg = workflow_config.quant
        if quant_cfg is None:
            return QuantResult(raw_model_dir=self.model_dir, skipped=True)
        algorithm = str(quant_cfg.get("algorithm", "gptqmodel")).lower().replace("-", "_")
        if algorithm not in {"gptq", "gptqmodel"}:
            raise NotImplementedError(
                "Xing4_0Workflow.quant supports quant=None for the floating-point HF path, "
                "or quant.algorithm='gptqmodel' for GPTQ/AutoRound."
            )
        from .quant_adapter import quantize_with_gptqmodel_api

        return quantize_with_gptqmodel_api(
            model_dir=self.model_dir,
            output_dir=output_dir,
            device=device,
            quant_cfg=quant_cfg,
            workflow_seed=self.seed,
        )

    def export(self, quant_result, output_dir, device, config_overrides=None) -> ExportResult:
        if quant_result.quanted_model_dir:
            from xhmodel_merak.model_integrations.xing4_0.checkpoint import inspect_quant_regions

            quant_cfg = self.workflow_config.with_overrides(config_overrides).quant
            routed_bits = int(quant_cfg.get("moe", {}).get("expert_bits", 4)) if quant_cfg else None
            inspect_quant_regions(quant_result.quanted_model_dir, routed_bits=routed_bits)
        result = super().export(quant_result, output_dir, device, config_overrides)
        finalize_merak_runtime_config(result)
        return result

    def dump_golden(self, export_result: ExportResult, device: str, input_messages: Any) -> str:
        with scoped_environment({"XHQUANT_MOE_LOW_M_TRITON": "0"}):
            return self._dump_golden(export_result, device, input_messages)

    def _dump_golden(self, export_result: ExportResult, device: str, input_messages: Any) -> str:
        from transformers import TextStreamer

        from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
        from xhquant.utils import ContextManagers

        meta_file = self._find_golden_meta_file(export_result)
        hmonnx_model = AutoLLMHONNXModel.from_pretrained(
            meta_file,
            device_map=[device] if isinstance(device, str) else device,
        )
        hmonnx_model.enable_golden = True
        tokenizer = hmonnx_model.get_tokenizer()
        prompt = input_messages if isinstance(input_messages, str) else input_messages.get("text", "")
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(hmonnx_model.device)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            eos_token_id = tokenizer.eos_token_id
            if isinstance(eos_token_id, (list, tuple)):
                eos_token_id = eos_token_id[0]
            if eos_token_id is None:
                raise ValueError("Xing4.0 tokenizer has no pad or EOS token id")
            pad_token_id = int(eos_token_id)
        with ContextManagers([LLMInferenceContextManager(hmonnx_model)]):
            hmonnx_model.generate(
                **model_inputs,
                max_new_tokens=2,
                streamer=TextStreamer(tokenizer),
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        return meta_file


__all__ = ["Xing4_0Workflow"]
