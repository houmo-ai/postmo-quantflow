"""Spark quantization dispatch and Merak export preparation."""

from collections.abc import Mapping

from ...workflows.base import BaseLLMWorkflow
from ...workflows.result import QuantResult
from ...workflows.runtime_config import finalize_merak_runtime_config


class Spark2_5Workflow(BaseLLMWorkflow):  # noqa: N801 - follows the Spark2_5 HF architecture name
    expected_model_cls_name = "XHSpark2_5Model"
    expected_model_config_cls_name = "XHSpark2_5ModelConfig"

    def quant(self, output_dir, device, config_overrides=None) -> QuantResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        quant_cfg = workflow_config.quant
        skipped = QuantResult(raw_model_dir=self.model_dir, skipped=True)
        # Validate before starting a producer; skip keeps the YAML's head requirement.
        self._prepare_export_head(skipped, workflow_config, device)
        if quant_cfg is None:
            return skipped

        for key in ("artifact_format", "output_format"):
            if quant_cfg.get(key, "gptqmodel_hf") != "gptqmodel_hf":
                raise ValueError(f"Spark {key} must be gptqmodel_hf")
        algorithm = str(quant_cfg.get("algorithm") or "gptqmodel").strip().lower().replace("-", "_")
        method = str(quant_cfg.get("method") or "").strip().lower().replace("-", "_")
        if algorithm in {"autoround", "auto_round"}:
            if method not in {"", "autoround", "auto_round", "mode1"}:
                raise ValueError(f"Spark algorithm={algorithm!r} conflicts with method={method!r}")
            method = "autoround"
        if algorithm in {"autoround", "auto_round"} or (
            algorithm == "gptqmodel" and method in {"autoround", "auto_round", "mode1"}
        ):
            # The CLI adapter consumes canonical names; do not mutate the YAML.
            normalized = {**quant_cfg, "algorithm": "gptqmodel", "method": "autoround"}
            normalized.pop("head", None)  # W8 head is validated above and handled by export RTN.
            return self._quant_autoround_api(output_dir=output_dir, device=device, quant_cfg=normalized)
        if algorithm in {"gptqmodel", "gptq"} and method in {"", "gptq"}:
            return self._quant_gptqmodel_api(output_dir=output_dir, device=device, quant_cfg=quant_cfg)
        raise NotImplementedError(
            f"Unsupported Spark quant algorithm={algorithm!r}, method={method!r}; "
            "use algorithm=gptqmodel, method=autoround. GPTQ has a reserved backend entry."
        )

    def _quant_autoround_api(self, *, output_dir, device, quant_cfg) -> QuantResult:
        from .quant_adapter import quantize_autoround

        return quantize_autoround(model_dir=self.model_dir, output_dir=output_dir, device=device, quant_cfg=quant_cfg)

    def _quant_gptqmodel_api(self, *, output_dir, device, quant_cfg) -> QuantResult:
        """Extension point for a Spark GPTQModel producer, separate from AutoRound."""
        raise NotImplementedError(
            "Spark GPTQ quantization is not implemented; connect a Spark GPTQModel backend "
            "in _quant_gptqmodel_api. Currently use algorithm=gptqmodel, method=autoround."
        )

    def _prepare_export_head(self, quant_result, workflow_config, device):
        """Keep floating W8 head for export RTN; reject unsupported calibration."""
        from ...shared_head import target_head_bits

        if workflow_config.export["model"].get("enable_page_attention"):
            raise ValueError("Spark sliding-window processing does not support PageAttention")
        quant_cfg = workflow_config.quant
        if quant_cfg is None:
            quant_cfg = self.workflow_config.quant or {}
        if quant_cfg.get("mtp") is not None:
            raise NotImplementedError("Spark MTP quantization is not implemented")
        export_bits = target_head_bits(workflow_config.export["model"].get("quant_scheme"))
        head = quant_cfg.get("head")
        if head is None:
            head = {}
        if not isinstance(head, Mapping):
            raise TypeError("quant.head must be a mapping")
        if head:
            if set(head) - {"bits", "group_size", "algorithm"}:
                raise ValueError("Unsupported Spark quant.head fields")
            if head.get("bits") not in (4, 5, 8):
                raise ValueError("quant.head.bits must be 4, 5 or 8")
            if head.get("group_size", 64) != 64:
                raise ValueError("quant.head.group_size must be 64")
            if head.get("algorithm", "inherit") not in {"inherit", "gptq", "autoround"}:
                raise ValueError("quant.head.algorithm must be inherit, gptq or autoround")
            if head["bits"] != export_bits:
                raise ValueError(f"quant.head.bits={head['bits']} conflicts with export lm_head W{export_bits}")
        if export_bits < 8:
            raise NotImplementedError("Spark low-bit lm_head calibration is not implemented; export lm_head with W8. ")
        return quant_result

    def export(self, quant_result, output_dir, device, config_overrides=None):
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        quant_result = self._prepare_export_head(quant_result, workflow_config, device)
        result = super().export(quant_result, output_dir, device, config_overrides)
        finalize_merak_runtime_config(result)
        return result
