"""Macaron-V1-Tall workflow built on the Qwen3.5-MoE implementation."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...workflows.result import ExportResult, QuantResult
from ..qwen3_5.workflow import Qwen35Workflow
from .lora_export_config import load_macaron_lora_export_config
from .lora_export_pipeline import (
    cleanup_macaron_base_export,
    inject_macaron_lora_export,
)


class MacaronV1TallWorkflow(Qwen35Workflow):
    """Reuse Qwen workflow stages and specialize only Macaron export."""

    expected_model_cls_names = (
        Qwen35Workflow.expected_model_cls_names
        | {"XHMacaronV1TallModel"}
    )
    family_label = "Macaron-V1-Tall"

    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        """Quantize the Qwen MoE backbone before Macaron export injection.

        ``MacaronV1TallForConditionalGeneration`` is an export-time wrapper
        around the Qwen3.5-MoE backbone.  The parent quantization workflow
        uses ``model_type`` to select its AutoRound topology, so it must see
        the underlying Qwen MoE type rather than the Macaron wrapper.
        """

        quant_overrides = dict(config_overrides or {})
        quant_overrides["export.model.model_type"] = (
            "Qwen3_5MoeForConditionalGeneration"
        )
        return super().quant(
            output_dir=output_dir,
            device=device,
            config_overrides=quant_overrides,
        )

    @staticmethod
    def export_overrides(
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Turn a Qwen baseline YAML into a Macaron graph export.

        The Macaron YAML already names this model so AutoLLM can select this
        workflow. The override remains idempotent for callers that reuse a
        Qwen baseline YAML directly.
        """

        overrides: dict[str, Any] = {
            "export.model.model_type": "MacaronV1TallForConditionalGeneration",
            "export.model.model_name": "macaron_v1_tall",
        }
        if extra:
            overrides.update(extra)
        # LoRA injection binds the recurrent projections as individual
        # Linear nodes; GDR fusion would remove those graph targets. Keep
        # these invariants after caller overrides have been applied.
        overrides["export.model.fuse_gdr_ops"] = False
        overrides["export.model.fuse_gdr_block_recurrent_ops"] = False
        return overrides

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        """Reuse Qwen export, then inject Macaron's runtime LoRA constants.

        Output is the compiler-facing export root. The temporary Qwen export
        uses a derived sibling directory and is removed after LoRA injection.
        """

        final_export_dir = Path(output_dir)
        base_export_dir = final_export_dir.with_name(
            f"{final_export_dir.name}_base_export"
        )
        if final_export_dir.exists():
            raise FileExistsError(
                f"Final export directory already exists: {final_export_dir}"
            )

        export_overrides = self.export_overrides(config_overrides)
        effective_config = self.workflow_config.with_overrides(export_overrides)
        lora_export_config = load_macaron_lora_export_config(
            effective_config.data
        )
        base_result = super().export(
            quant_result=quant_result,
            output_dir=str(base_export_dir),
            device=device,
            config_overrides=export_overrides,
        )
        inject_macaron_lora_export(
            model_dir=self.model_dir,
            base_export_dir=base_result.work_dir,
            base_model_export_dir=Path(
                self._find_golden_meta_file(base_result)
            ).parent,
            final_export_dir=final_export_dir,
            export_config=lora_export_config,
        )
        cleanup_macaron_base_export(
            base_export_dir=base_result.work_dir,
            final_export_dir=final_export_dir,
        )
        config_file = final_export_dir / Path(base_result.config_file).name
        return ExportResult(
            work_dir=str(final_export_dir),
            config_file=str(config_file),
            meta=base_result.meta,
        )

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any,
    ) -> str:
        """Generate visual and language golden through the Qwen workflow."""

        return super().dump_golden(
            export_result=export_result,
            device=device,
            input_messages=input_messages,
        )

__all__ = ["MacaronV1TallWorkflow"]
