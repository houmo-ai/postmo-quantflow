# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from ...workflows.base import BaseOtherModelWorkflow
from ...workflows.result import ExportResult, QuantResult


class SolH3Workflow(BaseOtherModelWorkflow):
    def quant(self, output_dir, device, config_overrides=None):
        from .build import validate_build_config

        config = self.workflow_config.with_overrides(config_overrides)
        export_config = validate_build_config(config.build_export_dict())
        if config.quant is None:
            return QuantResult(raw_model_dir=self.model_dir, skipped=True)
        from .assets import SourceAssets
        from .quant_adapter import quantize_conditioner

        root = Path(output_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        options = dict(config.quant)
        if options.get("calibration_file"):
            calibration = Path(options["calibration_file"]).expanduser()
            options["calibration_file"] = str(
                calibration if calibration.is_absolute() else Path(self.model_dir) / calibration
            )
        result = quantize_conditioner(
            SourceAssets.read(self.model_dir),
            root / "qwen3_vl_w4",
            options,
            layers=export_config["text_layers"],
            device=device,
        )
        return QuantResult(
            raw_model_dir=self.model_dir, quanted_model_dir=result.output_dir, meta={"conditioner": result.provenance}
        )

    def export(self, quant_result, output_dir, device, config_overrides=None):
        from .build import build_bundle

        config = self.workflow_config.with_overrides(config_overrides)
        if not quant_result.skipped and not quant_result.quanted_model_dir:
            raise ValueError("Sol-H3 quantization result is missing the conditioner checkpoint")
        root = Path(output_dir).expanduser().resolve() / "sol_h3_export"
        metadata = build_bundle(
            quant_result.raw_model_dir,
            root,
            config.build_export_dict(),
            device=device,
            conditioner_checkpoint=None if quant_result.skipped else quant_result.quanted_model_dir,
        )
        config_file = config.dump(str(root / "workflow.yaml"))
        return ExportResult(work_dir=str(root), config_file=config_file, meta=metadata)
