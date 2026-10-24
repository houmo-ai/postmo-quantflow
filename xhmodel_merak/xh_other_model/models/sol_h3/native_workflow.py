# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from ...workflows.base import BaseOtherModelWorkflow
from ...workflows.result import ExportResult, QuantResult


class NativeH3Workflow(BaseOtherModelWorkflow):
    def quant(self, output_dir, device, config_overrides=None):
        from .native_assets import NativeSourceAssets
        from .native_build import validate_native_config
        from .quant_adapter import quantize_conditioner

        config = self.workflow_config.with_overrides(config_overrides)
        export_config = validate_native_config(config.build_export_dict())
        if config.quant is None:
            return QuantResult(raw_model_dir=self.model_dir, skipped=True)
        options = dict(config.quant)
        if options.get("calibration_file"):
            path = Path(options["calibration_file"]).expanduser()
            options["calibration_file"] = str(path if path.is_absolute() else Path(self.model_dir) / path)
        result = quantize_conditioner(
            NativeSourceAssets.read(self.model_dir),
            Path(output_dir) / "qwen3_vl_w4",
            options,
            layers=export_config["text_layers"],
            device=device,
        )
        return QuantResult(
            raw_model_dir=self.model_dir, quanted_model_dir=result.output_dir, meta={"conditioner": result.provenance}
        )

    def export(self, quant_result, output_dir, device, config_overrides=None):
        from .native_build import build_native_bundle

        config = self.workflow_config.with_overrides(config_overrides)
        if not quant_result.skipped and not quant_result.quanted_model_dir:
            raise ValueError("Native H3 conditioner checkpoint is missing")
        root = Path(output_dir).expanduser().resolve() / "h3_export"
        meta = build_native_bundle(
            quant_result.raw_model_dir,
            root,
            config.build_export_dict(),
            device=device,
            conditioner_checkpoint=None if quant_result.skipped else quant_result.quanted_model_dir,
        )
        config_file = config.dump(str(root / "workflow.yaml"))
        return ExportResult(work_dir=str(root), config_file=config_file, meta=meta)
