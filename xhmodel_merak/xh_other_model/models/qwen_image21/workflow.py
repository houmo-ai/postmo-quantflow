# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Merak workflow for the complete Qwen-Image deployment bundle."""

from pathlib import Path

from ...workflows.base import BaseOtherModelWorkflow
from ...workflows.result import ExportResult, QuantResult
from .build import build_bundle, validate_build_config


class QwenImage21Workflow(BaseOtherModelWorkflow):
    def quant(self, output_dir, device, config_overrides=None):
        config = self.workflow_config.with_overrides(config_overrides)
        if config.quant is not None:
            raise ValueError("Qwen-Image applies PTQ during export; quant must be null")
        validate_build_config(config.build_export_dict())
        return QuantResult(raw_model_dir=self.model_dir, skipped=True)

    def export(self, quant_result, output_dir, device, config_overrides=None):
        config = self.workflow_config.with_overrides(config_overrides)
        source = self._resolve_export_model_dir(quant_result)
        if config.quant is not None or not quant_result.skipped:
            raise ValueError("this workflow exports the original floating-point checkpoint")
        export_config = config.build_export_dict()
        validate_build_config(export_config)
        root = Path(output_dir).expanduser().resolve() / "qwen_image21_export"
        metadata = build_bundle(source, root, export_config, device=device)
        config_file = config.dump(str(root / "workflow.yaml"))
        return ExportResult(work_dir=str(root), config_file=config_file, meta=metadata)
