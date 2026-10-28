# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from ...workflows.base import BaseOtherModelWorkflow
from ...workflows.result import ExportResult, QuantResult


class WallOSSWorkflow(BaseOtherModelWorkflow):
    def quant(self, output_dir, device, config_overrides=None):
        from .build import validate_config

        cfg = self.workflow_config.with_overrides(config_overrides)
        if cfg.quant is not None:
            raise ValueError("Wall applies PTQ during export; quant must be null")
        validate_config(cfg.build_export_dict())
        return QuantResult(raw_model_dir=self.model_dir, skipped=True)

    def export(self, quant_result, output_dir, device, config_overrides=None):
        from .build import build_bundle

        cfg = self.workflow_config.with_overrides(config_overrides)
        if cfg.quant is not None or not quant_result.skipped:
            raise ValueError("requires floating checkpoint")
        root = Path(output_dir) / "wall_oss_export"
        meta = build_bundle(self._resolve_export_model_dir(quant_result), root, cfg.build_export_dict(), device)
        return ExportResult(work_dir=str(root), config_file=cfg.dump(str(root / "workflow.yaml")), meta=meta)

    def dump_golden(self, export_result, device, input_messages=None):
        from .golden import dump_golden

        return dump_golden(export_result.work_dir, device=device, inputs=input_messages)
