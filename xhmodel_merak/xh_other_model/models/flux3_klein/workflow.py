# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

from ...workflows.base import BaseOtherModelWorkflow
from ...workflows.result import ExportResult, QuantResult


class Flux3KleinWorkflow(BaseOtherModelWorkflow):
    def quant(self, output_dir, device, config_overrides=None):
        from .build import validate_build_config

        config = self.workflow_config.with_overrides(config_overrides)
        if config.quant is not None:
            raise ValueError("FLUX 3 applies PTQ during export; quant must be null")
        validate_build_config(config.build_export_dict())
        return QuantResult(raw_model_dir=self.model_dir, skipped=True)

    def export(self, quant_result, output_dir, device, config_overrides=None):
        from .build import build_bundle

        config = self.workflow_config.with_overrides(config_overrides)
        if config.quant is not None or not quant_result.skipped:
            raise ValueError("FLUX 3 exports the original floating-point checkpoint")
        source = self._resolve_export_model_dir(quant_result)
        root = Path(output_dir).resolve() / "flux3_klein_export"
        meta = build_bundle(source, root, config.build_export_dict(), device=device)
        return ExportResult(work_dir=str(root), config_file=config.dump(str(root / "workflow.yaml")), meta=meta)

    def dump_golden(self, export_result, device, input_messages=None):
        from .golden import dump_golden

        return dump_golden(export_result.work_dir, device=device, inputs=input_messages)
