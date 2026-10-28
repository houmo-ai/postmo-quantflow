# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Export OCR vision and Qwen3.5 prefill/decode into a movable release."""

import json
import shutil
from datetime import datetime
from pathlib import Path

from xhmodel_merak.xh_llm.workflows.base import BaseLLMWorkflow
from xhmodel_merak.xh_llm.workflows.result import ExportResult


class MiniCPMOcrWorkflow(BaseLLMWorkflow):
    def export(self, quant_result, output_dir, device, config_overrides=None):
        import torch
        from xhquant.api import xhquant_init
        from xhmodel_merak.xh_llm.minicpm_v_series.token_capacity_vision_export import export_vision_profile
        from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import XHQwen3_5ModelConfig
        from .model import XHMiniCPMOcrModel
        from .native import register_native

        register_native()

        cfg = self.workflow_config.with_overrides(config_overrides)
        model_cfg = dict(cfg.build_export_dict()["model"])
        model_dir = self._resolve_export_model_dir(quant_result)
        root = Path(output_dir).resolve()
        if root.exists() and any(root.iterdir()):
            raise FileExistsError(root)
        root.mkdir(parents=True, exist_ok=True)
        if model_cfg["chip_arch"] != "XH2a":
            raise ValueError("MiniCPM-OCR export currently requires XH2a")
        xhquant_init(str(root / "export.log"), debug=self.debug)
        torch.manual_seed(self.seed)
        config_file = cfg.dump(str(root / "effective_workflow.yaml"))
        visual_cfg = model_cfg.pop("visual_config")
        capacity = int(visual_cfg["token_capacity"])
        formatted_name = self._format_model_name(cfg, model_dir)
        release_prefix = f"hmquant_{formatted_name}_{datetime.now():%Y%m%d}"
        profile = export_vision_profile(
            model_dir=model_dir,
            work_dir=root,
            target_device="XH2a",
            quant_type=visual_cfg["quant_scheme"]["quant_type"],
            token_capacity=capacity,
            downsample_mode="4x",
            validation_target=tuple(visual_cfg.get("validation_target", (32, 32))),
            device=device,
            model_label="MiniCPM-OCR-1.0",
            artifact_family=release_prefix,
        )
        old_visual = root / "vision_4x"
        visual_dir = root / f"visual_m{capacity // 4}"
        shutil.move(str(old_visual / "hmonnx"), visual_dir)
        shutil.rmtree(old_visual)
        profile["hmonnx"] = str((visual_dir / Path(profile["hmonnx"]).name).relative_to(root))
        model_cfg.update(hf_model=model_dir, visual_config=None, model_name=formatted_name)
        text = XHMiniCPMOcrModel(XHQwen3_5ModelConfig(**model_cfg))
        with torch.cuda.device(torch.device(device)):
            text.export_hmonnx(str(root / "text_work"))
        children = list((root / "text_work").glob("hmquant_*"))
        if len(children) != 1:
            raise RuntimeError("Expected one OCR text export")
        child = children[0]
        meta = json.loads((child / "golden_meta_info.json").read_text())
        for item in child.iterdir():
            shutil.move(str(item), root / item.name)
        shutil.rmtree(root / "text_work")
        # Preserve the native OCR processor config: the text shell is Qwen,
        # while image preprocessing and tokenizer remain MiniCPM-OCR.
        hf_config = root / meta["hf_config"]
        for name in (
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "preprocessor_config.json",
            "processor_config.json",
            "video_preprocessor_config.json",
            "chat_template.jinja",
        ):
            source = Path(model_dir) / name
            if source.is_file():
                shutil.copy2(source, hf_config / name)
        (root / "golden_meta_info.json").write_text(json.dumps(meta, indent=2) + "\n")
        runtime_meta = {
            "schema_version": 1,
            "model_type": "minicpm_ocr",
            "downsample_mode": "4x",
            "text_metadata": "golden_meta_info.json",
            "visual": profile,
            "components": ["visual", "prefill", "decode"],
            "hf_config": meta["hf_config"],
        }
        (root / "runtime_meta_info.json").write_text(json.dumps(runtime_meta, indent=2) + "\n")
        return ExportResult(work_dir=str(root), config_file=config_file, meta=runtime_meta)
