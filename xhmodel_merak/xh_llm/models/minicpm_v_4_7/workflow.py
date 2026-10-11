"""Merak workflow for MiniCPM-V-4.7 Dense and MoE variants."""

from __future__ import annotations

import copy
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xhmodel_merak.xh_llm.workflows.base import BaseLLMWorkflow
from xhmodel_merak.xh_llm.workflows.result import ExportResult, QuantResult


_DENSE_MODEL_TYPE = "MiniCPMV47ForConditionalGeneration"
_MOE_MODEL_TYPE = "MiniCPMV47MoeForConditionalGeneration"


class MiniCPMV47Workflow(BaseLLMWorkflow):
    """Export the two MiniCPM Vision profiles and Qwen3.5 text graphs."""

    SUPPORTED_COMPONENTS = {"vision_4x", "vision_16x", "llm"}

    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        if workflow_config.quant is None:
            return QuantResult(raw_model_dir=self.model_dir, skipped=True)

        quant_cfg = workflow_config.quant
        if not isinstance(quant_cfg, Mapping):
            raise TypeError("MiniCPM-V-4.7 quant must be a mapping")
        algorithm = str(quant_cfg.get("algorithm", "gptqmodel")).lower()
        method = str(quant_cfg.get("method", "gptq")).lower().replace("-", "_")
        if algorithm != "gptqmodel":
            raise NotImplementedError(
                "MiniCPMV47Workflow currently supports quant.algorithm=gptqmodel; "
                f"got algorithm={algorithm!r}"
            )
        if method in {"gptq", "gptqmodel"}:
            from .quant_llm import quantize_minicpm_v47_llm_gptq

            dequant_dir = quantize_minicpm_v47_llm_gptq(
                model_dir=self.model_dir,
                output_dir=output_dir,
                quant_cfg=quant_cfg,
                device=device,
            )
        elif method in {"autoround", "auto_round"}:
            from .quant_llm import quantize_minicpm_v47_llm_autoround

            dequant_dir = quantize_minicpm_v47_llm_autoround(
                model_dir=self.model_dir,
                output_dir=output_dir,
                quant_cfg=quant_cfg,
                device=device,
            )
        else:
            raise NotImplementedError(
                "MiniCPMV47Workflow supports Dense GPTQ (method=gptq) and MoE "
                "AutoRound (method=autoround); "
                f"got algorithm={algorithm!r}, method={method!r}"
            )
        return QuantResult(raw_model_dir=self.model_dir, quanted_model_dir=dequant_dir)

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        import torch

        from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import (
            XHQwen3_5ModelConfig,
        )
        from xhmodel_merak.xh_llm.models.qwen3_5_moe.xh_qwen3_5_moe_config import (
            XHQwen3_5MoeModelConfig,
        )
        from xhquant.api import xhquant_init

        from .model import XHMiniCPMV47Model, XHMiniCPMV47MoeModel
        from .vision_export import export_vision_profile

        workflow_config = self.workflow_config.with_overrides(config_overrides)
        # Host (vision + llm) always loads from the raw MiniCPM directory;
        # GPTQ/AutoRound only replaces the LLM backbone via quant_weight.
        model_dir = self._resolve_export_model_dir(quant_result)
        if not quant_result.skipped:
            model_dir = quant_result.raw_model_dir
        export_cfg = workflow_config.build_export_dict()
        model_cfg = export_cfg.get("model") or {}
        if not isinstance(model_cfg, Mapping):
            raise TypeError("MiniCPM export.model must be a mapping")
        target_device = str(model_cfg["chip_arch"])
        if target_device.lower() != "xh2a":
            raise ValueError("MiniCPM-V-4.7 has only been validated for export.model.chip_arch=XH2a")

        model_type = str(model_cfg.get("model_type", _DENSE_MODEL_TYPE))
        if model_type == _DENSE_MODEL_TYPE:
            model_cls = XHMiniCPMV47Model
            config_cls = XHQwen3_5ModelConfig
            default_model_name = "minicpm_v_4_7_qwen3_5_1b"
        elif model_type == _MOE_MODEL_TYPE:
            model_cls = XHMiniCPMV47MoeModel
            config_cls = XHQwen3_5MoeModelConfig
            default_model_name = "minicpm_v_4_7_qwen3_5_moe_35b_a3b"
        else:
            raise ValueError(
                "MiniCPM-V-4.7 export.model.model_type must be "
                f"{_DENSE_MODEL_TYPE!r} or {_MOE_MODEL_TYPE!r}, got {model_type!r}"
            )

        model_quant_scheme = model_cfg.get("quant_scheme")
        if model_quant_scheme is None:
            model_quant_scheme = {"quant_type": "w8a8h1_sefp"}
        if not isinstance(model_quant_scheme, Mapping):
            raise TypeError("export.model.quant_scheme must be a mapping")
        model_quant_scheme = copy.deepcopy(dict(model_quant_scheme))
        model_default_quant_type = str(model_quant_scheme.get("quant_type", "w8a8h1_sefp"))
        model_quant_scheme["quant_type"] = model_default_quant_type

        work_dir = Path(output_dir).resolve()
        if work_dir.exists() and any(work_dir.iterdir()):
            raise FileExistsError(
                f"MiniCPM export directory is not empty: {work_dir}. "
                "Remove it explicitly or use the example's --overwrite option."
            )
        work_dir.mkdir(parents=True, exist_ok=True)
        xhquant_init(work_dir / "convert.log", debug=self.debug)
        torch.manual_seed(self.seed)
        config_file = workflow_config.dump(str(work_dir / f"{workflow_config.name}.yaml"))

        components_cfg = export_cfg.get("components") or {}
        if not isinstance(components_cfg, Mapping):
            raise TypeError("MiniCPM export.components must be a mapping")
        unsupported = sorted(set(components_cfg) - self.SUPPORTED_COMPONENTS)
        if unsupported:
            raise ValueError(f"Unsupported MiniCPM component(s): {unsupported}")

        meta: dict[str, Any] = {
            "schema_version": 1,
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "model_type": "MiniCPM-V-4.7",
            "merak_model_type": model_type,
            "hf_model": model_dir,
            "target_device": target_device,
            "config": str(Path(config_file).relative_to(work_dir)),
            "components": [],
            "vision": {},
        }

        vision_cfg = export_cfg.get("vision") or {}
        if not isinstance(vision_cfg, Mapping):
            raise TypeError("MiniCPM export.vision must be a mapping")
        token_capacity = int(vision_cfg.get("token_capacity", 1536))
        keep_onnx = bool(vision_cfg.get("keep_onnx", False))
        for component_name, mode in (("vision_4x", "4x"), ("vision_16x", "16x")):
            component_cfg = components_cfg.get(component_name, False)
            if not _is_enabled(component_cfg):
                continue
            if not isinstance(component_cfg, Mapping):
                raise TypeError(f"export.components.{component_name} must be a mapping or false")
            validation_target = _pair(
                component_cfg.get("validation_target"),
                field=f"export.components.{component_name}.validation_target",
            )
            profile = export_vision_profile(
                model_dir=model_dir,
                work_dir=work_dir,
                target_device=target_device,
                quant_type=str(component_cfg.get("quant_type", "w8a8h1_sefp")),
                token_capacity=token_capacity,
                downsample_mode=mode,
                validation_target=validation_target,
                device=device,
                keep_onnx=keep_onnx,
            )
            meta["vision"][mode] = profile
            meta["components"].append(component_name)

        llm_component_cfg = components_cfg.get("llm", False)
        if _is_enabled(llm_component_cfg):
            if not isinstance(llm_component_cfg, Mapping):
                raise TypeError("export.components.llm must be a mapping or false")
            llm_model_cfg = copy.deepcopy(dict(model_cfg))
            # Prefer the full export.model.quant_scheme (nodes/ops) when present;
            # fall back to the component-level quant_type used by W8A8 configs.
            component_quant_type = llm_component_cfg.get("quant_type")
            if component_quant_type is not None and "quant_scheme" not in model_cfg:
                llm_quant_type = str(component_quant_type)
                llm_quant_scheme = {"quant_type": llm_quant_type}
            else:
                llm_quant_type = model_default_quant_type
                llm_quant_scheme = copy.deepcopy(model_quant_scheme)
                if component_quant_type is not None:
                    # Keep component override for naming/meta when explicitly set.
                    llm_quant_type = str(component_quant_type)
                    llm_quant_scheme["quant_type"] = llm_quant_type
            configured_name = str(llm_model_cfg.get("model_name", default_model_name))
            llm_model_cfg.update(
                {
                    "hf_model": model_dir,
                    "chip_arch": target_device,
                    "model_name": f"{configured_name}_{target_device}_{llm_quant_type}",
                    "visual_config": None,
                    "quant_scheme": llm_quant_scheme,
                }
            )
            if not quant_result.skipped and quant_result.quanted_model_dir:
                llm_model_cfg["quant_weight"] = str(
                    Path(quant_result.quanted_model_dir) / "pytorch_model.bin"
                )
            llm_output_dir = work_dir / "llm"
            llm_model = model_cls(config_cls(**llm_model_cfg))
            llm_model.export_hmonnx(str(llm_output_dir))
            artifact_dirs = sorted(llm_output_dir.glob("hmquant_*"))
            if len(artifact_dirs) != 1:
                raise RuntimeError(
                    "Expected exactly one MiniCPM LLM artifact directory, "
                    f"found {len(artifact_dirs)} under {llm_output_dir}"
                )
            artifact_dir = artifact_dirs[0]
            child_meta_file = artifact_dir / "golden_meta_info.json"
            child_meta = json.loads(child_meta_file.read_text(encoding="utf-8"))

            def child_path(value: str) -> str:
                return str((artifact_dir / value).relative_to(work_dir))

            meta["llm"] = {
                "quant_type": llm_quant_type,
                "artifact_dir": str(artifact_dir.relative_to(work_dir)),
                "metadata": str(child_meta_file.relative_to(work_dir)),
                "prefill_hmonnx": child_path(child_meta["prefill_hmonnx"]),
                "decode_hmonnx": child_path(child_meta["decode_hmonnx"]),
                "quant_embedding": child_path(child_meta["quant_embedding"]),
                "hf_config": child_path(child_meta["hf_config"]),
                "context_max_length": int(llm_model_cfg["context_max_length"]),
                "prefill_chunk_length": int(llm_model_cfg["prefill_chunk_length"]),
            }
            meta["components"].append("llm")

        if not meta["components"]:
            raise ValueError("MiniCPM export must enable at least one component")

        meta_file = work_dir / "export_meta_info.json"
        meta_file.write_text(
            json.dumps(meta, ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
        )
        return ExportResult(
            work_dir=str(work_dir),
            config_file=config_file,
            meta=meta,
        )

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any = None,
    ) -> str:
        from .inference import dump_minicpm_v47_golden

        return dump_minicpm_v47_golden(
            work_dir=Path(export_result.work_dir),
            device=device,
            input_messages=input_messages,
        )


def _is_enabled(config: Any) -> bool:
    return config is not False and (not isinstance(config, Mapping) or bool(config.get("enabled", True)))


def _pair(value: Any, *, field: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field} must contain [height, width]")
    return int(value[0]), int(value[1])


__all__ = ["MiniCPMV47Workflow"]
