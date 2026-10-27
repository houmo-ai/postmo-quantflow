"""Merak workflow for MiniCPM-V-4.7 Dense and MoE variants."""

from __future__ import annotations

import copy
import json
import shutil
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

        from .auto_classes import register_minicpmv47_auto_classes

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
        register_minicpmv47_auto_classes()
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

        from .dspark_config import MiniCPMV47ModelConfig as XHQwen3_5ModelConfig
        from xhmodel_merak.xh_llm.models.qwen3_5_moe.xh_qwen3_5_moe_config import (
            XHQwen3_5MoeModelConfig,
        )
        from xhquant.api import xhquant_init

        from .auto_classes import register_minicpmv47_auto_classes
        from .model import XHMiniCPMV47Model, XHMiniCPMV47MoeModel
        from .vision_export import export_vision_gears, export_vision_profile
        from xhmodel_merak.xh_llm.minicpm_v_series.vision_gears import image_token_gears_from_patches

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

        spec_mode = model_cfg.get("spec_decode_mode")
        prediction_k = model_cfg.get("les_prediction_k", 0)
        if type(prediction_k) is not int or prediction_k < 0:
            raise ValueError("les_prediction_k must be a nonnegative integer")
        if prediction_k:
            if model_type != _MOE_MODEL_TYPE or spec_mode:
                raise ValueError("LES requires MiniCPM MoE single-token ordinary decode")
            if (model_cfg.get("dynamic_prune") or {}).get("mode", "off") != "off":
                raise ValueError("LES cannot be combined with ACEP pruning")
            from .les import validate_prediction_k

            native_cfg = json.loads((Path(model_dir) / "config.json").read_text())
            validate_prediction_k(prediction_k, native_cfg["text_config"]["num_experts"])
        if spec_mode not in (None, "dflash"):
            raise ValueError("MiniCPM-V-4.7 supports spec_decode_mode=dflash")
        if spec_mode:
            from xhmodel_merak.xh_llm.models.minicpm_v_4_7.draft_checkpoint import validate_draft_target

            draft_dir = (model_cfg.get("dflash_config") or {}).get("hf_model")
            if not draft_dir:
                raise ValueError("MiniCPM DFlash requires export.model.dflash_config.hf_model")
            draft = validate_draft_target(draft_dir, model_dir)
            capacity = int(draft["dflash_config"]["block_size"])
            if draft.get("architectures") != ["Qwen3DSparkModel"]:
                capacity -= 1
            proposals = model_cfg.get("num_draft_tokens", 9)
            if type(proposals) is not int or not 1 <= proposals <= capacity:
                raise ValueError(f"MiniCPM num_draft_tokens must be between 1 and {capacity}")

        model_quant_scheme = model_cfg.get("quant_scheme")
        if model_quant_scheme is None:
            model_quant_scheme = {"quant_type": "w8a8h1_sefp"}
        if not isinstance(model_quant_scheme, Mapping):
            raise TypeError("export.model.quant_scheme must be a mapping")
        model_quant_scheme = copy.deepcopy(dict(model_quant_scheme))
        model_default_quant_type = str(model_quant_scheme.get("quant_type", "w8a8h1_sefp"))
        model_quant_scheme["quant_type"] = model_default_quant_type

        from xhmodel_merak.xh_llm.shared_head import target_head_bits

        head_bits = target_head_bits(model_quant_scheme)
        if head_bits < 8:
            from .quant_llm import validate_causal_head_checkpoint

            if quant_result.skipped or not quant_result.quanted_model_dir:
                raise ValueError("MiniCPM low-bit head export requires a calibrated quanted_model_dir")
            validate_causal_head_checkpoint(quant_result.quanted_model_dir, head_bits)

        register_minicpmv47_auto_classes()
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

        # Package tokenizer/processor resources so the artifact can move away
        # from the source checkpoint without retaining 35B model weights.
        resource_dir = work_dir / "hf_config"
        resource_dir.mkdir()
        for source in Path(model_dir).iterdir():
            if source.is_file() and source.suffix in {".json", ".jinja", ".txt", ".model", ".py"} and not source.name.endswith(".index.json"):
                shutil.copy2(source, resource_dir / source.name)

        meta: dict[str, Any] = {
            "schema_version": 1,
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "model_type": "MiniCPM-V-4.7",
            "merak_model_type": model_type,
            "hf_model": "hf_config",
            "target_device": target_device,
            "config": str(Path(config_file).relative_to(work_dir)),
            "components": [],
            "vision": {},
        }

        vision_cfg = export_cfg.get("vision") or {}
        if not isinstance(vision_cfg, Mapping):
            raise TypeError("MiniCPM export.vision must be a mapping")
        image_token_gears = vision_cfg.get("image_token_gears")
        patch_token_gears = vision_cfg.get("patch_token_gears")
        if sum(value is not None for value in (
            image_token_gears, patch_token_gears, vision_cfg.get("token_capacity")
        )) > 1:
            raise ValueError("Configure only one of patch_token_gears, image_token_gears or legacy token_capacity")
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
            mode_gears = (
                image_token_gears_from_patches(patch_token_gears, mode)
                if patch_token_gears is not None else image_token_gears
            )
            capacity_kwargs = (
                {"image_token_gears": mode_gears}
                if mode_gears is not None
                else {"token_capacity": int(vision_cfg.get("token_capacity", 1536))}
            )
            exporter = export_vision_gears if mode_gears is not None else export_vision_profile
            profile = exporter(
                model_dir=model_dir,
                work_dir=work_dir,
                target_device=target_device,
                quant_type=str(component_cfg.get("quant_type", "w8a8h1_sefp")),
                **capacity_kwargs,
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
                quant_dir = Path(quant_result.quanted_model_dir)
                llm_model_cfg["quant_weight"] = str(
                    quant_dir if (quant_dir / "model.safetensors.index.json").is_file()
                    else quant_dir / "pytorch_model.bin"
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
            if "les" in child_meta:
                meta["llm"]["les"] = copy.deepcopy(child_meta["les"])
            if spec_mode:
                spec = copy.deepcopy(child_meta["spec_decode"])
                for field in ("context_hmonnx", "context_decode_hmonnx", "decode_hmonnx"):
                    spec["draft"][field] = child_path(spec["draft"][field])
                for key in tuple(spec):
                    if key.endswith("_onnx"):
                        spec[key] = child_path(spec[key])
                if spec.get("verify_graphs"):
                    spec["verify_graphs"] = {length: child_path(path) for length, path in spec["verify_graphs"].items()}
                meta["llm"]["spec_decode"] = spec
                meta["components"].append("dflash")

        if spec_mode and "llm" not in meta:
            raise ValueError("MiniCPM DFlash requires the llm component")

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
