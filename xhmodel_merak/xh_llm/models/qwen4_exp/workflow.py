from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...workflows.base import BaseLLMWorkflow
from ...workflows.result import ExportResult, QuantResult
from ..qwen3_5.workflow import Qwen35Workflow
from .quant_adapter import quantize_with_qwen4_exp_api, resolve_quant_method_options


class Qwen4ExpWorkflow(Qwen35Workflow):
    """Qwen4-Exp quantization workflow.

    Qwen4-Exp uses its own explicit cache ABI because PLE/ngram state, QSA
    indexer keys, and hyper-connection streams are absent from Qwen3.5.
    """

    expected_model_config_cls_names = {"XHQwen4ExpModelConfig"}
    expected_model_cls_names = {"XHQwen4ExpModel"}
    family_label = "Qwen4-Exp"

    _COMMON_QUANT_KEYS = {
        "algorithm",
        "method",
        "output_format",
        "artifact_format",
        "bits",
        "group_size",
        "rotation",
        "language_quarot",
        "vision_quarot",
        "calibration",
        "runtime",
        "moe",
        "save_path",
        "output_dir",
        "batch_size",
        "nsamples",
        "seqlen",
        "iters",
        "disable_opt_rtn",
        "max_quant_layers",
        "seed",
        "preserve_mtp",
    }
    _SECTION_KEYS = {
        "calibration": {"dataset", "jsonl", "text_key", "nsamples", "seqlen"},
        "runtime": {
            "batch_size",
            "device",
            "device_map",
            "infer_device_map",
            "auto_forward_data_parallel",
            "dry_run",
            "extra_args",
            "trust_remote_code",
            "moe_routing_batch_size",
            "wait_for_submodule_finalizers",
            "gc_mode",
            "skip_eval",
        },
        "moe": {
            "attention_bits",
            "attn_bits",
            "linear_attention_bits",
            "shared_expert_bits",
            "expert_bits",
            "routing",
        },
    }

    def build_input_message(self, input_messages: Any) -> list[dict[str, Any]]:
        return super().build_input_message(input_messages)

    def export(self, quant_result, output_dir, device, config_overrides=None):
        return BaseLLMWorkflow.export(self, quant_result, output_dir, device, config_overrides)

    def _build_export_config(self, quant_result, workflow_config):
        config = BaseLLMWorkflow._build_export_config(self, quant_result, workflow_config)
        config["model"]["mtp_source_dir"] = self.model_dir
        return config

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any,
        **_: Any,
    ) -> str:
        meta_path = Path(export_result.work_dir) / "golden_meta_info.json"
        if meta_path.is_file():
            serving_meta = json.loads(meta_path.read_text())
            if "qwen4_exp_serving" in serving_meta:
                from .generation import run_text_generation

                messages = input_messages or "请用一句话介绍你自己。"
                if isinstance(messages, dict):
                    prompt = messages.get("prompt", messages.get("text", "请描述这张图片。"))
                    image = messages.get("image")
                elif isinstance(messages, str):
                    prompt, image = messages, None
                else:
                    raise ValueError("Qwen4 serving Golden expects text or a prompt/image mapping")
                components = ["prefill", "decode"]
                if serving_meta.get("qwen4_exp_vision") or serving_meta.get("visual_config"):
                    components.append("visual")
                if serving_meta.get("spec_decode", {}).get("mode") == "mtp":
                    components.append("mtp")
                golden_dir = Path(export_result.work_dir) / "golden" / f"request_{time.time_ns()}"
                from xhmodel_merak._environment import scoped_environment

                with scoped_environment({"ENABLE_HMINFERENCE_V2": "1", "XHQUANT_PACKED_W4": "1"}):
                    result = run_text_generation(
                        meta_path,
                        prompt=prompt,
                        image=image,
                        max_new_tokens=4,
                        device=device,
                        golden_components=components,
                        golden_dir=golden_dir,
                    )
                (golden_dir / "generation_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
                return str(golden_dir)
        meta = export_result.meta or {}
        golden_dir = meta.get("golden_dir")
        if not golden_dir or not Path(golden_dir).is_dir():
            raise FileNotFoundError("Qwen4-Exp export did not produce a Golden directory")
        conversation_report = meta.get("conversation_report")
        if not conversation_report or not Path(conversation_report).is_file():
            raise FileNotFoundError("Qwen4-Exp export did not produce a conversation validation report")
        return str(conversation_report)

    def _quant_autoround_api(
        self,
        output_dir: str,
        device: str,
        quant_cfg: Mapping[str, Any],
        export_model_cfg: Mapping[str, Any],
    ) -> QuantResult:
        self._validate_qwen4_exp_quant_config(quant_cfg)
        return quantize_with_qwen4_exp_api(
            model_dir=self.model_dir,
            output_dir=output_dir,
            device=device,
            quant_cfg=quant_cfg,
            export_model_cfg=export_model_cfg,
            workflow_seed=int(getattr(self, "seed", 1024)),
        )

    def _quant_gptqmodel_api(
        self,
        output_dir: str,
        device: str,
        quant_cfg: Mapping[str, Any],
        export_model_cfg: Mapping[str, Any],
    ) -> QuantResult:
        return self._quant_autoround_api(output_dir, device, quant_cfg, export_model_cfg)

    @classmethod
    def _validate_qwen4_exp_quant_config(cls, quant_cfg: Mapping[str, Any]) -> None:
        resolve_quant_method_options(quant_cfg)
        unknown = set(quant_cfg) - cls._COMMON_QUANT_KEYS
        if unknown:
            raise ValueError(f"Qwen4-Exp config contains keys not consumed by the qwen4_exp recipe: {sorted(unknown)}")
        for section, allowed in cls._SECTION_KEYS.items():
            value = quant_cfg.get(section)
            if value is None:
                continue
            if not isinstance(value, Mapping):
                raise TypeError(f"Qwen4-Exp quant.{section} must be a mapping")
            unknown = set(value) - allowed
            if unknown:
                raise ValueError(
                    f"Qwen4-Exp quant.{section} contains keys not consumed by the qwen4_exp recipe: {sorted(unknown)}"
                )
        if quant_cfg.get("rotation") not in (None, False):
            raise ValueError("Qwen4-Exp language/global QuaRot is not supported; use vision_quarot only")
        if int(quant_cfg.get("bits", 8)) != 8:
            raise ValueError("Qwen4-Exp quant profile requires base bits=8")
        if int(quant_cfg.get("group_size", 64)) != 64:
            raise ValueError("Qwen4-Exp quant profile requires group_size=64")
        moe = dict(quant_cfg.get("moe") or {})
        profile = (
            int(moe.get("attention_bits", moe.get("attn_bits", 8))),
            int(moe.get("linear_attention_bits", 8)),
            int(moe.get("shared_expert_bits", 8)),
            int(moe.get("expert_bits", 4)),
        )
        if profile != (8, 8, 8, 4):
            raise ValueError(
                "Qwen4-Exp quant profile requires attention/linear-attention/shared W8 and routed experts W4; "
                f"got attention={profile[0]}, linear_attention={profile[1]}, shared={profile[2]}, expert={profile[3]}"
            )


__all__ = ["Qwen4ExpWorkflow"]
