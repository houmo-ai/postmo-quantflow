from __future__ import annotations

import gc
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

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

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        from .hmonnx_export import (
            build_qwen4_exp_decode_sample_inputs,
            export_qwen4_exp_hmonnx,
            load_qwen4_exp_model_for_export,
            run_qwen4_exp_hmonnx_golden,
        )

        if quant_result is None:
            raise ValueError("quant_result can't be None")
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        model_dir = self._resolve_export_model_dir(quant_result)
        export_mode = workflow_config.export["model"].get("export_mode", "decode_validation")
        if export_mode == "serving":
            from .serving_workflow import export_qwen4_exp_serving_package

            model_cfg = workflow_config.export["model"]
            export_qwen4_exp_serving_package(
                model_dir=model_dir,
                output_dir=output_dir,
                model_config=model_cfg,
                mtp_source_dir=self.model_dir,
                device=device,
                device_map=model_cfg.get("device_map"),
            )
            config_file = Path(output_dir) / f"{workflow_config.name}.yaml"
            workflow_config.dump(str(config_file))
            meta = json.loads((Path(output_dir) / "golden_meta_info.json").read_text(encoding="utf-8"))
            return ExportResult(work_dir=str(output_dir), config_file=str(config_file), meta=meta)
        if export_mode != "decode_validation":
            raise ValueError(f"Unknown Qwen4-Exp export_mode: {export_mode!r}")
        work_dir = Path(output_dir)
        if work_dir.exists():
            raise FileExistsError(f"{work_dir} already exists")
        work_dir.mkdir(parents=True)
        config_file = work_dir / f"{workflow_config.name}.yaml"
        workflow_config.dump(str(config_file))

        model_cfg = workflow_config.export["model"]
        device_map = model_cfg.get("device_map")
        model = load_qwen4_exp_model_for_export(
            model_dir,
            device=device,
            device_map=device_map,
        )
        try:
            validation_prompt = str(model_cfg.get("validation_prompt") or "你好，请简要介绍一下你自己。")
            tokenizer, rendered_prompt, prefill_ids = self._build_validation_decode(
                model,
                model_dir,
                validation_prompt,
            )
            context_length = int(model_cfg.get("context_max_length", 262144))
            text_model, text_config, sample_inputs = build_qwen4_exp_decode_sample_inputs(
                model,
                prefill_ids=prefill_ids,
                context_length=context_length,
            )
            first_token_id = sample_inputs[3]
            export_result = export_qwen4_exp_hmonnx(
                text_model,
                work_dir / "text_decode",
                text_config=text_config,
                # Frontend/PTQ may execute in-place cache writers. Golden must
                # still start from the untouched prompt state built above.
                sample_inputs=tuple(value.detach().clone() for value in sample_inputs),
                quant_config=model_cfg.get("quant_scheme"),
                ple_quant_type=model_cfg.get("ple_quant_type", "sefp4"),
                batch_size=sample_inputs[0].shape[0],
                cache_length=context_length,
                quant_device=sample_inputs[0].device,
            )
            golden_dir = work_dir / "golden"
            golden_outputs = run_qwen4_exp_hmonnx_golden(
                export_result.hmonnx_path,
                sample_inputs,
                golden_dir,
                device=device,
            )
            # The exported wrapper already applies lm_head and returns logits.
            hmonnx_token_id = golden_outputs["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
            generated_token_ids = [int(first_token_id.item()), int(hmonnx_token_id.item())]
            conversation_report = work_dir / "qwen4_exp_hmonnx_conversation.json"
            conversation_report.write_text(
                json.dumps(
                    {
                        "prompt": validation_prompt,
                        "rendered_prompt": rendered_prompt,
                        "prompt_token_count": int(prefill_ids.shape[-1]),
                        "prompt_token_ids": prefill_ids.detach().cpu().tolist()[0],
                        "generated_token_ids": generated_token_ids,
                        "generated_text": tokenizer.decode(generated_token_ids, skip_special_tokens=True),
                        "native_seed_tokens": 1,
                        "hmonnx_decode_tokens": 1,
                        "prefill_state_source": "native_explicit_cache_wrapper",
                        "prefill_state_tensor_count": len(sample_inputs) - 4,
                        "cache_length": context_length,
                        "hmonnx_runtime_outputs": list(golden_outputs),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        meta = {
            "model_type": "qwen4_exp",
            "model_name": model_cfg.get("model_name", "qwen3_8_flash_next"),
            "source_model": model_dir,
            "hmonnx": export_result.hmonnx_path,
            "preflight": export_result.preflight_path,
            "io_manifest": export_result.io_manifest_path,
            "golden_dir": str(golden_dir),
            "golden_outputs": list(golden_outputs),
            "conversation_report": str(conversation_report),
            "precision_profile": (
                "W4A8 routed experts; W8A8 packed attention/shared; W8A8 graph default; "
                f"PLE {model_cfg.get('ple_quant_type', 'sefp4')} lookup"
            ),
        }
        meta_path = work_dir / "golden_meta_info.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return ExportResult(work_dir=str(work_dir), config_file=str(config_file), meta=meta)

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
                if serving_meta.get("visual_config"):
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

    @staticmethod
    @torch.no_grad()
    def _build_validation_decode(
        model: torch.nn.Module,
        model_dir: str,
        prompt: str,
    ) -> tuple[Any, str, torch.Tensor]:
        from transformers import AutoTokenizer

        from .modeling import extract_qwen4_exp_text_model

        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
        messages = [{"role": "user", "content": prompt}]
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        try:
            rendered_prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
        except TypeError:
            template_kwargs.pop("enable_thinking")
            rendered_prompt = tokenizer.apply_chat_template(messages, **template_kwargs)
        prefill_ids = tokenizer([rendered_prompt], return_tensors="pt", truncation=True).input_ids
        text_model = extract_qwen4_exp_text_model(model)
        embedding_device = text_model.embed_tokens.weight.device
        prefill_ids = prefill_ids.to(embedding_device)
        return tokenizer, rendered_prompt, prefill_ids

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
