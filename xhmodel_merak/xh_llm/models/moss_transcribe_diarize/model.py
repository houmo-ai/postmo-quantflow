from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from xhmodel_merak.xh_llm.builder import register_llm_model
from xhmodel_merak.xh_llm.text_llm_model import TextLLMModel

from .configuration import XHMossTranscribeDiarizeConfig
from .encoder_model import XHMossTranscribeDiarizeEncoder


def _relocate_meta(meta: dict[str, Any], component: str) -> dict[str, Any]:
    result = dict(meta)
    for key in ("onnx_file", "hmonnx_file"):
        if key in result:
            result[key] = f"{component}/{result[key]}"
    return result


def _require_decoder_geometry(prefill_length: int, cache_length: int, max_pe_length: int) -> None:
    if not (0 < prefill_length < cache_length <= max_pe_length):
        raise ValueError("Require 0 < prefill_chunk_length < context_max_length <= max_pe_length")


@register_llm_model("XHMossTranscribeDiarizeModel")
class XHMossTranscribeDiarizeModel(TextLLMModel):
    CONFIG_CLS = XHMossTranscribeDiarizeConfig
    WORKFLOW_CLS = (
        "xhmodel_merak.xh_llm.models.moss_transcribe_diarize.workflow:MossTranscribeDiarizeWorkflow"
    )

    def __init__(self, config: XHMossTranscribeDiarizeConfig):
        super().__init__(config)
        self.config = config
        self.encoder = XHMossTranscribeDiarizeEncoder(config.encode_model_config())
        self.wrap_cfg["flash_attention"] = dict(config.flash_attention or {})

    def get_native_model(self) -> Any:
        if self.config.upstream_module is not None:
            return self.config.upstream_module
        hf_model_dir = self.hf_model_dir or self.config.hf_model
        if not hf_model_dir:
            raise FileNotFoundError("MOSS export requires a user checkpoint; Reduced fixture is not an export fallback")
        from transformers import AutoModelForCausalLM
        import torch

        # AutoModel maps to the backbone without lm_head; CausalLM is the tied generation class.
        return AutoModelForCausalLM.from_pretrained(
            str(hf_model_dir), trust_remote_code=True, torch_dtype=torch.float32
        )

    def init_wrap_model(self, hf_model: Any) -> Any:
        from ._llm_model_impl import (
            attach_lm_head,
            language_model_from_hf,
            register_wrap_modules,
            restore_wrap_modules,
        )

        # Attach before registering wraps: register_wrap_modules picks the
        # lm_head-aware backbone only when the projection is already present.
        # Wrap the language model rather than the CausalLM root so the fp16 cast
        # in init_wrap_model leaves the fp32 Whisper encoder untouched.
        attach_lm_head(hf_model)
        replacements = register_wrap_modules(hf_model)
        try:
            return super().init_wrap_model(language_model_from_hf(hf_model))
        finally:
            restore_wrap_modules(replacements)

    def _get_language_model(self, hf_model: Any) -> Any:
        from ._llm_model_impl import language_model_from_hf

        return language_model_from_hf(hf_model)

    def get_export_cfg(self) -> dict[str, list[str]]:
        """Keep the base ``logits`` output name, which lm_head now makes accurate.

        The base class always names the first output ``logits``. Earlier MOSS
        graphs stopped at the backbone, so that name described a
        ``[1, 1, hidden]`` tensor; with lm_head in the graph it is the real
        vocabulary distribution.
        """
        return super().get_export_cfg()


    def get_export_info(self, output_dir):
        """Keep asr graphs under the workflow work_dir, plus hmquant_* serving metadata."""
        from datetime import datetime

        from xhmodel_merak.xh_llm.types import ExportData

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        serving_name = f"hmquant_{str(self.config.model_name).lower()}"
        (Path(output_dir) / serving_name).mkdir(parents=True, exist_ok=True)
        export_data = ExportData()
        export_data.exported_dir = str(Path(output_dir).resolve())
        export_data.meta = self._moss_export_metadata(output_dir)
        export_data.model_name = serving_name
        export_data.str_datetime = datetime.now().strftime("%Y%m%d")
        return export_data

    def _moss_export_metadata(self, output_dir: str):
        from datetime import datetime

        hf = self.config.hf_model
        if isinstance(hf, str) and hf:
            try:
                return self.create_export_metadata(output_dir)
            except (OSError, TypeError, ValueError):
                pass
        meta_info = self.get_export_metadata_cls()()
        meta_info.create_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        meta_info.model_config = self.config
        return meta_info

    def export_hmonnx(self, output_dir: str) -> dict[str, Any]:
        cfg = self.config
        work_dir = Path(output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        prefill_length = int(cfg.prefill_chunk_length)
        cache_length = int(cfg.context_max_length)
        max_pe_length = int(cfg.max_pe_length)
        _require_decoder_geometry(prefill_length, cache_length, max_pe_length)

        native = self.get_native_model()
        return self._export_text_llm(native, output_dir, prefill_length, cache_length, max_pe_length)

    def _export_text_llm(
        self,
        native: Any,
        output_dir: str,
        prefill_length: int,
        cache_length: int,
        max_pe_length: int,
    ) -> dict[str, Any]:
        from xhmodel_merak.xh_llm.types import LLMModelState

        if self._state != LLMModelState.QUANTED_ALIGNED:
            self.to_quanted_aligned()
        self._quanted_model.fixed()
        exported_info = self.get_export_info(output_dir)
        encoder_meta = self.encoder.export_hmonnx(
            str(Path(exported_info.exported_dir) / "encode"),
            hf_model=native,
        )
        self._export_hmonnx(exported_info)
        meta = self._stitch_asr_meta(
            encoder_meta,
            exported_info.meta,
            Path(exported_info.exported_dir),
            prefill_length,
            cache_length,
            max_pe_length,
        )
        self._save_eager_weights(native, Path(exported_info.exported_dir), meta)
        return meta

    def _save_eager_weights(self, native: Any, work_dir: Path, decoder_meta: dict[str, Any]) -> None:
        import torch

        from ._llm_model_impl import language_model_from_hf

        language = language_model_from_hf(native)
        lm_head = getattr(native, "lm_head", None)
        blob = {
            "encode": self.encoder.get_component(native).state_dict(),
            "language_model": language.state_dict(),
            "language_config": language.config.to_dict() if hasattr(language.config, "to_dict") else dict(language.config),
        }
        if lm_head is not None:
            from .transcribe import normalize_lm_head_state

            lm_state = normalize_lm_head_state(lm_head.state_dict())
            blob["lm_head"] = lm_state
            torch.save(lm_state, work_dir / "lm_head.pt")
            decoder_meta["lm_head_file"] = "lm_head.pt"
        torch.save(blob, work_dir / "weights.pt")
        decoder_meta["weights_file"] = "weights.pt"
        (work_dir / "export_meta_info.json").write_text(
            json.dumps(decoder_meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    def _stitch_asr_meta(
        self,
        encoder_meta: dict[str, Any],
        llm_meta: Any,
        work_dir: Path,
        prefill_length: int,
        cache_length: int,
        max_pe_length: int,
    ) -> dict[str, Any]:
        relocated_encoder = _relocate_meta(encoder_meta, "encode")
        flash_attention = dict(self.config.flash_attention or {})
        fa_enabled = bool(flash_attention.get("enable"))
        geometry = {
            "prefill_length": prefill_length,
            "cache_length": cache_length,
            "rope_max_positions": max_pe_length,
            "rope_mode": "fp32_table_integer_index",
            # XH2a lowering always emits per-layer CacheTensor inputs, FA on or off.
            "cache_mode": "inplace",
            # Graphs stay FA + inplace. Greedy runtime converts FA graphs to PA at
            # load; 2k FA-off stays on CacheTensor.
            "enable_page_attention": False,
        }
        encoder_hmonnx = relocated_encoder.get("hmonnx_file")
        prefill_hmonnx = getattr(llm_meta, "prefill_hmonnx", None)
        decode_hmonnx = getattr(llm_meta, "decode_hmonnx", None)
        if not encoder_hmonnx or not prefill_hmonnx or not decode_hmonnx:
            raise RuntimeError(
                "MOSS export requires published HMONNX graphs for encode, prefill, and decode"
            )
        prefill = {**geometry, "hmonnx_file": prefill_hmonnx}
        decode = {**geometry, "hmonnx_file": decode_hmonnx}
        quant_type = "w8a8_sefp"
        encode_config = {
            "enabled": True,
            "quant_type": str(self.config.encode_config.get("quant_type") or self.config.quant_type),
            "max_audio_length": self.config.encode_config.get("max_audio_length", 3000),
            "hmonnx_file": relocated_encoder.get("hmonnx_file"),
        }
        if relocated_encoder.get("onnx_file"):
            encode_config["onnx_file"] = relocated_encoder["onnx_file"]
        meta = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_device": str(self.config.chip_arch),
            "chip_arch": str(self.config.chip_arch),
            "quant_type": quant_type,
            "model_config": {
                "model_type": self.config.model_type,
                "chip_arch": str(self.config.chip_arch),
                "context_max_length": cache_length,
                "prefill_chunk_length": prefill_length,
                "max_pe_length": max_pe_length,
                "flash_attention": flash_attention,
                "enable_page_attention": fa_enabled,
            },
            "encode_config": encode_config,
            "components": {
                "encode": relocated_encoder,
                "prefill_decode": {"prefill": prefill, "decode": decode},
            },
            "prefill_hmonnx": prefill.get("hmonnx_file"),
            "decode_hmonnx": decode.get("hmonnx_file"),
        }
        (work_dir / "export_meta_info.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        serving = work_dir / f"hmquant_{self.config.model_name}"
        serving.mkdir(parents=True, exist_ok=True)
        (serving / "golden_meta_info.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return meta


__all__ = ["XHMossTranscribeDiarizeModel"]
