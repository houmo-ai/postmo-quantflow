from __future__ import annotations

from typing import Any

from xhmodel_merak.xh_llm.types import TextLLMModelConfig


class XHMossEncoderConfig:
    def __init__(
        self,
        *,
        model_name: str = "moss_transcribe_diarize_encode",
        chip_arch: str = "XH2a",
        hf_model: Any = None,
        encode_config: dict[str, Any] | None = None,
        convert_hmonnx: bool = True,
        quant_type: str = "w8a8_sefp",
        **kwargs: Any,
    ):
        del kwargs
        self.model_name = model_name
        self.chip_arch = chip_arch
        self.hf_model = hf_model
        self.encode_config = dict(encode_config or {})
        self.convert_hmonnx = bool(convert_hmonnx)
        self.quant_type = str(quant_type)


class XHMossTranscribeDiarizeConfig(TextLLMModelConfig):
    def __init__(
        self,
        *,
        model_name: str,
        chip_arch: str = "XH2a",
        model_type: str = "XHMossTranscribeDiarizeModel",
        hf_model: Any = None,
        context_max_length: int = 2048,
        prefill_chunk_length: int = 512,
        max_pe_length: int = 262144,
        encode_config: dict[str, Any] | None = None,
        flash_attention: dict[str, Any] | None = None,
        wrap_cfg: dict[str, Any] | None = None,
        export_cfg: dict[str, Any] | None = None,
        frontend_type: str | None = None,
        convert_hmonnx: bool = True,
        quant_type: str = "w8a8_sefp",
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name,
            chip_arch=chip_arch,
            model_type=model_type,
            hf_model=hf_model if isinstance(hf_model, str) or hf_model is None else None,
            context_max_length=context_max_length,
            prefill_chunk_length=prefill_chunk_length,
            max_pe_length=max_pe_length,
            **kwargs,
        )
        self.upstream_module = hf_model if hf_model is not None and not isinstance(hf_model, str) else None
        self.encode_config = dict(encode_config or {})
        self.flash_attention = dict(flash_attention or {})
        self.wrap_cfg = dict(wrap_cfg or {})
        self.export_cfg = dict(export_cfg or {})
        self.frontend_type = frontend_type
        self.convert_hmonnx = bool(convert_hmonnx)
        self.quant_type = str(quant_type)

    def encode_model_config(self) -> XHMossEncoderConfig:
        return XHMossEncoderConfig(
            model_name=f"{self.model_name}_encode",
            chip_arch=self.chip_arch,
            hf_model=self.upstream_module or self.hf_model,
            encode_config=self.encode_config,
            convert_hmonnx=bool(self.encode_config.get("convert_hmonnx", self.convert_hmonnx)),
            quant_type=str(self.encode_config.get("quant_type") or self.quant_type),
        )
