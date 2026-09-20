"""Non-master sibling that publishes the Encoder component."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from xhmodel_merak.xh_llm.builder import register_llm_model

from .configuration import XHMossEncoderConfig
from .encoder import (
    EncoderComponent,
    encoder_component_from_hf,
    export_encoder_component,
    load_encoder_component,
)


@register_llm_model("XHMossTranscribeDiarizeEncoder", master=False)
class XHMossTranscribeDiarizeEncoder:
    CONFIG_CLS = XHMossEncoderConfig

    def __init__(self, config: XHMossEncoderConfig):
        self.config = config
        self._component: EncoderComponent | None = None

    def get_component(self, hf_model: Any | None = None) -> EncoderComponent:
        if self._component is not None:
            return self._component
        if hf_model is not None:
            self._component = (
                hf_model if isinstance(hf_model, EncoderComponent) else encoder_component_from_hf(hf_model)
            )
            return self._component
        hf_model_dir = self.config.hf_model
        if isinstance(hf_model_dir, EncoderComponent):
            self._component = hf_model_dir
            return self._component
        if not hf_model_dir:
            raise FileNotFoundError("MOSS encoder export requires a checkpoint or an Official-shaped module")
        self._component = load_encoder_component(hf_model_dir)
        return self._component

    def export_hmonnx(self, output_dir: str, hf_model: Any | None = None) -> dict[str, Any]:
        encoder_cfg = dict(self.config.encode_config or {})
        component = self.get_component(hf_model)
        return export_encoder_component(
            component,
            Path(output_dir),
            target_device=str(self.config.chip_arch),
            quant_type=str(encoder_cfg.get("quant_type") or self.config.quant_type),
            convert_hmonnx=bool(encoder_cfg.get("convert_hmonnx", self.config.convert_hmonnx)),
            model_name=str(self.config.model_name),
        )


__all__ = ["XHMossTranscribeDiarizeEncoder"]
