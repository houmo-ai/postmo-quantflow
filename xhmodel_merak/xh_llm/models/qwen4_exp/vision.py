"""Qwen4-Exp visual tower using the shared Qwen static patch-token ABI.

Qwen4-Exp and Qwen3.5 have identical visual patch embedding, learned-position
interpolation, two-dimensional RoPE and merger semantics. Register the native
Qwen4 classes against those existing graph rewrites instead of maintaining a
second implementation of their head padding and quantizable attention.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoProcessor
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpForConditionalGeneration,
    Qwen4ExpVisionAttention,
    Qwen4ExpVisionBlock,
    Qwen4ExpVisionModel,
    Qwen4ExpVisionPatchEmbed,
    Qwen4ExpVisionPatchMerger,
)

from ...builder import register_llm_model
from ...hmonnx.hmonnx_model import HMONNXModel
from ...register import XHLLM_TRACEABLE_MODULES
from ...types import VisualModelMeta
from ..qwen3_5._vision_model_impl import (
    _Qwen3_5VisionAttention,
    _Qwen3_5VisionBlock,
    _Qwen3_5VisionModel,
    _Qwen3_5VisionPatchEmbed,
    _Qwen3_5VisionPatchMerger,
)
from ..qwen3_5.qwen3_5_hmonnx_inference import VisualTokenGearHMONNXModel
from ..qwen3_5.qwen3_5_vision_model import XHQwen3_5VisionModel
from ..qwen3_5.xh_qwen3_5_config import XHQwen3_5_VisualConfig


XHLLM_TRACEABLE_MODULES.register_module({Qwen4ExpVisionAttention: "Qwen4ExpVisionAttention"}, _Qwen3_5VisionAttention)
XHLLM_TRACEABLE_MODULES.register_module({Qwen4ExpVisionBlock: "Qwen4ExpVisionBlock"}, _Qwen3_5VisionBlock)
XHLLM_TRACEABLE_MODULES.register_module({Qwen4ExpVisionModel: "Qwen4ExpVisionModel"}, _Qwen3_5VisionModel)
XHLLM_TRACEABLE_MODULES.register_module(
    {Qwen4ExpVisionPatchEmbed: "Qwen4ExpVisionPatchEmbed"}, _Qwen3_5VisionPatchEmbed
)
XHLLM_TRACEABLE_MODULES.register_module(
    {Qwen4ExpVisionPatchMerger: "Qwen4ExpVisionPatchMerger"}, _Qwen3_5VisionPatchMerger
)


@register_llm_model("Qwen4ExpForConditionalGeneration_visual", master=False)
class XHQwen4ExpVisionModel(XHQwen3_5VisionModel):
    """Materialize only native ``model.visual.*`` weights, then export gears."""

    transformers_min_version = "5.16.0"
    HF_MODEL_CLS = Qwen4ExpForConditionalGeneration
    VISUAL_HF_MODEL_CLS = Qwen4ExpVisionModel

    def __init__(self, config):
        config.model_type = "Qwen4ExpForConditionalGeneration_visual"
        super().__init__(config)

    def get_tf_processor(self):
        # The published Flash Next checkpoint explicitly selects Qwen3VLProcessor.
        # Loading it from its own resources preserves its tokenizer/template and
        # image normalization rather than substituting Qwen3.5 defaults.
        return AutoProcessor.from_pretrained(self.hf_model_dir)

    def export_hmonnx(self, output_dir, *, write_standalone_metadata=True):
        from .vision_export import export_qwen4_exp_vision_components

        data = export_qwen4_exp_vision_components(
            self.hf_model_dir,
            output_dir,
            visual_config=self.config,
            quant_scheme=self.get_quant_cfg(),
            device=self.device,
            dtype=self.dtype,
        )
        metadata = VisualModelMeta()
        for name, value in data.items():
            setattr(metadata, name, value)
        if write_standalone_metadata:
            (Path(output_dir) / "visual_meta_info.json").write_text(json.dumps(metadata.to_dict(), indent=2) + "\n")
        return metadata


class _CapturedVisualGraph:
    """Keep capture scoped to the real SDK session of one component."""

    def __init__(self, graph, owner, name):
        self.graph, self.owner, self.name = graph, owner, name

    def __getattr__(self, name):
        return getattr(self.graph, name)

    @property
    def enable_golden(self):
        return self.graph.enable_golden

    @enable_golden.setter
    def enable_golden(self, enabled):
        self.graph.enable_golden = enabled
        self.graph.hmonnx_session.interpreter._enable_golden = enabled

    def forward(self, *args):
        capture = self.owner.capture
        scope = capture.step("visual", self.graph.hmonnx_session, graph_name=self.name) if capture else nullcontext()
        with scope:
            return self.graph.forward(*args)


class _ComponentGear:
    def __init__(self, components):
        self.components = components

    @property
    def device(self):
        return self.components["stem"].device

    def to(self, device):
        for graph in self.components.values():
            graph.to(device)
        return self

    def _set_dtype(self, dtype):
        for graph in self.components.values():
            graph._set_dtype(dtype)

    @property
    def enable_golden(self):
        return all(graph.enable_golden for graph in self.components.values())

    @enable_golden.setter
    def enable_golden(self, enabled):
        for graph in self.components.values():
            graph.enable_golden = enabled

    def update_step(self):
        for graph in self.components.values():
            graph.update_step()

    def forward(self, pixel_values, position_ids, position_weights, rotary_position_ids, attention_mask):
        hidden, cos, sin = self.components["stem"].forward(
            pixel_values,
            position_ids,
            position_weights,
            rotary_position_ids,
        )
        for name, graph in self.components.items():
            if name.startswith("block_"):
                hidden = graph.forward(hidden, cos, sin, attention_mask)
        return self.components["merger"].forward(hidden)


class _Qwen4VisionCapture:
    @property
    def max_image_tokens(self):
        return max(self.gears)

    def capture_all(self, capture):
        capture.required_graphs.update(self._capture_graph_names)
        previous = self.capture
        self.capture = capture
        try:
            return self.run_all_gears()
        finally:
            self.capture = previous


class Qwen4ComponentVisionEncoder(_Qwen4VisionCapture, VisualTokenGearHMONNXModel):
    """Retain component sessions and reuse the established patch/video routing."""

    def __init__(self, metadata, root, *, device, capture=None):
        from xhmodel_merak.exec_backends import MultiHMONNXLoader

        if not HMONNXModel._is_inference_v2_env_enabled():
            raise RuntimeError("Qwen4 component vision requires ENABLE_HMINFERENCE_V2=1")
        self.visual_meta = SimpleNamespace(**metadata)
        self.gears = tuple(metadata["image_token_gears"])
        self.capture = capture
        files = {
            f"visual_m{gear['image_token_capacity']}_{part['name']}": str(Path(root) / part["hmonnx"])
            for gear in metadata["gears"]
            for part in gear["components"]
        }
        self._capture_graph_names = tuple(files)
        self._loader = MultiHMONNXLoader(files)
        self.models = {}
        for gear in metadata["gears"]:
            capacity = gear["image_token_capacity"]
            components = {}
            for part in gear["components"]:
                name = f"visual_m{capacity}_{part['name']}"
                graph = HMONNXModel(
                    files[name], onnx_graph=self._loader.graphs[name], device_map=[device], enable_golden=True
                )
                graph.enable_golden = False
                graph.hmonnx_session.interpreter._enable_golden = False
                components[part["name"]] = _CapturedVisualGraph(graph, self, name)
            self.models[capacity] = _ComponentGear(components)
        self._device = self.models[self.gears[-1]].device
        self._dtype = torch.float16
        self._golden_capture_capable = True
        self._enable_golden = False

    @staticmethod
    def _session_input_info(model, name):
        component = (
            next(value for key, value in model.components.items() if key.startswith("block_"))
            if name == "attention_mask"
            else model.components["stem"]
        )
        return component.hmonnx_session.get_input(name)


class _LegacyQwen4VisionEncoder(_Qwen4VisionCapture, VisualTokenGearHMONNXModel):
    def __init__(self, metadata, *, device, capture=None):
        super().__init__(metadata, device=device, enable_golden=True)
        self.enable_golden = False
        for graph in self.models.values():
            graph.hmonnx_session.interpreter._enable_golden = False
        self.capture = capture
        self._capture_graph_names = tuple(f"visual_m{gear}" for gear in self.gears)
        self.models = {
            gear: _CapturedVisualGraph(graph, self, f"visual_m{gear}") for gear, graph in self.models.items()
        }


def load_qwen4_exp_vision(metadata, root, *, device, capture=None):
    """Load a relocatable component package or the former whole-gear ABI."""
    data = metadata.to_dict() if hasattr(metadata, "to_dict") else dict(metadata)
    if data.get("abi_version", 1) != 1 or data.get("layout") == "components":
        from .vision_contract import validate_qwen4_exp_vision

        validate_qwen4_exp_vision(data, root)
        return Qwen4ComponentVisionEncoder(data, root, device=device, capture=capture)
    gears = []
    for item in data["gears"]:
        gear = dict(item)
        path = Path(gear["hmonnx"])
        gear["hmonnx"] = str(path if path.is_absolute() else Path(root) / path)
        gears.append(SimpleNamespace(**gear))
    return _LegacyQwen4VisionEncoder(SimpleNamespace(**{**data, "gears": gears}), device=device, capture=capture)


def export_qwen4_exp_visual(
    model_dir,
    output_dir,
    *,
    image_token_gears,
    quant_scheme,
    device="cpu",
    dtype="float16",
):
    """Export ``visual_m*`` components and relocatable standalone metadata."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_dir).vision_config
    if getattr(config, "deepstack_visual_indexes", ()):
        raise ValueError("Qwen4-Exp Flash Next visual export does not support deepstack features")
    model = XHQwen4ExpVisionModel(
        XHQwen3_5_VisualConfig(
            model_name="qwen3_8_flash_next_visual",
            model_type="Qwen4ExpForConditionalGeneration_visual",
            hf_model=str(model_dir),
            chip_arch="XH2a",
            quant_scheme=quant_scheme,
            dtype=dtype,
            visual_input_mode="patches",
            image_token_gears=list(image_token_gears),
            image_token_capacity=max(image_token_gears),
            spatial_merge_size=config.spatial_merge_size,
            temporal_patch_size=config.temporal_patch_size,
            patch_size=config.patch_size,
        )
    )
    model.to(device=device)
    model.export_hmonnx(str(output_dir))

    return json.loads((Path(output_dir) / "visual_meta_info.json").read_text())


__all__ = ["XHQwen4ExpVisionModel", "VisualTokenGearHMONNXModel", "export_qwen4_exp_visual", "load_qwen4_exp_vision"]
