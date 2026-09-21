from __future__ import annotations

import gc
import inspect
from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image

from xhquant.api import (
    ConfigDict,
    QuantScheme,
    convert_dynamo_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    create_quant_config,
)

from . import MiniMaxH3VisionEncoderWrapper
from .audio_vae_decoder_wrapper import MiniMaxH3AudioVAEDecoderWrapper
from .audio_vae_encoder_wrapper import MiniMaxH3AudioVAEEncoderWrapper
from .lora import DEFAULT_LORA_ALPHA, DEFAULT_LORA_SCALE, load_minimax_h3_lora
from .text_encoder_wrapper import (
    DEFAULT_INPUT_SEQUENCE_LENGTH,
    DEFAULT_TEXT_ENCODER_LAYER,
    MiniMaxH3TextEncoderWrapper,
    build_minimax_h3_text_graph_inputs,
    build_minimax_h3_text_inputs,
)
from .transformer_minimax_h3 import (
    MINIMAX_H3_MODALITY_NUM,
    MiniMaxH3Transformer3DModel,
    load_fixed_point_calibration,
)
from .transformer_wrapper import (
    MiniMaxH3TransformerWrapper,
    build_pack_gather_indices,
    build_packed_attention_mask,
    disable_transformer_block_cpu_offload,
    prepare_transformer_block_cpu_offload,
)
from .vae_decoder_wrapper import MiniMaxH3VAEDecoderWrapper, build_vae_decoder_rotary_emb
from .vae_encoder_wrapper import MiniMaxH3VAEEncoderWrapper
from .vision_encoder_wrapper import (
    DEFAULT_VISION_MAX_PATCHES,
    build_vision_encoder_graph_inputs,
)


DEFAULT_FIXED_POINT_CALIBRATION_PATH = (
    Path(__file__).resolve().parents[4]
    / "examples_merak"
    / "aigc"
    / "minimaxh3"
    / "minimax_h3_fixed_point_calibration.json"
)


COMPONENTS = (
    "vae_encoder",
    "vae_decoder",
    "audio_vae_encoder",
    "audio_vae_decoder",
    "transformer",
    "vision_encoder",
    "text_encoder",
)
WRAPPERS = {
    "vae_encoder": ("vae", MiniMaxH3VAEEncoderWrapper),
    "vae_decoder": ("vae", MiniMaxH3VAEDecoderWrapper),
    "audio_vae_encoder": ("audio_vae", MiniMaxH3AudioVAEEncoderWrapper),
    "audio_vae_decoder": ("audio_vae", MiniMaxH3AudioVAEDecoderWrapper),
    "transformer": ("transformer", MiniMaxH3TransformerWrapper),
    "vision_encoder": ("text_encoder", MiniMaxH3VisionEncoderWrapper),
    "text_encoder": ("text_encoder", MiniMaxH3TextEncoderWrapper),
}
COMPONENT_GROUPS = (
    ("vae_encoder", "vae_decoder"),
    ("audio_vae_encoder", "audio_vae_decoder"),
    ("transformer",),
    ("text_encoder", "vision_encoder"),
)
_TEXT_VISION_COMPONENTS = frozenset({"text_encoder", "vision_encoder"})


def expand_minimax_h3_components(components: set[str] | tuple[str, ...] | list[str], export: bool = False) -> set[str]:
    """Expand MiniMax-H3 component selections that must be exported/run together."""
    selected = set(components)
    if selected & _TEXT_VISION_COMPONENTS:
        selected |= _TEXT_VISION_COMPONENTS
    # Transformer packing needs fixed-length text embeds + encoder_attention_mask.
    if "transformer" in selected and not export:
        selected |= _TEXT_VISION_COMPONENTS
    return selected


TENSOR_NAMES = {
    "vae_encoder": (("video",), ("latents",)),
    "vae_decoder": (("latents", "rotary_cos", "rotary_sin"), ("video",)),
    "audio_vae_encoder": (("waveform",), ("mean", "log_std")),
    "audio_vae_decoder": (("latents",), ("waveform",)),
    "transformer": (
        (
            "hidden_states",
            "audio_hidden_states",
            "encoder_hidden_states",
            "temb",
            "timestep_indices",
            "adaln_indices",
            "rotary_cos",
            "rotary_sin",
            "video_indices",
            "audio_indices",
            # text_indices are host-only (build pack_gather / packed mask); Dynamo drops unused args.
            "pack_gather_indices",
            "encoder_attention_mask",
            "packed_attention_mask",
        ),
        ("output_video_hidden_states", "output_audio_hidden_states"),
    ),
    "text_encoder": (
        (
            "hidden_states",
            "rotary_cos",
            "rotary_sin",
            "attention_mask",
            "deepstack_visual_embed_0",
            "deepstack_visual_embed_1",
            "deepstack_visual_embed_2",
        ),
        ("encoder_hidden_states",),
    ),
    "vision_encoder": (
        (
            "pixel_values",
            "position_embeddings",
            "rotary_cos",
            "rotary_sin",
            "attention_mask",
        ),
        ("image_embeds", "deepstack_feature_0", "deepstack_feature_1", "deepstack_feature_2"),
    ),
}


class MiniMaxH3Converter:
    def __init__(self, model_dir: str, export_cfg: dict[str, Any], *, device: str, seed: int):
        self.model_dir = model_dir
        self.cfg = export_cfg
        self.device = torch.device(device)
        self.seed = seed

    def export(self, output_dir: Path, components: tuple[str, ...]) -> dict[str, Any]:
        results = {}
        selected = expand_minimax_h3_components(components, export=True)
        for group in COMPONENT_GROUPS:
            group_components = tuple(component for component in group if component in selected)
            if not group_components:
                continue
            model_name = WRAPPERS[group_components[0]][0]
            pipe = self._load_pipeline(model_name)
            for component in group_components:
                results[component] = self._export_component(output_dir, component, pipe)
            del pipe
            self._release_memory()
        return results

    def _export_component(self, output_dir: Path, component: str, pipe) -> dict[str, Any]:
        name, wrapper_type = WRAPPERS[component]
        wrapper_kwargs = {}
        vision_meta: dict[str, Any] | None = None
        if component == "text_encoder":
            h3_cfg = self.cfg.get("minimax_h3") or {}
            wrapper_kwargs = {
                "input_sequence_length": int(h3_cfg.get("input_sequence_length", DEFAULT_INPUT_SEQUENCE_LENGTH)),
                "text_encoder_layer": int(h3_cfg.get("text_encoder_layer", DEFAULT_TEXT_ENCODER_LAYER)),
            }
        elif component == "vision_encoder":
            h3_cfg = self.cfg.get("minimax_h3") or {}
            wrapper_kwargs = {
                "max_patches": int(h3_cfg.get("vision_max_patches", DEFAULT_VISION_MAX_PATCHES)),
            }
        wrapped_module = getattr(pipe, name)
        if component == "vision_encoder":
            wrapped_module = getattr(getattr(wrapped_module, "model", None), "visual", None)
            if wrapped_module is None:
                raise AttributeError("MiniMax-H3 text encoder is missing model.visual")
        wrapper = wrapper_type(wrapped_module, **wrapper_kwargs).eval()
        if component == "vae_encoder":
            wrapper = wrapper.tile_encoder
        if component == "transformer":
            # FixedPointMulAdd is created during wrap; apply calibration afterwards so the ops refresh.
            self._apply_fixed_point_calibration(wrapper.transformer, self.cfg.get("minimax_h3") or {})
            # Keep activations on device, stream one transformer block at a time for FP warm-up.
            # Disable before Dynamo so `.to(cpu/cuda)` side effects are not captured.
            prepare_transformer_block_cpu_offload(wrapper.transformer, self.device)
        else:
            wrapper = wrapper.to(self.device)
        if component == "vision_encoder":
            inputs, vision_meta = self._build_vision_encoder_inputs(pipe, wrapper)
        else:
            inputs = self._build_inputs(component, pipe)
        with torch.no_grad():
            outputs = wrapper(*inputs)
            del outputs
            torch.cuda.empty_cache()

        quant_scheme = QuantScheme()
        quant_scheme.target_device = str(self.cfg["target_device"])
        quant_scheme.quant_type = str(
            (self.cfg.get("minimax_h3") or {}).get("quant_types", {}).get(component, "w8a8_ssfp")
        )
        quant_cfg = ConfigDict(create_quant_config(quant_scheme))
        if component == "transformer":
            quant_cfg.auto_release_unused_parameters = True
            quant_cfg.infer_shape = False
            disable_transformer_block_cpu_offload(wrapper.transformer)

        wrapper = wrapper.cpu()
        inputs = [tensor.cpu() for tensor in inputs]

        quanted = convert_dynamo_model_to_quanted_model(wrapper, inputs, quant_scheme.target_device, quant_cfg)
        hmonnx_path = (
            output_dir
            / component
            / f"minimax_h3_{component}-{quant_scheme.target_device}-{quant_scheme.quant_type}.onnx"
        )
        hmonnx_path.parent.mkdir(parents=True, exist_ok=True)
        input_names, output_names = TENSOR_NAMES[component]
        if len(input_names) != len(inputs):
            raise ValueError(
                f"{component} export input count mismatch: TENSOR_NAMES has {len(input_names)} names "
                f"{list(input_names)}, but built inputs has {len(inputs)}"
            )
        if component == "transformer":
            wrapper_param_names = tuple(
                name
                for name in inspect.signature(MiniMaxH3TransformerWrapper.forward).parameters
                if name != "self"
            )
            if wrapper_param_names != tuple(input_names):
                raise ValueError(
                    "transformer wrapper forward parameter names must match TENSOR_NAMES: "
                    f"wrapper={wrapper_param_names}, TENSOR_NAMES={tuple(input_names)}"
                )
        convert_quanted_model_to_hmonnx(
            quanted,
            list(inputs),
            str(hmonnx_path),
            list(input_names),
            list(output_names),
        )
        result = {
            "hmonnx_file": str(hmonnx_path.relative_to(output_dir)),
            "input_names": list(input_names),
            "output_names": list(output_names),
            "inputs": [self._tensor_spec(name, tensor) for name, tensor in zip(input_names, inputs, strict=True)],
        }
        if vision_meta is not None:
            result.update(vision_meta)
        del wrapper, inputs, quanted
        self._release_memory()
        return result

    def _load_pipeline(self, component_name: str):
        from diffusers import ComponentsManager
        from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ModularPipeline

        manager = ComponentsManager()
        manager.enable_auto_cpu_offload(device=str(self.device))
        workflow = str((self.cfg.get("minimax_h3") or {}).get("workflow", "fl2va"))
        pipe = MiniMaxH3ModularPipeline.from_pretrained(
            self.model_dir,
            workflow=workflow,
            components_manager=manager,
        )
        if component_name == "transformer":
            pipe.transformer = MiniMaxH3Transformer3DModel.from_pretrained(
                self.model_dir,
                subfolder="transformer",
                torch_dtype=torch.float16,
            )
            h3_cfg = self.cfg.get("minimax_h3") or {}
            lora_path = h3_cfg.get("lora_path")
            if lora_path:
                load_minimax_h3_lora(
                    pipe.transformer,
                    lora_path,
                    lora_alpha=int(h3_cfg.get("lora_alpha", DEFAULT_LORA_ALPHA)),
                    lora_scale=float(h3_cfg.get("lora_scale", DEFAULT_LORA_SCALE)),
                )
        else:
            component_names = [component_name]
            if component_name == "text_encoder":
                component_names.extend(("tokenizer", "processor"))

            pipe.load_components(
                names=component_names,
                dtype={
                    "default": torch.bfloat16,
                    "vae": torch.float16,
                    "audio_vae": torch.float16,
                    "text_encoder": torch.float16,
                },
            )
        return pipe

    def _release_memory(self) -> None:
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @staticmethod
    def _resolve_fixed_point_calibration_path(h3_cfg: Mapping[str, Any]) -> Path | None:
        if "fixed_point_calibration_path" in h3_cfg:
            configured = h3_cfg.get("fixed_point_calibration_path")
            if configured in (None, "", False):
                return None
            path = Path(str(configured)).expanduser()
        else:
            path = DEFAULT_FIXED_POINT_CALIBRATION_PATH
        if not path.is_absolute():
            path = Path.cwd() / path
        return path

    def _apply_fixed_point_calibration(self, transformer, h3_cfg: Mapping[str, Any]) -> None:
        calibration_path = self._resolve_fixed_point_calibration_path(h3_cfg)
        if calibration_path is None:
            return
        if not calibration_path.is_file():
            raise FileNotFoundError(f"MiniMax-H3 fixed-point calibration not found: {calibration_path}")
        load_fixed_point_calibration(transformer, calibration_path)

    @staticmethod
    def _tensor_spec(name: str, tensor: torch.Tensor) -> dict[str, Any]:
        return {
            "name": name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }

    @staticmethod
    def _spec_shape(specs: list[Mapping[str, Any]], name: str) -> tuple[int, ...]:
        for spec in specs:
            if str(spec.get("name", "")) == name:
                return tuple(int(value) for value in spec["shape"])
        raise KeyError(f"Missing export input shape for {name}")

    @classmethod
    def _build_transformer_index_tensors(
        cls,
        specs: list[Mapping[str, Any]],
        *,
        device: torch.device,
        generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        """Build gather/scatter indices that stay inside AdaLN / packing bounds.

        Export dummies previously used ``randint(0, 1024)`` for every int64 tensor. With
        ``temb`` of shape ``[T, ...]`` the AdaLN table only has ``T * MINIMAX_H3_MODALITY_NUM``
        rows, so values ``>= T * 3`` trigger CUDA ``index_select`` OOB asserts.
        """
        temb_shape = cls._spec_shape(specs, "temb")
        adaln_shape = cls._spec_shape(specs, "adaln_indices")
        timestep_shape = cls._spec_shape(specs, "timestep_indices")
        if len(temb_shape) < 1 or temb_shape[0] <= 0:
            raise ValueError(f"`temb` must have a positive leading timestep count, got {list(temb_shape)}")
        num_timesteps = int(temb_shape[0])
        sequence_length = int(adaln_shape[0])
        if timestep_shape != adaln_shape:
            raise ValueError(
                "`timestep_indices` and `adaln_indices` must share the same shape, got "
                f"{list(timestep_shape)} vs {list(adaln_shape)}"
            )

        # text_indices are not an exported graph input; derive text capacity from encoder embeds.
        text_len = int(cls._spec_shape(specs, "encoder_hidden_states")[1])
        video_len = int(cls._spec_shape(specs, "video_indices")[0])
        audio_len = int(cls._spec_shape(specs, "audio_indices")[0])
        if text_len + video_len + audio_len != sequence_length:
            raise ValueError(
                "Packed transformer indices must partition `adaln_indices` length: "
                f"text({text_len}) + video({video_len}) + audio({audio_len}) != {sequence_length}"
            )
        encoder_mask_shape = None
        try:
            encoder_mask_shape = cls._spec_shape(specs, "encoder_attention_mask")
        except KeyError:
            encoder_mask_shape = None
        if encoder_mask_shape is not None and int(encoder_mask_shape[-1]) != text_len:
            raise ValueError(
                "`encoder_attention_mask` last dim must match `encoder_hidden_states` text length: "
                f"{int(encoder_mask_shape[-1])} != {text_len}"
            )

        text_indices = torch.arange(text_len, device=device, dtype=torch.int64)
        video_indices = torch.arange(text_len, text_len + video_len, device=device, dtype=torch.int64)
        audio_indices = torch.arange(
            text_len + video_len,
            text_len + video_len + audio_len,
            device=device,
            dtype=torch.int64,
        )

        # Match runtime modality tags: video=0, text=1, audio=2.
        token_tags = torch.empty(sequence_length, device=device, dtype=torch.int64)
        token_tags.index_copy_(0, video_indices, torch.zeros(video_len, device=device, dtype=torch.int64))
        token_tags.index_copy_(0, text_indices, torch.ones(text_len, device=device, dtype=torch.int64))
        token_tags.index_copy_(
            0,
            audio_indices,
            torch.full((audio_len,), 2, device=device, dtype=torch.int64),
        )
        timestep_indices = torch.randint(
            0,
            num_timesteps,
            (sequence_length,),
            device=device,
            dtype=torch.int64,
            generator=generator,
        )
        adaln_indices = timestep_indices * MINIMAX_H3_MODALITY_NUM + token_tags
        pack_gather_indices = build_pack_gather_indices(
            text_indices,
            video_indices,
            audio_indices,
            sequence_length=sequence_length,
        )
        return {
            "timestep_indices": timestep_indices,
            "adaln_indices": adaln_indices,
            "video_indices": video_indices,
            "audio_indices": audio_indices,
            # Host-only helper tensor for building packed_attention_mask; not in TENSOR_NAMES.
            "text_indices": text_indices,
            "pack_gather_indices": pack_gather_indices,
        }

    def _build_inputs(self, component: str, pipe) -> tuple[torch.Tensor, ...]:
        if component == "text_encoder":
            return self._build_text_encoder_inputs(pipe)

        shapes = (self.cfg.get("minimax_h3") or {}).get("input_shapes", {})
        if component not in shapes:
            raise KeyError(f"Missing export.minimax_h3.input_shapes.{component}")
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        component_specs = list(shapes[component])
        index_tensors = (
            self._build_transformer_index_tensors(component_specs, device=self.device, generator=generator)
            if component == "transformer"
            else {}
        )
        inputs = []
        built_by_name: dict[str, torch.Tensor] = {}
        for spec in component_specs:
            dtype = getattr(torch, spec.get("dtype", "float16"))
            name = str(spec.get("name", ""))
            shape = tuple(int(value) for value in spec["shape"])
            if name in index_tensors:
                if name == "text_indices":
                    # Host-only: used to build pack_gather / packed mask, not an export input.
                    continue
                tensor = index_tensors[name]
                if tuple(tensor.shape) != shape:
                    raise ValueError(f"Generated {name} shape {tuple(tensor.shape)} != configured {shape}")
                if tensor.dtype != dtype:
                    tensor = tensor.to(dtype=dtype)
                inputs.append(tensor)
                built_by_name[name] = tensor
            elif name == "encoder_attention_mask":
                if not dtype.is_floating_point:
                    raise TypeError("transformer encoder_attention_mask must use a floating-point dtype")
                valid_length = int(spec.get("valid_length", shape[-1]))
                if not 0 < valid_length <= shape[-1]:
                    raise ValueError(
                        f"encoder_attention_mask valid_length must be in [1, {shape[-1]}], got {valid_length}"
                    )
                attention_mask = torch.zeros(shape, device=self.device, dtype=dtype)
                attention_mask[..., valid_length:] = torch.finfo(dtype).min
                inputs.append(attention_mask)
                built_by_name[name] = attention_mask
            elif name == "packed_attention_mask":
                if not dtype.is_floating_point:
                    raise TypeError("transformer packed_attention_mask must use a floating-point dtype")
                # Rebuild from the already-materialized text mask + host-only text_indices so export
                # dummies stay consistent with host-side packing.
                encoder_mask = built_by_name.get("encoder_attention_mask")
                text_indices = index_tensors.get("text_indices")
                if encoder_mask is None or text_indices is None:
                    raise KeyError(
                        "packed_attention_mask requires encoder_attention_mask earlier in "
                        "transformer input_shapes and host-built text_indices"
                    )
                packed_mask = build_packed_attention_mask(
                    encoder_mask,
                    text_indices,
                    sequence_length=int(shape[-1]),
                )
                if tuple(packed_mask.shape) != shape:
                    raise ValueError(
                        f"Generated packed_attention_mask shape {tuple(packed_mask.shape)} != configured {shape}"
                    )
                if packed_mask.dtype != dtype:
                    packed_mask = packed_mask.to(dtype=dtype)
                inputs.append(packed_mask)
                built_by_name[name] = packed_mask
            elif dtype.is_floating_point:
                tensor = torch.randn(
                    *shape,
                    device=self.device,
                    dtype=dtype,
                    generator=generator,
                )
                inputs.append(tensor)
                built_by_name[name] = tensor
            else:
                high = 2 if "mask" in name.lower() else 1024
                tensor = torch.randint(
                    0,
                    high,
                    shape,
                    device=self.device,
                    dtype=dtype,
                    generator=generator,
                )
                inputs.append(tensor)
                built_by_name[name] = tensor
        if component == "vae_decoder":
            rotary_cos, rotary_sin = build_vae_decoder_rotary_emb(pipe.vae.decoder, pipe.vae.post_quant_conv(inputs[0]))
            return inputs[0], rotary_cos.half(), rotary_sin.half()
        return tuple(inputs)

    def _build_vision_encoder_inputs(
        self,
        pipe,
        wrapper: MiniMaxH3VisionEncoderWrapper,
    ) -> tuple[tuple[torch.Tensor, ...], dict[str, Any]]:
        h3_cfg = self.cfg.get("minimax_h3") or {}
        image_path = h3_cfg.get("vision_image")
        if not image_path:
            raise KeyError("Missing export.minimax_h3.vision_image")
        image_path = Path(str(image_path)).expanduser()
        if not image_path.is_absolute():
            image_path = Path.cwd() / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"MiniMax-H3 vision export image not found: {image_path}")
        processor = getattr(pipe, "processor", None)
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            raise AttributeError("MiniMax-H3 pipeline is missing processor.image_processor")
        with Image.open(image_path) as image:
            vision_inputs = image_processor(images=image.convert("RGB"), return_tensors="pt")
        pixel_values = vision_inputs["pixel_values"].to(self.device, dtype=torch.float16)
        image_grid_thw = vision_inputs["image_grid_thw"].to(self.device, dtype=torch.long)
        graph_inputs, geometry = build_vision_encoder_graph_inputs(
            wrapper.visual,
            pixel_values,
            image_grid_thw,
            max_patches=wrapper.max_patches,
            dtype=torch.float16,
        )
        vision_meta = {
            "image_grid_thw": image_grid_thw.detach().cpu().tolist(),
            "vision_max_patches": int(geometry["max_patches"]),
            "vision_max_image_tokens": int(geometry["max_image_tokens"]),
            "valid_patches": int(geometry["valid_patches"]),
            "valid_image_tokens": int(geometry["valid_image_tokens"]),
            "attention_mask_format": "additive_key_padding_bias",
        }
        return graph_inputs, vision_meta

    def _build_text_encoder_inputs(self, pipe) -> tuple[torch.Tensor, ...]:
        h3_cfg = self.cfg.get("minimax_h3") or {}
        input_sequence_length = int(h3_cfg.get("input_sequence_length", DEFAULT_INPUT_SEQUENCE_LENGTH))
        prompt = str(h3_cfg.get("prompt", "a clay fox walking through a forest"))
        shapes = h3_cfg.get("input_shapes", {}).get("text_encoder")
        tokenizer = getattr(pipe, "tokenizer", None)
        if tokenizer is None:
            raise AttributeError("MiniMax-H3 pipeline is missing tokenizer for text_encoder export")

        if shapes:
            # Prefer explicit YAML shapes when provided, but keep integer semantics.
            named = {item.get("name"): item for item in shapes if isinstance(item, Mapping)}
            hidden_states_spec = named.get("hidden_states") or shapes[0]
            seq_len = int((hidden_states_spec.get("shape") or [1, input_sequence_length])[-2])
            input_sequence_length = seq_len

        input_ids, attention_mask, position_ids = build_minimax_h3_text_inputs(
            tokenizer,
            prompt,
            input_sequence_length,
            device=self.device,
        )
        language_model = getattr(getattr(pipe.text_encoder, "model", None), "language_model", None)
        if language_model is None:
            raise AttributeError("MiniMax-H3 text encoder is missing model.language_model")
        graph_inputs = build_minimax_h3_text_graph_inputs(
            language_model,
            input_ids,
            attention_mask,
            position_ids,
        )
        if shapes:
            expected = [
                (name, tensor, named.get(name) or shapes[index])
                for index, (name, tensor) in enumerate(zip(TENSOR_NAMES["text_encoder"][0], graph_inputs, strict=True))
            ]
            for name, tensor, spec in expected:
                if spec is None:
                    continue
                expected_shape = [int(item) for item in spec["shape"]]
                if list(tensor.shape) != expected_shape:
                    raise ValueError(
                        f"text_encoder {name} shape mismatch: constructed {list(tensor.shape)} vs yaml {expected_shape}"
                    )
        return graph_inputs
