from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_vl.modeling_qwen3_vl import BaseModelOutputWithDeepstackFeatures

from xhmodel_merak.exec_backends import HMONNXInferenceConfig, HMONNXInferenceV2

from .minimax_h3_converter import expand_minimax_h3_components
from .text_conditioning import install_fixed_length_text_conditioning
from .text_encoder_wrapper import build_minimax_h3_attention_bias
from .transformer_wrapper import (
    MINIMAX_H3_DEFAULT_NUM_INFERENCE_STEPS,
    build_minimax_h3_denoise_unique_timesteps,
    build_minimax_h3_transformer_conditioning,
    build_pack_gather_indices,
    build_packed_attention_mask,
)
from .vae_decoder_wrapper import build_vae_decoder_rotary_emb
from .vae_encoder_wrapper import MiniMaxH3VAEEncoderTileWrapper, MiniMaxH3VAEEncoderWrapper
from .vision_encoder_wrapper import (
    DEFAULT_VISION_MAX_PATCHES,
    build_vision_encoder_graph_inputs,
)


_SUPPORTED_COMPONENTS = (
    "vae_encoder",
    "vae_decoder",
    "audio_vae_encoder",
    "audio_vae_decoder",
    "transformer",
    "vision_encoder",
    "text_encoder",
)

_PIPELINE_CLEANUP_INSTALLED = "_xh_hmonnx_cleanup_installed"


def _is_hmonnx_wrapper(module: Any) -> bool:
    """True for adapters that still own a live HMONNX runtime/session."""
    return module is not None and hasattr(module, "runtime") and hasattr(module, "_fallback_offloaded")


def _holds_hmonnx_runtime(module: Any) -> bool:
    """Skip teardown for modules whose HMONNX session may still be live on CUDA."""
    if _is_hmonnx_wrapper(module):
        return True
    # audio_vae may keep HMONNX on bound encode/decode methods.
    for attr in ("encode", "decode"):
        if _is_hmonnx_wrapper(getattr(module, attr, None)):
            return True
    model = getattr(module, "model", None)
    if model is None:
        return False
    return _is_hmonnx_wrapper(getattr(model, "language_model", None)) or _is_hmonnx_wrapper(
        getattr(model, "visual", None)
    )


def _release_pipeline_components(pipe: Any) -> None:
    # Install already drops FP bodies via `_release_fallback`. Post-call cleanup must
    # not null live HMONNX wrappers: destroying their CUDA sessions + empty_cache
    # without synchronize can surface as illegal memory access after transformer runs.
    for name in ("vae", "audio_vae", "transformer", "text_encoder"):
        module = None
        if name in getattr(pipe, "_modules", {}):
            module = pipe._modules.get(name)
        elif hasattr(pipe, name):
            module = getattr(pipe, name)
        if module is None or _holds_hmonnx_runtime(module):
            continue
        if name in getattr(pipe, "_modules", {}):
            pipe._modules[name] = None
        elif hasattr(pipe, name):
            setattr(pipe, name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _install_pipeline_cleanup(pipe: Any) -> None:
    if getattr(pipe, _PIPELINE_CLEANUP_INSTALLED, False):
        return

    original_call = pipe.__class__.__call__

    def call_with_cleanup(self, *args, **kwargs):
        try:
            return original_call(self, *args, **kwargs)
        finally:
            _release_pipeline_components(self)

    pipe.__class__ = type(
        f"{pipe.__class__.__name__}WithHMONNXCleanup",
        (pipe.__class__,),
        {"__call__": call_with_cleanup},
    )
    setattr(pipe, _PIPELINE_CLEANUP_INSTALLED, True)


def _release_module_reference(owner: nn.Module, name: str) -> None:
    if name in owner._modules:
        owner._modules[name] = None
    else:
        setattr(owner, name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _remove_manager_components(pipe: Any, names: tuple[str, ...]) -> None:
    manager = getattr(pipe, "_components_manager", None)
    if manager is None or not hasattr(manager, "components") or not hasattr(manager, "remove"):
        return
    component_ids = [
        component_id for component_id in tuple(manager.components) if manager._id_to_name(component_id) in names
    ]
    for component_id in component_ids:
        manager.remove(component_id)


def _restore_vae_decode_modules(vae: nn.Module, device: torch.device | str) -> None:
    post_quant_conv = getattr(vae, "post_quant_conv", None)
    if isinstance(post_quant_conv, nn.Module):
        post_quant_conv.to(device)
    decoder = getattr(vae, "decoder", None)
    rope = getattr(decoder, "rope", None)
    if isinstance(rope, nn.Module):
        rope.to(device)


def _release_before_vae_decode(pipe: Any, vae: nn.Module) -> None:
    manager = getattr(pipe, "_components_manager", None)
    auto_offload_enabled = bool(getattr(manager, "_auto_offload_enabled", False))
    offload_device = getattr(manager, "_auto_offload_device", None)
    offload_strategy = getattr(manager, "_offload_strategy", None)
    if auto_offload_enabled:
        manager.disable_auto_cpu_offload()
    try:
        _remove_manager_components(pipe, ("transformer",))
        if "transformer" in getattr(pipe, "_modules", {}):
            pipe._modules["transformer"] = None
        elif hasattr(pipe, "transformer"):
            pipe.transformer = None
        # vae._encode = None
        # _release_module_reference(vae, "encoder")
        # _release_module_reference(vae, "quant_conv")
    finally:
        if auto_offload_enabled:
            manager.enable_auto_cpu_offload(offload_device, offload_strategy=offload_strategy)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _normalize_exec_devices(
    device: torch.device | str | int,
    exec_devices: list[torch.device | str | int] | tuple[torch.device | str | int, ...] | None = None,
) -> list[torch.device | str | int]:
    if exec_devices is None:
        return [torch.device(device)]
    normalized = [torch.device(item) if not isinstance(item, int) else item for item in exec_devices]
    if not normalized:
        raise ValueError("exec_devices must contain at least one device")
    return normalized


def _should_enable_auto_offload(exec_devices: list[torch.device | str | int]) -> bool:
    cuda_devices = []
    for item in exec_devices:
        device = torch.device(item) if not isinstance(item, int) else torch.device("cuda", item)
        if device.type == "cuda":
            cuda_devices.append(device)
    return len(cuda_devices) > 1


class _LazyHMONNXRuntime:
    def __init__(
        self,
        path: Path,
        device: torch.device,
        expected_shapes: tuple[tuple[int, ...], ...] | None = None,
        output_list: bool = False,
        exec_devices: list[torch.device | str | int] | tuple[torch.device | str | int, ...] | None = None,
        enable_golden: bool = False,
        golden_dir: str | Path | None = None,
    ):
        self.path = Path(path)
        self.device = torch.device(device)
        self.exec_devices = _normalize_exec_devices(self.device, exec_devices)
        self.enable_auto_offload = _should_enable_auto_offload(self.exec_devices)
        self.expected_shapes = expected_shapes
        self.session = None
        self.output_list = output_list
        self.enable_golden = bool(enable_golden)
        self.golden_dir = None if golden_dir is None else str(golden_dir)

    def supports(self, *inputs: torch.Tensor) -> bool:
        return self.expected_shapes is None or tuple(tuple(tensor.shape) for tensor in inputs) == self.expected_shapes

    def _create_session(self) -> HMONNXInferenceV2:
        session_config = HMONNXInferenceConfig(
            enable_auto_offload=self.enable_auto_offload,
            enable_golden=self.enable_golden,
            exec_devices=list(self.exec_devices),
        )
        session = HMONNXInferenceV2(str(self.path), session_config)
        if self.golden_dir is not None:
            session.save_golden_dir = self.golden_dir
            session.golden_dir = self.golden_dir
        return session

    def __call__(self, *inputs: torch.Tensor):
        if self.session is None:
            self.session = self._create_session()
        if self.output_list:
            print(
                "Warning: HMONNX runtime is returning outputs as a list. Ensure that the caller handles this correctly."
            )
            return [self.session(*inputs)]
        return self.session(*inputs)


class _HMONNXVAEEncoderTile(nn.Module):
    def __init__(self, runtime: _LazyHMONNXRuntime, fallback: nn.Module, owner: nn.Module):
        super().__init__()
        self.runtime = runtime
        self.fallback = fallback
        self.owner = owner
        self._fallback_offloaded = False

    def forward(self, tile: torch.Tensor) -> torch.Tensor:
        if not self.runtime.supports(tile):
            return self.fallback(tile)
        self._release_fallback()
        output = self.runtime(tile)
        return output[0] if isinstance(output, (tuple, list)) else output

    def _release_fallback(self) -> None:
        if self._fallback_offloaded:
            return
        fallback = self.fallback
        self.fallback = None
        _release_module_reference(self.owner, "encoder")
        self.owner = None
        self._fallback_offloaded = True
        del fallback
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _HMONNXModule(nn.Module):
    def __init__(self, runtime: _LazyHMONNXRuntime, fallback: nn.Module, input_names: tuple[str, ...]):
        super().__init__()
        self.runtime = runtime
        self.fallback = fallback
        self.input_names = input_names
        self._fallback_offloaded = False

    @property
    def dtype(self):
        return getattr(self.fallback, "dtype", torch.float16)

    def forward(self, *inputs: torch.Tensor, **kwargs):
        original_inputs = inputs
        original_kwargs = kwargs
        if kwargs:
            if not set(kwargs).issubset(self.input_names):
                return self.fallback(*original_inputs, **original_kwargs)
            try:
                inputs = tuple(kwargs[name] for name in self.input_names)
            except KeyError:
                return self.fallback(*original_inputs, **original_kwargs)
        if not self.runtime.supports(*inputs):
            return self.fallback(*original_inputs, **original_kwargs)
        self._release_fallback()
        output = self.runtime(*inputs)
        return output[0] if isinstance(output, (tuple, list)) and len(output) == 1 else output

    def _release_fallback(self) -> None:
        if self._fallback_offloaded:
            return
        fallback = self.fallback
        self.fallback = None
        self._fallback_offloaded = True
        del fallback
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _resolve_fixed_adaln_unique_timesteps(
    fixed_adaln_meta: dict[str, Any] | None,
    *,
    num_inference_steps: int = MINIMAX_H3_DEFAULT_NUM_INFERENCE_STEPS,
) -> list[list[float]]:
    """Prefer export-baked unique timesteps; otherwise rebuild the default fl2va schedule."""
    if fixed_adaln_meta:
        unique = fixed_adaln_meta.get("unique_timesteps")
        if isinstance(unique, list) and unique:
            return [list(map(float, row)) for row in unique]
        num_inference_steps = int(fixed_adaln_meta.get("num_inference_steps", num_inference_steps))
    rebuilt = build_minimax_h3_denoise_unique_timesteps(num_inference_steps=num_inference_steps)
    return [row.detach().float().cpu().tolist() for row in rebuilt]


class _HMONNXTransformerModule(nn.Module):
    """HMONNX transformer adapter with host-side step_index/adaln/rotary precompute."""

    def __init__(
        self,
        runtime: _LazyHMONNXRuntime,
        fallback: nn.Module,
        input_names: tuple[str, ...],
        *,
        fixed_adaln_meta: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.runtime = runtime
        self.fallback = fallback
        self.input_names = input_names
        self._dtype = getattr(fallback, "dtype", torch.float16)
        # Packing/layout reads transformer.config.patch_size after install.
        self.config = getattr(fallback, "config", None)
        # Keep only the host helpers needed to build the export ABI.
        self.time_proj = fallback.time_proj
        self.time_embedder = fallback.time_embedder
        self.rope = fallback.rope
        self._fallback_offloaded = False
        fallback_meta = getattr(fallback, "_minimax_fixed_adaln_meta", None)
        meta = fixed_adaln_meta or fallback_meta or {}
        self._fixed_adaln_unique_timesteps = _resolve_fixed_adaln_unique_timesteps(meta)
        self._fixed_adaln_num_evals = len(self._fixed_adaln_unique_timesteps)
        self._minimax_step_counter = 0

    @property
    def dtype(self):
        return self._dtype

    def _resolve_step_index(self, timestep: torch.Tensor) -> int:
        current = timestep.detach().float().cpu().reshape(-1).tolist()
        for idx, expected in enumerate(self._fixed_adaln_unique_timesteps):
            if len(expected) != len(current):
                continue
            if all(abs(float(a) - float(b)) < 1e-4 for a, b in zip(current, expected, strict=True)):
                self._minimax_step_counter = (idx + 1) % max(self._fixed_adaln_num_evals, 1)
                return idx
        if self._fixed_adaln_num_evals <= 0:
            raise RuntimeError(
                "MiniMax-H3 HMONNX transformer is missing baked AdaLN unique-timestep metadata; "
                "re-export after install_fixed_adaln_lut."
            )
        step_index = int(self._minimax_step_counter % self._fixed_adaln_num_evals)
        self._minimax_step_counter = (step_index + 1) % self._fixed_adaln_num_evals
        return step_index

    def forward(
        self,
        hidden_states: torch.Tensor,
        audio_hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        timestep_indices: torch.Tensor,
        token_tags: torch.Tensor,
        position_ids: torch.Tensor,
        video_indices: torch.Tensor,
        audio_indices: torch.Tensor,
        text_indices: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = False,
    ):
        del return_dict
        if attention_kwargs:
            raise ValueError("MiniMax-H3 HMONNX transformer does not support attention_kwargs")
        if encoder_attention_mask is None:
            raise ValueError("MiniMax-H3 HMONNX transformer requires encoder_attention_mask")

        # Host still builds adaln_indices / rotary from live timestep helpers; temb itself is
        # baked into the AdaLN LUT and replaced by scalar step_index on the graph ABI.
        legacy_temb, adaln_indices, rotary_cos, rotary_sin = build_minimax_h3_transformer_conditioning(
            self,
            timestep=timestep,
            timestep_indices=timestep_indices,
            token_tags=token_tags,
            position_ids=position_ids,
        )
        # Graph ABI uses INT32 scalar step_index (torch.onnx may downcast INT64 scalars).
        step_index = torch.tensor(
            self._resolve_step_index(timestep),
            device=hidden_states.device,
            dtype=torch.int32,
        )
        # Export wrapper has no dtype cast; host aligns live tensors to graph/body dtype.
        if hidden_states.dtype != self.dtype:
            hidden_states = hidden_states.to(dtype=self.dtype)
        if audio_hidden_states.dtype != self.dtype:
            audio_hidden_states = audio_hidden_states.to(dtype=self.dtype)
        if encoder_hidden_states.dtype != self.dtype:
            encoder_hidden_states = encoder_hidden_states.to(dtype=self.dtype)
        if rotary_cos.dtype != self.dtype:
            rotary_cos = rotary_cos.to(dtype=self.dtype)
        if rotary_sin.dtype != self.dtype:
            rotary_sin = rotary_sin.to(dtype=self.dtype)
        if encoder_attention_mask.dtype != self.dtype:
            encoder_attention_mask = encoder_attention_mask.to(dtype=self.dtype)
        sequence_length = int(adaln_indices.shape[0])
        packed_attention_mask = build_packed_attention_mask(
            encoder_attention_mask,
            text_indices,
            sequence_length=sequence_length,
        )
        if packed_attention_mask.dtype != self.dtype:
            packed_attention_mask = packed_attention_mask.to(dtype=self.dtype)
        pack_gather_indices = build_pack_gather_indices(
            text_indices,
            video_indices,
            audio_indices,
            sequence_length=sequence_length,
        )
        # text_indices remain host-only; export/HMONNX graph ABI does not consume them.
        named_inputs = {
            "hidden_states": hidden_states,
            "audio_hidden_states": audio_hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "step_index": step_index,
            "timestep_indices": timestep_indices.to(torch.int32),
            "adaln_indices": adaln_indices.to(torch.int32),
            "rotary_cos": rotary_cos,
            "rotary_sin": rotary_sin,
            "video_indices": video_indices.to(torch.int32),
            "audio_indices": audio_indices.to(torch.int32),
            "pack_gather_indices": pack_gather_indices.to(torch.int32),
            "encoder_attention_mask": encoder_attention_mask,
            "packed_attention_mask": packed_attention_mask,
        }
        if "temb" in self.input_names:
            named_inputs["temb"] = legacy_temb.to(device=hidden_states.device, dtype=self.dtype)
        try:
            inputs = tuple(named_inputs[name] for name in self.input_names)
        except KeyError as exc:
            raise KeyError(f"MiniMax-H3 HMONNX transformer is missing graph input: {exc}") from exc
        if not self.runtime.supports(*inputs):
            expected = self.runtime.expected_shapes
            actual = tuple(tuple(tensor.shape) for tensor in inputs)
            raise RuntimeError(
                "MiniMax-H3 HMONNX transformer input shapes do not match the exported graph: "
                f"expected {expected}, got {actual}"
            )
        output = self.runtime(*inputs)
        # Auto-offload may leave graph outputs on a secondary CUDA device; scheduler.step
        # expects noise preds on the same device as latents (usually the pipeline device).
        target_device = hidden_states.device

        def _align_device(value: Any) -> Any:
            if torch.is_tensor(value) and value.device != target_device:
                return value.to(device=target_device, non_blocking=True)
            return value

        if isinstance(output, tuple):
            return tuple(_align_device(item) for item in output)
        if isinstance(output, list):
            return [_align_device(item) for item in output]
        return _align_device(output)
        # return MiniMaxH3TransformerOutput(sample=output[0], audio_sample=output[1])

    def _release_fallback(self) -> None:
        if self._fallback_offloaded:
            return
        fallback = self.fallback
        self.fallback = None
        self._fallback_offloaded = True
        del fallback
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _HMONNXTextLanguageModel(nn.Module):
    def __init__(self, runtime: _LazyHMONNXRuntime, fallback: nn.Module):
        super().__init__()
        self.runtime = runtime
        self.fallback = fallback
        self.config = fallback.config
        # Re-parent host-side helpers off the heavy decoder so fallback can be freed.
        self.rotary_emb = fallback.rotary_emb
        # Qwen3VLModel.get_input_embeddings() delegates to language_model.
        self.embed_tokens = getattr(fallback, "embed_tokens", None)
        self._fallback_offloaded = False

    def get_input_embeddings(self) -> nn.Module:
        if self.embed_tokens is not None:
            return self.embed_tokens
        if self.fallback is not None and hasattr(self.fallback, "get_input_embeddings"):
            return self.fallback.get_input_embeddings()
        raise RuntimeError("MiniMax-H3 text language model embeddings are unavailable after HMONNX install")

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value
        if self.fallback is None:
            return
        if hasattr(self.fallback, "set_input_embeddings"):
            self.fallback.set_input_embeddings(value)
        else:
            self.fallback.embed_tokens = value

    def _release_fallback(self) -> None:
        if self._fallback_offloaded:
            return
        fallback = self.fallback
        self.fallback = None
        self._fallback_offloaded = True
        del fallback
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _fallback_forward(self, **kwargs) -> BaseModelOutputWithPast:
        if self.fallback is None:
            raise RuntimeError("The original text language model was released after HMONNX install/execution")
        return self.fallback(**kwargs)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (
            input_ids is not None
            or inputs_embeds is None
            or attention_mask is None
            or attention_mask.ndim != 2
            or kwargs.get("use_cache")
        ):
            return self._fallback_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                **kwargs,
            )

        expected_shapes = self.runtime.expected_shapes
        if expected_shapes is None:
            return self._fallback_forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                **kwargs,
            )
        batch_size, sequence_length, hidden_size = inputs_embeds.shape
        expected_batch, fixed_length, expected_hidden_size = expected_shapes[0]
        if (
            len(expected_shapes) != 7
            or batch_size != expected_batch
            or hidden_size != expected_hidden_size
            or sequence_length > fixed_length
        ):
            return self._fallback_forward(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                inputs_embeds=inputs_embeds,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                **kwargs,
            )

        dense_deepstack = [torch.zeros_like(inputs_embeds) for _ in range(3)]
        if deepstack_visual_embeds is not None:
            if visual_pos_masks is None or len(deepstack_visual_embeds) != len(dense_deepstack):
                return self._fallback_forward(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    visual_pos_masks=visual_pos_masks,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                    **kwargs,
                )
            try:
                for index, visual_embeds in enumerate(deepstack_visual_embeds):
                    dense_deepstack[index][visual_pos_masks] = visual_embeds.to(
                        device=inputs_embeds.device,
                        dtype=inputs_embeds.dtype,
                    )
            except (IndexError, RuntimeError):
                return self._fallback_forward(
                    input_ids=None,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    visual_pos_masks=visual_pos_masks,
                    deepstack_visual_embeds=deepstack_visual_embeds,
                    **kwargs,
                )

        if position_ids is None:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        padding_length = fixed_length - sequence_length
        if padding_length:
            inputs_embeds = torch.nn.functional.pad(inputs_embeds, (0, 0, 0, padding_length))
            attention_mask = torch.nn.functional.pad(attention_mask, (0, padding_length))
            position_ids = torch.nn.functional.pad(position_ids, (0, padding_length), value=1)
            dense_deepstack = [
                torch.nn.functional.pad(visual_embeds, (0, 0, 0, padding_length)) for visual_embeds in dense_deepstack
            ]

        rotary_cos, rotary_sin = self.rotary_emb(inputs_embeds, position_ids)
        attention_bias = build_minimax_h3_attention_bias(attention_mask, inputs_embeds.dtype)
        graph_inputs = (inputs_embeds, rotary_cos, rotary_sin, attention_bias, *dense_deepstack)
        if not self.runtime.supports(*graph_inputs):
            return self._fallback_forward(
                input_ids=None,
                attention_mask=attention_mask[:, :sequence_length],
                position_ids=position_ids[..., :sequence_length],
                inputs_embeds=inputs_embeds[:, :sequence_length],
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
                **kwargs,
            )

        hidden_states = self.runtime(*graph_inputs)
        del self.runtime
        torch.cuda.empty_cache()

        if isinstance(hidden_states, (tuple, list)):
            hidden_states = hidden_states[0]
        # Keep the fixed export length. Downstream packing/transformer consume the
        # padded rows through encoder_attention_mask instead of cropping here.
        self._release_fallback()
        captured_hidden_states = (None,) * 50 + (hidden_states,)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, hidden_states=captured_hidden_states)


class _VisionGeometrySource(nn.Module):
    """Keep only the modules needed to rebuild padded vision geometry at runtime."""

    def __init__(self, visual: nn.Module):
        super().__init__()
        self.spatial_merge_size = int(visual.spatial_merge_size)
        self.num_grid_per_side = visual.num_grid_per_side
        self.interpolation_mode = visual.interpolation_mode
        self.interpolation_align_corners = visual.interpolation_align_corners
        self.config = visual.config
        self.pos_embed = visual.pos_embed
        self.rotary_pos_emb = visual.rotary_pos_emb


class _HMONNXVisionModel(nn.Module):
    def __init__(
        self,
        runtime: _LazyHMONNXRuntime,
        fallback: nn.Module,
        max_patches: int | torch.Tensor | None = None,
        *,
        example_image_grid_thw: torch.Tensor | None = None,
    ):
        super().__init__()
        self.runtime = runtime
        self.fallback = fallback
        self.config = fallback.config
        self.spatial_merge_size = int(fallback.spatial_merge_size)
        self._dtype = fallback.dtype
        self._fallback_offloaded = False
        # Keep only geometry helpers so the heavy vision tower can be freed for demo.
        self.geometry_source = _VisionGeometrySource(fallback) if hasattr(fallback, "pos_embed") else fallback

        # Older call sites used ``_HMONNXVisionModel(runtime, fallback, image_grid_thw)``.
        if isinstance(max_patches, torch.Tensor):
            example_image_grid_thw = max_patches
            max_patches = None
        if max_patches is None:
            runtime_shapes = getattr(runtime, "expected_shapes", None)
            if runtime_shapes and runtime_shapes[0]:
                max_patches = int(runtime_shapes[0][0])
            else:
                max_patches = DEFAULT_VISION_MAX_PATCHES

        self.max_patches = int(max_patches)
        self.max_image_tokens = self.max_patches // (self.spatial_merge_size * self.spatial_merge_size)
        if example_image_grid_thw is not None:
            self.register_buffer(
                "example_image_grid_thw",
                example_image_grid_thw.to(dtype=torch.long),
                persistent=False,
            )
        else:
            self.example_image_grid_thw = None
        # Compatibility alias used by older tests/call sites.
        self.expected_image_tokens = self.max_image_tokens

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def _release_fallback(self) -> None:
        if self._fallback_offloaded:
            return
        fallback = self.fallback
        self.fallback = None
        self._fallback_offloaded = True
        del fallback
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _fallback(self, pixel_values, grid_thw=None, return_dict: bool = True, **kwargs):
        if self.fallback is None:
            raise RuntimeError("The original vision model was released after HMONNX install/execution")
        return self.fallback(pixel_values, grid_thw=grid_thw, return_dict=return_dict, **kwargs)

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor | None = None,
        return_dict: bool = True,
        **kwargs,
    ):
        # if kwargs or not return_dict or grid_thw is None:
        #     return self._fallback(pixel_values, grid_thw=grid_thw, return_dict=return_dict, **kwargs)

        try:
            graph_inputs, geometry = build_vision_encoder_graph_inputs(
                self.geometry_source,
                pixel_values,
                grid_thw,
                max_patches=self.max_patches,
                dtype=pixel_values.dtype if pixel_values.is_floating_point() else self.dtype,
            )
        except ValueError:
            return self._fallback(pixel_values, grid_thw=grid_thw, return_dict=return_dict, **kwargs)

        if not self.runtime.supports(*graph_inputs):
            return self._fallback(pixel_values, grid_thw=grid_thw, return_dict=return_dict, **kwargs)

        output = self.runtime(*graph_inputs)
        del self.runtime
        torch.cuda.empty_cache()

        if not isinstance(output, (tuple, list)) or len(output) != 4:
            raise RuntimeError(f"MiniMax-H3 vision HMONNX must return four tensors, got {type(output).__name__}")
        invalid_shapes = [tuple(tensor.shape) for tensor in output if tensor.shape[0] != self.max_image_tokens]
        if invalid_shapes:
            raise RuntimeError(
                f"MiniMax-H3 vision HMONNX must return {self.max_image_tokens} image tokens, "
                f"got shapes {invalid_shapes}"
            )

        valid_image_tokens = int(geometry["valid_image_tokens"])
        pooler_output = output[0][:valid_image_tokens]
        deepstack_features = [tensor[:valid_image_tokens] for tensor in output[1:]]
        self._release_fallback()
        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=None,
            pooler_output=pooler_output,
            deepstack_features=deepstack_features,
        )


class _HMONNXMethod:
    def __init__(
        self,
        runtime: _LazyHMONNXRuntime,
        fallback,
        input_names: tuple[str, ...],
        owner: nn.Module,
        release_names: tuple[str, ...],
        before_call=None,
    ):
        self.runtime = runtime
        self.fallback = fallback
        self.input_names = input_names
        self.owner = owner
        self.release_names = release_names
        self.before_call = before_call
        self._fallback_offloaded = False

    def _release_fallback(self) -> None:
        if self._fallback_offloaded:
            return
        fallback = self.fallback
        owner = getattr(fallback, "__self__", None)
        self.fallback = None
        self._fallback_offloaded = True
        del fallback
        for name in self.release_names:
            _release_module_reference(self.owner, name)
        self.owner = None
        del owner
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __call__(self, *inputs: torch.Tensor, **kwargs):
        if self.before_call is not None:
            self.before_call()
        original_inputs = inputs
        original_kwargs = kwargs
        if kwargs and "return_dict" not in kwargs:
            if not set(kwargs).issubset(self.input_names):
                return self.fallback(*original_inputs, **original_kwargs)
            try:
                inputs = tuple(kwargs[name] for name in self.input_names)
            except KeyError:
                return self.fallback(*original_inputs, **original_kwargs)
        if not self.runtime.supports(*inputs):
            return self.fallback(*original_inputs, **original_kwargs)
        self._release_fallback()
        output = self.runtime(*inputs)
        if self.runtime.output_list:
            return output
        return output[0] if isinstance(output, (tuple, list)) and len(output) == 1 else output


def _component_path(export_dir: Path, meta: dict[str, Any], name: str) -> Path | None:
    component = meta.get("components", {}).get(name)
    if not isinstance(component, dict) or not component.get("hmonnx_file"):
        return None
    path = (export_dir / str(component["hmonnx_file"])).resolve()
    if export_dir not in path.parents or not path.is_file():
        raise FileNotFoundError(f"Invalid HMONNX path for {name}: {path}")
    return path


def _component_shapes(meta: dict[str, Any], name: str) -> tuple[tuple[int, ...], ...] | None:
    component = meta.get("components", {}).get(name)
    inputs = component.get("inputs") if isinstance(component, dict) else None
    if not isinstance(inputs, list):
        return None
    shapes = [item.get("shape") for item in inputs if isinstance(item, dict)]
    if len(shapes) != len(inputs) or not all(isinstance(shape, list) for shape in shapes):
        return None
    return tuple(tuple(int(value) for value in shape) for shape in shapes)


def _load_meta(export_dir: Path) -> dict[str, Any]:
    meta_path = export_dir / "export_meta_info.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing MiniMax-H3 export metadata: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _install_method_component(
    owner: Any,
    method_name: str,
    runtime: _LazyHMONNXRuntime,
    input_names: tuple[str, ...],
    release_names: tuple[str, ...],
    before_call=None,
) -> None:
    setattr(
        owner,
        method_name,
        _HMONNXMethod(runtime, getattr(owner, method_name), input_names, owner, release_names, before_call),
    )


def install_hmonnx_components(
    pipe: Any,
    export_dir: str | Path,
    device: torch.device,
    components: tuple[str, ...] | list[str] | None = None,
    exec_devices: list[torch.device | str | int] | tuple[torch.device | str | int, ...] | None = None,
) -> tuple[str, ...]:
    requested = set(components) if components is not None else set(_SUPPORTED_COMPONENTS)
    supported = set(_SUPPORTED_COMPONENTS)
    unknown = sorted(requested - supported)
    if unknown:
        raise ValueError(f"Unsupported MiniMax-H3 HMONNX components: {unknown}; expected {_SUPPORTED_COMPONENTS}")
    if not requested:
        raise ValueError("At least one MiniMax-H3 HMONNX component must be selected")
    if components is not None:
        requested = expand_minimax_h3_components(requested)

    export_dir = Path(export_dir).resolve()
    meta = _load_meta(export_dir)
    available = {
        component for component in _SUPPORTED_COMPONENTS if _component_path(export_dir, meta, component) is not None
    }
    if components is None:
        requested = available
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"Export is missing requested MiniMax-H3 HMONNX components: {missing}")
    if not requested:
        raise ValueError("Export contains no supported MiniMax-H3 HMONNX components")

    primary_device = torch.device(device)
    runtime_exec_devices = _normalize_exec_devices(primary_device, exec_devices)

    vae = pipe.vae
    decode_prerequisites_released = False

    def release_decode_prerequisites() -> None:
        nonlocal decode_prerequisites_released
        if decode_prerequisites_released:
            return
        _release_before_vae_decode(pipe, vae)
        decode_prerequisites_released = True

    def _make_runtime(
        path: Path,
        shapes: tuple[tuple[int, ...], ...] | None,
        *,
        output_list: bool = False,
    ) -> _LazyHMONNXRuntime:
        return _LazyHMONNXRuntime(
            path,
            primary_device,
            shapes,
            output_list=output_list,
            exec_devices=runtime_exec_devices,
        )

    for component in sorted(requested - {"vae_encoder", "vae_decoder"}):
        path = _component_path(export_dir, meta, component)
        if component in ("audio_vae_decoder"):
            runtime = _make_runtime(path, _component_shapes(meta, component), output_list=True)
        else:
            runtime = _make_runtime(path, _component_shapes(meta, component))
        input_names = tuple(meta["components"][component].get("input_names", ()))
        if component == "audio_vae_encoder":
            _install_method_component(
                pipe.audio_vae,
                "encode",
                runtime,
                input_names,
                ("encoder", "pre_block", "mean_proj", "logs_proj"),
            )
        elif component == "audio_vae_decoder":
            _install_method_component(
                pipe.audio_vae,
                "decode",
                runtime,
                input_names,
                ("dec_in_proj", "decoder"),
                release_decode_prerequisites,
            )
        elif component == "transformer":
            component_meta = meta["components"][component]
            fixed_adaln_meta = component_meta.get("fixed_adaln")
            if not isinstance(fixed_adaln_meta, dict):
                fixed_adaln_meta = None
            pipe.transformer = _HMONNXTransformerModule(
                runtime,
                pipe.transformer,
                input_names,
                fixed_adaln_meta=fixed_adaln_meta,
            )
            # Keep only time_proj/time_embedder/rope for host precompute; drop the FP body.
            pipe.transformer._release_fallback()
        elif component == "vision_encoder":
            model = getattr(pipe.text_encoder, "model", None)
            visual = getattr(model, "visual", None)
            if not isinstance(visual, nn.Module):
                raise TypeError("MiniMax-H3 text encoder must expose model.visual")
            component_meta = meta["components"][component]
            max_patches = component_meta.get("vision_max_patches")
            if max_patches is None:
                inputs = component_meta.get("inputs")
                if isinstance(inputs, list) and inputs and isinstance(inputs[0], dict):
                    shape = inputs[0].get("shape")
                    if isinstance(shape, list) and shape:
                        max_patches = shape[0]
            if max_patches is None:
                max_patches = DEFAULT_VISION_MAX_PATCHES
            example_grid = component_meta.get("image_grid_thw")
            example_tensor = (
                torch.tensor(example_grid, device=primary_device, dtype=torch.long)
                if isinstance(example_grid, list)
                else None
            )
            model.visual = _HMONNXVisionModel(
                runtime,
                visual,
                max_patches=int(max_patches),
                example_image_grid_thw=example_tensor,
            )
            # Demo path must not keep the original vision tower in memory.
            model.visual._release_fallback()
        elif component == "text_encoder":
            model = getattr(pipe.text_encoder, "model", None)
            language_model = getattr(model, "language_model", None)
            if not isinstance(language_model, nn.Module):
                raise TypeError("MiniMax-H3 text encoder must expose model.language_model")
            model.language_model = _HMONNXTextLanguageModel(runtime, language_model)
            # Keep only embed_tokens/rotary_emb; drop the heavy decoder body immediately.
            model.language_model._release_fallback()

    encoder_path = _component_path(export_dir, meta, "vae_encoder") if "vae_encoder" in requested else None
    decoder_path = _component_path(export_dir, meta, "vae_decoder") if "vae_decoder" in requested else None
    encoder = _make_runtime(encoder_path, _component_shapes(meta, "vae_encoder")) if encoder_path else None
    decoder = _make_runtime(decoder_path, _component_shapes(meta, "vae_decoder")) if decoder_path else None

    if encoder is not None:
        fp_tile_encoder = MiniMaxH3VAEEncoderTileWrapper(vae.encoder, vae.quant_conv)
        tile_encoder = _HMONNXVAEEncoderTile(encoder, fp_tile_encoder, vae)
        encoder_wrapper = MiniMaxH3VAEEncoderWrapper(
            vae,
            tile_encoder=tile_encoder,
            wrap_modules=False,
        )
        vae._encode = encoder_wrapper.forward

    if decoder is not None:
        decoder_body_offloaded = False

        def hmonnx_decode_tile(tile: torch.Tensor) -> torch.Tensor:
            nonlocal decoder_body_offloaded
            hidden_states = vae.post_quant_conv(tile)
            rotary_cos, rotary_sin = build_vae_decoder_rotary_emb(vae.decoder, hidden_states)
            if not decoder.supports(tile, rotary_cos, rotary_sin):
                return vae.decoder(hidden_states)
            if not decoder_body_offloaded:
                decoder_module = vae.decoder
                for name in tuple(decoder_module._modules):
                    if name != "rope":
                        _release_module_reference(decoder_module, name)
                decoder_body_offloaded = True
            output = decoder(tile, rotary_cos.half(), rotary_sin.half())
            return output[0] if isinstance(output, (tuple, list)) else output

        def hmonnx_decode_clip(z: torch.Tensor) -> torch.Tensor:
            release_decode_prerequisites()
            _restore_vae_decode_modules(vae, device)
            if not vae.use_tiling:
                return hmonnx_decode_tile(z)

            height = z.shape[-2] * vae.spatial_compression_ratio
            width = z.shape[-1] * vae.spatial_compression_ratio
            y_indices, y_lengths, y_overlaps = vae._split_tiles(
                height,
                vae.tile_sample_min_height,
                vae.tile_sample_min_overlap_height,
            )
            x_indices, x_lengths, x_overlaps = vae._split_tiles(
                width,
                vae.tile_sample_min_width,
                vae.tile_sample_min_overlap_width,
            )

            ratio = vae.spatial_compression_ratio
            rows = []
            for i_pos, i_len in zip(y_indices, y_lengths, strict=True):
                row = []
                for j_pos, j_len in zip(x_indices, x_lengths, strict=True):
                    tile = z[
                        ...,
                        i_pos // ratio : i_pos // ratio + i_len // ratio,
                        j_pos // ratio : j_pos // ratio + j_len // ratio,
                    ]
                    row.append(hmonnx_decode_tile(tile))
                rows.append(row)

            return vae._stitch_tiles(rows, y_overlaps, x_overlaps)

        vae._decode_clip = hmonnx_decode_clip
    # else:
    #     original_decode_clip = vae._decode_clip

    #     def decode_clip_with_cleanup(*args, **kwargs):
    #         release_decode_prerequisites()
    #         return original_decode_clip(*args, **kwargs)

    #     vae._decode_clip = decode_clip_with_cleanup

    # Stock/other-env diffusers may not emit encoder_attention_mask. Install the
    # local fixed-512 normalize + mask path onto the live `pipe._blocks` tree.
    install_fixed_length_text_conditioning(pipe)
    _install_pipeline_cleanup(pipe)
    return tuple(component for component in _SUPPORTED_COMPONENTS if component in requested)


def install_hmonnx_vae(
    pipe: Any,
    export_dir: str | Path,
    device: torch.device,
    components: tuple[str, ...] | list[str] | None = None,
    exec_devices: list[torch.device | str | int] | tuple[torch.device | str | int, ...] | None = None,
) -> tuple[str, ...]:
    requested = None if components is None else tuple(components)
    if requested is not None and set(requested) - {"vae_encoder", "vae_decoder"}:
        raise ValueError("install_hmonnx_vae only accepts vae_encoder and vae_decoder")
    return install_hmonnx_components(pipe, export_dir, device, requested, exec_devices=exec_devices)


__all__ = ["install_hmonnx_components", "install_hmonnx_vae"]
