from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from typing import Any

import torch
from torch import nn


DEFAULT_LORA_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "models" / "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"
)
DEFAULT_LORA_ALPHA = 8
DEFAULT_LORA_SCALE = 1.0
DEFAULT_FIXED_POINT_CALIBRATION_PATH = Path(__file__).resolve().parent / "minimax_h3_fixed_point_calibration.json"
DEFAULT_MEMORY_RESERVE_MARGIN = "12GB"


class _ExportABITransformerAdapter(nn.Module):
    """Accept native pipeline kwargs and call the current export-ABI wrapper body."""

    def __init__(
        self,
        wrapped: nn.Module,
        *,
        config: Any,
        dtype: torch.dtype,
        log_step_index: bool = False,
        log_step_limit: int = 8,
        pad_temb_to_3: bool = False,
    ):
        super().__init__()
        self.wrapped = wrapped
        body = wrapped.transformer
        # DynamicModule conversion drops ConfigMixin.config; keep the native one for
        # packing / canvas_multiple (patch_size) readers on the modular pipeline.
        if config is None:
            raise ValueError(
                "MiniMax-H3 wrapper adapter requires the native transformer.config; "
                "capture it before MiniMaxH3TransformerWrapper(...)."
            )
        self.config = config
        self._dtype = dtype if dtype is not None else getattr(body, "dtype", torch.float16)
        self.time_proj = body.time_proj
        self.time_embedder = body.time_embedder
        self.rope = body.rope
        self.log_step_index = log_step_index
        self.log_step_limit = log_step_limit
        self._pad_temb_to_3 = bool(pad_temb_to_3)
        self._step_logs = 0
        meta = getattr(body, "_minimax_fixed_adaln_meta", None) or {}
        self._fixed_adaln_unique_timesteps = [list(map(float, row)) for row in (meta.get("unique_timesteps") or [])]
        self._fixed_adaln_num_evals = int(meta.get("num_denoise_evals") or len(self._fixed_adaln_unique_timesteps) or 0)
        self._minimax_step_counter = 0

    @property
    def dtype(self):
        return self._dtype

    @property
    def _body(self) -> nn.Module:
        return self.wrapped.transformer

    @property
    def _block_cpu_offload_enabled(self) -> bool:
        return bool(getattr(self._body, "enable_block_cpu_offload", False))

    @property
    def device(self) -> torch.device:
        # ComponentsManager CustomOffloadHook compares hook.model.device with execution_device.
        # With block offload, blocks stay on CPU while shared modules may already be on CUDA;
        # report a shared module device so pre_forward does not think the whole tree is on CPU.
        body = self._body
        for child_name, child in body.named_children():
            if child_name == "transformer_blocks":
                continue
            try:
                return next(child.parameters()).device
            except StopIteration:
                continue
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def get_memory_footprint(self, return_buffers: bool = True) -> int:
        # AutoOffloadStrategy requires ModelMixin-like footprint APIs on managed components.
        # When blocks are parked on CPU, only the shared modules remain live on the execution
        # device; the parked transformer blocks are intentionally excluded from the live footprint.
        body = self._body
        if self._block_cpu_offload_enabled:
            mem = 0
            for child_name, child in body.named_children():
                if child_name == "transformer_blocks":
                    continue
                mem += sum(param.nelement() * param.element_size() for param in child.parameters())
                if return_buffers:
                    mem += sum(buf.nelement() * buf.element_size() for buf in child.buffers())
            return mem
        mem = sum(param.nelement() * param.element_size() for param in self.parameters())
        if return_buffers:
            mem += sum(buf.nelement() * buf.element_size() for buf in self.buffers())
        return mem

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
                "MiniMax-H3 wrapper adapter is missing baked AdaLN unique-timestep metadata; "
                "call install_fixed_adaln_lut before wrapping."
            )
        step_index = int(self._minimax_step_counter % self._fixed_adaln_num_evals)
        self._minimax_step_counter = (step_index + 1) % self._fixed_adaln_num_evals
        return step_index

    def _prepare_block_cpu_offload(self, device: torch.device | str) -> None:
        from xhmodel_merak.xh_other_model.models.minimax_h3.transformer_wrapper import (
            prepare_transformer_block_cpu_offload,
        )

        prepare_transformer_block_cpu_offload(self._body, device)

    def to(self, *args, **kwargs):
        # CustomOffloadHook uses module.to(execution_device). The default nn.Module.to would
        # pull every transformer block onto CUDA at once and defeat enable_block_cpu_offload.
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if self._block_cpu_offload_enabled and device is not None:
            if device.type == "cpu":
                for _, child in self._body.named_children():
                    child.to(device=device, non_blocking=non_blocking)
            else:
                self._prepare_block_cpu_offload(device)
            if dtype is None and convert_to_format is None:
                return self
            # Cast in place without collapsing the mixed CPU/CUDA placement.
            return super().to(dtype=dtype, non_blocking=non_blocking, memory_format=convert_to_format)
        return super().to(*args, **kwargs)

    def cuda(self, device: int | torch.device | None = None):
        if self._block_cpu_offload_enabled:
            if device is None:
                target = torch.device("cuda", torch.cuda.current_device())
            elif isinstance(device, torch.device):
                target = device
            else:
                target = torch.device("cuda", device)
            self._prepare_block_cpu_offload(target)
            return self
        return super().cuda(device)

    def cpu(self):
        if self._block_cpu_offload_enabled:
            for _, child in self._body.named_children():
                child.cpu()
            return self
        return super().cpu()

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
        from xhmodel_merak.xh_other_model.models.minimax_h3.transformer_minimax_h3 import (
            MiniMaxH3TransformerOutput,
        )
        from xhmodel_merak.xh_other_model.models.minimax_h3.transformer_wrapper import (
            build_minimax_h3_transformer_conditioning,
            build_pack_gather_indices,
            build_packed_attention_mask,
        )

        if attention_kwargs:
            raise ValueError("MiniMax-H3 wrapper FP path does not support attention_kwargs")
        if encoder_attention_mask is None:
            raise ValueError("MiniMax-H3 wrapper FP path requires encoder_attention_mask")

        # The export ABI is scalar step_index-only: the host still computes the AdaLN/rotary
        # conditioning tables, but the graph itself no longer consumes a live temb tensor.
        _, adaln_indices, rotary_cos, rotary_sin = build_minimax_h3_transformer_conditioning(
            self,
            timestep=timestep,
            timestep_indices=timestep_indices,
            token_tags=token_tags,
            position_ids=position_ids,
        )
        step_index_value = self._resolve_step_index(timestep)
        step_index = torch.tensor(step_index_value, device=hidden_states.device, dtype=torch.int64)
        if self.log_step_index and self._step_logs < self.log_step_limit:
            print(
                "wrapper step_index: "
                f"step={step_index_value} "
                f"unique={int(timestep.numel())} "
                f"timestep_indices_max={int(timestep_indices.max())} "
                f"adaln_indices_max={int(adaln_indices.max())}",
                flush=True,
            )
            self._step_logs += 1

        # Export wrapper does not cast; host must align conditioning tensors to body dtype.
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
        pack_gather_indices = build_pack_gather_indices(
            text_indices,
            video_indices,
            audio_indices,
            sequence_length=sequence_length,
        )

        # text_indices stay host-only; export wrapper ABI no longer consumes them.
        video_out, audio_out = self.wrapped(
            hidden_states.to(torch.float16),
            audio_hidden_states.to(torch.float16),
            encoder_hidden_states.to(torch.float16),
            step_index,
            timestep_indices.to(torch.int32),
            adaln_indices.to(torch.int32),
            rotary_cos.to(torch.float16),
            rotary_sin.to(torch.float16),
            video_indices.to(torch.int32),
            audio_indices.to(torch.int32),
            pack_gather_indices.to(torch.int32),
            encoder_attention_mask.to(torch.float16),
            packed_attention_mask.to(torch.float16),
        )
        if return_dict:
            return MiniMaxH3TransformerOutput(sample=video_out, audio_sample=audio_out)
        return video_out, audio_out


def _module_param_device(module: nn.Module | None) -> str:
    if module is None:
        return "none"
    try:
        return str(next(module.parameters()).device)
    except StopIteration:
        return "no-params"


def _component_name_from_model_id(model_id: str | None) -> str:
    if not model_id:
        return ""
    # ComponentsManager ids look like "text_encoder_<python_id>".
    return str(model_id).rsplit("_", 1)[0]


class _PreferTextEncoderOffloadStrategy:
    """Force text_encoder(+vision) off GPU before transformer loads.

    Diffusers AutoOffloadStrategy only offloads when
    ``incoming_footprint >= free_vram - margin``. After prompt/vision encode,
    text_encoder stays on CUDA; with the default 3GB margin it often decides
    "space is enough" and leaves text+transformer coresident → OOM.
    """

    def __init__(self, memory_reserve_margin: str = DEFAULT_MEMORY_RESERVE_MARGIN):
        from diffusers.modular_pipelines.components_manager import AutoOffloadStrategy

        self.base = AutoOffloadStrategy(memory_reserve_margin=memory_reserve_margin)

    def __call__(self, hooks, model_id, model, execution_device):
        selected = list(self.base(hooks, model_id, model, execution_device))
        incoming_name = _component_name_from_model_id(model_id)
        if incoming_name != "transformer":
            return selected

        selected_ids = {hook.model_id for hook in selected}
        forced = []
        for hook in hooks:
            if hook.model_id in selected_ids:
                continue
            if _component_name_from_model_id(hook.model_id) == "text_encoder":
                forced.append(hook)
        if forced:
            print(
                "offload strategy: forcing text_encoder(+vision) to CPU before transformer load: "
                + ",".join(hook.model_id for hook in forced),
                flush=True,
            )
        return selected + forced


def _vision_encoder_module(pipe) -> nn.Module | None:
    text_encoder = getattr(pipe, "text_encoder", None)
    model = getattr(text_encoder, "model", None)
    return getattr(model, "visual", None)


def _log_cuda_memory(stage: str, device: torch.device | None = None) -> None:
    if not torch.cuda.is_available():
        return
    if device is None or device.type != "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    else:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    allocated_bytes = torch.cuda.memory_allocated(device)
    reserved_bytes = torch.cuda.memory_reserved(device)
    gib = 1024**3
    print(
        f"[{stage}] cuda-memory "
        f"free={free_bytes / gib:.2f}GiB total={total_bytes / gib:.2f}GiB "
        f"allocated={allocated_bytes / gib:.2f}GiB reserved={reserved_bytes / gib:.2f}GiB",
        flush=True,
    )


def _log_component_devices(pipe, stage: str, device: torch.device | None = None) -> None:
    text_encoder = getattr(pipe, "text_encoder", None)
    vision = _vision_encoder_module(pipe)
    transformer = getattr(pipe, "transformer", None)
    manager = getattr(pipe, "_components_manager", None)
    managed = []
    if manager is not None and getattr(manager, "components", None):
        managed = [manager._id_to_name(component_id) for component_id in manager.components]
    print(
        f"[{stage}] devices "
        f"text_encoder={_module_param_device(text_encoder)} "
        f"vision={_module_param_device(vision)} "
        f"transformer={_module_param_device(transformer)} "
        f"te_hook={type(getattr(text_encoder, '_hf_hook', None)).__name__} "
        f"tr_hook={type(getattr(transformer, '_hf_hook', None)).__name__} "
        f"managed={managed}",
        flush=True,
    )
    _log_cuda_memory(stage, device)


def _module_footprint_gib(module: nn.Module | None) -> float:
    if module is None:
        return 0.0
    bytes_ = sum(param.nelement() * param.element_size() for param in module.parameters())
    bytes_ += sum(buf.nelement() * buf.element_size() for buf in module.buffers())
    return bytes_ / (1024**3)


def _force_release_text_encoder(pipe, *, device: torch.device | None = None) -> None:
    """Drop text_encoder(+vision) right after prompt embeds are computed.

    Auto-offload only kicks residents when another hooked module first moves to CUDA.
    MiniMax-H3 encode manually pulls the whole text_encoder (including visual) up and
    leaves it there through VAE encode; waiting until transformer pre_forward is too late.
    """
    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is None:
        print("[after-prompt-embeds] text_encoder already absent", flush=True)
        return

    transformer = getattr(pipe, "transformer", None)
    print(
        "[before-release-text-encoder] footprints "
        f"text_encoder={_module_footprint_gib(text_encoder):.2f}GiB "
        f"transformer={_module_footprint_gib(transformer):.2f}GiB "
        f"(H3 transformer weights alone are ~61.7GiB fp16/bf16)",
        flush=True,
    )
    _log_component_devices(pipe, "before-release-text-encoder", device)
    # Prefer unload: removes from ComponentsManager so it cannot be pulled back up later.
    if hasattr(pipe, "unload_components"):
        pipe.unload_components("text_encoder")
    else:
        text_encoder.to("cpu")
        pipe.text_encoder = None

    # Qwen3-VL may stash non-parameter tensors (e.g. rope_deltas) that .to("cpu")/unload miss.
    model = getattr(text_encoder, "model", None)
    if model is not None and isinstance(getattr(model, "rope_deltas", None), torch.Tensor):
        model.rope_deltas = None

    # Drop the local reference before measuring; otherwise allocated still counts TE.
    del model
    del text_encoder
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _log_component_devices(pipe, "after-release-text-encoder", device)
    print(
        "[after-prompt-embeds] forced release of text_encoder(+vision); "
        "later ~62-70GiB during denoise is expected from transformer weights alone",
        flush=True,
    )


def _install_prompt_embed_release_hook(pipe, *, device: torch.device | None = None) -> None:
    from diffusers.modular_pipelines.minimax_h3 import encoders as minimax_h3_encoders

    original_get_qwen3vl_prompt_embeds = minimax_h3_encoders.get_qwen3vl_prompt_embeds

    def _wrapped_get_qwen3vl_prompt_embeds(*args, **kwargs):
        prompt_embeds = original_get_qwen3vl_prompt_embeds(*args, **kwargs)
        _force_release_text_encoder(pipe, device=device)
        return prompt_embeds

    minimax_h3_encoders.get_qwen3vl_prompt_embeds = _wrapped_get_qwen3vl_prompt_embeds
    print("installed prompt-embed hook: unload text_encoder(+vision) immediately after encode", flush=True)


def _remove_manager_components(pipe, names: tuple[str, ...]) -> None:
    """Drop named components from ComponentsManager so auto-offload hooks can be rebuilt."""
    manager = getattr(pipe, "_components_manager", None)
    if manager is None or not hasattr(manager, "components") or not hasattr(manager, "remove"):
        return
    component_ids = [
        component_id for component_id in tuple(manager.components) if manager._id_to_name(component_id) in names
    ]
    for component_id in component_ids:
        manager.remove(component_id)


def _register_managed_transformer(pipe, transformer: nn.Module) -> None:
    """Replace pipe.transformer through ModularPipeline so ComponentsManager keeps offload hooks."""
    manager = getattr(pipe, "_components_manager", None)
    auto_offload_enabled = bool(getattr(manager, "_auto_offload_enabled", False))
    offload_device = getattr(manager, "_auto_offload_device", None)
    offload_strategy = getattr(manager, "_offload_strategy", None)

    # Direct `pipe.transformer = ...` bypasses ComponentsManager: the new module has no
    # CustomOffloadHook, so it never moves to CUDA and never forces text/vision back to CPU.
    if auto_offload_enabled:
        manager.disable_auto_cpu_offload()
    try:
        _remove_manager_components(pipe, ("transformer",))
        if hasattr(pipe, "update_components"):
            pipe.update_components(transformer=transformer)
        else:
            pipe.transformer = transformer
            if manager is not None and hasattr(manager, "add"):
                manager.add("transformer", transformer, getattr(pipe, "_collection", None))
    finally:
        if auto_offload_enabled:
            # Pass the same strategy object back; enable() defaults margin to 3GB otherwise.
            manager.enable_auto_cpu_offload(offload_device, offload_strategy=offload_strategy)


def install_transformer_wrapper(
    pipe,
    *,
    transformer: nn.Module,
    num_inference_steps: int,
    log_step_index: bool,
    device: torch.device | str | None = None,
) -> _ExportABITransformerAdapter:
    from xhmodel_merak.xh_other_model.models.minimax_h3.transformer_minimax_h3 import (
        MiniMaxH3Transformer3DModel,
    )
    from xhmodel_merak.xh_other_model.models.minimax_h3.transformer_wrapper import (
        MiniMaxH3TransformerWrapper,
        install_fixed_adaln_lut,
    )
    from xhquant.utils.registry.dynamic_module import DynamicModule

    # wrap_transformer keys off Merak classes only; native diffusers MiniMax-H3 is a silent no-op.
    if type(transformer) is not MiniMaxH3Transformer3DModel:
        raise TypeError(
            "MiniMaxH3TransformerWrapper requires Merak MiniMaxH3Transformer3DModel, got "
            f"{type(transformer).__module__}.{type(transformer).__name__}"
        )

    # Capture before DynamicModule conversion: wrap_transformer drops ConfigMixin.config.
    native_config = getattr(transformer, "config", None)
    native_dtype = getattr(transformer, "dtype", torch.float16)
    if native_config is None or not hasattr(native_config, "patch_size"):
        raise AttributeError(
            "native transformer.config.patch_size is required for MiniMaxH3ModularPipeline.canvas_multiple"
        )
    # Bake AdaLN / final-norm constants before wrap so the export ABI uses step_index.
    tables = install_fixed_adaln_lut(
        transformer,
        num_inference_steps=int(num_inference_steps),
        device=device,
        dtype=torch.float16,
    )
    print(
        "installed fixed AdaLN LUT: "
        f"num_inference_steps={tables['num_inference_steps']} "
        f"num_denoise_evals={tables['num_denoise_evals']} "
        f"fixed_temb_timesteps={tables['fixed_temb_timesteps']}",
        flush=True,
    )
    wrapped = MiniMaxH3TransformerWrapper(transformer).eval()
    body = wrapped.transformer
    if not isinstance(body, DynamicModule):
        raise RuntimeError(
            "MiniMaxH3TransformerWrapper did not convert the Merak transformer to a DynamicModule; "
            f"got {type(body).__module__}.{type(body).__name__}"
        )
    if not hasattr(body.transformer_blocks[0], "fixed_point_mul_add_1"):
        raise RuntimeError("wrapped transformer is missing FixedPointMulAdd ops after wrap")
    adapter = _ExportABITransformerAdapter(
        wrapped,
        config=native_config,
        dtype=native_dtype,
        log_step_index=log_step_index,
    )
    _register_managed_transformer(pipe, adapter)
    if getattr(pipe, "transformer", None) is not adapter:
        raise RuntimeError("MiniMax-H3 transformer wrapper was not installed as the pipeline transformer")
    if getattr(pipe.transformer, "_hf_hook", None) is None:
        raise RuntimeError(
            "installed transformer adapter is missing CustomOffloadHook; text/vision will not be forced off GPU"
        )
    _log_component_devices(pipe, "after-wrapper-install")
    print(f"wrapper body={type(body).__name__}", flush=True)
    return adapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run MiniMax-H3 fl2va with the current Merak transformer wrapper (export ABI + host precompute). "
            "This isolates wrapper FP quality from HMONNX."
        )
    )
    parser.add_argument("--model-dir", required=True, help="MiniMax-H3 HF / diffusers model directory.")
    parser.add_argument("--workflow", default="fl2va", choices=("t2va", "fl2va"))
    parser.add_argument(
        "--prompt",
        default=(
            "The character in the picture is sitting inside an eggshell, with flames spewing from the bottom of the "
            "eggshell, causing him to fly up and he (S1) says, <d>[Chinese] 你好， 我的名字叫宇宙无敌大帅男许晨</d>."
        ),
    )
    parser.add_argument("--image", default="examples_merak/aigc/minimaxh3/astronaut.jpg")
    parser.add_argument("--output", default="examples_merak/aigc/minimaxh3/wrapper_output.mp4")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Execution device for ComponentsManager auto offload. Prefer a free GPU; cuda:0 may already be occupied.",
    )
    parser.add_argument(
        "--resolution",
        choices=("demo", "480p", "720p", "720p_16x9"),
        default=None,
        help=(
            "Named resolution preset. demo=608x352, 480p=832x480, "
            "720p=1344x768 (official short-side 768), 720p_16x9=1280x704. "
            "Overrides --height/--width when set."
        ),
    )
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=608)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--lora-path",
        type=Path,
        default=DEFAULT_LORA_PATH,
        help="Optional FL2V LoRA path. Missing path is skipped unless --require-lora is set.",
    )
    parser.add_argument(
        "--require-lora",
        action="store_true",
        help="Fail if --lora-path does not exist (default: skip missing LoRA).",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Do not load LoRA even if --lora-path exists.",
    )
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-scale", type=float, default=DEFAULT_LORA_SCALE)
    parser.add_argument(
        "--memory-reserve-margin",
        default=DEFAULT_MEMORY_RESERVE_MARGIN,
        help="AutoOffloadStrategy reserve margin (default: 12GB; Diffusers default 3GB is too small for H3).",
    )
    parser.add_argument(
        "--fixed-point-calibration-path",
        type=Path,
        default=DEFAULT_FIXED_POINT_CALIBRATION_PATH,
        help="Calibration JSON applied after wrapping so FixedPointMulAdd picks it up.",
    )
    parser.add_argument(
        "--disable-fixed-point-calibration",
        action="store_true",
        help="Do not load fixed-point calibration after wrapping.",
    )
    parser.add_argument(
        "--log-step-index",
        action="store_true",
        help="Print the first few resolved AdaLN LUT step_index values during denoise.",
    )
    parser.add_argument(
        "--block-cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Stream one transformer block at a time (default: true). "
            "Required on ~80GiB GPUs because the full H3 transformer weights alone are ~61.7GiB."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workflow == "fl2va" and not args.image:
        raise ValueError("--image is required for fl2va")
    if args.workflow == "t2va" and args.image:
        raise ValueError("--image is only valid for fl2va")
    if args.no_lora and args.require_lora:
        raise ValueError("--no-lora and --require-lora are mutually exclusive")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    from diffusers import ComponentsManager
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ModularPipeline
    from diffusers.utils import load_image
    from diffusers.utils.export_utils import encode_video

    from xhmodel_merak.xh_other_model.models.minimax_h3.lora import load_minimax_h3_lora
    from xhmodel_merak.xh_other_model.models.minimax_h3.resolution import get_resolution_preset
    from xhmodel_merak.xh_other_model.models.minimax_h3.transformer_minimax_h3 import (
        MiniMaxH3Transformer3DModel,
        load_fixed_point_calibration,
    )

    if args.resolution is not None:
        preset = get_resolution_preset(args.resolution)
        args.height = preset.height
        args.width = preset.width
        print(
            f"resolution preset={preset.name}: {preset.width}x{preset.height} "
            f"frames={args.num_frames} Nv={preset.num_video_tokens} "
            f"Na={preset.num_audio_tokens} S={preset.packed_sequence_length}",
            flush=True,
        )

    device = torch.device(args.device)
    if device.type == "cuda":
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        print(
            f"selected device {device}: free={free_bytes / 1024**3:.2f}GiB / total={total_bytes / 1024**3:.2f}GiB",
            flush=True,
        )
        if free_bytes < 20 * 1024**3:
            print(
                f"warning: {device} has little free VRAM; current machine often has VLLM on cuda:0-3. "
                "Prefer an idle GPU such as --device cuda:4 or cuda:5.",
                flush=True,
            )

    manager = ComponentsManager()
    offload_strategy = _PreferTextEncoderOffloadStrategy(memory_reserve_margin=args.memory_reserve_margin)
    manager.enable_auto_cpu_offload(device=str(device), offload_strategy=offload_strategy)
    pipe = MiniMaxH3ModularPipeline.from_pretrained(
        args.model_dir,
        workflow=args.workflow,
        components_manager=manager,
    )
    # Match converter: keep the native diffusers transformer out of the pipe and install Merak+wrapper.
    component_names = [name for name in pipe._component_specs if name != "transformer"]
    pipe.load_components(
        names=component_names,
        dtype={
            "default": torch.float16,
            "vae": torch.float16,
            "audio_vae": torch.float16,
            "text_encoder": torch.float16,
        },
    )
    _log_component_devices(pipe, "after-load-components")

    # Wrapper DynamicModule conversion keys off the Merak transformer classes.
    # Keep this replacement inside ComponentsManager so auto CPU offload hooks stay valid.
    merak_transformer = MiniMaxH3Transformer3DModel.from_pretrained(
        args.model_dir,
        subfolder="transformer",
        torch_dtype=torch.float16,
    )
    if args.no_lora:
        print("LoRA disabled via --no-lora", flush=True)
    else:
        lora_path = Path(args.lora_path).expanduser()
        if not lora_path.is_absolute():
            lora_path = Path.cwd() / lora_path
        if lora_path.is_file():
            load_minimax_h3_lora(
                merak_transformer,
                lora_path,
                lora_alpha=args.lora_alpha,
                lora_scale=args.lora_scale,
            )
        elif args.require_lora:
            raise FileNotFoundError(f"LoRA checkpoint does not exist: {lora_path}")
        else:
            print(f"LoRA not found, continuing without LoRA: {lora_path}", flush=True)

    adapter = install_transformer_wrapper(
        pipe,
        transformer=merak_transformer,
        num_inference_steps=args.num_inference_steps,
        log_step_index=args.log_step_index,
        device=device,
    )
    # FixedPointMulAdd is created during wrap; apply calibration afterwards so the ops refresh.
    if not args.disable_fixed_point_calibration:
        calibration_path = Path(str(args.fixed_point_calibration_path)).expanduser()
        if not calibration_path.is_absolute():
            calibration_path = Path.cwd() / calibration_path
        if not calibration_path.is_file():
            raise FileNotFoundError(f"fixed-point calibration not found: {calibration_path}")
        body = adapter.wrapped.transformer
        load_fixed_point_calibration(body, calibration_path)
        block0 = body.transformer_blocks[0]
        attn_cfg = tuple(block0.fixed_point_mul_add_1.config)
        ffn_cfg = tuple(block0.fixed_point_mul_add_3.config)
        if attn_cfg != tuple(block0.fixed_point_configs[1]) or ffn_cfg != tuple(block0.fixed_point_configs[3]):
            raise RuntimeError(
                "fixed-point calibration did not refresh wrapped FixedPointMulAdd configs: "
                f"ops=({attn_cfg}, {ffn_cfg}) stored=({block0.fixed_point_configs[1]}, {block0.fixed_point_configs[3]})"
            )
        if attn_cfg == (8, 8, 8, 8, 0) and ffn_cfg == (8, 8, 8, 8, 0):
            raise RuntimeError(
                "fixed-point calibration appears inactive after wrap; FixedPointMulAdd still uses defaults"
            )
        print(
            f"loaded fixed-point calibration after wrap: {calibration_path} attn={attn_cfg} ffn={ffn_cfg}",
            flush=True,
        )
    else:
        print("fixed-point calibration disabled", flush=True)

    if args.block_cpu_offload:
        # Must run after ComponentsManager init_hook. Adapter.to() is overridden so later
        # CustomOffloadHook onloads keep shared modules on device and blocks on CPU.
        adapter._prepare_block_cpu_offload(device)
        parked = sum(
            1
            for block in adapter.wrapped.transformer.transformer_blocks
            if next(block.parameters()).device.type == "cpu"
        )
        print(
            f"enabled transformer block CPU offload on {device}: "
            f"parked_blocks={parked}/{len(adapter.wrapped.transformer.transformer_blocks)} "
            f"adapter_device={adapter.device} footprint_gib={adapter.get_memory_footprint() / 1024**3:.2f}",
            flush=True,
        )
        _log_component_devices(pipe, "after-block-cpu-offload", device)
    else:
        print("transformer block CPU offload disabled", flush=True)

    print(
        "installed Merak transformer wrapper via export-ABI adapter: "
        f"num_inference_steps={args.num_inference_steps} dtype={adapter.dtype}",
        flush=True,
    )
    print(
        "note: MiniMaxH3Scheduler.step keeps sample/model_output dtype, "
        "but evaluates the Euler blend in float32 for fp16/bf16 samples",
        flush=True,
    )

    # Auto-offload is demand-driven; encode leaves text/vision on CUDA until something else
    # needs memory. Force-release immediately after prompt embeds, before VAE/denoise.
    _install_prompt_embed_release_hook(pipe, device=device)

    call_kwargs = {
        "prompt": args.prompt,
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "num_inference_steps": args.num_inference_steps,
        "generator": torch.Generator(device=device).manual_seed(args.seed),
        "video_decode_autocast_dtype": torch.float16,
        "output": ["videos", "audio", "sampling_rate"],
    }
    if args.image:
        call_kwargs["image"] = load_image(args.image)

    results = pipe(**call_kwargs)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    encode_video(
        results["videos"][0],
        fps=24,
        output_path=str(output),
        audio=results["audio"][0],
        audio_sample_rate=results["sampling_rate"],
    )
    print(f"saved: {output}", flush=True)


if __name__ == "__main__":
    main()
