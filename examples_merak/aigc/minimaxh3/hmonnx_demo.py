from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


DEFAULT_LORA_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "models" / "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"
)
DEFAULT_LORA_ALPHA = 8
DEFAULT_LORA_SCALE = 1.0


def parse_exec_devices(raw: str | None, primary_device: torch.device) -> list[torch.device | str | int]:
    if raw is None or not str(raw).strip():
        return [primary_device]
    devices: list[torch.device | str | int] = []
    for token in str(raw).split(","):
        item = token.strip()
        if not item:
            continue
        if item.isdigit():
            devices.append(int(item))
        else:
            devices.append(torch.device(item))
    if not devices:
        raise ValueError("--exec-devices must contain at least one device")
    return devices


def resolve_hmonnx_demo_components(components: list[str] | None) -> list[str] | None:
    """Normalize demo ``--components``.

    Returns ``None`` to install every available exported component.
    Selecting ``transformer`` also enables ``text_encoder``/``vision_encoder`` so the
    host path keeps fixed-length text embeds and ``encoder_attention_mask``.
    """
    from xhmodel_merak.xh_other_model.models.minimax_h3.minimax_h3_converter import (
        COMPONENTS,
        expand_minimax_h3_components,
    )

    if components is None:
        return None
    if "all" in components:
        if components != ["all"]:
            raise ValueError("`all` cannot be combined with other --components values")
        return None

    selected = set(components)
    unknown = sorted(selected - set(COMPONENTS))
    if unknown:
        raise ValueError(f"Unsupported MiniMax-H3 HMONNX components: {unknown}; expected {COMPONENTS}")
    selected = expand_minimax_h3_components(selected)
    return [component for component in COMPONENTS if component in selected]


def resolve_demo_pretrained_component_names(pipe, *, use_light_text_host: bool) -> list[str]:
    """Choose ``load_components`` names, optionally skipping the full text_encoder."""
    preferred = (
        "image_processor",
        "tokenizer",
        "processor",
        "vae",
        "scheduler",
        "audio_scheduler",
        "transformer",
        "video_processor",
        "audio_vae",
    )
    names = [name for name in preferred if name in getattr(pipe, "_component_specs", {})]
    if not use_light_text_host and "text_encoder" in getattr(pipe, "_component_specs", {}):
        # Keep text_encoder near the front so tokenizer/processor pairing stays familiar.
        insert_at = 1 if names and names[0] == "image_processor" else 0
        names.insert(insert_at, "text_encoder")
    return names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MiniMax-H3 with exported HMONNX components.")
    parser.add_argument("--model-dir", required=True, help="MiniMax-H3 HF / diffusers model directory.")
    parser.add_argument("--export-dir", required=True, help="Merak export directory with export_meta_info.json.")
    parser.add_argument(
        "--components",
        nargs="+",
        choices=(
            "all",
            "vae_encoder",
            "vae_decoder",
            "audio_vae_encoder",
            "audio_vae_decoder",
            "transformer",
            "vision_encoder",
            "text_encoder",
        ),
        default=None,
        help=(
            "One or more HMONNX components to use, or `all` for every available export. "
            "Default is also all available exports. Selecting text_encoder or vision_encoder "
            "enables both; selecting transformer also enables text_encoder/vision_encoder "
            "so fixed-length text embeds and encoder_attention_mask are available."
        ),
    )
    parser.add_argument("--workflow", default="fl2va", choices=("t2va", "fl2va"))
    parser.add_argument(
        "--prompt",
        default="""
        The character in the picture is sitting inside an eggshell, with flames spewing from the bottom of the
        eggshell, causing him to fly up and he (S1) says, <d>[Chinese] 你好， 我的名字叫宇宙无敌大帅男许晨</d>.
    """,
    )
    parser.add_argument("--image", default="examples_merak/aigc/minimaxh3/astronaut.jpg")
    parser.add_argument("--output", default="examples_merak/aigc/minimaxh3/output.mp4")
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Primary host/runtime device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--exec-devices",
        default=None,
        help=(
            "Comma-separated HMONNX execution devices for multi-GPU auto-offload, "
            "e.g. cuda:0,cuda:1. Defaults to --device."
        ),
    )
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=608)
    parser.add_argument("--num-frames", type=int, default=124)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-path", type=Path, default=DEFAULT_LORA_PATH)
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Do not load LoRA even if --lora-path exists.",
    )
    parser.add_argument(
        "--require-lora",
        action="store_true",
        help="Fail if --lora-path does not exist (default: skip missing LoRA).",
    )
    parser.add_argument("--lora-alpha", type=int, default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--lora-scale", type=float, default=DEFAULT_LORA_SCALE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_lora and args.require_lora:
        raise ValueError("--no-lora and --require-lora are mutually exclusive")
    if args.workflow == "fl2va" and not args.image:
        raise ValueError("--image is required for fl2va")
    if args.workflow == "t2va" and args.image:
        raise ValueError("--image is only valid for fl2va")

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from diffusers import ComponentsManager
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ModularPipeline
    from diffusers.utils import load_image
    from diffusers.utils.export_utils import encode_video

    from xhmodel_merak.xh_other_model.models.minimax_h3.components_hmonnx import install_hmonnx_components
    from xhmodel_merak.xh_other_model.models.minimax_h3.lora import load_minimax_h3_lora
    from xhmodel_merak.xh_other_model.models.minimax_h3.text_embedding_host import (
        install_hmonnx_text_encoder_host,
        text_embedding_path,
        vision_geometry_path,
    )

    device = torch.device(args.device)
    manager = ComponentsManager()
    manager.enable_auto_cpu_offload(device=str(device))
    pipe = MiniMaxH3ModularPipeline.from_pretrained(
        args.model_dir,
        workflow=args.workflow,
        components_manager=manager,
    )
    selected_components = resolve_hmonnx_demo_components(args.components)
    if args.components and "transformer" in args.components and selected_components is not None:
        print(
            "transformer validation requires text_encoder/vision_encoder for fixed-length "
            f"text embeds + encoder_attention_mask; enabling: {', '.join(selected_components)}",
            flush=True,
        )

    export_dir = Path(args.export_dir).expanduser().resolve()
    use_light_text_host = text_embedding_path(export_dir).is_file()
    require_vision_geometry = args.workflow == "fl2va" or (
        selected_components is None or "vision_encoder" in selected_components
    )
    if use_light_text_host and require_vision_geometry and not vision_geometry_path(export_dir).is_file():
        raise FileNotFoundError(
            f"Export has text embeddings but is missing vision geometry: {vision_geometry_path(export_dir)}"
        )

    component_names = resolve_demo_pretrained_component_names(pipe, use_light_text_host=use_light_text_host)
    pipe.load_components(
        names=component_names,
        dtype={
            "default": torch.float16,
            "vae": torch.float16,
            "audio_vae": torch.float16,
            "text_encoder": torch.float16,
            "transformer": torch.float16,
        },
    )
    if use_light_text_host:
        install_hmonnx_text_encoder_host(
            pipe,
            args.model_dir,
            export_dir,
            dtype=torch.float16,
            require_vision_geometry=require_vision_geometry,
        )
        print(
            "Loaded lightweight text_encoder host from exported embeddings "
            f"({text_embedding_path(export_dir).name}"
            + (
                f", {vision_geometry_path(export_dir).name}"
                if vision_geometry_path(export_dir).is_file()
                else ""
            )
            + "); skipped full original text_encoder weights",
            flush=True,
        )
    if args.no_lora:
        print("LoRA disabled via --no-lora", flush=True)
    else:
        lora_path = Path(args.lora_path).expanduser()
        if not lora_path.is_absolute():
            lora_path = Path.cwd() / lora_path
        if lora_path.is_file():
            load_minimax_h3_lora(
                pipe.transformer,
                lora_path,
                lora_alpha=args.lora_alpha,
                lora_scale=args.lora_scale,
            )
            print(f"LoRA loaded: {lora_path}", flush=True)
        elif args.require_lora:
            raise FileNotFoundError(f"LoRA checkpoint does not exist: {lora_path}")
        else:
            print(f"LoRA not found, continuing without LoRA: {lora_path}", flush=True)
    exec_devices = parse_exec_devices(args.exec_devices, device)
    installed_components = install_hmonnx_components(
        pipe,
        args.export_dir,
        device,
        selected_components,
        exec_devices=exec_devices,
    )
    print(
        f"HMONNX components: {', '.join(installed_components)}; exec_devices={[str(item) for item in exec_devices]}",
        flush=True,
    )
    if "transformer" in installed_components:
        print(
            "transformer HMONNX ABI: host precomputes temb/adaln/rotary/pack_gather_indices/packed_attention_mask; "
            "encoder_attention_mask is required for token_refiner; FP transformer body is released at install",
            flush=True,
        )
    visual = getattr(getattr(getattr(pipe, "text_encoder", None), "model", None), "visual", None)
    max_patches = getattr(visual, "max_patches", None)
    if max_patches is not None:
        print(
            f"vision HMONNX capacity: max_patches={max_patches} (shorter images are padded and attention-masked)",
            flush=True,
        )

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
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
