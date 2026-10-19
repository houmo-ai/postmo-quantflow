from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_CONFIG_PATH = "configs_merak/workflows/xh2a/other_models/minimax_h3/minimax_h3.yaml"
DEFAULT_OUTPUT_DIR = "work_dirs/minimax_h3_w8a8ssfp"
DEFAULT_LORA_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "models" / "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"
)
DEFAULT_LORA_ALPHA = 8
DEFAULT_LORA_SCALE = 1.0


def _remove_output_dir_if_needed(output_dir: str, force: bool) -> None:
    path = Path(output_dir)
    if force and path.exists():
        shutil.rmtree(path)


def resolve_workflow_components(components: list[str] | None) -> list[str] | None:
    """Normalize workflow ``--components``; ``all`` expands to the full supported set."""
    from xhmodel_merak.xh_other_model.models.minimax_h3.minimax_h3_converter import (
        COMPONENTS,
        expand_minimax_h3_components,
    )

    if components is None:
        return None
    if "all" in components:
        if components != ["all"]:
            raise ValueError("`all` cannot be combined with other --components values")
        return list(COMPONENTS)

    selected = set(components)
    unknown = sorted(selected - set(COMPONENTS))
    if unknown:
        raise ValueError(f"Unsupported MiniMax-H3 export components: {unknown}; expected {COMPONENTS}")
    # Export path: keep transformer-only selections intact. text/vision still bind to each other.
    selected = expand_minimax_h3_components(selected, export=True)
    return [component for component in COMPONENTS if component in selected]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MiniMax-H3 Merak workflow.")
    parser.add_argument("--model-dir", required=True, help="MiniMax-H3 HF / diffusers model directory.")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
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
            "One or more MiniMax-H3 export components, or `all` for the full supported set. "
            "Selecting text_encoder/vision_encoder enables both. Selecting transformer alone "
            "exports only transformer; HMONNX install/demo still expands text/vision at runtime."
        ),
    )
    parser.add_argument(
        "--resolution",
        choices=("demo", "480p", "720p", "720p_16x9"),
        default=None,
        help=(
            "Override transformer packing shapes for a named resolution preset. "
            "demo=608x352, 480p=832x480, 720p=1344x768, 720p_16x9=1280x704. "
            "VAE tile examples stay unchanged."
        ),
    )
    parser.add_argument("--input-sequence-length", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dump-golden", action="store_true")
    parser.add_argument("--debug", action="store_true")
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
        "--num-inference-steps",
        type=int,
        default=None,
        help=(
            "Override AdaLN LUT bake steps for transformer export "
            "(export.minimax_h3.num_inference_steps). Use 8 for the official 8step LoRA."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_lora and args.require_lora:
        raise ValueError("--no-lora and --require-lora are mutually exclusive")

    from xhmodel_merak.workflows import AutoWorkflow
    from xhmodel_merak.xh_other_model.models.minimax_h3.resolution import (
        build_transformer_input_shapes,
        get_resolution_preset,
    )

    _remove_output_dir_if_needed(args.output_dir, args.overwrite)
    workflow = AutoWorkflow.from_config(model_dir=args.model_dir, config_path=args.config_path, debug=args.debug)
    overrides: dict[str, object] = {}
    selected_components = resolve_workflow_components(args.components)
    if selected_components is not None:
        overrides["export.minimax_h3.components"] = selected_components
    if args.input_sequence_length is not None:
        overrides["export.minimax_h3.input_sequence_length"] = args.input_sequence_length
    if args.resolution is not None:
        preset = get_resolution_preset(args.resolution)
        # Only override transformer packing; keep VAE/text/audio tile examples intact.
        overrides["export.minimax_h3.input_shapes.transformer"] = build_transformer_input_shapes(preset)
        print(
            f"resolution preset={preset.name}: {preset.width}x{preset.height} "
            f"Nv={preset.num_video_tokens} Na={preset.num_audio_tokens} "
            f"S={preset.packed_sequence_length}",
            flush=True,
        )

    if args.no_lora:
        overrides["export.minimax_h3.lora_path"] = None
        print("LoRA disabled via --no-lora", flush=True)
    else:
        lora_path = Path(args.lora_path).expanduser()
        if not lora_path.is_absolute():
            lora_path = Path.cwd() / lora_path
        if lora_path.is_file():
            overrides["export.minimax_h3.lora_path"] = str(lora_path.resolve())
        elif args.require_lora:
            raise FileNotFoundError(f"LoRA checkpoint does not exist: {lora_path}")
        else:
            overrides["export.minimax_h3.lora_path"] = None
            print(f"LoRA not found, exporting without LoRA: {lora_path}", flush=True)
    overrides["export.minimax_h3.lora_alpha"] = args.lora_alpha
    overrides["export.minimax_h3.lora_scale"] = args.lora_scale
    if args.num_inference_steps is not None:
        if int(args.num_inference_steps) < 2:
            raise ValueError(f"--num-inference-steps must be >= 2, got {args.num_inference_steps}")
        overrides["export.minimax_h3.num_inference_steps"] = int(args.num_inference_steps)
        print(f"num_inference_steps override={int(args.num_inference_steps)}", flush=True)
    quant_result = workflow.quant(output_dir=args.output_dir, device=args.device, config_overrides=overrides)
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=args.output_dir,
        device=args.device,
        config_overrides=overrides,
    )
    print(f"export_result: {export_result}")
    if args.dump_golden:
        print(workflow.dump_golden(export_result=export_result, device=args.device))


if __name__ == "__main__":
    main()
