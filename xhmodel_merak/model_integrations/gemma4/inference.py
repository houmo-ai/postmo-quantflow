from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from xhmodel_merak.model_catalog.contracts import ModelRecipe
from xhmodel_merak.model_inference.contracts import TextInferenceResult
from xhmodel_merak.model_inference.inputs import (
    load_json,
    require_local_file,
    scoped_environment,
    validate_inputs,
)
from xhmodel_merak.model_integrations.autoregressive.inference import run_text_generation


_FRAME_DIRECTORY_FPS = 2.0


class _Gemma4Adapter:
    def validate_artifact(self, *, recipe, root, manifest, meta_file):
        variant = recipe.get_variant(manifest.get("recipe", {}).get("variant", recipe.default_variant.name))
        if variant.speculative_mode == "mtp":
            from .artifact import validate_mtp
            validate_mtp(root, load_json(meta_file))

    def run(
        self,
        *,
        recipe: ModelRecipe,
        root: Path,
        manifest: dict[str, Any],
        meta_file: Path,
        inputs: dict[str, Any],
        device: str,
        components: tuple[str, ...],
        selected_golden: tuple[str, ...],
        golden_artifacts: dict[str, tuple[str, ...]],
    ) -> TextInferenceResult:
        normalized = validate_inputs(recipe, inputs)
        media = [name for name in ("image", "video", "audio") if normalized.get(name) is not None]
        variant = recipe.get_variant(manifest.get("recipe", {}).get("variant", recipe.default_variant.name))
        spec_mode = variant.speculative_mode
        if spec_mode == "mtp":
            if media or selected_golden:
                raise ValueError("Gemma4 unified MTP currently accepts text only, without Golden")
            from xhmodel_merak.xh_llm.models.gemma4_series.mtp_runtime import run_mtp_generation
            output_text, output_tokens, latency_s, tokens_per_second, stats = run_mtp_generation(
                meta_file, prompt=normalized["prompt"], max_new_tokens=int(normalized["max_new_tokens"]), device=device)
        elif not media:
            output_text, output_tokens, latency_s, tokens_per_second, stats = run_text_generation(
                meta_file,
                prompt=normalized["prompt"],
                max_new_tokens=int(normalized["max_new_tokens"]),
                device=device,
                device_map=True,
                pack_w4=True,
            )
        else:
            output_text, output_tokens, latency_s, tokens_per_second, stats = _run_multimodal(
                meta_file,
                recipe=recipe,
                prompt=normalized["prompt"],
                modality=media[0],
                media_path=require_local_file(
                    recipe,
                    media[0],
                    normalized[media[0]],
                    allow_directory=media[0] == "video",
                ),
                max_new_tokens=int(normalized["max_new_tokens"]),
                device=device,
            )
        return TextInferenceResult(
            model_id=recipe.model_id,
            task=recipe.task,
            output_type="text",
            artifact_dir=str(root),
            meta_file=str(meta_file),
            latency_s=latency_s,
            components=components,
            golden_components=selected_golden,
            golden_artifacts=golden_artifacts,
            output_text=output_text,
            output_tokens=output_tokens,
            tokens_per_second=tokens_per_second,
            stats=stats,
            spec_decode_mode=spec_mode,
        )


def _run_multimodal(
    meta_file: Path,
    *,
    recipe: ModelRecipe,
    prompt: str,
    modality: str,
    media_path: Path,
    max_new_tokens: int,
    device: str,
) -> tuple[str, int, float, float, dict[str, Any]]:
    loaded_video = _load_video(media_path) if modality == "video" else None

    import torch
    from PIL import Image

    resolved_device = torch.device(device)
    device_context = torch.cuda.device(resolved_device) if resolved_device.type == "cuda" else nullcontext()
    with device_context, scoped_environment({"ENABLE_HMINFERENCE_V2": "1", "XHQUANT_PACKED_W4": "1"}):
        from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager

        runtime_meta = load_json(meta_file)
        loader_kwargs = {"device_map": [resolved_device]} if resolved_device.type == "cuda" else {}
        model = AutoLLMHONNXModel.from_pretrained(str(meta_file), **loader_kwargs)
        processor = model.get_tf_processor()
        content: list[dict[str, Any]] = []
        if modality == "image":
            content.append({"type": "image", "image": Image.open(media_path).convert("RGB")})
        elif modality == "video":
            assert loaded_video is not None
            video, video_metadata = loaded_video
            video_item = {"type": "video", "video": video}
            if video_metadata is not None:
                video_item["video_metadata"] = video_metadata
            content.append(video_item)
        elif modality == "audio":
            if getattr(model, "audio", None) is None:
                raise ValueError(f"{recipe.model_id} Artifact has no audio runtime component")
            content.append({"type": "audio", "audio": str(media_path)})
        content.append({"type": "text", "text": prompt})
        model_inputs = processor.apply_chat_template([{"role": "user", "content": content}]).to(device)
        context_max_length = int(runtime_meta.get("model_config", {}).get("context_max_length", 0))
        prompt_length = int(model_inputs.input_ids.shape[1])
        if context_max_length and prompt_length + max_new_tokens > context_max_length:
            raise ValueError(
                f"prompt tokens ({prompt_length}) + max_new_tokens ({max_new_tokens}) exceed "
                f"context_max_length ({context_max_length})"
            )
        tokenizer = processor.tokenizer
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        model.to(device)
        _synchronize(device)
        started = time.perf_counter()
        with LLMInferenceContextManager(model, devices=[device]):
            runtime_inputs = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in model_inputs.items()
            }
            generated = model.generate(
                **runtime_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        _synchronize(device)
        latency_s = time.perf_counter() - started
        output_ids = generated[0, prompt_length:]
        output_tokens = int(output_ids.numel())
        return (
            tokenizer.decode(output_ids.tolist(), skip_special_tokens=True).strip(),
            output_tokens,
            latency_s,
            output_tokens / latency_s if latency_s > 0 else 0.0,
            {"prompt_tokens": prompt_length, "modality": modality},
        )


def _load_video(path: Path) -> tuple[Any, dict[str, Any] | None]:
    from PIL import Image

    if path.is_dir():
        suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
        frame_paths = [frame for frame in sorted(path.iterdir()) if frame.suffix.lower() in suffixes]
        if not frame_paths:
            raise ValueError(f"video frame directory contains no supported images: {path}")
        frames = []
        frame_size = None
        for frame_path in frame_paths:
            try:
                with Image.open(frame_path) as image:
                    frame = image.convert("RGB")
            except OSError as error:
                raise ValueError(f"failed to load video frame {frame_path}: {error}") from error
            if frame_size is None:
                frame_size = frame.size
            elif frame.size != frame_size:
                raise ValueError(
                    "video frame directory images must have identical dimensions: "
                    f"expected {frame_size}, got {frame.size} for {frame_path}"
                )
            frames.append(frame)
        width, height = frame_size
        num_frames = len(frames)
        return frames, {
            "fps": _FRAME_DIRECTORY_FPS,
            "width": width,
            "height": height,
            "duration": num_frames / _FRAME_DIRECTORY_FPS,
            "total_num_frames": num_frames,
            "frames_indices": list(range(num_frames)),
            "video_backend": "image_directory",
        }
    return str(path), None


def _synchronize(device: str) -> None:
    import torch

    resolved = torch.device(device)
    if resolved.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(resolved)


adapter = _Gemma4Adapter()


__all__ = ["adapter"]
