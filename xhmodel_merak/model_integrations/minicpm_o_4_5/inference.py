from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from xhmodel_merak.errors import invalid_parameter
from xhmodel_merak.model_artifact.files import artifact_file
from xhmodel_merak.model_catalog.contracts import ModelRecipe
from xhmodel_merak.model_inference.contracts import AudioInferenceResult, TextInferenceResult
from xhmodel_merak.model_inference.inputs import (
    collect_golden_dirs,
    require_local_file,
    scoped_environment,
    snapshot_golden_dirs,
    validate_inputs,
)

from .artifact import load_metadata, validate_runtime_contract


def _load_contract(root: Path, manifest: Mapping[str, Any], meta_file: Path) -> dict[str, Any]:
    meta = load_metadata(meta_file)
    validate_runtime_contract(
        meta, resolve_file=lambda value, field: artifact_file(root, manifest, value, field),
        deployment_precision=manifest["parameters"]["precision"]["deployment"],
        variant=manifest["recipe"]["variant"],
    )
    return meta


def _run_session(runtime: Any, chunks: list[Any], media: dict[str, Path], inputs: dict[str, Any]) -> tuple:
    import librosa
    import numpy as np

    generate_audio = inputs["generate_audio"]
    runtime.reset_session()
    if generate_audio:
        runtime.init_token2wav_cache(
            librosa.load(media["reference_audio"], sr=16000, mono=True, res_type="kaiser_best")[0]
        )
    for index, (waveform, frames) in enumerate(chunks):
        content = ([inputs["prompt"]] if index == 0 else []) + frames
        if waveform is not None:
            content.append(waveform)
        runtime.streaming_prefill(
            "unified", [{"role": "user", "content": content}], tokenizer=runtime.tokenizer,
            is_last_chunk=index == len(chunks) - 1, max_slice_nums=1,
            use_tts_template=generate_audio,
        )
    texts, waves, segments = [], [], []
    for index, (value, text) in enumerate(runtime.streaming_generate(
        "unified", tokenizer=runtime.tokenizer, generate_audio=generate_audio,
        use_tts_template=generate_audio, do_sample=False, max_new_tokens=inputs["max_new_tokens"],
    )):
        if generate_audio:
            wave = value.detach().float().cpu().numpy().reshape(-1)
            waves.append(wave)
            texts.append(str(text))
            segments.append({"chunk_index": index, "text": str(text), "num_samples": int(wave.size)})
        else:
            texts.append(value)
            segments.append({"chunk_index": index, "text": value, "finished": bool(text), "num_samples": 0})
    text = "".join(texts)
    wave = np.concatenate(waves) if waves else None
    return text, wave, segments, {}


def _run_duplex(runtime: Any, chunks: list[Any], media: dict[str, Path], inputs: dict[str, Any]) -> tuple:
    import numpy as np

    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import bind_duplex_hmonnx_decoder

    generate_audio = inputs["generate_audio"]
    prompt_path = str(media["reference_audio"]) if generate_audio else None
    duplex = runtime.as_duplex(
        device=str(runtime.llm.device), generate_audio=generate_audio, omni="image" in media or "video" in media,
    )
    duplex = bind_duplex_hmonnx_decoder(duplex)
    duplex.prepare(prefix_system_prompt=inputs["prompt"], prompt_wav_path=prompt_path)
    texts, waves, segments = [], [], []
    for index, (waveform, frames) in enumerate(chunks):
        prefill = duplex.streaming_prefill(audio_waveform=waveform, frame_list=frames or None, max_slice_nums=1)
        if not prefill["success"]:
            raise RuntimeError(f"MiniCPM-o-4.5 duplex prefill failed: {prefill['reason']}")
        result = duplex.streaming_generate(
            prompt_wav_path=prompt_path, max_new_speak_tokens_per_chunk=inputs["max_new_tokens"],
            decode_mode="greedy", top_k=1, top_p=1.0, temperature=0.0,
        )
        text = result["text"]
        texts.append(text)
        wave = result["audio_waveform"]
        if generate_audio and wave is not None:
            wave = np.asarray(wave, dtype=np.float32).reshape(-1)
            waves.append(wave)
        segments.append({
            "chunk_index": index, "text": text, "is_listen": result["is_listen"],
            "end_of_turn": result["end_of_turn"], "n_tokens": result["n_tokens"],
            "n_tts_tokens": result["n_tts_tokens"], "num_samples": 0 if wave is None else int(wave.size),
        })
    return "".join(texts), np.concatenate(waves) if waves else None, segments, {
        "output_tokens": sum(segment["n_tokens"] for segment in segments),
        "listen_turns": sum(segment["is_listen"] for segment in segments),
        "speak_turns": sum(not segment["is_listen"] for segment in segments),
    }


class _MiniCPMO45Adapter:
    def validate_artifact(
        self, *, root: Path, manifest: Mapping[str, Any], meta_file: Path, **_kwargs: Any,
    ) -> None:
        _load_contract(root, manifest, meta_file)

    def run(
        self, *, recipe: ModelRecipe, root: Path, manifest: dict[str, Any], meta_file: Path,
        inputs: dict[str, Any], device: str, components: tuple[str, ...], selected_golden: tuple[str, ...],
        golden_artifacts: dict[str, tuple[str, ...]],
    ) -> TextInferenceResult | AudioInferenceResult:
        import librosa
        import numpy as np
        from PIL import Image

        from xhmodel_merak.xh_llm.models.minicpm_o_4_5.golden import enable_runtime_golden
        from xhmodel_merak.xh_llm.models.minicpm_o_4_5.media import (
            normalize_minicpmo_video,
            streaming_media_chunks,
        )
        from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime import MiniCPMO45HMONNXRuntime

        normalized = validate_inputs(recipe, inputs)
        mode = manifest["recipe"]["variant"]
        generate_audio = normalized["generate_audio"]
        if generate_audio and "reference_audio" not in normalized:
            raise invalid_parameter(recipe.model_id, "inputs.reference_audio", None, "required with generate_audio")
        media = {name: require_local_file(recipe, name, normalized[name]) for name in (
            "image", "audio", "video", "reference_audio",
        ) if name in normalized}
        active_media = set(media) - {"reference_audio"}
        if mode == "offline" and len(active_media) > 1:
            raise invalid_parameter(recipe.model_id, "inputs", sorted(active_media), "offline accepts one media input")
        if mode == "session" and (not active_media or not active_media <= {"audio", "video"}):
            raise invalid_parameter(recipe.model_id, "inputs", sorted(active_media), "session requires audio or video")
        if mode != "offline" and "language" in inputs:
            raise invalid_parameter(
                recipe.model_id, "inputs.language", inputs["language"],
                "offline-only; set streaming language in prompt",
            )
        if mode == "stream" and (not active_media or ("video" in active_media and len(active_media) > 1)):
            raise invalid_parameter(
                recipe.model_id, "inputs", sorted(active_media),
                "stream requires audio, image, audio + image, or video",
            )
        if mode == "session" and len(active_media) > 1:
            raise invalid_parameter(recipe.model_id, "inputs", sorted(active_media), "session accepts one media input")
        contents: list[Any] = []
        if mode == "offline" and "image" in media:
            with Image.open(media["image"]) as image:
                contents.append(image.convert("RGB"))
        elif mode == "offline" and "audio" in media:
            contents.append(librosa.load(media["audio"], sr=16000, mono=True, res_type="kaiser_best")[0])
        elif mode == "offline" and "video" in media:
            contents.extend(normalize_minicpmo_video(media["video"], include_audio=True, stack_frames=1))
        contents.append(normalized["prompt"])
        messages = [{"role": "user", "content": contents}]
        chunks = streaming_media_chunks(media) if mode != "offline" else []
        meta = _load_contract(root, manifest, meta_file)
        golden_before = snapshot_golden_dirs(root, selected_golden) if selected_golden else {}
        with scoped_environment({"ENABLE_HMINFERENCE_V2": "1"}):
            runtime = MiniCPMO45HMONNXRuntime.from_artifact(root, metadata=meta, device=device)
            started = time.perf_counter()
            try:
                enable_runtime_golden(runtime, root, meta, selected_golden)
                segments: list[dict[str, Any]] = []
                stats: dict[str, Any] = {}
                if mode == "session":
                    text, waveform, segments, stats = _run_session(runtime, chunks, media, normalized)
                elif mode == "stream":
                    text, waveform, segments, stats = _run_duplex(runtime, chunks, media, normalized)
                elif generate_audio:
                    reference = librosa.load(media["reference_audio"], sr=16000, mono=True, res_type="kaiser_best")[0]
                    messages.insert(0, runtime.host_model.get_sys_prompt(
                        ref_audio=reference, mode="omni", language=normalized.get("language", "zh"),
                    ))
                if mode == "offline":
                    # The official chat API only runs TTS when an output path is supplied.
                    with TemporaryDirectory(prefix="minicpmo45_inference_") as temporary:
                        text = runtime.chat(
                            msgs=messages, tokenizer=runtime.tokenizer, omni_mode=True,
                            sampling=False, do_sample=False, num_beams=1, max_new_tokens=normalized["max_new_tokens"],
                            generate_audio=generate_audio, use_tts_template=generate_audio, max_slice_nums=1,
                            output_audio_path=str(Path(temporary) / "output.wav") if generate_audio else None,
                        )
                    waveform = runtime.last_waveform
                    if waveform is not None:
                        waveform = waveform.detach().float().cpu().numpy().reshape(-1)
                output_tokens = stats.get(
                    "output_tokens", len(runtime.tokenizer.encode(text, add_special_tokens=False)),
                )
                stats.update(
                    mode=mode, output_text=text, output_tokens=output_tokens,
                    observed_token2wav_counts=(runtime.token2wav_execution_counts if mode == "offline" else
                                              runtime.token2wav_streaming_execution_counts),
                )
                if segments:
                    stats["turns"] = segments
                if mode != "offline":
                    stats["input_chunks"] = len(chunks)
                latency = time.perf_counter() - started
            finally:
                runtime.release()
        produced_golden = (
            collect_golden_dirs(root, selected_golden, before=golden_before) if selected_golden else golden_artifacts
        )
        common = dict(
            model_id=recipe.model_id, task=recipe.task, artifact_dir=str(root), meta_file=str(meta_file),
            latency_s=latency, components=components, golden_components=selected_golden,
            golden_artifacts=produced_golden,
        )
        if generate_audio:
            if waveform is None or waveform.size == 0 or not np.isfinite(waveform).all():
                raise RuntimeError("MiniCPM-o-4.5 produced no finite audio")
            sample_rate = 24000
            duration = waveform.size / sample_rate
            return AudioInferenceResult(
                **common, output_type="audio", waveform=waveform, sample_rate=sample_rate,
                num_samples=int(waveform.size), duration_s=duration, real_time_factor=latency / duration,
                voice="reference_audio", speed=1.0, segments=tuple(segments), stats=stats,
            )
        return TextInferenceResult(
            **common, output_type="text", output_text=text, output_tokens=output_tokens,
            tokens_per_second=output_tokens / latency,
            stats=stats,
        )


adapter = _MiniCPMO45Adapter()
