from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, DynamicCache

from .hf_compatible import (
    ensure_tts_sampling_config,
    patch_audio_attention_return_compat,
    patch_dynamic_cache_seen_tokens,
)
from .media import normalize_minicpmo_video
from .runtime import MiniCPMO45HMONNXRuntime
from .streaming_fixtures import (
    get_streaming_case,
    run_streaming_case,
    streaming_case_result_manifest,
)


_SAMPLE_RATE = 16000
_STREAM_ROLES = (
    "vision",
    "audio_stream_prefill",
    "audio_stream_decode",
    "llm_prefill",
    "llm_decode",
    "tts_prefill",
    "tts_decode",
    "token2wav_speech_tokenizer",
    "token2wav_campplus",
    "token2wav_prompt_frontend",
    "token2wav_prompt_estimator_step",
    "token2wav_flow_frontend",
    "token2wav_estimator_step",
    "token2wav_hift_stream_part1",
    "token2wav_hift_stream_part2",
)


def _session_media(video_path: Path, prompt_path: Path) -> dict[str, Any]:
    import librosa
    from moviepy import VideoFileClip

    with VideoFileClip(str(video_path)) as video:
        audio = np.asarray(video.audio.to_soundarray(fps=_SAMPLE_RATE), dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    chunks = []
    for index in range(2):
        chunk = audio[index * _SAMPLE_RATE : (index + 1) * _SAMPLE_RATE]
        chunks.append((np.pad(chunk, (0, _SAMPLE_RATE - len(chunk))), []))
    prompt_waveform, _ = librosa.load(prompt_path, sr=_SAMPLE_RATE, mono=True)
    return {
        "chunks": chunks,
        "prompt_waveform": np.asarray(prompt_waveform, dtype=np.float32),
        "prompt_wav_path": str(prompt_path),
    }


def default_real_request(meta: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    model_dir = Path(meta["hf_model"])
    mode = meta["export_mode"]
    request: dict[str, Any] = {"mode": "real", "first_step_only": bool(overrides.get("first_step_only", True))}
    if mode == "offline":
        video = Path(overrides.get("video_sample", model_dir / "assets" / "Skiing.mp4"))
        contents = normalize_minicpmo_video(video, include_audio=True, stack_frames=1)
        request.update(
            {
                "sample": str(video),
                "messages": [
                    {
                        "role": "user",
                        "content": [*contents, overrides.get("question", "请简短描述视频内容。")],
                    }
                ],
                "ref_audio": str(overrides.get("ref_audio", model_dir / "assets" / "system_ref_audio.wav")),
                "chat_kwargs": {
                    "omni_mode": True,
                    "max_new_tokens": int(overrides.get("max_new_tokens", 32)),
                    "generate_audio": True,
                    "use_tts_template": True,
                    "max_slice_nums": 1,
                },
                "output_audio_name": "golden_sample_output.wav",
            }
        )
    elif mode == "session":
        video = Path(overrides.get("video_sample", model_dir / "assets" / "Skiing.mp4"))
        prompt_path = Path(overrides.get("ref_audio", model_dir / "assets" / "system_ref_audio.wav"))
        request.update(
            {
                "sample": str(video),
                "streaming_case": "session_audio_reply",
                "media": _session_media(video, prompt_path),
            }
        )
    elif mode == "stream":
        sample = Path(overrides.get("audio_sample", model_dir / "assets" / "haimianbaobao.wav"))
        if not sample.is_file():
            raise FileNotFoundError(f"Golden audio sample not found: {sample}; pass --golden-audio")
        request.update(
            {
                "duplex_audio_sample": str(sample),
                "duplex_image_sample": str(overrides.get("image_sample", model_dir / "assets" / "highway.png")),
                "seed": int(overrides.get("seed", 20260918)),
            }
        )
    else:
        raise ValueError(f"unsupported MiniCPM-o-4.5 export mode: {mode}")
    return request


def build_offline_messages(host: Any, request: Mapping[str, Any]) -> list[dict[str, Any]]:
    import librosa

    ref_audio, _ = librosa.load(request["ref_audio"], sr=_SAMPLE_RATE, mono=True)
    system_prompt = host.get_sys_prompt(ref_audio=ref_audio, mode="omni", language="zh")
    return [system_prompt, *request["messages"]]


def run_duplex_golden_sample(
    runtime: Any,
    sample_path: Path,
    image_path: Path,
    model_dir: Path,
    directories: Mapping[str, Path],
    seed: int = 20260918,
) -> dict[str, Any]:
    import librosa
    from PIL import Image

    waveform, _ = librosa.load(sample_path, sr=_SAMPLE_RATE, mono=True)
    waveform = np.asarray(waveform, dtype=np.float32)
    with Image.open(image_path) as image:
        frame = image.convert("RGB")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    runtime.reset_state()
    prompt_path = model_dir / "assets" / "system_ref_audio.wav"
    duplex = runtime.as_duplex(generate_audio=True)
    duplex.prepare(
        prefix_system_prompt="You enjoy having a good conversation.",
        prompt_wav_path=str(prompt_path),
    )

    offset = 0
    length = 16560
    index = 0
    audible_rms = 0.0
    while offset < len(waveform):
        chunk = waveform[offset : offset + length]
        offset += len(chunk)
        if len(chunk) < length:
            chunk = np.pad(chunk, (0, length - len(chunk)))
        duplex.streaming_prefill(audio_waveform=chunk, frame_list=[frame] if index == 0 else None)
        result = duplex.streaming_generate(
            prompt_wav_path=str(prompt_path),
            max_new_speak_tokens_per_chunk=20,
            decode_mode="greedy",
            top_k=1,
            top_p=1.0,
            temperature=0.0,
        )
        if not result.get("is_listen", False) and result.get("audio_waveform") is not None:
            audio = np.asarray(result["audio_waveform"], dtype=np.float32)
            audible_rms = max(audible_rms, float(np.sqrt(np.mean(np.square(audio)))))
        captured = [name for name in _STREAM_ROLES if (directories[name] / "step_0").is_dir()]
        if len(captured) == len(_STREAM_ROLES) and audible_rms > 1e-5:
            return {
                "sample": str(sample_path),
                "image_sample": str(image_path),
                "seed": seed,
                "capture_stopped_after_chunk_index": index,
                "observed_audible_turn_rms": audible_rms,
            }
        length = 16000
        index += 1
    raise RuntimeError("default duplex sample did not capture all streaming graphs with audible output")


def _load_real_runtime(root: Path, meta: Mapping[str, Any]) -> tuple[MiniCPMO45HMONNXRuntime, Any]:
    model_dir = str(meta["hf_model"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    host = AutoModel.from_pretrained(
        model_dir,
        config=ensure_tts_sampling_config(config),
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",
    ).eval()
    patch_dynamic_cache_seen_tokens(DynamicCache)
    patch_audio_attention_return_compat(host)
    return MiniCPMO45HMONNXRuntime(root, host), tokenizer


def _runtime_sessions(runtime: MiniCPMO45HMONNXRuntime) -> dict[str, Any]:
    sessions: dict[str, Any] = {}
    session_paths = {
        "vision": ("vision",),
        "audio": ("audio", "session"),
        "llm_prefill": ("llm", "prefill_model"),
        "llm_decode": ("llm", "decode_model"),
        "tts_prefill": ("tts", "prefill_model"),
        "tts_decode": ("tts", "decode_model"),
    }
    for name in (
        "stream_prefill_session",
        "stream_decode_session",
        "session_prefill_session",
        "session_decode_session",
    ):
        session_paths[f"audio_{name.removesuffix('_session')}"] = ("audio", name)
    for name in (
        "frontend_session",
        "decoder_session",
        "hift_part1_session",
        "hift_part2_session",
        "flow_frontend_session",
        "prompt_frontend_session",
        "estimator_step_session",
        "prompt_estimator_step_session",
        "hift_stream_part1_session",
        "hift_stream_part2_session",
        "campplus_session",
        "speech_tokenizer_session",
    ):
        session_paths[f"token2wav_{name.removesuffix('_session')}"] = ("token2wav", name)
    for name, path in session_paths.items():
        owner = runtime
        for attribute in path:
            owner = getattr(owner, attribute, None)
            if owner is None:
                break
        if owner is not None:
            sessions[name] = owner
    return sessions


def _enable_runtime_golden(
    runtime: MiniCPMO45HMONNXRuntime, root: Path, *, first_step_only: bool = False
) -> dict[str, Path]:
    golden_root = root / "real_golden"
    if golden_root.exists():
        shutil.rmtree(golden_root)
    directories: dict[str, Path] = {}
    for name, session in _runtime_sessions(runtime).items():
        if session is None:
            continue
        directory = golden_root / name
        if hasattr(session, "hmonnx_session"):
            # Keep the wrapper's state in sync: its setter also resets the
            # backend step counter.  Golden files themselves belong to the
            # concrete backend session below.
            session.enable_golden = True
        backend_session = getattr(session, "hmonnx_session", session)
        if not hasattr(backend_session, "save_golden"):
            raise RuntimeError(f"{name} backend does not support golden export")
        # Real Golden lives outside the graph directory, so copy its graph
        # before xhquant creates relative links for each step.
        backend_session.legacy_mode = True
        backend_session.save_golden = True
        backend_session.golden_dir = str(directory)
        reset_step = getattr(backend_session, "reset_step", None)
        if callable(reset_step):
            reset_step()
        else:
            backend_session.step = 0
        if first_step_only:
            forward = backend_session.forward

            def capture_first_step(*args: Any, _forward=forward, _backend=backend_session) -> Any:
                try:
                    return _forward(*args)
                finally:
                    _backend.save_golden = False

            backend_session.forward = capture_first_step
        directories[name] = directory
    return directories


def dump_real_golden(
    work_dir: Path,
    meta: Mapping[str, Any],
    device: str,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    if "duplex_audio_sample" in request:
        sample = Path(request["duplex_audio_sample"])
        if not sample.is_file():
            raise FileNotFoundError(sample)
        runtime, _ = _load_real_runtime(work_dir, meta)
        runtime.set_exec_device(device)
        directories = _enable_runtime_golden(
            runtime, work_dir, first_step_only=bool(request.get("first_step_only", False))
        )
        try:
            result = run_duplex_golden_sample(
                runtime,
                sample,
                Path(request["duplex_image_sample"]),
                Path(meta["hf_model"]),
                directories,
                seed=int(request.get("seed", 20260918)),
            )
        finally:
            runtime.release()
        components = {
            name: {
                "directory": str(directory.relative_to(work_dir)),
                "files": [str(path.relative_to(work_dir)) for path in sorted(directory.rglob("*.npy"))],
            }
            for name, directory in directories.items()
            if directory.is_dir() and any(directory.rglob("*.npy"))
        }
        return {"mode": "real", **result, "components": components}

    streaming_case_name = request.get("streaming_case")
    if isinstance(streaming_case_name, str):
        case = get_streaming_case(streaming_case_name)
        runtime, tokenizer = _load_real_runtime(work_dir, meta)
        runtime.set_exec_device(device)
        directories = _enable_runtime_golden(
            runtime, work_dir, first_step_only=bool(request.get("first_step_only", False))
        )
        runtime.reset_state()
        processor = getattr(getattr(runtime, "host_model", None), "processor", None)
        media = request.get("media")
        try:
            result = run_streaming_case(runtime, tokenizer, processor, case, media)
        finally:
            runtime.release()
        components = {
            name: {
                "directory": str(directory.relative_to(work_dir)),
                "files": [str(path.relative_to(work_dir)) for path in sorted(directory.rglob("*.npy"))],
            }
            for name, directory in directories.items()
            if directory.is_dir() and any(directory.rglob("*.npy"))
        }
        return {
            "mode": "real",
            "sample": request.get("sample"),
            "streaming_case": result.case_name,
            "final": result.final,
            "api_events": list(result.api_events),
            "text_chunks": list(result.text_chunks),
            "token_ids": list(result.token_ids),
            "waveform_chunks": list(result.waveform_chunks),
            "backend_counters": dict(result.backend_counters),
            "components": components,
            "streaming_manifest": streaming_case_result_manifest(meta, case),
        }
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("real MiniCPM-o-4.5 Golden requires non-empty messages")
    runtime, tokenizer = _load_real_runtime(work_dir, meta)
    runtime.set_exec_device(device)
    directories = _enable_runtime_golden(
        runtime, work_dir, first_step_only=bool(request.get("first_step_only", False))
    )
    runtime.reset_state()
    chat_kwargs = dict(request.get("chat_kwargs", {}))
    if "ref_audio" in request:
        messages = build_offline_messages(runtime.host_model, request)
    audio_output = work_dir / Path(str(request["output_audio_name"])).name if "output_audio_name" in request else None
    if audio_output is not None:
        chat_kwargs["output_audio_path"] = str(audio_output)
    chat_kwargs.update(
        {
            "msgs": messages,
            "tokenizer": tokenizer,
            "sampling": False,
            "do_sample": False,
        }
    )
    try:
        text = runtime.chat(**chat_kwargs)
    finally:
        runtime.release()
        if audio_output is not None:
            audio_output.unlink(missing_ok=True)
    components = {
        name: {
            "directory": str(directory.relative_to(work_dir)),
            "files": [str(path.relative_to(work_dir)) for path in sorted(directory.rglob("*.npy"))],
        }
        for name, directory in directories.items()
        if directory.is_dir() and any(directory.rglob("*.npy"))
    }
    return {
        "mode": "real",
        "sample": request.get("sample"),
        "text": text if isinstance(text, str) else str(text),
        "components": components,
    }


__all__ = ["default_real_request", "dump_real_golden"]
