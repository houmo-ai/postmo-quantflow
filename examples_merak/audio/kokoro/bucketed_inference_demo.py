from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path
from typing import Any

import numpy as np

from xhmodel_merak.xh_other_model.models.kokoro.assets import sha256
from xhmodel_merak.xh_other_model.models.kokoro.bucketed_runtime import KokoroBucketedRuntime
from xhmodel_merak.xh_other_model.models.kokoro.buckets import audio_seconds_to_frames
from xhmodel_merak.xh_other_model.models.kokoro.host import (
    DEFAULT_INPUT_IDS,
    DEFAULT_PHONEMES,
    DEFAULT_TEXT,
    SAMPLE_RATE,
    load_voice_style,
)
from xhmodel_merak.xh_other_model.models.kokoro.text import prepare_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the four-route Kokoro static pipeline")
    parser.add_argument("--text", help="Chinese text to synthesize (CPU ZHG2P v1.1)")
    parser.add_argument("--config-path", help="Kokoro v1.1-zh config.json containing the phoneme vocabulary")
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output-wav", default="work_dirs/kokoro_bucketed.wav")
    parser.add_argument("--backend", choices=("ort", "hmonnx"), default="hmonnx")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--token-bucket", type=int, help="Force one paired route by T capacity")
    parser.add_argument("--audio-seconds", type=int, help="Force one paired route by audio capacity")
    parser.add_argument("--list-buckets", "--list-routes", action="store_true")
    parser.add_argument("--exercise-all-buckets", "--exercise-all-routes", action="store_true")
    parser.add_argument("--route-report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = KokoroBucketedRuntime.from_export(
        args.export_dir,
        backend=args.backend,
        device=args.device,
    )
    routes = runtime.available_routes()
    if args.list_buckets:
        print(json.dumps({"routes": routes}, indent=2, ensure_ascii=False))
        if not args.exercise_all_buckets:
            return

    if args.text is None:
        text, phonemes = DEFAULT_TEXT, DEFAULT_PHONEMES
        tokens = np.asarray(DEFAULT_INPUT_IDS, dtype=np.int32)
    else:
        text = args.text.strip()
        config_path = _resolve_config_path(Path(args.export_dir).resolve(), runtime.meta, config_path=args.config_path)
        tokens, phonemes = prepare_text(
            text, config_path, max_tokens=max(int(route["token_max_length"]) for route in routes)
        )
    text_info = {"text": text, "phonemes": phonemes}

    voice_path = _resolve_voice_path(
        Path(args.export_dir).expanduser().resolve(),
        runtime.meta,
    )
    style = load_voice_style(voice_path, phoneme_count=tokens.size - 2).numpy()

    if args.exercise_all_buckets:
        requests = _exercise_requests(
            routes,
            token_bucket=args.token_bucket,
            audio_seconds=args.audio_seconds,
        )
        results: list[dict[str, Any]] = []
        for request in requests:
            try:
                waveform, synthesis = runtime.synthesize(
                    tokens,
                    style,
                    speed=args.speed,
                    token_bucket=request.get("token_bucket"),
                    frame_bucket=request.get("frame_bucket"),
                )
                results.append(
                    {
                        "exercise": request["exercise"],
                        "status": "ok",
                        "samples": int(waveform.size),
                        "rms": float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))),
                        **text_info,
                        **synthesis,
                    }
                )
            except Exception as error:
                results.append(
                    {
                        "exercise": request["exercise"],
                        "status": "failed",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        **text_info,
                    }
                )
            finally:
                runtime.clear_runner_cache()
        report = json.dumps(results, indent=2, ensure_ascii=False)
        print(report)
        if args.route_report:
            report_path = Path(args.route_report).expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report + "\n", encoding="utf-8")
        failed = [result["exercise"] for result in results if result["status"] != "ok"]
        if failed:
            raise RuntimeError(f"Kokoro bucket inference failed for: {failed}")
        return

    forced_frame = None if args.audio_seconds is None else audio_seconds_to_frames(args.audio_seconds)
    waveform, synthesis = runtime.synthesize(
        tokens,
        style,
        speed=args.speed,
        token_bucket=args.token_bucket,
        frame_bucket=forced_frame,
    )
    destination = Path(args.output_wav).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(waveform, -1, 1) * 32767).astype("<i2")
    with wave.open(str(destination), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(pcm.tobytes())
    print(
        json.dumps(
            {
                "output_wav": str(destination),
                "seconds": waveform.size / SAMPLE_RATE,
                **text_info,
                **synthesis,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _resolve_config_path(export_dir: Path, metadata: dict[str, Any], *, config_path: str | None) -> Path:
    path = Path(config_path).expanduser() if config_path else export_dir / metadata["runtime_assets"]["config_file"]
    if not path.is_file():
        raise FileNotFoundError(f"missing vocabulary: {path}; provide --config-path")
    return path.resolve()


def _exercise_requests(
    routes: tuple[dict[str, Any], ...],
    *,
    token_bucket: int | None,
    audio_seconds: int | None,
) -> tuple[dict[str, Any], ...]:
    frame_bucket = None if audio_seconds is None else audio_seconds_to_frames(audio_seconds)
    if token_bucket is not None or frame_bucket is not None:
        matches = tuple(
            route
            for route in routes
            if (token_bucket is None or int(route["token_max_length"]) == token_bucket)
            and (frame_bucket is None or int(route["frame_max_length"]) == frame_bucket)
        )
        if len(matches) != 1:
            raise ValueError(
                f"no unique paired route for token bucket T={token_bucket} and frame bucket F={frame_bucket}"
            )
        route = matches[0]
        return (
            {
                "exercise": str(route["key"]),
                "token_bucket": int(route["token_max_length"]),
                "frame_bucket": int(route["frame_max_length"]),
            },
        )

    return tuple(
        {
            "exercise": str(route["key"]),
            "token_bucket": int(route["token_max_length"]),
            "frame_bucket": int(route["frame_max_length"]),
        }
        for route in routes
    )


def _resolve_voice_path(export_dir: Path, metadata: dict[str, Any]) -> Path:
    voice = metadata["runtime_assets"]["voice_pack"]
    path = export_dir / voice["file"]
    if not path.is_file():
        raise FileNotFoundError(f"exported NumPy voice pack is missing: {path}")
    if sha256(path) != voice["sha256"]:
        raise RuntimeError(f"exported NumPy voice pack SHA256 mismatch: {path}")
    return path


if __name__ == "__main__":
    main()
