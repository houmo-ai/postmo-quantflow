"""Real-text regression for the released HMONNX pipeline (not part of export)."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

from xhmodel_merak.xh_other_model.models.kokoro.assets import sha256
from xhmodel_merak.xh_other_model.models.kokoro.bucketed_runtime import KokoroBucketedRuntime
from xhmodel_merak.xh_other_model.models.kokoro.host import SAMPLE_RATE, load_voice_style
from xhmodel_merak.xh_other_model.models.kokoro.text import prepare_text


TEXTS = (
    "你好。",
    "千里之行，始于足下。",
    "今天阳光很好，我们出去走走吧。",
    "请在下一个路口右转，前方就是目的地。",
    "您的订单已经发货，预计明天下午3点送达，请留意电话通知。",
    "欢迎来到语音合成体验区，输入一段中文，就能听到自然流畅的声音。",
    "现在开始介绍这个项目，我们希望把文字转换成语音，让设备能够清楚地播报信息，并与用户交流。",
    "这个语音助手可以朗读消息、播报天气和提醒日程，也能为智能家居提供语音反馈，帮助大家更方便地使用日常设备。",
    "我们的任务是把语音合成模型部署到芯片上，先完成文本处理和模型导出，再验证不同长度的输入，最后检查声音是否完整、清晰，以及运行结果是否稳定。",
    "本次任务是为设备增加中文语音能力，我们先分析模型并完成静态图导出，再做好量化适配，让文字自动选择模型，最后用真实句子检查发音、停顿和完整性，并提供演示程序。",
)
# Prior approved minimal-CPU graph-patch experiment, same seed and PCM encoder.
APPROVED_CASE10_WAV_SHA256 = "b8c9ec62f6c299c852671739bfd04ff9deed2502ae9c660551fb8e7c670220bd"


def write_wav(path: Path, waveform: np.ndarray) -> None:
    pcm = np.round(np.clip(waveform, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm.tobytes())


def validate(export_dir: Path, output_dir: Path, device: str) -> dict:
    runtime = KokoroBucketedRuntime.from_export(export_dir, device=device)
    assets = runtime.meta["runtime_assets"]
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"complete": False, "cases": [], "all_routes_exercised": False}
    report_path = output_dir / "report.json"
    try:
        for index, text in enumerate(TEXTS, 1):
            print(f"Validating text {index}/10", flush=True)
            tokens, phonemes = prepare_text(text, export_dir / assets["config_file"], max_tokens=256)
            style = load_voice_style(export_dir / assets["voice_pack"]["file"], tokens.size - 2).numpy()
            waveform, synthesis = runtime.synthesize(tokens, style)
            if not np.isfinite(waveform).all() or not np.any(waveform):
                raise RuntimeError(f"invalid waveform for text {index}")
            if waveform.size != synthesis["frame_length"] * 600:
                raise RuntimeError(f"incorrect waveform crop for text {index}")
            path = output_dir / f"{index:02d}.wav"
            write_wav(path, waveform)
            digest = sha256(path)
            report["cases"].append(
                {
                    "index": index,
                    "text": text,
                    "phonemes": phonemes,
                    **synthesis,
                    "wav_file": path.name,
                    "wav_sha256": digest,
                    "seconds": waveform.size / SAMPLE_RATE,
                    "peak_abs": float(np.max(np.abs(waveform))),
                    "rms": float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))),
                    "clip_fraction": float(np.mean(np.abs(waveform) >= 1)),
                }
            )
            if index == 10:
                report["approved_case10_pcm_exact"] = digest == APPROVED_CASE10_WAV_SHA256
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            runtime.clear_runner_cache()
        exercised = {case["route"] for case in report["cases"]}
        expected = {route["key"] for route in runtime.available_routes()}
        if exercised != expected:
            raise RuntimeError(f"text suite did not cover every route: {exercised} vs {expected}")
        report["all_routes_exercised"] = True
        report["complete"] = True
        return report
    finally:
        runtime.clear_runner_cache()
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    root = Path(args.export_dir).resolve()
    output = Path(args.output_dir).resolve() if args.output_dir else root / "validation"
    report = validate(root, output, args.device)
    print(
        json.dumps(
            {"report": str(output / "report.json"), "approved_case10_pcm_exact": report["approved_case10_pcm_exact"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
