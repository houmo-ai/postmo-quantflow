"""Transcribe a local recording through the public artifact interface."""

from __future__ import annotations

import argparse
import json

from xhmodel_merak.model_inference import inference_model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, help="build_model 返回的 output_dir")
    parser.add_argument("--audio", required=True, help="本地 16 kHz 单声道 16-bit PCM WAV/FLAC")
    parser.add_argument("--prompt", help="覆盖默认转写指令；省略则用模型默认 prompt")
    parser.add_argument("--max-new-tokens", type=int, help="最多生成多少个 token；省略则按音频时长估计")
    parser.add_argument("--device", help="推理设备，例如 cuda:0")
    parser.add_argument("--golden", nargs="+", help="额外生成指定组件的 golden，例如 encode prefill decode")
    args = parser.parse_args(argv)

    inputs: dict[str, object] = {"audio": args.audio}
    if args.prompt is not None:
        inputs["prompt"] = args.prompt
    if args.max_new_tokens is not None:
        inputs["max_new_tokens"] = args.max_new_tokens
    result = inference_model(
        artifact_dir=args.artifact_dir,
        expected_model_id="moss-transcribe-diarize-0.9b",
        inputs=inputs,
        device=args.device,
        golden=args.golden,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
