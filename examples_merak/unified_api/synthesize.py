"""Synthesize Chinese speech through the public artifact interface."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

from xhmodel_merak.model_inference import inference_model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output-wav", required=True)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--device")
    args = parser.parse_args(argv)

    import numpy as np

    result = inference_model(
        artifact_dir=args.artifact_dir,
        expected_model_id="kokoro-82m-v1.1-zh",
        inputs={"text": args.text, "speed": args.speed},
        device=args.device,
    )
    path = Path(args.output_wav).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(result.waveform, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(result.sample_rate)
        target.writeframes(pcm.tobytes())
    print(
        json.dumps(
            {
                "output_wav": str(path),
                "sample_rate": result.sample_rate,
                "duration_s": result.duration_s,
                "segments": result.segments,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
