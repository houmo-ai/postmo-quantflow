"""Run a packaged WeMM embedding artifact through the unified API."""

from __future__ import annotations

import argparse
import json

from xhmodel_merak.model_inference.api import inference_model


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--text")
    parser.add_argument("--image")
    parser.add_argument("--video")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    inputs = {name: value for name in ("text", "image", "video") if (value := getattr(args, name)) is not None}
    if not inputs:
        parser.error("provide at least one of --text, --image, or --video")
    result = inference_model(artifact_dir=args.artifact_dir, inputs=inputs, device=args.device)
    print(json.dumps(result.to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
