"""Run the same scoring runtime through the public inference interface."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--input-json", help="JSON with query, documents and optional instruction")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from xhmodel_merak.model_inference import inference_model

    inputs = (
        json.loads(Path(args.input_json).read_text())
        if args.input_json
        else {
            "query": {"text": "What is the capital of France?"},
            "documents": [{"text": "Paris is the capital of France."}, {"text": "Whales live in the ocean."}],
        }
    )
    result = inference_model(artifact_dir=args.artifact_dir, inputs=inputs, device=args.device)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
