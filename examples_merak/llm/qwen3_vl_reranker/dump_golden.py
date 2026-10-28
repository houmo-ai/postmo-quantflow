"""Generate synthetic or real image Golden for both exported graphs."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image")
    parser.add_argument("--query", default="A woman and a dog playing on the beach.")
    args = parser.parse_args()
    from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.workflow import dump_reranker_golden

    inputs = {"query": {"text": args.query}, "documents": [{"image": args.image}]} if args.image else None
    result = dump_reranker_golden(
        args.metadata, args.device, inputs, golden_name="golden_real" if args.image else "golden_synthetic"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
