from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
from gptqmodel import GPTQModel
from transformers import AutoTokenizer, TextStreamer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run causal language model inference with PyTorch and Transformers on CPU or CUDA GPU."
    )
    parser.add_argument("--model", required=True, help="Hugging Face model path or model ID.")

    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask the chat template to append the model's thinking prefix.",
    )
    parser.add_argument(
        "--input",
        help="Input JSONL file. Each line may contain prompt, query, messages, or raw_data.",
    )
    parser.add_argument("--prompt", default="Reply with only the letter A.")
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="float16",
        help="Model and computation dtype.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, help="Maximum number of input records to process.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling during generation.")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)

    return parser


def read_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.input:
        return [{"id": "smoke-1", "prompt": args.prompt}]

    input_path = Path(args.input)
    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        records = parsed if isinstance(parsed, list) else [parsed]

    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{input_path} must contain JSON objects")
    if args.limit is not None:
        records = records[: args.limit]
    for line_number, record in enumerate(records, start=1):
        record.setdefault("id", str(line_number))
    return records


def get_prompt_source(record: dict[str, Any]) -> dict[str, Any]:
    raw_data = record.get("raw_data")
    if isinstance(raw_data, dict):
        return {**raw_data, **record}
    return record


def render_prompt(record: dict[str, Any], tokenizer: Any, enable_thinking: bool = True) -> str:
    source = get_prompt_source(record)
    messages = source.get("messages")
    metadata = source.get("metadata")
    metadata_prompt = metadata.get("prompt") if isinstance(metadata, dict) else None
    if metadata_prompt is not None:
        messages = [{"role": "user", "content": str(metadata_prompt)}]
    if messages is None:
        for field in ("prompt", "query", "text"):
            if source.get(field) is not None:
                messages = [{"role": "user", "content": str(source[field])}]
                break
    if messages is not None:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("Input has text or messages, but tokenizer has no chat_template")
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    raise ValueError("Each input record needs prompt, query, text, messages, or raw_data containing one of them")


def make_generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        for name in ("temperature", "top_p", "top_k"):
            value = getattr(args, name)
            if value is not None:
                generation_kwargs[name] = value
    if args.repetition_penalty is not None:
        generation_kwargs["repetition_penalty"] = args.repetition_penalty
    return generation_kwargs


def resolve_tokenizer_path(model_path: str) -> str:
    path = Path(model_path)
    if not path.is_dir():
        return model_path
    if (path / "tokenizer.json").is_file():
        return model_path

    tokenizer_dirs = sorted(
        child for child in path.iterdir() if child.is_dir() and (child / "tokenizer.json").is_file()
    )
    if len(tokenizer_dirs) == 1:
        return str(tokenizer_dirs[0])
    return model_path


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    args = build_parser().parse_args()
    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.gptqmodel_wrap import register_spark2_5_gptqmodel
    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.spark_2_5 import register_spark2_5

    register_spark2_5()
    register_spark2_5_gptqmodel()
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; falling back to CPU")
        device_name = "cpu"
    device = torch.device(device_name)

    tokenizer_path = resolve_tokenizer_path(args.model)
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading GPTQModel checkpoint from {args.model} on {device}")
    model: Any = GPTQModel.load(
        args.model,
        device=str(device),
        dtype=getattr(torch, args.dtype),
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    print(f"Using device: {device}")

    if args.seed is not None:
        set_seed(args.seed)
        print(f"Random seed set to {args.seed}")

    generation_kwargs = make_generation_kwargs(args)
    records = read_records(args)
    for record in records:
        prompt = render_prompt(record, tokenizer, enable_thinking=args.thinking)
        encoded = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
        encoded.pop("token_type_ids", None)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        started = time.perf_counter()
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                **generation_kwargs,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                streamer=streamer,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        prompt_length = encoded["input_ids"].shape[1]
        output_ids = generated[0, prompt_length:].tolist()
        result = {
            "id": str(record["id"]),
            "prompt": prompt,
            "text": tokenizer.decode(output_ids, skip_special_tokens=False),
            "input_tokens": prompt_length,
            "output_tokens": len(output_ids),
            "latency_seconds": round(time.perf_counter() - started, 6),
            "device": str(device),
            "dtype": args.dtype,
            "sampling": {
                "do_sample": args.do_sample,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "repetition_penalty": args.repetition_penalty,
                "max_new_tokens": args.max_new_tokens,
            },
            "raw_data": record.get("raw_data"),
        }
        print(f"Prompt:\n{prompt}\n\nResponse:\n{result['text']}")


if __name__ == "__main__":
    main()
