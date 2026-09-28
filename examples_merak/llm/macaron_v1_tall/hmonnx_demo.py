"""Run Macaron adapters exported by the shared Qwen3.5 workflow."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

def find_meta_file(export_dir: Path) -> str:
    meta = export_dir / "golden_meta_info.json"
    if meta.is_file():
        return str(meta)
    for child in export_dir.iterdir():
        if child.is_dir() and child.name.startswith("hmquant"):
            candidate = child / "golden_meta_info.json"
            if candidate.is_file():
                return str(candidate)
    raise FileNotFoundError(
        f"No golden_meta_info.json found under {export_dir}"
    )


def run_generation(
    model,
    prompt: str,
    max_new_tokens: int,
    runtime_device: str,
    think: bool = False,
):
    """Generate text through the repository HMONNX runtime."""
    from xhmodel_merak.xh_llm import LLMInferenceContextManager

    tokenizer = model.get_tokenizer()
    if model.pad_token_id is None:
        if tokenizer.pad_token_id is None:
            raise ValueError("The tokenizer does not define pad_token_id")
        # Compatibility for exports created before padding was serialized.
        model.pad_token_id = tokenizer.pad_token_id

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=think,
    )
    model_inputs = tokenizer(
        [text], return_tensors="pt", truncation=True
    ).to(runtime_device)
    prompt_len = model_inputs.input_ids.shape[1]

    generate_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=model.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    start = time.perf_counter()
    with LLMInferenceContextManager(model, devices=[runtime_device]):
        generated_ids = model.generate(
            **model_inputs, **generate_kwargs
        )
    latency_s = time.perf_counter() - start

    output_ids = generated_ids[0][prompt_len:].tolist()
    eos = tokenizer.eos_token_id
    if eos is not None and eos in output_ids:
        output_ids = output_ids[: output_ids.index(eos)]
    output_text = tokenizer.decode(
        output_ids, skip_special_tokens=True
    ).strip("\n")
    num_tokens = len(output_ids)
    return {
        "output_text": output_text,
        "prompt_token_ids": model_inputs.input_ids[0].tolist(),
        "output_token_ids": output_ids,
        "num_tokens": num_tokens,
        "latency_s": latency_s,
        "tokens_per_second": num_tokens / latency_s
        if latency_s > 0
        else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Macaron-V1-Tall HMONNX generation demo"
    )
    parser.add_argument(
        "--export-dir",
        type=str,
        required=True,
        help="Path to a Qwen workflow export containing LoRA adapters",
    )
    parser.add_argument(
        "--adapters",
        nargs="+",
        default=None,
        help="Exported adapter names; defaults to all entries in metadata.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a Python function to sort a list.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--think",
        action="store_true",
        help="enable thinking mode in the chat template",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="0",
        help="Execution device: 'cpu' or a GPU id like '0'/'cuda:0'",
    )
    parser.add_argument(
        "--auto-offload",
        action="store_true",
        help="enable auto offload (models exceeding single-card memory)",
    )
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    from xhmodel_merak.xh_llm import AutoLLMHONNXModel


    device_token = args.device.lower().removeprefix("cuda:")
    if device_token == "cpu":
        device_map, runtime_device = ["cpu"], "cpu"
    elif device_token.isdigit():
        device_map = [int(device_token)]
        runtime_device = f"cuda:{device_token}"
    else:
        raise ValueError("--device must be 'cpu' or one GPU id")
    print(f"Device: {runtime_device} (device_map={device_map})")

    root_meta = Path(find_meta_file(export_dir))
    metadata = json.loads(root_meta.read_text())
    entries = {item["name"]: item for item in metadata.get("lora_adapters", [])}
    selected = args.adapters or list(entries)
    if not selected:
        raise ValueError("The export contains no LoRA adapters")
    for adapter in selected:
        if adapter not in entries:
            raise ValueError(f"Unknown exported adapter {adapter!r}; available: {list(entries)}")
        from xhmodel_merak.model_artifact.files import confined_file
        meta_file = str(confined_file(root_meta.parent.resolve(), entries[adapter]["meta_file"], "lora.meta_file"))
        print(f"[{adapter}] Meta: {meta_file}")
        print(f"[{adapter}] Loading Qwen3.5 MoE adapter...")
        t0 = time.time()
        model = AutoLLMHONNXModel.from_pretrained(
            meta_file,
            device_map=device_map,
            enable_auto_offload=args.auto_offload,
            enable_golden=False,
        )
        model.to(runtime_device)
        print(f"[{adapter}] Model loaded in {time.time() - t0:.1f}s")
        result = run_generation(
            model,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            runtime_device=runtime_device,
            think=args.think,
        )
        print(
            f"[{adapter}] Output ({result['num_tokens']} tokens, "
            f"{result['tokens_per_second']:.1f} tok/s):"
        )
        print(result["output_text"][:500])
        del model
        gc.collect()


if __name__ == "__main__":
    main()
