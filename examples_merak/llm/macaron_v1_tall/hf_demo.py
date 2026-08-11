"""Run the Hugging Face + PEFT reference for one or all Macaron adapters."""

import argparse
import time
from pathlib import Path

import torch


ADAPTER_NAMES = ["L0", "L1", "L2", "L3"]
ADAPTER_ROLES = {
    "L0": "Chat",
    "L1": "Agent",
    "L2": "Coding",
    "L3": "GenUI",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Macaron-V1-Tall MoL inference demo."
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Macaron-V1-Tall model directory (contains loras/L0..L3).",
    )
    parser.add_argument(
        "--prompt",
        default="Write a Python function to sort a list.",
        help="User prompt for generation.",
    )
    parser.add_argument(
        "--adapter",
        default=None,
        choices=ADAPTER_NAMES,
        help="Run a specific adapter; without it, the demo prints all adapters.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate. Default: 128",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device. Default: cuda",
    )
    return parser.parse_args()


def load_base_model(model_dir: str):
    """Load the base Qwen3.6-35B-A3B model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(">>> Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_dir, trust_remote_code=True
    )

    print(">>> Loading base model (FP16, device_map=auto)...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(
        f"    Loaded in {time.time() - t0:.1f}s: {type(model).__name__}, {n_params:.2f}B params"
    )
    return model, tokenizer


def load_lora_adapters(
    model, model_dir: str, adapter_names: list[str] | None = None
):
    """Load the requested LoRA adapters via PEFT (all four by default)."""
    from peft import PeftModel

    loras_dir = Path(model_dir) / "loras"
    adapter_names = adapter_names or ADAPTER_NAMES
    unknown = set(adapter_names) - set(ADAPTER_NAMES)
    if unknown:
        raise ValueError(f"Unknown adapters: {sorted(unknown)}")
    if not adapter_names:
        raise ValueError("At least one LoRA adapter is required")
    print(">>> Loading LoRA adapters...")

    t0 = time.time()
    peft_model = PeftModel.from_pretrained(
        model,
        str(loras_dir / adapter_names[0]),
        adapter_name=adapter_names[0],
        is_trainable=False,
    )
    first_adapter = adapter_names[0]
    print(
        f"    {first_adapter} ({ADAPTER_ROLES[first_adapter]}) loaded in {time.time() - t0:.1f}s"
    )

    for name in adapter_names[1:]:
        t0 = time.time()
        peft_model.load_adapter(
            str(loras_dir / name), adapter_name=name
        )
        print(
            f"    {name} ({ADAPTER_ROLES[name]}) loaded in {time.time() - t0:.1f}s"
        )

    # Verify dynamic mode (not merged)
    merged = sum(
        1
        for _, m in peft_model.named_modules()
        if getattr(m, "merged", False)
    )
    lora_count = sum(
        1
        for _, m in peft_model.named_modules()
        if hasattr(m, "lora_A") and len(m.lora_A) > 0
    )
    print(
        f"    Total: {lora_count} LoRA modules, merged={merged} (0=dynamic)"
    )

    return peft_model


def generate(
    peft_model,
    tokenizer,
    prompt: str,
    adapter: str,
    max_new_tokens: int,
) -> str:
    """Generate text with a specific adapter."""
    peft_model.set_adapter(adapter)

    messages = [{"role": "user", "content": prompt}]
    # Match the HMONNX demo default (hmonnx_demo.py think=False),
    # so HF and HMONNX comparisons use the same non-thinking chat template.
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(peft_model.device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    t0 = time.time()
    with torch.no_grad():
        out = peft_model.generate(**inputs, **gen_kwargs)
    elapsed = time.time() - t0

    response = tokenizer.decode(
        out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    n_tokens = out.shape[1] - inputs.input_ids.shape[1]
    print(
        f"    Generated {n_tokens} tokens in {elapsed:.1f}s ({n_tokens / elapsed:.1f} tok/s)"
    )
    return response


def main() -> None:
    args = parse_args()

    model, tokenizer = load_base_model(args.model_dir)
    peft_model = load_lora_adapters(model, args.model_dir)

    if args.adapter:
        # Manual adapter selection
        print(
            f"\n>>> Using adapter {args.adapter} ({ADAPTER_ROLES[args.adapter]})"
        )
        response = generate(
            peft_model,
            tokenizer,
            args.prompt,
            args.adapter,
            args.max_new_tokens,
        )
        print(
            f"\n=== {args.adapter} ({ADAPTER_ROLES[args.adapter]}) Response ==="
        )
        print(response)
    else:
        # Auto-route: generate with all adapters for comparison
        print(f"\n>>> Prompt: {args.prompt!r}")
        print(">>> Generating with all 4 adapters for comparison...\n")

        for name in ADAPTER_NAMES:
            print(f"--- {name} ({ADAPTER_ROLES[name]}) ---")
            response = generate(
                peft_model,
                tokenizer,
                args.prompt,
                name,
                args.max_new_tokens,
            )
            # Truncate for display
            display = (
                response[:300] + "..."
                if len(response) > 300
                else response
            )
            print(f"    {display}\n")

    print("=== Done ===")


if __name__ == "__main__":
    main()
