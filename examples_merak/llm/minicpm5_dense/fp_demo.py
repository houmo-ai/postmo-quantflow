import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MiniCPM5 FP inference.")
    parser.add_argument(
        "--model-dir",
        default="/data01/datasets/MiniCPM5-2.6B-0822/models",
    )
    parser.add_argument("--prompt", default="Who are you? Please briefly introduce yourself.")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype="auto",
        device_map="auto",
    )
    messages = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    prompt_length = inputs["input_ids"].shape[-1]
    print(tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True))


if __name__ == "__main__":
    main()