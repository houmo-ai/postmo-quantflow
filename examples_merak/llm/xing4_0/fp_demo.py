"""Run the original Xing4.0 checkpoint through Transformers."""

import argparse


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", default="Briefly explain the basic principles of quantum computing.")
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, trust_remote_code=True, device_map="auto", dtype=torch.bfloat16,
    )
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **inputs, do_sample=True, top_k=0, top_p=0.95, temperature=1.0,
        repetition_penalty=1.05, max_new_tokens=args.max_new_tokens,
    )
    response = tokenizer.decode(generated_ids[0, inputs.input_ids.shape[1]:], skip_special_tokens=False)
    print(response.split("</think>")[-1].strip())


if __name__ == "__main__":
    main()
