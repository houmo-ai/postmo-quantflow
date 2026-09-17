import argparse

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

from examples_merak.llm.qwen3.debug_scripts.qwen3_xh2a_forward import _read_prompt


def main(args):
    model_dir = args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model: Qwen3ForCausalLM = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch.float16, device_map="auto"
    )  # type: ignore
    model.eval()
    prompt = _read_prompt(args.prompt)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Switches between thinking and non-thinking modes. Default is True.
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    max_new_tokens = 1024
    prefill_len = model_inputs["input_ids"].shape[1]
    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs, pad_token_id=tokenizer.pad_token_id, max_new_tokens=max_new_tokens, do_sample=False
        )
    output_ids = generated_ids[0][len(model_inputs["input_ids"][0]) :].tolist()

    # parsing thinking content
    try:
        # rindex finding 151668 (<tool_call>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    logger.info(f"prefill_len: {prefill_len}")
    logger.info(f"thinking content:{thinking_content}")
    logger.info(f"content:{content}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test Qwen3 model")
    parser.add_argument("--model-dir", type=str, default="./data/models/Qwen3-8B")
    parser.add_argument("--prompt", type=str, default="你多大了？用中文回答。")
    args = parser.parse_args()
    main(args)
