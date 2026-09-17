import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from transformers import TextStreamer

from xhmodel_merak.xh_llm import (
    AutoLLMHONNXModel,
    LLMInferenceContextManager,
)
from xhquant.api import get_xhquant_logger, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.qwen3 import XHQwen3HMONNXModel


@dataclass(frozen=True)
class GenerationStep:
    """Information produced by one HMONNX forward call."""

    index: int
    phase: str
    past_length: int
    input_length: int
    input_token_id: int | None
    output_token_id: int


def _normalize_token_ids(value: Any) -> set[int]:
    if value is None:
        return set()
    if isinstance(value, int):
        return {value}
    if isinstance(value, (list, tuple, set)):
        return {int(token_id) for token_id in value}
    raise TypeError(f"token IDs must be an int or a sequence of ints, got {type(value).__name__}")


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    logits = getattr(output, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    if isinstance(output, dict) and isinstance(output.get("logits"), torch.Tensor):
        return output["logits"]
    raise TypeError(f"model forward output does not contain tensor logits: {type(output).__name__}")


def _next_token_id(output: Any) -> int:
    logits = _extract_logits(output)
    if logits.ndim < 2:
        raise ValueError(f"logits must have at least two dimensions, got shape {tuple(logits.shape)}")
    last_logits = logits[:, -1, :] if logits.ndim >= 3 else logits
    next_tokens = last_logits.argmax(dim=-1)
    if next_tokens.numel() != 1:
        raise ValueError(f"step-by-step generation requires batch size 1, got {next_tokens.numel()}")
    return int(next_tokens.item())


def _make_step_reporter(tokenizer: Any, logger: Any) -> Callable[[GenerationStep], None]:
    def report(step: GenerationStep) -> None:
        token_text = tokenizer.decode([step.output_token_id], skip_special_tokens=False)
        input_token = "prompt" if step.input_token_id is None else str(step.input_token_id)
        logger.info(
            f"[step {step.index:04d}] phase={step.phase} past_length={step.past_length} "
            f"input={input_token} -> token_id={step.output_token_id} token={token_text!r}"
        )

    return report


def _generate_hmonnx_step_by_step(
    model,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
    step_callback: Callable[[GenerationStep], None] | None = None,
) -> list[int]:
    """Greedily generate with explicit HMONNX prefill/decode forward calls."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"input_ids must have shape [1, sequence], got {tuple(input_ids.shape)}")
    prompt_length = int(input_ids.shape[1])
    if prompt_length == 0:
        raise ValueError("input_ids must contain at least one token")
    if max_new_tokens == 0:
        return []

    model.set_prefill()
    data_processor = model.get_data_preprocessor()
    prefill_chunk_length = int(model.get_input_sequence_length())
    if prefill_chunk_length <= 0:
        raise RuntimeError(f"invalid HMONNX prefill chunk length: {prefill_chunk_length}")

    prefill_output = None
    for start in range(0, prompt_length, prefill_chunk_length):
        end = min(start + prefill_chunk_length, prompt_length)
        prefill_inputs = data_processor(
            {
                "input_ids": input_ids[:, start:end],
                "past_seq_length": start,
            }
        )
        prefill_output = model.forward(*prefill_inputs)
    if prefill_output is None:
        raise RuntimeError("HMONNX prefill did not execute")

    next_token = _next_token_id(prefill_output)
    generated_ids = [next_token]
    if step_callback is not None:
        step_callback(GenerationStep(0, "prefill", 0, prompt_length, None, next_token))

    if len(generated_ids) < max_new_tokens and next_token not in eos_token_ids:
        model.set_decode()
    while len(generated_ids) < max_new_tokens and next_token not in eos_token_ids:
        generation_index = len(generated_ids)
        past_length = prompt_length + generation_index - 1
        input_token = next_token
        decode_ids = torch.tensor([[input_token]], dtype=torch.long)
        decode_inputs = data_processor(
            {
                "input_ids": decode_ids,
                "past_seq_length": past_length,
            }
        )
        next_token = _next_token_id(model.forward(*decode_inputs))
        generated_ids.append(next_token)
        if step_callback is not None:
            step_callback(GenerationStep(generation_index, "decode", past_length, 1, input_token, next_token))
    return generated_ids


def _extract_query_from_prompt_object(prompt_data: Any, *, source: str) -> str:
    if not isinstance(prompt_data, dict):
        raise TypeError(f"{source} must contain an object, got {type(prompt_data).__name__}")
    query = prompt_data.get("query")
    if not isinstance(query, str):
        raise TypeError(f"{source} field 'query' must be a string")
    return query


def _read_prompt(value: str) -> str:
    prompt_path = Path(value)
    if not prompt_path.is_file():
        return value

    prompt_text = prompt_path.read_text(encoding="utf-8")
    suffix = prompt_path.suffix.lower()
    if suffix == ".jsonl":
        lines = [line.strip() for line in prompt_text.splitlines() if line.strip()]
        if not lines:
            raise ValueError(f"JSONL prompt file is empty: {prompt_path}")
        return _extract_query_from_prompt_object(json.loads(lines[0]), source="JSONL prompt")
    if suffix == ".json":
        return _extract_query_from_prompt_object(json.loads(prompt_text), source="JSON prompt")
    return prompt_text


def main(args):
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative")
    debug = args.debug
    xhquant_init(None, debug)
    logger = get_xhquant_logger()
    model_meta_file = args.config
    hmonnx_model: XHQwen3HMONNXModel = AutoLLMHONNXModel.from_pretrained(model_meta_file)
    if type(hmonnx_model).__name__ != "XHQwen3HMONNXModel":
        raise TypeError(f"Expected model type XHQwen3HMONNXModel, but got {type(hmonnx_model).__name__}")

    hmonnx_model.enable_auto_offload = args.auto_offload

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enable_think = args.think
    prompt = _read_prompt(args.prompt)
    messages = [
        {"role": "user", "content": prompt},
    ]

    tokenizer = hmonnx_model.get_tokenizer()
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_think,  # Switches between thinking and non-thinking modes. Default is True.
    )

    model_inputs = tokenizer([text], return_tensors="pt", truncation=True).to(device)
    streamer = TextStreamer(tokenizer)
    hmonnx_model.to(device)
    if args.fast:
        hmonnx_model.to_fast()
    max_new_tokens = args.max_new_tokens
    if args.golden:
        max_new_tokens = 2
        hmonnx_model.enable_golden = True

    eos_token_ids = _normalize_token_ids(getattr(tokenizer, "eos_token_id", None))
    step_reporter = _make_step_reporter(tokenizer, logger)

    def report_step(step: GenerationStep) -> None:
        step_reporter(step)
        streamer.put(torch.tensor([[step.output_token_id]], dtype=torch.long))

    contexts = [
        TimeProfiler("hmonnx_forward", logger),
        MemoryTracker(device=device, name="forward", logger=logger),
        LLMInferenceContextManager(hmonnx_model),
        torch.inference_mode(),
    ]
    try:
        with ContextManagers(contexts):
            output_ids = _generate_hmonnx_step_by_step(
                hmonnx_model,
                model_inputs.input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_ids=eos_token_ids,
                step_callback=report_step,
            )
    finally:
        streamer.end()

    # parsing thinking content
    try:
        think_end_token_id = tokenizer.encode("</think>")[-1]  # 获取 </think> 的 token ID
        index = len(output_ids) - output_ids[::-1].index(think_end_token_id)
    except (ValueError, IndexError):
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    logger.info(f"think: {enable_think}")
    stop_reason = "eos" if output_ids and output_ids[-1] in eos_token_ids else "max_new_tokens"
    logger.info(f"generated_tokens: {len(output_ids)}, stop_reason: {stop_reason}")
    if len(thinking_content) > 0:
        logger.info(f"{'-' * 20} thinking content {'-' * 20}")
    logger.info(f"{thinking_content}")
    logger.info(f"{'-' * 20} content {'-' * 20}")
    logger.info(f"{content}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="work_dirs/qwen3_8b_xh2a_2k/hmquant_xh2_qwen3-1.7b_w8a8_20260309/golden_meta_info.json",
    )

    parser.add_argument("--fast", action="store_true", help="run in fast mode")
    parser.add_argument("--debug", action="store_true", help="Whether to run in debug mode")
    parser.add_argument("--prompt", type=str, default="你多大了？用中文回答。")
    parser.add_argument("--think", action="store_true", help="enable think mode")
    parser.add_argument("--golden", action="store_true", help="Whether to save golden outputs for testing.")
    parser.add_argument("--max-new-tokens", type=int, default=10, help="The maximum number of tokens to generate.")
    parser.add_argument(
        "--auto-offload", action="store_true", help="Whether to enable auto offload, only for debug and development"
    )
    args = parser.parse_args()
    main(args)
