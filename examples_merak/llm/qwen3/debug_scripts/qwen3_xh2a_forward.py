"""Run Qwen3 through a Merak state, one forward call per decoding step.

This entry supports ``wrap`` and the eager/fronted/quanted Merak states.  It
explicitly switches from prefill to single-token decode and prints the token
predicted by every call.  Use ``native_qwen3_forward.py`` for the corresponding
Hugging Face native path.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


DEFAULT_CONFIG_PATH = "configs_merak/xh2a/llm_models/qwen3/8b/qwen3_8b_xh2a_w4a8_hf_gptq_2k.py"
DEFAULT_MODEL_DIR = None
DEFAULT_MAX_NEW_TOKENS = 1024
MERAK_EVAL_TYPES = (
    "wrap",
    "eager_fast",
    "eager_aligned",
    "fronted",
    "quanted_disable",
    "quanted_fast",
    "quanted_aligned",
)


@dataclass(frozen=True)
class GenerationStep:
    """Information produced by one model forward call."""

    index: int
    phase: str
    past_length: int
    input_length: int
    input_token_id: int | None
    output_token_id: int


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


def _torch_dtype(name: str) -> torch.dtype:
    dtypes = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return dtypes[name.lower()]
    except KeyError as error:
        raise ValueError(f"unsupported dtype {name!r}; choose one of {sorted(dtypes)}") from error


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


def _next_token_id(output: Any, *, valid_length: int | None = None) -> int:
    logits = _extract_logits(output)
    if logits.ndim < 2:
        raise ValueError(f"logits must have at least two dimensions, got shape {tuple(logits.shape)}")
    position = valid_length - 1 if valid_length is not None and logits.ndim >= 3 and logits.shape[1] != 1 else -1
    last_logits = logits[:, position, :] if logits.ndim >= 3 else logits
    next_tokens = last_logits.argmax(dim=-1)
    if next_tokens.numel() != 1:
        raise ValueError(f"step-by-step generation requires batch size 1, got {next_tokens.numel()}")
    return int(next_tokens.item())


def _new_dynamic_cache(model: Any) -> Any:
    from transformers import DynamicCache

    try:
        return DynamicCache(config=model.config)
    except TypeError:
        return DynamicCache()


def _generate_hf_step_by_step(
    model: Any,
    input_ids: torch.Tensor,
    *,
    device: torch.device,
    max_new_tokens: int,
    eos_token_ids: set[int],
    step_callback: Callable[[GenerationStep], None] | None = None,
) -> list[int]:
    """Greedily decode a HF-compatible model with an explicit DynamicCache."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"input_ids must have shape [1, sequence], got {tuple(input_ids.shape)}")
    if max_new_tokens == 0:
        return []

    model_ids = input_ids.to(device=device, dtype=torch.long)
    prompt_length = int(model_ids.shape[1])
    cache = _new_dynamic_cache(model)
    output = model(
        input_ids=model_ids,
        attention_mask=torch.ones_like(model_ids),
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
        return_dict=True,
    )
    next_token = _next_token_id(output)
    cache = getattr(output, "past_key_values", None)
    if cache is None:
        raise RuntimeError("HF-compatible prefill did not return past_key_values")

    generated_ids = [next_token]
    if step_callback is not None:
        step_callback(GenerationStep(0, "prefill", 0, prompt_length, None, next_token))

    while len(generated_ids) < max_new_tokens and next_token not in eos_token_ids:
        generation_index = len(generated_ids)
        past_length = prompt_length + generation_index - 1
        input_token = next_token
        decode_ids = torch.tensor([[input_token]], dtype=torch.long, device=device)
        output = model(
            input_ids=decode_ids,
            attention_mask=torch.ones((1, past_length + 1), dtype=torch.long, device=device),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            return_dict=True,
        )
        next_token = _next_token_id(output)
        cache = getattr(output, "past_key_values", None)
        if cache is None:
            raise RuntimeError(f"HF-compatible decode step {generation_index} did not return past_key_values")
        generated_ids.append(next_token)
        if step_callback is not None:
            step_callback(GenerationStep(generation_index, "decode", past_length, 1, input_token, next_token))
    return generated_ids


def _generate_wrap_step_by_step(
    xh_model: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_ids: set[int],
    step_callback: Callable[[GenerationStep], None] | None = None,
) -> list[int]:
    """Greedily decode with fixed-size prefill chunks and single-token decode."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"input_ids must have shape [1, sequence], got {tuple(input_ids.shape)}")
    if max_new_tokens == 0:
        return []

    prompt_length = int(input_ids.shape[1])
    if prompt_length == 0:
        raise ValueError("input_ids must contain at least one token")
    chunk_length = int(xh_model.config.prefill_chunk_length)
    if chunk_length <= 0:
        raise ValueError(f"prefill chunk length must be positive, got {chunk_length}")
    xh_model.set_prefill()
    xh_model.set_input_sequence_length(chunk_length)
    data_processor = xh_model.get_data_preprocessor()
    for start in range(0, prompt_length, chunk_length):
        end = min(start + chunk_length, prompt_length)
        prefill_inputs = data_processor({"input_ids": input_ids[:, start:end], "past_seq_length": start})
        prefill_output = xh_model.forward(*prefill_inputs)
    next_token = _next_token_id(prefill_output, valid_length=end - start)

    generated_ids = [next_token]
    if step_callback is not None:
        step_callback(
            GenerationStep(
                index=0,
                phase="prefill",
                past_length=0,
                input_length=prompt_length,
                input_token_id=None,
                output_token_id=next_token,
            )
        )

    if len(generated_ids) < max_new_tokens and next_token not in eos_token_ids:
        xh_model.set_decode()
    while len(generated_ids) < max_new_tokens and next_token not in eos_token_ids:
        generation_index = len(generated_ids)
        past_length = prompt_length + generation_index - 1
        decode_ids = torch.tensor([[next_token]], dtype=torch.long)
        decode_inputs = data_processor({"input_ids": decode_ids, "past_seq_length": past_length})
        input_token = next_token
        next_token = _next_token_id(xh_model.forward(*decode_inputs))
        generated_ids.append(next_token)
        if step_callback is not None:
            step_callback(
                GenerationStep(
                    index=generation_index,
                    phase="decode",
                    past_length=past_length,
                    input_length=1,
                    input_token_id=input_token,
                    output_token_id=next_token,
                )
            )
    return generated_ids


def _load_merak_config(config_path: str, model_dir: str | None, seed: int) -> tuple[Any, Any]:
    from xhmodel_merak.xh_llm import AutoLLMConfig
    from xhquant.api import Config

    cfg = Config.fromfile(config_path)
    if model_dir:
        if hasattr(cfg, "hf_model_dir"):
            cfg.hf_model_dir = model_dir
        cfg.model.hf_model = model_dir
    cfg.seed = seed
    return cfg, AutoLLMConfig.from_pretrained(cfg.model)


def _build_xh_model(model_cfg: Any, work_dir: Path) -> Any:
    from xhmodel_merak.xh_llm import AutoLLMModel

    xh_model = AutoLLMModel.from_pretrained(config=model_cfg)
    xh_model.work_dir = str(work_dir)
    return xh_model


def _validate_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested, but CUDA is unavailable: {device}")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"invalid CUDA device index {index}")
        device = torch.device("cuda", index)
        torch.cuda.set_device(device)
    return device


def _make_step_reporter(tokenizer: Any, logger: Any) -> Callable[[GenerationStep], None]:
    def report(step: GenerationStep) -> None:
        token_text = tokenizer.decode([step.output_token_id], skip_special_tokens=False)
        input_token = "prompt" if step.input_token_id is None else str(step.input_token_id)
        logger.info(
            f"[step {step.index:04d}] phase={step.phase} past_length={step.past_length} "
            f"input={input_token} -> token_id={step.output_token_id} token={token_text!r}"
        )

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help="Merak Python model config.",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Optional override for model.hf_model in the config.",
    )
    parser.add_argument(
        "--eval-type",
        default="wrap",
        choices=MERAK_EVAL_TYPES,
        help="Merak model state to execute. Default: wrap.",
    )
    parser.add_argument(
        "--prompt",
        default="1+1等于几",
        help="Prompt text, UTF-8 text file, or .json/.jsonl file with a string 'query' field.",
    )
    parser.add_argument("--device", default="cuda:0", help="Execution device. Default: cuda:0.")
    parser.add_argument("--dtype", default="fp16", choices=("fp16", "bf16", "fp32"))
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--think", action="store_true", help="Enable Qwen3 thinking mode.")
    parser.add_argument("--work-dir", default="work_dirs/qwen3_forward")
    parser.add_argument("--debug", action="store_true", help="Enable xhquant debug logging.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative")
    config_path = Path(args.config_path)
    if not config_path.is_file():
        raise FileNotFoundError(f"model config does not exist: {config_path}")

    device = _validate_device(args.device)
    dtype = _torch_dtype(args.dtype)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    from xhquant.api import get_xhquant_logger, set_random_seed, xhquant_init

    xhquant_init(str(work_dir / f"qwen3_forward_{args.eval_type}.log"), args.debug)
    set_random_seed(args.seed)
    logger = get_xhquant_logger()

    cfg, model_cfg = _load_merak_config(str(config_path), args.model_dir, args.seed)
    model_dir = Path(model_cfg.hf_model)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_dir}")
    logger.info(f"Config:\n{cfg.pretty_text}")
    logger.info(f"Model Config:\n{model_cfg.to_json_string()}")

    xh_model = _build_xh_model(model_cfg, work_dir)
    prompt = _read_prompt(args.prompt)
    messages = [{"role": "user", "content": prompt}]
    processor = xh_model.get_tf_processor()
    tokenizer = processor.tokenizer
    model_inputs = processor.apply_chat_template(messages, enable_think=args.think)
    input_ids = model_inputs.input_ids.to(dtype=torch.long, device="cpu")
    total_length = int(input_ids.shape[1]) + args.max_new_tokens
    if total_length > int(model_cfg.context_max_length):
        raise ValueError(
            f"prompt plus generated tokens ({total_length}) exceeds model context limit {model_cfg.context_max_length}"
        )

    eos_token_ids = _normalize_token_ids(getattr(tokenizer, "eos_token_id", None))
    eos_token_ids.update(_normalize_token_ids(getattr(xh_model.config, "eos_token_id", None)))
    step_callback = _make_step_reporter(tokenizer, logger)
    logger.info(
        f"Running {args.eval_type} step-by-step generation: input_tokens={input_ids.shape[1]}, "
        f"max_new_tokens={args.max_new_tokens}, device={device}, dtype={dtype}"
    )

    from xhmodel_merak.xh_llm import LLMInferenceContextManager, LLMModelState
    from xhquant.core.context import get_current_context

    if not bool(xh_model.config.use_cache):
        raise RuntimeError("Merak step-by-step generation requires model.use_cache=true")
    get_current_context().kvcache.fast_mode = False
    # This script schedules prefill chunks explicitly; disable internal chunking.
    xh_model.config.enable_prefill_chunk = False
    xh_model.set_state(LLMModelState.from_string(args.eval_type))
    xh_model.to(device=device, dtype=dtype)
    xh_model.eval()
    state_model = xh_model.get_inference_model()
    eos_token_ids.update(
        _normalize_token_ids(getattr(getattr(state_model, "generation_config", None), "eos_token_id", None))
    )
    with LLMInferenceContextManager(xh_model, devices=[device]), torch.no_grad():
        if args.eval_type in {"eager_fast", "eager_aligned"}:
            generated_ids = _generate_hf_step_by_step(
                state_model,
                input_ids,
                device=device,
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=eos_token_ids,
                step_callback=step_callback,
            )
        else:
            generated_ids = _generate_wrap_step_by_step(
                xh_model,
                input_ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_ids=eos_token_ids,
                step_callback=step_callback,
            )

    generated_text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    stop_reason = "eos" if generated_ids and generated_ids[-1] in eos_token_ids else "max_new_tokens"
    logger.info(
        f"Generation complete: generated_tokens={len(generated_ids)}, stop_reason={stop_reason}\n{generated_text}"
    )
    print(generated_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
