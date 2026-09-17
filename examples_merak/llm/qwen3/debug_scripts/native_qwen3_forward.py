"""Run native Qwen3 greedy generation one explicit forward call at a time."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from examples_merak.llm.qwen3.debug_scripts.qwen3_xh2a_forward import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_DIR,
    _build_xh_model,
    _generate_hf_step_by_step,
    _load_merak_config,
    _make_step_reporter,
    _normalize_token_ids,
    _read_prompt,
    _torch_dtype,
    _validate_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help="Merak Python model config used to load/dequantize the checkpoint.",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Optional override for model.hf_model in the config.",
    )
    parser.add_argument("--prompt", default="1+1等于几", help="Prompt text or UTF-8 text file.")
    parser.add_argument("--device", default="cuda:0", help="Execution device. Default: cuda:0.")
    parser.add_argument("--dtype", default="fp16", choices=("fp16", "bf16", "fp32"))
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--think", action="store_true", help="Enable Qwen3 thinking mode.")
    parser.add_argument("--work-dir", default="work_dirs/native_qwen3_forward")
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

    xhquant_init(str(work_dir / "native_qwen3_forward.log"), args.debug)
    set_random_seed(args.seed)
    logger = get_xhquant_logger()

    cfg, model_cfg = _load_merak_config(str(config_path), args.model_dir, args.seed)
    model_dir = Path(model_cfg.hf_model)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_dir}")
    logger.info(f"Config:\n{cfg.pretty_text}")
    logger.info(f"Model Config:\n{model_cfg.to_json_string()}")

    xh_model = _build_xh_model(model_cfg, work_dir)
    processor = xh_model.get_tf_processor()
    tokenizer = processor.tokenizer
    prompt = _read_prompt(args.prompt)
    model_inputs = processor.apply_chat_template(
        [{"role": "user", "content": prompt}],
        enable_think=args.think,
    )
    input_ids = model_inputs.input_ids.to(dtype=torch.long, device="cpu")
    total_length = int(input_ids.shape[1]) + args.max_new_tokens
    if total_length > int(model_cfg.context_max_length):
        raise ValueError(
            f"prompt plus generated tokens ({total_length}) exceeds model context "
            f"limit {model_cfg.context_max_length}"
        )

    native_model = xh_model.get_native_model()
    native_model.config._attn_implementation = "eager"
    native_model.to(device=device, dtype=dtype)
    native_model.eval()
    eos_token_ids = _normalize_token_ids(getattr(tokenizer, "eos_token_id", None))
    eos_token_ids.update(_normalize_token_ids(getattr(native_model.generation_config, "eos_token_id", None)))
    logger.info(
        f"Running native step-by-step generation: input_tokens={input_ids.shape[1]}, "
        f"max_new_tokens={args.max_new_tokens}, device={device}, dtype={dtype}"
    )
    with torch.no_grad():
        generated_ids = _generate_hf_step_by_step(
            native_model,
            input_ids,
            device=device,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos_token_ids,
            step_callback=_make_step_reporter(tokenizer, logger),
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
