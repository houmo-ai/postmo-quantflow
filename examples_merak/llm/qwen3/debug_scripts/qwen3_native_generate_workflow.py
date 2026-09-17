import argparse
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import TextStreamer

from examples_merak.llm.qwen3.debug_scripts.qwen3_xh2a_forward import _read_prompt
from xhquant.core.context import get_current_context
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.qwen3 import XHQwen3Model
from xhquant.api import get_xhquant_logger, set_random_seed, xhquant_init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run native HF generate for a Qwen3 Merak workflow model.",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="HF model directory.",
    )
    parser.add_argument(
        "--config-path",
        required=True,
        help="Workflow YAML path, choose one in configs_merak/workflows/xh2a/llm_models/qwen3",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for native generate when --auto-offload is not set. Default: cuda",
    )
    parser.add_argument(
        "--prompt",
        default="请用中文介绍一下你自己。",
        help="Prompt text, UTF-8 text file, or .json/.jsonl file with a string 'query' field.",
    )
    parser.add_argument("--think", action="store_true", help="Enable think mode.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument(
        "--auto-offload",
        action="store_true",
        help="Load the native HF model with device_map=auto.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg_name = Path(args.config_path).stem
    if args.debug:
        cfg_name += "_debug"
    work_dir = Path("work_dirs") / cfg_name
    work_dir.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(work_dir / "generate.log"), args.debug)
    set_random_seed(1024)
    logger = get_xhquant_logger()
    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    get_current_context().kvcache.fast_mode = True
    workflow = AutoLLMWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
        seed=1024,
        debug=args.debug,
    )
    logger.info(f"Config chip_arch: {workflow.workflow_config.export['model']['chip_arch']}")
    xh_model: XHQwen3Model = workflow.get_model(work_dir=str(work_dir))
    if args.auto_offload:
        xh_model.config.enable_auto_offload = True

    native_model = xh_model.get_native_model()
    native_model.config._attn_implementation = "eager"
    native_model.eval()

    prompt = _read_prompt(args.prompt)
    messages = [{"role": "user", "content": prompt}]

    device = args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu"
    dtype = torch.float16
    logger.info(f"Using device: {device}, dtype: {dtype}, auto_offload: {args.auto_offload}")
    processor = xh_model.get_tf_processor()
    tokenizer = processor.tokenizer
    model_inputs = processor.apply_chat_template(messages, enable_think=args.think)
    if args.auto_offload:
        native_device = native_model.device
    else:
        native_model.to(device=device, dtype=dtype)
        native_device = device
    model_inputs = model_inputs.to(native_device)
    streamer = TextStreamer(tokenizer=tokenizer)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    contexts = [
        TimeProfiler("generate", logger),
        MemoryTracker(device=device, name="generate", logger=logger),
        torch.no_grad(),
    ]
    with ContextManagers(contexts):
        generated_ids = native_model.generate(
            **model_inputs,
            max_new_tokens=1024,
            streamer=streamer,
            do_sample=False,
            pad_token_id=pad_token_id,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    try:
        think_end_token_id = tokenizer.encode("</think>")[-1]
        index = len(output_ids) - output_ids[::-1].index(think_end_token_id)
    except (ValueError, IndexError):
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    logger.info(f"{'-' * 20} native generate {'-' * 20}")
    logger.info(f"think: {args.think}")
    if thinking_content:
        logger.info(f"{'-' * 20} thinking content {'-' * 20}")
        logger.info(thinking_content)
    logger.info(f"{'-' * 20} content {'-' * 20}")
    logger.info(content)


if __name__ == "__main__":
    main()
