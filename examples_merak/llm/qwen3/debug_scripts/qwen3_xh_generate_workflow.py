import argparse
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import TextStreamer

from xhmodel_merak.xh_llm import (
    LLMInferenceContextManager,
    LLMModelState,
)
from xhquant.core.context import get_current_context
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.qwen3 import XHQwen3Model
from examples_merak.llm.qwen3.debug_scripts.qwen3_xh2a_forward import _read_prompt
from xhquant.api import get_xhquant_logger, set_random_seed, xhquant_init


def _remove_output_dir_if_needed(output_dir: str, force: bool) -> None:
    path = Path(output_dir)
    if force and path.exists():
        shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Qwen3 Merak quant/export workflow.",
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
        "--quant-output-dir",
        default="work_dirs/qwen3_quant",
        help="Quantization output directory. Default: work_dirs/qwen3_quant",
    )
    parser.add_argument(
        "--export-output-dir",
        default="work_dirs/qwen3_export",
        help="Export output directory. Default: work_dirs/qwen3_export",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for quant/export/golden/quick test. Default: cuda",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing export output directories before running.",
    )
    parser.add_argument(
        "--dump-golden",
        action="store_true",
        help="Dump golden data after export.",
    )
    parser.add_argument(
        "--export-from-quanted-model",
        action="store_true",
        help="if --model-dir is a quanted model, set this param to True",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=None,
        help="quantization bits, set this param to override config.yaml",
    )
    parser.add_argument(
        "--eval-type",
        default="quanted_aligned",
        choices=LLMModelState.get_all_values(),
        help="Model state used for generation. Default: wrap",
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
        help="Enable automatic device offload for generation.",
    )
    parser.add_argument(
        "--enable-prefill-chunk",
        action="store_true",
        help="Enable prefill chunking for generation.",
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
    get_current_context().kvcache.fast_mode = True
    # 初始化工作流
    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

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
    if args.enable_prefill_chunk:
        xh_model.config.enable_prefill_chunk = True
        xh_model.config.use_cache = True

    eval_type = LLMModelState.from_string(args.eval_type)
    xh_model.set_state(eval_type)

    prompt = _read_prompt(args.prompt)
    messages = [{"role": "user", "content": prompt}]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    logger.info(f"Using device: {device}, dtype: {dtype}")
    processor = xh_model.get_tf_processor()
    tokenizer = processor.tokenizer
    model_inputs = processor.apply_chat_template(messages, enable_think=args.think).to(device)
    streamer = TextStreamer(tokenizer=tokenizer)
    xh_model.to(device=device, dtype=dtype)
    xh_model.eval()

    contexts = [
        TimeProfiler("generate", logger),
        MemoryTracker(device=device, name="generate", logger=logger),
        LLMInferenceContextManager(xh_model),
        torch.no_grad(),
    ]
    with ContextManagers(contexts):
        generated_ids = xh_model.generate(
            **model_inputs,
            max_new_tokens=1024,
            streamer=streamer,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
    try:
        think_end_token_id = tokenizer.encode("</think>")[-1]
        index = len(output_ids) - output_ids[::-1].index(think_end_token_id)
    except (ValueError, IndexError):
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    logger.info(f"{'-' * 20} eval type: {eval_type} {'-' * 20}")
    logger.info(f"think: {args.think}")
    if thinking_content:
        logger.info(f"{'-' * 20} thinking content {'-' * 20}")
        logger.info(thinking_content)
    logger.info(f"{'-' * 20} content {'-' * 20}")
    logger.info(content)


if __name__ == "__main__":
    main()
