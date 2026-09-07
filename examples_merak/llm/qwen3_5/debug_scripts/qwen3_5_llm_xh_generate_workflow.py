import argparse
from pathlib import Path

import torch
from PIL import Image
from transformers import TextStreamer

from xhmodel_merak.xh_llm import LLMInferenceContextManager, LLMModelState
from xhquant.api import get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Qwen3.5 generation through the Merak workflow.")
    parser.add_argument("--model-dir", required=True, help="HF model directory.")
    parser.add_argument("--config-path", required=True, help="Qwen3.5 workflow YAML path.")
    parser.add_argument(
        "--eval-type",
        default="quanted_aligned",
        choices=LLMModelState.get_all_values(),
        help="Model state used for generation.",
    )
    parser.add_argument("--prompt", default="你是谁？", help="Prompt text or a file containing the prompt.")
    parser.add_argument("--image-path", default=None, help="Optional image path for visual Qwen3.5 models.")
    parser.add_argument("--think", action="store_true", help="Enable thinking mode.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--auto-offload", action="store_true", help="Enable automatic device offload.")
    parser.add_argument("--enable-prefill-chunk", action="store_true", help="Enable prefill chunking.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Maximum number of generated tokens.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompt = args.prompt
    if Path(prompt).is_file():
        prompt = Path(prompt).read_text()

    image = None
    if args.image_path:
        image = Image.open(args.image_path).convert("RGB")

    cfg_name = Path(args.config_path).stem
    if args.debug:
        cfg_name += "_debug"
    work_dir = Path("work_dirs") / cfg_name
    work_dir.mkdir(parents=True, exist_ok=True)

    xhquant_init(str(work_dir / f"generate_{args.eval_type}.log"), args.debug)
    set_random_seed(1024)
    logger = get_xhquant_logger()

    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    workflow = AutoLLMWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
        seed=1024,
        debug=args.debug,
    )
    logger.info(f"Config chip_arch: {workflow.workflow_config.export['model']['chip_arch']}")
    xh_model = workflow.get_model(work_dir=str(work_dir))

    if args.auto_offload:
        xh_model.config.enable_auto_offload = True
    if args.enable_prefill_chunk:
        xh_model.config.enable_prefill_chunk = True
        xh_model.config.use_cache = True

    eval_type = LLMModelState.from_string(args.eval_type)
    xh_model.set_state(eval_type)

    message_content = []
    if image is not None:
        message_content.append({"type": "image", "image": image})
    message_content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": message_content}]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    logger.info(f"Using device: {device}, dtype: {dtype}")
    processor = xh_model.get_tf_processor()
    tokenizer = processor.tokenizer
    model_inputs = processor.apply_chat_template(messages, enable_thinking=args.think).to(device=device)
    streamer = TextStreamer(tokenizer=tokenizer)

    contexts = [
        TimeProfiler("generate", logger),
        MemoryTracker(device=device, name="generate", logger=logger),
        LLMInferenceContextManager(xh_model),
        torch.no_grad(),
    ]
    with ContextManagers(contexts):
        xh_model.to(device=device, dtype=dtype)
        xh_model.eval()
        generated_ids = xh_model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    output_text = output_text[0] if isinstance(output_text, list) else output_text
    logger.info(f"{'-' * 20} eval type: {eval_type} {'-' * 20}")
    logger.info(f"think: {args.think}")
    logger.info(f"{'-' * 20} Output {'-' * 20}")
    logger.info(output_text)


if __name__ == "__main__":
    main()
