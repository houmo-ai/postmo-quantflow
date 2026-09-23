import argparse
from pathlib import Path

import torch
from transformers import TextStreamer

from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
from xhquant.api import get_xhquant_logger, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


DEFAULT_CONFIG = (
    "/data01/home/xuchen/xh2/xh2modelzoo/work_dirs/minicpm5_export/"
    """hmquant_xh2_minicpm5_2_6b_w8a8_256_2k_mpe128k_20260909/golden_meta_info.json"""
)


def main(args: argparse.Namespace) -> None:
    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"HMONNX metadata file not found: {config_path}")

    xhquant_init(None, args.debug)
    logger = get_xhquant_logger()
    hmonnx_model = AutoLLMHONNXModel.from_pretrained(str(config_path))
    actual_class = type(hmonnx_model).__name__
    if actual_class != "XHMiniCPM5HMONNXModel":
        raise TypeError(f"Expected XHMiniCPM5HMONNXModel, got {actual_class}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = hmonnx_model.get_tokenizer()
    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.think,
    )
    model_inputs = tokenizer([text], return_tensors="pt", truncation=True).to(device)

    hmonnx_model.to(device)
    hmonnx_model.enable_auto_offload = args.auto_offload
    if args.fast:
        hmonnx_model.to_fast()
    if args.golden:
        hmonnx_model.enable_golden = True

    streamer = TextStreamer(tokenizer)
    contexts = [
        TimeProfiler("minicpm5_hmonnx_generate", logger),
        MemoryTracker(device=device, name="minicpm5_generate", logger=logger),
        LLMInferenceContextManager(hmonnx_model),
    ]
    with ContextManagers(contexts):
        generated_ids = hmonnx_model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            do_sample=args.do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_ids = generated_ids[0][model_inputs.input_ids.shape[-1] :].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    logger.info(f"{'-' * 20} MiniCPM5 response {'-' * 20}")
    logger.info(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one MiniCPM5 conversation with HMONNX.")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to golden_meta_info.json")
    parser.add_argument("--prompt", type=str, default="请用中文简单介绍一下你自己。")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--think", action="store_true", help="Enable thinking mode in the chat template.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling during generation.")
    parser.add_argument("--fast", action="store_true", help="Run the HMONNX model in fast mode.")
    parser.add_argument("--golden", action="store_true", help="Use golden mode for comparison/debugging.")
    parser.add_argument("--debug", action="store_true", help="Enable xhquant debug logging.")
    parser.add_argument("--auto-offload", action="store_true", help="Enable automatic device offload.")
    main(parser.parse_args())
