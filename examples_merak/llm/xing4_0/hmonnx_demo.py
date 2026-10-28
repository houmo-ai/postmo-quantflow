"""Run a short generation with an exported Xing4.0 HMONNX model."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from transformers import TextStreamer

from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
from xhquant.api import get_xhquant_logger, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


DEFAULT_CONFIG = (
    "work_dirs/xing4_0_29b_a4b_export_moe_ml3/"
    "hmquant_xh2_xing4_0_29b_a4b_w4a8_256_2k_mpe256k_20261009/golden_meta_info.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Xing4.0 HMONNX inference.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to golden_meta_info.json")
    parser.add_argument("--prompt", default="请用中文简单介绍 Xing4.0。")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--golden", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Triton 3.4 rejects low-M MoE direct-route kernels; use grouped GEMM.
    os.environ.setdefault("XHQUANT_MOE_LOW_M_TRITON", "0")

    xhquant_init(None, args.debug)
    logger = get_xhquant_logger()
    hmonnx_model = AutoLLMHONNXModel.from_pretrained(args.config)
    assert type(hmonnx_model).__name__ == "XHXing4_0HMONNXModel", type(hmonnx_model).__name__

    prompt = args.prompt
    if Path(prompt).is_file():
        prompt = Path(prompt).read_text(encoding="utf-8")
    tokenizer = hmonnx_model.get_tokenizer()
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(args.device)
    hmonnx_model.to(args.device)
    if args.fast:
        hmonnx_model.to_fast()
    if args.golden:
        hmonnx_model.enable_golden = True

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        eos_token_id = tokenizer.eos_token_id
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0]
        if eos_token_id is None:
            raise ValueError("Xing4.0 tokenizer has no pad or EOS token id")
        pad_token_id = int(eos_token_id)

    max_new_tokens = 2 if args.golden else args.max_new_tokens
    streamer = TextStreamer(tokenizer)
    with ContextManagers(
        [
            TimeProfiler("xing4_0_hmonnx_generate", logger),
            MemoryTracker(device=args.device, name="xing4_0_generate", logger=logger),
            LLMInferenceContextManager(hmonnx_model),
        ]
    ):
        generated_ids = hmonnx_model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            streamer=streamer,
            do_sample=False,
            pad_token_id=pad_token_id,
        )

    output_ids = generated_ids[0, model_inputs.input_ids.shape[1] :]
    content = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    logger.info(f"hmonnx Xing4.0 output: {content}")
    print(f"DEMO_OK tokens={len(output_ids)} content={content!r}")


if __name__ == "__main__":
    main()
