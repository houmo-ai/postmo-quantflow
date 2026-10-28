"""Run Xing4.0 generation through Merak QuantGraph states (e.g. quanted_fast).

This exercises the xhquant QuantGraph path:
`wrap -> frontend -> convert_to_quant_graph -> ptq_quantize(FAST)`.
It is NOT HMONNX / `to_fast()`.

Typical usage with an existing GPTQ checkpoint:

```bash
cd /path/to/xh2modelzoo
source ../gptqmodel/env.sh
export CUDA_VISIBLE_DEVICES=5
export XHQUANT_MOE_LOW_M_TRITON=0
python examples_merak/llm/xing4_0/xing4_0_xh_generate_workflow.py \
  --model-dir work_dirs/xing4_0_29b_a4b_quant \
  --config-path configs_merak/workflows/xh2a/llm_models/xing4_0/29b_a4b/xing4_0_29b_a4b_xh2a_w4a8.yaml \
  --eval-type quanted_fast \
  --think \
  --max-new-tokens 64
```
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import TextStreamer

from xhmodel_merak.xh_llm import LLMInferenceContextManager, LLMModelState
from xhquant.api import get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.core.context import get_current_context
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler

if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.xing4_0 import XHXing4_0Model


DEFAULT_CONFIG_PATH = (
    "configs_merak/workflows/xh2a/llm_models/xing4_0/29b_a4b/xing4_0_29b_a4b_xh2a_w4a8.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Xing4.0 Merak QuantGraph generation (quanted_fast / fronted / wrap).",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="HF model directory. Pass the GPTQ checkpoint to exercise the post-quant QuantGraph path.",
    )
    parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help="Workflow YAML path under configs_merak/workflows/xh2a/llm_models/xing4_0.",
    )
    parser.add_argument(
        "--eval-type",
        default="quanted_fast",
        choices=LLMModelState.get_all_values(),
        help="Model state used for generation. Default: quanted_fast (QuantGraph FAST).",
    )
    parser.add_argument(
        "--prompt",
        default="请用中文简单介绍 Xing4.0，并给出 1+1 的结果。",
        help="Prompt text or a UTF-8 text file path.",
    )
    parser.add_argument("--think", action="store_true", help="Enable thinking mode in chat template.")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging.")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum generated tokens.")
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


def _read_prompt(prompt: str) -> str:
    path = Path(prompt)
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return prompt


def main() -> None:
    args = parse_args()
    # Triton 3.4 rejects low-M MoE direct-route kernels; use grouped GEMM.
    os.environ.setdefault("XHQUANT_MOE_LOW_M_TRITON", "0")

    cfg_name = Path(args.config_path).stem
    if args.debug:
        cfg_name += "_debug"
    work_dir = Path("work_dirs") / f"{cfg_name}_quantgraph_generate"
    work_dir.mkdir(parents=True, exist_ok=True)

    xhquant_init(str(work_dir / f"generate_{args.eval_type}.log"), args.debug)
    set_random_seed(1024)
    logger = get_xhquant_logger()
    get_current_context().kvcache.fast_mode = True

    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    workflow = AutoLLMWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
        seed=1024,
        debug=args.debug,
    )
    logger.info(f"Config chip_arch: {workflow.workflow_config.export['model']['chip_arch']}")
    # get_model() with skipped quant uses --model-dir as hf_model. Passing the
    # GPTQ checkpoint here loads/dequantizes it, then set_state(quanted_fast)
    # builds the QuantGraph.
    xh_model: "XHXing4_0Model" = workflow.get_model(work_dir=str(work_dir))

    if args.auto_offload:
        xh_model.config.enable_auto_offload = True
    if args.enable_prefill_chunk:
        xh_model.config.enable_prefill_chunk = True
        xh_model.config.use_cache = True

    eval_type = LLMModelState.from_string(args.eval_type)
    logger.info(f"Converting Xing4.0 to state={eval_type.value}")
    xh_model.set_state(eval_type)

    prompt = _read_prompt(args.prompt)
    messages = [{"role": "user", "content": prompt}]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    logger.info(f"Using device: {device}, dtype: {dtype}")

    # Xing4 remote-code tokenizer requires trust_remote_code; avoid interactive
    # HF prompts that break under nohup/tee (Bad file descriptor).
    tokenizer = xh_model.get_tokenizer(trust_remote_code=True)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=args.think,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        eos_token_id = tokenizer.eos_token_id
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0]
        if eos_token_id is None:
            raise ValueError("Xing4.0 tokenizer has no pad or EOS token id")
        pad_token_id = int(eos_token_id)

    streamer = TextStreamer(tokenizer=tokenizer)
    xh_model.to(device=device, dtype=dtype)
    xh_model.eval()

    contexts = [
        TimeProfiler("xing4_0_quantgraph_generate", logger),
        MemoryTracker(device=device, name="xing4_0_quantgraph_generate", logger=logger),
        LLMInferenceContextManager(xh_model),
        torch.no_grad(),
    ]
    with ContextManagers(contexts):
        generated_ids = xh_model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer,
            do_sample=False,
            pad_token_id=pad_token_id,
        )

    input_len = model_inputs["input_ids"].shape[-1]
    output_ids = generated_ids[0][input_len:].tolist()
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
    print(
        f"QUANTGRAPH_OK state={eval_type.value} tokens={len(output_ids)} "
        f"content={content!r}"
    )


if __name__ == "__main__":
    main()
