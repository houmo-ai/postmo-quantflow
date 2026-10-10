import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

# A direct script invocation must use this checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from transformers import TextStreamer

from xhmodel_merak.xh_llm import (
    LLMInferenceContextManager,
    LLMModelState,
)
from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow
from xhquant.api import get_xhquant_logger, set_random_seed, xhquant_init
from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler


if TYPE_CHECKING:
    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5 import XHSpark2_5Model


def load_model(args: argparse.Namespace, work_dir: Path) -> "XHSpark2_5Model":
    workflow = AutoLLMWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config,
        seed=args.seed,
        debug=args.debug,
    )
    workflow.workflow_config.dump(str(work_dir / Path(args.config).name))
    model = workflow.get_model(work_dir=str(work_dir))
    dtype = torch.float16
    model.to(dtype=dtype)
    model.set_state(LLMModelState.from_string(args.eval_type))
    model.to(device=args.device, dtype=dtype)
    model.eval()
    return model


def read_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    if not args.input:
        return [{"id": "smoke-1", "prompt": args.prompt}]

    input_path = Path(args.input)
    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        records = parsed if isinstance(parsed, list) else [parsed]

    if not all(isinstance(record, dict) for record in records):
        raise ValueError(f"{input_path} must contain JSON objects")
    if args.limit is not None:
        records = records[: args.limit]
    for line_number, record in enumerate(records, start=1):
        record.setdefault("id", str(line_number))
    return records


def get_prompt_source(record: dict[str, Any]) -> dict[str, Any]:
    raw_data = record.get("raw_data")
    if isinstance(raw_data, dict):
        return {**raw_data, **record}
    return record


def render_prompt(record: dict[str, Any], tokenizer: Any, enable_thinking: bool = True) -> str:
    source = get_prompt_source(record)
    messages = source.get("messages")
    metadata = source.get("metadata")
    metadata_prompt = metadata.get("prompt") if isinstance(metadata, dict) else None
    if metadata_prompt is not None:
        messages = [{"role": "user", "content": str(metadata_prompt)}]
    if messages is None:
        for field in ("prompt", "query", "text"):
            if source.get(field) is not None:
                messages = [{"role": "user", "content": str(source[field])}]
                break
    if messages is not None:
        if not getattr(tokenizer, "chat_template", None):
            raise ValueError("Input has text or messages, but tokenizer has no chat_template")
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    raise ValueError("Each input record needs prompt, query, text, messages, or raw_data containing one of them")


def make_generation_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "repetition_penalty": args.repetition_penalty,
    }
    if args.do_sample:
        generation_kwargs.update(
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    return generation_kwargs


def collect_ssfp_scale_stats(model) -> list[dict[str, Any]]:
    """Inspect signed GPTQ/SSFP scales without changing the packed payload."""
    records = []
    for name, module in model.get_inference_model().named_modules():
        scale = getattr(module, "scales", None)
        bits = getattr(module, "logical_bits", None)
        if bits is None:
            spec = getattr(module, "w_cfg", {}).get("qspec", {})
            if spec.get("fp_mode") != "ssfp":
                continue
            scale = getattr(module, "scale_or_exp", None)
            bits = spec.get("man_bit")
        if not isinstance(scale, torch.Tensor) or scale.is_meta or not scale.numel():
            continue
        scale = scale.detach()
        records.append({
            "module": name,
            "bits": bits,
            "count": scale.numel(),
            "negative": int((scale < 0).sum().item()),
            "minimum": float(scale.min().item()),
            "maximum": float(scale.max().item()),
        })
    return records


def main(args):
    cfg_name = Path(args.config).stem
    eval_type = args.eval_type

    work_dir = Path(args.work_dir) if args.work_dir else Path("work_dirs") / cfg_name / eval_type
    work_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(work_dir / f"generate_{eval_type}.log")

    xhquant_init(log_file, args.debug)
    seed = args.seed
    set_random_seed(seed)
    logger = get_xhquant_logger()
    dtype = torch.float16
    device = args.device
    logger.info(f"Using device: {device}, dtype: {dtype}")
    logger.info(f"Workflow config: {Path(args.config).resolve()}")
    logger.info(f"Inference checkpoint: {Path(args.model_dir).resolve()}, eval_type={args.eval_type}")

    xh_model = load_model(args, work_dir)
    eval_type = LLMModelState.from_string(args.eval_type)
    logger.info(f"Workflow model loaded: {type(xh_model).__name__}, state={xh_model._state}")
    if args.diagnose_ssfp_scales:
        scale_stats = collect_ssfp_scale_stats(xh_model)
        negative = sum(record["negative"] for record in scale_stats)
        total = sum(record["count"] for record in scale_stats)
        logger.info(f"SSFP scale diagnostics: {negative}/{total} negative scales in {len(scale_stats)} projections")
        for record in scale_stats:
            logger.info(f"SSFP scales: {json.dumps(record, ensure_ascii=False)}")
        if negative and eval_type == LLMModelState.QUANTED_ALIGNED:
            logger.warning("SSFP kernels require nonnegative scales. Check that GPTQPackedWeight.to_hm_ssfp() folds negative scale signs into qweight; see the Spark README diagnosis.")

    enable_think = args.think

    processor = xh_model.get_tf_processor()
    tokenizer = processor.tokenizer

    records = read_records(args)
    generation_kwargs = make_generation_kwargs(args)
    logger.info(f"Generation parameters: {generation_kwargs}, seed={seed}")
    for record in records:
        prompt = render_prompt(record, tokenizer, enable_thinking=enable_think)
        logger.info(f"{prompt}")

        model_inputs = tokenizer(prompt, return_tensors="pt", return_token_type_ids=False)
        model_inputs.pop("token_type_ids", None)
        model_inputs = model_inputs.to(device)
        streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True)

        logger.info(f"{'-' * 20} stream output {'-' * 20}")
        contexts = [
            TimeProfiler("generate", logger),
            MemoryTracker(device=device, name="generate", logger=logger),
            # KV cache tensors must be allocated on the same device as the
            # wrapped graph.  Without ``devices``, the context manager creates
            # them on CPU even though the model is on CUDA.
            LLMInferenceContextManager(xh_model, devices=[device]),
            torch.no_grad(),
        ]
        with ContextManagers(contexts):
            generated_ids = xh_model.generate(
                **model_inputs,
                **generation_kwargs,
                streamer=streamer,
                pad_token_id=tokenizer.eos_token_id,
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        logger.info(f"Generated {len(output_ids)} tokens: {output_ids}")
        logger.info(f"Raw output: {tokenizer.decode(output_ids, skip_special_tokens=False)}")

        # parsing thinking content
        try:
            # 动态获取 </think> token 的 ID
            think_end_token_id = tokenizer.encode("</think>")[-1]  # 获取 </think> 的 token ID
            # rindex finding think_end_token_id (</think>)
            index = len(output_ids) - output_ids[::-1].index(think_end_token_id)
        except (ValueError, IndexError):
            index = 0

        thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        logger.info(f"{'-' * 20} eval type: {eval_type} {'-' * 20}")
        logger.info(f"think: {enable_think}")
        if len(thinking_content) > 0:
            logger.info(f"{'-' * 20} thinking content {'-' * 20}")
            logger.info(f"{thinking_content}")
        logger.info(f"{'-' * 20} content {'-' * 20}")
        logger.info(f"{content}")


def parse_args() -> argparse.Namespace:
    eval_types = [state.value for state in (
        LLMModelState.WRAP, LLMModelState.FRONTED,
        LLMModelState.QUANTED_DISABLE, LLMModelState.QUANTED_FAST, LLMModelState.QUANTED_ALIGNED,
    )]
    parser = argparse.ArgumentParser(description="Run Spark X2.5 generation through a Merak workflow.")
    parser.add_argument("--model-dir", required=True, help="Native HF or already quantized checkpoint directory.")
    parser.add_argument(
        "--config", "--config-path",
        type=str,
        required=True,
        help="Spark X2.5 workflow YAML.",
    )
    parser.add_argument("--eval-type", type=str, default="quanted_aligned", choices=eval_types)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--work-dir", help="Directory for inference logs and the workflow config copy.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--diagnose-ssfp-scales", action="store_true",
        help="Log signed packed-weight scale statistics for investigating ALIGNED output differences.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input JSONL file. Each line may contain prompt, query, messages, or raw_data.",
    )

    parser.add_argument("--prompt", type=str, default="用中文简单介绍自己。")
    parser.add_argument(
        "--think",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask the chat template to append the model's thinking prefix.",
    )

    parser.add_argument("--limit", type=int, help="Maximum number of input records to process.")
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Maximum number of tokens to generate.")
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable sampling during generation (disabled by default for deterministic greedy decoding).",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1024, help="Random seed used by model generation.")
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


if __name__ == "__main__":
    main(parse_args())
