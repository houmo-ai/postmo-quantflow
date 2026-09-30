import os
import time

import torch
from vllm import LLM, EngineArgs, SamplingParams
from vllm.config import CompilationConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.utils.argparse_utils import FlexibleArgumentParser

import vllm_merak.config  # noqa: F401
from vllm_merak.constants import MERAK_AUTO_OFFLOAD, MERAK_CUDA_GRAPH, MERAK_USE_V2


def _render_messages_to_prompt(tokenizer, messages) -> list[str]:
    if getattr(tokenizer, "chat_template", None):
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    elif (
        getattr(tokenizer, "bos_token", None) == "<｜begin▁of▁sentence｜>"
        and getattr(tokenizer, "eos_token", None) == "<｜end▁of▁sentence｜>"
    ):
        # DeepSeek-V4 tokenizer assets intentionally omit a Jinja template.
        content = messages[-1]["content"]
        rendered = (
            "<｜begin▁of▁sentence｜><｜User｜>"
            f"{content}<｜Assistant｜></think>"
        )
    else:
        raise ValueError("tokenizer has no supported chat template")
    return [rendered]


def _allow_merak_config_format(parser: FlexibleArgumentParser) -> None:
    for action in parser._actions:
        if action.dest != "config_format" or action.choices is None:
            continue
        choices = list(action.choices)
        if "merak_llm" not in choices:
            choices.append("merak_llm")
            action.choices = choices


# /data01/home/lihui/projects/xhquant_projects/xh2modelzoo/work_dirs/qwen3_5_moe_35b_a3b_instruct_hf_autoround_xh2a_2k/hmquant_xh2_qwen3.5-35b-a3b_w4a8_256_2k_448x448_20260427
# def use_my_npu_platform():
#     import vllm.platforms as platforms

#     if getattr(platforms, "_current_platform", None) is not None:
#         platforms._current_platform = None

#     platforms.resolve_current_platform_cls_qualname = lambda: "vllm_merak.platform.MerakPlatform"


def main(args):
    model_path = os.path.normpath(args.model)
    warmup_tokens = int(args.warmup_tokens or os.environ.get("MERAK_CHAT_WARMUP_TOKENS", "8"))
    measure_tokens = int(args.measure_tokens or os.environ.get("MERAK_CHAT_MEASURE_TOKENS", "8"))
    if measure_tokens < 6:
        raise ValueError("--measure-tokens must be at least 6")
    enable_cuda_profiler = os.environ.get("MERAK_CHAT_CUDA_PROFILER", "0") == "1"
    compilation_config = CompilationConfig()
    compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
    os.environ["VLLM_LOGGING_PREFIX"] = "[vLLM Merak]"

    if args.auto_offload and args.cuda_graph and not args.use_v2:
        raise ValueError("--auto-offload and --cuda-graph require --use-v2 when enabled together.")
    if args.use_v2:
        os.environ[MERAK_USE_V2] = "1"
    if args.auto_offload:
        os.environ[MERAK_AUTO_OFFLOAD] = "1"
    if args.cuda_graph:
        os.environ[MERAK_CUDA_GRAPH] = "1"
    # use_my_npu_platform()

    llm = None
    try:
        llm = LLM(
            model=model_path,
            max_num_seqs=1,
            max_model_len=int(args.max_model_len or 262144),
            enforce_eager=True,
            compilation_config=compilation_config,
            trust_remote_code=True,
            config_format="merak_llm",
            load_format="merak_llm",
            disable_log_stats=False,
        )

        prompt = args.prompt
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        rendered_prompt = _render_messages_to_prompt(
            llm.get_tokenizer(),
            messages,
        )

        # Warm up the exact decode shape before enabling vLLM's progress bar.
        # ``ignore_eos`` guarantees that the warmup crosses the same
        # PageAttention block-table capacity buckets as the measured request,
        # so CUDA Graph capture/recapture time is excluded from the throughput
        # that vLLM prints below. The same LLM instance must be reused so the
        # captured graphs and runtime caches remain warm.
        warmup_params = SamplingParams(
            temperature=0.7,
            max_tokens=warmup_tokens,
            ignore_eos=True,
        )
        print("Warming up Merak prefill/decode CUDA Graphs...", flush=True)
        llm.generate(
            rendered_prompt,
            sampling_params=warmup_params,
            use_tqdm=False,
        )
        print("Warmup complete; measuring steady-state throughput.", flush=True)

        params = SamplingParams(temperature=0.7, max_tokens=measure_tokens, ignore_eos=True)
        if enable_cuda_profiler:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
        try:
            measure_start = time.perf_counter()
            output = llm.generate(
                rendered_prompt,
                sampling_params=params,
                use_tqdm=True,
            )
            measure_seconds = time.perf_counter() - measure_start
        finally:
            if enable_cuda_profiler:
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaProfilerStop()
        request_output = output[0]
        generated_tokens = len(request_output.outputs[0].token_ids)
        metrics = request_output.metrics
        decode_intervals = max(generated_tokens - 1, 0)
        decode_seconds = (
            float(metrics.last_token_ts - metrics.first_token_ts)
            if metrics is not None and decode_intervals
            else 0.0
        )
        print(
            {
                "prompt_tokens": len(request_output.prompt_token_ids),
                "generated_tokens": generated_tokens,
                "first_token_seconds": (
                    round(float(metrics.first_token_latency), 3)
                    if metrics is not None
                    else None
                ),
                "decode_seconds": round(decode_seconds, 3),
                "decode_tokens_per_second": round(
                    decode_intervals / decode_seconds if decode_seconds else 0.0,
                    3,
                ),
                "end_to_end_seconds": round(measure_seconds, 3),
                "end_to_end_tokens_per_second": round(
                    generated_tokens / measure_seconds if measure_seconds else 0.0,
                    3,
                ),
                "text": request_output.outputs[0].text,
            }
        )
    finally:
        engine_core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "shutdown"):
            engine_core.shutdown()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    _allow_merak_config_format(parser)
    parser.add_argument(
        "--use-v2",
        action="store_true",
        help="Enable Merak HMONNX runtime use_v2 mode.",
    )
    parser.add_argument(
        "--auto-offload",
        action="store_true",
        help="Enable Merak HMONNX runtime auto offload mode.",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Enable Merak HMONNX runtime CUDA graph mode.",
    )
    parser.add_argument("--prompt", default="北京市是", help="User prompt text.")
    parser.add_argument("--warmup-tokens", type=int, default=None)
    parser.add_argument("--measure-tokens", type=int, default=None)

    args = parser.parse_args()
    main(args)
