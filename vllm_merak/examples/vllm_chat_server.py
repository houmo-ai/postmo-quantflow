import json
import os
from pathlib import Path
from typing import Any, cast

import uvloop

import vllm_merak.config  # noqa: F401
from vllm_merak.audit_logging import (
    AUDIT_LOG_DIR_ENV,
    AUDIT_LOG_MIDDLEWARE,
    AUDIT_LOG_MODE_ENV,
)
from vllm_merak.constants import MERAK_AUTO_OFFLOAD, MERAK_CUDA_GRAPH, MERAK_USE_V2

os.environ["VLLM_USE_V1"] = "1"

from vllm.entrypoints.openai.api_server import run_server
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.utils.argparse_utils import FlexibleArgumentParser


def _allow_merak_config_format(parser: FlexibleArgumentParser) -> None:
    for action in parser._actions:
        if action.dest != "config_format" or action.choices is None:
            continue
        choices = list(action.choices)
        if "merak_llm" not in choices:
            choices.append("merak_llm")
            action.choices = choices


def build_arg_parser() -> FlexibleArgumentParser:
    parser = make_arg_parser(FlexibleArgumentParser())
    _allow_merak_config_format(parser)
    parser.add_argument(
        "--use-v2",
        action="store_true",
        help="Enable Merak HMONNX runtime use_v2 mode.",
    )
    parser.add_argument(
        "--auto-offload",
        action="store_true",
        help="Enable Merak HMONNX runtime auto_offload mode.",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Enable Merak HMONNX runtime CUDA graph mode.",
    )
    parser.add_argument(
        "--audit-log-mode",
        choices=("off", "aggregate", "sse", "both"),
        default="off",
        help=(
            "Write request/response audit JSONL independently of vLLM's log "
            "level. 'aggregate' stores one reconstructed record per request; "
            "'sse' stores lossless ASGI response chunks; 'both' stores both."
        ),
    )
    parser.add_argument(
        "--audit-log-dir",
        default="logs/audit",
        help="Directory for Merak audit JSONL files.",
    )
    return parser


def _is_gemma4_merak_artifact(model: object) -> bool:
    """Identify local Gemma4 artifacts before vLLM resolves the config."""

    if not isinstance(model, str):
        return False
    model_dir = Path(model)
    if not model_dir.is_dir():
        return False
    try:
        merak_config = json.loads((model_dir / "merak_config.json").read_text(encoding="utf-8"))
        meta_path = Path(merak_config["xh_model"]["meta_info"])
        if not meta_path.is_absolute():
            meta_path = model_dir / meta_path
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        model_type = meta.get("model_config", {}).get("model_type", "")
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError):
        return False
    return str(model_type).lower().startswith("gemma4")


def apply_merak_runtime_flags(args: Any, parser: FlexibleArgumentParser) -> None:
    if getattr(args, "model_tag", None) is not None:
        args.model = args.model_tag
    if not getattr(args, "reasoning_parser", None) and not _is_gemma4_merak_artifact(args.model):
        # Qwen3.5 emits thinking text by default, but clients such as Cherry
        # Studio only render it as a thinking block when the OpenAI response
        # carries separated reasoning content.  Enable vLLM's Qwen3 parser by
        # default for this Merak chat-server example; callers can still override
        # it explicitly with --reasoning-parser.
        args.reasoning_parser = "qwen3"
    if args.auto_offload and args.cuda_graph and not args.use_v2:
        parser.error("--auto-offload and --cuda-graph require --use-v2 when enabled together.")
    if args.use_v2:
        os.environ[MERAK_USE_V2] = "1"
    if args.auto_offload:
        os.environ[MERAK_AUTO_OFFLOAD] = "1"
    if args.cuda_graph:
        os.environ[MERAK_CUDA_GRAPH] = "1"

    audit_log_mode = getattr(args, "audit_log_mode", "off")
    if audit_log_mode == "off":
        os.environ.pop(AUDIT_LOG_MODE_ENV, None)
        os.environ.pop(AUDIT_LOG_DIR_ENV, None)
        return

    os.environ[AUDIT_LOG_MODE_ENV] = audit_log_mode
    os.environ[AUDIT_LOG_DIR_ENV] = str(getattr(args, "audit_log_dir", "logs/audit"))
    middleware = list(getattr(args, "middleware", None) or [])
    if AUDIT_LOG_MIDDLEWARE not in middleware:
        middleware.append(AUDIT_LOG_MIDDLEWARE)
    args.middleware = middleware


def main():
    parser = build_arg_parser()
    args = cast(Any, parser.parse_args())
    apply_merak_runtime_flags(args, parser)
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
