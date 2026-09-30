import os
from pathlib import Path
from typing import Any, cast

import torch
from vllm import LLM, EngineArgs, SamplingParams
from vllm.config import CompilationConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.utils.argparse_utils import FlexibleArgumentParser


def _render_messages_to_prompt(tokenizer, messages) -> list[str]:
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return [rendered]


def _resolve_image_url(image: str) -> str:
    if image.startswith(("http://", "https://", "file://", "data:image/")):
        return image

    image_path = Path(image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    return image_path.as_uri()


def _resolve_allowed_local_media_path(
    image: str | None,
    configured_path: str | None,
) -> str | None:
    if configured_path:
        return str(Path(configured_path).expanduser().resolve())

    if image is None or image.startswith(("http://", "https://", "file://", "data:image/")):
        return None

    image_path = Path(image).expanduser().resolve()
    if not image_path.is_file():
        return None

    return str(image_path.parent)


def _build_user_messages(prompt: str, image: str | None = None) -> list[dict[str, object]]:
    if image is None:
        content: str | list[dict[str, object]] = prompt
    else:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _resolve_image_url(image)}},
        ]

    return [
        {
            "role": "user",
            "content": content,
        }
    ]


def main(args):
    model_path = os.path.normpath(args.model)
    allowed_local_media_path = _resolve_allowed_local_media_path(
        args.image,
        getattr(args, "allowed_local_media_path", None),
    )
    compilation_config = CompilationConfig()
    compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # os.environ["VLLM_USE_V1"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_LOGGING_LEVEL"] = "INFO"
    os.environ["VLLM_LOGGING_PREFIX"] = "[vLLM Merak]"
    os.environ["VLLM_CPU_KVCACHE_SPACE"] = "16"

    llm = None
    try:
        llm_kwargs = dict(
            model=model_path,
            max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs,
            enforce_eager=True,
            compilation_config=compilation_config,
            trust_remote_code=True,
            config_format="merak_llm",
            load_format="merak_llm",
        )
        if allowed_local_media_path is not None:
            llm_kwargs["allowed_local_media_path"] = allowed_local_media_path

        llm = LLM(**cast(Any, llm_kwargs))

        messages = _build_user_messages(args.prompt, args.image)

        params = SamplingParams(temperature=0.7, max_tokens=128)
        if args.image is None:
            rendered_prompt = _render_messages_to_prompt(
                llm.get_tokenizer(),
                messages,
            )
            output = llm.generate(rendered_prompt, sampling_params=params)
        else:
            output = llm.chat(cast(Any, messages), sampling_params=params)
        print(output)
    finally:
        engine_core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "shutdown"):
            engine_core.shutdown()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    EngineArgs.add_cli_args(parser)
    parser.set_defaults(max_num_seqs=1)
    parser.add_argument("--prompt", default="北京市是", help="Input prompt text.")
    parser.add_argument("--image", default=None, help="Optional image path or URL for multimodal generation.")

    args = parser.parse_args()
    main(args)
