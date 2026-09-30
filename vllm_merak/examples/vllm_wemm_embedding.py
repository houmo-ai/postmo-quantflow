# SPDX-License-Identifier: Apache-2.0
"""Run native floating-point WeMM text, image, or video embeddings."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, cast


# Plugin selection must happen before importing vLLM in every process.
os.environ["VLLM_PLUGINS"] = "wemm"
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
from PIL import Image
from vllm import LLM
from vllm.config import CompilationConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.multimodal.media import ImageMediaIO, VideoMediaIO
from vllm_merak.model_executor.model_loader.wemm_embedding_native import (
    prepare_wemm_prompt_token_ids,
)


def _render_prompt(tokenizer: Any, text: str, modality: str | None) -> str:
    content: list[dict[str, str]] = []
    if modality is not None:
        content.append({"type": modality})
    content.append({"type": "text", "text": text})
    return cast(
        str,
        tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        ),
    )


def _build_request(args: argparse.Namespace, tokenizer: Any) -> object:
    modality = "image" if args.image else "video" if args.video else None
    prompt = _render_prompt(tokenizer, args.text, modality)
    request: dict[str, object] = {"prompt_token_ids": prepare_wemm_prompt_token_ids(tokenizer, prompt)}

    if args.image:
        image_path = Path(args.image).expanduser().resolve()
        request["multi_modal_data"] = {"image": Image.open(image_path).convert("RGB")}
    elif args.video:
        video_path = Path(args.video).expanduser().resolve()
        request["multi_modal_data"] = {
            "video": VideoMediaIO(
                ImageMediaIO(),
                num_frames=args.video_frames,
            ).load_file(video_path)
        }
    return request


def main(args: argparse.Namespace) -> None:
    model_path = Path(args.model).expanduser().resolve()
    compilation_config = CompilationConfig()
    compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    llm = None
    try:
        kwargs: dict[str, object] = {
            "model": str(model_path),
            "runner": "pooling",
            "max_num_seqs": args.max_num_seqs,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "enforce_eager": True,
            "compilation_config": compilation_config,
            "trust_remote_code": True,
            "enable_prefix_caching": False,
            "skip_mm_profiling": True,
            "limit_mm_per_prompt": {"image": 1, "video": 1},
        }
        chat_template = model_path / "embedding_chat_template.jinja"
        if chat_template.is_file():
            kwargs["chat_template"] = chat_template

        llm = LLM(**kwargs)
        request = _build_request(args, llm.get_tokenizer())
        output = llm.embed(cast(Any, request), use_tqdm=False)[0]
        embedding = torch.tensor(output.outputs.embedding, dtype=torch.float32)
        print(f"shape={tuple(embedding.shape)}")
        print(f"dtype={embedding.dtype}")
        print(f"norm={torch.linalg.vector_norm(embedding).item():.8f}")
        if args.output:
            output_path = Path(args.output).expanduser().resolve()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(embedding.cpu(), output_path)
            print(f"output={output_path}")
        if args.print_values:
            print(embedding.tolist())
    finally:
        engine_core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "shutdown"):
            engine_core.shutdown()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Official WeMM checkpoint")
    parser.add_argument("--text", default="Represent this input for retrieval.")
    media = parser.add_mutually_exclusive_group()
    media.add_argument("--image", help="Local image file")
    media.add_argument("--video", help="Local video file")
    parser.add_argument("--video-frames", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--output", help="Optional path for the FP32 embedding tensor")
    parser.add_argument("--print-values", action="store_true")
    main(parser.parse_args())
