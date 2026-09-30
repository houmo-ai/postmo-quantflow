# SPDX-License-Identifier: Apache-2.0
"""Run text, image, or video WeMM embeddings with vLLM Merak."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("ENABLE_HMINFERENCE_V2", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import torch
from PIL import Image
from vllm import LLM
from vllm.config import CompilationConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.multimodal.media import ImageMediaIO, VideoMediaIO


EMBEDDING_TOKEN = "<embedding>"


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


def _tokenize_embedding_prompt(tokenizer: Any, prompt: str) -> dict[str, list[int]]:
    """Build a pretokenized request whose pooling position is explicit."""

    prompt_token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    embedding_token_ids = tokenizer.encode(EMBEDDING_TOKEN, add_special_tokens=False)
    if len(embedding_token_ids) != 1:
        raise ValueError(
            f"Expected {EMBEDDING_TOKEN!r} to encode as one token, "
            f"got {embedding_token_ids!r}"
        )
    if not prompt_token_ids or prompt_token_ids[-1] != embedding_token_ids[0]:
        prompt_token_ids.append(embedding_token_ids[0])
    return {"prompt_token_ids": prompt_token_ids}


def _build_request(args: argparse.Namespace, tokenizer: Any) -> object:
    if args.image is not None:
        modality = "image"
    elif args.video is not None:
        modality = "video"
    else:
        modality = None
    request: dict[str, object] = _tokenize_embedding_prompt(
        tokenizer,
        _render_prompt(tokenizer, args.text, modality),
    )
    if args.image is not None:
        image_path = Path(args.image).expanduser().resolve()
        image = Image.open(image_path).convert("RGB")
        request["multi_modal_data"] = {"image": image}
        return request

    if args.video is not None:
        video_path = Path(args.video).expanduser().resolve()
        video = VideoMediaIO(
            ImageMediaIO(),
            num_frames=args.video_frames,
        ).load_file(video_path)
        request["multi_modal_data"] = {"video": video}
        return request

    return request


def main(args: argparse.Namespace) -> None:
    compilation_config = CompilationConfig()
    compilation_config.cudagraph_mode = CUDAGraphMode.NONE

    llm = None
    try:
        llm = LLM(
            model=str(Path(args.model).expanduser().resolve()),
            runner="auto",
            max_num_seqs=1,
            max_model_len=args.max_model_len,
            enforce_eager=True,
            compilation_config=compilation_config,
            trust_remote_code=True,
            config_format="merak_llm",
            load_format="merak_llm",
            limit_mm_per_prompt={"image": 1, "video": 1},
        )
        request = _build_request(args, llm.get_tokenizer())
        output = llm.embed(cast(Any, request), use_tqdm=False)[0]
        embedding = torch.tensor(output.outputs.embedding, dtype=torch.float32)
        print(f"shape={tuple(embedding.shape)}")
        print(f"dtype={embedding.dtype}")
        print(f"norm={torch.linalg.vector_norm(embedding).item():.8f}")
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
    parser.add_argument("--model", required=True, help="WeMM HMONNX export directory")
    parser.add_argument("--text", default="Represent this input for retrieval.")
    media = parser.add_mutually_exclusive_group()
    media.add_argument("--image", help="Local image file")
    media.add_argument("--video", help="Local video file")
    parser.add_argument("--video-frames", type=int, default=4)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument(
        "--print-values",
        action="store_true",
        help="Print the full embedding vector in addition to shape and norm.",
    )
    main(parser.parse_args())
