#!/usr/bin/env python3
"""Run the release-gate sampling matrix against a generative HMONNX model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from vllm.config import CompilationConfig
from vllm.config.compilation import CUDAGraphMode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=4)
    return parser


def main() -> int:
    args = _parser().parse_args()
    model = args.model.expanduser().resolve()
    meta_path = model / "golden_meta_info.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing golden_meta_info.json: {model}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    has_visual = isinstance(meta.get("visual_config"), dict)
    smoke_image = Path(__file__).resolve().parents[1] / "data/images/qwen2_vl_demo.jpeg"
    if has_visual and not smoke_image.is_file():
        raise FileNotFoundError(f"missing visual smoke image: {smoke_image}")

    compilation = CompilationConfig()
    compilation.cudagraph_mode = CUDAGraphMode.NONE
    llm = None
    results: list[dict[str, object]] = []
    try:
        llm = LLM(
            model=str(model),
            max_model_len=args.max_model_len,
            max_num_seqs=1,
            enforce_eager=True,
            compilation_config=compilation,
            trust_remote_code=True,
            config_format="merak_llm",
            load_format="merak_llm",
            allowed_local_media_path=str(smoke_image.parent),
        )
        cases = (
            (
                "greedy",
                SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
            ),
            (
                "temperature_topk_topp",
                SamplingParams(
                    temperature=0.7,
                    top_k=20,
                    top_p=0.9,
                    seed=17,
                    max_tokens=args.max_tokens,
                ),
            ),
            (
                "penalties",
                SamplingParams(
                    temperature=0.0,
                    repetition_penalty=1.1,
                    presence_penalty=0.1,
                    frequency_penalty=0.1,
                    max_tokens=args.max_tokens,
                ),
            ),
        )
        for index, (name, params) in enumerate(cases):
            prompt = f"Reply with one short word for release smoke case {index}."
            output = llm.generate(prompt, params, use_tqdm=False)[0]
            token_ids = list(output.outputs[0].token_ids)
            if not token_ids:
                raise RuntimeError(f"{name} returned no generated tokens")
            results.append(
                {
                    "case": name,
                    "token_count": len(token_ids),
                    "finished": output.finished,
                }
            )
        if has_visual:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image briefly."},
                        {
                            "type": "image_url",
                            "image_url": {"url": smoke_image.as_uri()},
                        },
                    ],
                }
            ]
            output = llm.chat(
                messages,
                sampling_params=SamplingParams(
                    temperature=0.0,
                    max_tokens=args.max_tokens,
                ),
                use_tqdm=False,
            )[0]
            token_ids = list(output.outputs[0].token_ids)
            if not token_ids:
                raise RuntimeError("image_greedy returned no generated tokens")
            results.append(
                {
                    "case": "image_greedy",
                    "token_count": len(token_ids),
                    "finished": output.finished,
                }
            )
    finally:
        engine_core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        if engine_core is not None and hasattr(engine_core, "shutdown"):
            engine_core.shutdown()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    print(json.dumps({"model": str(model), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
