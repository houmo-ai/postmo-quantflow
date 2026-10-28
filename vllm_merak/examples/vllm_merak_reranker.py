# SPDX-License-Identifier: Apache-2.0
"""Run Qwen3-VL-Reranker HMONNX scoring with the vLLM Merak pooling runner."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path


def materialize_external_data(model_dir: Path, metadata: dict) -> None:
    """Keep ONNX 1.22's single-link requirement without changing payloads."""
    import onnx

    from xhmodel_merak.model_artifact.files import confined_file

    graphs = [metadata["prefill_hmonnx"]]
    graphs.extend(gear["hmonnx"] for gear in metadata["visual_config"]["gears"])
    for relative in graphs:
        graph_file = confined_file(model_dir, relative, "hmonnx")
        graph = onnx.load(graph_file, load_external_data=False).graph
        locations = {
            dict((entry.key, entry.value) for entry in tensor.external_data).get("location")
            for tensor in graph.initializer
            if tensor.external_data
        }
        for location in locations:
            if not location:
                raise ValueError("External weight tensor is missing its location")
            relative_payload = (graph_file.parent / location).relative_to(model_dir).as_posix()
            payload = confined_file(model_dir, relative_payload, "external_data")
            if payload.stat().st_nlink <= 1:
                continue
            with payload.open("rb") as handle:
                expected = hashlib.file_digest(handle, "sha256").hexdigest()
            descriptor, temporary = tempfile.mkstemp(prefix=".merak-external-", dir=payload.parent)
            os.close(descriptor)
            try:
                shutil.copyfile(payload, temporary)
                with open(temporary, "rb") as handle:
                    actual = hashlib.file_digest(handle, "sha256").hexdigest()
                if actual != expected:
                    raise ValueError("External weight copy changed its SHA-256")
                os.replace(temporary, payload)
            finally:
                Path(temporary).unlink(missing_ok=True)


def prepare_model(model_dir: Path, *, materialize: bool = False) -> Path:
    from xhmodel_merak.model_integrations.qwen3_vl_reranker.artifact import validate_score_contract

    metadata = json.loads((model_dir / "golden_meta_info.json").read_text())
    validate_score_contract(model_dir, metadata)
    if materialize:
        materialize_external_data(model_dir, metadata)
    expected = {
        "architectures": ["Qwen3VLRerankerMerakForSequenceClassification"],
        "runner_type": "pooling",
        "num_labels": 1,
        "enable_page_attention": False,
        "xh_model": {"model_type": "hmonnx", "meta_info": "golden_meta_info.json"},
    }
    path = model_dir / "merak_config.json"
    if path.exists():
        config = json.loads(path.read_text())
        if any(config.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{path} conflicts with the Reranker pooling contract")
    else:
        path.write_text(json.dumps(expected, indent=2) + "\n")
    return path


def score_item(item):
    from PIL import Image

    from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.processor import normalize_item

    # Accept vLLM's standard content arrays, including video_url, unchanged.
    # Their validation and media loading belong to vLLM's scoring frontend.
    if isinstance(item, Mapping) and "content" in item:
        return dict(item)
    item = normalize_item(item)
    content = []
    # Keep the ModelZoo query/document ordering: images precede text.
    for source in item["image"]:
        picture = Image.open(source) if isinstance(source, str) else source
        buffer = io.BytesIO()
        picture.save(buffer, format="PNG")
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")},
            }
        )
    if item["text"] is not None:
        content.append({"type": "text", "text": item["text"]})
    return {"content": content}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--input-json", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--materialize-external-data",
        action="store_true",
        help="Copy hard-linked external weights for ONNX 1.22; preserve their SHA-256",
    )
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--allowed-local-media-path", type=Path, help="vLLM root allowed for file:// image/video URLs")
    args = parser.parse_args()
    os.environ["VLLM_PLUGINS"] = "merak"
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    print(f"Config: {prepare_model(args.model.resolve(), materialize=args.materialize_external_data)}", flush=True)
    if args.prepare_only:
        return
    from vllm import LLM
    from vllm.config import CompilationConfig
    from vllm.config.compilation import CUDAGraphMode

    from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.processor import DEFAULT_INSTRUCTION

    inputs = (
        json.loads(args.input_json.read_text())
        if args.input_json
        else {
            "query": {"text": "What is the capital of France?"},
            "documents": [{"text": "Paris is the capital of France."}, {"text": "Whales live in the ocean."}],
        }
    )
    llm = None
    try:
        instruction = inputs.get("instruction", DEFAULT_INSTRUCTION)
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        template = Path(__file__).with_name("qwen3_vl_reranker.jinja").read_text()
        # vLLM 0.23's offline score API has no chat_template_kwargs argument.
        # A quoted Jinja string binds the instruction without executing it.
        template = "{% set instruction = " + json.dumps(instruction, ensure_ascii=False) + " %}" + template
        llm = LLM(
            model=str(args.model.resolve()),
            runner="pooling",
            config_format="merak_llm",
            load_format="merak_llm",
            max_model_len=args.max_model_len,
            max_num_seqs=1,
            max_num_batched_tokens=args.max_num_batched_tokens,
            enforce_eager=True,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 8, "video": 2},
            allowed_local_media_path=str(args.allowed_local_media_path.resolve())
            if args.allowed_local_media_path
            else "",
            compilation_config=CompilationConfig(cudagraph_mode=CUDAGraphMode.NONE),
        )
        output = llm.score(
            score_item(inputs["query"]),
            [score_item(item) for item in inputs["documents"]],
            chat_template=template,
            use_tqdm=False,
        )
        scores = [item.outputs.score for item in output]
        print(json.dumps({"scores": scores, "ranking": sorted(range(len(scores)), key=lambda i: -scores[i])}, indent=2))
    finally:
        core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        if core is not None:
            core.shutdown()


if __name__ == "__main__":
    main()
