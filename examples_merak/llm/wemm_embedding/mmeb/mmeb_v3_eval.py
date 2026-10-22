"""Run the pinned official MMEB-v3 evaluator with HF, HMONNX, or vLLM.

The official dataset parsers, instructions, candidate sets, and ranking metrics
remain in the upstream evaluator.  Only model loading and encoding are adapted
for HMONNX.  Run this script from the upstream ``mmeb_v3_eval`` directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch


class HMONNXWeMM:
    def __init__(self, meta: str, device: str, image_min_pixels: int, image_max_pixels: int):
        from examples_merak.llm.wemm_embedding.runtime_utils import load_hmonnx_embedder

        self.runtime, self.meta = load_hmonnx_embedder(meta, device)
        self.processor = self.runtime.get_tf_processor()
        self.rep_dim = int(self.runtime.meta_info.embedding_dimension)
        self.device = torch.device(device)
        self.image_min_pixels = int(image_min_pixels)
        self.image_max_pixels = int(image_max_pixels)

    def to(self, device, **_kwargs):
        self.device = torch.device(device)
        self.runtime.to(self.device)
        return self

    def eval(self):
        return self

    def __call__(self, qry=None, tgt=None):
        from src.model.processor import VLM_IMAGE_TOKENS, VLM_VIDEO_TOKENS, WEMM_EMBEDDING

        inputs = qry if qry is not None else tgt
        if inputs is None:
            raise ValueError("Expected query or target inputs")
        if inputs.get("videos") and any(v is not None for v in inputs["videos"]):
            raise ValueError("This HMONNX adapter currently supports image tasks only")
        image_token = VLM_IMAGE_TOKENS[WEMM_EMBEDDING]
        video_token = VLM_VIDEO_TOKENS[WEMM_EMBEDDING]
        rows = []
        for raw_text, images in zip(inputs["text"], inputs["images"], strict=True):
            if video_token in raw_text:
                raise ValueError("Video placeholder in image-only evaluation")
            content = []
            images = [image for image in (images or []) if image is not None]
            if image_token in raw_text and images:
                parts = raw_text.split(image_token)
                image_iter = iter(images[:16])
                for index, part in enumerate(parts):
                    part = part.strip()
                    if part:
                        content.append({"type": "text", "text": part})
                    if index < len(parts) - 1:
                        image = next(image_iter, None)
                        if image is not None:
                            content.append(
                                {
                                    "type": "image",
                                    "image": image,
                                    "min_pixels": self.image_min_pixels,
                                    "max_pixels": self.image_max_pixels,
                                }
                            )
            else:
                for image in images[:16]:
                    content.append(
                        {
                            "type": "image",
                            "image": image,
                            "min_pixels": self.image_min_pixels,
                            "max_pixels": self.image_max_pixels,
                        }
                    )
                text = raw_text.replace(image_token, "").strip()
                if text or not content:
                    content.append({"type": "text", "text": text})
            messages = [{"role": "user", "content": content}]
            rows.append(self.runtime.embed({"messages": messages}, processor=self.processor).to(self.device))
        reps = torch.cat(rows, dim=0)
        return {"qry_reps" if qry is not None else "tgt_reps": reps}


def shard_bounds(length: int, index: int, count: int) -> tuple[int, int]:
    """Return a deterministic contiguous half-open range for one shard."""
    if length < 0:
        raise ValueError("length must be non-negative")
    if count <= 0:
        raise ValueError("shard count must be positive")
    if index < 0 or index >= count:
        raise ValueError(f"shard index {index} is outside [0, {count})")
    start = length * index // count
    end = length * (index + 1) // count
    return start, end


def _write_npy_atomic(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value, allow_pickle=False)
    temporary.replace(path)


def _write_jsonl_atomic(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _select_range(dataset, start: int, end: int):
    if hasattr(dataset, "select"):
        return dataset.select(range(start, end))
    return torch.utils.data.Subset(dataset, range(start, end))


def _run_embedding_shard(
    official_eval,
    options: argparse.Namespace,
    upstream_args: list[str],
) -> None:
    """Encode one query/candidate range without computing a partial score."""
    from src.arguments import DataArguments, ModelArguments, TrainingArguments
    from src.data.collator.eval_collator import MultimodalEvalDataCollator
    from src.data.eval_dataset.base_eval_dataset import AutoEvalPairDataset, generate_cand_dataset
    from src.model.model import MMEBModel
    from src.model.processor import load_processor
    from torch.utils.data import DataLoader
    from transformers import HfArgumentParser

    old_argv = sys.argv
    try:
        sys.argv = [str(Path(official_eval.__file__).resolve()), *upstream_args]
        model_args, data_args, training_args = HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments)
        ).parse_args_into_dataclasses()
    finally:
        sys.argv = old_argv

    task_configs = __import__("yaml").safe_load(Path(data_args.dataset_config).read_text())
    if len(task_configs) != 1:
        raise ValueError("Embedding shard mode requires exactly one dataset in --dataset_config")
    dataset_name, raw_task_config = next(iter(task_configs.items()))
    task_config = dict(raw_task_config)
    for key in (
        "image_root",
        "video_root",
        "frame_root",
        "clip_root",
        "audio_root",
        "data_path",
        "query_file",
        "candidate_file",
        "qrels_file",
    ):
        if data_args.data_basedir and task_config.get(key):
            task_config[key] = os.path.join(data_args.data_basedir, task_config[key])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_args.device = device
    processor = load_processor(model_args, data_args)
    model = MMEBModel.load(model_args, is_trainable=False, processor=processor)
    model.eval()
    model = model.to(device, dtype=torch.bfloat16)

    full_queries, corpus = AutoEvalPairDataset.instantiate(
        model_args=model_args,
        data_args=data_args,
        **task_config,
    )
    full_candidates = generate_cand_dataset(full_queries, corpus)
    shard_root = Path(data_args.encode_output_path) / "embedding_shard"
    shard_root.mkdir(parents=True, exist_ok=True)
    side_results: dict[str, dict[str, int]] = {}

    for encode_side, full_dataset in (("qry", full_queries), ("cand", full_candidates)):
        start, end = shard_bounds(len(full_dataset), options.embedding_shard_index, options.embedding_shard_count)
        shard_dataset = _select_range(full_dataset, start, end)
        batch_size = training_args.per_device_eval_batch_size
        if dataset_name in {"Charades-STA", "QVHighlight", "MomentSeeker", "YouCook2", "Video-MME"}:
            batch_size = min(batch_size, 4)
        collator = MultimodalEvalDataCollator(processor, model_args, data_args, encode_side)
        loader = DataLoader(
            shard_dataset,
            batch_size=batch_size,
            collate_fn=collator,
            num_workers=training_args.dataloader_num_workers,
            timeout=(
                int(os.environ.get("MMEB_DATALOADER_TIMEOUT", "600")) if training_args.dataloader_num_workers > 0 else 0
            ),
        )
        embeddings, infos = official_eval.encode_embeddings(
            model,
            loader,
            training_args,
            model_args,
            shard_dataset,
            encode_side=encode_side,
            description=f"{encode_side} shard {options.embedding_shard_index + 1}/"
            f"{options.embedding_shard_count} for {dataset_name}",
        )
        expected = end - start
        if len(embeddings) != expected or len(infos) != expected:
            raise RuntimeError(
                f"{encode_side} shard produced {len(embeddings)} embeddings and {len(infos)} IDs; expected {expected}"
            )
        _write_npy_atomic(shard_root / f"{encode_side}.npy", np.asarray(embeddings, dtype=np.float32))
        _write_jsonl_atomic(shard_root / f"{encode_side}_info.jsonl", infos)
        side_results[encode_side] = {"start": start, "end": end, "total": len(full_dataset)}

    manifest = {
        "schema_version": 1,
        "dataset_name": dataset_name,
        "dataset_config_sha256": hashlib.sha256(Path(data_args.dataset_config).read_bytes()).hexdigest(),
        "shard_index": options.embedding_shard_index,
        "shard_count": options.embedding_shard_count,
        "query": side_results["qry"],
        "candidate": side_results["cand"],
        "embedding_dimension": int(model.rep_dim),
    }
    _write_json_atomic(shard_root / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


class _ScoreOnlyModel:
    rep_dim = 0

    def eval(self):
        return self

    def to(self, *_args, **_kwargs):
        return self


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", choices=("hf", "hmonnx", "vllm", "vllm_merak"), required=True)
    parser.add_argument("--hmonnx-meta")
    parser.add_argument("--image-min-pixels", type=int, default=64 * 32 * 32)
    parser.add_argument("--image-max-pixels", type=int, default=1536 * 4 * 16 * 16)
    parser.add_argument("--protocol-file")
    parser.add_argument("--vllm-max-model-len", type=int, default=262144)
    parser.add_argument("--vllm-max-num-seqs", type=int, default=1)
    parser.add_argument("--vllm-max-images", type=int, default=16)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--vllm-merak-source-sha256")
    parser.add_argument("--video-min-pixels", type=int, default=4 * 32 * 32)
    parser.add_argument("--video-max-pixels", type=int, default=256 * 32 * 32)
    parser.add_argument("--video-total-pixels", type=int, default=8192 * 32 * 32)
    parser.add_argument("--embedding-shard-count", type=int, default=1)
    parser.add_argument("--embedding-shard-index", type=int, default=0)
    parser.add_argument("--embedding-shard-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    options, upstream_args = parser.parse_known_args()
    if options.backend in {"hmonnx", "vllm_merak"} and not options.hmonnx_meta:
        parser.error("--hmonnx-meta is required for HMONNX-backed backends")
    if options.vllm_max_num_seqs <= 0:
        parser.error("--vllm-max-num-seqs must be positive")
    if options.backend == "vllm_merak" and options.vllm_max_num_seqs != 1:
        parser.error("vLLM Merak hybrid linear state requires --vllm-max-num-seqs 1")
    if options.image_min_pixels <= 0 or options.image_max_pixels < options.image_min_pixels:
        parser.error("Expected 0 < --image-min-pixels <= --image-max-pixels")
    if options.embedding_shard_count <= 0:
        parser.error("--embedding-shard-count must be positive")
    if not 0 <= options.embedding_shard_index < options.embedding_shard_count:
        parser.error("--embedding-shard-index must be in [0, --embedding-shard-count)")
    if options.embedding_shard_only and options.embedding_shard_count < 2:
        parser.error("--embedding-shard-only requires --embedding-shard-count >= 2")
    if options.embedding_shard_only and options.score_only:
        parser.error("--embedding-shard-only and --score-only are mutually exclusive")
    if "--dataset_config" not in upstream_args:
        parser.error("--dataset_config is required")
    dataset_config = upstream_args[upstream_args.index("--dataset_config") + 1]
    import yaml

    tasks = yaml.safe_load(Path(dataset_config).read_text())
    if not tasks:
        parser.error("The MMEB-v3 task configuration is empty")
    if options.backend == "hmonnx":
        image_parsers = {"image_cls", "image_qa", "image_i2t", "image_t2i", "image_i2i_vg", "mcmr"}
        if any(task.get("dataset_parser") not in image_parsers for task in tasks.values()):
            parser.error("The direct HMONNX adapter accepts image-only MMEB-v3 task configurations")

    # The upstream package imports a video parser while registering all tasks.
    # imageio is only needed when that video parser is actually executed.
    try:
        import imageio  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["imageio"] = types.ModuleType("imageio")
    try:
        import decord  # noqa: F401
    except ModuleNotFoundError:
        decord_stub = types.ModuleType("decord")
        decord_stub.VideoReader = None
        decord_stub.cpu = None
        sys.modules["decord"] = decord_stub

    import eval as official_eval
    from src.data.collator.eval_collator import MultimodalEvalDataCollator
    from src.model import processor as official_processor
    from src.model.model import MMEBModel

    # Both backends preserve aspect ratio and use the same pixel bounds.  The
    # maximum matches the largest exported HMONNX visual gear (m1536).
    official_processor.WEMM_IMAGE_KWARGS.pop("resized_height", None)
    official_processor.WEMM_IMAGE_KWARGS.pop("resized_width", None)
    official_processor.WEMM_IMAGE_KWARGS.update(
        min_pixels=options.image_min_pixels,
        max_pixels=options.image_max_pixels,
    )
    official_processor.WEMM_VIDEO_KWARGS.update(
        min_pixels=options.video_min_pixels,
        max_pixels=options.video_max_pixels,
        total_pixels=options.video_total_pixels,
    )

    if options.backend in {"hmonnx", "vllm", "vllm_merak"}:
        original_load = MMEBModel.load
        original_collate = MultimodalEvalDataCollator.__call__
        original_encode = official_eval.encode_embeddings

        def load_model(cls, model_args, is_trainable=True, **kwargs):
            if model_args.model_backbone == official_processor.WEMM_EMBEDDING:
                if options.score_only:
                    return _ScoreOnlyModel()
                if options.backend == "vllm":
                    from examples_merak.llm.wemm_embedding.mmeb.mmeb_v3_vllm_merak import VLLMWeMM

                    return VLLMWeMM(
                        model_args.model_name,
                        options.vllm_max_model_len,
                        options.image_min_pixels,
                        options.image_max_pixels,
                        options.video_min_pixels,
                        options.video_max_pixels,
                        options.video_total_pixels,
                        options.vllm_max_images,
                        options.vllm_gpu_memory_utilization,
                        options.vllm_max_num_seqs,
                    )
                if options.backend == "vllm_merak":
                    from examples_merak.llm.wemm_embedding.mmeb.mmeb_v3_vllm_merak import VLLMMerakWeMM

                    return VLLMMerakWeMM(
                        options.hmonnx_meta,
                        options.vllm_max_model_len,
                        options.image_min_pixels,
                        options.image_max_pixels,
                        options.video_min_pixels,
                        options.video_max_pixels,
                        options.video_total_pixels,
                        options.vllm_max_images,
                        options.vllm_gpu_memory_utilization,
                        options.vllm_max_num_seqs,
                    )
                device = os.environ.get("MMEB_DEVICE", f"cuda:{os.environ.get('LOCAL_RANK', '0')}")
                return HMONNXWeMM(
                    options.hmonnx_meta,
                    device,
                    options.image_min_pixels,
                    options.image_max_pixels,
                )
            return original_load(model_args, is_trainable=is_trainable, **kwargs)

        def collate(self, examples):
            if self.model_args.model_backbone not in (official_processor.WEMM_EMBEDDING, "wemm_hmonnx"):
                return original_collate(self, examples)
            if self.encode_side == "qry":
                inputs = self._get_batch_inputs(examples, "query_text", "query_image", "query_audio", "image")
            else:
                if "cand_video" in examples[0]:
                    inputs = self._get_batch_inputs(examples, "cand_text", "cand_video", "cand_audio", "video")
                else:
                    inputs = self._get_batch_inputs(examples, "cand_text", "cand_image", "cand_audio", "image")
            if inputs.get("audios") and any(a is not None for a in inputs["audios"]):
                raise ValueError("Audio is unsupported by WeMM HMONNX")
            return inputs, [example["dataset_infos"] for example in examples]

        def encode(model, loader, training_args, model_args, full_dataset, encode_side, description="Encoding"):
            if model_args.model_backbone != official_processor.WEMM_EMBEDDING:
                return original_encode(
                    model,
                    loader,
                    training_args,
                    model_args,
                    full_dataset,
                    encode_side,
                    description,
                )
            if torch.distributed.is_initialized():
                raise RuntimeError("The vLLM MMEB adapters run one independent evaluator per GPU")

            from tqdm import tqdm

            rows = []
            gt_infos = []
            model.eval()
            with torch.no_grad():
                for inputs, dataset_info in tqdm(loader, desc=f"{description} (rank 0)"):
                    output = model(qry=inputs) if encode_side == "qry" else model(tgt=inputs)
                    key = "qry_reps" if encode_side == "qry" else "tgt_reps"
                    rows.append(output[key].detach().float().cpu())
                    if encode_side == "qry":
                        gt_infos.extend(dataset_info)
                    else:
                        gt_infos.extend(info["cand_name"] for info in dataset_info)
            if not rows:
                import numpy as np

                return np.empty((0, int(model.rep_dim)), dtype=np.float32), []
            return torch.cat(rows, dim=0).numpy(), gt_infos

        MMEBModel.load = classmethod(load_model)
        MultimodalEvalDataCollator.__call__ = collate
        official_eval.encode_embeddings = encode
        if options.score_only:
            official_eval.load_processor = lambda *_args, **_kwargs: object()

    if options.protocol_file:

        def value_after(flag: str):
            return upstream_args[upstream_args.index(flag) + 1] if flag in upstream_args else None

        dataset_sha256 = hashlib.sha256(Path(dataset_config).read_bytes()).hexdigest()
        model_name = value_after("--model_name")
        hf_config = Path(model_name) / "config.json" if model_name else None
        hf_config_sha256 = (
            hashlib.sha256(hf_config.read_bytes()).hexdigest() if hf_config and hf_config.is_file() else None
        )
        if options.hmonnx_meta:
            from examples_merak.llm.wemm_embedding.runtime_utils import resolve_hmonnx_meta

            meta_path = resolve_hmonnx_meta(options.hmonnx_meta).resolve()
            meta_sha256 = hashlib.sha256(meta_path.read_bytes()).hexdigest()
        else:
            meta_path = None
            meta_sha256 = None
        upstream_root = Path(official_eval.__file__).resolve().parent.parent
        upstream_revision = subprocess.check_output(
            ["git", "-C", str(upstream_root), "rev-parse", "HEAD"], text=True
        ).strip()
        import transformers

        adapter_path = Path(__file__).with_name("mmeb_v3_vllm_merak.py")
        if options.backend == "vllm":
            from vllm_merak.model_executor.model_loader import (
                wemm_embedding_native,
            )

            native_wemm_path = Path(inspect.getfile(wemm_embedding_native)).resolve()
        else:
            native_wemm_path = None

        protocol = {
            "evaluator": "WeMM official mmeb_v3_eval",
            "official_repo_commit": upstream_revision,
            "mmeb_v3_data_revision": "4a5560b2b64384204b6fea8a82ea986eba51f5aa",
            "mmeb_v2_media_revision": "e7bbfeb69a70dfe32ff36da3d6d8dbe31fc36af1",
            "backend": options.backend,
            "transformers_version": transformers.__version__,
            "torch_version": torch.__version__,
            "vllm_version": (importlib.metadata.version("vllm") if options.backend in {"vllm", "vllm_merak"} else None),
            "vllm_merak_version": (
                importlib.metadata.version("vllm-merak") if options.backend == "vllm_merak" else None
            ),
            "vllm_merak_source_sha256": options.vllm_merak_source_sha256,
            "wemm_vllm_adapter_sha256": (
                hashlib.sha256(adapter_path.read_bytes()).hexdigest()
                if options.backend in {"vllm", "vllm_merak"}
                else None
            ),
            "mmeb_v3_eval_adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "vllm_wemm_model_sha256": (
                hashlib.sha256(native_wemm_path.read_bytes()).hexdigest() if native_wemm_path is not None else None
            ),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "image_preprocessing": {
                "mode": "dynamic",
                "resized_height": None,
                "resized_width": None,
                "min_pixels": options.image_min_pixels,
                "max_pixels": options.image_max_pixels,
                "preserve_aspect_ratio": True,
            },
            "video_preprocessing": {
                "num_frames": 64,
                "frame_size": 224,
                "min_pixels": options.video_min_pixels,
                "max_pixels": options.video_max_pixels,
                "total_pixels": options.video_total_pixels,
                "processor_max_pixels": ("min(num_frames * max_pixels, 2 * total_pixels), with temporal padding"),
                "sample_fps": 1,
            },
            "vllm": {
                "max_model_len": options.vllm_max_model_len,
                "max_num_seqs": options.vllm_max_num_seqs,
                "long_text_safety_margin_tokens": 1,
                "max_images": options.vllm_max_images,
                "gpu_memory_utilization": options.vllm_gpu_memory_utilization,
            }
            if options.backend in {"vllm", "vllm_merak"}
            else None,
            "dataset_config_sha256": dataset_sha256,
            "model_name": model_name,
            "hf_config_sha256": hf_config_sha256,
            "hmonnx_meta": str(meta_path) if meta_path else None,
            "hmonnx_meta_sha256": meta_sha256,
            "upstream_args": upstream_args,
            "embedding_shard": (
                {
                    "index": options.embedding_shard_index,
                    "count": options.embedding_shard_count,
                    "shard_only": options.embedding_shard_only,
                }
                if options.embedding_shard_only
                else None
            ),
            "score_only": options.score_only,
        }
        path = Path(options.protocol_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")

    if options.embedding_shard_only:
        _run_embedding_shard(official_eval, options, upstream_args)
        return

    sys.argv = [str(Path(official_eval.__file__).resolve()), *upstream_args]
    try:
        official_eval.main()
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
