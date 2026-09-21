"""Shared image/video/text generation used by the CLI and public inference API."""

import gc
import time


def generate_glm53(meta_file, *, prompt, max_new_tokens=64, image=None, video=None, device="cuda", devices=None):
    import torch
    from xhmodel_merak._environment import scoped_environment
    from ...hmonnx.auto_llm_hmonnx_model import AutoLLMHONNXModel
    from .vision import load_glm53_video

    selected = torch.device(device or "cuda")
    if devices is None:
        devices = (
            [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            if selected.type == "cuda" and selected.index is None
            else [str(selected)]
        )
    model = None
    with scoped_environment({
        "ENABLE_HMINFERENCE_V2": "1", "XHQUANT_PACKED_W4": "1", "XHQUANT_DISABLE_INITIALIZER_DIGEST_CACHE": "1"
    }):
        try:
            model = AutoLLMHONNXModel.from_pretrained(str(meta_file), device_map=devices, enable_auto_offload=True)
            processor = model.get_processor()
            image_paths = list(image) if isinstance(image, (list, tuple)) else ([image] if image else [])
            video_paths = list(video) if isinstance(video, (list, tuple)) else ([video] if video else [])
            videos = [load_glm53_video(processor, path) for path in video_paths]
            content = [{"type": "image", "image": path} for path in image_paths]
            content += [{"type": "video", "video": frames} for frames, _ in videos]
            content.append({"type": "text", "text": prompt})
            encoded = processor.apply_chat_template(
                [{"role": "user", "content": content}],
                add_generation_prompt=True,
                tokenize=True,
                return_tensors="pt",
                return_dict=True,
                processor_kwargs={"do_sample_frames": False, "video_metadata": [metadata for _, metadata in videos]}
                if videos
                else {},
            )
            length = encoded["input_ids"].shape[1]
            if length + max_new_tokens > model.cache_spec.max_context_length:
                raise ValueError("prompt plus generation exceeds the exported context capacity")
            encoded.pop("video_metadata", None)
            started = time.perf_counter()
            output = model.generate(**encoded.to(model.device), max_new_tokens=max_new_tokens, do_sample=False)
            if selected.type == "cuda":
                for item in devices:
                    torch.cuda.synchronize(item)
            elapsed = time.perf_counter() - started
            ids = output[0, length:].tolist()
            return {
                "output_text": processor.tokenizer.decode(ids, skip_special_tokens=True),
                "output_tokens": len(ids),
                "latency_s": elapsed,
                "tokens_per_second": len(ids) / elapsed,
                "stats": {"prompt_tokens": length, "output_ids": ids},
            }
        finally:
            if model is not None:
                model.release_inference_model()
            model = None
            gc.collect()
            if selected.type == "cuda":
                torch.cuda.empty_cache()
