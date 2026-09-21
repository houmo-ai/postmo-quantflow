# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Compare real GLM vision artifacts and optionally a bounded text prefix with HF."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import DynamicCache
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel
from transformers.video_utils import VideoMetadata

from xhmodel_merak.xh_llm.hmonnx.auto_llm_hmonnx_model import AutoLLMHONNXModel
from xhmodel_merak.xh_llm.models.glm5_3_flash.checkpoint import Glm53Checkpoint
from xhmodel_merak.xh_llm.models.glm5_3_flash.vision import create_glm53_processor, load_vision_encoder


def metrics(actual, expected):
    a, b = actual.detach().float().cpu().flatten(), expected.detach().float().cpu().flatten()
    return dict(
        cosine=torch.nn.functional.cosine_similarity(a, b, dim=0).item(),
        max_abs=(a - b).abs().max().item(),
        finite=bool(torch.isfinite(a).all()),
    )


def sample_image(height, width, seed):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:height, :width]
    pixels = np.stack(((x * 255 // width), (y * 255 // height), ((x + y) % 64) * 4), axis=-1).astype(np.uint8)
    pixels[: height // 3, : width // 3] = rng.integers(0, 255, 3, dtype=np.uint8)
    return Image.fromarray(pixels)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="weights/GLM-5.3-Flash")
    parser.add_argument("--meta", required=True, help="multimodal artifact, vision is loaded independently")
    parser.add_argument("--text-meta", help="optional real text prefix artifact for bounded end-to-end checks")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    os.environ.setdefault("ENABLE_HMINFERENCE_V2", "1")
    torch.set_num_threads(1)
    torch.manual_seed(5311)
    meta_path = Path(args.meta)
    meta = json.loads(meta_path.read_text())
    encoder = load_vision_encoder(meta["glm53_vision"], meta_path.parent, device=args.device)
    reader = Glm53Checkpoint(args.model)
    with torch.device("meta"):
        native = Glm5NextVisionModel(reader.config.vision_config)
    reader.load_module(native, "model.visual.", device=args.device)
    # The source constructor's nonpersistent rotary frequencies are not checkpoint tensors.
    rotary_dim = reader.config.vision_config.hidden_size // reader.config.vision_config.num_heads // 2
    native.rotary_pos_emb.inv_freq = 1.0 / (
        10000 ** (torch.arange(0, rotary_dim, 2, device=args.device).float() / rotary_dim)
    )
    processor = create_glm53_processor(args.model, encoder.capacity)
    images = [sample_image(448, 448, 1), sample_image(196, 308, 2), sample_image(56, 112, 3)]
    frames = np.stack([np.asarray(sample_image(112, 168, i)) for i in range(4)])
    cases = {
        "square_image": [{"type": "image", "image": images[0]}],
        "two_rectangular_images": [{"type": "image", "image": images[1]}, {"type": "image", "image": images[2]}],
        "video": [{"type": "video", "video": frames}],
        "image_and_video": [{"type": "image", "image": images[2]}, {"type": "video", "video": frames}],
    }
    text_model = text_source = head = None
    if args.text_meta:
        text_model = AutoLLMHONNXModel.from_pretrained(
            args.text_meta, device_map=[args.device], enable_auto_offload=True
        )
        text_model.vision_encoder = encoder
        layers = len(text_model.cache_spec.layer_types)
        if layers > 4:
            raise ValueError("this bounded HF comparison accepts at most four text layers")
        text_source, head = reader.load_text_model(max_layers=layers, device=args.device)
    results = {}
    with torch.inference_mode():
        for name, content in cases.items():
            video_kwargs = {}
            if any(c["type"] == "video" for c in content):
                video_kwargs = dict(
                    do_sample_frames=False,
                    video_metadata=[VideoMetadata(total_num_frames=4, fps=2, frames_indices=[0, 1, 2, 3])],
                )
            content = content + [{"type": "text", "text": "请描述画面中的颜色。"}]
            inputs = processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
                processor_kwargs=video_kwargs,
            )
            inputs.pop("video_metadata", None)
            ids = inputs["input_ids"].to(args.device)
            features, native_features, detail = {}, {}, {"prompt_tokens": ids.shape[1]}
            for kind, pixels_key, grid_key in (
                ("image", "pixel_values", "image_grid_thw"),
                ("video", "pixel_values_videos", "video_grid_thw"),
            ):
                if pixels_key not in inputs:
                    continue
                pixels, grid = inputs[pixels_key], inputs[grid_key]
                actual = encoder(pixels, grid)
                expected = native(
                    pixels.to(device=args.device, dtype=torch.float16), grid_thw=grid.to(args.device)
                ).pooler_output
                detail[kind] = {**metrics(actual, expected), "grid": grid.tolist(), "shape": list(actual.shape)}
                assert detail[kind]["finite"] and detail[kind]["cosine"] > 0.98, detail
                features[kind], native_features[kind] = actual, expected
            if text_model is not None:
                embedding = text_model.get_input_embeddings()
                fused = embedding(ids.to(embedding.weight.device)).to(args.device)
                reference = fused.clone()
                video_depth = (ids == reader.config.video_start_token_id).int().cumsum(1)
                video_depth -= (ids == reader.config.video_end_token_id).int().cumsum(1)
                for kind in features:
                    mask = (ids == reader.config.image_token_id) & (
                        (video_depth > 0) if kind == "video" else (video_depth == 0)
                    )
                    fused[mask], reference[mask] = features[kind], native_features[kind]
                text_model.get_kvcache_mixin().clear_kv_cache()
                cache = DynamicCache(config=text_source.config)
                past, comparisons = 0, []
                text_model.set_prefill()
                p = text_model.get_input_sequence_length()
                for start in range(0, ids.shape[1], p):
                    text_model.set_prefill()
                    chunk, ref = fused[:, start : start + p], reference[:, start : start + p]
                    actual = text_model.forward(
                        *text_model.get_data_preprocessor()({"inputs_embeds": chunk, "past_seq_length": past})
                    )
                    expected = head(
                        text_source(inputs_embeds=ref, past_key_values=cache, use_cache=True).last_hidden_state[:, -1:]
                    )
                    comparisons.append(metrics(actual, expected))
                    past += chunk.shape[1]
                for _ in range(2):
                    next_id = actual.argmax(-1)
                    text_model.set_decode()
                    actual = text_model.forward(
                        *text_model.get_data_preprocessor()({"input_ids": next_id, "past_seq_length": past})
                    )
                    expected = head(
                        text_source(input_ids=next_id, past_key_values=cache, use_cache=True).last_hidden_state[:, -1:]
                    )
                    comparisons.append(metrics(actual, expected))
                    past += 1
                detail["logits"] = comparisons
                assert all(x["finite"] and x["cosine"] > 0.98 for x in comparisons), detail
                generated = text_model.generate(
                    **inputs.to(args.device), max_new_tokens=2, do_sample=False, eos_token_id=None
                )
                assert generated.shape[1] == ids.shape[1] + 2
                detail["generated_ids"] = generated[0, -2:].cpu().tolist()
            results[name] = detail
            print(name, json.dumps(detail), flush=True)
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
