"""Encode an ordered list of RGB frame files with an exported Qwen3.6 ViT."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_hmonnx_inference import VisualTokenGearHMONNXModel
from xhmodel_merak.xh_llm.models.qwen3_5_moe.qwen3_6_video_vision import Qwen3_6VideoVision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="original HF checkpoint containing video processor config")
    parser.add_argument("--visual-meta", required=True, help="standalone visual release runtime metadata JSON")
    parser.add_argument("--frames", nargs="+", required=True, help="frame paths in chronological order")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", required=True, type=Path, help="output .pt file")
    args = parser.parse_args()

    visual = VisualTokenGearHMONNXModel.from_meta_file(args.visual_meta, device=args.device)
    encoder = Qwen3_6VideoVision.from_pretrained(visual, args.model_dir)
    frames = []
    for path in args.frames:
        with Image.open(path) as image:
            frames.append(image.convert("RGB"))
    inputs = encoder.preprocess(frames)
    features = encoder.encode_patches(inputs["pixel_values_videos"], inputs["video_grid_thw"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"video_embeds": features.cpu(), "video_grid_thw": inputs["video_grid_thw"].cpu()}, args.output)
    print(f"frames={len(frames)}, grid={inputs['video_grid_thw'].tolist()}, features={tuple(features.shape)}")
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
