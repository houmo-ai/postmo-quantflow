# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Wall-X image and prompt geometry, preserving actual non-square grids."""

import json
from pathlib import Path

import numpy as np
import torch


def resize_image(image, long_edge=448):
    from qwen_vl_utils.vision_process import smart_resize

    image = image.convert("RGB")
    w, h = image.size
    if long_edge <= 0:
        raise ValueError("long_edge must be positive")
    if w > h:
        size = (long_edge, max(1, int(long_edge * h / w)))
    else:
        size = (max(1, int(long_edge * w / h)), long_edge)
    image = image.resize(size)
    h, w = smart_resize(image.height, image.width, factor=28, min_pixels=3136, max_pixels=12845056)
    return image.resize((w, h))


def vision_geometry(height, width, *, capacity, head_dim, window_size=112, patch_size=14):
    if height <= 0 or width <= 0 or height % 2 or width % 2 or height * width > capacity or capacity % 4:
        raise ValueError("even actual patch grid must fit visual capacity")
    mh, mw = height // 2, width // 2
    side = window_size // patch_size // 2
    groups, spans = [], []
    for row in range(0, mh, side):
        for col in range(0, mw, side):
            current = [h * mw + w for h in range(row, min(row + side, mh)) for w in range(col, min(col + side, mw))]
            begin = len(groups) * 4
            groups.extend(current)
            spans.extend([(begin, len(groups) * 4)] * (len(current) * 4))
    n = height * width
    group_order = torch.tensor(groups, dtype=torch.long)
    order = (group_order[:, None] * 4 + torch.arange(4)).flatten()
    reverse = group_order.argsort()
    row = torch.arange(height)[:, None].expand(height, width).reshape(mh, 2, mw, 2).permute(0, 2, 1, 3).flatten()
    col = torch.arange(width)[None, :].expand(height, width).reshape(mh, 2, mw, 2).permute(0, 2, 1, 3).flatten()
    inv = 1 / (10000.0 ** (torch.arange(0, head_dim // 2, 2).float() / (head_dim // 2)))
    angles = torch.cat((row[:, None] * inv, col[:, None] * inv), -1)[order]
    angles = torch.cat((angles, angles), -1)
    pad = capacity - n
    order = torch.cat((order, torch.arange(n, capacity))).to(torch.int32)
    reverse = torch.cat((reverse, torch.arange(n // 4, capacity // 4))).to(torch.int32)
    cos = torch.nn.functional.pad(angles.cos(), (0, 0, 0, pad), value=1)[None, None]
    sin = torch.nn.functional.pad(angles.sin(), (0, 0, 0, pad))[None, None]
    mask = (torch.arange(capacity) < n).float().reshape(1, capacity, 1)
    full = torch.zeros(1, capacity, 2, dtype=torch.int32)
    full[0, :n, 1] = n
    window = torch.zeros_like(full)
    window[0, :n] = torch.tensor(spans, dtype=torch.int32)
    return order, reverse, cos, sin, mask, full, window


def language_rope(position_ids, head_dim=128, theta=1000000.0, sections=(16, 24, 24)):
    inv = 1 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    angle = position_ids.float().unsqueeze(-1) * inv
    angle = torch.cat((angle, angle), -1)
    chunks = angle.split(tuple(sections) * 2, -1)
    selected = torch.cat([chunk[i % 3] for i, chunk in enumerate(chunks)], -1)
    return selected.cos().unsqueeze(1), selected.sin().unsqueeze(1)


class WallProcessor:
    def __init__(self, root, *, prefix_capacity=2048, visual_capacity=1024):
        from transformers import AutoProcessor

        self.root = Path(root)
        self.config = json.loads((self.root / "wall_config.json").read_text())
        self.processor = AutoProcessor.from_pretrained(self.root / "processor", local_files_only=True, use_fast=True)
        self.tokenizer = self.processor.tokenizer
        self.image_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self.normalizers = json.loads((self.root / "normalizers.json").read_text())
        self.prefix_capacity, self.visual_capacity = prefix_capacity, visual_capacity

    def normalize(self, values, key, kind):
        if key not in self.normalizers[kind]:
            raise ValueError(f"unknown {kind} norm_key: {key!r}")
        stats = self.normalizers[kind][key]
        lo, delta = np.array(stats["min"], np.float32), np.array(stats["delta"], np.float32)
        return np.clip((np.asarray(values, np.float32) - lo) / np.where(delta == 0, 1, delta) * 2 - 1, -1, 1)

    def prepare(self, instruction, images, state, norm_key, state_mask=None):
        if not instruction.strip() or len(images) != 2 or np.shape(state) != (26,):
            raise ValueError("requires instruction, two ordered camera images and 26 state values")
        if not np.isfinite(state).all():
            raise ValueError("state must be finite")
        normalized = self.normalize(state, norm_key, "propri")
        active = np.ones(26, bool) if state_mask is None else np.asarray(state_mask, bool)
        if active.shape != (26,):
            raise ValueError("state mask must have 26 entries")
        bins = np.digitize(normalized, np.linspace(-1, 1, 513)[:-1]) - 1
        state_text = " ".join(map(str, bins[active]))
        prepared = [resize_image(image, self.config["long_edge"]) for image in images]
        prompt = (
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\nObservation: front view: <|vision_start|><|image_pad|><|vision_end|>"
            " right wrist view: <|vision_start|><|image_pad|><|vision_end|>\nInstruction: "
            + instruction
            + "\nPredict the next action in robot action.\nProprioception: "
            + state_text
            + "\n<|im_end|>\n<|im_start|>assistant\n"
        )
        encoded = self.processor(text=[prompt], images=prepared, return_tensors="pt", do_resize=False)
        ids, grid = encoded["input_ids"], encoded["image_grid_thw"]
        if ids.shape[1] + self.config["horizon"] > self.prefix_capacity:
            raise ValueError("prompt and action exceed exported cache capacity")
        all_ids = torch.cat(
            (ids, torch.full((1, self.config["horizon"]), self.tokenizer.convert_tokens_to_ids("<|action|>"))), 1
        )
        positions, start, image_index, cursor = [], 0, 0, 0
        flat = all_ids[0].tolist()
        while cursor < len(flat):
            if flat[cursor] != self.image_token_id:
                positions.append(torch.full((3, 1), start, dtype=torch.long))
                start += 1
                cursor += 1
                continue
            _, h, w = grid[image_index].tolist()
            h //= 2
            w //= 2
            count = h * w
            if flat[cursor : cursor + count] != [self.image_token_id] * count:
                raise ValueError("image placeholders disagree with actual grid")
            positions.append(
                torch.stack(
                    (
                        torch.zeros(count, dtype=torch.long),
                        torch.arange(h).repeat_interleave(w),
                        torch.arange(w).repeat(h),
                    )
                )
                + start
            )
            start += max(h, w)
            cursor += count
            image_index += 1
        pos = torch.cat(positions, 1).unsqueeze(1)
        patches, geometry, offset = [], [], 0
        vc = self.config["vision_config"]
        for _, h, w in grid.tolist():
            count = h * w
            if count > self.visual_capacity:
                raise ValueError("image grid exceeds exported patch capacity")
            patches.append(
                torch.nn.functional.pad(
                    encoded["pixel_values"][offset : offset + count], (0, 0, 0, self.visual_capacity - count)
                )[None]
            )
            geometry.append(
                vision_geometry(
                    h,
                    w,
                    capacity=self.visual_capacity,
                    head_dim=vc["hidden_size"] // vc["num_heads"],
                    window_size=vc["window_size"],
                    patch_size=vc["patch_size"],
                )
            )
            offset += count
        return dict(
            ids=ids,
            grid=grid,
            patches=patches,
            geometry=geometry,
            rope=language_rope(
                pos,
                head_dim=self.config["model_config"]["hidden_size"]
                // self.config["model_config"]["num_attention_heads"],
                theta=self.config["model_config"]["rope_theta"],
                sections=self.config["model_config"]["rope_scaling"]["mrope_section"],
            ),
            prompt=prompt,
            sizes=[image.size for image in prepared],
        )
