# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Write one editable ComfyUI workflow and its equivalent API prompt."""

import argparse
import json
import uuid
from pathlib import Path


def build_workflow(prompt="A red panda reading a book in a cozy library."):
    values = {"prompt": prompt, "width": 1024, "height": 1024, "seed": 42}
    kind = "MerakQwenImage21TurboW8A8"
    prefix = "Qwen_Image_21_Turbo/w8a8/merak_turbo"
    api = {
        "1": {"class_type": kind, "inputs": values},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": prefix}},
    }
    workflow = {
        "id": str(uuid.uuid4()),
        "revision": 0,
        "last_node_id": 2,
        "last_link_id": 1,
        "nodes": [
            {
                "id": 1,
                "type": kind,
                "pos": [100, 100],
                "size": [550, 360],
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [{"name": "reference_image", "type": "IMAGE", "link": None, "shape": 7}],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [1], "slot_index": 0}],
                "properties": {"Node name for S&R": kind},
                "widgets_values": [prompt, 1024, 1024, 42, "fixed"],
            },
            {
                "id": 2,
                "type": "SaveImage",
                "pos": [740, 100],
                "size": [550, 600],
                "flags": {},
                "order": 1,
                "mode": 0,
                "inputs": [{"name": "images", "type": "IMAGE", "link": 1}],
                "outputs": [],
                "properties": {"Node name for S&R": "SaveImage"},
                "widgets_values": [prefix],
            },
        ],
        "links": [[1, 1, 0, 2, 0, "IMAGE"]],
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 0.85, "offset": [0, 0]}},
        "version": 0.4,
    }
    return workflow, api


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, value in zip((".json", "_api.json"), build_workflow(), strict=True):
        path = args.output_dir / ("qwen_image21_turbo_w8a8" + suffix)
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
