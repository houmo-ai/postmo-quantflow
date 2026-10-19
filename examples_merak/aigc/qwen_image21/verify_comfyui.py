# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Execute a real ComfyUI W8A8 prompt, inspect history and saved PNG metadata."""

import argparse
import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path

import aiohttp
import numpy as np
from PIL import Image

from .comfyui_workflow import build_workflow


async def verify(args):
    workflow, prompt = build_workflow(args.prompt)
    prompt["1"]["inputs"].update(width=args.width, height=args.height, seed=args.seed)
    workflow["nodes"][0]["widgets_values"] = [args.prompt, args.width, args.height, args.seed, "fixed"]
    prefix = f"Qwen_Image_21_Turbo/w8a8/{args.name}"
    prompt["2"]["inputs"]["filename_prefix"] = prefix
    workflow["nodes"][1]["widgets_values"] = [prefix]
    if args.reference:
        prompt["3"] = {"class_type": "LoadImage", "inputs": {"image": args.reference}}
        prompt["1"]["inputs"]["reference_image"] = ["3", 0]
        workflow["nodes"].append(
            {
                "id": 3,
                "type": "LoadImage",
                "pos": [-430, 100],
                "size": [400, 450],
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "links": [2]},
                    {"name": "MASK", "type": "MASK", "links": None},
                ],
                "properties": {"Node name for S&R": "LoadImage"},
                "widgets_values": [args.reference, "image"],
            }
        )
        workflow["nodes"][0]["inputs"][0]["link"] = 2
        workflow["nodes"][0]["order"], workflow["nodes"][1]["order"] = 1, 2
        workflow["links"].append([2, 3, 0, 1, 0, "IMAGE"])
        workflow["last_node_id"], workflow["last_link_id"] = 3, 2
    client_id = str(uuid.uuid4())
    started = time.perf_counter()
    timeout = aiohttp.ClientTimeout(total=7200)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        if args.clear_cache:
            async with session.get(args.url + "/queue") as response:
                queue = await response.json()
            if queue["queue_running"] or queue["queue_pending"]:
                raise RuntimeError("ComfyUI queue must be idle before clearing its execution cache")
            async with session.post(args.url + "/free", json={"free_memory": True}) as response:
                response.raise_for_status()
            await asyncio.sleep(1)
        async with session.ws_connect(args.url + "/ws?clientId=" + client_id) as websocket:
            async with session.post(
                args.url + "/prompt",
                json={
                    "prompt": prompt,
                    "client_id": client_id,
                    "extra_data": {"extra_pnginfo": {"workflow": workflow}},
                },
            ) as response:
                submission = await response.json()
                if response.status != 200 or submission.get("node_errors"):
                    raise RuntimeError(submission)
            prompt_id = submission["prompt_id"]
            print("submitted", args.name, prompt_id, flush=True)
            async for message in websocket:
                if message.type != aiohttp.WSMsgType.TEXT:
                    continue
                event = json.loads(message.data)
                data = event.get("data", {})
                if data.get("prompt_id") != prompt_id:
                    continue
                if event["type"] in ("execution_error", "execution_interrupted"):
                    raise RuntimeError(data)
                if event["type"] == "executing" and data.get("node") is None:
                    break
            async with session.get(args.url + "/history/" + prompt_id) as response:
                history = (await response.json())[prompt_id]
    if history["status"]["status_str"] != "success":
        raise RuntimeError(history["status"])
    output = history["outputs"]["2"]["images"][0]
    path = args.comfy_root / "output" / output["subfolder"] / output["filename"]
    with Image.open(path) as image:
        assert image.size == (args.width, args.height) and image.mode == "RGBA"
        embedded_prompt = json.loads(image.info["prompt"])
        embedded_workflow = json.loads(image.info["workflow"])
        for key, value in prompt.items():
            assert embedded_prompt[key]["class_type"] == value["class_type"]
            assert embedded_prompt[key]["inputs"] == value["inputs"]
        assert embedded_workflow == workflow
        assert np.asarray(image).std() > 1
    if "1" not in history["outputs"]:
        raise RuntimeError("ComfyUI reused a cached node result; rerun with --clear-cache to verify real inference")
    stats = json.loads(history["outputs"]["1"]["text"][0])
    assert stats["steps"] == 8 and stats["reference_count"] == int(bool(args.reference))
    report = {
        "prompt_id": prompt_id,
        "output": str(path),
        "wall_s": time.perf_counter() - started,
        "stats": stats,
        "png_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "prompt": prompt,
        "workflow": workflow,
        "history": history,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key not in ("history", "workflow", "prompt")},
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8188")
    parser.add_argument("--comfy-root", type=Path, required=True)
    parser.add_argument("--name", default="verification")
    parser.add_argument("--prompt", default="A red panda reading a book in a cozy library.")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference", help="Filename already in ComfyUI/input")
    parser.add_argument("--clear-cache", action="store_true", help="Require idle queue and clear ComfyUI result cache")
    parser.add_argument("--report", type=Path, required=True)
    asyncio.run(verify(parser.parse_args(argv)))


if __name__ == "__main__":
    main()
