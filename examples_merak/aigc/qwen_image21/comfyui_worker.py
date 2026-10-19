# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Serial loopback HTTP bridge to the validated public image-inference API."""

import argparse
import gc
import json
import logging
import tempfile
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from xhmodel_merak.errors import ModelError
from xhmodel_merak.model_inference import inference_model
from xhmodel_merak.xh_other_model.models.qwen_image21.bundle import read_bundle_metadata
from xhmodel_merak.xh_other_model.models.qwen_image21.capacity import fit_reference_size, validate_size

from .comfyui_nodes.client import decode_image, encode_image


MODEL_ID = "qwen-image-2.1-turbo"
QUANT_TYPE = "w8a8_sefp"
MAX_REQUEST_BYTES = 64 * 1024 * 1024
LOGGER = logging.getLogger(__name__)


def error_payload(error):
    payload = {"error": str(error)}
    if isinstance(error, ModelError):
        payload.update(code=error.code.value, details=error.details)
        if error.details.get("reason"):
            payload["error"] += ": " + str(error.details["reason"])
    return payload


def validate_request(values, metadata):
    if not isinstance(values, dict):
        raise ValueError("request must be a JSON object")
    allowed = {"prompt", "width", "height", "seed", "reference_image"}
    if values.keys() - allowed:
        raise ValueError(f"unknown request fields: {sorted(values.keys() - allowed)}")
    prompt = values.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    seed = values.get("seed", 42)
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    capacity = metadata["deployment_config"]
    height = values.get("height", capacity["output_size"][0])
    width = values.get("width", capacity["output_size"][1])
    validate_size((height, width), capacity["canvas_size"], capacity["target_capacity"])
    reference = decode_image(values["reference_image"]) if "reference_image" in values else None
    return {"prompt": prompt, "width": width, "height": height, "seed": seed, "reference_image": reference}


class ImageWorker:
    def __init__(self, artifact_dir, device, metadata):
        if metadata.get("model_id") != MODEL_ID or metadata.get("quant_type") != QUANT_TYPE:
            raise ValueError("worker requires the official Turbo model exported as w8a8_sefp")
        self.artifact_dir, self.device, self.metadata = str(artifact_dir), device, metadata

    def info(self):
        return {
            "model_id": MODEL_ID,
            "quant_type": QUANT_TYPE,
            "backend": "HMONNX GPU simulation",
            "steps": 8,
            "guidance_scale": 1.0,
            "device": self.device,
            "capacity": self.metadata["deployment_config"],
            "artifact_dir": self.artifact_dir,
        }

    def generate(self, values):
        reference = values.pop("reference_image")
        started = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="qwen-image21-comfyui-") as temporary:
                images = []
                if reference is not None:
                    path = Path(temporary) / "reference.png"
                    reference.save(path)
                    images = [str(path)]
                result = inference_model(
                    artifact_dir=self.artifact_dir,
                    expected_model_id=MODEL_ID,
                    device=self.device,
                    inputs={**values, "images": images, "guidance_scale": 1.0},
                )
            stats = {
                **result.stats,
                "seed": result.seed,
                "width": result.width,
                "height": result.height,
                "inference_s": result.latency_s,
                "total_s": time.perf_counter() - started,
            }
            if reference is not None:
                cap = self.metadata["deployment_config"]
                fitted = fit_reference_size(
                    reference.height, reference.width, cap["canvas_size"], cap["reference_capacity"]
                )
                stats["reference_size_hw"] = list(fitted)
            LOGGER.info("generation completed: %s", stats)
            return {
                "model_id": MODEL_ID,
                "quant_type": QUANT_TYPE,
                "image": encode_image(Image.fromarray(np.asarray(result.image))),
                "stats": stats,
            }
        finally:
            # The public API owns per-request sessions and validates package hashes
            # on every invocation. Release CUDA allocator memory before returning
            # control to other ComfyUI workflows on the same GPU.
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def create_server(host, port, *, artifact_dir, device, metadata):
    if host != "127.0.0.1":
        raise ValueError("worker must bind to the IPv4 loopback address 127.0.0.1")
    worker = ImageWorker(artifact_dir, device, metadata)

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status, value):
            body = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/info":
                self.send_json(404, {"error": "unknown endpoint"})
                return
            self.send_json(200, worker.info())

        def do_POST(self):
            if self.path != "/generate":
                self.send_json(404, {"error": "unknown endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= MAX_REQUEST_BYTES:
                    raise ValueError("request body must contain 1..64 MiB of JSON")
                values = validate_request(json.loads(self.rfile.read(length)), worker.metadata)
            except (ValueError, TypeError) as error:
                self.send_json(400, {"error": str(error)})
                return
            try:
                self.send_json(200, worker.generate(values))
            except Exception as error:
                LOGGER.exception("W8A8 generation failed")
                self.send_json(500, error_payload(error))

    # One request at a time: HMONNX output buffers and GPU capacity are shared.
    server = HTTPServer((host, port), Handler)
    return server


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--port", type=int, default=8190)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    metadata = read_bundle_metadata(Path(args.artifact_dir))
    server = create_server(
        "127.0.0.1", args.port, artifact_dir=args.artifact_dir, device=args.device, metadata=metadata
    )
    LOGGER.info("Turbo W8A8 worker ready at http://127.0.0.1:%s/info", server.server_port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
