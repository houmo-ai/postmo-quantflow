# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Lossless loopback transport; no Merak or quantization SDK imports."""

import base64
import binascii
import io
import json
import os
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image


def encode_image(image):
    output = io.BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def decode_image(value):
    if not isinstance(value, str):
        raise ValueError("reference_image must be a base64 PNG string")
    try:
        raw = base64.b64decode(value, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PNG" or image.mode not in ("RGB", "RGBA"):
                raise ValueError("image must be an RGB/RGBA PNG")
            image.load()
            return image.copy()
    except (binascii.Error, OSError, Image.DecompressionBombError) as error:
        raise ValueError("invalid PNG image") from error


def generate(values):
    url = os.environ.get("MERAK_QWEN_IMAGE21_URL", "http://127.0.0.1:8190").rstrip("/")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.username or parsed.password:
        raise ValueError("MERAK_QWEN_IMAGE21_URL must be an HTTP URL on 127.0.0.1")
    request = urllib.request.Request(
        url + "/generate", data=json.dumps(values).encode("utf-8"), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=7200) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        try:
            detail = json.load(error)["error"]
        except (ValueError, KeyError):
            detail = str(error)
        raise RuntimeError(f"Merak W8A8 worker: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Cannot connect to Merak W8A8 worker at {url}; start comfyui_worker first") from error
    if result.get("quant_type") != "w8a8_sefp" or result.get("model_id") != "qwen-image-2.1-turbo":
        raise RuntimeError("worker returned an unexpected model or precision")
    return decode_image(result["image"]), result["stats"]
