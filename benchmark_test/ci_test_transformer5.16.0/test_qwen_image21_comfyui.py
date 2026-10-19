# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Real image transport and published-capacity preflight, without graph mocks."""

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
from PIL import Image

from examples_merak.aigc.qwen_image21.comfyui_nodes.client import decode_image, encode_image
from examples_merak.aigc.qwen_image21.comfyui_nodes.nodes import MerakQwenImage21TurboW8A8
from examples_merak.aigc.qwen_image21.comfyui_worker import create_server, error_payload, validate_request
from examples_merak.aigc.qwen_image21.comfyui_workflow import build_workflow


@pytest.fixture
def metadata():
    from _qwen_image21_bundle_fixture import bundle_metadata

    meta, _, _ = bundle_metadata("w8a8_sefp")
    meta["model_id"] = "qwen-image-2.1-turbo"
    return meta


@pytest.mark.parametrize("mode", ["RGB", "RGBA"])
def test_lossless_image_transport(mode):
    pixels = np.arange(32 * 64 * len(mode), dtype=np.uint8).reshape(32, 64, len(mode))
    restored = decode_image(encode_image(Image.fromarray(pixels)))
    assert restored.mode == mode
    np.testing.assert_array_equal(restored, pixels)


@pytest.mark.parametrize(
    "values",
    [
        {"width": 2048},
        {"width": 1536, "height": 1536},
        {"height": 1000},
        {"seed": -1},
        {"seed": True},
        {"prompt": ""},
        {"prompt": 7},
        {"steps": 4},
        {"reference_image": "not-png"},
    ],
)
def test_reject_invalid_request_before_runtime(metadata, values):
    with pytest.raises((ValueError, TypeError)):
        validate_request({"prompt": "A flower", "width": 1024, "height": 1024, "seed": 42, **values}, metadata)


@pytest.mark.parametrize("size", [(1024, 1024), (768, 1344), (1536, 704)])
def test_admitted_output_sizes_and_one_reference(metadata, size):
    image = Image.new("RGBA", (64, 32), (20, 40, 60, 255))
    values = validate_request(
        {
            "prompt": "Make it blue",
            "width": size[0],
            "height": size[1],
            "seed": 42,
            "reference_image": encode_image(image),
        },
        metadata,
    )
    assert values["width"] == size[0] and values["height"] == size[1]
    assert values["reference_image"].size == image.size


def test_real_http_info_and_preflight_errors(metadata):
    # No learned graph is required to reject invalid inputs or report capabilities.
    server = create_server("127.0.0.1", 0, artifact_dir="/unused", device="cpu", metadata=metadata)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(url + "/info") as response:
            info = json.load(response)
        assert info["model_id"] == "qwen-image-2.1-turbo"
        assert info["quant_type"] == "w8a8_sefp" and info["steps"] == 8
        request = urllib.request.Request(
            url + "/generate", data=b'{"prompt":"test","width":2048}', headers={"Content-Type": "application/json"}
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 400
        assert "canvas" in json.load(error.value)["error"]
        request = urllib.request.Request(
            url + "/generate", data=b'{"prompt":"test"}', headers={"Content-Type": "application/json"}
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 500
        failure = json.load(error.value)
        assert failure["code"] == "ARTIFACT_VALIDATION_FAILED"
        assert failure["details"]["artifact_dir"] == "/unused"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_worker_rejects_nonloopback_bind(metadata):
    with pytest.raises(ValueError, match="loopback"):
        create_server("0.0.0.0", 0, artifact_dir="/unused", device="cpu", metadata=metadata)


@pytest.mark.parametrize("field,value", [("model_id", "qwen-image-2.1"), ("quant_type", "w8a16_sefp")])
def test_worker_rejects_wrong_identity(metadata, field, value):
    metadata[field] = value
    with pytest.raises(ValueError, match="Turbo.*w8a8_sefp"):
        create_server("127.0.0.1", 0, artifact_dir="/unused", device="cpu", metadata=metadata)


@pytest.mark.parametrize("shape", [(2, 32, 32, 3), (1, 32, 32, 1), (32, 32, 3)])
def test_node_rejects_incompatible_reference_tensors(shape):
    import torch

    with pytest.raises(ValueError, match="exactly one"):
        MerakQwenImage21TurboW8A8().run("Make it blue", 1024, 1024, 42, torch.zeros(shape))


def test_workflow_widget_order_and_optional_reference():
    workflow, prompt = build_workflow("A flower")
    node = workflow["nodes"][0]
    assert node["type"] == prompt["1"]["class_type"]
    assert node["widgets_values"] == ["A flower", 1024, 1024, 42, "fixed"]
    assert node["inputs"] == [{"name": "reference_image", "type": "IMAGE", "link": None, "shape": 7}]
    assert prompt["2"]["inputs"]["images"] == ["1", 0]
    assert workflow["links"] == [[1, 1, 0, 2, 0, "IMAGE"]]


def test_inference_failure_exposes_actionable_reason():
    from xhmodel_merak.errors import ErrorCode, ModelError

    payload = error_payload(
        ModelError(
            ErrorCode.INFERENCE_FAILED, "Model runtime failed", reason="prompt exceeds the published token budget"
        )
    )
    assert "prompt exceeds the published token budget" in payload["error"]
    assert payload["code"] == "INFERENCE_FAILED"
