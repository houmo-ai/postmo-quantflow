from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from xhmodel_merak.xh_llm.types import VisualModelMeta


REPO_ROOT = Path(__file__).resolve().parents[2]
PADDLEOCR_PACKAGE = REPO_ROOT / "xhmodel_merak" / "xh_llm" / "models" / "paddleocr_vl"


def _load_visual_token_gears():
    module_path = PADDLEOCR_PACKAGE / "visual_token_gears.py"
    spec = importlib.util.spec_from_file_location("paddleocr_visual_token_gears_under_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_runtime_with_lightweight_neighbors(monkeypatch):
    package_name = "xhmodel_merak.xh_llm.models.paddleocr_vl"
    package = types.ModuleType(package_name)
    package.__path__ = [str(PADDLEOCR_PACKAGE)]
    monkeypatch.setitem(sys.modules, package_name, package)

    data_module = types.ModuleType(f"{package_name}.data")
    data_module.PaddleOCRLLMInputProcessor = object
    monkeypatch.setitem(sys.modules, f"{package_name}.data", data_module)

    generation_module = types.ModuleType(f"{package_name}.generation")

    class PaddleOCRGenerationMixin:
        pass

    generation_module.PaddleOCRGenerationMixin = PaddleOCRGenerationMixin

    def validate_visual_pixel_values_shape(pixel_values, grid, patch_size):
        expected = (1, int(grid[0]) * int(grid[1]) * int(grid[2]), 3, int(patch_size), int(patch_size))
        if tuple(pixel_values.shape) != expected:
            raise ValueError(
                "PaddleOCR pixel_values shape must match image_grid_thw exactly: "
                f"expected {expected}, got {tuple(pixel_values.shape)}"
            )

    generation_module._validate_visual_pixel_values_shape = validate_visual_pixel_values_shape
    monkeypatch.setitem(sys.modules, f"{package_name}.generation", generation_module)

    hf_module = types.ModuleType(f"{package_name}.hf")
    hf_module.load_paddleocr_config = lambda *args, **kwargs: None
    hf_module.load_paddleocr_processor = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, f"{package_name}.hf", hf_module)

    module_path = PADDLEOCR_PACKAGE / "runtime.py"
    spec = importlib.util.spec_from_file_location(f"{package_name}.runtime", module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, f"{package_name}.runtime", module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_config_with_lightweight_package(monkeypatch):
    package_name = "xhmodel_merak.xh_llm.models.paddleocr_vl"
    package = types.ModuleType(package_name)
    package.__path__ = [str(PADDLEOCR_PACKAGE)]
    monkeypatch.setitem(sys.modules, package_name, package)
    module_path = PADDLEOCR_PACKAGE / "config.py"
    spec = importlib.util.spec_from_file_location(f"{package_name}.config", module_path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, f"{package_name}.config", module)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_paddleocr_visual_config_normalizes_and_validates_image_token_gears(monkeypatch, tmp_path):
    config_module = _load_config_with_lightweight_package(monkeypatch)

    config = config_module.XHPaddleOCRVLVisualConfig(
        model_name="paddleocr_vl_visual",
        hf_model=str(tmp_path),
        image_token_gears=[4, 1, 4, 2],
        image_token_capacity=4,
    )

    assert config.image_token_gears == (1, 2, 4)
    assert config.image_token_capacity == 4
    with pytest.raises(ValueError, match="image token gears must be positive"):
        config_module.XHPaddleOCRVLVisualConfig(
            model_name="paddleocr_vl_visual",
            hf_model=str(tmp_path),
            image_token_gears=[0, 4],
            image_token_capacity=4,
        )
    with pytest.raises(TypeError, match="image token gears must be integers"):
        config_module.XHPaddleOCRVLVisualConfig(
            model_name="paddleocr_vl_visual",
            hf_model=str(tmp_path),
            image_token_gears=[2, 4.5],
            image_token_capacity=5,
        )
    with pytest.raises(ValueError, match="must cover image_token_gears"):
        config_module.XHPaddleOCRVLVisualConfig(
            model_name="paddleocr_vl_visual",
            hf_model=str(tmp_path),
            image_token_gears=[2, 4],
            image_token_capacity=2,
        )


def test_paddleocr_visual_token_gears_select_smallest_capacity_that_fits():
    gears = _load_visual_token_gears()

    assert gears.normalize_image_token_gears([4, 1, 4, 2]) == (1, 2, 4)
    assert gears.select_image_token_gear(2, [4, 1, 2]) == 2
    assert gears.select_image_token_gear(3, [4, 1, 2]) == 4

    with pytest.raises(ValueError, match="exceeds the largest configured gear"):
        gears.select_image_token_gear(5, [1, 2, 4])


def test_paddleocr_visual_token_gears_convert_image_capacity_to_patch_capacity():
    gears = _load_visual_token_gears()

    assert gears.patch_token_capacity(2, spatial_merge_size=2) == 8

    with pytest.raises(ValueError, match="image_token_capacity must be positive"):
        gears.patch_token_capacity(0, spatial_merge_size=2)


def test_paddleocr_visual_token_gears_reorder_non_square_patches_to_merge_major_before_padding():
    gears = _load_visual_token_gears()
    pixel_values = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1, 1, 1)

    padded, valid_patch_tokens = gears.pad_patch_images(
        pixel_values,
        capacity=16,
        grid_thw=torch.tensor([[1, 2, 4]]),
        spatial_merge_size=2,
    )

    assert valid_patch_tokens == 8
    assert tuple(padded.shape) == (1, 16, 1, 1, 1)
    torch.testing.assert_close(
        padded[:, :8, 0, 0, 0],
        torch.tensor([[0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]]),
    )
    torch.testing.assert_close(padded[:, 8:], torch.zeros_like(padded[:, 8:]))


def test_paddleocr_visual_token_gears_build_geometry_mask_from_true_grid():
    gears = _load_visual_token_gears()

    geometry = gears.build_visual_token_gear_inputs(
        torch.tensor([[1, 2, 4]]),
        patch_capacity=16,
        num_positions=16,
        spatial_merge_size=2,
        dtype=torch.float32,
    )

    mask = geometry["attention_mask"]
    assert set(geometry) == {
        "position_ids",
        "position_weights",
        "rotary_position_ids",
        "attention_mask",
        "valid_patch_tokens",
        "valid_image_tokens",
    }
    assert geometry["valid_patch_tokens"] == 8
    assert geometry["valid_image_tokens"] == 2
    assert tuple(geometry["position_ids"].shape) == (4, 16)
    assert tuple(geometry["position_weights"].shape) == (4, 16)
    assert tuple(geometry["rotary_position_ids"].shape) == (2, 16)
    assert tuple(mask.shape) == (1, 1, 1, 16)
    torch.testing.assert_close(
        geometry["rotary_position_ids"][:, :8],
        torch.tensor(
            [
                [0, 0, 1, 1, 0, 0, 1, 1],
                [0, 1, 0, 1, 2, 3, 2, 3],
            ],
            dtype=torch.long,
        ),
    )
    torch.testing.assert_close(geometry["position_weights"][:, 8:], torch.zeros(4, 8))
    torch.testing.assert_close(mask[..., :8], torch.zeros(1, 1, 1, 8))
    assert bool(torch.all(mask[..., 8:] < 0).item())


def test_paddleocr_visual_router_routes_by_token_capacity_and_crops_padding(monkeypatch):
    runtime = _load_runtime_with_lightweight_neighbors(monkeypatch)
    calls = []

    class FakeHMONNXModel:
        @staticmethod
        def _is_inference_v2_env_enabled():
            return True

        def __init__(self, path, *, onnx_graph, enable_golden, device_map):
            self.path = path
            self.device = torch.device("cpu")
            self._input_info = {
                "pixel_values": SimpleNamespace(dtype=torch.float32),
                "position_ids": SimpleNamespace(dtype=torch.int64),
                "position_weights": SimpleNamespace(dtype=torch.float32),
                "rotary_position_ids": SimpleNamespace(dtype=torch.int64),
                "attention_mask": SimpleNamespace(dtype=torch.float32),
            }
            self.hmonnx_session = SimpleNamespace(get_input=lambda name: self._input_info[name])

        def forward(self, *inputs):
            assert len(inputs) == 5
            pixel_values = inputs[0]
            calls.append((self.path, tuple(pixel_values.shape)))
            return torch.arange(4 * 3, dtype=torch.float32).reshape(4, 3)

        def to(self, device):
            self.device = torch.device(device)
            return self

        def _set_dtype(self, dtype):
            return self

        def update_step(self):
            pass

    monkeypatch.setattr(runtime, "HMONNXModel", FakeHMONNXModel)
    monkeypatch.setattr(
        runtime,
        "MultiHMONNXLoader",
        lambda graph_files: SimpleNamespace(
            graphs={name: object() for name in graph_files},
            initializer_pool=SimpleNamespace(summary=lambda: {}),
        ),
    )
    visual_meta = VisualModelMeta()
    visual_meta.patch_size = 14
    visual_meta.spatial_merge_size = 2
    visual_meta.routing_policy = "smallest_fit"
    visual_meta.gears = [
        SimpleNamespace(image_token_capacity=2, patch_token_capacity=8, hmonnx="visual_m2.onnx"),
        SimpleNamespace(image_token_capacity=4, patch_token_capacity=16, hmonnx="visual_m4.onnx"),
    ]

    router = runtime.PaddleOCRVLVisualHMONNXRouter(visual_meta, device_map=["cpu"])
    output = router.forward(torch.ones(1, 8, 3, 14, 14), torch.tensor([[1, 2, 4]]))

    assert calls == [("visual_m2.onnx", (1, 8, 3, 14, 14))]
    torch.testing.assert_close(output, torch.arange(6, dtype=torch.float32).reshape(2, 3))


def test_paddleocr_visual_router_uses_larger_gear_when_image_needs_padding(monkeypatch):
    runtime = _load_runtime_with_lightweight_neighbors(monkeypatch)
    calls = []

    class FakeHMONNXModel:
        @staticmethod
        def _is_inference_v2_env_enabled():
            return True

        def __init__(self, path, *, onnx_graph, enable_golden, device_map):
            self.path = path
            self.device = torch.device("cpu")
            self._input_info = {
                "pixel_values": SimpleNamespace(dtype=torch.float32),
                "position_ids": SimpleNamespace(dtype=torch.int64),
                "position_weights": SimpleNamespace(dtype=torch.float32),
                "rotary_position_ids": SimpleNamespace(dtype=torch.int64),
                "attention_mask": SimpleNamespace(dtype=torch.float32),
            }
            self.hmonnx_session = SimpleNamespace(get_input=lambda name: self._input_info[name])

        def forward(self, *inputs):
            assert len(inputs) == 5
            pixel_values = inputs[0]
            attention_mask = inputs[-1]
            calls.append((self.path, tuple(pixel_values.shape), attention_mask.reshape(-1).tolist()))
            return torch.arange(4 * 2, dtype=torch.float32).reshape(4, 2)

        def to(self, device):
            self.device = torch.device(device)
            return self

        def _set_dtype(self, dtype):
            return self

        def update_step(self):
            pass

    monkeypatch.setattr(runtime, "HMONNXModel", FakeHMONNXModel)
    monkeypatch.setattr(
        runtime,
        "MultiHMONNXLoader",
        lambda graph_files: SimpleNamespace(
            graphs={name: object() for name in graph_files},
            initializer_pool=SimpleNamespace(summary=lambda: {}),
        ),
    )
    visual_meta = VisualModelMeta()
    visual_meta.patch_size = 14
    visual_meta.spatial_merge_size = 2
    visual_meta.routing_policy = "smallest_fit"
    visual_meta.gears = [
        SimpleNamespace(image_token_capacity=2, patch_token_capacity=8, hmonnx="visual_m2.onnx"),
        SimpleNamespace(image_token_capacity=4, patch_token_capacity=16, hmonnx="visual_m4.onnx"),
    ]

    router = runtime.PaddleOCRVLVisualHMONNXRouter(visual_meta, device_map=["cpu"])
    output = router.forward(torch.ones(1, 12, 3, 14, 14), torch.tensor([[1, 2, 6]]))

    assert calls[0][0] == "visual_m4.onnx"
    assert calls[0][1] == (1, 16, 3, 14, 14)
    assert calls[0][2][:12] == [0.0] * 12
    assert all(value < 0.0 for value in calls[0][2][12:])
    torch.testing.assert_close(output, torch.arange(6, dtype=torch.float32).reshape(3, 2))
