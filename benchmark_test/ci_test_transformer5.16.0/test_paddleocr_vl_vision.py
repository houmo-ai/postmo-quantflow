from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import (
    PaddleOCRTextConfig,
    PaddleOCRVisionConfig,
    PaddleOCRVLConfig,
    PaddleOCRVLForConditionalGeneration,
)

from xhmodel_merak.xh_llm.models.paddleocr_vl._vision_model_impl import build_static_image_embedding_model
from xhmodel_merak.xh_llm.models.paddleocr_vl.config import XHPaddleOCRVLVisualConfig
from xhmodel_merak.xh_llm.models.paddleocr_vl.generation import PaddleOCRGenerationMixin, _supported_visual_grids
from xhmodel_merak.xh_llm.models.paddleocr_vl.runtime import (
    PaddleOCRVLVisualHMONNXRouter,
    XHPaddleOCRVLHMONNXModel,
)
from xhmodel_merak.xh_llm.models.paddleocr_vl.visual import XHPaddleOCRVLVisualModel
from xhmodel_merak.xh_llm.models.paddleocr_vl.visual_token_gears import (
    build_visual_token_gear_inputs,
    pad_patch_images,
)
from xhmodel_merak.xh_llm.types import LLMModelState, VisualModelMeta, VLLMModelMeta


REQUIRED_PROFILE_GRIDS = [
    [1, 64, 64],
    [1, 82, 60],
    [1, 60, 84],
    [1, 50, 100],
    [1, 48, 74],
]


def _tiny_paddleocr_model(vision_num_attention_heads: int = 4) -> PaddleOCRVLForConditionalGeneration:
    config = PaddleOCRVLConfig(
        vision_config=PaddleOCRVisionConfig(
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=vision_num_attention_heads,
            num_channels=3,
            image_size=28,
            patch_size=14,
            spatial_merge_size=2,
            attention_dropout=0.0,
        ),
        text_config=PaddleOCRTextConfig(
            vocab_size=32,
            hidden_size=24,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=6,
            tie_word_embeddings=False,
        ),
        tie_word_embeddings=False,
    )
    config.vision_config._attn_implementation = "eager"
    model = PaddleOCRVLForConditionalGeneration(config)
    return model.eval()


@pytest.mark.parametrize("grid_thw", [torch.tensor([[1, 2, 2]]), torch.tensor([[1, 2, 4]])])
def test_static_visual_projector_matches_native_for_square_and_rectangular_grids(grid_thw):
    torch.manual_seed(20260908)
    native = _tiny_paddleocr_model()
    pixel_values = torch.randn(1, int(grid_thw.prod().item()), 3, 14, 14)
    wrapper = build_static_image_embedding_model(native.model, grid_thw).eval()

    with torch.no_grad():
        native_outputs = native.model.visual(
            pixel_values=pixel_values,
            cu_seqlens=torch.tensor([0, int(grid_thw.prod().item())]),
            grid_thw=grid_thw,
        )
        expected = native.model.projector(native_outputs.last_hidden_state, grid_thw)
        actual = wrapper(pixel_values)

    assert tuple(actual.shape) == (int(grid_thw.prod().item()) // 4, 24)
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-4)


def test_visual_model_declares_patch_token_gear_contract(tmp_path):
    config = XHPaddleOCRVLVisualConfig(
        model_name="paddleocr_vl_visual",
        hf_model=str(tmp_path),
        max_size_h=896,
        max_size_w=896,
        patch_size=14,
        spatial_merge_size=2,
        image_token_gears=[4],
        image_token_capacity=4,
    )
    model = XHPaddleOCRVLVisualModel(config)

    dummy = model.get_dummy_inputs()
    metadata = model.create_export_metadata(str(tmp_path))

    assert tuple(dummy["pixel_values"].shape) == (1, 16, 3, 14, 14)
    assert tuple(dummy["position_ids"].shape) == (4, 16)
    assert tuple(dummy["rotary_position_ids"].shape) == (2, 16)
    assert tuple(dummy["attention_mask"].shape) == (1, 1, 1, 16)
    assert metadata.image_token_capacity == 4
    assert metadata.input_shape == [1, 16, 3, 14, 14]
    assert metadata.output_shape == [4, 1024]
    assert model.get_export_cfg() == {
        "input_names": [
            "pixel_values",
            "position_ids",
            "position_weights",
            "rotary_position_ids",
            "attention_mask",
        ],
        "output_names": ["image_embeds"],
    }


def test_visual_config_defaults_to_maximum_processor_token_gear(tmp_path):
    config = XHPaddleOCRVLVisualConfig(model_name="paddleocr_vl_visual", hf_model=str(tmp_path))

    assert config.max_size_h == 896
    assert config.max_size_w == 896
    assert config.image_token_gears == (1280,)
    assert config.image_token_capacity == 1280


def test_visual_config_validates_visual_geometry(tmp_path):
    with pytest.raises(ValueError, match="patch_size must be positive"):
        XHPaddleOCRVLVisualConfig(
            model_name="paddleocr_vl_visual",
            hf_model=str(tmp_path),
            patch_size=0,
        )
    with pytest.raises(ValueError, match="spatial_merge_size must be positive"):
        XHPaddleOCRVLVisualConfig(
            model_name="paddleocr_vl_visual",
            hf_model=str(tmp_path),
            spatial_merge_size=0,
        )


def test_visual_model_metadata_uses_checkpoint_text_hidden_size(tmp_path):
    native = _tiny_paddleocr_model()
    native.config.save_pretrained(tmp_path)
    config = XHPaddleOCRVLVisualConfig(
        model_name="paddleocr_vl_visual",
        hf_model=str(tmp_path),
        max_size_h=28,
        max_size_w=56,
        patch_size=14,
        spatial_merge_size=2,
        image_token_gears=[2],
        image_token_capacity=2,
    )
    model = XHPaddleOCRVLVisualModel(config)

    metadata = model.create_export_metadata(str(tmp_path))

    assert metadata.output_shape == [2, 24]


def test_visual_model_wraps_full_native_model_without_copying_projector_math(tmp_path):
    torch.manual_seed(20260908)
    native = _tiny_paddleocr_model()
    grid_thw = torch.tensor([[1, 2, 6]])
    pixel_values = torch.randn(1, int(grid_thw.prod().item()), 3, 14, 14)
    config = XHPaddleOCRVLVisualConfig(
        model_name="paddleocr_vl_visual",
        hf_model=str(tmp_path),
        max_size_h=28,
        max_size_w=56,
        patch_size=14,
        spatial_merge_size=2,
        dtype="float32",
        image_token_gears=[4],
        image_token_capacity=4,
    )
    merak_visual = XHPaddleOCRVLVisualModel(config)

    wrapper = merak_visual.init_wrap_model(native).eval()
    assert wrapper.visual is not merak_visual._native_visual_source.visual
    assert wrapper.projector.linear_1 is not merak_visual._native_visual_source.projector.linear_1
    assert merak_visual.create_export_metadata(str(tmp_path)).num_positions == wrapper.num_positions

    with torch.no_grad():
        native_outputs = native.model.visual(
            pixel_values=pixel_values,
            cu_seqlens=torch.tensor([0, 12]),
            grid_thw=grid_thw,
        )
        expected = native.model.projector(native_outputs.last_hidden_state, grid_thw)
        padded, _ = pad_patch_images(pixel_values, 16, grid_thw, spatial_merge_size=2)
        geometry = build_visual_token_gear_inputs(
            grid_thw,
            patch_capacity=16,
            num_positions=wrapper.num_positions,
            spatial_merge_size=2,
            dtype=torch.float32,
            rotary_cache_length=wrapper.rotary_cache_length,
        )
        actual = wrapper(
            padded,
            geometry["position_ids"],
            geometry["position_weights"],
            geometry["rotary_position_ids"],
            geometry["attention_mask"],
        )[:3]

    assert tuple(actual.shape) == (3, 24)
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-4)


@pytest.mark.parametrize("grid_hw", [(2, 2), (2, 4)])
@pytest.mark.parametrize("loaded_dtype", [torch.float32, torch.float16])
def test_visual_profile_rebuild_preserves_rotary_constants_across_parent_dtype_casts(tmp_path, grid_hw, loaded_dtype):
    native = _tiny_paddleocr_model(vision_num_attention_heads=1).to(dtype=loaded_dtype)
    pristine = copy.deepcopy(native)
    model = XHPaddleOCRVLVisualModel(XHPaddleOCRVLVisualConfig(
        model_name="paddleocr_vl_visual",
        hf_model=str(tmp_path),
        max_size_h=28,
        max_size_w=28,
        dtype="float16",
        image_token_gears=[2],
        image_token_capacity=2,
    ))
    model.init_wrap_model(native)
    # Visual frontend export and parent language wrapping cast shared modules
    # before subsequent profile wrappers are constructed.
    native.half()
    model._wrap_model.float()
    delattr(native.model, "visual")
    delattr(native.model, "projector")
    model._reset_compiled_state_for_profile()
    model.config.image_token_gears = (grid_hw[0] * grid_hw[1] // 4,)
    model.config.image_token_capacity = grid_hw[0] * grid_hw[1] // 4

    rebuilt = model.init_wrap_model()
    expected = build_static_image_embedding_model(pristine.model, torch.tensor([[1, *grid_hw]])).half()

    torch.testing.assert_close(rebuilt.rotary_cos_cached, expected.rotary_cos_cached, rtol=0, atol=0)
    torch.testing.assert_close(rebuilt.rotary_sin_cached, expected.rotary_sin_cached, rtol=0, atol=0)


def test_visual_multi_token_gear_metadata_is_relocatable(tmp_path, monkeypatch):
    config = XHPaddleOCRVLVisualConfig(
        model_name="paddleocr_vl_visual",
        hf_model=str(tmp_path),
        max_size_h=896,
        max_size_w=896,
        patch_size=14,
        spatial_merge_size=2,
        image_token_gears=[1, 2, 4],
        image_token_capacity=4,
    )
    model = XHPaddleOCRVLVisualModel(config)

    def export_profile(self, output_dir, profile):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        graph = Path(output_dir) / f"{self.config.model_name}.onnx"
        graph.write_text("graph\n")
        meta = self._metadata_for_profile(output_dir, profile)
        meta.hmonnx = str(graph)
        return meta

    monkeypatch.setattr(XHPaddleOCRVLVisualModel, "_export_single_profile_hmonnx", export_profile)

    metadata = model.export_hmonnx(str(tmp_path / "export"))
    metadata.hmonnx = Path(metadata.hmonnx).relative_to(tmp_path / "export").as_posix()
    meta_file = tmp_path / "export" / "golden_meta_info.json"
    meta_file.write_text(json.dumps({"visual_config": metadata.to_dict()}), encoding="utf-8")

    relocated = (tmp_path / "export").rename(tmp_path / "relocated")
    relocated_meta = relocated / "golden_meta_info.json"
    loaded = VLLMModelMeta.from_dict({
        "hf_config": "hf_config",
        "quant_embedding": "quant_embedding.pt",
        "prefill_hmonnx": "prefill/model.onnx",
        "decode_hmonnx": "decode/model.onnx",
        "visual_config": json.loads(relocated_meta.read_text())["visual_config"],
        "_meta_path_": str(relocated_meta),
    }).visual_config

    assert metadata.routing_policy == "smallest_fit"
    assert [gear["image_token_capacity"] for gear in metadata.gears] == [1, 2, 4]
    assert metadata.hmonnx == "visual_m4/paddleocr_vl_visual.onnx"
    assert metadata.gear_manifest == "visual_gears.json"
    manifest = json.loads((relocated / "visual_gears.json").read_text())
    assert manifest["routing_policy"] == "smallest_fit"
    assert manifest["gears"] == json.loads(relocated_meta.read_text())["visual_config"]["gears"]
    assert [Path(gear.hmonnx) for gear in loaded.gears] == [
        relocated / "visual_m1/paddleocr_vl_visual.onnx",
        relocated / "visual_m2/paddleocr_vl_visual.onnx",
        relocated / "visual_m4/paddleocr_vl_visual.onnx",
    ]


def test_visual_multi_profile_export_reuses_supplied_native_owner_and_isolates_frontend_cache(tmp_path, monkeypatch):
    native = _tiny_paddleocr_model()
    config = XHPaddleOCRVLVisualConfig(
        model_name="paddleocr_vl_visual",
        hf_model=str(tmp_path / "missing_gptq_checkpoint"),
        max_size_h=28,
        max_size_w=28,
        patch_size=14,
        spatial_merge_size=2,
        image_token_gears=[1, 2],
        image_token_capacity=2,
    )
    model = XHPaddleOCRVLVisualModel(config)
    model.init_wrap_model(native)
    exports = []

    def export_without_hmonnx(self, output_dir):
        exports.append({
            "output_dir": Path(output_dir).name,
            "work_dir": Path(self.config.work_dir).name,
            "dummy_shape": tuple(self.get_dummy_inputs()["pixel_values"].shape),
            "native_root_id": id(self._native_model_root),
        })
        graph = Path(output_dir) / "graph.onnx"
        graph.parent.mkdir(parents=True, exist_ok=True)
        graph.write_text("graph\n")
        return graph

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.paddleocr_vl.visual.BaseVisionModel._export_hmonnx",
        export_without_hmonnx,
    )

    original_work_dir = model.config.work_dir
    original_native_root = id(native.model)
    metadata = model.export_hmonnx(str(tmp_path / "export" / "visual"))

    assert model.config.work_dir == original_work_dir
    assert [item["output_dir"] for item in exports] == ["visual_m1", "visual_m2"]
    assert [item["work_dir"] for item in exports] == ["visual_m1", "visual_m2"]
    assert [item["dummy_shape"] for item in exports] == [(1, 4, 3, 14, 14), (1, 8, 3, 14, 14)]
    assert [item["native_root_id"] for item in exports] == [original_native_root, original_native_root]
    assert [gear["hmonnx"] for gear in metadata.gears] == [
        "visual/visual_m1/graph.onnx",
        "visual/visual_m2/graph.onnx",
    ]


def test_visual_single_profile_export_rebuilds_when_explicit_profile_differs_from_compiled_default(
    tmp_path,
    monkeypatch,
):
    native = _tiny_paddleocr_model()
    config = XHPaddleOCRVLVisualConfig(
        model_name="paddleocr_vl_visual",
        hf_model=str(tmp_path / "missing_gptq_checkpoint"),
        max_size_h=56,
        max_size_w=56,
        patch_size=14,
        spatial_merge_size=2,
        image_token_gears=[2],
        image_token_capacity=4,
    )
    model = XHPaddleOCRVLVisualModel(config)
    model.init_wrap_model(native)
    model._state = LLMModelState.QUANTED_ALIGNED
    model._quanted_model = object()
    observed = {}

    def export_without_hmonnx(self, output_dir):
        observed["state_at_export"] = self._state
        observed["dummy_shape"] = tuple(self.get_dummy_inputs()["pixel_values"].shape)
        observed["native_root_id"] = id(self._native_model_root)
        graph = Path(output_dir) / "graph.onnx"
        graph.parent.mkdir(parents=True, exist_ok=True)
        graph.write_text("graph\n")
        return graph

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.paddleocr_vl.visual.BaseVisionModel._export_hmonnx",
        export_without_hmonnx,
    )

    metadata = model.export_hmonnx(str(tmp_path / "export"))

    assert observed == {
        "state_at_export": LLMModelState.NONE,
        "dummy_shape": (1, 8, 3, 14, 14),
        "native_root_id": id(native.model),
    }
    assert metadata.image_token_capacity == 2
    assert metadata.input_shape == [1, 8, 3, 14, 14]
    assert metadata.output_shape == [2, 24]


def test_visual_profile_rebuild_uses_snapshot_after_owner_root_loses_visual_modules(tmp_path, monkeypatch):
    native = _tiny_paddleocr_model()
    config = XHPaddleOCRVLVisualConfig(
        model_name="paddleocr_vl_visual",
        hf_model=str(tmp_path / "missing_gptq_checkpoint"),
        max_size_h=28,
        max_size_w=28,
        patch_size=14,
        spatial_merge_size=2,
        image_token_gears=[1, 2],
        image_token_capacity=2,
    )
    model = XHPaddleOCRVLVisualModel(config)
    model.init_wrap_model(native)
    snapshot_visual = model._native_visual_source.visual
    snapshot_projector = model._native_visual_source.projector
    delattr(native.model, "visual")
    delattr(native.model, "projector")
    model._state = LLMModelState.QUANTED_ALIGNED
    model._quanted_model = object()
    exports = []

    def export_without_hmonnx(self, output_dir):
        exports.append({
            "dummy_shape": tuple(self.get_dummy_inputs()["pixel_values"].shape),
            "source_visual_id": id(self._native_visual_source.visual),
            "source_projector_id": id(self._native_visual_source.projector),
            "root_has_visual": hasattr(self._native_model_root, "visual"),
        })
        graph = Path(output_dir) / "graph.onnx"
        graph.parent.mkdir(parents=True, exist_ok=True)
        graph.write_text("graph\n")
        return graph

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.paddleocr_vl.visual.BaseVisionModel._export_hmonnx",
        export_without_hmonnx,
    )

    metadata = model.export_hmonnx(str(tmp_path / "export" / "visual"))

    assert [item["dummy_shape"] for item in exports] == [(1, 4, 3, 14, 14), (1, 8, 3, 14, 14)]
    assert {item["source_visual_id"] for item in exports} == {id(snapshot_visual)}
    assert {item["source_projector_id"] for item in exports} == {id(snapshot_projector)}
    assert [item["root_has_visual"] for item in exports] == [False, False]
    assert [gear["image_token_capacity"] for gear in metadata.gears] == [1, 2]



def test_live_visual_supported_grids_uses_compiled_grid_not_configured_profiles(tmp_path):
    config = XHPaddleOCRVLVisualConfig(
        model_name="paddleocr_vl_visual",
        hf_model=str(tmp_path),
        max_size_h=56,
        max_size_w=56,
        patch_size=14,
        spatial_merge_size=2,
    )
    visual = SimpleNamespace(
        config=config,
        _wrap_model=SimpleNamespace(grid_thw=torch.tensor([[1, 4, 4]])),
        _inference_model=None,
    )
    visual_meta = VisualModelMeta()
    visual_meta.patch_size = 14
    visual_meta.max_size_h = 56
    visual_meta.max_size_w = 56
    visual_meta.resolution_profiles = [
        {"height": 56, "width": 56, "name": "compiled"},
        {"height": 28, "width": 56, "name": "uncompiled"},
    ]

    assert _supported_visual_grids(visual, visual_meta) == {(1, 4, 4)}


def test_live_generate_rejects_overflow_grid_and_bad_pixel_shape_before_visual_call(tmp_path):
    config = XHPaddleOCRVLVisualConfig(
        model_name="paddleocr_vl_visual",
        hf_model=str(tmp_path),
        max_size_h=56,
        max_size_w=56,
        patch_size=14,
        spatial_merge_size=2,
        image_token_gears=[4],
        image_token_capacity=4,
    )

    class FailingVisual:
        def __init__(self):
            self.config = config
            self._wrap_model = SimpleNamespace(grid_thw=torch.tensor([[1, 4, 4]]))
            self._inference_model = None

        def forward(self, pixel_values):
            raise AssertionError(f"visual graph should not run for invalid input {tuple(pixel_values.shape)}")

    class FakePaddleOCR(PaddleOCRGenerationMixin):
        dtype = torch.float16
        device = torch.device("cpu")
        paddleocr_hf_config = SimpleNamespace(image_token_id=5)

        def prepare_for_inference(self):
            pass

        def get_input_embeddings(self):
            return torch.nn.Embedding(8, 4)

    model = FakePaddleOCR()
    model.config = SimpleNamespace(context_max_length=8, prefill_chunk_length=4, visual_config=config)
    model.visual = FailingVisual()
    input_ids = torch.tensor([[5]], dtype=torch.long)

    with pytest.raises(ValueError, match="exceeds the largest configured gear"):
        model.generate(
            input_ids,
            pixel_values=torch.zeros(1, 24, 3, 14, 14),
            image_grid_thw=torch.tensor([[1, 4, 6]]),
            max_new_tokens=1,
        )
    with pytest.raises(ValueError, match="pixel_values shape must match image_grid_thw"):
        model.generate(
            input_ids,
            pixel_values=torch.zeros(1, 15, 3, 14, 14),
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            max_new_tokens=1,
        )


@pytest.mark.parametrize("grid", REQUIRED_PROFILE_GRIDS)
def test_visual_router_selects_each_required_grid_and_validates_pixel_shape(monkeypatch, grid):
    calls = []

    class FakeHMONNXModel:
        def __init__(self, path, *, enable_golden, device_map):
            self.path = path
            self.device = torch.device("cpu")
            self.enable_golden = enable_golden
            self.device_map = device_map

        def to(self, device):
            self.device = torch.device(device)
            calls.append((f"{self.path}:to", str(self.device)))
            return self

        def forward(self, pixel_values):
            calls.append((self.path, tuple(pixel_values.shape)))
            return torch.full((1, 1), float(len(calls)))

        def update_step(self):
            calls.append((f"{self.path}:step", ()))

    monkeypatch.setattr("xhmodel_merak.xh_llm.models.paddleocr_vl.runtime.HMONNXModel", FakeHMONNXModel)
    visual_meta = VisualModelMeta()
    visual_meta.patch_size = 14
    visual_meta.gears = [
        SimpleNamespace(image_grid_thw=candidate, hmonnx=f"grid_{candidate[1]}_{candidate[2]}.onnx")
        for candidate in REQUIRED_PROFILE_GRIDS
    ]

    router = PaddleOCRVLVisualHMONNXRouter(visual_meta, device_map=["cpu"], enable_golden=True)
    patch_count = grid[0] * grid[1] * grid[2]
    output = router.forward(torch.zeros(1, patch_count, 3, 14, 14), torch.tensor([grid]))
    router.enable_golden = False
    router.update_step()

    torch.testing.assert_close(output, torch.ones(1, 1))
    assert calls[0] == (f"grid_{grid[1]}_{grid[2]}.onnx", (1, patch_count, 3, 14, 14))
    assert all(not model.enable_golden for model in router.models.values())
    assert len([call for call in calls if call[0].endswith(":step")]) == len(REQUIRED_PROFILE_GRIDS)
    with pytest.raises(ValueError, match="pixel_values shape must match image_grid_thw"):
        router.forward(torch.zeros(1, patch_count - 1, 3, 14, 14), torch.tensor([grid]))
    with pytest.raises(ValueError, match="pixel_values shape must match image_grid_thw"):
        router.forward(torch.zeros(1, patch_count, 1, 14, 14), torch.tensor([grid]))
    with pytest.raises(ValueError, match="pixel_values shape must match image_grid_thw"):
        router.forward(torch.zeros(1, patch_count, 3, 28, 28), torch.tensor([grid]))
    with pytest.raises(ValueError, match="Unsupported PaddleOCR image_grid_thw"):
        router.forward(torch.zeros(1, 4, 3, 14, 14), torch.tensor([[1, 2, 2]]))


def test_paddleocr_hmonnx_parent_device_move_updates_visual_router():
    class MoveRecorder:
        def __init__(self, name):
            self.name = name
            self.devices = []

        def to(self, device):
            self.devices.append(torch.device(device))
            return self

    runtime = XHPaddleOCRVLHMONNXModel.__new__(XHPaddleOCRVLHMONNXModel)
    prefill = MoveRecorder("prefill")
    decode = MoveRecorder("decode")
    visual = MoveRecorder("visual")
    embed_tokens = MoveRecorder("embed")
    kvcache = MoveRecorder("cache")
    runtime._models = {"prefill": prefill, "decode": decode}
    runtime._device = torch.device("cpu")
    runtime.embed_tokens = embed_tokens
    runtime._kvcache_mixin = kvcache
    runtime.visual = visual

    result = runtime._set_device(torch.device("cpu"))

    assert result is runtime
    assert prefill.devices == [torch.device("cpu")]
    assert decode.devices == [torch.device("cpu")]
    assert embed_tokens.devices == [torch.device("cpu")]
    assert kvcache.devices == [torch.device("cpu")]
    assert visual.devices == [torch.device("cpu")]
