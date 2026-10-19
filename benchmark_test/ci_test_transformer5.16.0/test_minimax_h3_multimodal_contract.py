"""CPU contracts for MiniMax-H3 Qwen3-VL text and vision HMONNX adapters."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import Qwen3VLVisionConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from xhmodel_merak.xh_other_model.models.minimax_h3 import (
    MiniMaxH3VisionEncoderWrapper,
    build_vision_encoder_graph_inputs,
)
from xhmodel_merak.xh_other_model.models.minimax_h3.components_hmonnx import (
    _HMONNXTextLanguageModel,
    _HMONNXTransformerModule,
    _HMONNXVisionModel,
    _release_pipeline_components,
    install_hmonnx_components,
)
from xhmodel_merak.xh_other_model.models.minimax_h3.minimax_h3_converter import expand_minimax_h3_components
from xhmodel_merak.xh_other_model.models.minimax_h3.workflow import MiniMaxH3Workflow
from xhmodel_merak.xh_other_model.workflows.result import QuantResult


class _RotaryEmbedding(nn.Module):
    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor):
        del position_ids
        return torch.zeros_like(hidden_states), torch.zeros_like(hidden_states)


class _TextFallback(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace()
        self.rotary_emb = _RotaryEmbedding()
        self.embed_tokens = nn.Embedding(8, 4)

    def forward(self, **kwargs):
        raise AssertionError(f"unexpected text fallback: {sorted(kwargs)}")


class _RecordingRuntime:
    def __init__(self, expected_shapes, outputs=None):
        self.expected_shapes = expected_shapes
        self.outputs = outputs
        self.inputs = None

    def supports(self, *inputs: torch.Tensor) -> bool:
        return tuple(tuple(tensor.shape) for tensor in inputs) == self.expected_shapes

    def __call__(self, *inputs: torch.Tensor):
        self.inputs = inputs
        return self.outputs if self.outputs is not None else inputs[0]


def _tiny_vision_config() -> Qwen3VLVisionConfig:
    return Qwen3VLVisionConfig(
        depth=3,
        hidden_size=16,
        intermediate_size=32,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        out_hidden_size=12,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0, 1, 2],
    )


class _VisionFallback(nn.Module):
    def __init__(self, source: Qwen3VLVisionModel | None = None):
        super().__init__()
        if source is None:
            source = Qwen3VLVisionModel(_tiny_vision_config()).eval()
        self.config = source.config
        self.spatial_merge_size = int(source.spatial_merge_size)
        self.num_grid_per_side = source.num_grid_per_side
        self.interpolation_mode = source.interpolation_mode
        self.interpolation_align_corners = source.interpolation_align_corners
        self.pos_embed = source.pos_embed
        self.rotary_pos_emb = source.rotary_pos_emb
        self._dtype = torch.float16
        self.called = False

    @property
    def dtype(self):
        return self._dtype

    def forward(self, pixel_values, grid_thw=None, return_dict=True, **kwargs):
        del pixel_values, grid_thw, return_dict, kwargs
        self.called = True
        return "fallback"


def _vision_expected_shapes(max_patches: int = 8, patch_dim: int = 24, rope_dim: int = 4, hidden: int = 16):
    return (
        (max_patches, patch_dim),
        (max_patches, hidden),
        (max_patches, rope_dim),
        (max_patches, rope_dim),
        (1, 1, 1, max_patches),
    )


def test_text_runtime_densifies_three_deepstack_inputs_at_fixed_length():
    # Unit-test densify/pad against an arbitrary fixed capacity from export meta.
    expected_shapes = (
        (1, 2048, 4),
        (1, 2048, 4),
        (1, 2048, 4),
        (1, 1, 2048, 2048),
        (1, 2048, 4),
        (1, 2048, 4),
        (1, 2048, 4),
    )
    runtime = _RecordingRuntime(expected_shapes)
    model = _HMONNXTextLanguageModel(runtime, _TextFallback())
    inputs_embeds = torch.arange(20, dtype=torch.float32).reshape(1, 5, 4)
    attention_mask = torch.ones(1, 5, dtype=torch.long)
    visual_pos_masks = torch.tensor([[False, True, False, True, False]])
    deepstack = [torch.full((2, 4), float(index + 1)) for index in range(3)]

    output = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack,
    )

    assert runtime.inputs is not None
    # HMONNX text keeps the fixed export length; pad rows are masked later.
    assert output.last_hidden_state.shape == (1, 2048, 4)
    torch.testing.assert_close(output.last_hidden_state[:, :5], inputs_embeds)
    assert torch.count_nonzero(output.last_hidden_state[:, 5:]) == 0
    for index, dense in enumerate(runtime.inputs[4:]):
        torch.testing.assert_close(dense[:, :5][visual_pos_masks], deepstack[index])
        assert torch.count_nonzero(dense[:, 5:]) == 0


def test_text_hmonnx_wrapper_exposes_input_embeddings_for_qwen3vl_model():
    fallback = _TextFallback()
    original_embeddings = fallback.embed_tokens
    wrapper = _HMONNXTextLanguageModel(_RecordingRuntime(((1, 2048, 4),) * 7), fallback)
    parent = SimpleNamespace(language_model=wrapper)

    # Mimic transformers EmbeddingAccessMixin step 4 for Qwen3VLModel.
    assert hasattr(parent.language_model, "get_input_embeddings")
    assert wrapper.get_input_embeddings() is original_embeddings
    assert wrapper.embed_tokens is original_embeddings

    replacement = nn.Embedding(8, 4)
    wrapper.set_input_embeddings(replacement)
    assert wrapper.get_input_embeddings() is replacement

    # Demo install drops the heavy decoder body but must keep host embeddings.
    wrapper._release_fallback()
    assert wrapper.fallback is None
    assert wrapper.get_input_embeddings() is replacement


def test_vision_runtime_pads_masks_and_crops_valid_tokens():
    torch.manual_seed(3)
    max_patches = 8
    max_tokens = 2
    valid_tokens = 1
    image_embeds = torch.randn(max_tokens, 8)
    deepstack = [torch.randn(max_tokens, 8) for _ in range(3)]
    runtime = _RecordingRuntime(_vision_expected_shapes(max_patches), (image_embeds, *deepstack))
    fallback = _VisionFallback()
    model = _HMONNXVisionModel(
        runtime,
        fallback,
        max_patches=max_patches,
        example_image_grid_thw=torch.tensor([[1, 2, 2]]),
    )
    pixel_values = torch.randn(4, 24)

    output = model(pixel_values, grid_thw=torch.tensor([[1, 2, 2]]))

    assert runtime.inputs is not None
    assert runtime.inputs[0].shape == (max_patches, 24)
    assert torch.equal(runtime.inputs[0][:4], pixel_values)
    assert torch.count_nonzero(runtime.inputs[0][4:]) == 0
    assert runtime.inputs[4].shape == (1, 1, 1, max_patches)
    assert torch.all(runtime.inputs[4][..., :4] == 0)
    assert torch.all(runtime.inputs[4][..., 4:] < 0)
    torch.testing.assert_close(output.pooler_output, image_embeds[:valid_tokens])
    assert len(output.deepstack_features) == 3
    for actual, expected in zip(output.deepstack_features, deepstack, strict=True):
        torch.testing.assert_close(actual, expected[:valid_tokens])
    assert not fallback.called
    assert model.fallback is None
    assert model.geometry_source is not None


def test_static_vision_wrapper_matches_native_qwen3_vl_on_valid_prefix():
    torch.manual_seed(17)
    config = _tiny_vision_config()
    native = Qwen3VLVisionModel(config).eval()
    wrapped_source = Qwen3VLVisionModel(config).eval()
    wrapped_source.load_state_dict(native.state_dict())
    grid = torch.tensor([[1, 2, 2]])
    pixel_values = torch.randn(4, 24)
    wrapper = MiniMaxH3VisionEncoderWrapper(wrapped_source, max_patches=8).eval()
    graph_inputs, geometry = build_vision_encoder_graph_inputs(
        wrapper.visual,
        pixel_values,
        grid,
        max_patches=8,
    )

    assert wrapper.max_patches == 8
    assert wrapper.max_image_tokens == 2
    assert type(wrapper.visual.blocks[0].attn).__name__ == "_Qwen3VLVisionAttention"
    assert hasattr(wrapper.visual.blocks[0].attn, "q_proj")
    assert hasattr(wrapper.visual.blocks[0].attn, "k_proj")
    assert hasattr(wrapper.visual.blocks[0].attn, "v_proj")
    assert not hasattr(wrapper.visual.blocks[0].attn, "qkv")
    assert type(wrapper.visual.merger).__name__ == "_Qwen3VLVisionPatchMerger"

    with torch.no_grad():
        expected = native(pixel_values, grid_thw=grid)
        actual = wrapper(*graph_inputs)

    valid_tokens = int(geometry["valid_image_tokens"])
    torch.testing.assert_close(actual[0][:valid_tokens], expected.pooler_output)
    for got, want in zip(actual[1:], expected.deepstack_features, strict=True):
        torch.testing.assert_close(got[:valid_tokens], want)


def test_vision_runtime_falls_back_when_grid_exceeds_capacity():
    runtime = _RecordingRuntime(_vision_expected_shapes(8), tuple(torch.randn(2, 8) for _ in range(4)))
    fallback = _VisionFallback()
    model = _HMONNXVisionModel(runtime, fallback, max_patches=8)

    # 1x4x4 = 16 patches > capacity 8
    assert model(torch.randn(16, 24), grid_thw=torch.tensor([[1, 4, 4]])) == "fallback"
    assert fallback.called
    assert runtime.inputs is None


def test_vision_runtime_rejects_wrong_full_capacity_token_count():
    runtime = _RecordingRuntime(_vision_expected_shapes(8), tuple(torch.randn(1, 8) for _ in range(4)))
    model = _HMONNXVisionModel(runtime, _VisionFallback(), max_patches=8)

    with pytest.raises(RuntimeError, match="must return 2 image tokens"):
        model(torch.randn(4, 24), grid_thw=torch.tensor([[1, 2, 2]]))


@pytest.mark.parametrize("component", ("text_encoder", "vision_encoder"))
def test_expand_minimax_h3_components_binds_text_and_vision(component):
    assert expand_minimax_h3_components({component}) == {"text_encoder", "vision_encoder"}
    assert expand_minimax_h3_components({"vae_encoder", component}) == {
        "vae_encoder",
        "text_encoder",
        "vision_encoder",
    }


def test_expand_minimax_h3_components_binds_transformer_to_text_and_vision():
    assert expand_minimax_h3_components({"transformer"}) == {
        "transformer",
        "text_encoder",
        "vision_encoder",
    }
    assert expand_minimax_h3_components({"vae_encoder", "transformer"}) == {
        "vae_encoder",
        "transformer",
        "text_encoder",
        "vision_encoder",
    }


def test_expand_minimax_h3_components_keeps_transformer_only_for_export():
    assert expand_minimax_h3_components({"transformer"}, export=True) == {"transformer"}
    assert expand_minimax_h3_components({"vae_encoder", "transformer"}, export=True) == {
        "vae_encoder",
        "transformer",
    }
    # text/vision still bind to each other even during export.
    assert expand_minimax_h3_components({"text_encoder"}, export=True) == {
        "text_encoder",
        "vision_encoder",
    }


def test_resolve_workflow_components_keeps_transformer_only_for_export():
    from examples_merak.aigc.minimaxh3.minimax_h3_workflow import resolve_workflow_components

    assert resolve_workflow_components(["transformer"]) == ["transformer"]
    assert resolve_workflow_components(["text_encoder"]) == ["vision_encoder", "text_encoder"]


def _write_text_vision_export(tmp_path: Path) -> Path:
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "text_encoder.hmonnx").write_bytes(b"text")
    (export_dir / "vision_encoder.hmonnx").write_bytes(b"vision")
    meta = {
        "components": {
            "text_encoder": {
                "hmonnx_file": "text_encoder.hmonnx",
                "input_names": [
                    "hidden_states",
                    "rotary_cos",
                    "rotary_sin",
                    "attention_mask",
                    "deepstack_visual_embed_0",
                    "deepstack_visual_embed_1",
                    "deepstack_visual_embed_2",
                ],
                "inputs": [{"shape": [1, 2048, 4]} for _ in range(7)],
            },
            "vision_encoder": {
                "hmonnx_file": "vision_encoder.hmonnx",
                "input_names": [
                    "pixel_values",
                    "position_embeddings",
                    "rotary_cos",
                    "rotary_sin",
                    "attention_mask",
                ],
                "inputs": [
                    {"shape": [8, 24]},
                    {"shape": [8, 16]},
                    {"shape": [8, 4]},
                    {"shape": [8, 4]},
                    {"shape": [1, 1, 1, 8]},
                ],
                "image_grid_thw": [[1, 2, 2]],
                "vision_max_patches": 8,
                "vision_max_image_tokens": 2,
                "attention_mask_format": "additive_key_padding_bias",
            },
        }
    }
    (export_dir / "export_meta_info.json").write_text(json.dumps(meta), encoding="utf-8")
    return export_dir


def test_install_hmonnx_components_expands_text_or_vision(tmp_path, monkeypatch):
    export_dir = _write_text_vision_export(tmp_path)

    class _FakeLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()
            self.rotary_emb = nn.Identity()
            self.embed_tokens = nn.Embedding(8, 4)

    class _FakeVisionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()
            self.spatial_merge_size = 2
            self.dtype = torch.float32
            self.num_grid_per_side = 4
            self.interpolation_mode = "bilinear"
            self.interpolation_align_corners = False
            self.pos_embed = nn.Embedding(16, 16)
            self.rotary_pos_emb = nn.Identity()

    language_model = _FakeLanguageModel()
    visual = _FakeVisionModel()
    pipe = SimpleNamespace(
        text_encoder=SimpleNamespace(model=SimpleNamespace(language_model=language_model, visual=visual)),
        vae=SimpleNamespace(),
        audio_vae=SimpleNamespace(),
        transformer=nn.Linear(2, 2),
        _modules={},
    )

    class _FakeRuntime:
        def __init__(self, path, device, shapes, output_list=False, **kwargs):
            del path, device, shapes, output_list, kwargs

    monkeypatch.setattr(
        "xhmodel_merak.xh_other_model.models.minimax_h3.components_hmonnx._LazyHMONNXRuntime",
        _FakeRuntime,
    )
    monkeypatch.setattr(
        "xhmodel_merak.xh_other_model.models.minimax_h3.components_hmonnx._install_pipeline_cleanup",
        lambda pipe: None,
    )

    installed = install_hmonnx_components(pipe, export_dir, torch.device("cpu"), ("vision_encoder",))
    assert installed == ("vision_encoder", "text_encoder")
    assert isinstance(pipe.text_encoder.model.visual, _HMONNXVisionModel)
    assert isinstance(pipe.text_encoder.model.language_model, _HMONNXTextLanguageModel)
    assert pipe.text_encoder.model.visual.fallback is None
    assert pipe.text_encoder.model.language_model.fallback is None
    assert pipe.text_encoder.model.language_model.get_input_embeddings() is language_model.embed_tokens
    assert pipe.text_encoder.model.visual.geometry_source is not visual


def _write_transformer_text_vision_export(tmp_path: Path) -> Path:
    export_dir = _write_text_vision_export(tmp_path)
    (export_dir / "transformer.hmonnx").write_bytes(b"transformer")
    meta = json.loads((export_dir / "export_meta_info.json").read_text(encoding="utf-8"))
    meta["components"]["transformer"] = {
        "hmonnx_file": "transformer.hmonnx",
        "input_names": [
            "hidden_states",
            "audio_hidden_states",
            "encoder_hidden_states",
            "step_index",
            "timestep_indices",
            "adaln_indices",
            "rotary_cos",
            "rotary_sin",
            "video_indices",
            "audio_indices",
            "pack_gather_indices",
            "encoder_attention_mask",
            "packed_attention_mask",
        ],
        "inputs": [{"shape": [1, 2, 2]} for _ in range(13)],
    }
    (export_dir / "export_meta_info.json").write_text(json.dumps(meta), encoding="utf-8")
    return export_dir


def test_release_pipeline_components_preserves_hmonnx_transformer_runtime(tmp_path, monkeypatch):
    export_dir = _write_transformer_text_vision_export(tmp_path)

    class _FakeLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()
            self.rotary_emb = nn.Identity()
            self.embed_tokens = nn.Embedding(8, 4)

    class _FakeVisionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()
            self.spatial_merge_size = 2
            self.dtype = torch.float32
            self.num_grid_per_side = 4
            self.interpolation_mode = "bilinear"
            self.interpolation_align_corners = False
            self.pos_embed = nn.Embedding(16, 16)
            self.rotary_pos_emb = nn.Identity()

    class _FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(patch_size=(1, 1, 1))
            self.dtype = torch.float16
            self.time_proj = nn.Identity()
            self.time_embedder = nn.Identity()
            self.rope = nn.Identity()

    language_model = _FakeLanguageModel()
    visual = _FakeVisionModel()
    transformer = _FakeTransformer()
    pipe = SimpleNamespace(
        text_encoder=SimpleNamespace(model=SimpleNamespace(language_model=language_model, visual=visual)),
        vae=nn.Linear(2, 2),
        audio_vae=nn.Linear(2, 2),
        transformer=transformer,
        _modules={},
    )

    class _FakeRuntime:
        def __init__(self, path, device, shapes, output_list=False, **kwargs):
            del path, device, shapes, output_list, kwargs
            self.session = object()

    monkeypatch.setattr(
        "xhmodel_merak.xh_other_model.models.minimax_h3.components_hmonnx._LazyHMONNXRuntime",
        _FakeRuntime,
    )
    monkeypatch.setattr(
        "xhmodel_merak.xh_other_model.models.minimax_h3.components_hmonnx._install_pipeline_cleanup",
        lambda pipe: None,
    )

    installed = install_hmonnx_components(pipe, export_dir, torch.device("cpu"), ("transformer",))
    assert installed == ("transformer", "vision_encoder", "text_encoder")
    assert isinstance(pipe.transformer, _HMONNXTransformerModule)
    assert pipe.transformer.runtime is not None
    assert isinstance(pipe.text_encoder.model.language_model, _HMONNXTextLanguageModel)
    assert isinstance(pipe.text_encoder.model.visual, _HMONNXVisionModel)

    kept_transformer = pipe.transformer
    kept_text_encoder = pipe.text_encoder
    kept_language_runtime = pipe.text_encoder.model.language_model.runtime
    kept_vision_runtime = pipe.text_encoder.model.visual.runtime
    _release_pipeline_components(pipe)

    assert pipe.transformer is kept_transformer
    assert pipe.transformer.runtime is kept_transformer.runtime
    assert pipe.text_encoder is kept_text_encoder
    assert pipe.text_encoder.model.language_model.runtime is kept_language_runtime
    assert pipe.text_encoder.model.visual.runtime is kept_vision_runtime
    assert pipe.vae is None
    assert pipe.audio_vae is None


def test_install_hmonnx_components_requires_paired_text_vision_export(tmp_path):
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    (export_dir / "vision_encoder.hmonnx").write_bytes(b"vision")
    meta = {
        "components": {
            "vision_encoder": {
                "hmonnx_file": "vision_encoder.hmonnx",
                "input_names": [
                    "pixel_values",
                    "position_embeddings",
                    "rotary_cos",
                    "rotary_sin",
                    "attention_mask",
                ],
                "inputs": [
                    {"shape": [8, 24]},
                    {"shape": [8, 16]},
                    {"shape": [8, 4]},
                    {"shape": [8, 4]},
                    {"shape": [1, 1, 1, 8]},
                ],
                "image_grid_thw": [[1, 2, 2]],
                "vision_max_patches": 8,
            }
        }
    }
    (export_dir / "export_meta_info.json").write_text(json.dumps(meta), encoding="utf-8")
    pipe = SimpleNamespace(
        text_encoder=SimpleNamespace(model=SimpleNamespace(language_model=nn.Linear(4, 4), visual=nn.Linear(3, 3))),
        vae=SimpleNamespace(),
        audio_vae=SimpleNamespace(),
        transformer=nn.Linear(2, 2),
        _modules={},
    )

    with pytest.raises(ValueError, match="missing requested MiniMax-H3 HMONNX components: \\['text_encoder'\\]"):
        install_hmonnx_components(pipe, export_dir, torch.device("cpu"), ("vision_encoder",))


def test_minimax_h3_workflow_merges_existing_export_meta(tmp_path, monkeypatch):
    config_path = Path("configs_merak/workflows/xh2a/other_models/minimax_h3/minimax_h3.yaml").resolve()
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    existing = {
        "create_time": "old",
        "config": "old.yaml",
        "model_type": "XHMiniMaxH3Model",
        "target_device": "XH2a",
        "components": {
            "text_encoder": {
                "hmonnx_file": "text_encoder/minimax_h3_text_encoder-XH2a-w8a8_ssfp.onnx",
                "input_names": ["hidden_states"],
            }
        },
    }
    (export_dir / "export_meta_info.json").write_text(json.dumps(existing), encoding="utf-8")

    class _FakeConverter:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def export(self, output_dir, components):
            del output_dir, components
            return {
                "vision_encoder": {
                    "hmonnx_file": "vision_encoder/minimax_h3_vision_encoder-XH2a-w8a8_ssfp.onnx",
                    "input_names": [
                        "pixel_values",
                        "position_embeddings",
                        "rotary_cos",
                        "rotary_sin",
                        "attention_mask",
                    ],
                    "vision_max_patches": 4352,
                }
            }

    monkeypatch.setattr(
        "xhmodel_merak.xh_other_model.models.minimax_h3.workflow.MiniMaxH3Converter",
        _FakeConverter,
    )
    workflow = MiniMaxH3Workflow(model_dir=str(tmp_path / "model"), config_path=str(config_path))
    result = workflow.export(
        QuantResult(raw_model_dir=str(tmp_path / "model"), skipped=True),
        str(export_dir),
        "cpu",
        {"export.minimax_h3.components": ["vision_encoder"]},
    )
    meta = json.loads(Path(result.work_dir, "export_meta_info.json").read_text(encoding="utf-8"))
    assert "text_encoder" in meta["components"]
    assert "vision_encoder" in meta["components"]
    assert meta["components"]["text_encoder"]["hmonnx_file"].endswith(
        "text_encoder/minimax_h3_text_encoder-XH2a-w8a8_ssfp.onnx"
    )


def _tiny_qwen3vl_config():
    from transformers import Qwen3VLConfig

    return Qwen3VLConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 4,
            "max_position_embeddings": 64,
            "rope_theta": 10000.0,
        },
        vision_config={
            "depth": 2,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 12,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0, 1],
        },
    )


def test_export_and_build_hmonnx_text_encoder_host(tmp_path):
    from safetensors.torch import load_file

    from xhmodel_merak.xh_other_model.models.minimax_h3.text_embedding_host import (
        build_hmonnx_text_encoder_host,
        export_text_embedding_weights,
        export_vision_geometry_weights,
        install_hmonnx_text_encoder_host,
        text_embedding_path,
        vision_geometry_path,
    )

    model_dir = tmp_path / "model"
    text_encoder_dir = model_dir / "text_encoder"
    text_encoder_dir.mkdir(parents=True)
    _tiny_qwen3vl_config().save_pretrained(text_encoder_dir)

    export_dir = tmp_path / "export"
    language_model = SimpleNamespace(embed_tokens=nn.Embedding(32, 16))
    visual = SimpleNamespace(pos_embed=nn.Embedding(16, 16))
    with torch.no_grad():
        language_model.embed_tokens.weight.copy_(torch.arange(32 * 16, dtype=torch.float32).reshape(32, 16))
        visual.pos_embed.weight.copy_(torch.arange(16 * 16, dtype=torch.float32).reshape(16, 16) + 1)

    text_meta = export_text_embedding_weights(language_model, export_dir)
    vision_meta = export_vision_geometry_weights(visual, export_dir)
    assert text_meta["text_embeddings_file"] == "text_encoder/text_embeddings.safetensors"
    assert vision_meta["vision_geometry_file"] == "vision_encoder/vision_geometry.safetensors"
    assert text_embedding_path(export_dir).is_file()
    assert vision_geometry_path(export_dir).is_file()
    assert load_file(str(text_embedding_path(export_dir)))["embed_tokens.weight"].shape == (32, 16)

    host = build_hmonnx_text_encoder_host(model_dir, export_dir, dtype=torch.float16)
    assert len(host.model.language_model.layers) == 0
    assert len(host.model.visual.blocks) == 0
    assert host.model.language_model.embed_tokens.weight.dtype == torch.float16
    assert host.model.visual.pos_embed.weight.dtype == torch.float16
    torch.testing.assert_close(
        host.model.language_model.embed_tokens.weight.float(),
        language_model.embed_tokens.weight.float(),
    )
    torch.testing.assert_close(
        host.model.visual.pos_embed.weight.float(),
        visual.pos_embed.weight.float(),
    )
    embeds = host.get_input_embeddings()(torch.tensor([[1, 2, 3]]))
    assert embeds.shape == (1, 3, 16)
    assert embeds.dtype == torch.float16

    pipe = SimpleNamespace(_modules={"text_encoder": None})
    installed = install_hmonnx_text_encoder_host(pipe, model_dir, export_dir, dtype=torch.float16)
    assert pipe.text_encoder is installed
    assert pipe._modules["text_encoder"] is installed


def test_resolve_demo_pretrained_component_names_skips_text_encoder_for_light_host():
    from examples_merak.aigc.minimaxh3.hmonnx_demo import resolve_demo_pretrained_component_names

    pipe = SimpleNamespace(
        _component_specs={
            "image_processor": object(),
            "text_encoder": object(),
            "tokenizer": object(),
            "processor": object(),
            "vae": object(),
            "scheduler": object(),
            "audio_scheduler": object(),
            "transformer": object(),
            "video_processor": object(),
            "audio_vae": object(),
        }
    )
    light = resolve_demo_pretrained_component_names(pipe, use_light_text_host=True)
    full = resolve_demo_pretrained_component_names(pipe, use_light_text_host=False)
    assert "text_encoder" not in light
    assert light[0] == "image_processor"
    assert "tokenizer" in light and "processor" in light
    assert full[1] == "text_encoder"
    assert "text_encoder" in full
