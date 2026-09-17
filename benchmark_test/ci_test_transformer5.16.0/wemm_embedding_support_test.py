from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from xhmodel_merak.xh_llm import AutoLLMConfig
from xhmodel_merak.xh_llm.builder import get_model_class
from xhquant.api import Config


REPO_ROOT = Path(__file__).parents[2]
CONFIG_ROOT = REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/wemm_embedding"
MODEL_DIMENSIONS = {"2b": 2048, "4b": 2560, "9b": 4096}


def _load_standalone_module(module_name: str, relative_path: str):
    import importlib.util

    path = REPO_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_export_checks():
    return _load_standalone_module(
        "wemm_embedding_export_checks",
        "xhmodel_merak/xh_llm/models/wemm_embedding/export_checks.py",
    )


def _config_path(size: str, method: str = "autoround") -> Path:
    suffix = "" if method == "autoround" else f"_{method}"
    return CONFIG_ROOT / size / f"wemm_embedding_{size}_xh2a_w4a8{suffix}.yaml"


def _hf_model_contract_workflow(model_dir: str | Path, size: str = "2b"):
    from xhmodel_merak.xh_llm.models.wemm_embedding.workflow import XHWeMMEmbeddingWorkflow

    return XHWeMMEmbeddingWorkflow(
        model_dir=str(model_dir),
        config_path=str(_config_path(size)),
    )


def _checkpoint(tmp_path: Path, size: str) -> Path:
    model_dir = tmp_path / f"WeMM-Embedding-{size.upper()}"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"text_config": {"hidden_size": MODEL_DIMENSIONS[size]}}),
        encoding="utf-8",
    )
    return model_dir


def _load_export_config(model_dir: Path, size: str = "2b"):
    cfg = Config.fromfile(str(_config_path(size))).export.model
    cfg.hf_model = str(model_dir)
    return cfg


def _assert_recipe_export_contract(cfg, size: str) -> None:
    assert cfg.export.model.hf_model is None
    assert cfg.export.model.model_name == f"wemm_embedding_{size}"
    assert cfg.export.model.model_type == "WeMMEmbedding"
    assert cfg.export.model.context_max_length == 2048
    assert cfg.export.model.prefill_chunk_length == 256
    assert cfg.export.model.quant_scheme.quant_type == "w4a8h0_ssfp"
    assert cfg.export.model.visual_config.quant_scheme.quant_type == "w8a8h1_sefp"
    assert cfg.export.model.quant_scheme.ops == {}
    assert cfg.export.model.visual_config.quant_scheme.ops == {}


def test_wemm_embedding_config_resolves_to_dedicated_adapter(tmp_path):
    cfg = _load_export_config(_checkpoint(tmp_path, "2b"))
    model_cfg = AutoLLMConfig.from_pretrained(cfg)
    assert type(model_cfg).__name__ == "XHWeMMEmbeddingModelConfig"
    assert model_cfg.num_logits_to_keep == 0
    model_cls = get_model_class(cfg)
    assert model_cls.__name__ == "XHWeMMEmbeddingModel"
    assert "qwen3_5" not in model_cls.__module__


def test_wemm_embedding_uses_prefill_only_workflow(tmp_path):
    cfg = _load_export_config(_checkpoint(tmp_path, "2b"))
    model_cfg = AutoLLMConfig.from_pretrained(cfg)
    assert model_cfg.model_type == "WeMMEmbedding"
    assert "xhmodel_merak.xh_llm.models.wemm_embedding.workflow" in get_model_class(cfg).WORKFLOW_CLS


@pytest.mark.parametrize("size,expected_dimension", MODEL_DIMENSIONS.items())
def test_wemm_embedding_recipes_are_isolated_and_consistent(tmp_path, size, expected_dimension):
    model_dir = _checkpoint(tmp_path, size)
    cfg = Config.fromfile(str(_config_path(size)))
    assert cfg.quant.algorithm == "gptqmodel"
    assert cfg.quant.method == "autoround"
    assert cfg.quant.mode == "llm-only"
    assert cfg.quant.bits == 4
    assert cfg.quant.group_size == 64
    assert cfg.quant.artifact_format == "gptqmodel_hf"
    assert cfg.quant.calibration.seqlen == 2048
    assert str(cfg.quant.calibration.jsonl).startswith("xh2modelzoo://")
    assert "validation" not in cfg.quant
    _assert_recipe_export_contract(cfg, size)
    checkpoint_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    assert checkpoint_config["text_config"]["hidden_size"] == expected_dimension


@pytest.mark.parametrize("size", MODEL_DIMENSIONS)
def test_wemm_embedding_gptq_recipes_are_isolated_and_consistent(size):
    cfg = Config.fromfile(str(_config_path(size, "gptq")))
    assert cfg.quant.algorithm == "gptqmodel"
    assert cfg.quant.method == "gptq"
    assert "mode" not in cfg.quant  # The GPTQ recipe API has no mode argument.
    assert cfg.quant.bits == 4
    assert cfg.quant.group_size == 64
    assert cfg.quant.artifact_format == "gptqmodel_hf"
    assert cfg.quant.calibration.nsamples == 256
    assert cfg.quant.calibration.seqlen == 1024
    assert cfg.quant.runtime.batch_size == 1
    assert cfg.quant.runtime.wait_for_submodule_finalizers is True
    assert cfg.quant.validation.check_quant_vision_demo is False
    assert cfg.quant.validation.skip_inference is True
    _assert_recipe_export_contract(cfg, size)


def test_wemm_embedding_gptq_recipe_translates_to_qwen35_backend_kwargs():
    from xhmodel_merak.xh_llm.models.qwen3_5.quant_adapter import build_qwen35_gptqmodel_kwargs

    cfg = Config.fromfile(str(_config_path("2b", "gptq")))
    kwargs = build_qwen35_gptqmodel_kwargs(
        hf_model_dir="weights/WeMM-Embedding-2B",
        output_dir="work_dirs/wemm_embedding/2b/wemm_embedding_gptq_quant",
        device="cuda:0",
        quant_cfg=cfg.quant,
        export_model_cfg=cfg.export.model,
        workflow_seed=42,
    )
    assert kwargs["method"] == "gptq"
    assert "mode" not in kwargs
    assert kwargs["topology"] == "dense"
    assert kwargs["bits"] == 4
    assert kwargs["group_size"] == 64
    assert kwargs["batch_size"] == 1
    assert kwargs["nsamples"] == 256
    assert kwargs["seqlen"] == 1024
    assert kwargs["calibration_jsonl"].endswith("Qwen3.5-27B.jsonl")
    assert kwargs["check_quant_vision_demo"] is False
    assert kwargs["skip_inference"] is True


def test_wemm_export_replaces_lm_head(monkeypatch):
    from xhmodel_merak.xh_llm.models.wemm_embedding.model import XHWeMMEmbeddingModel

    wrapped = SimpleNamespace(lm_head=object())
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model.XHQwen3_5Model.init_wrap_model",
        lambda self, hf_model: wrapped,
    )
    model = object.__new__(XHWeMMEmbeddingModel)
    assert XHWeMMEmbeddingModel.init_wrap_model(model, object()) is wrapped
    assert isinstance(wrapped.lm_head, nn.Identity)


@pytest.mark.parametrize("size,expected_dimension", MODEL_DIMENSIONS.items())
def test_wemm_export_metadata_uses_checkpoint_dimension(monkeypatch, size, expected_dimension):
    from xhmodel_merak.xh_llm.models.wemm_embedding.model import XHWeMMEmbeddingModel

    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model.XHQwen3_5Model._extra_export_metadata",
        lambda self, output_dir, meta_info: meta_info,
    )
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.models.wemm_embedding.model.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(text_config=SimpleNamespace(hidden_size=expected_dimension)),
    )
    model = object.__new__(XHWeMMEmbeddingModel)
    model.hf_model_dir = f"unused-{size}"
    meta = XHWeMMEmbeddingModel._extra_export_metadata(model, "unused", SimpleNamespace())
    assert meta.embedding_output == "hidden_states"
    assert meta.embedding_pooling == "last_token"
    assert meta.embedding_normalize is True
    assert meta.embedding_dimension == expected_dimension


def test_wemm_pooling_normalizes_last_valid_hidden_state():
    from xhmodel_merak.xh_llm.models.wemm_embedding.hmonnx_inference import (
        XHWeMMEmbeddingHMONNXModel,
    )

    hidden_states = torch.tensor([[[9.0, 0.0, 0.0], [3.0, 4.0, 0.0], [0.0, 0.0, 8.0]]])
    embedding = XHWeMMEmbeddingHMONNXModel._pool_hidden_states(
        hidden_states,
        input_length=2,
        expected_dimension=3,
    )
    assert embedding.dtype == torch.float32
    assert torch.allclose(embedding, torch.tensor([[0.6, 0.8, 0.0]]))
    assert embedding.norm(dim=-1).tolist() == pytest.approx([1.0])


def test_wemm_pooling_rejects_logits_dimension():
    from xhmodel_merak.xh_llm.models.wemm_embedding.hmonnx_inference import (
        XHWeMMEmbeddingHMONNXModel,
    )

    with pytest.raises(ValueError, match="does not match metadata embedding_dimension"):
        XHWeMMEmbeddingHMONNXModel._pool_hidden_states(
            torch.zeros(1, 2, 248078),
            input_length=2,
            expected_dimension=4096,
        )


@pytest.mark.parametrize(
    ("hidden_states", "input_length", "expected_dimension", "error"),
    [
        (None, 1, 3, "hidden-state tensor"),
        (torch.zeros(1, 1, 3), 0, 3, "input length must be positive"),
        (torch.zeros(1, 1, 3), 1, 0, "embedding_dimension must be positive"),
        (torch.zeros(1, 1, 3), 1, -1, "embedding_dimension must be positive"),
        (torch.zeros(0, 1, 3), 1, 3, "empty batch or embedding dimension"),
        (torch.zeros(1, 1, 0), 1, 3, "empty batch or embedding dimension"),
    ],
)
def test_wemm_pooling_rejects_invalid_inputs(hidden_states, input_length, expected_dimension, error):
    from xhmodel_merak.xh_llm.models.wemm_embedding.hmonnx_inference import XHWeMMEmbeddingHMONNXModel

    with pytest.raises((TypeError, ValueError), match=error):
        XHWeMMEmbeddingHMONNXModel._pool_hidden_states(hidden_states, input_length, expected_dimension)


@pytest.mark.parametrize("input_length", [1, 255, 256, 257, 512, 1024, 2048])
def test_wemm_context_length_accepts_supported_boundaries(input_length):
    from xhmodel_merak.xh_llm.models.wemm_embedding.hmonnx_inference import (
        XHWeMMEmbeddingHMONNXModel,
    )

    XHWeMMEmbeddingHMONNXModel._validate_input_length(input_length, 2048)


@pytest.mark.parametrize(
    ("input_length", "message"),
    [
        (0, "at least one token"),
        (2049, "exported context length 2048"),
    ],
)
def test_wemm_context_length_rejects_unsupported_boundaries(input_length, message):
    from xhmodel_merak.xh_llm.models.wemm_embedding.hmonnx_inference import (
        XHWeMMEmbeddingHMONNXModel,
    )

    with pytest.raises(ValueError, match=message):
        XHWeMMEmbeddingHMONNXModel._validate_input_length(input_length, 2048)


def _run_fake_multi_chunk(input_length: int, image_start: int | None = None, image_count: int = 0):
    from xhmodel_merak.xh_llm.models.wemm_embedding.hmonnx_inference import (
        XHWeMMEmbeddingHMONNXModel,
    )
    from xhmodel_merak.xh_llm.types import CacheList

    class FakeDataProcessor:
        image_token_id = 7
        pad_token_id = 0

        def __init__(self):
            self.rope_deltas = None
            self.past_key_caches = CacheList()
            self.past_value_caches = CacheList()
            self.past_conv_caches = CacheList()
            self.past_recurrent_states = CacheList()

        def get_rope_index(
            self,
            input_ids,
            inputs_embeds,
            image_grid_thw=None,
            video_grid_thw=None,
            attention_mask=None,
        ):
            del input_ids, image_grid_thw, video_grid_thw, attention_mask
            positions = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            position_ids = positions.view(1, 1, -1).expand(3, 1, -1)
            rope_deltas = torch.zeros((1, 1), dtype=torch.long, device=inputs_embeds.device)
            return position_ids, rope_deltas

    data_processor = FakeDataProcessor()
    forward_calls = []

    def fake_forward(*args):
        past_seq_length = int(args[4].item())
        current_input_length = int(args[5].item())
        forward_calls.append(
            {
                "past_seq_length": past_seq_length,
                "current_input_length": current_input_length,
                "linear_mask_length": int(args[6].sum().item()),
                "time_position_ids": args[1].tolist(),
                "cache_ids": tuple(id(cache) for cache in args[7:11]),
                "inputs_embeds": args[0].detach().clone(),
            }
        )
        hidden_states = torch.zeros((1, 256, 3), dtype=torch.float16)
        hidden_states[0, :, 0] = args[1].to(torch.float16)
        hidden_states[0, :, 1] = 1
        return hidden_states, [], []

    embed_tokens = nn.Embedding(32, 3)
    with torch.no_grad():
        embed_tokens.weight.zero_()
    runtime = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float16,
        embed_tokens=embed_tokens,
        get_data_preprocessor=lambda: data_processor,
        forward=fake_forward,
    )
    input_ids = torch.ones((1, input_length), dtype=torch.long)
    image_embeds = None
    if image_start is not None:
        input_ids[:, image_start : image_start + image_count] = data_processor.image_token_id
        image_embeds = torch.arange(image_count * 3, dtype=torch.float32).reshape(image_count, 3)
    hidden_states, last_chunk_length = XHWeMMEmbeddingHMONNXModel._run_multi_chunk_prefill(
        runtime,
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        image_grid_thw=None,
        video_grid_thw=None,
        image_embeds=image_embeds,
        video_embeds=None,
        prefill_length=256,
        input_length=input_length,
    )
    return forward_calls, hidden_states, last_chunk_length, image_embeds


def test_wemm_multi_chunk_prefill_reuses_cache_state_and_tail_length():
    from xhmodel_merak.xh_llm.models.wemm_embedding.hmonnx_inference import (
        XHWeMMEmbeddingHMONNXModel,
    )

    forward_calls, hidden_states, last_chunk_length, image_embeds = _run_fake_multi_chunk(
        300,
        image_start=250,
        image_count=10,
    )

    assert [(call["past_seq_length"], call["current_input_length"]) for call in forward_calls] == [
        (0, 256),
        (256, 44),
    ]
    assert [call["linear_mask_length"] for call in forward_calls] == [256, 44]
    assert forward_calls[0]["time_position_ids"] == list(range(256))
    assert forward_calls[1]["time_position_ids"][:44] == list(range(256, 300))
    assert forward_calls[0]["cache_ids"] == forward_calls[1]["cache_ids"]
    assert torch.equal(forward_calls[0]["inputs_embeds"][0, 250:256], image_embeds[:6])
    assert torch.equal(forward_calls[1]["inputs_embeds"][0, :4], image_embeds[6:])
    assert last_chunk_length == 44
    embedding = XHWeMMEmbeddingHMONNXModel._pool_hidden_states(
        hidden_states,
        input_length=last_chunk_length,
        expected_dimension=3,
    )
    expected = torch.tensor([[299.0, 1.0, 0.0]])
    expected = torch.nn.functional.normalize(expected, p=2, dim=-1)
    assert torch.allclose(embedding, expected)


@pytest.mark.parametrize(
    ("input_length", "expected_chunk_lengths"),
    [
        (256, [256]),
        (512, [256, 256]),
        (513, [256, 256, 1]),
        (1024, [256] * 4),
        (2048, [256] * 8),
    ],
)
def test_wemm_multi_chunk_prefill_covers_tail_and_context_limit(input_length, expected_chunk_lengths):
    from xhmodel_merak.xh_llm.models.wemm_embedding.hmonnx_inference import (
        XHWeMMEmbeddingHMONNXModel,
    )

    forward_calls, hidden_states, last_chunk_length, _ = _run_fake_multi_chunk(input_length)
    assert [call["past_seq_length"] for call in forward_calls] == list(range(0, input_length, 256))
    assert [call["current_input_length"] for call in forward_calls] == expected_chunk_lengths
    assert [call["linear_mask_length"] for call in forward_calls] == expected_chunk_lengths
    assert len({call["cache_ids"] for call in forward_calls}) == 1
    assert last_chunk_length == expected_chunk_lengths[-1]

    embedding = XHWeMMEmbeddingHMONNXModel._pool_hidden_states(
        hidden_states,
        input_length=last_chunk_length,
        expected_dimension=3,
    )
    expected = torch.tensor([[float(input_length - 1), 1.0, 0.0]])
    expected = torch.nn.functional.normalize(expected, p=2, dim=-1)
    assert torch.allclose(embedding, expected)


def test_wemm_embed_clears_cache_state_between_requests():
    from xhmodel_merak.xh_llm.models.wemm_embedding.hmonnx_inference import (
        XHWeMMEmbeddingHMONNXModel,
    )

    class FakeCacheMixin:
        def __init__(self):
            self.kv = []
            self.other = []
            self.prepare_count = 0
            self.clear_count = 0

        def prepare_kv_cache(self):
            assert not self.kv
            self.kv.append(self.prepare_count)
            self.prepare_count += 1

        def _prepare_other_cache_on_device(self, device):
            assert device == torch.device("cpu")
            assert not self.other
            self.other.append(self.prepare_count)

        def clear_kv_cache(self):
            self.kv.clear()
            self.clear_count += 1

        def clear_other_cache(self):
            self.other.clear()

    class FakeEmbedding:
        def to(self, device):
            assert device == torch.device("cpu")
            return self

        def __call__(self, input_ids):
            return torch.zeros((*input_ids.shape, 3), dtype=torch.float16)

    class FakeProcessor:
        def apply_chat_template(self, messages, add_generation_prompt=False):
            assert messages[0]["content"][0]["text"] in {"first", "second"}
            assert add_generation_prompt is False
            return {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
            }

    cache_mixin = FakeCacheMixin()
    lifecycle = []

    runtime = SimpleNamespace(
        device=torch.device("cpu"),
        dtype=torch.float16,
        embed_tokens=FakeEmbedding(),
        meta_info=SimpleNamespace(
            embedding_dimension=3,
            model_config=SimpleNamespace(prefill_chunk_length=256, context_max_length=2048),
        ),
        to=lambda device: lifecycle.append(("to", device)),
        get_kvcache_mixin=lambda: cache_mixin,
        prepare_for_inference=lambda: lifecycle.append(("prepare", cache_mixin.prepare_count)),
        release_inference_model=lambda: lifecycle.append(("release", cache_mixin.clear_count)),
        set_prefill=lambda: lifecycle.append(("prefill", cache_mixin.prepare_count)),
        get_data_preprocessor=lambda: lambda data: (),
        forward=lambda: torch.tensor([[[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]]),
        _messages=XHWeMMEmbeddingHMONNXModel._messages,
        _validate_input_length=XHWeMMEmbeddingHMONNXModel._validate_input_length,
        _pool_hidden_states=XHWeMMEmbeddingHMONNXModel._pool_hidden_states,
    )

    first = XHWeMMEmbeddingHMONNXModel.embed(runtime, {"text": "first"}, processor=FakeProcessor())
    assert not cache_mixin.kv
    assert not cache_mixin.other
    second = XHWeMMEmbeddingHMONNXModel.embed(runtime, {"text": "second"}, processor=FakeProcessor())

    assert torch.equal(first, second)
    assert cache_mixin.prepare_count == 2
    assert cache_mixin.clear_count == 2
    assert not cache_mixin.kv
    assert not cache_mixin.other
    assert [event[0] for event in lifecycle].count("prepare") == 2
    assert [event[0] for event in lifecycle].count("release") == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for a real embedding device check")
def test_wemm_single_chunk_moves_processor_inputs_to_cuda(monkeypatch):
    from contextlib import nullcontext

    from xhmodel_merak.xh_llm.models.wemm_embedding import hmonnx_inference

    monkeypatch.setattr(hmonnx_inference, "LLMInferenceContextManager", lambda *args, **kwargs: nullcontext())
    embedding = nn.Embedding(8, 3, device="cuda:0")

    class FakeProcessor:
        def apply_chat_template(self, messages, add_generation_prompt=False):
            return {
                "input_ids": torch.tensor([[1, 2]], dtype=torch.long),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
            }

    def preprocess(data):
        assert data["input_ids"].device == embedding.weight.device
        return (embedding(data["input_ids"]),)

    model_cls = hmonnx_inference.XHWeMMEmbeddingHMONNXModel
    runtime = SimpleNamespace(
        device=torch.device("cuda:0"),
        dtype=torch.float16,
        embed_tokens=embedding,
        meta_info=SimpleNamespace(
            embedding_dimension=3,
            model_config=SimpleNamespace(prefill_chunk_length=256, context_max_length=2048),
        ),
        to=lambda device: None,
        set_prefill=lambda: None,
        get_data_preprocessor=lambda: preprocess,
        forward=lambda inputs_embeds: inputs_embeds,
        _messages=model_cls._messages,
        _validate_input_length=model_cls._validate_input_length,
        _pool_hidden_states=model_cls._pool_hidden_states,
    )
    result = model_cls.embed(runtime, {"text": "hello"}, processor=FakeProcessor())
    assert result.device == torch.device("cuda:0")
    assert result.shape == (1, 3)


@pytest.mark.parametrize(
    ("item", "expected_types"),
    [
        ({"text": "hello"}, ["text"]),
        ({"image": "image.jpg"}, ["image"]),
        ({"image": "image.jpg", "text": "describe"}, ["image", "text"]),
        ({"video": "video.mp4", "text": "describe"}, ["video", "text"]),
    ],
)
def test_wemm_messages_preserve_supported_modalities(item, expected_types):
    from xhmodel_merak.xh_llm.models.wemm_embedding.hmonnx_inference import (
        XHWeMMEmbeddingHMONNXModel,
    )

    messages = XHWeMMEmbeddingHMONNXModel._messages(item)
    assert messages[0]["role"] == "user"
    assert [part["type"] for part in messages[0]["content"]] == expected_types


def test_wemm_messages_preserve_interleaved_video():
    from xhmodel_merak.xh_llm.models.wemm_embedding.hmonnx_inference import XHWeMMEmbeddingHMONNXModel

    item = {"messages": [{"role": "user", "content": [{"type": "video", "video": "video.mp4"}]}]}
    assert XHWeMMEmbeddingHMONNXModel._messages(item) == item["messages"]


def test_wemm_video_patches_split_into_visual_frame_grids():
    from xhmodel_merak.xh_llm.models.wemm_embedding.processor import video_frame_grids

    temporal, channels, patch, merge, grid_t = 2, 3, 16, 2, 2
    height = width = 32
    frames = torch.arange(grid_t * temporal * channels * height * width, dtype=torch.float32)
    frames = frames.reshape(grid_t, temporal, channels, height, width).remainder(256)
    normalized = (frames / 255 - 0.5) / 0.5
    flat = normalized.reshape(grid_t, temporal, channels, 1, merge, patch, 1, merge, patch)
    flat = flat.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(-1, channels * temporal * patch * patch)
    grids = video_frame_grids(flat, torch.tensor([[grid_t, merge, merge]]), merge)
    assert torch.equal(grids, torch.tensor([[1, merge, merge], [1, merge, merge]]))
    assert torch.equal(flat.reshape(grid_t, merge * merge, -1)[0], flat[: merge * merge])
    assert torch.equal(flat.reshape(grid_t, merge * merge, -1)[1], flat[merge * merge :])


def test_wemm_video_rope_expands_each_timestamped_frame():
    from xhmodel_merak.xh_llm.models.wemm_embedding.data_preprocess import WeMMEmbeddingDataPreprocess

    data_processor = WeMMEmbeddingDataPreprocess(
        token_embedding=nn.Embedding(16, 3),
        spatial_merge_size=2,
        image_token_id=3,
        video_token_id=5,
        vision_start_token_id=4,
    )
    input_ids = torch.tensor([[4, 5, 5, 5, 5, 6, 4, 5, 5, 5, 5, 6, 7]])
    video_grid = torch.tensor([[2, 4, 4]])
    positions, rope_delta = data_processor.get_rope_index(
        input_ids,
        video_grid_thw=video_grid,
    )
    assert positions.shape == (3, 1, input_ids.shape[1])
    assert rope_delta.shape == (1, 1)
    assert not torch.equal(positions[:, :, 1:5], positions[:, :, 7:11])
    assert torch.equal(video_grid, torch.tensor([[2, 4, 4]]))


def test_wemm_single_chunk_injects_video_features():
    from xhmodel_merak.xh_llm.kv_cache_mixin import CacheList
    from xhmodel_merak.xh_llm.models.wemm_embedding.data_preprocess import WeMMEmbeddingDataPreprocess

    embedding = nn.Embedding(16, 3)
    processor = WeMMEmbeddingDataPreprocess(
        token_embedding=embedding,
        input_sequence_length=8,
        past_key_caches=CacheList(),
        past_value_caches=CacheList(),
        past_conv_caches=CacheList(),
        past_recurrent_states=CacheList(),
        video_token_id=5,
    )
    video_feature = torch.tensor([[1.0, 2.0, 3.0]])
    inputs = processor.forward(
        {"input_ids": torch.tensor([[5, 1]]), "video_embeds": video_feature, "past_seq_length": 0}
    )
    assert torch.equal(inputs[0][0, 0], video_feature[0])
    assert torch.equal(inputs[0][0, 1], embedding.weight[1])


def test_wemm_runtime_builds_video_data_preprocessor():
    from xhmodel_merak.xh_llm.kv_cache_mixin import CacheList
    from xhmodel_merak.xh_llm.models.wemm_embedding.data_preprocess import WeMMEmbeddingDataPreprocess
    from xhmodel_merak.xh_llm.models.wemm_embedding.hmonnx_inference import XHWeMMEmbeddingHMONNXModel

    runtime = SimpleNamespace(
        embed_tokens=nn.Embedding(16, 3),
        get_input_sequence_length=lambda: 8,
        visual_meta=SimpleNamespace(max_size_w=448, max_size_h=448, patch_size=16),
        past_key_caches=CacheList(),
        past_value_caches=CacheList(),
        past_conv_caches=CacheList(),
        past_recurrent_states=CacheList(),
        meta_info=SimpleNamespace(
            model_config=SimpleNamespace(
                image_token_id=3,
                video_token_id=5,
                vision_start_token_id=4,
                vision_end_token_id=6,
                spatial_merge_size=2,
            )
        ),
        enable_page_attention=False,
    )
    processor = XHWeMMEmbeddingHMONNXModel._get_data_preprocessor(runtime)
    assert isinstance(processor, WeMMEmbeddingDataPreprocess)
    assert processor.video_token_id == 5


@pytest.mark.parametrize("item", [{}, {"text": None, "image": None, "video": None}])
def test_wemm_messages_reject_missing_modalities(item):
    from xhmodel_merak.xh_llm.models.wemm_embedding.hmonnx_inference import (
        XHWeMMEmbeddingHMONNXModel,
    )

    with pytest.raises(ValueError, match="must contain text, image, or video"):
        XHWeMMEmbeddingHMONNXModel._messages(item)


@pytest.mark.parametrize("method", ["embed_items", "embed_texts", "embed_images"])
def test_wemm_batch_interfaces_reject_empty_inputs(method):
    from xhmodel_merak.xh_llm.models.wemm_embedding.hmonnx_inference import (
        XHWeMMEmbeddingHMONNXModel,
    )

    with pytest.raises(ValueError, match="At least one WeMM"):
        getattr(XHWeMMEmbeddingHMONNXModel, method)(SimpleNamespace(), [])


def test_wemm_convenience_embedding_interfaces_reuse_embed_items():
    from xhmodel_merak.xh_llm.models.wemm_embedding.hmonnx_inference import (
        XHWeMMEmbeddingHMONNXModel,
    )

    calls = []
    runtime = SimpleNamespace(
        embed_items=lambda items, processor=None: calls.append((items, processor)) or torch.ones(len(items), 3)
    )
    processor = object()
    text_embeddings = XHWeMMEmbeddingHMONNXModel.embed_texts(
        runtime,
        ["first", "second"],
        processor=processor,
    )
    image_embeddings = XHWeMMEmbeddingHMONNXModel.embed_images(
        runtime,
        ["a.jpg"],
        processor=processor,
    )
    assert text_embeddings.shape == (2, 3)
    assert image_embeddings.shape == (1, 3)
    assert calls == [
        ([{"text": "first"}, {"text": "second"}], processor),
        ([{"image": "a.jpg"}], processor),
    ]


def test_wemm_provenance_alias_preserves_backend_record(tmp_path):
    from xhmodel_merak.xh_llm.models.wemm_embedding.workflow import (
        QWEN35_AUTOROUND_PROVENANCE,
        WEMM_AUTOROUND_PROVENANCE,
        XHWeMMEmbeddingWorkflow,
    )

    source = tmp_path / QWEN35_AUTOROUND_PROVENANCE
    source.write_text('{"recipe": "gptqmodel.recipes.qwen35_autoround"}\n', encoding="utf-8")
    destination = XHWeMMEmbeddingWorkflow.ensure_autoround_provenance_alias(tmp_path)
    assert destination == tmp_path / WEMM_AUTOROUND_PROVENANCE
    assert destination.read_bytes() == source.read_bytes()


def test_wemm_gptq_provenance_alias_preserves_backend_record(tmp_path):
    from xhmodel_merak.xh_llm.models.wemm_embedding.workflow import (
        QWEN35_GPTQ_PROVENANCE,
        WEMM_GPTQ_PROVENANCE,
        XHWeMMEmbeddingWorkflow,
    )

    source = tmp_path / QWEN35_GPTQ_PROVENANCE
    source.write_text('{"recipe": "gptqmodel.recipes.qwen35"}\n', encoding="utf-8")
    destination = XHWeMMEmbeddingWorkflow.ensure_gptq_provenance_alias(tmp_path)
    assert destination == tmp_path / WEMM_GPTQ_PROVENANCE
    assert destination.read_bytes() == source.read_bytes()


def test_wemm_quant_creates_delivery_provenance_alias(monkeypatch, tmp_path):
    from xhmodel_merak.xh_llm.models.qwen3_5.workflow import Qwen35Workflow
    from xhmodel_merak.xh_llm.models.wemm_embedding.workflow import (
        QWEN35_AUTOROUND_PROVENANCE,
        WEMM_AUTOROUND_PROVENANCE,
        XHWeMMEmbeddingWorkflow,
    )
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    (tmp_path / QWEN35_AUTOROUND_PROVENANCE).write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        Qwen35Workflow,
        "quant",
        lambda self, output_dir, device, config_overrides=None: QuantResult(
            raw_model_dir="raw",
            quanted_model_dir=str(tmp_path),
        ),
    )
    workflow = object.__new__(XHWeMMEmbeddingWorkflow)
    result = XHWeMMEmbeddingWorkflow.quant(workflow, "unused", "cpu")
    assert result.quanted_model_dir == str(tmp_path)
    assert (tmp_path / WEMM_AUTOROUND_PROVENANCE).read_text(encoding="utf-8") == "{}\n"


def test_wemm_quant_creates_gptq_delivery_provenance_alias(monkeypatch, tmp_path):
    from xhmodel_merak.xh_llm.models.qwen3_5.workflow import Qwen35Workflow
    from xhmodel_merak.xh_llm.models.wemm_embedding.workflow import (
        QWEN35_GPTQ_PROVENANCE,
        WEMM_GPTQ_PROVENANCE,
        XHWeMMEmbeddingWorkflow,
    )
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    (tmp_path / QWEN35_GPTQ_PROVENANCE).write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        Qwen35Workflow,
        "quant",
        lambda self, output_dir, device, config_overrides=None: QuantResult(
            raw_model_dir="raw",
            quanted_model_dir=str(tmp_path),
        ),
    )
    workflow = object.__new__(XHWeMMEmbeddingWorkflow)
    result = XHWeMMEmbeddingWorkflow.quant(workflow, "unused", "cpu")
    assert result.quanted_model_dir == str(tmp_path)
    assert (tmp_path / WEMM_GPTQ_PROVENANCE).read_text(encoding="utf-8") == "{}\n"


def test_workflow_cli_quantizes_and_exports(monkeypatch, tmp_path):
    from examples_merak.llm.wemm_embedding.wemm_embedding_workflow import main
    from xhmodel_merak.workflows import AutoWorkflow
    from xhmodel_merak.xh_llm.workflows.result import ExportResult, QuantResult

    calls = []
    quant_dir = tmp_path / "quant"
    quant_dir.mkdir()

    class FakeWorkflow:
        def quant(self, output_dir, device, config_overrides):
            calls.append(("quant", output_dir, device, config_overrides))
            return QuantResult(raw_model_dir="native", quanted_model_dir=str(quant_dir))

        def export(self, quant_result, output_dir, device, config_overrides):
            calls.append(("export", output_dir, device, config_overrides))
            return ExportResult(work_dir=output_dir, config_file="config.yaml")

    monkeypatch.setattr(AutoWorkflow, "from_config", lambda **kwargs: FakeWorkflow())
    monkeypatch.setattr(
        "examples_merak.llm.wemm_embedding.wemm_embedding_workflow.preflight_recipe",
        lambda **kwargs: {"ok": True},
    )
    main(
        [
            "--size",
            "2b",
            "--quant-method",
            "gptq",
            "--model-dir",
            "native",
            "--quant-output-dir",
            str(tmp_path / "quant"),
            "--export-output-dir",
            str(tmp_path / "export"),
            "--device",
            "cuda:0",
        ]
    )
    assert [call[0] for call in calls] == ["quant", "export"]


def test_wemm_workflow_routes_gptq_recipe_to_shared_backend(monkeypatch, tmp_path):
    from xhmodel_merak.xh_llm.models.qwen3_5.workflow import Qwen35Workflow
    from xhmodel_merak.xh_llm.models.wemm_embedding.workflow import XHWeMMEmbeddingWorkflow
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    calls = {}

    def fake_quant(self, output_dir, device, quant_cfg, export_model_cfg):
        calls.update(
            output_dir=output_dir,
            device=device,
            method=quant_cfg["method"],
            model_type=export_model_cfg["model_type"],
        )
        return QuantResult(raw_model_dir=self.model_dir, quanted_model_dir=str(tmp_path))

    monkeypatch.setattr(Qwen35Workflow, "_quant_gptqmodel_api", fake_quant)
    workflow = XHWeMMEmbeddingWorkflow.from_config(
        model_dir="weights/WeMM-Embedding-2B",
        config_path=str(_config_path("2b", "gptq")),
        seed=42,
    )
    result = workflow.quant("work_dirs/gptq", "cuda:0")
    assert result.quanted_model_dir == str(tmp_path)
    assert calls == {
        "output_dir": "work_dirs/gptq",
        "device": "cuda:0",
        "method": "gptq",
        "model_type": "WeMMEmbedding",
    }


def test_workflow_cli_resolves_size_specific_defaults(monkeypatch):
    from examples_merak.llm.wemm_embedding.wemm_embedding_workflow import parse_args, resolve_paths

    monkeypatch.setenv("WEMM_MODEL_ROOT", "/models")
    paths = resolve_paths(parse_args(["--size", "9b"]))
    assert paths["model_dir"] == "/models/WeMM-Embedding-9B"
    assert paths["config_path"].endswith("9b/wemm_embedding_9b_xh2a_w4a8.yaml")
    assert paths["quant_output_dir"].endswith("9b/wemm_embedding_quant")
    assert paths["export_output_dir"].endswith("9b/wemm_embedding_export")


def test_workflow_cli_resolves_gptq_specific_defaults(monkeypatch):
    from examples_merak.llm.wemm_embedding.wemm_embedding_workflow import parse_args, resolve_paths

    monkeypatch.setenv("WEMM_MODEL_ROOT", "/models")
    paths = resolve_paths(parse_args(["--size", "4b", "--quant-method", "gptq"]))
    assert paths["model_dir"] == "/models/WeMM-Embedding-4B"
    assert paths["config_path"].endswith("4b/wemm_embedding_4b_xh2a_w4a8_gptq.yaml")
    assert paths["quant_output_dir"].endswith("4b/wemm_embedding_gptq_quant")
    assert paths["export_output_dir"].endswith("4b/wemm_embedding_gptq_export")


@pytest.mark.parametrize("size", MODEL_DIMENSIONS)
@pytest.mark.parametrize("method", ["autoround", "gptq"])
def test_wemm_preflight_accepts_null_hf_model_placeholder(tmp_path, size, method):
    from examples_merak.llm.wemm_embedding.wemm_embedding_workflow import preflight_recipe

    model_dir = _checkpoint(tmp_path, size)
    (tmp_path / "quant").mkdir()
    report = preflight_recipe(
        config_path=_config_path(size, method),
        model_dir=model_dir,
        quant_model_dir=tmp_path / "quant",
        skip_quant=True,
    )
    assert report["hf_model_placeholder"] is True
    assert report["calibration_jsonl"] is None
    assert report["model_dir"] == str(model_dir.resolve())


def test_wemm_preflight_runs_autoround_quant_asset_checks(tmp_path):
    from examples_merak.llm.wemm_embedding.wemm_embedding_workflow import preflight_recipe
    from xhmodel_merak.xh_llm.models.qwen3_5.quant_adapter import build_qwen35_autoround_kwargs

    cfg = Config.fromfile(str(_config_path("2b")))
    model_dir = _checkpoint(tmp_path, "2b")
    report = preflight_recipe(
        config_path=_config_path("2b"),
        model_dir=model_dir,
        skip_quant=False,
    )
    assert Path(report["calibration_jsonl"]).is_file()
    kwargs = build_qwen35_autoround_kwargs(
        hf_model_dir=str(model_dir),
        output_dir=str(tmp_path / "quant"),
        device="cuda:0",
        quant_cfg=cfg.quant,
        export_model_cfg=cfg.export.model,
        workflow_seed=42,
    )
    assert kwargs["dataset"] == report["calibration_jsonl"]
    assert "check_quant_vision_demo" not in kwargs
    assert "skip_inference" not in kwargs


def test_resolve_xh2modelzoo_uri_loads_autoround_calibration_jsonl():
    from examples_merak.llm.wemm_embedding.wemm_embedding_workflow import resolve_xh2modelzoo_uri

    cfg = Config.fromfile(str(_config_path("2b")))
    jsonl = str(cfg.quant.calibration.jsonl)
    assert jsonl.startswith("xh2modelzoo://")
    resolved_jsonl = resolve_xh2modelzoo_uri(jsonl, "quantization calibration jsonl")
    assert resolved_jsonl.is_file()
    assert resolved_jsonl.name == "NeelNanda-pile-10k.jsonl"


def test_wemm_gptq_calibration_resolves_in_preflight_and_quant_adapter(monkeypatch, tmp_path):
    import importlib.util

    from examples_merak.llm.wemm_embedding.wemm_embedding_workflow import (
        preflight_recipe,
        resolve_xh2modelzoo_uri,
    )
    from xhmodel_merak.xh_llm.models.qwen3_5.quant_adapter import build_qwen35_gptqmodel_kwargs

    cfg = Config.fromfile(str(_config_path("2b", "gptq")))
    uri = str(cfg.quant.calibration.jsonl)
    assert uri.startswith("gptqmodel://")
    package_root = tmp_path / "gptqmodel"
    calibration_file = package_root / uri.removeprefix("gptqmodel://")
    calibration_file.parent.mkdir(parents=True)
    calibration_file.write_text('{"text": "sample"}\n', encoding="utf-8")
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: (
            SimpleNamespace(submodule_search_locations=[str(package_root)])
            if name == "gptqmodel"
            else original_find_spec(name)
        ),
    )

    assert resolve_xh2modelzoo_uri(uri) == calibration_file.resolve()
    model_dir = _checkpoint(tmp_path, "2b")
    report = preflight_recipe(config_path=_config_path("2b", "gptq"), model_dir=model_dir)
    assert report["calibration_jsonl"] == str(calibration_file.resolve())
    kwargs = build_qwen35_gptqmodel_kwargs(
        hf_model_dir=str(model_dir),
        output_dir=str(tmp_path / "quant"),
        device="cuda:0",
        quant_cfg=cfg.quant,
        export_model_cfg=cfg.export.model,
        workflow_seed=42,
    )
    assert kwargs["calibration_jsonl"] == report["calibration_jsonl"]


def test_wemm_gptq_calibration_missing_resource_fails_early(monkeypatch, tmp_path):
    import importlib.util

    from examples_merak.llm.wemm_embedding.wemm_embedding_workflow import preflight_recipe

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(FileNotFoundError, match="gptqmodel resource not found"):
        preflight_recipe(config_path=_config_path("2b", "gptq"), model_dir=_checkpoint(tmp_path, "2b"))


def test_resolve_xh2modelzoo_uri_rejects_missing_resource():
    from examples_merak.llm.wemm_embedding.wemm_embedding_workflow import resolve_xh2modelzoo_uri

    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_xh2modelzoo_uri("xh2modelzoo://data/missing-wemm-resource.jsonl", "missing resource")


def test_resolve_xh2modelzoo_uri_supports_file_and_relative_paths():
    from examples_merak.llm.wemm_embedding.wemm_embedding_workflow import resolve_xh2modelzoo_uri

    relative = "data/calib_data/NeelNanda-pile-10k.jsonl"
    resolved_relative = resolve_xh2modelzoo_uri(relative, "relative calibration jsonl")
    assert resolved_relative.is_file()
    resolved_file = resolve_xh2modelzoo_uri(f"file://{resolved_relative}", "file URI calibration jsonl")
    assert resolved_file == resolved_relative
    with pytest.raises(ValueError, match="unsupported URI scheme"):
        resolve_xh2modelzoo_uri("http://example.com/missing.jsonl", "illegal uri")


def test_wemm_preflight_rejects_empty_calibration_file(tmp_path):
    from examples_merak.llm.wemm_embedding.wemm_embedding_workflow import preflight_recipe

    empty_file = tmp_path / "empty.jsonl"
    empty_file.touch()
    recipe = _config_path("2b").read_text(encoding="utf-8")
    config_path = tmp_path / "empty_calibration.yaml"
    config_path.write_text(
        recipe.replace("xh2modelzoo://data/calib_data/NeelNanda-pile-10k.jsonl", f"file://{empty_file}"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="quantization calibration jsonl is empty"):
        preflight_recipe(config_path=config_path, model_dir=_checkpoint(tmp_path, "2b"))


def test_resolve_xh2modelzoo_uri_rejects_directory_and_unreadable_file(monkeypatch, tmp_path):
    from examples_merak.llm.wemm_embedding.wemm_embedding_workflow import resolve_xh2modelzoo_uri

    with pytest.raises(FileNotFoundError, match="not found"):
        resolve_xh2modelzoo_uri(f"file://{tmp_path}", "calibration directory")
    blocked = tmp_path / "blocked.jsonl"
    blocked.write_text('{"text": "sample"}\n', encoding="utf-8")
    original_open = Path.open

    def deny_blocked(path, *args, **kwargs):
        if path == blocked:
            raise PermissionError("permission denied")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny_blocked)
    with pytest.raises(PermissionError, match="calibration file is not readable"):
        resolve_xh2modelzoo_uri(f"file://{blocked}", "calibration file")


def test_wemm_preflight_rejects_missing_model_directory(tmp_path):
    from examples_merak.llm.wemm_embedding.wemm_embedding_workflow import preflight_recipe

    missing = tmp_path / "missing-model"
    with pytest.raises(FileNotFoundError, match="model directory does not exist"):
        preflight_recipe(config_path=_config_path("2b"), model_dir=missing)


def test_wemm_preflight_rejects_model_file_instead_of_directory(tmp_path):
    from examples_merak.llm.wemm_embedding.wemm_embedding_workflow import preflight_recipe

    model_file = tmp_path / "model.bin"
    model_file.write_text("not-a-directory", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a directory"):
        preflight_recipe(config_path=_config_path("2b"), model_dir=model_file)


def test_wemm_preflight_rejects_directory_without_config_json(tmp_path):
    from examples_merak.llm.wemm_embedding.wemm_embedding_workflow import preflight_recipe

    model_dir = tmp_path / "empty-model"
    model_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="missing config.json"):
        preflight_recipe(config_path=_config_path("2b"), model_dir=model_dir)


def test_wemm_preflight_rejects_missing_quant_directory(tmp_path):
    from examples_merak.llm.wemm_embedding.wemm_embedding_workflow import preflight_recipe

    with pytest.raises(FileNotFoundError, match="Quantized model directory not found"):
        preflight_recipe(
            config_path=_config_path("2b"),
            model_dir=_checkpoint(tmp_path, "2b"),
            quant_model_dir=tmp_path / "missing-quant",
            skip_quant=True,
        )


def test_wemm_preflight_rejects_missing_workflow_config(tmp_path):
    from examples_merak.llm.wemm_embedding.wemm_embedding_workflow import preflight_recipe

    with pytest.raises(FileNotFoundError, match="workflow config not found"):
        preflight_recipe(
            config_path=tmp_path / "missing.yaml",
            model_dir=_checkpoint(tmp_path, "2b"),
        )


def test_flickr30k_loader_fails_when_images_are_missing(tmp_path):
    from examples_merak.llm.wemm_embedding.flickr30k_utils import load_flickr30k

    with pytest.raises(FileNotFoundError, match="No Flickr30K image directory"):
        load_flickr30k(tmp_path)


def test_flickr30k_loader_fails_when_annotations_are_missing(tmp_path):
    from examples_merak.llm.wemm_embedding.flickr30k_utils import load_flickr30k

    (tmp_path / "images").mkdir()
    with pytest.raises(RuntimeError, match="No supported Flickr30K annotations"):
        load_flickr30k(tmp_path)


def test_retrieval_dataset_rejects_empty_rows(tmp_path):
    from examples_merak.llm.wemm_embedding.retrieval_dataset import (
        RetrievalDataset,
        load_retrieval_dataset,
        register_retrieval_dataset,
    )

    name = f"empty-{tmp_path.name}"
    register_retrieval_dataset(
        name,
        lambda root: RetrievalDataset(name="empty", split="test", rows=[]),
        force=True,
    )
    with pytest.raises(ValueError, match="has no evaluation rows"):
        load_retrieval_dataset(name, tmp_path)


def test_resize_with_padding_rejects_corrupt_image(tmp_path):
    from examples_merak.llm.wemm_embedding.flickr30k_utils import resize_with_padding

    image_path = tmp_path / "broken.jpg"
    image_path.write_bytes(b"not-an-image")
    with pytest.raises(OSError):
        resize_with_padding(image_path, 448)


def test_summarize_accuracy_fails_when_reports_are_missing(tmp_path):
    from examples_merak.llm.wemm_embedding.summarize_accuracy import load_accuracy_reports

    missing = tmp_path / "missing-reports"
    with pytest.raises(FileNotFoundError, match="report directory not found"):
        load_accuracy_reports(missing)
    missing.mkdir()
    with pytest.raises(FileNotFoundError, match="No Flickr30K accuracy reports"):
        load_accuracy_reports(missing)


def test_compare_full_cosine_fails_when_embedding_files_are_missing(tmp_path):
    from examples_merak.llm.wemm_embedding.compare_full_cosine import _load_tensor

    with pytest.raises(FileNotFoundError):
        _load_tensor(tmp_path, "text_emb.pt")


def test_wemm_yaml_recipes_only_differ_by_size_fields():
    from copy import deepcopy

    for method in ("autoround", "gptq"):
        normalized = []
        for size in MODEL_DIMENSIONS:
            cfg = Config.fromfile(str(_config_path(size, method))).to_dict()
            assert cfg["export"]["model"]["hf_model"] is None
            assert cfg["export"]["model"]["model_name"] == f"wemm_embedding_{size}"
            assert cfg["export"]["model"]["context_max_length"] == 2048
            assert cfg["export"]["model"]["prefill_chunk_length"] == 256
            assert cfg["export"]["model"]["quant_scheme"]["quant_type"] == "w4a8h0_ssfp"
            assert cfg["export"]["model"]["visual_config"]["quant_scheme"]["quant_type"] == "w8a8h1_sefp"
            assert cfg["export"]["model"]["quant_scheme"]["ops"] == {}
            assert cfg["export"]["model"]["visual_config"]["quant_scheme"]["ops"] == {}
            payload = deepcopy(cfg)
            payload["export"]["model"]["model_name"] = "<size>"
            normalized.append(payload)
        assert normalized[0] == normalized[1] == normalized[2]


@pytest.mark.parametrize("size", MODEL_DIMENSIONS)
def test_wemm_autoround_and_gptq_share_export_contract(size):
    autoround = Config.fromfile(str(_config_path(size, "autoround"))).to_dict()
    gptq = Config.fromfile(str(_config_path(size, "gptq"))).to_dict()
    assert autoround["export"] == gptq["export"]


@pytest.mark.parametrize("size,expected_dimension", MODEL_DIMENSIONS.items())
@pytest.mark.parametrize("method", ["autoround", "gptq"])
def test_wemm_yaml_fake_artifacts_match_recipe_metadata(tmp_path, size, expected_dimension, method):
    from examples_merak.llm.wemm_embedding.runtime_utils import resolve_hmonnx_meta
    from xhmodel_merak.xh_llm.models.wemm_embedding.workflow import XHWeMMEmbeddingWorkflow
    from xhmodel_merak.xh_llm.workflows.result import ExportResult

    cfg = Config.fromfile(str(_config_path(size, method)))
    export_dir = tmp_path / size / method
    meta = export_dir / "hmquant_test" / "golden_meta_info.json"
    meta.parent.mkdir(parents=True)
    payload = {
        "model_name": cfg.export.model.model_name,
        "embedding_dimension": expected_dimension,
        "embedding_output": "hidden_states",
        "embedding_pooling": "last_token",
        "embedding_normalize": True,
        "quant_method": cfg.quant.method,
        "quant_type": cfg.export.model.quant_scheme.quant_type,
        "visual_quant_type": cfg.export.model.visual_config.quant_scheme.quant_type,
        "prefill_chunk_length": cfg.export.model.prefill_chunk_length,
        "context_max_length": cfg.export.model.context_max_length,
    }
    meta.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert resolve_hmonnx_meta(export_dir) == meta
    found = XHWeMMEmbeddingWorkflow._find_golden_meta_file(
        ExportResult(work_dir=str(export_dir), config_file="unused.yaml")
    )
    metadata = json.loads(Path(found).read_text(encoding="utf-8"))
    assert metadata["model_name"] == f"wemm_embedding_{size}"
    assert metadata["embedding_dimension"] == expected_dimension
    assert metadata["quant_method"] == ("autoround" if method == "autoround" else "gptq")
    assert metadata["quant_type"] == "w4a8h0_ssfp"
    assert metadata["visual_quant_type"] == "w8a8h1_sefp"
    assert metadata["prefill_chunk_length"] == 256
    assert metadata["context_max_length"] == 2048


def test_load_hmonnx_embedder_uses_resolved_metadata(monkeypatch, tmp_path):
    from examples_merak.llm.wemm_embedding.runtime_utils import load_hmonnx_embedder

    meta = tmp_path / "golden_meta_info.json"
    meta.write_text("{}\n", encoding="utf-8")
    captured = {}

    class FakeModel:
        def to(self, device):
            captured["device"] = str(device)
            return self

    class FakeAuto:
        @staticmethod
        def from_pretrained(path, device_map=None):
            captured["path"] = path
            captured["device_map"] = device_map
            return FakeModel()

    monkeypatch.setattr("xhmodel_merak.xh_llm.AutoLLMHONNXModel", FakeAuto)
    model, resolved = load_hmonnx_embedder(tmp_path, "cpu")
    assert isinstance(model, FakeModel)
    assert resolved == meta
    assert captured["path"] == str(meta)
    assert captured["device_map"] == ["cpu"]
    assert captured["device"] == "cpu"


def test_wemm_embed_rejects_inputs_over_exported_context():
    from xhmodel_merak.xh_llm.models.wemm_embedding.hmonnx_inference import (
        XHWeMMEmbeddingHMONNXModel,
    )

    class FakeProcessor:
        def apply_chat_template(self, messages, add_generation_prompt=False):
            del messages, add_generation_prompt
            return {
                "input_ids": torch.ones((1, 2049), dtype=torch.long),
                "attention_mask": torch.ones((1, 2049), dtype=torch.long),
            }

    runtime = SimpleNamespace(
        device=torch.device("cpu"),
        embed_tokens=SimpleNamespace(to=lambda device: None),
        meta_info=SimpleNamespace(
            embedding_dimension=3,
            model_config=SimpleNamespace(prefill_chunk_length=256, context_max_length=2048),
        ),
        to=lambda device: None,
        _messages=XHWeMMEmbeddingHMONNXModel._messages,
        _validate_input_length=XHWeMMEmbeddingHMONNXModel._validate_input_length,
    )
    with pytest.raises(ValueError, match="exported context length 2048"):
        XHWeMMEmbeddingHMONNXModel.embed(runtime, {"text": "too long"}, processor=FakeProcessor())


def test_wemm_workflow_overwrites_null_hf_model_placeholder(tmp_path):
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    checks = _load_export_checks()
    quant_dir = tmp_path / "quant"
    quant_dir.mkdir()
    native_dir = tmp_path / "native"
    native_dir.mkdir()
    workflow = _hf_model_contract_workflow(native_dir)
    yaml_export = workflow.workflow_config.export
    assert yaml_export["model"]["hf_model"] is None
    assert checks.is_placeholder_hf_model(yaml_export["model"]["hf_model"]) is True
    checks.require_placeholder_hf_model(yaml_export)
    export_cfg = workflow._build_export_config(
        QuantResult(raw_model_dir=workflow.model_dir, quanted_model_dir=str(quant_dir)),
        workflow.workflow_config,
    )
    assert export_cfg["model"]["hf_model"] == str(quant_dir)
    assert checks.require_overwritten_hf_model(export_cfg) == str(quant_dir)


def test_is_placeholder_hf_model_detects_yaml_null():
    checks = _load_export_checks()
    assert checks.is_placeholder_hf_model(None) is True
    assert checks.is_placeholder_hf_model("") is False
    assert checks.is_placeholder_hf_model("null") is False
    assert checks.is_placeholder_hf_model("/models/WeMM-Embedding-2B") is False


def test_hf_model_contract_rejects_non_null_original(tmp_path):
    checks = _load_export_checks()
    quant_dir = tmp_path / "quant"
    quant_dir.mkdir()
    with pytest.raises(ValueError, match="must stay null before overwrite"):
        checks.require_placeholder_hf_model({"model": {"hf_model": str(quant_dir)}})
    export_cfg = {"model": {"hf_model": str(quant_dir)}}
    checks.require_placeholder_hf_model({"model": {"hf_model": None}})
    assert checks.require_overwritten_hf_model(export_cfg) == str(quant_dir)


def test_hf_model_contract_rejects_file_and_missing_paths(tmp_path):
    checks = _load_export_checks()
    quant_file = tmp_path / "quant.bin"
    quant_file.write_text("weights", encoding="utf-8")
    with pytest.raises(ValueError, match="is a file"):
        checks.require_overwritten_hf_model({"model": {"hf_model": str(quant_file)}})
    with pytest.raises(FileNotFoundError, match="not a directory"):
        checks.require_overwritten_hf_model({"model": {"hf_model": str(tmp_path / "missing")}})
    with pytest.raises(ValueError, match="still null after workflow overwrite"):
        checks.require_overwritten_hf_model({"model": {"hf_model": None}})


def test_wemm_export_rejects_non_null_hf_model_before_overwrite():
    from examples_merak.llm.wemm_embedding.wemm_embedding_workflow import (
        require_placeholder_hf_model as cli_require_placeholder_hf_model,
    )

    for require_placeholder_hf_model in (
        _load_export_checks().require_placeholder_hf_model,
        cli_require_placeholder_hf_model,
    ):
        require_placeholder_hf_model({"model": {"hf_model": None}})
        for invalid in ("null", ""):
            with pytest.raises(ValueError, match="must stay null before overwrite"):
                require_placeholder_hf_model({"model": {"hf_model": invalid}})
        with pytest.raises(ValueError, match="must stay null before overwrite"):
            require_placeholder_hf_model({"model": {"hf_model": "/models/WeMM-Embedding-2B"}})
        with pytest.raises(ValueError, match="missing a model mapping"):
            require_placeholder_hf_model({})
        with pytest.raises(ValueError, match="hf_model is missing"):
            require_placeholder_hf_model({"model": {}})


def test_wemm_cli_preflight_rejects_missing_hf_model_field(tmp_path):
    from examples_merak.llm.wemm_embedding.wemm_embedding_workflow import preflight_recipe

    recipe = _config_path("2b").read_text(encoding="utf-8")
    config_path = tmp_path / "missing_hf_model.yaml"
    config_path.write_text(recipe.replace("    hf_model: null\n", "", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="hf_model is missing"):
        preflight_recipe(config_path=config_path, model_dir=tmp_path / "unused")


def test_wemm_export_rejects_leftover_null_hf_model(tmp_path):
    require_overwritten_hf_model = _load_export_checks().require_overwritten_hf_model
    with pytest.raises(ValueError, match="still null after workflow overwrite"):
        require_overwritten_hf_model({"model": {"hf_model": None}})
    with pytest.raises(ValueError, match="still null after workflow overwrite"):
        require_overwritten_hf_model({"model": {"hf_model": "null"}})
    with pytest.raises(ValueError, match="still null after workflow overwrite"):
        require_overwritten_hf_model({"model": {"hf_model": ""}})
    with pytest.raises(ValueError, match="missing a model mapping"):
        require_overwritten_hf_model({})
    missing = tmp_path / "missing-quant"
    with pytest.raises(FileNotFoundError, match="not a directory"):
        require_overwritten_hf_model({"model": {"hf_model": str(missing)}})
    quant_file = tmp_path / "quant.bin"
    quant_file.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="is a file"):
        require_overwritten_hf_model({"model": {"hf_model": str(quant_file)}})


def test_wemm_export_rejects_overwritten_hf_model_missing_directory(tmp_path):
    from examples_merak.llm.wemm_embedding.wemm_embedding_workflow import require_export_hf_model_dir
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    missing = tmp_path / "missing-quant"
    native_dir = tmp_path / "native"
    native_dir.mkdir()
    workflow = _hf_model_contract_workflow(native_dir)
    with pytest.raises(FileNotFoundError, match="not a directory"):
        workflow._build_export_config(
            QuantResult(raw_model_dir=workflow.model_dir, quanted_model_dir=str(missing)),
            workflow.workflow_config,
        )
    with pytest.raises(FileNotFoundError, match="overwrite target is not a directory"):
        require_export_hf_model_dir(missing)


def test_wemm_cli_rejects_invalid_hf_model_overwrite_directory(tmp_path):
    from examples_merak.llm.wemm_embedding.wemm_embedding_workflow import require_export_hf_model_dir

    with pytest.raises(ValueError, match="overwrite target is empty"):
        require_export_hf_model_dir(None)
    quant_file = tmp_path / "quant.bin"
    quant_file.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="overwrite target is a file"):
        require_export_hf_model_dir(quant_file)
    missing = tmp_path / "missing-quant"
    with pytest.raises(FileNotFoundError, match="overwrite target is not a directory"):
        require_export_hf_model_dir(missing)
    quant_dir = tmp_path / "quant"
    quant_dir.mkdir()
    assert require_export_hf_model_dir(quant_dir) == str(quant_dir)


def test_wemm_workflow_rejects_hf_model_file_path(tmp_path):
    from examples_merak.llm.wemm_embedding.wemm_embedding_workflow import require_export_hf_model_dir
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    quant_file = tmp_path / "quant.bin"
    quant_file.write_text("weights", encoding="utf-8")
    native_dir = tmp_path / "native"
    native_dir.mkdir()
    workflow = _hf_model_contract_workflow(native_dir)
    with pytest.raises(ValueError, match="is a file"):
        workflow._build_export_config(
            QuantResult(raw_model_dir=workflow.model_dir, quanted_model_dir=str(quant_file)),
            workflow.workflow_config,
        )
    with pytest.raises(ValueError, match="overwrite target is a file"):
        require_export_hf_model_dir(quant_file)


def test_wemm_workflow_rejects_non_null_original_hf_model(tmp_path):
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    quant_dir = tmp_path / "quant"
    quant_dir.mkdir()
    native_dir = tmp_path / "native"
    native_dir.mkdir()
    workflow = _hf_model_contract_workflow(native_dir)
    workflow.workflow_config.data["export"]["model"]["hf_model"] = str(native_dir)
    with pytest.raises(ValueError, match="must stay null before overwrite"):
        workflow._build_export_config(
            QuantResult(raw_model_dir=workflow.model_dir, quanted_model_dir=str(quant_dir)),
            workflow.workflow_config,
        )


def test_wemm_workflow_rejects_missing_original_hf_model(tmp_path):
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    quant_dir = tmp_path / "quant"
    quant_dir.mkdir()
    native_dir = tmp_path / "native"
    native_dir.mkdir()
    workflow = _hf_model_contract_workflow(native_dir)
    del workflow.workflow_config.data["export"]["model"]["hf_model"]
    with pytest.raises(ValueError, match="hf_model is missing"):
        workflow._build_export_config(
            QuantResult(raw_model_dir=workflow.model_dir, quanted_model_dir=str(quant_dir)),
            workflow.workflow_config,
        )


def test_wemm_workflow_rejects_leftover_null_hf_model_after_overwrite(monkeypatch, tmp_path):
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    quant_dir = tmp_path / "quant"
    quant_dir.mkdir()
    native_dir = tmp_path / "native"
    native_dir.mkdir()
    workflow = _hf_model_contract_workflow(native_dir)
    monkeypatch.setattr(workflow, "_resolve_export_model_dir", lambda quant_result: None)
    with pytest.raises(ValueError, match="still null after workflow overwrite"):
        workflow._build_export_config(
            QuantResult(raw_model_dir=workflow.model_dir, quanted_model_dir=str(quant_dir)),
            workflow.workflow_config,
        )
