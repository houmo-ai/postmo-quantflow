import json
from types import SimpleNamespace

import pytest
import torch

from vllm_merak.config import MerakConfigParser


def _write_wemm_export(tmp_path, *, runner_type: str | None = "pooling") -> None:
    (tmp_path / "hf_config").mkdir()
    (tmp_path / "hf_config" / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "hidden_size": 64,
                "num_attention_heads": 4,
                "num_hidden_layers": 2,
                "vocab_size": 128,
                "max_position_embeddings": 2048,
            }
        ),
        encoding="utf-8",
    )
    visual = tmp_path / "visual" / "visual.hmonnx"
    visual.parent.mkdir()
    visual.touch()
    (tmp_path / "golden_meta_info.json").write_text(
        json.dumps(
            {
                "embedding_output": "hidden_states",
                "embedding_pooling": "last_token",
                "embedding_normalize": True,
                "embedding_dimension": 64,
                "model_config": {"model_type": "WeMMEmbedding"},
                "visual_config": {
                    "hmonnx": "visual/visual.hmonnx",
                    "image_size_w": 448,
                    "image_size_h": 448,
                    "patch_size": 16,
                    "spatial_merge_size": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    merak_config = {
        "architectures": ["WeMMEmbeddingMerakForEmbedding"],
        "enable_page_attention": True,
        "xh_model": {
            "model_type": "hmonnx",
            "meta_info": "golden_meta_info.json",
        },
    }
    if runner_type is not None:
        merak_config["runner_type"] = runner_type
    (tmp_path / "merak_config.json").write_text(
        json.dumps(merak_config),
        encoding="utf-8",
    )


def test_wemm_metadata_selects_pooling_loader(tmp_path) -> None:
    _write_wemm_export(tmp_path)

    config_dict, config = MerakConfigParser().parse(
        tmp_path,
        trust_remote_code=False,
    )

    assert config_dict["architectures"] == ["WeMMEmbeddingMerakForEmbedding"]
    assert config_dict["merak_loader_architecture"] == "WeMMEmbeddingMerakForEmbedding"
    assert config_dict["runner_type"] == "pooling"
    assert config_dict["enable_page_attention"] is True
    assert config.runner_type == "pooling"


def test_wemm_metadata_does_not_infer_missing_pooling_runner_type(tmp_path) -> None:
    _write_wemm_export(tmp_path, runner_type=None)

    config_dict, config = MerakConfigParser().parse(
        tmp_path,
        trust_remote_code=False,
    )

    # The xh2modelzoo export workflow owns the WeMM runner contract.  Missing
    # fields retain Merak's generic legacy-generation fallback.
    assert config_dict["runner_type"] == "generate"
    assert config.runner_type == "generate"


def test_wemm_config_parser_preserves_explicit_generate_runner_type(tmp_path) -> None:
    _write_wemm_export(tmp_path, runner_type="generate")

    config_dict, config = MerakConfigParser().parse(
        tmp_path,
        trust_remote_code=False,
    )

    assert config_dict["runner_type"] == "generate"
    assert config.runner_type == "generate"


def test_wemm_loader_declares_embedding_contract() -> None:
    from vllm_merak.constants import HYBRID_CACHE_POLICY_PAGED
    from vllm_merak.model_executor.model_loader.wemm_embedding import (
        WeMMEmbeddingMerakForEmbedding,
        WeMMEmbeddingProcessingInfo,
    )

    assert WeMMEmbeddingMerakForEmbedding.is_pooling_model is True
    assert WeMMEmbeddingMerakForEmbedding.default_seq_pooling_type == "LAST"
    assert (
        WeMMEmbeddingMerakForEmbedding.hybrid_cache_policy
        == HYBRID_CACHE_POLICY_PAGED
    )
    info = object.__new__(WeMMEmbeddingProcessingInfo)
    assert info.get_supported_mm_limits() == {"image": None, "video": None}


def test_wemm_exposes_four_split_linear_state_copy_functions() -> None:
    from vllm.model_executor.layers.mamba.mamba_utils import get_temporal_copy_spec
    from vllm_merak.model_executor.model_loader.wemm_embedding import (
        WeMMEmbeddingMerakForEmbedding,
        _get_wemm_split_conv_copy_spec,
    )

    copy_funcs = WeMMEmbeddingMerakForEmbedding.get_mamba_state_copy_func()

    assert copy_funcs == (
        _get_wemm_split_conv_copy_spec,
        _get_wemm_split_conv_copy_spec,
        _get_wemm_split_conv_copy_spec,
        get_temporal_copy_spec,
    )

    state = torch.arange(4 * 3 * 2).reshape(4, 3, 2)
    copy_spec = copy_funcs[0](state, [3, 1], 1, 1)
    assert copy_spec.start_addr == state[1].data_ptr()
    assert copy_spec.num_elements == state[1].numel()

    with pytest.raises(NotImplementedError, match="offset copies"):
        copy_funcs[0](state, [3, 1], 0, 2)


def test_wemm_request_uses_finalized_attention_block_size() -> None:
    from vllm_merak.model_executor.model_loader.wemm_embedding import (
        WeMMEmbeddingMerakForEmbedding,
    )

    model = WeMMEmbeddingMerakForEmbedding.__new__(WeMMEmbeddingMerakForEmbedding)
    model.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=16)
    )
    model._merak_attention_block_size = 64

    assert model._get_block_size() == 64


def test_wemm_pooler_casts_to_fp32_before_normalization() -> None:
    from vllm.config import PoolerConfig
    from vllm.model_executor.layers.pooler.activations import PoolerNormalize
    from vllm.model_executor.layers.pooler.seqwise import SequencePooler
    from vllm.model_executor.layers.pooler.seqwise.heads import EmbeddingPoolerHead
    from vllm_merak.model_executor.model_loader.wemm_embedding import (
        _build_wemm_pooler,
    )

    pooler = _build_wemm_pooler(PoolerConfig(seq_pooling_type="LAST"))

    assert pooler.get_supported_tasks() == {"embed"}
    embed_pooler = pooler.poolers_by_task["embed"]
    assert isinstance(embed_pooler, SequencePooler)
    assert isinstance(embed_pooler.head, EmbeddingPoolerHead)
    assert embed_pooler.head.head_dtype == torch.float32
    assert isinstance(embed_pooler.head.activation, PoolerNormalize)


def test_wemm_hmonnx_loader_pins_ordinary_runtime_to_primary_device(
    monkeypatch,
) -> None:
    import vllm_merak.model_executor.model_loader.wemm_embedding as wemm_module
    from vllm_merak.model_executor.model_loader.wemm_embedding import (
        WeMMEmbeddingMerakForEmbedding,
    )

    captured = {}
    sentinel = object()

    def fake_load(meta_file, **kwargs):
        captured.update(meta_file=meta_file, **kwargs)
        return sentinel

    monkeypatch.setattr(
        wemm_module.AutoLLMHONNXModel,
        "from_pretrained",
        fake_load,
    )
    model = WeMMEmbeddingMerakForEmbedding.__new__(WeMMEmbeddingMerakForEmbedding)
    model.config = SimpleNamespace(use_v2=True, auto_offload=False)
    model.inference_device = torch.device("cuda:3")

    result = model._load_hmonnx_model("meta.json", {"enable_cuda_graph": True})

    assert result is sentinel
    assert captured == {
        "meta_file": "meta.json",
        "enable_cuda_graph": True,
        "device_map": [torch.device("cuda:3")],
    }


@pytest.mark.parametrize("requested_phase", [None, "decode"])
def test_wemm_runtime_forces_prefill_for_single_token_continuations(
    monkeypatch,
    requested_phase,
) -> None:
    from vllm_merak.model_executor.model_loader.qwen3_5 import (
        Qwen3_5MerakForCausalLM,
    )
    from vllm_merak.model_executor.model_loader.wemm_embedding import (
        WeMMEmbeddingMerakForEmbedding,
    )

    captured = {}

    def fake_run(self, past_len, **kwargs):
        captured.update(past_len=past_len, **kwargs)
        return torch.zeros(64)

    monkeypatch.setattr(Qwen3_5MerakForCausalLM, "_run_request", fake_run)
    model = WeMMEmbeddingMerakForEmbedding.__new__(WeMMEmbeddingMerakForEmbedding)
    kwargs = {"input_ids": torch.tensor([1])}
    if requested_phase is not None:
        kwargs["execution_phase"] = requested_phase
    model._run_request(255, **kwargs)

    assert captured["past_len"] == 255
    assert captured["execution_phase"] == "prefill"


def test_wemm_video_encoder_groups_frame_outputs_per_video() -> None:
    from vllm_merak.model_executor.model_loader.wemm_embedding import (
        WeMMEmbeddingMerakForEmbedding,
    )

    class Visual:
        device = torch.device("cpu")
        dtype = torch.float16

        @staticmethod
        def encode_many(pixel_values, frame_grids):
            assert pixel_values.shape == (12, 3)
            assert frame_grids.tolist() == [[1, 2, 2], [1, 2, 2], [1, 2, 2]]
            return [
                torch.full((1, 1, 4), 1.0),
                torch.full((1, 1, 4), 2.0),
                torch.full((1, 1, 4), 3.0),
            ]

    model = WeMMEmbeddingMerakForEmbedding.__new__(WeMMEmbeddingMerakForEmbedding)
    model.inference_model = SimpleNamespace(visual=Visual())
    model.input_embeddings = SimpleNamespace(weight=torch.empty(1, 4, dtype=torch.float16))
    model._runner_device = torch.device("cpu")
    model.spatial_merge_size = 1

    outputs = model.embed_multimodal(
        pixel_values_videos=torch.zeros(12, 3),
        video_grid_thw=torch.tensor([[2, 2, 2], [1, 2, 2]]),
    )

    assert len(outputs) == 2
    assert outputs[0].shape == (2, 4)
    assert outputs[1].shape == (1, 4)
    assert outputs[0][:, 0].tolist() == [1.0, 2.0]
    assert outputs[1][:, 0].tolist() == [3.0]


def test_wemm_routes_image_and_video_rows_to_each_request() -> None:
    from vllm_merak.model_executor.model_loader.wemm_embedding import (
        WeMMEmbeddingMerakForEmbedding,
    )

    model = WeMMEmbeddingMerakForEmbedding.__new__(WeMMEmbeddingMerakForEmbedding)
    model.vision_start_token_id = 10
    model.image_token_id = 11
    model.video_token_id = 12
    model.spatial_merge_size = 1

    input_ids = torch.tensor([10, 11, 11, 20, 10, 12, 12, 12, 21])
    inputs_embeds = torch.arange(9 * 4, dtype=torch.float32).reshape(9, 4)
    image_grids = torch.tensor([[1, 2, 2]])
    video_grids = torch.tensor([[3, 2, 2]])

    kwargs, image_offset, video_offset = model._extract_request_multimodal_kwargs(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        image_grid_thw=image_grids,
        image_grid_offset=0,
        video_grid_thw=video_grids,
        video_grid_offset=0,
    )

    assert image_offset == 1
    assert video_offset == 1
    assert torch.equal(kwargs["image_embeds"], inputs_embeds[1:3])
    assert torch.equal(kwargs["video_embeds"], inputs_embeds[5:8])
    assert torch.equal(kwargs["image_grid_thw"], image_grids)
    assert torch.equal(kwargs["video_grid_thw"], video_grids)


def test_wemm_maps_multiple_video_frame_spans_to_one_video_grid() -> None:
    from vllm_merak.model_executor.model_loader.wemm_embedding import (
        WeMMEmbeddingMerakForEmbedding,
    )

    model = WeMMEmbeddingMerakForEmbedding.__new__(WeMMEmbeddingMerakForEmbedding)
    model.vision_start_token_id = 10
    model.image_token_id = 11
    model.video_token_id = 12
    model.spatial_merge_size = 2
    # One Qwen3-VL video is expanded into two timestamped frame spans.  The
    # corresponding video grid remains one [T, H, W] row.
    input_ids = torch.tensor([10, 12, 12, 20, 10, 12, 12, 21])
    inputs_embeds = torch.arange(8 * 4, dtype=torch.float32).reshape(8, 4)
    video_grids = torch.tensor([[2, 4, 2]])

    kwargs, image_offset, video_offset = model._extract_request_multimodal_kwargs(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        image_grid_thw=None,
        image_grid_offset=0,
        video_grid_thw=video_grids,
        video_grid_offset=0,
    )

    assert image_offset == 0
    assert video_offset == 1
    assert torch.equal(kwargs["video_embeds"], inputs_embeds[[1, 2, 5, 6]])
    assert torch.equal(kwargs["video_grid_thw"], video_grids)


def test_wemm_routes_resumed_video_placeholder_without_consuming_grid() -> None:
    from vllm_merak.model_executor.model_loader.wemm_embedding import (
        WeMMEmbeddingMerakForEmbedding,
    )

    model = WeMMEmbeddingMerakForEmbedding.__new__(WeMMEmbeddingMerakForEmbedding)
    model.vision_start_token_id = 10
    model.image_token_id = 11
    model.video_token_id = 12
    input_ids = torch.tensor([12, 12])
    inputs_embeds = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    kwargs, image_offset, video_offset = model._extract_request_multimodal_kwargs(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        image_grid_thw=None,
        image_grid_offset=3,
        video_grid_thw=None,
        video_grid_offset=4,
    )

    assert image_offset == 3
    assert video_offset == 4
    assert torch.equal(kwargs["video_embeds"], inputs_embeds)
    assert "video_grid_thw" not in kwargs


def test_merak_runner_delegates_pooling_task_discovery(monkeypatch) -> None:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    from vllm_merak.worker_v1.merak_model_runner_v1 import MerakModelRunnerV1

    monkeypatch.setattr(
        GPUModelRunner,
        "get_supported_tasks",
        lambda _self: ("embed",),
    )
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.model_config = SimpleNamespace(runner_type="pooling")

    assert runner.get_supported_tasks() == ("embed",)


@pytest.mark.parametrize("runner_type", ["generate", "pooling"])
def test_merak_worker_accepts_supported_runner_types(runner_type) -> None:
    from vllm_merak.worker_v1.merak_worker_v1 import _validate_runner_type

    _validate_runner_type(runner_type)


def test_merak_worker_rejects_unknown_runner_type() -> None:
    from vllm_merak.worker_v1.merak_worker_v1 import _validate_runner_type

    with pytest.raises(NotImplementedError, match="score"):
        _validate_runner_type("score")
