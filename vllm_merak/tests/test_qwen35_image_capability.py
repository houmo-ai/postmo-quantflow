import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm_merak.config import MerakConfigParser
from vllm_merak.model_executor import _get_architecture_candidates
from vllm_merak.model_executor.model_loader import qwen3_5 as qwen35


def _write_qwen35_export(tmp_path, *, visual_hmonnx: str | None) -> None:
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
    (tmp_path / "merak_config.json").write_text(
        json.dumps({"xh_model": {"model_type": "hmonnx"}}),
        encoding="utf-8",
    )
    meta = {"model_config": {"model_type": "Qwen3_5ForConditionalGeneration"}}
    if visual_hmonnx is not None:
        meta["visual_config"] = {
            "hmonnx": visual_hmonnx,
            "image_size_w": 448,
            "image_size_h": 448,
            "patch_size": 16,
            "spatial_merge_size": 2,
        }
    (tmp_path / "golden_meta_info.json").write_text(json.dumps(meta), encoding="utf-8")


def test_qwen35_export_with_visual_selects_multimodal_model(tmp_path) -> None:
    visual_path = tmp_path / "visual" / "visual.hmonnx"
    visual_path.parent.mkdir()
    visual_path.touch()
    _write_qwen35_export(tmp_path, visual_hmonnx="visual/visual.hmonnx")

    config_dict, config = MerakConfigParser().parse(tmp_path, trust_remote_code=False)

    assert config_dict["architectures"] == ["Qwen3_5MerakForCausalLM"]
    assert config_dict["merak_loader_architecture"] == "Qwen3_5MerakForCausalLM"
    assert config_dict["image_input_supported"] is True
    assert config.image_input_supported is True


def test_qwen35_export_without_visual_selects_text_only_model(tmp_path) -> None:
    _write_qwen35_export(tmp_path, visual_hmonnx=None)

    config_dict, config = MerakConfigParser().parse(tmp_path, trust_remote_code=False)

    assert config_dict["architectures"] == ["Qwen3_5MerakTextForCausalLM"]
    assert config_dict["merak_loader_architecture"] == "Qwen3_5MerakTextForCausalLM"
    assert config_dict["image_input_supported"] is False
    assert config.image_input_supported is False


def test_text_only_runtime_repairs_legacy_null_spatial_merge_size() -> None:
    meta = SimpleNamespace(
        model_config=SimpleNamespace(
            spatial_merge_size=None,
            visual_config={"spatial_merge_size": 2},
        )
    )

    qwen35._repair_text_only_runtime_model_config(meta)

    assert meta.model_config.spatial_merge_size == 2


def test_qwen35_export_rejects_missing_declared_visual_hmonnx(tmp_path) -> None:
    _write_qwen35_export(tmp_path, visual_hmonnx="visual/missing.hmonnx")

    with pytest.raises(FileNotFoundError, match="visual HMONNX.*missing.hmonnx"):
        MerakConfigParser().parse(tmp_path, trust_remote_code=False)


def test_qwen35_export_rejects_missing_token_gear_hmonnx(tmp_path) -> None:
    largest_visual = tmp_path / "visual" / "gears" / "m196" / "visual.onnx"
    largest_visual.parent.mkdir(parents=True)
    largest_visual.touch()
    _write_qwen35_export(
        tmp_path,
        visual_hmonnx="visual/gears/m196/visual.onnx",
    )
    meta_path = tmp_path / "golden_meta_info.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["visual_config"].update(
        {
            "visual_input_mode": "patches",
            "attention_mask_format": "additive_key_padding_bias",
            "attention_mask_shape": "[1,1,1,patch_token_capacity]",
            "attention_mask_operator": "xhquant.nn.MaskedAdd+Softmax",
            "rotary_position_format": "height_width_2d_ids",
            "visual_rope_cache_length": 392,
            "spatial_merge_size": 2,
            "gears": [
                {
                    "image_token_capacity": 96,
                    "patch_token_capacity": 384,
                    "hmonnx": "visual/gears/m96/missing.onnx",
                },
                {
                    "image_token_capacity": 196,
                    "patch_token_capacity": 784,
                    "hmonnx": "visual/gears/m196/visual.onnx",
                },
            ],
        }
    )
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="visual gear 96 HMONNX"):
        MerakConfigParser().parse(tmp_path, trust_remote_code=False)


def test_qwen35_export_rejects_causal_mask_contract_for_token_gears(tmp_path) -> None:
    visual_path = tmp_path / "visual" / "gears" / "m96" / "visual.onnx"
    visual_path.parent.mkdir(parents=True)
    visual_path.touch()
    _write_qwen35_export(tmp_path, visual_hmonnx="visual/gears/m96/visual.onnx")
    meta_path = tmp_path / "golden_meta_info.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["visual_config"].update(
        {
            "visual_input_mode": "patches",
            "attention_mask_format": "additive_key_padding_bias",
            "attention_mask_shape": "[1,1,1,patch_token_capacity]",
            "attention_mask_operator": "xhquant.nn.MaskedSoftmax",
            "rotary_position_format": "height_width_2d_ids",
            "visual_rope_cache_length": 192,
            "spatial_merge_size": 2,
            "gears": [
                {
                    "image_token_capacity": 96,
                    "patch_token_capacity": 384,
                    "hmonnx": "visual/gears/m96/visual.onnx",
                }
            ],
        }
    )
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match=r"MaskedAdd\+Softmax"):
        MerakConfigParser().parse(tmp_path, trust_remote_code=False)


def test_qwen35_concrete_classes_advertise_distinct_image_capabilities() -> None:
    assert qwen35.Qwen3_5MerakForCausalLM.supports_multimodal is True
    text_model_cls = getattr(qwen35, "Qwen3_5MerakTextForCausalLM", None)
    assert text_model_cls is not None
    assert text_model_cls.supports_multimodal is False
    assert text_model_cls.requires_raw_input_tokens is True
    assert text_model_cls.supports_mrope is True


def test_qwen35_loader_architecture_has_priority_over_source_architecture() -> None:
    hf_config = SimpleNamespace(
        merak_loader_architecture="Qwen3_5MerakTextForCausalLM",
        source_architectures=["Qwen3_5ForConditionalGeneration"],
        architectures=["Qwen3_5MerakTextForCausalLM"],
    )

    field, candidates = _get_architecture_candidates(hf_config)

    assert field == "merak_loader_architecture"
    assert candidates == ["Qwen3_5MerakTextForCausalLM"]


def test_qwen35_visual_runtime_must_be_on_gpu() -> None:
    with pytest.raises(RuntimeError, match="visual HMONNX.*GPU"):
        qwen35.Qwen3_5MerakForCausalLM._validate_visual_gpu_runtime(
            SimpleNamespace(device=torch.device("cpu"))
        )

    assert qwen35.Qwen3_5MerakForCausalLM._validate_visual_gpu_runtime(
        SimpleNamespace(device=torch.device("cuda:3"))
    ) == torch.device("cuda:3")


def test_qwen35_multimodal_embeddings_are_staged_to_merge_tensor_device_and_dtype() -> (
    None
):
    inputs_embeds = torch.empty((4, 8), device="meta", dtype=torch.float16)
    image_embeds = (torch.ones((2, 8), device="cpu", dtype=torch.float32),)

    normalized = qwen35.Qwen3_5MerakForCausalLM._stage_multimodal_embeddings_for_merge(
        inputs_embeds,
        image_embeds,
    )

    assert len(normalized) == 1
    assert normalized[0].device == inputs_embeds.device
    assert normalized[0].dtype == inputs_embeds.dtype


def test_qwen35_staging_preserves_batched_multimodal_tensor_shape() -> None:
    inputs_embeds = torch.empty((4, 8), device="meta", dtype=torch.float16)
    image_embeds = torch.ones((2, 8), device="cpu", dtype=torch.float32)

    normalized = qwen35.Qwen3_5MerakForCausalLM._stage_multimodal_embeddings_for_merge(
        inputs_embeds,
        image_embeds,
    )

    assert isinstance(normalized, torch.Tensor)
    assert normalized.shape == image_embeds.shape
    assert normalized.device == inputs_embeds.device
    assert normalized.dtype == inputs_embeds.dtype


def test_qwen35_merak_does_not_set_a_model_level_image_count_limit() -> None:
    info = qwen35.MerakQwen3_5ProcessingInfo.__new__(qwen35.MerakQwen3_5ProcessingInfo)

    assert info.get_supported_mm_limits() == {"image": None}


def test_qwen35_token_gears_configure_dynamic_pixel_range() -> None:
    image_processor = SimpleNamespace(
        size={"shortest_edge": 4096, "longest_edge": 1_003_520}
    )
    visual_config = {
        "visual_input_mode": "patches",
        "patch_size": 16,
        "spatial_merge_size": 2,
        "gears": [
            {"image_token_capacity": 96, "patch_token_capacity": 384},
            {"image_token_capacity": 196, "patch_token_capacity": 784},
        ],
    }

    qwen35._configure_merak_image_processor(image_processor, visual_config)

    assert image_processor.size["shortest_edge"] == 4096
    assert image_processor.size["longest_edge"] == 784 * 16 * 16
    assert image_processor.min_pixels == 4096
    assert image_processor.max_pixels == 784 * 16 * 16


def test_qwen35_token_gear_processor_preserves_source_image(monkeypatch) -> None:
    visual_config = {
        "visual_input_mode": "patches",
        "image_token_gears": [96, 196],
    }
    monkeypatch.setattr(
        qwen35,
        "_load_merak_visual_config",
        lambda _model_id, *, trust_remote_code: visual_config,
    )
    captured = {}

    def fake_hf_processor(self, prompt, mm_data, mm_kwargs, tok_kwargs):
        del self, prompt, mm_kwargs, tok_kwargs
        captured.update(mm_data)
        return {
            "pixel_values": torch.zeros((8, 6)),
            "image_grid_thw": torch.tensor([[1, 2, 4]]),
        }

    monkeypatch.setattr(
        qwen35.Qwen3VLMultiModalProcessor,
        "_call_hf_processor",
        fake_hf_processor,
    )
    processor = qwen35.MerakQwen3_5MultiModalProcessor.__new__(
        qwen35.MerakQwen3_5MultiModalProcessor
    )
    processor.info = SimpleNamespace(
        model_id="unused",
        ctx=SimpleNamespace(
            model_config=SimpleNamespace(trust_remote_code=False),
        ),
    )
    source_image = object()

    outputs = processor._call_hf_processor(
        prompt="describe",
        mm_data={"images": [source_image]},
        mm_kwargs={},
        tok_kwargs={},
    )

    assert captured["images"][0] is source_image
    assert "pixel_values" in outputs
    assert "image_grid_thw" in outputs
    assert "hm_pixel_values" not in outputs


def test_qwen35_visual_encoder_preserves_multiple_image_order() -> None:
    model = qwen35.Qwen3_5MerakForCausalLM.__new__(qwen35.Qwen3_5MerakForCausalLM)
    model._runner_device = torch.device("cpu")
    model.input_embeddings = SimpleNamespace(weight=torch.empty(1, dtype=torch.float16))

    class FakeVisual:
        device = torch.device("cpu")
        dtype = torch.float32

        def forward(self, pixels):
            marker = float(pixels.flatten()[0].item())
            return torch.full((1, 2, 3), marker)

    model.inference_model = SimpleNamespace(visual=FakeVisual())

    embeddings = model.embed_multimodal(
        hm_pixel_values=(
            torch.full((1, 1), 1.0),
            torch.full((1, 1), 2.0),
        )
    )

    assert isinstance(embeddings, tuple)
    assert len(embeddings) == 2
    assert embeddings[0].shape == (2, 3)
    assert embeddings[1].shape == (2, 3)
    assert embeddings[0].unique().item() == 1
    assert embeddings[1].unique().item() == 2


def test_qwen35_visual_token_gears_encode_many_once_and_preserve_order() -> None:
    model = qwen35.Qwen3_5MerakForCausalLM.__new__(qwen35.Qwen3_5MerakForCausalLM)
    model._runner_device = torch.device("cpu")
    model.input_embeddings = SimpleNamespace(weight=torch.empty(1, dtype=torch.float16))

    class FakeVisualTokenGears:
        device = torch.device("cpu")
        dtype = torch.float32

        def __init__(self):
            self.calls = []

        def encode_many(self, pixels, grids):
            self.calls.append((pixels, grids))
            patch_counts = [int(grid.prod()) for grid in grids]
            images = pixels.split(patch_counts, dim=0)
            return [
                torch.full(
                    (1, patch_count // 4, 3),
                    float(image[0, 0]),
                )
                for image, patch_count in zip(images, patch_counts, strict=True)
            ]

    visual = FakeVisualTokenGears()
    model.inference_model = SimpleNamespace(visual=visual)
    image_grid_thw = torch.tensor([[1, 2, 4], [1, 4, 4]])
    pixel_values = torch.cat(
        (
            torch.full((8, 6), 1.0),
            torch.full((16, 6), 2.0),
        )
    )

    embeddings = model.embed_multimodal(
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )

    assert len(visual.calls) == 1
    assert visual.calls[0][0] is pixel_values
    assert visual.calls[0][1] is image_grid_thw
    assert isinstance(embeddings, tuple)
    assert [embedding.shape for embedding in embeddings] == [(2, 3), (4, 3)]
    assert embeddings[0].unique().item() == 1
    assert embeddings[1].unique().item() == 2


def test_qwen35_request_extracts_multiple_image_grids_and_embeddings_in_order() -> None:
    model = qwen35.Qwen3_5MerakForCausalLM.__new__(qwen35.Qwen3_5MerakForCausalLM)
    model.image_token_id = 99
    model.video_token_id = 100
    model.vision_start_token_id = 98
    input_ids = torch.tensor([98, 99, 99, 7, 98, 99, 99, 99, 8])
    inputs_embeds = torch.arange(36, dtype=torch.float32).reshape(9, 4)
    image_grid_thw = torch.tensor([[1, 2, 4], [1, 4, 6]])

    kwargs, image_offset, video_offset = model._extract_request_multimodal_kwargs(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        image_grid_thw=image_grid_thw,
        image_grid_offset=0,
        video_grid_thw=None,
        video_grid_offset=0,
    )

    image_mask = input_ids == model.image_token_id
    assert torch.equal(kwargs["image_embeds"], inputs_embeds[image_mask])
    assert torch.equal(kwargs["image_grid_thw"], image_grid_thw)
    assert image_offset == 2
    assert video_offset == 0


def test_qwen35_request_rejects_missing_grid_from_multiple_images() -> None:
    model = qwen35.Qwen3_5MerakForCausalLM.__new__(qwen35.Qwen3_5MerakForCausalLM)
    model.image_token_id = 99
    model.video_token_id = 100
    model.vision_start_token_id = 98

    with pytest.raises(ValueError, match="expected 2 image grids, got 1"):
        model._extract_request_multimodal_kwargs(
            input_ids=torch.tensor([98, 99, 7, 98, 99, 8]),
            inputs_embeds=torch.zeros((6, 4)),
            image_grid_thw=torch.tensor([[1, 2, 4]]),
            image_grid_offset=0,
            video_grid_thw=None,
            video_grid_offset=0,
        )


def _run_qwen35_two_image_prefill() -> None:
    model = qwen35.Qwen3_5MerakForCausalLM.__new__(qwen35.Qwen3_5MerakForCausalLM)
    model.inference_device = torch.device("cpu")
    model.input_embeddings = SimpleNamespace(weight=torch.empty(1, dtype=torch.float16))
    model.logits_processor = SimpleNamespace(vocab_size=2)
    model.image_token_id = 99
    model._prefer_input_ids_runtime = True
    model._ensure_runtime_cache_tensors = lambda: None

    class FakeProcessor:
        rope_deltas = None

        def __init__(self):
            self.image_embed_chunks = []

        def __call__(self, data):
            self.image_embed_chunks.append(data.get("image_embeds"))
            padded_len = 4
            return (
                torch.zeros((1, padded_len, 2)),
                torch.full((padded_len,), -1, dtype=torch.long),
                torch.full((padded_len,), -2, dtype=torch.long),
                torch.full((padded_len,), -3, dtype=torch.long),
            )

    class FakeRuntime:
        def __init__(self):
            self.processor = FakeProcessor()
            self.forward_calls = []

        def set_prefill(self):
            pass

        def get_input_sequence_length(self):
            return 4

        def get_data_preprocessor(self):
            return self.processor

        def forward(self, *args):
            self.forward_calls.append(args)
            return torch.zeros((1, 4, 2))

    runtime = FakeRuntime()
    model.inference_model = runtime
    input_ids = torch.tensor([98, 99, 99, 7, 98, 99, 99, 99, 99, 8])
    image_embeds = torch.arange(12, dtype=torch.float16).reshape(6, 2)
    positions = torch.stack(
        [torch.arange(10), torch.arange(10) + 100, torch.arange(10) + 200]
    )

    model._run_request(
        0,
        input_ids=input_ids,
        positions=positions,
        image_embeds=image_embeds,
        image_grid_thw=torch.tensor([[1, 2, 4], [1, 4, 8]]),
    )

    assert len(runtime.forward_calls) == 3
    assert torch.equal(runtime.processor.image_embed_chunks[0], image_embeds[:2])
    assert torch.equal(runtime.processor.image_embed_chunks[1], image_embeds[2:5])
    assert torch.equal(runtime.processor.image_embed_chunks[2], image_embeds[5:])
    assert torch.equal(runtime.forward_calls[0][1], positions[0, :4])
    assert torch.equal(runtime.forward_calls[1][2], positions[1, 4:8])
    assert torch.equal(runtime.forward_calls[2][3][:2], positions[2, 8:10])


def test_qwen35_runtime_routes_two_images_across_prefill_chunks_in_order() -> None:
    _run_qwen35_two_image_prefill()


def test_qwen35_does_not_use_gemma_atomic_chunk_planner(monkeypatch) -> None:
    from vllm_merak.model_executor.model_loader import gemma4

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Qwen3.5 must not use the Gemma4 atomic chunk planner")

    monkeypatch.setattr(gemma4, "plan_gemma4_prefill_chunks", fail_if_called)
    _run_qwen35_two_image_prefill()


def test_qwen35_runtime_routes_cross_chunk_image_embeddings_and_mrope_positions() -> (
    None
):
    model = qwen35.Qwen3_5MerakForCausalLM.__new__(qwen35.Qwen3_5MerakForCausalLM)
    model.inference_device = torch.device("cpu")
    model.input_embeddings = SimpleNamespace(weight=torch.empty(1, dtype=torch.float16))
    model.logits_processor = SimpleNamespace(vocab_size=2)
    model.image_token_id = 99
    model._prefer_input_ids_runtime = True
    model._ensure_runtime_cache_tensors = lambda: None

    class FakeProcessor:
        rope_deltas = None

        def __init__(self):
            self.image_embed_chunks = []

        def __call__(self, data):
            self.image_embed_chunks.append(data.get("image_embeds"))
            padded_len = 4
            return (
                torch.zeros((1, padded_len, 2)),
                torch.full((padded_len,), -1, dtype=torch.long),
                torch.full((padded_len,), -2, dtype=torch.long),
                torch.full((padded_len,), -3, dtype=torch.long),
            )

    class FakeRuntime:
        def __init__(self):
            self.processor = FakeProcessor()
            self.forward_calls = []

        def set_prefill(self):
            pass

        def get_input_sequence_length(self):
            return 4

        def get_data_preprocessor(self):
            return self.processor

        def forward(self, *args):
            self.forward_calls.append(args)
            return torch.zeros((1, 4, 2))

    runtime = FakeRuntime()
    model.inference_model = runtime
    input_ids = torch.tensor([1, 2, 3, 99, 99, 99, 99, 99, 4, 5])
    image_embeds = torch.arange(10, dtype=torch.float16).reshape(5, 2)
    positions = torch.stack(
        [torch.arange(10), torch.arange(10) + 100, torch.arange(10) + 200]
    )

    model._run_request(
        0,
        input_ids=input_ids,
        positions=positions,
        image_embeds=image_embeds,
        image_grid_thw=torch.tensor([[1, 2, 10]]),
    )

    assert len(runtime.forward_calls) == 3
    assert [call[0].shape[1] for call in runtime.forward_calls] == [4, 4, 4]
    assert torch.equal(runtime.processor.image_embed_chunks[0], image_embeds[:1])
    assert torch.equal(runtime.processor.image_embed_chunks[1], image_embeds[1:5])
    assert runtime.processor.image_embed_chunks[2] is None
    assert torch.equal(runtime.forward_calls[0][1], positions[0, :4])
    assert torch.equal(runtime.forward_calls[1][2], positions[1, 4:8])
    assert torch.equal(runtime.forward_calls[2][3][:2], positions[2, 8:10])
    assert runtime.forward_calls[2][3][2:].tolist() == [-3, -3]


def test_qwen35_runtime_rope_placeholder_uses_hmonnx_processor_device() -> None:
    processor = SimpleNamespace(_device=torch.device("cpu"), rope_deltas=None)
    positions = torch.empty((3, 2), device="meta", dtype=torch.long)

    qwen35.Qwen3_5MerakForCausalLM._prepare_runtime_data_processor_for_positions(
        SimpleNamespace(),
        processor,
        positions,
    )

    assert processor.rope_deltas.device == processor._device


def test_qwen35_request_extracts_image_embeddings_without_requiring_vision_start() -> (
    None
):
    model = qwen35.Qwen3_5MerakForCausalLM.__new__(qwen35.Qwen3_5MerakForCausalLM)
    model.image_token_id = 99
    model.video_token_id = 100
    model.vision_start_token_id = 98
    input_ids = torch.tensor([99, 99, 7])
    inputs_embeds = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    kwargs, image_offset, video_offset = model._extract_request_multimodal_kwargs(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        image_grid_thw=None,
        image_grid_offset=2,
        video_grid_thw=None,
        video_grid_offset=0,
    )

    assert torch.equal(kwargs["image_embeds"], inputs_embeds[:2])
    assert "image_grid_thw" not in kwargs
    assert image_offset == 2
    assert video_offset == 0


def test_qwen35_request_handles_vision_start_split_from_image_tokens() -> None:
    model = qwen35.Qwen3_5MerakForCausalLM.__new__(qwen35.Qwen3_5MerakForCausalLM)
    model.image_token_id = 99
    model.video_token_id = 100
    model.vision_start_token_id = 98

    first_kwargs, image_offset, video_offset = model._extract_request_multimodal_kwargs(
        input_ids=torch.tensor([7, 98]),
        inputs_embeds=torch.zeros((2, 4)),
        image_grid_thw=None,
        image_grid_offset=0,
        video_grid_thw=None,
        video_grid_offset=0,
    )
    resumed_embeds = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    resumed_kwargs, image_offset, video_offset = (
        model._extract_request_multimodal_kwargs(
            input_ids=torch.tensor([99, 99, 8]),
            inputs_embeds=resumed_embeds,
            image_grid_thw=None,
            image_grid_offset=image_offset,
            video_grid_thw=None,
            video_grid_offset=video_offset,
        )
    )

    assert first_kwargs == {}
    assert torch.equal(resumed_kwargs["image_embeds"], resumed_embeds[:2])
    assert "image_grid_thw" not in resumed_kwargs
    assert image_offset == 0
    assert video_offset == 0


def test_qwen35_request_handles_resumed_chunk_starting_with_an_image() -> None:
    model = qwen35.Qwen3_5MerakForCausalLM.__new__(qwen35.Qwen3_5MerakForCausalLM)
    model.image_token_id = 99
    model.video_token_id = 100
    model.vision_start_token_id = 98
    inputs_embeds = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    kwargs, image_offset, video_offset = model._extract_request_multimodal_kwargs(
        input_ids=torch.tensor([98, 99, 99, 7]),
        inputs_embeds=inputs_embeds,
        image_grid_thw=None,
        image_grid_offset=3,
        video_grid_thw=None,
        video_grid_offset=0,
    )

    assert torch.equal(kwargs["image_embeds"], inputs_embeds[1:3])
    assert "image_grid_thw" not in kwargs
    assert image_offset == 3
    assert video_offset == 0


def test_qwen35_example_does_not_require_image_before_text() -> None:
    example = runpy.run_path(
        str(Path(__file__).parents[1] / "examples" / "vllm_merak_generate.py")
    )
    messages = example["_build_user_messages"](
        "describe", "https://example.com/image.png"
    )

    content = messages[0]["content"]
    assert isinstance(content, list)
    assert [item["type"] for item in content] == ["text", "image_url"]


def test_qwen35_text_runtime_uses_nested_visual_shape_without_loading_visual(
    monkeypatch,
) -> None:
    def fake_base_init(self, meta_info, **kwargs):
        del kwargs
        self.meta_info = meta_info
        self.kvcache_config = SimpleNamespace()
        self.enable_page_attention = False

    class FakeCacheMixin:
        split_conv_cache = False

        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(qwen35.VisonLLMHMONNXModel, "__init__", fake_base_init)
    monkeypatch.setattr(qwen35, "Qwen3_5HMONNXKVCacheMixin", FakeCacheMixin)
    meta_info = SimpleNamespace(
        model_config=SimpleNamespace(
            visual_config={"max_size_w": 640, "max_size_h": 384, "patch_size": 16},
            split_conv_cache=False,
        )
    )

    runtime = qwen35._TextOnlyXHQwen3_5HMONNXModel(meta_info)

    assert not hasattr(runtime, "visual")
    assert runtime.visual_meta.max_size_w == 640
    assert runtime.visual_meta.max_size_h == 384
