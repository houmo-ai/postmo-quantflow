"""Actual plugin discovery, DeepStack merge and FP32 classifier contracts."""

import importlib.util

import pytest
import torch


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None, reason="Requires the separate vLLM service environment"
)


@pytest.fixture
def reranker_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = [
        "<unk>",
        "<|im_start|>",
        "<|im_end|>",
        "<|vision_start|>",
        "<|image_pad|>",
        "<|vision_end|>",
        "\n",
        "assistant",
        "text",
        "<|video_pad|>",
        "user",
    ]
    backend = Tokenizer(models.WordLevel({word: index for index, word in enumerate(words)}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Split("\n", behavior="isolated"), pre_tokenizers.Split(" ", behavior="removed")]
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", additional_special_tokens=words[1:6] + [words[9]]
    )


@pytest.mark.parametrize("ordinary_tokens", [0, 3, 4, 5, 40])
def test_reranker_truncation_preserves_all_special_tokens_and_final_template(reranker_tokenizer, ordinary_tokens):
    from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.processor import reranker_token_indices

    suffix = reranker_tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    assert len(suffix) == 5
    body = [1, *([8] * ordinary_tokens), 3, 4, 4, 5]
    ids = body + suffix
    selected = reranker_token_indices(ids, reranker_tokenizer, max_length=14)
    actual = [ids[index] for index in selected]
    assert actual == [1, *([8] * min(ordinary_tokens, 4)), 3, 4, 4, 5, *suffix]
    assert actual[-5:] == suffix
    assert len(actual) <= 14
    assert selected == sorted(set(selected))


def test_reranker_truncation_rejects_missing_tail_and_special_token_overflow(reranker_tokenizer):
    from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.processor import reranker_token_indices

    suffix = reranker_tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    with pytest.raises(ValueError, match="template suffix"):
        reranker_token_indices([1, 8, *suffix[:-1]], reranker_tokenizer, max_length=16)
    with pytest.raises(ValueError, match="special tokens"):
        reranker_token_indices([1, *([4] * 12), *suffix], reranker_tokenizer, max_length=16)


def test_reranker_truncation_preserves_special_tokens_added_after_vllm_cache(reranker_tokenizer):
    from vllm.tokenizers.hf import get_cached_tokenizer

    from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.processor import reranker_token_indices

    tokenizer = get_cached_tokenizer(reranker_tokenizer)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<late_visual>"]})
    visual_id = tokenizer.convert_tokens_to_ids("<late_visual>")
    assert visual_id not in tokenizer.all_special_ids
    suffix = tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    ids = [1, *([8] * 40), visual_id, *suffix]
    indices = reranker_token_indices(ids, tokenizer, max_length=14)
    assert [ids[index] for index in indices] == [1, *([8] * 7), visual_id, *suffix]


def test_reranker_truncation_relocates_late_image_without_dropping_visual_tokens(reranker_tokenizer):
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm_merak.reranker_processing import truncate_reranker_engine_input

    suffix = reranker_tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    ids = [1, *([8] * 30), 3, 4, 4, 5, *suffix]
    image_range = PlaceholderRange(offset=32, length=2)
    features = object()
    original = {"prompt_token_ids": ids, "mm_placeholders": {"image": [image_range]}, "mm_kwargs": features}
    actual = truncate_reranker_engine_input(original, reranker_tokenizer, max_length=14)
    assert actual["prompt_token_ids"] == [1, 8, 8, 8, 8, 3, 4, 4, 5, *suffix]
    assert actual["mm_placeholders"]["image"] == [PlaceholderRange(offset=6, length=2)]
    assert actual["mm_kwargs"] is features
    assert original["prompt_token_ids"] == ids
    assert original["mm_placeholders"]["image"] == [image_range]


def test_native_reranker_processor_truncates_masks_and_types_with_input_ids(tmp_path, reranker_tokenizer):
    from transformers import Qwen2VLImageProcessor, Qwen3VLProcessor, Qwen3VLVideoProcessor

    from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.processor import RerankerProcessor

    template = (
        "{% for message in messages %}<|im_start|>{{ message.role }}\n"
        "{% for part in message.content %}{{ part.text }}{% endfor %}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    hf_processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(),
        tokenizer=reranker_tokenizer,
        video_processor=Qwen3VLVideoProcessor(),
        chat_template=template,
    )
    hf_processor.save_pretrained(tmp_path)
    processor = RerankerProcessor(tmp_path, patch_size=16, context_length=24)
    messages = [{"role": "user", "content": [{"type": "text", "text": "text " * 40}]}]
    result = processor.apply_chat_template(messages)
    suffix = reranker_tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    assert result["input_ids"].shape == (1, 24)
    assert result["input_ids"][0, -5:].tolist() == suffix
    assert result["attention_mask"].shape == (1, 24)
    assert result["attention_mask"].all()
    assert result["mm_token_type_ids"].shape == (1, 24)


def test_reranker_merak_is_lazily_discovered_without_overriding_qwen3vl():
    from vllm_merak.model_executor.register import get_model_registration_paths, get_registered_model_cls

    paths = get_model_registration_paths()
    assert paths["Qwen3VLRerankerMerakForSequenceClassification"].endswith(
        "qwen3_vl_reranker:Qwen3VLRerankerMerakForSequenceClassification"
    )
    assert get_registered_model_cls("Qwen3VLForConditionalGeneration") is None


def test_reranker_processing_info_exposes_upstream_video_path():
    from vllm_merak.model_executor.model_loader.qwen3_vl_reranker import (
        Qwen3VLRerankerMerakForSequenceClassification,
        RerankerProcessingInfo,
    )

    info = object.__new__(RerankerProcessingInfo)
    assert info.get_supported_mm_limits() == {"image": None, "video": None}
    assert Qwen3VLRerankerMerakForSequenceClassification.get_placeholder_str("video", 0)


@pytest.mark.parametrize("instruction", [None, "Retrieve matching images.", '判断相关性，保留 "yes" / "no"。'])
def test_reranker_template_preserves_official_separators_and_instruction(instruction):
    from pathlib import Path

    from jinja2 import Template

    from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.processor import DEFAULT_INSTRUCTION, SYSTEM_PROMPT

    path = Path(__file__).resolve().parents[2] / "vllm_merak/examples/qwen3_vl_reranker.jinja"
    query, document = "查询中有标点：\n<|im_start|>", "<|vision_start|><|image_pad|><|vision_end|>候选文本。"
    actual = Template(path.read_text()).render(
        instruction=instruction,
        messages=[{"role": "query", "content": query}, {"role": "document", "content": document}],
    )
    assert actual == (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n"
        f"<Instruct>: {instruction or DEFAULT_INSTRUCTION}<Query>:{query}\n<Document>:{document}"
        "<|im_end|>\n<|im_start|>assistant\n"
    )


@pytest.fixture
def reranker_processing_info(tmp_path, reranker_tokenizer):
    from types import SimpleNamespace

    from transformers import Qwen2VLImageProcessor, Qwen3VLConfig, Qwen3VLProcessor, Qwen3VLVideoProcessor
    from vllm.config import MultiModalConfig
    from vllm.multimodal.processing import InputProcessingContext
    from vllm_merak.model_executor.model_loader.qwen3_vl_reranker import RerankerProcessingInfo

    config = Qwen3VLConfig(image_token_id=4, video_token_id=9, vision_start_token_id=3, vision_end_token_id=5)
    config.save_pretrained(tmp_path)
    processor = Qwen3VLProcessor(
        tokenizer=reranker_tokenizer,
        image_processor=Qwen2VLImageProcessor(patch_size=16, temporal_patch_size=2, merge_size=2),
        video_processor=Qwen3VLVideoProcessor(patch_size=16, temporal_patch_size=2, merge_size=2),
    )
    processor.save_pretrained(tmp_path)
    model_config = SimpleNamespace(
        model=str(tmp_path),
        hf_config=config,
        trust_remote_code=False,
        multimodal_config=MultiModalConfig(limit_per_prompt={"image": 8, "video": 2}),
        encoder_config={},
        max_model_len=32768,
        dtype=torch.float32,
    )
    model_config.get_multimodal_config = lambda: model_config.multimodal_config
    return RerankerProcessingInfo(InputProcessingContext(model_config, reranker_tokenizer))


@pytest.mark.parametrize("token_prompt", [False, True])
def test_reranker_video_expansion_matches_hf_outer_boundaries_and_pixels(reranker_processing_info, token_prompt):
    from types import MethodType, SimpleNamespace

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
    from vllm.model_executor.models.qwen3_vl import Qwen3VLDummyInputsBuilder, Qwen3VLForConditionalGeneration
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.multimodal.processing import ProcessorInputs, TimingContext
    from vllm_merak.model_executor.model_loader.qwen3_vl_reranker import RerankerMultiModalProcessor

    info = reranker_processing_info
    processor = RerankerMultiModalProcessor(info, Qwen3VLDummyInputsBuilder(info))
    video = torch.arange(4 * 3 * 64 * 64, dtype=torch.int64).remainder(256).to(torch.uint8).reshape(4, 3, 64, 64)
    metadata = {"fps": 2.0, "frames_indices": [0, 1, 2, 3], "total_num_frames": 4}
    prompt = "<|vision_start|><|video_pad|><|vision_end|>"
    expected = info.get_hf_processor()(
        text=[prompt],
        videos=[video],
        video_metadata=[dict(metadata)],
        do_resize=False,
        do_sample_frames=False,
        return_tensors="pt",
    )
    result = processor.apply(
        ProcessorInputs(
            prompt=info.get_tokenizer().encode(prompt) if token_prompt else prompt,
            mm_data_items=info.get_data_parser().parse_mm_data({"video": [(video, dict(metadata))]}),
            hf_processor_mm_kwargs={"do_resize": False, "do_sample_frames": False},
        ),
        TimingContext(enabled=False),
    )
    assert result["prompt_token_ids"] == expected["input_ids"][0].tolist()
    item = result["mm_kwargs"]["video"][0]
    torch.testing.assert_close(item["pixel_values_videos"].data, expected["pixel_values_videos"], rtol=0, atol=0)
    assert result["prompt_token_ids"].count(3) == 3
    assert result["prompt_token_ids"].count(5) == 3
    feature = MultiModalFeatureSpec(
        data=item, modality="video", identifier="video", mm_position=result["mm_placeholders"]["video"][0]
    )
    actual_positions, actual_delta = Qwen3VLForConditionalGeneration._get_mrope_input_positions(
        result["prompt_token_ids"], [feature], info.get_hf_config()
    )
    geometry = SimpleNamespace(config=info.get_hf_config())
    geometry.get_vision_position_ids = MethodType(Qwen3VLModel.get_vision_position_ids, geometry)
    expected_positions, expected_delta = Qwen3VLModel.get_rope_index(
        geometry, expected["input_ids"], expected["mm_token_type_ids"], video_grid_thw=expected["video_grid_thw"]
    )
    torch.testing.assert_close(actual_positions, expected_positions[:, 0], rtol=0, atol=0)
    assert actual_delta == expected_delta.item()


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_reranker_scoring_media_matches_official_image_resize_and_order(tmp_path, mode):
    import numpy as np
    from PIL import Image
    from qwen_vl_utils import fetch_image
    from vllm.multimodal.media import MediaConnector
    from vllm_merak.reranker_processing import prepare_reranker_score_content

    picture = Image.fromarray(np.arange(37 * 91 * 3, dtype=np.uint8).reshape(37, 91, 3)).convert(mode)
    if mode == "RGBA":
        picture.putalpha(128)
    path = tmp_path / "image.png"
    picture.save(path)
    content = [{"type": "text", "text": "text"}, {"type": "image_url", "image_url": {"url": path.as_uri()}}]
    prompt, media = prepare_reranker_score_content(content, MediaConnector(allowed_local_media_path=str(tmp_path)))
    expected = fetch_image({"image": picture, "min_pixels": 4096, "max_pixels": 1800 * 1024}, image_patch_size=16)
    assert prompt == "<|vision_start|><|image_pad|><|vision_end|>text"
    np.testing.assert_array_equal(np.asarray(media["image"][0]), np.asarray(expected))
    assert prepare_reranker_score_content([], MediaConnector()) == ("NULL", {})
    assert prepare_reranker_score_content("", MediaConnector()) == ("", {})


@pytest.mark.parametrize("modality", ["image", "video"])
def test_reranker_rejects_official_pixels_above_export_capacity_before_inference(reranker_processing_info, modality):
    import json
    from pathlib import Path

    from PIL import Image
    from vllm.model_executor.models.qwen3_vl import Qwen3VLDummyInputsBuilder
    from vllm_merak.model_executor.model_loader.qwen3_vl_reranker import RerankerMultiModalProcessor

    info = reranker_processing_info
    Path(info.model_id, "golden_meta_info.json").write_text(
        json.dumps(
            {
                "visual_config": {
                    "visual_input_mode": "patches",
                    "spatial_merge_size": 2,
                    "gears": [{"image_token_capacity": 4}],
                }
            }
        )
    )
    processor = RerankerMultiModalProcessor(info, Qwen3VLDummyInputsBuilder(info))
    media = (
        [Image.new("RGB", (96, 64))]
        if modality == "image"
        else [
            (
                torch.zeros(4, 3, 64, 96),
                {
                    "fps": 2,
                    "frames_indices": [0, 1, 2, 3],
                    "total_num_frames": 4,
                },
            )
        ]
    )
    with pytest.raises(ValueError, match="exceeds exported visual capacity"):
        processor(
            f"<|vision_start|><|{modality}_pad|><|vision_end|>",
            info.get_data_parser().parse_mm_data({modality: media}),
            hf_processor_mm_kwargs={"do_resize": False, "do_sample_frames": False},
        )


def test_reranker_video_io_matches_official_mp4_sampling_and_pixels(tmp_path):
    import cv2
    import numpy as np
    from qwen_vl_utils import fetch_video
    from vllm.multimodal.media import MediaConnector
    from vllm_merak.reranker_processing import RerankerVideoIO

    path = tmp_path / "sample.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (96, 64))
    assert writer.isOpened()
    for index in range(16):
        writer.write(np.full((64, 96, 3), index * 12, dtype=np.uint8))
    writer.release()
    expected, metadata = fetch_video(
        {"video": str(path), "fps": 1, "max_frames": 64}, image_patch_size=16, return_video_metadata=True
    )
    actual, actual_metadata = MediaConnector(allowed_local_media_path=str(tmp_path)).load_from_url(
        path.as_uri(), RerankerVideoIO()
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual_metadata["frames_indices"] == list(metadata["frames_indices"])
    assert actual_metadata["fps"] == metadata["fps"]
    assert actual_metadata["do_sample_frames"] is False
    with pytest.raises(RuntimeError, match="allowed-local-media-path"):
        MediaConnector().load_from_url(path.as_uri(), RerankerVideoIO())


@pytest.mark.parametrize("frame_count", [3, 66])
def test_reranker_video_frame_sequence_keeps_official_sampling_and_metadata(frame_count):
    import numpy as np
    from PIL import Image
    from qwen_vl_utils import fetch_video
    from vllm.multimodal.utils import encode_video_url
    from vllm_merak.reranker_processing import RerankerVideoIO

    frames = np.stack([np.full((64, 64, 3), index, dtype=np.uint8) for index in range(frame_count)])
    indices = np.linspace(0, frame_count - 1, min(frame_count, 64), dtype=int)
    expected, expected_metadata = fetch_video(
        {"video": [Image.fromarray(frames[i]) for i in indices], "total_pixels": 7680 * 1024},
        image_patch_size=16,
        return_video_metadata=True,
    )
    url = encode_video_url(frames)
    actual, metadata = RerankerVideoIO().load_base64("video/jpeg", url.split(",", 1)[1])
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert metadata["frames_indices"] == expected_metadata["frames_indices"]
    assert metadata["fps"] == expected_metadata["fps"]
    assert metadata["total_num_frames"] == expected_metadata["total_num_frames"]
    assert metadata["do_sample_frames"] is False
    if frame_count == 3:
        _, explicit = RerankerVideoIO(
            sequence_metadata={
                "fps": 6,
                "frames_indices": [100, 200, 300],
                "total_num_frames": 600,
                "duration": 100,
            }
        ).load_base64("video/jpeg", url.split(",", 1)[1])
        assert explicit["frames_indices"] == [100, 200, 300, 300]
        assert explicit["fps"] == 6
        assert explicit["total_num_frames"] == 600


def test_reranker_video_and_image_features_preserve_order_and_all_deepstack_levels():
    from types import SimpleNamespace

    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    from vllm_merak.model_executor.model_loader.qwen3_vl_reranker import Qwen3VLRerankerMerakForSequenceClassification

    config = Qwen3VLVisionConfig(
        depth=3,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        out_hidden_size=256,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        deepstack_visual_indexes=[0, 1, 2],
    )
    native = Qwen3VLVisionModel(config).eval()

    class NativeVisualEncoder(torch.nn.Module):
        def encode(self, pixels, grid):
            result = native(pixels, grid_thw=grid)
            return result.pooler_output[None], [part[None] for part in result.deepstack_features]

    model = object.__new__(Qwen3VLRerankerMerakForSequenceClassification)
    torch.nn.Module.__init__(model)
    model.inference_model = SimpleNamespace(visual=NativeVisualEncoder())
    model.spatial_merge_size = 2
    video_grids = torch.tensor([[2, 2, 4], [1, 4, 2]])
    video_pixels = torch.randn(24, 3 * 2 * 16 * 16)
    image_grid = torch.tensor([[1, 2, 2]])
    image_pixels = torch.randn(4, 3 * 2 * 16 * 16)
    with torch.inference_mode():
        videos = native(video_pixels, grid_thw=video_grids)
        expected_videos = torch.cat([videos.pooler_output, *videos.deepstack_features], dim=-1).split([4, 2])
        picture = native(image_pixels, grid_thw=image_grid)
        expected_image = torch.cat([picture.pooler_output, *picture.deepstack_features], dim=-1)
        actual = model.embed_multimodal(
            pixel_values_videos=video_pixels,
            video_grid_thw=video_grids,
            pixel_values=image_pixels,
            image_grid_thw=image_grid,
        )
    assert len(actual) == 3
    for result, expected in zip(actual, [*expected_videos, expected_image], strict=True):
        torch.testing.assert_close(result, expected, atol=2e-5, rtol=2e-5)


def test_reranker_truncation_keeps_video_timestamps_and_relocates_full_span(reranker_tokenizer):
    from vllm.multimodal.inputs import PlaceholderRange
    from vllm_merak.reranker_processing import truncate_reranker_engine_input

    suffix = reranker_tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    span = [8, 3, 9, 9, 5, 8, 3, 9, 9, 5]
    # Ordinary timestamp tokens inside the video span must stay aligned too.
    prompt = {
        "prompt_token_ids": [1, *([8] * 40), *span, *suffix],
        "mm_placeholders": {"video": [PlaceholderRange(offset=41, length=10)]},
    }
    result = truncate_reranker_engine_input(prompt, reranker_tokenizer, max_length=20)
    assert result["prompt_token_ids"] == [1, *([8] * 4), *span, *suffix]
    assert result["mm_placeholders"]["video"][0].offset == 5


def test_reranker_deepstack_merge_clears_features_for_following_text_request():
    from vllm_merak.model_executor.model_loader.qwen3_vl_reranker import Qwen3VLRerankerMerakForSequenceClassification

    model = object.__new__(Qwen3VLRerankerMerakForSequenceClassification)
    torch.nn.Module.__init__(model)
    model.input_embeddings = torch.nn.Embedding(16, 8)
    model.image_token_id = 2
    model.video_token_id = 4
    model.visual_dim = 8
    model.multiscale_dim = 24
    model.deepstack_num_level = 3
    ids = torch.tensor([1, 2, 2, 3])
    mask = torch.tensor([False, True, True, False])
    visual = torch.arange(64, dtype=torch.float32).reshape(2, 32)
    output = model.embed_input_ids(ids, [visual], is_multimodal=mask)
    torch.testing.assert_close(output[mask], visual[:, :8])
    for index in range(3):
        torch.testing.assert_close(
            model._deepstack_input_embeds[index, mask], visual[:, 8 * (index + 1) : 8 * (index + 2)]
        )
        assert torch.count_nonzero(model._deepstack_input_embeds[index, ~mask]) == 0
    model.embed_input_ids(ids)
    assert model._deepstack_input_embeds is None


def test_reranker_video_merge_keeps_timestamp_and_vision_delimiter_embeddings():
    from vllm_merak.model_executor.model_loader.qwen3_vl_reranker import Qwen3VLRerankerMerakForSequenceClassification

    model = object.__new__(Qwen3VLRerankerMerakForSequenceClassification)
    torch.nn.Module.__init__(model)
    model.input_embeddings = torch.nn.Embedding(16, 8)
    model.image_token_id, model.video_token_id = 2, 4
    model.visual_dim, model.multiscale_dim, model.deepstack_num_level = 8, 24, 3
    ids = torch.tensor([1, 3, 4, 4, 5, 6])
    text_embeddings = model.input_embeddings(ids).detach()
    visual = torch.arange(64, dtype=torch.float32).reshape(2, 32)
    result = model.embed_input_ids(ids, [visual], is_multimodal=torch.tensor([False, True, True, True, True, False]))
    torch.testing.assert_close(result[[0, 1, 4, 5]], text_embeddings[[0, 1, 4, 5]])
    torch.testing.assert_close(result[2:4], visual[:, :8])
    assert torch.count_nonzero(model._deepstack_input_embeds[:, [0, 1, 4, 5]]) == 0


def test_reranker_pooler_selects_final_valid_logit_and_uses_fp32_sigmoid():
    from vllm.pooling_params import PoolingParams
    from vllm.v1.pool.metadata import PoolingCursor, PoolingMetadata, PoolingStates
    from vllm_merak.model_executor.model_loader.qwen3_vl_reranker import build_reranker_pooler

    lengths = torch.tensor([3, 2])
    metadata = PoolingMetadata(
        prompt_lens=lengths,
        prompt_token_ids=None,
        prompt_token_ids_cpu=None,
        pooling_params=[PoolingParams(task="classify", use_activation=True) for _ in range(2)],
        pooling_states=[PoolingStates() for _ in range(2)],
        pooling_cursor=PoolingCursor(torch.tensor([0, 3]), torch.tensor([2, 4]), lengths, lengths, lengths),
    )
    logits = torch.tensor([[99.0], [98.0], [0.125], [-99.0], [-0.25]], dtype=torch.float16)
    result = build_reranker_pooler()(logits, metadata)
    torch.testing.assert_close(
        torch.stack(result) if isinstance(result, list) else result, logits[[2, 4]].float().sigmoid(), atol=0, rtol=0
    )


def test_internal_hmonnx_scheduler_cache_device_does_not_inspect_flash_nodes():
    from types import SimpleNamespace

    from vllm_merak.worker_v1.merak_model_runner_v1 import MerakModelRunnerV1

    runner = object.__new__(MerakModelRunnerV1)
    runner.device = torch.device("cpu")
    runner.model = SimpleNamespace(
        uses_internal_hmonnx_cache=True,
        enable_page_attention=False,
        inference_device=torch.device("cuda:0"),
        placeholder_attn_layers=[SimpleNamespace(layer_name="model.layers.0.attn")],
    )
    assert runner._get_kv_cache_layer_devices() == {"model.layers.0.attn": torch.device("cuda:0")}
    runner.model.enable_page_attention = True
    with pytest.raises(ValueError, match="Internal HMONNX cache"):
        runner._get_kv_cache_layer_devices()


def test_reranker_metadata_selects_classification_without_changing_source_architecture(tmp_path):
    import json

    import onnx
    from onnx import TensorProto, helper
    from vllm_merak.config import MerakConfigParser

    (tmp_path / "hf_config").mkdir()
    source = {
        "model_type": "qwen3_vl",
        "architectures": ["Qwen3VLForConditionalGeneration"],
        "text_config": {
            "hidden_size": 256,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_hidden_layers": 1,
            "vocab_size": 64,
            "max_position_embeddings": 128,
        },
    }
    (tmp_path / "hf_config/config.json").write_text(json.dumps(source))
    (tmp_path / "embedding.pt").touch()
    output = helper.make_tensor_value_info("score_logits", TensorProto.FLOAT16, [1, 1, 64])
    graph = helper.make_graph([], "score_abi", [], [output])
    onnx.save(helper.make_model(graph), tmp_path / "prefill.onnx")
    metadata = {
        "model_config": {"model_type": "Qwen3VLForConditionalGeneration_reranker"},
        "hf_config": "hf_config",
        "quant_embedding": "embedding.pt",
        "prefill_hmonnx": "prefill.onnx",
        "reranker": {
            "version": 1,
            "output": "score_logits",
            "score_index": 0,
            "head_width": 64,
            "activation": "sigmoid",
            "projection": "yes_minus_no",
        },
        "visual_config": {"hmonnx": "visual.onnx"},
    }
    (tmp_path / "visual.onnx").touch()
    (tmp_path / "golden_meta_info.json").write_text(json.dumps(metadata))
    (tmp_path / "merak_config.json").write_text(
        json.dumps(
            {
                "runner_type": "pooling",
                "num_labels": 1,
                "enable_page_attention": False,
                "xh_model": {"model_type": "hmonnx", "meta_info": "golden_meta_info.json"},
            }
        )
    )
    parsed, _ = MerakConfigParser().parse(tmp_path, trust_remote_code=False)
    assert parsed["architectures"] == ["Qwen3VLRerankerMerakForSequenceClassification"]
    assert parsed["runner_type"] == "pooling"
    assert parsed["num_labels"] == 1
    assert json.loads((tmp_path / "hf_config/config.json").read_text()) == source
    metadata["reranker"]["score_index"] = 1
    (tmp_path / "golden_meta_info.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="score contract"):
        MerakConfigParser().parse(tmp_path, trust_remote_code=False)


def test_pooling_scheduler_consumes_exact_capacity_on_resumed_prefill():
    from types import SimpleNamespace

    from vllm.config import CacheConfig, DeviceConfig, SchedulerConfig, VllmConfig
    from vllm.pooling_params import PoolingParams
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
    from vllm.v1.request import Request
    from vllm_merak.worker_v1.pooling_scheduler import MerakPoolingScheduler

    counts = []
    for cls in (Scheduler, MerakPoolingScheduler):
        config = VllmConfig(device_config=DeviceConfig(device="cpu"))
        config.model_config = SimpleNamespace(
            runner_type="pooling",
            max_model_len=8,
            is_encoder_decoder=False,
            is_multimodal_model=False,
            enable_return_routed_experts=False,
        )
        config.cache_config = CacheConfig(block_size=64, enable_prefix_caching=False)
        config.cache_config.num_gpu_blocks = 16
        config.scheduler_config = SchedulerConfig(
            max_model_len=8,
            max_num_batched_tokens=4,
            is_encoder_decoder=False,
            max_num_seqs=1,
            enable_chunked_prefill=True,
        )
        spec = FullAttentionSpec(block_size=64, num_kv_heads=1, head_size=1, dtype=torch.float16)
        cache = KVCacheConfig(
            num_blocks=16, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["placeholder"], spec)]
        )
        register_all_kvcache_specs(config)
        scheduler = cls(config, cache, structured_output_manager=None, block_size=64)
        request = Request(
            "exact_capacity", [1] * 8, sampling_params=None, pooling_params=PoolingParams(task="classify")
        )
        scheduler.add_request(request)
        first = scheduler.schedule()
        second = scheduler.schedule()
        counts.append([first.num_scheduled_tokens[request.request_id], second.num_scheduled_tokens[request.request_id]])
        assert scheduler.max_model_len == 8
    assert counts == [[4, 3], [4, 4]]


def test_materializing_external_weights_preserves_bytes_and_golden(tmp_path):
    import hashlib
    import importlib.util
    import os
    from pathlib import Path

    import onnx
    from onnx import TensorProto, helper
    from onnx.external_data_helper import set_external_data

    spec = importlib.util.spec_from_file_location(
        "reranker_example", Path(__file__).resolve().parents[2] / "vllm_merak/examples/vllm_merak_reranker.py"
    )
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    original = torch.arange(64, dtype=torch.float16).numpy().tobytes()
    payload = tmp_path / "weights.bin"
    payload.write_bytes(original)
    golden = tmp_path / "golden_weights.bin"
    os.link(payload, golden)
    tensor = helper.make_tensor("score_logits", TensorProto.FLOAT16, [1, 1, 64], original, raw=True)
    set_external_data(tensor, location="weights.bin", offset=0, length=len(original))
    tensor.ClearField("raw_data")
    model = helper.make_model(
        helper.make_graph(
            [],
            "external",
            [],
            [helper.make_tensor_value_info("score_logits", TensorProto.FLOAT16, [1, 1, 64])],
            [tensor],
        )
    )
    onnx.save(model, tmp_path / "prefill.onnx")
    metadata = {"prefill_hmonnx": "prefill.onnx", "visual_config": {"gears": [{"hmonnx": "prefill.onnx"}]}}
    example.materialize_external_data(tmp_path.resolve(), metadata)
    assert payload.stat().st_nlink == 1
    assert golden.stat().st_nlink == 1
    assert payload.read_bytes() == golden.read_bytes() == original
    assert hashlib.sha256(payload.read_bytes()).digest() == hashlib.sha256(original).digest()
    assert not list(tmp_path.glob(".merak-external-*"))
    # The real ONNX loader must now accept and read the previously linked data.
    loaded = onnx.load(tmp_path / "prefill.onnx")
    assert loaded.graph.initializer[0].raw_data == original
