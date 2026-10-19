# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Real tiny GLM vision and hybrid language blocks, including HMONNX round-trip."""

import json
from pathlib import Path

import pytest
import torch
from test_glm53_static import tiny_config
from transformers.models.glm5_next import Glm5NextConfig, Glm5NextForConditionalGeneration, Glm5NextVisionConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel

from xhmodel_merak.xh_llm.models.glm5_3_flash.vision import (
    Glm53VisionEncoder,
    StaticVisionModel,
    fuse_multimodal_embeddings,
    load_glm53_video,
    vision_grid_frames,
)


def vision_config():
    return Glm5NextVisionConfig(
        depth=2,
        hidden_size=256,
        num_heads=4,
        intermediate_size=48,
        out_hidden_size=32,
        projection_intermediate_size=64,
        patch_size=2,
        image_size=8,
        temporal_patch_size=2,
        spatial_merge_size=2,
        swiglu_limit=0.2,
    )


def tiny_multimodal_config():
    text = tiny_config()
    text.num_hidden_layers = 2
    text.layer_types = ["linear_attention", "deepseek_sparse_attention"]
    text.mlp_layer_types = ["dense", "sparse"]
    text.indexer_types = ["full", "full"]
    return Glm5NextConfig(
        text_config=text.to_dict(),
        vision_config=vision_config().to_dict(),
        image_token_id=50,
        video_token_id=51,
        image_start_token_id=52,
        image_end_token_id=53,
        video_start_token_id=54,
        video_end_token_id=55,
    )


@pytest.mark.parametrize("grids", [[[1, 4, 6]], [[2, 4, 4], [1, 2, 4]], [[1, 2, 2]]])
def test_vision_matches_native_and_isolates_frames(grids):
    torch.manual_seed(538)
    config = vision_config()
    source = Glm5NextVisionModel(config).eval()
    wrapped = StaticVisionModel(source, 32).eval()
    encoder = Glm53VisionEncoder(wrapped.stem, wrapped.blocks, wrapped.merger, config, 32, dtype=torch.float32)
    grid = torch.tensor(grids)
    pixels = torch.randn(sum(t * h * w for t, h, w in grids), 24)
    with torch.no_grad():
        expected = source(pixels, grid_thw=grid).pooler_output
        actual = encoder(pixels, grid)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=1e-5)
    assert torch.isfinite(actual).all()
    with pytest.raises(ValueError, match="layout|grid_thw"):
        encoder(pixels[:-1], grid)
    for bad in (torch.tensor([[1, 2, 18]]), torch.tensor([[0, 2, 2]]), torch.tensor([[1, 3, 4]]), grid.float()):
        with pytest.raises(ValueError):
            vision_grid_frames(pixels, bad, config, 32)


def multimodal_inputs():
    # One standalone image and two temporal video groups, all using image ID 50.
    ids = torch.tensor([[2, 52, 50, 50, 50, 50, 53, 3, 54, 52, 50, 50, 50, 50, 53, 4, 52, 50, 50, 50, 50, 53, 55, 5]])
    return dict(
        input_ids=ids,
        pixel_values=torch.randn(16, 24),
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        pixel_values_videos=torch.randn(32, 24),
        video_grid_thw=torch.tensor([[2, 4, 4]]),
    )


def save_tiny_processor(directory):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from transformers.models.glm5_next import Glm5NextImageProcessor, Glm5NextProcessor, Glm5NextVideoProcessor

    vocab = {f"t{i}": i for i in range(64)}
    specials = [
        "<|image|>",
        "<|video|>",
        "<|begin_of_image|>",
        "<|end_of_image|>",
        "<|begin_of_video|>",
        "<|end_of_video|>",
    ]
    for i, token in enumerate(specials, 50):
        del vocab[f"t{i}"]
        vocab[token] = i
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="t63")), unk_token="t63", pad_token="t0", eos_token="t1"
    )
    tokenizer.add_special_tokens({"additional_special_tokens": specials})
    template = (
        "{% for m in messages %}{% for c in m['content'] %}"
        "{% if c['type'] == 'image' %}<|begin_of_image|><|image|><|end_of_image|>"
        "{% elif c['type'] == 'video' %}<|begin_of_video|><|video|><|end_of_video|>"
        "{% else %}{{ c['text'] }}{% endif %}{% endfor %}{% endfor %}t2"
    )
    options = dict(patch_size=2, temporal_patch_size=2, merge_size=2, min_image_tokens=4, max_image_tokens=8)
    processor = Glm5NextProcessor(
        Glm5NextImageProcessor(**options), tokenizer, Glm5NextVideoProcessor(**options), chat_template=template
    )
    processor.save_pretrained(directory)


def processor_inputs(processor):
    import numpy as np
    from PIL import Image
    from transformers.video_utils import VideoMetadata

    image = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    content = [
        {"type": "image", "image": Image.fromarray(image)},
        {"type": "video", "video": np.stack([image, 255 - image, image, 255 - image])},
        {"type": "text", "text": "t3"},
    ]
    return processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        processor_kwargs=dict(
            do_sample_frames=False,
            video_metadata=[VideoMetadata(total_num_frames=4, fps=2, frames_indices=[0, 1, 2, 3])],
        ),
    )


def test_video_file_sampling_preserves_repeated_frames(tmp_path):
    import av
    import numpy as np
    from transformers import AutoProcessor

    save_tiny_processor(tmp_path / "processor")
    processor = AutoProcessor.from_pretrained(tmp_path / "processor")
    path = tmp_path / "low_fps.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=1)
        stream.width, stream.height, stream.pix_fmt = 12, 8, "yuv420p"
        for color in (30, 120, 210):
            frame = av.VideoFrame.from_ndarray(np.full((8, 12, 3), color, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    pixels, metadata = load_glm53_video(processor, path)
    expected = processor.video_processor.sample_frames(metadata=metadata)
    assert pixels.shape == (len(expected), 8, 12, 3)
    assert len(expected) > len(set(expected))
    assert list(metadata.frames_indices) == list(expected)
    for i, index in enumerate(expected):
        assert abs(float(pixels[i].mean()) - (30, 120, 210)[index]) < 5
        assert metadata.timestamps[i] == index / metadata.fps


def test_fusion_uses_native_video_spans():
    torch.manual_seed(539)
    config = tiny_multimodal_config()
    model = Glm5NextForConditionalGeneration(config).eval()
    wrapped = StaticVisionModel(model.model.visual, 32)
    encoder = Glm53VisionEncoder(
        wrapped.stem, wrapped.blocks, wrapped.merger, config.vision_config, 32, dtype=torch.float32
    )
    inputs = multimodal_inputs()
    ids = inputs.pop("input_ids")
    with torch.no_grad():
        expected = model.get_input_embeddings()(ids)
        images = torch.cat(model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"]).pooler_output)
        videos = torch.cat(
            model.get_video_features(inputs["pixel_values_videos"], inputs["video_grid_thw"]).pooler_output
        )
        imask, vmask = model.model.get_placeholder_mask(ids, expected, images, videos)
        expected = expected.masked_scatter(imask, images).masked_scatter(vmask, videos)
        actual = fuse_multimodal_embeddings(ids, model.get_input_embeddings(), encoder, config, **inputs)
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=1e-5)
    with pytest.raises(ValueError):
        fuse_multimodal_embeddings(
            ids,
            model.get_input_embeddings(),
            encoder,
            config,
            **{**inputs, "pixel_values": inputs["pixel_values"][:-1]},
        )
    missing_placeholder = ids.clone()
    missing_placeholder[0, 2] = 2
    with pytest.raises(ValueError, match="placeholder"):
        fuse_multimodal_embeddings(missing_placeholder, model.get_input_embeddings(), encoder, config, **inputs)


def test_multimodal_w8_export_move_reload_generate(tmp_path, monkeypatch):
    import yaml

    from xhmodel_merak.workflows import AutoWorkflow
    from xhmodel_merak.xh_llm.hmonnx.auto_llm_hmonnx_model import AutoLLMHONNXModel

    torch.manual_seed(5310)
    config = tiny_multimodal_config()
    source = Glm5NextForConditionalGeneration(config).eval()
    checkpoint = tmp_path / "checkpoint"
    source.save_pretrained(checkpoint)
    save_tiny_processor(checkpoint)
    recipe = yaml.safe_load(
        Path("configs_merak/workflows/xh2a/llm_models/glm5_3_flash/glm5_3_flash_w8a16.yaml").read_text()
    )
    recipe["export"]["model"].update(
        prefill_chunk_length=8,
        linear_chunk_size=8,
        context_max_length=64,
        placeholder_export_workers=1,
        vision_max_patches=32,
    )
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(yaml.safe_dump(recipe))
    workflow = AutoWorkflow.from_config(model_dir=str(checkpoint), config_path=str(recipe_path))
    quant = workflow.quant(output_dir=str(tmp_path / "unused"), device="cuda:0")
    workflow.export(quant, output_dir=str(tmp_path / "export"), device="cuda:0")
    original = next((tmp_path / "export").rglob("golden_meta_info.json"))
    moved = tmp_path / "relocated"
    original.parent.rename(moved)
    meta_path = moved / original.name
    meta = json.loads(meta_path.read_text())
    assert meta["supported_modalities"] == ["text", "image", "video"]
    assert len(meta["glm53_vision"]["components"]) == 4
    assert meta["glm53_vision"]["abi_version"] == 2
    import onnx

    for component in meta["glm53_vision"]["components"][1:-1]:
        graph = onnx.load(moved / meta["glm53_vision"]["directory"] / component["hmonnx"], load_external_data=False)
        attention = [node for node in graph.graph.node if node.op_type == "FlashAttention"]
        assert len(attention) == 1
        assert not any(node.op_type in ("MatMul", "Softmax") for node in graph.graph.node)
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in attention[0].attribute}
        assert attrs["is_causal"] == 0
        assert "s_bits" not in attrs
        for name in ("q_bits", "k_bits", "v_bits", "p_bits"):
            assert attrs[name] == 16
    monkeypatch.setenv("ENABLE_HMINFERENCE_V2", "1")
    model = AutoLLMHONNXModel.from_pretrained(str(meta_path), device_map=["cuda:0"])
    inputs = multimodal_inputs()
    source = source.cuda().half()
    with torch.no_grad():
        for pixel_name, grid_name, native in (
            ("pixel_values", "image_grid_thw", source.get_image_features),
            ("pixel_values_videos", "video_grid_thw", source.get_video_features),
        ):
            actual = model.vision_encoder(inputs[pixel_name], inputs[grid_name])
            expected = torch.cat(native(inputs[pixel_name].cuda().half(), inputs[grid_name].cuda()).pooler_output)
            cosine = torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
            assert torch.isfinite(actual).all() and cosine > 0.99
        ids = inputs.pop("input_ids").cuda()
        fused = fuse_multimodal_embeddings(ids, model.embed_tokens, model.vision_encoder, config, **inputs)
        for start in range(0, ids.shape[1], 8):
            actual = model.forward(
                *model.get_data_preprocessor()({"inputs_embeds": fused[:, start : start + 8], "past_seq_length": start})
            )
        expected = source(input_ids=ids, **{k: v.cuda() for k, v in inputs.items()}).logits[:, -1:]
        cosine = torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
        assert cosine > 0.98 and torch.isfinite(actual).all()
    kwargs = dict(input_ids=ids, **inputs, max_new_tokens=3, do_sample=False, eos_token_id=None, pad_token_id=0)
    generated = model.generate(**kwargs)
    assert generated.shape == (1, ids.shape[1] + 3)
    torch.testing.assert_close(model.generate(**kwargs), generated)
    processed = processor_inputs(model.get_processor())
    assert processed["input_ids"].shape[1] < 64
    assert processed["video_grid_thw"][0, 0] == 2
    assert 2 in processed["mm_token_type_ids"]
    assert model.generate(**processed, max_new_tokens=2, eos_token_id=None, do_sample=False).shape[1] == (
        processed["input_ids"].shape[1] + 2
    )
    image_only = dict(
        input_ids=ids[:, :8], pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"]
    )
    assert model.generate(**image_only, max_new_tokens=2, do_sample=False, eos_token_id=None).shape == (1, 10)
    with pytest.raises(ValueError):
        model.generate(**{**kwargs, "image_grid_thw": torch.tensor([[1, 2, 4]])})
    torch.testing.assert_close(model.generate(**kwargs), generated)
