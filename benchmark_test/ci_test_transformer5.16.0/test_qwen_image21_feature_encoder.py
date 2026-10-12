# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Match native Qwen3-VL features through Qwen-Image's CPU loader and capacity graphs."""

import copy

import pytest
import torch

from xhmodel_merak.xh_other_model.models.qwen_image21.feature_graph import (
    CapacityFeatureSession,
    CapacityTextFeatures,
    FeatureSession,
    load_feature_encoder,
)


def tiny_config():
    from transformers import Qwen3VLConfig

    return Qwen3VLConfig(
        text_config=dict(
            vocab_size=64,
            eos_token_id=2,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=128,
            max_position_embeddings=256,
            rope_parameters={"rope_type": "default", "rope_theta": 1000000.0, "mrope_section": [24, 20, 20]},
        ),
        vision_config=dict(
            depth=3,
            hidden_size=64,
            intermediate_size=128,
            num_heads=4,
            out_hidden_size=256,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            deepstack_visual_indexes=[0, 1, 2],
        ),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )


@pytest.mark.parametrize("capacity,length", [(12, 8), (1664, 8), (1792, 8), (256, 256), (512, 257), (512, 512)])
def test_capacity_flash_features_preserve_native_mrope_deepstack_and_hide_padding(capacity, length):
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration

    from xhmodel_merak.xh_other_model.models.qwen_image21.feature_graph import CapacityTextFeatures

    torch.manual_seed(19)
    source = Qwen3VLForConditionalGeneration(tiny_config()).eval().model.language_model
    source.config._attn_implementation = "sdpa"
    graph = CapacityTextFeatures(source, capacity, max(256, capacity))
    embeddings = torch.randn(1, length, 256) * 0.2
    positions = torch.arange(length, dtype=torch.int32).expand(3, -1).clone()
    if length == 8:
        positions[1] = torch.tensor([0, 1, 1, 2, 2, 3, 3, 4])
        positions[2] = torch.tensor([0, 1, 2, 1, 2, 1, 2, 3])
    deep = [torch.randn_like(embeddings) * 0.03 for _ in range(3)]
    captured = []
    hook = source.norm.register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
    with torch.inference_mode():
        source(
            inputs_embeds=embeddings,
            position_ids=positions[:, None].long(),
            use_cache=False,
            visual_pos_masks=torch.ones(1, length, dtype=torch.bool),
            deepstack_visual_embeds=[x[0] for x in deep],
        )
        hook.remove()
        padding = capacity - embeddings.shape[1]
        padded = torch.nn.functional.pad(embeddings, (0, 0, 0, padding))
        p = torch.nn.functional.pad(positions, (0, padding))
        d = [torch.nn.functional.pad(x, (0, 0, 0, padding)) for x in deep]
        current_length = torch.tensor([length], dtype=torch.int32)
        actual = graph(padded, *p.unbind(0), current_length, *d)
        assert actual.shape == (1, capacity, 256)
        torch.testing.assert_close(actual[:, :length], captured[0], atol=2e-5, rtol=3e-5)
        assert torch.count_nonzero(actual[:, length:]) == 0
        padded[:, length:] = 1000
        for x in d:
            x[:, length:] = -1000
        torch.testing.assert_close(graph(padded, *p.unbind(0), current_length, *d), actual, atol=0, rtol=0)


@pytest.mark.parametrize("length", [0, 255, 256, 257])
def test_capacity_feature_session_length_boundaries(length):
    session = CapacityFeatureSession(None, 256, 256, 512, device="cpu")
    embeddings = torch.zeros(1, length, 256)
    positions = torch.arange(length, dtype=torch.int32).expand(3, -1)
    if length == 0 or length > session.capacity:
        with pytest.raises(ValueError, match="sequence length"):
            session.prepare_inputs(embeddings, positions)
    else:
        inputs = session.prepare_inputs(embeddings, positions)
        assert inputs[0].shape == (1, session.capacity, 256)
        assert inputs[4].dtype == torch.int32 and inputs[4].tolist() == [length]


@pytest.mark.parametrize("ling_compat", [False, True])
def test_loaded_native_encoder_matches_capacity_features(tmp_path, ling_compat):
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration as NativeModel

    from xhmodel_merak.xh_llm.models.ling_3_flash._compat import patch_ling_remote_code_compatibility
    from xhquant.frontend.utils import module_graph_forward_guard

    torch.manual_seed(31)
    cfg = tiny_config()
    native = NativeModel(copy.deepcopy(cfg)).eval()
    native.save_pretrained(tmp_path)
    previous_initializers = ROPE_INIT_FUNCTIONS.copy()
    try:
        ROPE_INIT_FUNCTIONS.pop("default", None)
        if ling_compat:
            patch_ling_remote_code_compatibility()
        registered_initializers = ROPE_INIT_FUNCTIONS.copy()
        # Use the exact CPU loader used by the public six-graph builder. Its
        # text/vision buffers must survive reload independently of import order.
        loaded = load_feature_encoder(tmp_path)
        assert ROPE_INIT_FUNCTIONS == registered_initializers
    finally:
        ROPE_INIT_FUNCTIONS.clear()
        ROPE_INIT_FUNCTIONS.update(previous_initializers)
    assert all(parameter.device.type == "cpu" for parameter in loaded.parameters())
    torch.testing.assert_close(
        loaded.model.language_model.rotary_emb.inv_freq,
        native.model.language_model.rotary_emb.inv_freq,
        atol=0,
        rtol=0,
    )
    # Nonpersistent visual RoPE frequencies must be rebuilt after meta loading.
    torch.testing.assert_close(
        loaded.model.visual.rotary_pos_emb.inv_freq,
        native.model.visual.rotary_pos_emb.inv_freq,
        atol=0,
        rtol=0,
    )
    inputs = torch.randn(1, 8, 256) * 0.2
    positions = torch.arange(8, dtype=torch.int64)
    heights = torch.tensor([0, 1, 1, 1, 2, 2, 3, 4], dtype=torch.int64)
    widths = torch.tensor([0, 1, 2, 3, 1, 2, 3, 4], dtype=torch.int64)
    deep = [torch.randn(1, 8, 256) * 0.03 for _ in range(3)]
    captured = []
    hook = native.model.language_model.norm.register_forward_pre_hook(lambda m, args: captured.append(args[0].clone()))
    with torch.no_grad():
        normalized = native.model.language_model(
            inputs_embeds=inputs,
            position_ids=torch.stack((positions, heights, widths))[:, None],
            use_cache=False,
            visual_pos_masks=torch.ones(1, 8, dtype=torch.bool),
            deepstack_visual_embeds=[x[0] for x in deep],
        ).last_hidden_state
    hook.remove()
    graph = CapacityTextFeatures(loaded.model.language_model, 12, 256).eval()
    padded = torch.nn.functional.pad(inputs, (0, 0, 0, 4))
    padded_positions = torch.nn.functional.pad(torch.stack((positions, heights, widths)), (0, 4)).int()
    padded_deep = [torch.nn.functional.pad(x, (0, 0, 0, 4)) for x in deep]
    with torch.no_grad(), module_graph_forward_guard(graph):
        actual = graph(padded, *padded_positions.unbind(0), torch.tensor([8], dtype=torch.int32), *padded_deep)
    torch.testing.assert_close(actual[:, :8], captured[0], atol=1e-4, rtol=1e-4)
    assert torch.count_nonzero(actual[:, 8:]) == 0
    # Qwen-Image consumes the residual before final RMSNorm, not pooled or
    # normalized embedding features. Keep this distinction observable.
    assert (captured[0] - normalized).abs().max() > 0.1
    graph.half()
    with torch.no_grad(), module_graph_forward_guard(graph):
        features = CapacityFeatureSession(graph, 12, 256, 256, device="cpu")(
            inputs, torch.stack((positions, heights, widths)), deep
        )
    torch.testing.assert_close(features.float(), captured[0], atol=2e-3, rtol=2e-3)
    assert features.shape == (1, 8, 256)


def test_feature_session_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="capacity exceeds"):
        FeatureSession(None, 64, 16, 32, device="cpu")
    session = FeatureSession(None, 8, 16, 32, device="cpu")
    with pytest.raises(ValueError, match="sequence length"):
        session(torch.zeros(1, 9, 16), torch.zeros(3, 9, dtype=torch.int32))
    with pytest.raises(ValueError, match="position exceeds"):
        session(torch.zeros(1, 4, 16), torch.full((3, 4), 32, dtype=torch.int64))
    with pytest.raises(ValueError, match="three DeepStack"):
        session(torch.zeros(1, 4, 16), torch.zeros(3, 4, dtype=torch.int64), [])


@pytest.mark.parametrize("image_count", [0, 1, 2])
@pytest.mark.parametrize("size", [(32, 32), (32, 64)])
def test_prompt_encoder_matches_native_multimodal_features(tmp_path, image_count, size):
    from PIL import Image
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast, Qwen2VLImageProcessorFast, Qwen3VLProcessor, Qwen3VLVideoProcessor
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration as NativeModel

    from xhmodel_merak.xh_other_model.models.qwen_image21.processor import PromptEncoder
    from xhmodel_merak.xh_other_model.models.qwen_image21.vision_graph import PackedVision, PackedVisionSession
    from xhquant.frontend.utils import module_graph_forward_guard

    torch.manual_seed(39)
    height, width = size
    vocab = {f"token{i}": i for i in range(64)}
    specials = {
        "<unk>": 0,
        "<pad>": 1,
        "<|im_end|>": 2,
        "<|im_start|>": 3,
        "<|image_pad|>": 60,
        "<|video_pad|>": 61,
        "<|vision_start|>": 62,
        "<|vision_end|>": 63,
    }
    for token, index in specials.items():
        del vocab[f"token{index}"]
        vocab[token] = index
    backend = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        pad_token="<pad>",
        additional_special_tokens=list(specials),
    )
    tokenizer.chat_template = "{{ '<|im_start|>system\\n' + messages[0]['content'][0]['text'] + '<|im_end|>\\n' }}"
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessorFast(
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
        ),
        tokenizer=tokenizer,
        video_processor=Qwen3VLVideoProcessor(),
        chat_template=tokenizer.chat_template,
    )
    cfg = tiny_config()
    native = NativeModel(copy.deepcopy(cfg)).eval()
    native.save_pretrained(tmp_path)
    loaded = load_feature_encoder(tmp_path)
    embedding = copy.deepcopy(loaded.model.language_model.embed_tokens)
    visual = PackedVision(loaded.model.visual, 8).eval().half()
    graph = CapacityTextFeatures(loaded.model.language_model, 128, 256).eval().half()
    vision_session = PackedVisionSession(visual, 8, loaded.model.visual.num_grid_per_side, 16, device="cpu")
    encoder = PromptEncoder(
        processor,
        embedding,
        CapacityFeatureSession(graph, 128, 256, 256, "cpu"),
        vision_session,
        prompt_budget=32,
        max_references=2,
    )
    images = [Image.new("RGBA", (width, height), (30 + i * 80, 70, 210, 128)) for i in range(image_count)]
    captured = []
    hook = native.model.language_model.norm.register_forward_pre_hook(lambda m, a: captured.append(a[0].clone()))
    prefix = " ".join(f"<image{i + 1}><|vision_start|><|image_pad|><|vision_end|>" for i in range(image_count))
    text = (
        f"<|im_start|>system\n{encoder.SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{prefix}token8<|im_end|>\n<|im_start|>assistant\n"
    )
    kwargs = dict(text=[text], padding=True, padding_side="left", return_tensors="pt")
    if images:
        kwargs.update(images=[encoder.vision_image(x) for x in images], do_resize=False)
    batch = processor(**kwargs)
    with torch.no_grad():
        native.model(**batch, use_cache=False)
    hook.remove()
    with torch.no_grad(), module_graph_forward_guard(graph), module_graph_forward_guard(visual):
        actual, mask = encoder("token8", images)
    torch.testing.assert_close(actual.float(), captured[0][:, encoder.drop_tokens :], atol=2e-3, rtol=2e-2)
    assert int(mask.sum()) == height * width // 1024 * image_count
    assert images == [] or images[0].mode == "RGBA"
    with pytest.raises(ValueError, match="patch capacity"):
        encoder.prepare("token8", [Image.new("RGB", (96, 32))])
