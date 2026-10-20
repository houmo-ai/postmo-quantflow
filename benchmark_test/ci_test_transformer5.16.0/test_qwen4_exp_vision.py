"""Small real Qwen4 visual rewrites, native positions and HMONNX export."""

from __future__ import annotations

import copy
import json

import onnx
import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen4ExpConfig, Qwen4ExpVisionConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

from xhmodel_merak.xh_llm.models.qwen3_5.visual_token_gears import (
    build_visual_token_gear_inputs,
    pad_flattened_patches,
)
from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import XHQwen3_5_VisualConfig
from xhmodel_merak.xh_llm.models.qwen4_exp.multimodal import (
    image_rope_positions,
    merge_image_embeddings,
    prepare_image_prompt,
)
from xhmodel_merak.xh_llm.models.qwen4_exp.vision import XHQwen4ExpVisionModel, export_qwen4_exp_visual


def _tiny_checkpoint(path, *, head_dim=8):
    torch.manual_seed(37)
    config = Qwen4ExpVisionConfig(
        depth=1,
        hidden_size=2 * head_dim,
        num_heads=2,
        intermediate_size=32,
        out_hidden_size=16,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        num_position_embeddings=16,
    )
    config._attn_implementation = "eager"
    native = Qwen4ExpVisionModel(config).eval()
    path.mkdir()
    Qwen4ExpConfig(vision_config=config.to_dict()).save_pretrained(path)
    save_file(
        {"model.visual." + name: value for name, value in native.state_dict().items()}, path / "model.safetensors"
    )
    return native


@pytest.mark.parametrize("head_dim", [8, 72])
@pytest.mark.parametrize("height,width,capacity", [(2, 4, 4), (4, 4, 4), (2, 8, 8)])
def test_visual_wrapper_preserves_native_patch_bias_positions_and_padded_attention(
    tmp_path, head_dim, height, width, capacity
):
    native = _tiny_checkpoint(tmp_path / "source", head_dim=head_dim)
    model = XHQwen4ExpVisionModel(
        XHQwen3_5_VisualConfig(
            model_name="tiny_visual",
            hf_model=str(tmp_path / "source"),
            dtype="float32",
            visual_input_mode="patches",
            image_token_gears=[capacity],
            image_token_capacity=capacity,
            spatial_merge_size=2,
            patch_size=2,
            temporal_patch_size=2,
        )
    )
    # Rewriting the native Qwen4 classes exercises the actual export wrappers,
    # including 72->128 head padding used by Flash Next's real visual tower.
    wrapped = model.init_wrap_model(copy.deepcopy(native))
    pixels = torch.randn(height * width, 24)
    grid = torch.tensor([[1, height, width]])
    geometry = build_visual_token_gear_inputs(
        grid,
        patch_capacity=capacity * 4,
        num_position_embeddings=16,
        spatial_merge_size=2,
        dtype=torch.float32,
    )
    padded, valid = pad_flattened_patches(pixels, capacity * 4)
    with torch.inference_mode():
        reference = native(pixels, grid).pooler_output
        actual = wrapped(
            padded,
            geometry["position_ids"],
            geometry["position_weights"],
            geometry["rotary_position_ids"],
            geometry["attention_mask"],
        )
    assert valid == height * width
    torch.testing.assert_close(actual[0, : reference.shape[0]], reference, atol=2e-6, rtol=2e-5)


def test_multimodal_positions_keep_image_height_width_and_continuation_offset():
    config = Qwen4ExpConfig(image_token_id=90, video_token_id=91, vision_config={"spatial_merge_size": 2})
    # Text prefix, a 2x3 merged grid, text suffix. Temporal/height/width
    # coordinates differ; flattening them to sequential IDs changes RoPE.
    ids = torch.tensor([[3, 4, 90, 90, 90, 90, 90, 90, 5]])
    positions, delta = image_rope_positions(config, ids, torch.tensor([[1, 4, 6]]))
    assert positions.tolist() == [
        [[0, 1, 2, 2, 2, 2, 2, 2, 5]],
        [[0, 1, 2, 2, 2, 3, 3, 3, 5]],
        [[0, 1, 2, 3, 4, 2, 3, 4, 5]],
    ]
    assert delta.tolist() == [[-3]]
    with pytest.raises(ValueError, match="placeholder"):
        image_rope_positions(config, ids, torch.tensor([[1, 2, 2]]))


def test_image_embedding_insertion_preserves_text_and_rejects_mismatched_features():
    embedding = torch.nn.Embedding(10, 4)
    ids = torch.tensor([[1, 8, 8, 2]])
    visual = torch.arange(8).float().reshape(1, 2, 4)
    actual = merge_image_embeddings(ids, embedding, [visual], 8)
    torch.testing.assert_close(actual[:, 1:3], visual)
    torch.testing.assert_close(actual[:, [0, 3]], embedding(ids[:, [0, 3]]))
    assert ids.tolist() == [[1, 8, 8, 2]]
    with pytest.raises(ValueError, match="do not match"):
        merge_image_embeddings(ids, embedding, [visual[:, :1]], 8)


def test_native_image_processor_preserves_normalization_and_respects_exported_capacity(tmp_path):
    from PIL import Image
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessor,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    vocab = {
        "<unk>": 0,
        "<|image_pad|>": 1,
        "<|video_pad|>": 2,
        "<|vision_start|>": 3,
        "<|vision_end|>": 4,
        "describe": 5,
    }
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(vocab, unk_token="<unk>")),
        unk_token="<unk>",
        additional_special_tokens=list(vocab)[1:5],
    )
    processor = Qwen3VLProcessor(
        image_processor=Qwen2VLImageProcessor(
            patch_size=2,
            temporal_patch_size=2,
            merge_size=2,
            min_pixels=16,
            max_pixels=256,
            image_mean=[0.5] * 3,
            image_std=[0.5] * 3,
        ),
        video_processor=Qwen3VLVideoProcessor(patch_size=2, temporal_patch_size=2, merge_size=2),
        tokenizer=tokenizer,
        chat_template="<|vision_start|><|image_pad|><|vision_end|>describe",
    )
    processor.save_pretrained(tmp_path)
    Qwen4ExpConfig(
        image_token_id=1,
        video_token_id=2,
        vision_config={"patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2},
    ).save_pretrained(tmp_path)
    inputs = prepare_image_prompt(tmp_path, "describe", Image.new("RGB", (8, 16), "white"), max_image_tokens=4)
    assert inputs["pixel_values"].shape[1] == 24
    assert inputs["pixel_values"].shape[0] <= 16
    torch.testing.assert_close(inputs["pixel_values"], torch.ones_like(inputs["pixel_values"]))
    assert inputs["position_ids"].shape == (3, 1, inputs["input_ids"].shape[1])
    assert (inputs["input_ids"] == 1).sum() == inputs["image_grid_thw"].prod() // 4


def test_real_visual_hmonnx_export_has_five_input_contract_and_flat_gears(tmp_path):
    _tiny_checkpoint(tmp_path / "source")
    result = export_qwen4_exp_visual(
        str(tmp_path / "source"),
        tmp_path / "export",
        image_token_gears=[2, 4],
        quant_scheme={"quant_type": "w16a16h1_sefp"},
        dtype="float32",
    )
    assert result["image_token_gears"] == [2, 4]
    assert result["patch_size"] == 2
    assert result["num_heads"] == 2
    assert result["hidden_size"] == 16
    assert result["input_names"] == [
        "pixel_values",
        "position_ids",
        "position_weights",
        "rotary_position_ids",
        "attention_mask",
    ]
    manifest = json.loads((tmp_path / "export" / "visual_gears.json").read_text())
    assert manifest["routing_policy"] == "smallest_fit"
    for gear in result["gears"]:
        graph_path = tmp_path / "export" / gear["hmonnx"]
        assert graph_path.parent.name == f"visual_m{gear['image_token_capacity']}"
        graph = onnx.load(graph_path)
        assert [value.name for value in graph.graph.input] == result["input_names"]
        assert [value.name for value in graph.graph.output] == ["image_embeds"]
        # MaskedAdd lowers to its two explicit additive-bias operations.
        assert sum("attention_mask" in node.input for node in graph.graph.node) == 2
        assert not any(node.op_type in {"Conv", "Conv3D"} for node in graph.graph.node)
