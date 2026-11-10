# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Real CPU sessions for visual prefill, M-RoPE and speculative scheduling."""

from types import MethodType, SimpleNamespace

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from xhmodel_merak.xh_llm.models.qwen3_5.data_preprocess import Qwen3_5_DataPreprocess
from xhmodel_merak.xh_llm.models.qwen3_5.spec_decode_v2 import Qwen3_5SpecDecodeV2Model
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_onnx_model import _alloc_cache_inputs
from xhmodel_merak.xh_llm.types import CacheList


CONFIG = dict(
    image_token_id=10, video_token_id=11, vision_start_token_id=12, vision_end_token_id=13, spatial_merge_size=2
)


def _graph(path, length, role):
    def info(name, shape, dtype=TensorProto.FLOAT):
        return helper.make_tensor_value_info(name, dtype, shape)

    nodes, weights = [], []
    inputs = [info("past_seq_length", [1], TensorProto.INT32), info("current_input_length", [1], TensorProto.INT32)]
    if role == "target":
        inputs += [info("past_key_cache_0", [1, 1, 4, 64]), info("past_value_cache_0", [1, 1, 64, 4])]
        source = "inputs_embeds"
        inputs += [info(source, [1, length, 4]), info("linear_attn_mask", [1, length])]
        for name in ("time_position_ids", "hight_position_ids", "width_position_ids"):
            inputs.append(info(name, [length], TensorProto.INT64))
        # A numerical target sensitive to all three axes and fused features.
        weights.append(numpy_helper.from_array(np.array([1, length, 1], np.int64), "shape"))
        hidden = source
        for axis in ("time_position_ids", "hight_position_ids", "width_position_ids"):
            nodes += [
                helper.make_node("Cast", [axis], [axis + "_float"], to=TensorProto.FLOAT),
                helper.make_node("Reshape", [axis + "_float", "shape"], [axis + "_3d"]),
                helper.make_node("Add", [hidden, axis + "_3d"], [axis + "_hidden"]),
            ]
            hidden = axis + "_hidden"
    elif role == "mtp":
        inputs += [info("post_norm_hidden", [1, length, 4]), info("next_token_embedding", [1, length, 4])]
        nodes += [helper.make_node("Add", ["post_norm_hidden", "next_token_embedding"], ["merged"])]
        hidden = "merged"
    elif role == "context":
        inputs += [info("target_hidden", [1, length, 4])]
        hidden = "target_hidden"
    else:
        inputs += [info("noise_embedding", [1, length, 4])]
        hidden = "noise_embedding"
    nodes += [
        helper.make_node("Identity", [hidden], ["post_norm_out"]),
        helper.make_node("Concat", [hidden] * 4, ["wide_hidden"], axis=2),
        helper.make_node("Mul", ["wide_hidden", "head"], ["raw_logits"]),
        helper.make_node("Add", ["raw_logits", "bias"], ["logits"]),
    ]
    # Deterministic target/draft agreement exercises full accept and MTP tail.
    weights += [
        numpy_helper.from_array(np.zeros((1, 1, 16), np.float32), "head"),
        numpy_helper.from_array(np.eye(16, dtype=np.float32)[7] * 10, "bias"),
    ]
    for index, node in enumerate(nodes):
        node.name = f"{role}_{index}"
    graph = helper.make_graph(
        nodes, role, inputs, [info("logits", [1, length, 16]), info("post_norm_out", [1, length, 4])], weights
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return {"onnx": str(path)}


def make_runtime(tmp_path, mode):
    target = {stage: _graph(tmp_path / f"{stage}.onnx", 3, "target") for stage in ("prefill", "decode")}
    draft = {
        "prefill" if mode == "mtp" else "context": _graph(
            tmp_path / "draft_prefill.onnx", 3, "mtp" if mode == "mtp" else "context"
        ),
        "decode": _graph(
            tmp_path / "draft_decode.onnx", 1 if mode == "mtp" else 3, "mtp" if mode == "mtp" else "dflash"
        ),
    }
    model = Qwen3_5SpecDecodeV2Model(
        **target,
        draft=draft,
        session_device="cpu",
        auto_offload=False,
        spec_decode_mode=mode,
        hidden_output_name="post_norm_out",
        block_size=2,
        dflash_noise_token_id=0 if mode == "dflash" else None,
        max_context_tokens=64,
    )
    model.to(torch.float32)
    model.set_input_embeddings(torch.nn.Embedding.from_pretrained(torch.arange(64).reshape(16, 4).float()))
    model.configure_multimodal({"model_config": CONFIG}, tmp_path, tmp_path)
    return model


@pytest.fixture(params=["mtp", "dflash"])
def runtime(tmp_path, request):
    return make_runtime(tmp_path, request.param)


def _prompt(kind):
    image = [12, 10, 10, 10, 10, 13]
    video = [12, 11, 11, 11, 11, 13]
    spans = {"image": image, "video": video + [3] + video, "mixed": video + [3] + video + [2] + image}
    ids = torch.tensor([[1] + spans[kind] + [2]])
    data = {}
    if kind in {"image", "mixed"}:
        data.update(image_grid_thw=torch.tensor([[1, 4, 4]]), image_embeds=torch.arange(16).reshape(4, 4).float() + 100)
    if kind in {"video", "mixed"}:
        data.update(video_grid_thw=torch.tensor([[2, 4, 4]]), video_embeds=torch.arange(32).reshape(8, 4).float() + 200)
    return ids, data


@pytest.mark.parametrize("kind", ["image", "video", "mixed"])
def test_target_prefill_matches_regular_fusion_and_native_mrope(runtime, kind):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    ids, data = _prompt(kind)
    processor = Qwen3_5_DataPreprocess(
        token_embedding=runtime.token_embedding,
        input_sequence_length=ids.shape[1],
        past_key_caches=CacheList(),
        past_value_caches=CacheList(),
        past_conv_caches=CacheList(),
        past_recurrent_states=CacheList(),
        **CONFIG,
    )
    regular = processor(dict(input_ids=ids, past_seq_length=0, **data))
    prepared = runtime._prepare_visual_prompt(ids, **data)
    torch.testing.assert_close(prepared.embeddings, regular[0])
    native = SimpleNamespace(
        config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2)), get_vision_position_ids=None
    )
    native.get_vision_position_ids = MethodType(Qwen3_5Model.get_vision_position_ids, native)
    types = (ids == 10).long() + (ids == 11).long() * 2
    positions, delta = Qwen3_5Model.get_rope_index(
        native, ids, types, image_grid_thw=data.get("image_grid_thw"), video_grid_thw=data.get("video_grid_thw")
    )
    torch.testing.assert_close(prepared.positions, positions[:, 0])
    torch.testing.assert_close(prepared.rope_delta, delta)
    assert int(delta) < 0
    runtime._visual_prompt = prepared
    caches = _alloc_cache_inputs(runtime.prefill_session, runtime.device)
    for start in range(0, ids.shape[1], 3):
        valid = min(3, ids.shape[1] - start)
        feed = runtime._build_prefill_feed(ids[:, start : start + valid], valid, start, caches)
        for name, value in caches.items():
            assert feed[name] is value
        _, actual = runtime._run_hmonnx(runtime.prefill_session, feed)
        expected = regular[0][:, start : start + valid] + positions[:, :, start : start + valid].sum(0).unsqueeze(-1)
        torch.testing.assert_close(actual["post_norm_out"][:, :valid], expected)
        assert feed["past_seq_length"].item() == start
        assert feed["current_input_length"].item() == valid
        if valid < 3:
            torch.testing.assert_close(
                feed["inputs_embeds"][:, valid:], torch.zeros_like(feed["inputs_embeds"][:, valid:])
            )
    verify = runtime._build_decode_feed(torch.tensor([[7, 7, 7]]), ids.shape[1], caches, 3)
    expected = torch.arange(3) + ids.shape[1] + delta.item()
    torch.testing.assert_close(verify["time_position_ids"], expected)
    assert verify["past_seq_length"].item() == ids.shape[1]
    if runtime.spec_decode_mode == "mtp":
        for start in (0, 3):
            tokens = ids[:, start + 1 : start + 4]
            feed = runtime._build_mtp_prefill_feed(torch.ones(1, 3, 4), tokens, start, {})
            torch.testing.assert_close(feed["next_token_embedding"], prepared.embeddings[:, start + 1 : start + 4])


@pytest.mark.parametrize("kind", ["image", "video", "mixed"])
def test_visual_generation_then_text_resets_request_state(runtime, kind):
    tokenizer = SimpleNamespace(eos_token_id=15, decode=lambda tokens, **kw: " ".join(map(str, tokens)))
    ids, data = _prompt(kind)
    visual, stats = runtime.generate(ids, tokenizer, max_new_tokens=7, return_stats=True, **data)
    assert visual == "7 7 7 7 7 7 7"
    assert stats["draft_tokens_total"] > 0
    assert runtime._visual_prompt is None
    text, _ = runtime.generate(torch.tensor([[1, 2]]), tokenizer, max_new_tokens=7, return_stats=True)
    assert text == visual and runtime._visual_prompt is None


def test_malformed_visual_request_cannot_truncate_or_leak(runtime):
    ids, data = _prompt("video")
    tokenizer = SimpleNamespace(eos_token_id=15)
    with pytest.raises(ValueError, match="do not match"):
        runtime.generate(ids, tokenizer, **{**data, "video_embeds": data["video_embeds"][:-1]})
    assert runtime._visual_prompt is None
    runtime.max_context_tokens = 4
    with pytest.raises(ValueError, match="truncation"):
        runtime.generate(ids, tokenizer, **data)
    assert runtime._visual_prompt is None


def test_precomputed_full_embeddings_reuse_positions_and_generation(runtime):
    ids, features = _prompt("mixed")
    prepared = runtime._prepare_visual_prompt(ids, **features)
    grids = {name: value for name, value in features.items() if name.endswith("grid_thw")}
    fused = runtime._prepare_visual_prompt(ids, inputs_embeds=prepared.embeddings, **grids)
    torch.testing.assert_close(fused.embeddings, prepared.embeddings)
    torch.testing.assert_close(fused.positions, prepared.positions)
    tokenizer = SimpleNamespace(eos_token_id=15, decode=lambda tokens, **kw: " ".join(map(str, tokens)))
    assert runtime.generate(ids, tokenizer, inputs_embeds=fused.embeddings, max_new_tokens=4, **grids) == "7 7 7 7"
    assert runtime._visual_prompt is None
    with pytest.raises(ValueError, match="hidden_size"):
        runtime.generate(ids, tokenizer, **{**features, "image_embeds": features["image_embeds"][:, :2]})
    assert runtime._visual_prompt is None
