"""Real small checkpoint contracts for Qwen4's bounded module loader."""

from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from xhmodel_merak.xh_llm.models.qwen4_exp.streaming_loader import Qwen4ExpStreamingLoader


@pytest.fixture(params=[1, 2], ids=["one_ple_shard", "two_ple_shards"])
def native_checkpoint(tmp_path, request):
    from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

    config = Qwen4ExpTextConfig(
        max_position_embeddings=16,
        vocab_size=16, hidden_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_key_head_dim=4, linear_value_head_dim=4,
        linear_num_key_heads=2, linear_num_value_heads=2,
        moe_intermediate_size=8, shared_expert_intermediate_size=8,
        num_experts=2, num_experts_per_tok=1, hc_count=2, hc_lowrank=4,
        layer_types=["linear_attention", "full_attention"], ple_layer_ids=[1],
        ple_embed_dim=16, ngram_size=2, heads_per_ngram=2,
        ngram_vocab_size_base=16, make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=request.param, eos_token_id=0,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8,
        indexer_budget=4, indexer_compress_ratio=2,
        rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.5},
    )
    torch.manual_seed(11)
    model = Qwen4ExpForCausalLM(config).eval()
    source = tmp_path / "native"
    model.save_pretrained(source)
    return source, model


def _loader(native_checkpoint):
    return Qwen4ExpStreamingLoader(native_checkpoint[0], dtype=torch.float32)


def test_only_requested_native_subtree_is_materialized_and_released(native_checkpoint):
    loader = _loader(native_checkpoint)
    reference = native_checkpoint[1]
    assert all(value.is_meta for value in loader.model.parameters())
    module = loader.load_module("model.layers.0")
    assert all(not value.is_meta for value in module.parameters())
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, reference.model.layers[0].state_dict()[name], rtol=0, atol=0)
    assert all(value.is_meta for value in loader.model.parameters())
    loader.release(module)
    assert all(value.is_meta for value in module.parameters())
    assert all(value.is_meta for value in module.buffers())


@pytest.mark.parametrize("prefix", ["model.embed_tokens", "model.hyper_connection_mixer", "lm_head"])
def test_main_graph_weights_can_be_loaded_individually(native_checkpoint, prefix):
    loader = _loader(native_checkpoint)
    with loader.materialized(prefix) as module:
        reference = native_checkpoint[1].get_submodule(prefix)
        for name, value in module.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[name], rtol=0, atol=0)
        if prefix == "lm_head":
            inputs = torch.randn(2, 3, 16)
            torch.testing.assert_close(module(inputs), reference(inputs), rtol=0, atol=0)
        assert all(value.is_meta for value in loader.model.parameters())
    assert all(value.is_meta for value in module.parameters())


def test_materialized_scope_releases_on_exception(native_checkpoint):
    loader = _loader(native_checkpoint)
    with pytest.raises(RuntimeError, match="consumer failed"):
        with loader.materialized("lm_head", clone=False) as module:
            assert not loader.model.lm_head.weight.is_meta
            raise RuntimeError("consumer failed")
    assert module is loader.model.lm_head
    assert module.weight.is_meta


def test_nonpersistent_rope_buffers_are_recreated_after_release(native_checkpoint):
    loader = _loader(native_checkpoint)
    reference = native_checkpoint[1].model.rotary_emb
    for _ in range(2):
        with loader.materialized("model.rotary_emb", clone=False) as rope:
            torch.testing.assert_close(rope.inv_freq, reference.inv_freq, rtol=0, atol=0)
            torch.testing.assert_close(rope.original_inv_freq, reference.original_inv_freq, rtol=0, atol=0)
        assert rope.inv_freq.is_meta


def test_ple_ranges_are_bounded_and_preserve_saved_hash_buffers(native_checkpoint):
    loader = _loader(native_checkpoint)
    key = "model.layers.0.ple.ple_embedding.ngram_embedding.weight"
    expected = native_checkpoint[1].model.layers[0].ple.ple_embedding.ngram_embedding.weight
    assert loader.tensor_info(key) == {"shape": tuple(expected.shape), "dtype": "F32"}
    torch.testing.assert_close(loader.read_tensor(key, rows=slice(2, 5)), expected[2:5], rtol=0, atol=0)
    boundary = expected.shape[0] // native_checkpoint[1].config.split_ngram_parts
    start, stop = max(0, boundary - 1), min(expected.shape[0], boundary + 2)
    torch.testing.assert_close(loader.read_tensor(key, rows=slice(start, stop)), expected[start:stop], rtol=0, atol=0)
    chunks = list(loader.iter_tensor_rows(key, chunk_rows=7))
    assert all(chunk.shape[0] <= 7 for chunk in chunks)
    torch.testing.assert_close(torch.cat(chunks), expected, rtol=0, atol=0)
    with loader.materialized("model.layers.0.ple.ple_embedding") as embedding:
        original = native_checkpoint[1].model.layers[0].ple.ple_embedding
        for name in ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets"):
            actual = getattr(embedding, name)
            torch.testing.assert_close(actual, getattr(original, name), rtol=0, atol=0)
            assert actual.dtype == torch.int64


def test_explicit_ple_exclusion_never_reads_the_full_table(native_checkpoint, monkeypatch):
    from xhmodel_merak.xh_llm import big_hf_model_helper

    loader = _loader(native_checkpoint)
    key = "model.layers.0.ple.ple_embedding.ngram_embedding.weight"
    original_load = big_hf_model_helper.load_weight_from_safetensor
    original_read = loader.read_tensor
    seen = []

    def observe(weight_map, names):
        seen.extend(names)
        assert key not in names
        return original_load(weight_map, names)

    monkeypatch.setattr(big_hf_model_helper, "load_weight_from_safetensor", observe)

    def observe_read(name, **kwargs):
        assert name != key and ".ngram_embedding.shard_" not in name
        return original_read(name, **kwargs)

    monkeypatch.setattr(loader, "read_tensor", observe_read)
    with loader.materialized("model.layers.0", skip_tensors=(key,)) as layer:
        assert layer.ple.ple_embedding.ngram_embedding.weight.is_meta
        assert all(value.is_meta == (name == "ple.ple_embedding.ngram_embedding.weight")
                   for name, value in layer.named_parameters())
    assert seen
    assert all(value.is_meta for value in loader.model.parameters())


@pytest.mark.parametrize("rows", [slice(-1, 2), slice(0, 1000000), slice(0, 2, 2), slice(True, 2)])
def test_invalid_range_is_rejected(native_checkpoint, rows):
    with pytest.raises(ValueError, match="row"):
        _loader(native_checkpoint).read_tensor("lm_head.weight", rows=rows)


@pytest.mark.parametrize("excluded", ["model.layers.1.self_attn.q_proj.weight", "model.layers.0.missing.weight"])
def test_exclusion_requires_existing_tensor_within_requested_subtree(native_checkpoint, excluded):
    loader = _loader(native_checkpoint)
    with pytest.raises(ValueError, match="Excluded tensor"):
        loader.load_module("model.layers.0", skip_tensors=(excluded,))
    assert all(value.is_meta for value in loader.model.parameters())


def test_missing_other_weight_is_rejected_even_when_ple_is_excluded(native_checkpoint):
    source, _ = native_checkpoint
    values = load_file(source / "model.safetensors")
    missing = next(name for name in values if name.startswith("model.layers.0.") and ".mlp." in name)
    values.pop(missing)
    save_file(values, source / "model.safetensors")
    loader = _loader(native_checkpoint)
    key = "model.layers.0.ple.ple_embedding.ngram_embedding.weight"
    with pytest.raises(RuntimeError, match="absent from the safetensors weight map"):
        loader.load_module("model.layers.0", skip_tensors=(key,), clone=False)
    assert all(value.is_meta for value in loader.model.parameters())


@pytest.mark.parametrize("bits", [4, 8])
def test_real_sdk_packed_payload_loads_losslessly_without_dense_weight(native_checkpoint, bits):
    from xhquant.nn import GPTQPackedLinear, GPTQPackedWeight

    source, _ = native_checkpoint
    pack = 32 // bits
    zero = sum((1 << (bits - 1)) << (lane * bits) for lane in range(pack))
    zero = zero if zero < (1 << 31) else zero - (1 << 32)
    payload = GPTQPackedWeight(
        qweight=torch.randint(-(1 << 31), (1 << 31) - 1, (64 // pack, 16), dtype=torch.int32),
        qzeros=torch.full((1, 16 // pack), zero, dtype=torch.int32),
        scales=torch.rand((1, 16), dtype=torch.float32) + 0.1,
        g_idx=torch.zeros(64, dtype=torch.int32),
        bits=bits, group_size=64, in_features=64, out_features=16,
    )
    reference = GPTQPackedLinear(payload)
    state = {"packed." + name: tensor for name, tensor in reference.state_dict().items()}
    save_file(state, source / "packed.safetensors")
    loader = _loader(native_checkpoint)
    import copy

    module = copy.deepcopy(reference).to_empty(device="meta")
    loader.load_into(module, "packed")
    assert module.weight is None
    for name, expected in reference.state_dict().items():
        actual = module.state_dict()[name]
        assert actual.dtype == expected.dtype
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    inputs = torch.randn(2, 64)
    torch.testing.assert_close(module(inputs), reference(inputs), rtol=0, atol=0)
    loader.release(module)
    assert all(value.is_meta for value in module.buffers())


def test_quantized_remote_config_redirect_is_rejected_before_backend_lookup(native_checkpoint):
    source, _ = native_checkpoint
    path = source / "config.json"
    config = json.loads(path.read_text())
    config.update(quantization_config={"quant_method": "gptq", "bits": 4, "group_size": 64},
                  auto_map={"AutoModelForCausalLM": "custom.Custom"})
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="auto_map"):
        _loader(native_checkpoint)


def test_native_conditional_checkpoint_uses_actual_language_prefix(native_checkpoint, tmp_path):
    from transformers import Qwen4ExpConfig, Qwen4ExpForConditionalGeneration

    config = Qwen4ExpConfig(
        text_config=native_checkpoint[1].config.to_dict(),
        vision_config={
            "depth": 1, "hidden_size": 16, "intermediate_size": 32,
            "num_heads": 2, "out_hidden_size": 16, "patch_size": 2,
            "temporal_patch_size": 1, "spatial_merge_size": 2,
            "num_position_embeddings": 32, "deepstack_visual_indexes": [],
        },
    )
    model = Qwen4ExpForConditionalGeneration(config).eval()
    path = tmp_path / "conditional"
    model.save_pretrained(path)
    loader = Qwen4ExpStreamingLoader(path, dtype=torch.float32)
    assert all(value.is_meta for value in loader.model.parameters())
    key = "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.weight"
    assert key not in loader.weight_map
    assert any(name.startswith(key.removesuffix(".weight") + ".shard_") for name in loader.weight_map)
    with loader.materialized("model.language_model.layers.0", skip_tensors=(key,)) as layer:
        original = model.model.language_model.layers[0]
        torch.testing.assert_close(layer.mlp.experts.gate_up_proj, original.mlp.experts.gate_up_proj, rtol=0, atol=0)
        assert layer.ple.ple_embedding.ngram_embedding.weight.is_meta
    assert all(value.is_meta for value in loader.model.parameters())


def test_missing_ple_shard_cannot_be_hidden_by_exclusion(native_checkpoint):
    source, _ = native_checkpoint
    tensors = load_file(source / "model.safetensors")
    shard = next(name for name in tensors if ".ngram_embedding.shard_0.weight" in name)
    tensors.pop(shard)
    save_file(tensors, source / "model.safetensors")
    loader = _loader(native_checkpoint)
    key = "model.layers.0.ple.ple_embedding.ngram_embedding.weight"
    with pytest.raises(ValueError, match="checkpoint-backed"):
        loader.load_module("model.layers.0", skip_tensors=(key,))


def test_excluded_table_shape_must_match_native_contract(native_checkpoint):
    source, _ = native_checkpoint
    tensors = load_file(source / "model.safetensors")
    for name in list(tensors):
        if ".ngram_embedding.shard_" in name:
            tensors[name] = tensors[name][:, :-1].contiguous()
    save_file(tensors, source / "model.safetensors")
    loader = _loader(native_checkpoint)
    key = "model.layers.0.ple.ple_embedding.ngram_embedding.weight"
    with pytest.raises(ValueError, match="Shape mismatch"):
        loader.load_module("model.layers.0", skip_tensors=(key,))
