"""Spark mixed-cache allocation, metadata and real sliding attention."""

import json

import pytest
import torch


@pytest.fixture
def spark(tmp_path):
    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.spark2_5_model import XHSpark2_5Model, XHSpark2_5ModelConfig
    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.spark_2_5 import Spark2_5Config, Spark2_5ForCausalLM

    torch.manual_seed(251)
    config = Spark2_5Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
        headwise_attn_output_gate=True,
        max_position_embeddings=128,
    )
    source = tmp_path / "source"
    config.save_pretrained(source)
    model = XHSpark2_5Model(
        XHSpark2_5ModelConfig(
            model_name="spark",
            hf_model=str(source),
            context_max_length=64,
            prefill_chunk_length=4,
            max_pe_length=128,
            dtype="float32",
            flash_attention={"enable": False},
        )
    )
    model.kvcache_config.cache_dtype = "float32"
    model.init_wrap_model(Spark2_5ForCausalLM(config).eval())
    return model


@pytest.mark.parametrize(
    "context,chunk,expected", [(2048, 256, 768), (131072, 256, 768), (1048576, 256, 768), (2048, 1, 528)]
)
def test_checkpoint_layer_capacities(context, chunk, expected):
    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.kv_cache import build_layer_kv_shapes

    # Released 4B layout, without allocating the full model or its caches.
    types = ["sliding_attention"] * 3 + ["full_attention"]
    shapes = build_layer_kv_shapes(types * 9, [1, 4, context, 256], 512, chunk)
    assert len(shapes) == 36
    assert shapes.count([1, 4, expected, 256]) == 27
    assert shapes.count([1, 4, context, 256]) == 9


def test_compact_cache_matches_full_storage_across_windows(spark):
    from xhmodel_merak.xh_llm.types import CacheList
    from xhquant.core import CacheTensor, HybridCacheTensor
    from xhquant.frontend.utils import module_graph_forward_guard

    mixin = spark.get_kvcache_mixin()
    mixin.prepare_kv_cache()
    assert [list(c.shape) for c in spark.past_key_caches] == [[1, 1, 16, 16], [1, 1, 64, 16]]
    assert isinstance(spark.past_key_caches[0], HybridCacheTensor)
    assert type(spark.past_key_caches[1]) is CacheTensor
    first = spark.past_key_caches[0]
    mixin.prepare_kv_cache()
    assert len(spark.past_key_caches) == 2 and spark.past_key_caches[0] is first
    reference_keys = CacheList([CacheTensor(torch.zeros(1, 1, 64, 16)) for _ in range(2)])
    reference_values = CacheList([CacheTensor(torch.zeros(1, 1, 64, 16)) for _ in range(2)])
    processor = spark._get_data_preprocessor().to(dtype=torch.float32)
    past = 0
    with torch.no_grad(), module_graph_forward_guard(spark._wrap_model):
        # Repeated shifts beyond physical capacity, partial prefill, then decode.
        for width, valid in [(4, 4)] * 5 + [(4, 2)] + [(1, 1)] * 3:
            spark.wrap_cfg.input_sequence_length = width
            for module in spark._wrap_model.modules():
                if hasattr(module, "_update_cfg"):
                    module._update_cfg(spark.wrap_cfg)
            processor.input_sequence_length = width
            args = processor({"input_ids": torch.randint(3, 32, (1, valid)), "past_seq_length": past})
            actual = spark._wrap_model(*args)
            expected = spark._wrap_model(*args[:4], reference_keys, reference_values)
            torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
            retained = min(past, 7) + valid
            past += valid
            for compact, full in [(spark.past_key_caches, reference_keys), (spark.past_value_caches, reference_values)]:
                assert compact[0].cache_valid_len == retained
                torch.testing.assert_close(compact[0][:, :, :retained], full[0][:, :, past - retained : past])
                torch.testing.assert_close(compact[1], full[1])
    mixin.clear_kv_cache()
    mixin.prepare_kv_cache()
    assert spark.past_key_caches[0].cache_valid_len == 0
    assert not torch.count_nonzero(spark.past_key_caches[0])


@pytest.mark.parametrize(
    "layer_types,context",
    [
        (["sliding_attention"], 64),
        (["full_attention", "sliding_attention"], 16),
        (["sliding_attention", "full_attention"], 8),
    ],
)
def test_cache_kind_follows_layer_type_not_relative_capacity(layer_types, context):
    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.kv_cache import SparkKVCacheMixin, build_layer_kv_shapes
    from xhmodel_merak.xh_llm.types import KVCacheConfig
    from xhquant.core import CacheTensor, HybridCacheTensor

    shape = [1, 1, context, 4]
    config = KVCacheConfig(num_layers=len(layer_types), kv_cache_shape=shape)
    mixin = SparkKVCacheMixin(config, layer_types, build_layer_kv_shapes(layer_types, shape, 8, 4))
    with mixin.kv_cache_scope():
        for layer_type, cache in zip(layer_types, mixin.past_key_caches, strict=True):
            expected = HybridCacheTensor if layer_type == "sliding_attention" else CacheTensor
            assert type(cache) is expected
            assert cache.dtype == torch.float16
    assert not mixin.past_key_caches and not mixin.past_value_caches


def test_export_metadata_drives_runtime_cache(spark, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from xhmodel_merak.xh_llm.hmonnx import base_llm_hmonnx_model as base
    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.spark2_5_hmonnx_inference import XHSpark2_5HMONNXModel
    from xhmodel_merak.xh_llm.types import LLMModelMeta
    from xhquant.core import CacheTensor, HybridCacheTensor

    output = tmp_path / "release"
    output.mkdir()
    meta = spark.create_export_metadata(str(output))
    spark._save_export_token_embedding(output, meta)
    data = json.loads(json.dumps(meta.to_dict()))
    assert data["layer_types"] == ["sliding_attention", "full_attention"]
    assert data["layer_kv_shapes"] == [[1, 1, 16, 16], [1, 1, 64, 16]]
    data["hf_config"] = str(output / data["hf_config"])
    data["quant_embedding"] = str(output / data["quant_embedding"])
    meta = LLMModelMeta.from_dict(data)
    # Replace only file/session execution; run metadata loading, embedding
    # loading, backend capture and the actual runtime cache allocator.
    monkeypatch.setenv("XHMODEL_INFERENCE_BACKEND", "hmonnx")
    monkeypatch.setenv("ENABLE_HMINFERENCE_V2", "1")
    monkeypatch.setattr(base.BaseLLMHMONNXModel, "_load_hmonnx_graphs", lambda *args: (None, None, {}))
    monkeypatch.setattr(
        base,
        "HMONNXModel",
        lambda *args, **kwargs: SimpleNamespace(hmonnx_session=SimpleNamespace(get_layer_infos=lambda: [])),
    )
    runtime = XHSpark2_5HMONNXModel(meta, device_map="cpu")
    monkeypatch.setenv("XHMODEL_INFERENCE_BACKEND", "invalid-after-binding")
    runtime._kvcache_mixin.prepare_kv_cache()
    assert isinstance(runtime.past_key_caches[0], HybridCacheTensor)
    assert type(runtime.past_key_caches[1]) is CacheTensor
    assert [list(c.shape) for c in runtime.past_key_caches] == data["layer_kv_shapes"]


@pytest.mark.parametrize("corruption", ["missing", "shape", "count"])
def test_runtime_rejects_inconsistent_cache_metadata(spark, tmp_path, corruption):
    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.spark2_5_hmonnx_inference import XHSpark2_5HMONNXModel

    meta = spark.create_export_metadata(str(tmp_path))
    if corruption == "missing":
        del meta.layer_kv_shapes
    elif corruption == "shape":
        meta.layer_kv_shapes[0][2] = 64
    else:
        meta.layer_types.pop()
    with pytest.raises(ValueError, match="re-export"):
        XHSpark2_5HMONNXModel(meta, device_map="cpu")
