"""Real checkpoint row streaming and Host PLE state contracts."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding

from xhmodel_merak.xh_llm.models.qwen4_exp.ple_sefp4 import quantize_ple_sefp4_rows
from xhmodel_merak.xh_llm.models.qwen4_exp.streaming_ple import (
    HostPLEInput,
    prepare_streamed_ple,
    set_streamed_ple_inputs,
    streamed_ple_tensor_names,
)


def _source(tmp_path):
    config = Qwen4ExpTextConfig(
        ngram_size=3, heads_per_ngram=1, ngram_vocab_size_base=5,
        eos_token_id=0, vocab_size=16, make_ngram_vocab_size_divisible_by=1,
    )
    native = Qwen4ExpTextNGramEmbedding(config, embedding_dim=128, layer_idx=1).half()
    layer = torch.nn.Module()
    layer.ple_embedding = deepcopy(native)
    shape = native.ngram_embedding.weight.shape
    layer.ple_embedding.ngram_embedding = torch.nn.Embedding(*shape, device="meta", dtype=torch.float16)
    key = "model.language_model.layers.1.ple_embedding.ngram_embedding.weight"
    path = tmp_path / "weights.safetensors"
    save_file({key: native.ngram_embedding.weight.detach()}, path)

    class Reader:
        dtype = torch.float16

        def __init__(self):
            self.read_sizes = []

        def tensor_info(self, name):
            assert name == key
            with safe_open(path, framework="pt") as reader:
                return {"shape": reader.get_slice(name).get_shape(), "dtype": "F16"}

        def iter_tensor_rows(self, name, *, chunk_rows):
            with safe_open(path, framework="pt") as reader:
                value = reader.get_slice(name)
                for start in range(0, shape[0], chunk_rows):
                    chunk = value[start:start + chunk_rows]
                    self.read_sizes.append(len(chunk))
                    yield chunk

    return layer, Reader(), key, native


def test_streamed_sidecars_match_real_sefp4_rows_and_host_updates_history(tmp_path):
    layer, loader, key, native = _source(tmp_path)
    assert streamed_ple_tensor_names(layer, "model.language_model.layers.1") == (key,)
    specs = prepare_streamed_ple(loader, layer, "model.language_model.layers.1", tmp_path, chunk_rows=2)
    assert isinstance(layer.ple_embedding, HostPLEInput)
    assert list(layer.parameters()) == []
    assert max(loader.read_sizes) == 2
    assert sum(loader.read_sizes) == native.ngram_embedding.num_embeddings
    spec, = specs
    packed, exponent = quantize_ple_sefp4_rows(native.ngram_embedding.weight.detach())
    assert (tmp_path / spec["packed"]["path"]).read_bytes() == packed.numpy().tobytes()
    assert (tmp_path / spec["exponent"]["path"]).read_bytes() == exponent.numpy().astype("<i2").tobytes()
    embeddings = torch.randn(1, 3, 2, 64, dtype=torch.float16)
    set_streamed_ple_inputs(layer, specs, [embeddings])
    cache = SimpleNamespace(
        layers=[None, SimpleNamespace(conv_states=[None, None, torch.tensor([[4, 5]], dtype=torch.int32)])],
        current_input_length=torch.tensor([2], dtype=torch.int32),
    )
    output = layer.ple_embedding(torch.tensor([[6, 7, 0]], dtype=torch.int32), cache)
    assert torch.equal(output, embeddings.flatten(-2))
    assert torch.equal(cache.layers[1].conv_states[2], torch.tensor([[6, 7]], dtype=torch.int32))
    replacement = embeddings + 1
    set_streamed_ple_inputs(layer, specs, [replacement], cache=cache)
    assert layer.ple_embedding.embeddings is embeddings
    assert torch.equal(layer.ple_embedding(torch.tensor([[6, 7, 0]], dtype=torch.int32), cache),
                       replacement.flatten(-2))


def test_failed_stream_cleans_only_its_own_sidecars_and_keeps_meta_layer(tmp_path):
    layer, loader, _, _ = _source(tmp_path)
    original = loader.iter_tensor_rows

    def incomplete(name, *, chunk_rows):
        yield next(original(name, chunk_rows=chunk_rows))

    loader.iter_tensor_rows = incomplete
    keep = tmp_path / "keep.packed.bin"
    keep.write_bytes(b"unrelated")
    with pytest.raises(ValueError, match="Incomplete"):
        prepare_streamed_ple(loader, layer, "model.language_model.layers.1", tmp_path, chunk_rows=2)
    assert list(tmp_path.glob("*.packed.bin")) == [keep]
    assert not list(tmp_path.glob("*.exponent.bin"))
    assert layer.ple_embedding.ngram_embedding.weight.is_meta


def test_host_ple_requires_explicit_input_and_checks_count():
    module = HostPLEInput(0, 2)
    with pytest.raises(ValueError, match="explicit Host"):
        module(torch.zeros(1, 1, dtype=torch.int32), None)
    with pytest.raises(ValueError, match="count"):
        set_streamed_ple_inputs(module, [], [np.zeros(1)])
