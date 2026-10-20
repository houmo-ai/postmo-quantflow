"""Standalone NumPy host PLE matches SDK packing and native/exported hashing."""

import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding

from xhmodel_merak.xh_llm.models.qwen4_exp import ple_demo
from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import _replace_native_modules_for_export
from xhmodel_merak.xh_llm.models.qwen4_exp.ple_sefp4 import lookup_ple_sefp4, quantize_ple_sefp4_rows


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_numpy_unpack_and_dequant_matches_sdk_selected_rows(dtype):
    weight = torch.linspace(-0.1, 0.1, 5 * 160).reshape(5, 160).to(dtype)
    if dtype == torch.float16:
        weight[0] *= 2 ** -15  # FP16 subnormal scale/rounding contract.
    else:
        weight[0] *= 2 ** -60
    packed, exp = quantize_ple_sefp4_rows(weight)
    ids = np.array([[0, -1, 3, 3, -5]], dtype=np.int64)
    rows, exponents = ple_demo.gather_ple_rows(packed.numpy(), exp.numpy(), ids)
    actual = ple_demo.dequantize_ple_rows(rows, exponents, 160, str(dtype).split(".")[-1])
    expected = lookup_ple_sefp4(packed, exp, torch.from_numpy(ids), embedding_dim=160, output_dtype=dtype)
    np.testing.assert_array_equal(actual, expected.float().numpy() if dtype == torch.bfloat16 else expected.numpy())
    empty = np.empty((2, 0), dtype=np.int32)
    rows, exponents = ple_demo.gather_ple_rows(packed.numpy(), exp.numpy(), empty)
    assert ple_demo.dequantize_ple_rows(rows, exponents, 160).shape == (2, 0, 160)
    with pytest.raises(IndexError):
        ple_demo.gather_ple_rows(packed.numpy(), exp.numpy(), np.array([5]))


@pytest.mark.parametrize("valid", [0, 1, 3, 6])
def test_host_hash_matches_native_and_exportable_ids_without_mutating_history(valid):
    config = Qwen4ExpTextConfig(
        vocab_size=256, ngram_size=3, heads_per_ngram=2, ngram_vocab_size_base=32,
        make_ngram_vocab_size_divisible_by=8, eos_token_id=0,
    )
    module = Qwen4ExpTextNGramEmbedding(config, embedding_dim=32, layer_idx=0)
    spec = {
        "ngram_size": 3, "heads_per_ngram": 2, "eos_token_id": 0,
        "multipliers": module.layer_multipliers.tolist(),
        "vocab_sizes": module.ngram_heads_vocab_sizes.tolist(),
        "offsets": module.ngram_heads_offsets.tolist(),
    }
    ids = np.array([[1, 2, 0, 7, 9, 255], [255, 254, 3, 0, 12, 1]], dtype=np.int64)
    history = np.array([[4, 0], [9, 8]], dtype=np.int64)
    before = history.copy()
    actual = ple_demo.compute_ngram_ids(ids, history, spec, valid)
    np.testing.assert_array_equal(history, before)
    padded = ids.copy()
    padded[:, valid:] = 0
    captured = []
    handle = module.ngram_embedding.register_forward_pre_hook(lambda _m, args: captured.append(args[0].clone()))
    # Native no-cache history is EOS; prepend explicit history and crop its outputs.
    module(torch.from_numpy(np.concatenate((history, padded), axis=1)), None)
    np.testing.assert_array_equal(actual, captured[-1].numpy()[:, -ids.shape[1]:])
    _replace_native_modules_for_export(module, batch_size=2, sequence_length=6, cache_length=8)
    cache = SimpleNamespace(
        current_input_length=torch.tensor([valid], dtype=torch.int32),
        layers=[SimpleNamespace(conv_states=[None, None, torch.from_numpy(history.copy()).to(torch.int32)])],
    )
    module(torch.from_numpy(padded), cache)
    np.testing.assert_array_equal(actual, captured[-1].numpy())
    handle.remove()


def _package(tmp_path):
    (tmp_path / "ple").mkdir()
    packed, exp = quantize_ple_sefp4_rows(torch.linspace(-0.1, 0.1, 5 * 160).half().reshape(5, 160))
    packed.numpy().tofile(tmp_path / "ple/packed.bin")
    exp.numpy().tofile(tmp_path / "ple/exp.bin")
    spec = {
        "layer_idx": 1, "input_name": "ple_embeddings_1", "embedding_dim": 160, "output_dtype": "float16",
        "packed": {"path": "ple/packed.bin", "shape": list(packed.shape), "dtype": "int8"},
        "exponent": {"path": "ple/exp.bin", "shape": list(exp.shape), "dtype": "int16"},
    }
    return spec, packed, exp


def test_standalone_script_replays_manifest_sample_and_confines_mmap_paths(tmp_path):
    spec, packed, exp = _package(tmp_path)
    mapped = ple_demo.load_tables(tmp_path, spec)
    assert all(isinstance(item, np.memmap) and not item.flags.writeable for item in mapped)
    ids = np.array([[[0, 4], [3, 3]]], dtype=np.int64)
    expected = lookup_ple_sefp4(packed, exp, torch.from_numpy(ids), embedding_dim=160).numpy()
    np.save(tmp_path / "ple/ids.npy", ids)
    np.save(tmp_path / "ple/expected.npy", expected)
    portable_spec = json.loads(json.dumps(spec))
    portable_spec["packed"]["path"] = "packed.bin"
    portable_spec["exponent"]["path"] = "exp.bin"
    manifest = {"version": 1, "layers": [portable_spec], "samples": [
        {"layer_idx": 1, "indices": "ids.npy", "expected": "expected.npy", "component": "prefill", "step": 0},
    ]}
    (tmp_path / "ple/manifest.json").write_text(json.dumps(manifest))
    shutil.copyfile(Path(ple_demo.__file__), tmp_path / "ple_demo.py")
    result = subprocess.run(
        [sys.executable, "-I", str(tmp_path / "ple_demo.py"), "--output", str(tmp_path / "actual.npy")],
        cwd=tmp_path, check=True, text=True, capture_output=True,
    )
    assert json.loads(result.stdout)["matched_expected"] is True
    np.testing.assert_array_equal(np.load(tmp_path / "actual.npy"), expected)
    np.save(tmp_path / "other_ids.npy", np.array([[[1]]], dtype=np.int64))
    result = subprocess.run(
        [sys.executable, "-I", str(tmp_path / "ple_demo.py"), "--indices", str(tmp_path / "other_ids.npy")],
        cwd=tmp_path, check=True, text=True, capture_output=True,
    )
    assert json.loads(result.stdout)["matched_expected"] is False
    assert json.loads(result.stdout)["shape"] == [1, 1, 1, 160]
    spec["packed"]["path"] = "../escape.bin"
    with pytest.raises(ValueError, match="escapes"):
        ple_demo.load_tables(tmp_path, spec)
    spec["packed"]["path"] = "ple/packed.bin"
    with (tmp_path / "ple/packed.bin").open("ab") as stream:
        stream.write(b"x")
    with pytest.raises(ValueError, match="file size"):
        ple_demo.load_tables(tmp_path, spec)


def test_numpy_unpack_matches_all_signed_sdk_nibbles():
    from xhquant.backend.xh2a.functions.te.xh2a_matmul_triton import pack_signed_w4

    signed = (torch.arange(64, dtype=torch.int8) % 16 - 8).reshape(1, 1, 64, 1)
    packed = pack_signed_w4(signed).squeeze(-1).numpy()
    exponent = np.zeros((1, 1, 1), dtype=np.int16)
    result = ple_demo.dequantize_ple_rows(packed, exponent, 64)
    np.testing.assert_array_equal(result, signed.reshape(1, 64).half().numpy())
