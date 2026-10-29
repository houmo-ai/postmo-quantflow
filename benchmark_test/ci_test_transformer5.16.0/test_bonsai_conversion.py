"""Unit tests for Bonsai → SSFP4 Phase A conversion and Hadamard MatMul.

These tests intentionally load Phase A modules by file path. Offline conversion stays usable without initializing the quantization SDK.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


bonsai_ssfp4 = _load(
    "test_pq2_bonsai_ssfp4",
    "xhmodel_merak/xh_llm/models/bonsai/bonsai_ssfp4.py",
)
hadamard_matmul = _load(
    "test_pq2_hadamard_matmul",
    "xhmodel_merak/xh_llm/models/bonsai/hadamard_matmul.py",
)


def _pack_codes_to_u32(codes: torch.Tensor) -> torch.Tensor:
    rows, width = codes.shape
    assert width % 16 == 0
    words = width // 16
    out = torch.zeros((rows, words), dtype=torch.int64)
    for lane in range(16):
        out |= (codes[:, lane::16].to(torch.int64) & 3) << (2 * lane)
    return out.to(torch.int64)


def test_bonsai_name_to_hf_name():
    assert (
        bonsai_ssfp4.bonsai_name_to_hf_name(
            "language_model.model.layers.0.mlp.gate_proj.weight"
        )
        == "model.language_model.layers.0.mlp.gate_proj"
    )
    assert bonsai_ssfp4.bonsai_name_to_hf_name("language_model.lm_head.weight") == "lm_head"
    assert (
        bonsai_ssfp4.bonsai_name_to_hf_name("language_model.model.embed_tokens")
        == "model.language_model.embed_tokens"
    )


def test_unpack_and_k64_scale_duplication_synthetic():
    rows, width = 4, 128
    codes = torch.randint(0, 3, (rows, width), dtype=torch.int64)
    weight_u32 = _pack_codes_to_u32(codes)
    scales = torch.rand((rows, width // 128), dtype=torch.float16) * 0.02 + 0.01
    biases = -scales

    converted = bonsai_ssfp4.convert_bonsai_linear_to_ssfp4(weight_u32, scales, biases)
    assert converted["quant_weight"].dtype == torch.int8
    assert set(converted["quant_weight"].unique().tolist()).issubset({-1, 0, 1})
    assert converted["w_scale_k64"].shape == (rows, width // 64)
    assert torch.equal(converted["w_scale_k64"][:, 0], scales[:, 0])
    assert torch.equal(converted["w_scale_k64"][:, 1], scales[:, 0])
    assert converted["weight"].shape == (rows, width)

    meta = bonsai_ssfp4.build_phase_a_metadata(num_modules=1)
    assert meta["pq2_weight"] is False
    assert meta["format"] == bonsai_ssfp4.PHASE_A_FORMAT


def test_duplicate_k128_rejects_negative_scale():
    scales = torch.tensor([[0.1, -0.2]], dtype=torch.float16)
    with pytest.raises(ValueError, match="Negative"):
        bonsai_ssfp4.duplicate_k128_scales_to_k64(scales)


def test_hadamard_matmul_matches_fwht_reference():
    width = 1024
    block = 1024
    signs = torch.randint(0, 2, (width,), dtype=torch.float32) * 2 - 1
    x = torch.randn(2, 3, width, dtype=torch.float32)

    y_ref = hadamard_matmul.fwht_reference(x, signs, block_size=block, inverse=False)
    y_mm = hadamard_matmul.apply_block_hadamard_matmul(
        x, signs, block_size=block, inverse=False
    )
    assert torch.allclose(y_ref, y_mm, atol=1e-5, rtol=1e-5)

    y_ref_inv = hadamard_matmul.fwht_reference(x, signs, block_size=block, inverse=True)
    y_mm_inv = hadamard_matmul.apply_block_hadamard_matmul(
        x, signs, block_size=block, inverse=True
    )
    assert torch.allclose(y_ref_inv, y_mm_inv, atol=1e-5, rtol=1e-5)

    w = hadamard_matmul.build_hadamard_matmul_weight(
        width, signs, block_size=block, dtype=torch.float32
    )
    assert w.shape == (width, width)
    gram = w @ w.T
    assert torch.allclose(gram, torch.eye(width), atol=1e-5, rtol=1e-5)


def test_split_export_and_aux_tensors():
    state = {
        "model.language_model.layers.0.mlp.gate_proj.weight": torch.zeros(2, 4),
        "model.language_model.layers.0.mlp.gate_proj.quant_weight": torch.zeros(
            2, 4, dtype=torch.int8
        ),
        "model.language_model.layers.0.mlp.gate_proj.w_scale_k64": torch.ones(2, 1),
        "model.language_model.layers.0.mlp.gate_proj.hadamard_signs": torch.ones(4),
        "hadamard_matmul.forward.w4": torch.eye(4),
        "model.language_model.layers.0.input_layernorm.weight": torch.ones(4),
    }
    export_sd, aux_sd = bonsai_ssfp4.split_phase_a_export_tensors(state)
    assert "model.language_model.layers.0.mlp.gate_proj.weight" in export_sd
    assert "model.language_model.layers.0.mlp.gate_proj.quant_weight" in export_sd
    assert "model.language_model.layers.0.input_layernorm.weight" in export_sd
    assert "model.language_model.layers.0.mlp.gate_proj.w_scale_k64" in aux_sd
    assert "hadamard_matmul.forward.w4" in aux_sd


@pytest.mark.skipif(
    not Path("/data01/datasets/Ternary-Bonsai-2-27B-mlx-2bit/model.safetensors").exists(),
    reason="Bonsai pack not available on this machine",
)
def test_real_bonsai_gate_proj_layer0():
    from safetensors import safe_open

    path = "/data01/datasets/Ternary-Bonsai-2-27B-mlx-2bit/model.safetensors"
    base = "language_model.model.layers.0.mlp.gate_proj"
    with safe_open(path, framework="pt") as f:
        weight = f.get_slice(base + ".weight")[:2]
        scales = f.get_slice(base + ".scales")[:2]
        biases = f.get_slice(base + ".biases")[:2]
        signs = f.get_tensor(base + ".signs")

    codes = bonsai_ssfp4.unpack_bonsai_u32_codes(weight)
    assert set(codes.unique().tolist()).issubset({0, 1, 2})
    converted = bonsai_ssfp4.convert_bonsai_linear_to_ssfp4(weight, scales, biases)
    assert converted["quant_weight"].shape == (2, 5120)
    assert converted["w_scale_k64"].shape == (2, 80)
    assert signs.numel() == 5120

    x = torch.randn(1, 5120, dtype=torch.float32)
    y_ref = hadamard_matmul.fwht_reference(
        x, signs.float(), block_size=1024, inverse=False
    )
    y_mm = hadamard_matmul.apply_block_hadamard_matmul(
        x, signs.float(), block_size=1024, inverse=False
    )
    assert torch.allclose(y_ref, y_mm, atol=1e-4, rtol=1e-4)


def test_full_pack_converts_embedding_norm_and_conv(tmp_path):
    from safetensors.torch import save_file
    import json
    convert = bonsai_ssfp4
    had = hadamard_matmul
    codes = torch.arange(4 * 128).reshape(4, 128) % 3
    words = torch.zeros(4, 8, dtype=torch.int64)
    for lane in range(16):
        words |= codes[:, lane::16] << (2 * lane)
    scale = torch.full((4, 1), 0.03125, dtype=torch.float16)
    signs = (torch.arange(128) % 2 * 2 - 1).float()
    source = {'language_model.model.embed_tokens.' + leaf: value for leaf, value in
              {'weight': words.to(torch.uint32), 'scales': scale, 'biases': -scale, 'signs': signs}.items()}
    source['language_model.model.layers.0.input_layernorm.weight'] = torch.linspace(0.2, 1.2, 128)
    source['language_model.model.layers.0.linear_attn.norm.weight'] = torch.linspace(0.1, 1.1, 16)
    source['language_model.model.layers.0.linear_attn.conv1d.weight'] = torch.arange(32).float().reshape(8, 4, 1)
    save_file(source, tmp_path / 'model.safetensors')
    (tmp_path / 'hadamard.json').write_text(json.dumps({'prism.hadamard.block_size': 32}))
    state, meta = convert.convert_bonsai_pack_to_ssfp4_state_dict(tmp_path)
    decoded = (codes.float() - 1) * scale
    expected = had.apply_block_hadamard_matmul(decoded, signs, block_size=32, inverse=True).half()
    torch.testing.assert_close(state['model.language_model.embed_tokens.weight'], expected, rtol=0, atol=0)
    assert 'model.language_model.embed_tokens.quant_weight' not in state
    assert meta['version'] == 2 and meta['complete_language_model']
    assert meta['float_tensor_count'] == 3 and meta['embedding_inverse_applied']
    torch.testing.assert_close(state['model.language_model.layers.0.input_layernorm.weight'],
                               source['language_model.model.layers.0.input_layernorm.weight'] - 1)
    torch.testing.assert_close(state['model.language_model.layers.0.linear_attn.norm.weight'],
                               source['language_model.model.layers.0.linear_attn.norm.weight'])
    torch.testing.assert_close(state['model.language_model.layers.0.linear_attn.conv1d.weight'],
                               source['language_model.model.layers.0.linear_attn.conv1d.weight'].transpose(1, 2))


def test_converter_cli_emits_only_required_forward_bank(tmp_path):
    from safetensors.torch import save_file

    source_dir, output_dir = tmp_path / 'source', tmp_path / 'converted'
    source_dir.mkdir()
    codes = torch.arange(4 * 128).reshape(4, 128) % 3
    scales = torch.full((4, 1), 0.03125, dtype=torch.float16)
    signs = (torch.arange(128) % 2 * 2 - 1).float()
    tensors = {}
    for name in ('language_model.model.embed_tokens', 'language_model.model.layers.0.mlp.gate_proj'):
        for suffix, value in {'weight': _pack_codes_to_u32(codes).to(torch.uint32),
                              'scales': scales, 'biases': -scales, 'signs': signs}.items():
            tensors[f'{name}.{suffix}'] = value.clone()
    save_file(tensors, source_dir / 'model.safetensors')
    (source_dir / 'hadamard.json').write_text(json.dumps({'prism.hadamard.block_size': 32}))
    (source_dir / 'config.json').write_text(json.dumps({
        'model_type': 'qwen3_5', 'text_config': {'hidden_size': 128},
        'quantization': {'bits': 2}, 'quantization_config': {'group_size': 128},
    }))
    (source_dir / 'tokenizer_config.json').write_text('{}')
    subprocess.run([sys.executable, str(ROOT / 'examples_merak/llm/bonsai/convert_bonsai_to_ssfp4_phase_a.py'),
                    '--bonsai-dir', str(source_dir), '--output-dir', str(output_dir), '--filename', 'payload.pt'],
                   check=True, capture_output=True, text=True)
    aux = torch.load(output_dir / 'phase_a_aux.pt', weights_only=True)
    assert aux['hadamard_matmul.forward.w128'].shape == (128, 128)
    assert not any(key.startswith('hadamard_matmul.inverse.') for key in aux)
    metadata = json.loads((output_dir / 'phase_a_metadata.json').read_text())
    assert metadata['embedding_inverse_applied'] and metadata['hadamard']['block_size'] == 32
    assert metadata['checkpoint_file'] == 'payload.pt' and (output_dir / 'payload.pt').is_file()
    assert (output_dir / 'tokenizer_config.json').is_file()
    config = json.loads((output_dir / 'config.json').read_text())
    assert config == {'model_type': 'qwen3_5', 'text_config': {'hidden_size': 128}}
