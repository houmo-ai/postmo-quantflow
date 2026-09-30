import json
import os
from types import SimpleNamespace

import onnx
import pytest
import transformers.configuration_utils as transformers_config
from onnx import helper
from transformers import AutoConfig

from vllm_merak.config import MerakConfig, MerakConfigParser
from vllm_merak.constants import (
    MERAK_AUTO_OFFLOAD,
    MERAK_CUDA_GRAPH,
    MERAK_PARALLEL_LINEAR_FUSION,
    MERAK_USE_V2,
)
from vllm_merak.model_executor.model_loader.merak_for_causal_lm import _resolve_hmfp_block_size
from vllm_merak.utils import get_hmonnx_runtime_kwargs, should_preserve_hmonnx_runtime_device_map


def _write_flash_attention_hmonnx(
    tmp_path,
    *,
    filename="prefill.onnx",
    k_bits=8,
    v_bits=8,
) -> str:
    attributes = {}
    if k_bits is not None:
        attributes["k_bits"] = k_bits
    if v_bits is not None:
        attributes["v_bits"] = v_bits
    node = helper.make_node(
        "FlashAttention",
        inputs=[],
        outputs=[],
        name="flash_attention",
        domain="ai.houmo.xh2a",
        **attributes,
    )
    onnx.save(helper.make_model(helper.make_graph([node], "test", [], [])), tmp_path / filename)
    return filename


def _write_page_attention_config(tmp_path, *, prefill_hmonnx=None, **config) -> None:
    (tmp_path / "merak_config.json").write_text(
        json.dumps({"xh_model": {"model_type": "hmonnx"}, "enable_page_attention": True, **config}),
        encoding="utf-8",
    )
    if prefill_hmonnx is not None:
        (tmp_path / "golden_meta_info.json").write_text(
            json.dumps({"prefill_hmonnx": prefill_hmonnx}),
            encoding="utf-8",
        )


def test_merak_config_accepts_runtime_flags() -> None:
    config = MerakConfig(
        xh_model={"model_type": "dummy"},
        use_v2=True,
        auto_offload=True,
        cuda_graph=True,
    )

    assert config.use_v2 is True
    assert config.auto_offload is True
    assert config.cuda_graph is True


def test_hmonnx_runtime_kwargs_can_be_read_from_config(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("ENABLE_HMINFERENCE_V2", raising=False)
    monkeypatch.setattr("vllm_merak.utils.torch.cuda.device_count", lambda: 2)
    merak_config_path = tmp_path / "merak_config.json"
    merak_config_path.write_text(json.dumps({"xh_model": {"model_type": "dummy"}}), encoding="utf-8")

    _, config = MerakConfigParser().parse(
        tmp_path,
        trust_remote_code=False,
        hf_overrides={"use_v2": True, "auto_offload": True, "cuda_graph": True},
    )

    kwargs = get_hmonnx_runtime_kwargs(config)
    assert kwargs["enable_cuda_graph"] is True
    assert kwargs["enable_auto_offload"] is True
    assert kwargs["enable_parallel_linear_fusion"] is True
    assert kwargs["device_map"] == [0, 1]
    assert should_preserve_hmonnx_runtime_device_map(config) is True
    assert os.environ["ENABLE_HMINFERENCE_V2"] == "1"


def test_hmonnx_runtime_kwargs_can_be_read_from_env(monkeypatch) -> None:
    monkeypatch.delenv("ENABLE_HMINFERENCE_V2", raising=False)
    monkeypatch.setenv(MERAK_USE_V2, "1")
    monkeypatch.setenv(MERAK_AUTO_OFFLOAD, "true")
    monkeypatch.setenv(MERAK_CUDA_GRAPH, "yes")
    monkeypatch.setattr("vllm_merak.utils.torch.cuda.device_count", lambda: 1)

    kwargs = get_hmonnx_runtime_kwargs(None)

    assert kwargs == {
        "enable_cuda_graph": True,
        "enable_auto_offload": True,
        "enable_parallel_linear_fusion": True,
        "device_map": [0],
    }
    assert should_preserve_hmonnx_runtime_device_map(None) is True
    assert os.environ["ENABLE_HMINFERENCE_V2"] == "1"


def test_v2_without_auto_offload_uses_one_explicit_device(monkeypatch) -> None:
    monkeypatch.delenv("ENABLE_HMINFERENCE_V2", raising=False)
    monkeypatch.setenv(MERAK_USE_V2, "1")
    monkeypatch.setattr("vllm_merak.utils.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("vllm_merak.utils.torch.cuda.current_device", lambda: 1)

    kwargs = get_hmonnx_runtime_kwargs(None)

    assert kwargs["device_map"] == [1]
    assert kwargs["enable_parallel_linear_fusion"] is True


def test_hmonnx_parallel_linear_fusion_can_be_disabled_from_env(monkeypatch) -> None:
    monkeypatch.setenv(MERAK_USE_V2, "1")
    monkeypatch.setenv(MERAK_PARALLEL_LINEAR_FUSION, "0")

    kwargs = get_hmonnx_runtime_kwargs(None)

    assert kwargs["enable_parallel_linear_fusion"] is False


def test_merak_config_loads_chip_arch_from_meta_info(tmp_path) -> None:
    merak_config_path = tmp_path / "merak_config.json"
    merak_config_path.write_text(
        json.dumps({"xh_model": {"model_type": "hmonnx", "meta_info": "custom_meta.json"}}),
        encoding="utf-8",
    )
    meta_path = tmp_path / "custom_meta.json"
    meta_path.write_text(json.dumps({"model_config": {"chip_arch": "XH2a"}}), encoding="utf-8")

    config_dict, config = MerakConfigParser().parse(tmp_path, trust_remote_code=False)

    assert config_dict["chip_arch"] == "XH2a"
    assert config.chip_arch == "XH2a"


def test_merak_config_isolates_source_hf_weight_quantization_metadata(
    tmp_path,
) -> None:
    (tmp_path / "merak_config.json").write_text(
        json.dumps({"xh_model": {"model_type": "hmonnx"}}), encoding="utf-8"
    )
    hf_config_dir = tmp_path / "hf_config"
    hf_config_dir.mkdir()
    (hf_config_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["UnknownSourceArchitecture"],
                "quantization_config": {
                    "quant_method": "gptq",
                    "kcache_man_bit": 7,
                },
                "text_config": {
                    "quantization_config": {
                        "quant_method": "gptq",
                        "vcache_man_bit": 8,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    config_dict, config = MerakConfigParser().parse(tmp_path, trust_remote_code=False)

    assert "quantization_config" not in config_dict
    assert "quantization_config" not in config_dict["text_config"]
    assert getattr(config, "quantization_config", None) is None
    assert getattr(config.text_config, "quantization_config", None) is None
    assert config_dict["merak_source_weight_format"] == {
        "model": {"quant_method": "gptq", "kcache_man_bit": 7},
        "text": {"quant_method": "gptq", "vcache_man_bit": 8},
    }
    assert (
        config.merak_source_weight_format == config_dict["merak_source_weight_format"]
    )


def test_merak_config_allows_explicit_runtime_quantization_override(tmp_path) -> None:
    (tmp_path / "merak_config.json").write_text(
        json.dumps({"xh_model": {"model_type": "hmonnx"}}), encoding="utf-8"
    )
    hf_config_dir = tmp_path / "hf_config"
    hf_config_dir.mkdir()
    (hf_config_dir / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "gptq",
                    "desc_act": False,
                }
            }
        ),
        encoding="utf-8",
    )
    runtime_quantization = {"quant_method": "custom_merak_test"}

    config_dict, config = MerakConfigParser().parse(
        tmp_path,
        trust_remote_code=False,
        hf_overrides={"quantization_config": runtime_quantization},
    )

    assert config_dict["quantization_config"] == runtime_quantization
    assert config.quantization_config == runtime_quantization


def test_merak_config_allows_hash_moe_during_tokenizer_config_reload(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        transformers_config,
        "ALLOWED_LAYER_TYPES",
        tuple(
            layer_type
            for layer_type in transformers_config.ALLOWED_LAYER_TYPES
            if layer_type != "hash_moe"
        ),
    )
    (tmp_path / "merak_config.json").write_text(
        json.dumps({"xh_model": {"model_type": "hmonnx"}}), encoding="utf-8"
    )
    hf_config_dir = tmp_path / "hf_config"
    hf_config_dir.mkdir()
    (hf_config_dir / "config.json").write_text(
        json.dumps(
                {
                    "architectures": ["DeepseekV4ForCausalLM"],
                    "model_type": "deepseek_v4",
                    "mlp_layer_types": ["hash_moe"] * 3 + ["moe"] * 40,
                }
        ),
        encoding="utf-8",
    )

    config_dict, _ = MerakConfigParser().parse(tmp_path, trust_remote_code=False)
    source_config = AutoConfig.from_pretrained(hf_config_dir)

    assert "mlp_layer_types" not in config_dict
    assert source_config.mlp_layer_types == ["hash_moe"] * 3 + ["moe"] * 40


def test_merak_config_auto_detects_sefp_kv_cache_format_from_hmonnx(tmp_path) -> None:
    prefill_hmonnx = _write_flash_attention_hmonnx(tmp_path)
    _write_page_attention_config(tmp_path, prefill_hmonnx=prefill_hmonnx)

    config_dict, config = MerakConfigParser().parse(tmp_path, trust_remote_code=False)

    assert config_dict["kv_cache_format"] == "sefp_kint8_vint8"
    assert config.kv_cache_format == "sefp_kint8_vint8"


@pytest.mark.parametrize(
    ("k_bits", "v_bits", "expected_format"),
    [
        (8, 8, "sefp_kint8_vint8"),
        (8, 16, "sefp_kint8_vint16"),
        (16, 8, "sefp_kint16_vint8"),
    ],
)
def test_merak_config_infers_kv_payload_from_hmonnx(tmp_path, k_bits, v_bits, expected_format) -> None:
    prefill_hmonnx = _write_flash_attention_hmonnx(tmp_path, k_bits=k_bits, v_bits=v_bits)
    _write_page_attention_config(tmp_path, prefill_hmonnx=prefill_hmonnx)

    config_dict, config = MerakConfigParser().parse(tmp_path, trust_remote_code=False)

    assert config_dict["kv_cache_format"] == expected_format
    assert config.kv_cache_format == expected_format


@pytest.mark.parametrize(
    ("k_bits", "v_bits", "message"), [(8.5, 16, "must declare integer"), (4, 16, "exactly 8 or 16")]
)
def test_merak_config_rejects_invalid_hmonnx_flash_kv_bits(tmp_path, k_bits, v_bits, message) -> None:
    prefill_hmonnx = _write_flash_attention_hmonnx(tmp_path, k_bits=k_bits, v_bits=v_bits)
    _write_page_attention_config(tmp_path, prefill_hmonnx=prefill_hmonnx)

    with pytest.raises(ValueError, match=message):
        MerakConfigParser().parse(tmp_path, trust_remote_code=False)


def test_merak_config_rejects_incomplete_hmonnx_flash_kv_bits(tmp_path) -> None:
    prefill_hmonnx = _write_flash_attention_hmonnx(tmp_path, k_bits=16, v_bits=None)
    _write_page_attention_config(tmp_path, prefill_hmonnx=prefill_hmonnx)

    with pytest.raises(ValueError, match="must declare integer k_bits and v_bits"):
        MerakConfigParser().parse(tmp_path, trust_remote_code=False)


def test_merak_config_keeps_explicit_kv_cache_format(tmp_path) -> None:
    prefill_hmonnx = _write_flash_attention_hmonnx(tmp_path)
    _write_page_attention_config(tmp_path, prefill_hmonnx=prefill_hmonnx, kv_cache_format="float16")

    config_dict, config = MerakConfigParser().parse(tmp_path, trust_remote_code=False)

    assert config_dict["kv_cache_format"] == "float16"
    assert config.kv_cache_format == "float16"


def test_merak_config_does_not_auto_detect_sefp_when_page_attention_is_disabled(tmp_path) -> None:
    prefill_hmonnx = _write_flash_attention_hmonnx(tmp_path)
    (tmp_path / "merak_config.json").write_text(
        json.dumps({"xh_model": {"model_type": "hmonnx"}, "enable_page_attention": False}),
        encoding="utf-8",
    )
    (tmp_path / "golden_meta_info.json").write_text(
        json.dumps({"prefill_hmonnx": prefill_hmonnx}),
        encoding="utf-8",
    )

    config_dict, config = MerakConfigParser().parse(tmp_path, trust_remote_code=False)

    assert "kv_cache_format" not in config_dict
    assert config.kv_cache_format == "float16"


def test_hmfp_block_size_defaults_to_64() -> None:
    assert _resolve_hmfp_block_size(None) == 64
    assert _resolve_hmfp_block_size(SimpleNamespace(block_size=None)) == 64


def test_hmfp_block_size_accepts_multiple_of_64() -> None:
    assert _resolve_hmfp_block_size(SimpleNamespace(block_size=128)) == 128


def test_hmfp_block_size_rejects_non_multiple_of_64() -> None:
    try:
        _resolve_hmfp_block_size(SimpleNamespace(block_size=96))
    except ValueError as exc:
        assert "positive multiple of 64" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non-64-multiple block_size")
