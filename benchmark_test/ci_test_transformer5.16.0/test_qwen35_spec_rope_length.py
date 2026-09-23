"""Spec draft RoPE tables follow the target max_pe_length, not the KV cache."""

import json
from pathlib import Path

from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import (
    XHQwen3_5_DFlashConfig,
    XHQwen3_5_MTPConfig,
    XHQwen3_5ModelConfig,
)


def _assistant(path: Path) -> str:
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps({"dflash_config": {"mask_token_id": 7, "block_size": 16}}),
        encoding="utf-8",
    )
    return str(path)


def test_spec_draft_rope_follows_target_max_pe_length_not_kv_cache(tmp_path):
    context_max_length = 8192
    max_pe_length = 262144
    assistant = _assistant(tmp_path / "assistant")

    mtp = XHQwen3_5ModelConfig(
        model_name="qwen3_5",
        context_max_length=context_max_length,
        max_pe_length=max_pe_length,
        spec_decode_mode="mtp",
        mtp_config={"hf_model": "weights/target"},
    )
    dflash = XHQwen3_5ModelConfig(
        model_name="qwen3_5",
        context_max_length=context_max_length,
        max_pe_length=max_pe_length,
        spec_decode_mode="dflash",
        dflash_config={"hf_model": assistant},
    )

    assert mtp.mtp_config.context_max_length == context_max_length
    assert mtp.mtp_config.max_pe_length == max_pe_length
    assert dflash.dflash_config.max_sequence_length == context_max_length
    assert dflash.dflash_config.max_pe_length == max_pe_length
    assert dflash.dflash_config.max_pe_length != dflash.dflash_config.max_sequence_length


def test_prebuilt_spec_configs_are_realigned_to_the_target(tmp_path):
    context_max_length = 8192
    max_pe_length = 131072
    assistant = _assistant(tmp_path / "assistant")
    mtp_config = XHQwen3_5_MTPConfig(
        model_name="qwen3_5_mtp",
        hf_model="weights/stale",
        context_max_length=2048,
        max_pe_length=32768,
    )
    dflash_config = XHQwen3_5_DFlashConfig(
        model_name="qwen3_5_dflash",
        hf_model=assistant,
        max_sequence_length=2048,
        max_pe_length=32768,
    )

    config = XHQwen3_5ModelConfig(
        model_name="qwen3_5",
        hf_model="weights/target",
        context_max_length=context_max_length,
        max_pe_length=max_pe_length,
        spec_decode_mode="dflash",
        mtp_config=mtp_config,
        dflash_config=dflash_config,
    )

    assert config.mtp_config.context_max_length == context_max_length
    assert config.mtp_config.max_pe_length == max_pe_length
    assert config.dflash_config.max_sequence_length == context_max_length
    assert config.dflash_config.max_pe_length == max_pe_length
    assert config.dflash_config.target_model_dir == "weights/target"


def test_draft_rope_falls_back_to_cache_length_when_target_pe_is_none(tmp_path):
    assistant = _assistant(tmp_path / "assistant")
    mtp_source = {"hf_model": "weights/target", "max_pe_length": 2048}
    dflash_source = {"hf_model": assistant, "max_pe_length": 2048}
    config = XHQwen3_5ModelConfig(
        model_name="qwen3_5",
        context_max_length=8192,
        max_pe_length=None,
        spec_decode_mode="dflash",
        mtp_config=mtp_source,
        dflash_config=dflash_source,
    )

    assert config.mtp_config.context_max_length == 8192
    assert config.mtp_config.max_pe_length == 8192
    assert config.dflash_config.max_sequence_length == 8192
    assert config.dflash_config.max_pe_length == 8192
    assert mtp_source["max_pe_length"] == 2048
    assert dflash_source["max_pe_length"] == 2048
