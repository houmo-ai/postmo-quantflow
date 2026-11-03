"""MiniCPM DSpark configuration layered over the unchanged Qwen3.5 config."""

from __future__ import annotations

import json
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

from ..qwen3_5.xh_qwen3_5_config import XHQwen3_5_DFlashConfig, XHQwen3_5ModelConfig
from ..qwen3_5_moe.xh_qwen3_5_moe_config import XHQwen3_5MoeModelConfig


def _mapping(value):
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    return dict(vars(value))


class MiniCPMV47DSparkConfig(XHQwen3_5_DFlashConfig):
    """Keep DSpark fields that the restored Qwen DFlash config no longer stores."""

    def __init__(self, **kwargs):
        self.num_output_draft_tokens = kwargs.pop("num_output_draft_tokens", None)
        truncation = kwargs.pop("enable_confidence_truncation", False)
        if type(truncation) is not bool:
            raise ValueError("enable_confidence_truncation must be boolean")
        self.enable_confidence_truncation = truncation
        self.confidence_threshold = kwargs.pop("confidence_threshold", None)
        self.is_dspark = bool(kwargs.pop("is_dspark", False))
        if self.enable_confidence_truncation:
            if type(self.confidence_threshold) not in (int, float) or not 0 < float(self.confidence_threshold) <= 1:
                raise ValueError("MiniCPM confidence_threshold must be in (0, 1]")
        elif self.confidence_threshold is not None:
            raise ValueError("confidence_threshold requires enable_confidence_truncation")
        with self._prepared_flat_checkpoint(kwargs.get("hf_model")):
            super().__init__(**kwargs)
        if self.num_output_draft_tokens is not None:
            self.num_output_draft_tokens = int(self.num_output_draft_tokens)
        path = Path(str(self.hf_model)) / "config.json"
        if not path.is_file() or self.architecture != "Qwen3DSparkModel":
            return
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.is_dspark = True
        self.num_target_layers = len(raw["target_layer_ids"])
        self.block_size = int(raw["block_size"])
        self.noise_token_id = int(raw["mask_token_id"])

    @staticmethod
    @contextmanager
    def _prepared_flat_checkpoint(model_dir):
        path = Path(str(model_dir)) / "config.json" if model_dir else None
        original = None
        if path is not None and path.is_file():
            original = path.read_text(encoding="utf-8")
            raw = json.loads(original)
            if raw.get("architectures") == ["Qwen3DSparkModel"] and "dflash_config" not in raw:
                raw["dflash_config"] = {
                    "target_layer_ids": raw["target_layer_ids"],
                    "mask_token_id": raw["mask_token_id"],
                    "block_size": raw["block_size"],
                }
                raw["num_target_layers"] = len(raw["target_layer_ids"])
                path.write_text(json.dumps(raw), encoding="utf-8")
        try:
            yield
        finally:
            if original is not None:
                path.write_text(original, encoding="utf-8")


class MiniCPMV47ModelConfig(XHQwen3_5ModelConfig):
    """Apply DSpark capacity and flash-attention rules after the Qwen parent."""

    def __init__(self, **kwargs):
        # Public MiniCPM settings use DSpark; the reused Qwen exporter
        # consumes its existing DFlash configuration and dispatch fields.
        dspark = kwargs.pop("dspark_config", None)
        if dspark is not None:
            if kwargs.get("dflash_config") is not None:
                raise ValueError("Configure only dspark_config, not both draft aliases")
            kwargs["dflash_config"] = dspark
        if kwargs.get("spec_decode_mode") == "dspark":
            kwargs["spec_decode_mode"] = "dflash"
        raw_dflash = kwargs.get("dflash_config")
        raw = _mapping(raw_dflash)
        if isinstance(raw_dflash, Mapping):
            explicit_flash = raw.get("flash_attention") if "flash_attention" in raw else None
        else:
            explicit_flash = getattr(raw_dflash, "flash_attention", None)
        is_dspark = bool(raw.get("is_dspark", False)) or raw.get("architecture") == "Qwen3DSparkModel"
        draft_dir = raw.get("hf_model")
        if draft_dir and Path(str(draft_dir), "config.json").is_file():
            checkpoint = json.loads(Path(str(draft_dir), "config.json").read_text(encoding="utf-8"))
            is_dspark = is_dspark or checkpoint.get("architectures") == ["Qwen3DSparkModel"]
            raw.setdefault("block_size", checkpoint.get("block_size"))
            truncation = raw.get("enable_confidence_truncation", False)
            if type(truncation) is not bool:
                raise ValueError("enable_confidence_truncation must be boolean")
            if truncation and checkpoint.get("enable_confidence_head") is not True:
                raise ValueError("confidence truncation requires a checkpoint confidence head")
        requested_tokens = kwargs.get("num_draft_tokens")
        if is_dspark and requested_tokens is not None and raw.get("block_size") is not None:
            if int(requested_tokens) > int(raw["block_size"]):
                raise ValueError(
                    "MiniCPM DSpark num_draft_tokens exceeds checkpoint "
                    f"capacity: proposals={requested_tokens}, block_size={raw['block_size']}"
                )
            # The restored parent rejects block_size == num_draft_tokens.
            # Pass a legal value through that check, then restore the request.
            kwargs["num_draft_tokens"] = max(0, int(raw["block_size"]) - 1)
            if isinstance(raw_dflash, Mapping):
                adjusted = dict(raw_dflash)
                adjusted["flash_attention"] = raw.get("flash_attention")
                kwargs["dflash_config"] = adjusted
        with MiniCPMV47DSparkConfig._prepared_flat_checkpoint(raw.get("hf_model")):
            super().__init__(**kwargs)
        if self.dflash_config is None:
            return
        is_dspark = is_dspark or getattr(self.dflash_config, "architecture", None) == "Qwen3DSparkModel"
        if not is_dspark:
            return
        if requested_tokens is not None:
            self.num_draft_tokens = int(requested_tokens)
        values = _mapping(self.dflash_config)
        values.update(
            is_dspark=True,
            architecture=values.get("architecture") or "Qwen3DSparkModel",
            num_output_draft_tokens=self.num_draft_tokens,
            enable_confidence_truncation=bool(raw.get("enable_confidence_truncation", False)),
            confidence_threshold=raw.get("confidence_threshold"),
        )
        # A non-causal draft must mask padding explicitly. Do not inherit
        # the target's causal FA when loading older presets without this field.
        values["flash_attention"] = explicit_flash if explicit_flash is not None else {"enable": False}
        self.dflash_config = MiniCPMV47DSparkConfig(**values)

    @property
    def dspark_config(self):
        return self.dflash_config

    def to_dict(self):
        saved = super().to_dict()
        if getattr(self.dflash_config, "is_dspark", False):
            saved["spec_decode_mode"] = "dspark" if self.spec_decode_mode else None
            saved["dspark_config"] = saved.pop("dflash_config")
        return saved

    @classmethod
    def from_dict(cls, config_dict: dict):
        raw = dict(config_dict)
        dflash = raw.get("dflash_config")
        if isinstance(dflash, Mapping):
            raw["dflash_config"] = dict(dflash)
        return cls(**raw)


class MiniCPMV47MoeModelConfig(MiniCPMV47ModelConfig, XHQwen3_5MoeModelConfig):
    """Preserve both MiniCPM DSpark overrides and Qwen MoE pruning validation."""


# Read compatibility for code importing the old MiniCPM class name.
MiniCPMV47DFlashConfig = MiniCPMV47DSparkConfig
