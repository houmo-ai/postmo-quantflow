"""Read MiniCPM DSpark checkpoints without changing the Qwen3.5 DFlash loader."""

from __future__ import annotations

import json
from pathlib import Path


def read_draft_checkpoint(model_dir) -> dict:
    path = Path(model_dir) / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    architectures = config.get("architectures") or []
    if architectures == ["Qwen3DSparkModel"]:
        if config.get("markov_head_type") != "vanilla":
            raise ValueError("MiniCPM DSpark requires markov_head_type=vanilla")
        draft_vocab = config.get("draft_vocab_size", config["vocab_size"])
        if int(draft_vocab) != int(config["vocab_size"]):
            raise ValueError("MiniCPM DSpark draft vocab must match the target vocab")
        if config.get("attention_mode", "gqa") != "gqa":
            raise ValueError("MiniCPM DSpark requires attention_mode=gqa")
        rank = config.get("markov_rank")
        if type(rank) is not int or rank <= 0:
            raise ValueError("MiniCPM DSpark markov_rank must be a positive integer")
        target_layer_ids = list(config["target_layer_ids"])
        config = dict(config)
        config["dflash_config"] = {
            "target_layer_ids": target_layer_ids,
            "mask_token_id": config["mask_token_id"],
            "block_size": config["block_size"],
        }
    return config


def validate_draft_target(draft_dir, target_dir) -> dict:
    draft = read_draft_checkpoint(draft_dir)
    target = json.loads((Path(target_dir) / "config.json").read_text(encoding="utf-8"))
    text = target.get("text_config") or target
    if int(draft["hidden_size"]) != int(text["hidden_size"]):
        raise ValueError("DSpark hidden_size does not match the MiniCPM target")
    if int(draft["vocab_size"]) != int(text["vocab_size"]):
        raise ValueError("DSpark vocab_size does not match the MiniCPM target")
    ids = [int(item) for item in draft["dflash_config"]["target_layer_ids"]]
    if ids != sorted(set(ids)) or ids[-1] >= int(text["num_hidden_layers"]):
        raise ValueError("DSpark target_layer_ids must be strictly increasing and in range")
    if int(draft.get("num_target_layers", text["num_hidden_layers"])) != int(text["num_hidden_layers"]):
        raise ValueError("DSpark num_target_layers does not match target_layer_ids")
    if "mask_token_id" not in draft["dflash_config"]:
        raise ValueError("DSpark checkpoint is missing mask_token_id")
    return draft
