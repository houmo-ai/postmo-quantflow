# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Reduced native Xing checkpoint from bundled config/code, without weights."""

import json
import os
from dataclasses import replace
from pathlib import Path

from _unified_preparation import prepare_context
from _unified_small_model import read_spec


def _configure_calibration(calibration, monkeypatch):
    import xhmodel_merak.model_build.api as api

    original = api.resolve_build_plan

    def resolve(**kwargs):
        plan = original(**kwargs)
        replacements = dict(plan.config_replacements)
        replacements.update({
            "quant.calibration.jsonl": str(calibration),
            "quant.calibration.nsamples": 2,
            "quant.calibration.seqlen": 64,
            "quant.runtime.batch_size": 1,
        })
        return replace(plan, config_replacements=replacements)

    monkeypatch.setattr(api, "resolve_build_plan", resolve)


def prepare(case, root, monkeypatch):
    import sentencepiece as spm
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from gptqmodel.models.definitions.xing4_0_mla import (
        patch_loaded_xing_classes,
        prepare_xing_quantization_dtype,
    )

    native_path = os.environ.get("XING4_0_MODEL_DIR") or (
        Path(__file__).resolve().parents[1] / "fixtures/unified/xing_native"
    )
    native_path = Path(native_path).resolve()
    for name in ("config.json", "configuration_xing4_0.py", "modeling_xing4_0.py", "tokenization_xing4_0.py"):
        if not (native_path / name).is_file():
            raise FileNotFoundError(native_path / name)
    catalog, recipe, source, calibration, _, kwargs, inputs, _, _ = prepare_context(case, root)
    spec = read_spec("xing")
    config = AutoConfig.from_pretrained(str(native_path), trust_remote_code=True)
    config.update(spec["config"])
    source.mkdir()
    get_class_from_dynamic_module("modeling_xing4_0.Xing4_0ForCausalLM", str(native_path))
    patch_loaded_xing_classes()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4891)
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, attn_implementation="eager", dtype=torch.float32
        ).eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "norm" in name and parameter.ndim == 1 and parameter.numel() > 1:
                    parameter.add_(torch.linspace(-0.02, 0.02, parameter.numel()))
    prepare_xing_quantization_dtype(model, torch.float16)
    model.save_pretrained(source)
    # Match the official source representation: head_dim is derived by native code.
    config_path = source / "config.json"
    source_config = json.loads(config_path.read_text())
    source_config.pop("head_dim", None)
    config_path.write_text(json.dumps(source_config))
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(["hello world test calibration " + " ".join(str(i) for i in range(256))]),
        model_prefix=str(source / "tokenizer"),
        vocab_size=512,
        hard_vocab_limit=False,
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        num_threads=1,
        shuffle_input_sentence=False,
    )
    tokenizer_cls = get_class_from_dynamic_module("tokenization_xing4_0.Xing4_0Tokenizer", str(native_path))
    tokenizer = tokenizer_cls(
        vocab_file=str(source / "tokenizer.model"), bos_token="<s>", eos_token="</s>", pad_token="<pad>"
    )
    tokenizer.add_tokens([f"token_{i}" for i in range(len(tokenizer), config.vocab_size)])
    tokenizer.register_for_auto_class("AutoTokenizer")
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    tokenizer.save_pretrained(source)
    (source / "tokenizer.vocab").unlink()
    kwargs.update(context_length=64, prefill_chunk_length=8, quant_output_dir=str(root / "quant"))
    monkeypatch.setitem(
        catalog._recipes,
        case.model_id,
        replace(recipe, source_constraints=(("model_type", "xing4_0"),), max_pe_length=128),
    )
    _configure_calibration(calibration, monkeypatch)
    return kwargs, inputs


def validate_build(case, result, record):
    from xhmodel_merak.model_artifact import MANIFEST_NAME

    assert result.quant_artifact is not None
    config = json.loads((Path(result.quant_artifact) / "config.json").read_text())
    assert config["mla_weight_layout"] == "xh2_absorbed_v1"
    assert config["mla_transform"]["version"] == 1
    assert config["mla_transform"]["cache_layout"] == "separate_k_latent_rope_v_latent"
    assert "head_dim" not in config
    manifest = json.loads((Path(result.output_dir) / MANIFEST_NAME).read_text())
    regions = manifest["artifact"]["checkpoint_precision_regions"]
    assert {name: entry["weight_bits"] for name, entry in regions.items()} == {
        "attention": 8, "dense_mlp": 8, "shared_expert": 8, "routed_expert": 4,
    }
    record["mla_transform"] = config["mla_transform"]
