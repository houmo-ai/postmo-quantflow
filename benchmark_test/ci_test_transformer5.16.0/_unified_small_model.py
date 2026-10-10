# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Offline, deterministic native checkpoints for public API integration tests.

Only model geometry and processor settings are stored in the repository. Each
test creates fresh weights; no /tmp probe, dataset snapshot or pretrained model
is used by these factories. Two attention kinds and routed/shared experts are
retained where the architecture needs them.
"""

from __future__ import annotations

import json
from pathlib import Path


FIXTURES = Path(__file__).with_name("fixtures") / "unified"


def read_spec(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def write_processor(source: Path, spec: dict):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    settings = spec["tokenizer"]
    vocab = dict(settings["vocab"])
    for index in range(max(vocab.values()) + 1, 512):
        vocab[f"token_{index}"] = index
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    special = dict(settings["special_tokens"])
    special["unk_token"] = "[UNK]"
    special.setdefault("pad_token", "[UNK]")
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, **special)
    # Exercise real processor dispatch for text and typed multimodal messages.
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{% if message['content'] is string %}{{ message['content'] }}"
        "{% else %}{% for item in message['content'] %}"
        "{% if item['type'] == 'image' %}{{ image_token }}"
        "{% elif item['type'] == 'text' %}{{ item['text'] }}{% endif %}"
        "{% endfor %}{% endif %} {% endfor %}"
    )
    tokenizer.save_pretrained(source)
    for filename, value in spec["processor_files"].items():
        (source / filename).write_text(json.dumps(value))
    return tokenizer


def write_native_checkpoint(source: Path, family: str):
    import torch
    import transformers

    spec = read_spec(family)
    config = spec["config"]
    source.mkdir(parents=True)
    config_cls = transformers.AutoConfig.for_model
    raw = dict(config)
    model_type = raw.pop("model_type")
    if family == "spark":
        from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.spark_2_5 import Spark2_5ForCausalLM

        model_cls = Spark2_5ForCausalLM
    config = config_cls(model_type, **raw)
    if family != "spark":
        model_cls = getattr(transformers, spec["config"]["architectures"][0])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4891)
        model = model_cls(config).eval()
        # Trained norm vectors are not identical constants. Keep their channels
        # distinguishable so ONNX constant folding does not erase the norm ABI.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "norm" in name and parameter.ndim == 1 and parameter.numel() > 1:
                    parameter.add_(torch.linspace(-0.02, 0.02, parameter.numel()))
    model.to(torch.bfloat16).save_pretrained(source, safe_serialization=True)
    write_processor(source, spec)
    return spec["config"]


def write_auxiliary(source: Path, name: str, *, target_config=None):
    """Make native-format MTP/LoRA/draft tensors from audited small shapes."""
    import torch
    from safetensors.torch import load_file, save_file

    spec = read_spec(name)
    generator = torch.Generator().manual_seed(4891)
    tensors = {}
    for key, shape in spec["tensor_shapes"].items():
        tensor = torch.randn(shape, generator=generator) * 0.02
        if "norm" in key and len(shape) == 1:
            tensor.fill_(1)
        tensors[key] = tensor.to(torch.bfloat16).contiguous()
    if name.endswith("_mtp"):
        path = source / "model.safetensors"
        weights = load_file(path)
        weights.update(tensors)
        save_file(weights, path)
        config_path = source / "config.json"
        config = json.loads(config_path.read_text())
        config.get("text_config", config)["mtp_num_hidden_layers"] = 1
        config_path.write_text(json.dumps(config))
    elif name == "qwen_lora":
        source.mkdir(parents=True)
        (source / "adapter_config.json").write_text(json.dumps(spec["config"]))
        save_file(tensors, source / "adapter_model.safetensors")
    else:
        source.mkdir(parents=True)
        config = spec["config"]
        config.update(vocab_size=512, eos_token_id=target_config["text_config"]["eos_token_id"])
        config.pop("auto_map", None)
        config["pad_token_id"] = config["eos_token_id"]
        (source / "config.json").write_text(json.dumps(config))
        save_file(tensors, source / "model.safetensors")


def write_calibration(root: Path):
    """Two calibration rows plus one held-out MTP row; never fetch a dataset."""
    from PIL import Image

    path = root / "calibration.jsonl"
    image = root / "image.png"
    Image.new("RGB", (56, 56), "red").save(image)
    path.write_text("".join(json.dumps({
        "text": "hello world test calibration " * 128,
        "image": str(image), "prompt": "OCR:", "answer": "42",
    }) + "\n" for _ in range(3)))
    return path, image
