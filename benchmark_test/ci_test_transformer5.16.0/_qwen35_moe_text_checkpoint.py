# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Native Qwen3.5 MoE text checkpoint for public model integration tests."""

from _unified_small_model import read_spec, write_processor


def write_native_moe_text_checkpoint(root):
    """Save the native text architecture with both attention types and experts."""
    import torch
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    source = root / "source"
    spec = read_spec("qwen_moe")
    config = Qwen3_5MoeTextConfig(**spec["config"]["text_config"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4891)
        model = Qwen3_5MoeForCausalLM(config).to(torch.float16).eval()
    model.save_pretrained(source, safe_serialization=True)
    write_processor(source, {**spec, "processor_files": {}})
    return source, model
