# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Use actual small Qwen3-VL weights and processors, with no model/runtime mocks."""

from dataclasses import replace

from _unified_preparation import prepare_context


def prepare(case, root, monkeypatch):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import (
        PreTrainedTokenizerFast,
        Qwen2VLImageProcessor,
        Qwen3VLConfig,
        Qwen3VLForConditionalGeneration,
        Qwen3VLProcessor,
        Qwen3VLVideoProcessor,
    )

    catalog, recipe, source, calibration, image, kwargs, inputs, replacements, family = prepare_context(case, root)
    source.mkdir()
    torch.manual_seed(23)
    config = Qwen3VLConfig(
        text_config=dict(
            vocab_size=64,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=128,
            max_position_embeddings=256,
            rope_parameters={"rope_type": "default", "rope_theta": 5000000.0, "mrope_section": [24, 20, 20]},
            eos_token_id=2,
        ),
        vision_config=dict(
            depth=3,
            hidden_size=64,
            intermediate_size=128,
            num_heads=4,
            out_hidden_size=256,
            patch_size=16,
            temporal_patch_size=2,
            spatial_merge_size=2,
            deepstack_visual_indexes=[0, 1, 2],
        ),
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )
    Qwen3VLForConditionalGeneration(config).eval().save_pretrained(source)
    vocab = {
        "[UNK]": 0,
        "[PAD]": 1,
        "<|im_end|>": 2,
        "<|im_start|>": 3,
        "yes": 4,
        "no": 5,
        "hello": 6,
        "world": 7,
        "test": 8,
        "calibration": 9,
        "\n": 10,
        "assistant": 11,
        "<|image_pad|>": 60,
        "<|video_pad|>": 61,
        "<|vision_start|>": 62,
        "<|vision_end|>": 63,
    }
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    # Keep the real scoring anchor: WhitespaceSplit silently discarded the
    # newline at the end of the assistant template in this tiny fixture.
    backend.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Split("\n", behavior="isolated"), pre_tokenizers.Split(" ", behavior="removed")]
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<|im_end|>",
        additional_special_tokens=[
            "<|im_start|>",
            "<|image_pad|>",
            "<|video_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
        ],
    )
    assert tokenizer.encode("<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False) == [2, 10, 3, 11, 10]
    template = (
        "{% for message in messages %}<|im_start|>{{ message['role'] }}\n"
        "{% for item in message['content'] %}{% if item['type'] == 'image' %}"
        "<|vision_start|><|image_pad|><|vision_end|>{% else %}{{ item['text'] }}{% endif %}{% endfor %}"
        "<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    processor = Qwen3VLProcessor(
        tokenizer=tokenizer,
        chat_template=template,
        image_processor=Qwen2VLImageProcessor(
            patch_size=16, temporal_patch_size=2, merge_size=2, image_mean=[0.5] * 3, image_std=[0.5] * 3
        ),
        video_processor=Qwen3VLVideoProcessor(patch_size=16, temporal_patch_size=2, merge_size=2),
    )
    processor.save_pretrained(source)
    monkeypatch.setitem(
        catalog._recipes,
        case.model_id,
        replace(
            recipe,
            source_constraints=(("model_type", "qwen3_vl"),),
            max_pe_length=256,
            vision_gears=(4, 8),
            config_replacements=(
                ("export.model.visual_config.image_token_gears", [4, 8]),
                ("export.model.visual_config.image_token_capacity", 8),
            ),
        ),
    )
    kwargs.update(context_length=256, prefill_chunk_length=256)
    return kwargs, {
        "query": {"text": "Find the image.", "image": str(image)},
        "documents": [{"text": "hello", "image": str(image)}, {"text": "world"}],
    }


def validate_output(case, output, result, moved, inputs, root, record):
    import math
    from pathlib import Path

    from xhmodel_merak.model_inference import inference_model
    from xhmodel_merak.xh_llm.models.qwen3_vl_reranker.workflow import dump_reranker_golden

    assert len(output.scores) == 2 and all(math.isfinite(value) and 0 <= value <= 1 for value in output.scores)
    assert set(output.ranking) == {0, 1}
    golden = dump_reranker_golden(
        output.meta_file, "cuda:0", {"query": inputs["query"], "documents": inputs["documents"][:1]}
    )
    assert all(any((Path(path) / "step_0").glob("*.npy")) for path in golden["golden_dirs"])
    assert not any(path.is_symlink() for path in moved.rglob("*"))
    record["golden_components"] = ["visual", "prefill"]
    repeated = inference_model(artifact_dir=str(moved), inputs=inputs, device="cuda:0")
    assert output.scores == repeated.scores
    swapped = inference_model(
        artifact_dir=str(moved), inputs={**inputs, "documents": inputs["documents"][::-1]}, device="cuda:0"
    )
    assert output.scores == swapped.scores[::-1]
    record["repeat_bit_exact"] = True
