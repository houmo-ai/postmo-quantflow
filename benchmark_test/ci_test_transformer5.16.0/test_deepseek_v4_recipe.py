import json

import pytest

from xhmodel_merak.model_catalog.registry import get_recipe
from xhmodel_merak.model_build.resolver import resolve_build_plan
from xhmodel_merak.model_integrations.deepseek_v4.planning import resolve_config
from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig


def _checkpoint_config(root):
    from transformers import DeepseekV4Config

    config = DeepseekV4Config()
    config.architectures = ["DeepseekV4ForCausalLM"]
    config.quantization_config = {"quant_method": "gptq", "bits": 8, "group_size": 64,
                                  "sym": True, "desc_act": False}
    config.save_pretrained(root)
    # Planning reads configuration only. Numerical tests use real model blocks
    # in test_deepseek_v4_dspark.py rather than allocating a 43-block fixture.
    (root / "dspark").mkdir()
    (root / "dspark" / "dspark_meta_info.json").write_text(json.dumps({"format_version": 1}))


def test_flash_recipe_discovers_both_workflows_and_internal_adapters():
    from xhmodel_merak.model_integrations.deepseek_v4 import build, inference

    recipe = get_recipe("deepseek-v4-flash-0731")
    assert [variant.name for variant in recipe.variants] == ["full", "dspark"]
    assert recipe.get_variant("dspark").speculative_mode == "dspark"
    assert len(recipe.get_variant("dspark").extra_components) == 4
    assert callable(build.validate_artifact) and callable(inference.adapter.run)
    config = WorkflowConfig.from_file(recipe.get_variant("dspark").workflow_config)
    assert config.export["model"]["dspark"]["head_weight_bits"] == 5
    assert config.export["model"]["num_draft_tokens"] == 5


def test_flash_dspark_public_plan_preserves_checkpoint_precision_and_resolves_graphs(tmp_path):
    source = tmp_path / "source"
    _checkpoint_config(source)
    plan = resolve_build_plan(model_id="deepseek-v4-flash-0731", model_dir=str(source),
                              output_dir=str(tmp_path / "output"),
                              speculative={"mode": "dspark", "num_draft_tokens": 7},
                              deployment_precision={"prefill": "w8a16h1_sefp", "decode": "w8a16h1_sefp"})
    assert plan.variant == "dspark"
    assert plan.config_overrides["export.model.num_draft_tokens"] == 7
    assert set(plan.components) == {"prefill", "decode", "dspark_context", "dspark_context_decode",
                                    "dspark_draft_decode", "dspark_markov"}


def test_flash_dspark_plan_rejects_missing_bundled_weights(tmp_path):
    _checkpoint_config(tmp_path)
    (tmp_path / "dspark" / "dspark_meta_info.json").unlink()
    recipe = get_recipe("deepseek-v4-flash-0731")
    with pytest.raises(ValueError, match="producer-preserved"):
        resolve_config(recipe=recipe, variant=recipe.get_variant("dspark"), model_dir=tmp_path,
                       sources={}, workflow=None)


def test_flash_dspark_rejects_draft_features_from_different_target_layers(tmp_path):
    from torch import nn
    from xhmodel_merak.xh_llm.models.deepseek_v4.dspark_checkpoint import load_dspark_checkpoint
    from xhmodel_merak.model_integrations.deepseek_v4.build import validate_dspark
    from types import SimpleNamespace

    (tmp_path / "dspark_bf16_preparation.json").write_text(json.dumps({"target_layer_ids": [40, 41, 42]}))
    # Reject before constructing the large draft or loading its tensors.
    with pytest.raises(ValueError, match="feature layers"):
        load_dspark_checkpoint(tmp_path, nn.Linear(4, 8), target_layer_ids=(39, 40, 41))
    metadata = {"dspark": {"components": dict.fromkeys(
        ("dspark_context", "dspark_context_decode", "dspark_draft_decode", "dspark_markov"), {}),
        "spec": {"max_context_length": 262144, "target_layer_ids": [40, 41, 42]}},
        "model_config": {"context_max_length": 262144}, "deepseek_v4_auxiliary_layer_ids": [39, 40, 41]}
    with pytest.raises(ValueError, match="feature layers"):
        validate_dspark(tmp_path, metadata, SimpleNamespace(components=("dspark_draft_decode",)))


def test_flash_package_preserves_official_thinking_template_after_relocation(tmp_path):
    from types import SimpleNamespace
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    from xhmodel_merak.model_integrations.deepseek_v4.build import package_runtime

    hf = tmp_path / "stage" / "hf_config"
    vocabulary = {"[UNK]": 0, "<｜begin▁of▁sentence｜>": 1, "<｜end▁of▁sentence｜>": 2}
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(vocabulary, unk_token="[UNK]")),
        unk_token="[UNK]", bos_token="<｜begin▁of▁sentence｜>", eos_token="<｜end▁of▁sentence｜>")
    tokenizer.save_pretrained(hf)
    package_runtime(hf.parent, SimpleNamespace(quant_checkpoint=None))
    destination = tmp_path / "moved"
    hf.parent.rename(destination)
    tokenizer = AutoTokenizer.from_pretrained(destination / "hf_config", local_files_only=True)
    messages = [{"role": "user", "content": "17×3=?"}]
    thinking = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=True, reasoning_effort="max")
    assert thinking.startswith("<｜begin▁of▁sentence｜>Reasoning Effort: Beyond maximum")
    assert thinking.endswith("<｜User｜>17×3=?<｜Assistant｜><think>")
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    assert chat == "<｜begin▁of▁sentence｜><｜User｜>17×3=?<｜Assistant｜></think>"
    assert tokenizer.encode(thinking).count(tokenizer.bos_token_id) == 1
    assert "MIT License" in tokenizer.chat_template
    with pytest.raises(Exception, match="one user string"):
        tokenizer.apply_chat_template([{"role": "system", "content": "x"}], tokenize=False)


def test_flash_public_answer_separates_reasoning_and_reports_truncation():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from xhmodel_merak.xh_llm.models.deepseek_v4.chat import decode_flash_thinking_completion

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
        {"[UNK]": 0, "reason": 1, "</think>": 2, "51": 3, "<eos>": 4}, unk_token="[UNK]")),
        unk_token="[UNK]", eos_token="<eos>", additional_special_tokens=["</think>"])
    answer, stats = decode_flash_thinking_completion(tokenizer, [1, 2, 3, 4])
    assert answer == "51"
    assert stats == {"reasoning_tokens": 2, "thinking_complete": True, "reasoning_text": "reason"}
    answer, stats = decode_flash_thinking_completion(tokenizer, [1])
    assert answer == "" and stats["thinking_complete"] is False and stats["reasoning_tokens"] == 1


def test_flash_checkpoint_verifies_auxiliary_weights_without_hiding_unknown_files(tmp_path):
    from types import SimpleNamespace
    import torch
    from safetensors.torch import save_file
    from _deepseek_v4_small_fixture import write_deepseek_v4_checkpoint
    from xhmodel_merak.errors import ModelError
    from xhmodel_merak.model_build.checkpoint import _inspect_quant_weight_files

    source = tmp_path / "source"
    write_deepseek_v4_checkpoint(source)
    plan = SimpleNamespace(model_id="deepseek-v4-flash-0731", model_dir=str(source),
                           build_adapter="deepseek-v4.build")
    result = _inspect_quant_weight_files(source, plan)
    assert [entry["file"] for entry in result["weights"]] == ["model.safetensors"]
    assert result["auxiliary_components"]["dspark"]["num_draft_layers"] == 3
    save_file({"unexpected.weight": torch.ones(2, 2)}, source / "dspark" / "unexpected.safetensors")
    with pytest.raises(ModelError, match="requires one authoritative index"):
        _inspect_quant_weight_files(source, plan)
