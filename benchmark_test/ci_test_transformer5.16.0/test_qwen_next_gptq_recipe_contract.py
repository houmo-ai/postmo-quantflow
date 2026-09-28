"""Exercise the registered Next Workflows against the pinned producer API.

Only the producer's real dry-run path is used. A tiny safetensors fixture
exercises MTP detection without running calibration or quantization.
"""

import json
from copy import deepcopy
from inspect import signature

import pytest
import torch
import yaml
from safetensors.torch import save_file

from xhmodel_merak.model_catalog.registry import get_recipe
from xhmodel_merak.model_catalog.resources import resolve_workflow_config


@pytest.mark.parametrize("variant", ["full", "mtp"])
def test_registered_qwen_next_workflow_matches_gptq_58_producer(tmp_path, variant):
    from gptqmodel.recipes.qwen3_next import quantize_qwen3_next

    from xhmodel_merak.xh_llm.models.qwen3_next.workflow import Qwen3NextWorkflow

    recipe = get_recipe("qwen3-next-80b-a3b")
    config = yaml.safe_load(resolve_workflow_config(recipe.get_variant(variant).workflow_config).read_text())
    quant = deepcopy(config["quant"])
    quant["runtime"]["dry_run"] = True
    workflow = Qwen3NextWorkflow.__new__(Qwen3NextWorkflow)
    source = tmp_path / "Qwen3-Next-80B-A3B-Instruct"
    source.mkdir()
    save_file({"mtp.norm.weight": torch.ones(4)}, str(source / "model.safetensors"))
    workflow.model_dir = str(source)
    result = workflow._quant_gptqmodel_api(str(tmp_path / "quant"), "cpu", quant, config["export"]["model"])

    producer_signature = signature(quantize_qwen3_next)
    parameters = set(producer_signature.parameters)
    assert {"method", "calibration_jsonl", "dry_run"} <= parameters
    # Validate the GPTQ path we consume, not the absence of optional arguments
    # added by the producer for other methods. Its real dry-run below remains
    # the contract check, including the emitted GPTQ command.
    assert producer_signature.parameters["method"].default == "gptq"
    assert result.meta.method == "gptq"
    assert result.meta.model_family == "qwen3_next"
    assert result.meta.has_mtp is True
    provenance = result.meta.provenance
    assert provenance["dry_run"] is True
    assert "--calibration-jsonl" in provenance["command"]
    assert not {"--sym", "--iters", "--dataset"} & set(provenance["command"])


def test_reused_next_target_requires_mtp_in_same_checkpoint(tmp_path):
    from xhmodel_merak.model_integrations.qwen.planning import resolve_config

    recipe = get_recipe("qwen3-next-80b-a3b")
    variant = recipe.get_variant("mtp")
    workflow = yaml.safe_load(resolve_workflow_config(variant.workflow_config).read_text())
    raw, packed = tmp_path / "raw", tmp_path / "packed"
    raw.mkdir()
    packed.mkdir()
    (raw / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"mtp.layers.0.input_layernorm.weight": "raw.safetensors"},
    }))
    (packed / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    with pytest.raises(ValueError, match="cannot combine a potentially rotated target"):
        resolve_config(
            recipe=recipe, variant=variant, model_dir=raw, sources={},
            workflow=workflow, quant_checkpoint=str(packed),
        )
    (packed / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"mtp.layers.0.input_layernorm.weight": "packed.safetensors"},
    }))
    replacements = resolve_config(
        recipe=recipe, variant=variant, model_dir=raw, sources={},
        workflow=workflow, quant_checkpoint=str(packed),
    )
    assert "export.model.mtp_config.hf_model" not in replacements
