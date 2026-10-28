"""Resolve native and W4 builds from the maintained activation workflows."""

from copy import deepcopy

import yaml

from xhmodel_merak.model_catalog.resources import resolve_workflow_config
from xhmodel_merak.model_integrations.autoregressive.vision import validate_visual_workflow


W4_WORKFLOW = (
    "configs_merak/workflows/xh2a/llm_models/qwen3_vl_reranker/2b/qwen3_vl_reranker_2b_xh2a_w4a8_a16_nodes2.yaml"
)
W4_A16_WORKFLOW = "configs_merak/workflows/xh2a/llm_models/qwen3_vl_reranker/2b/qwen3_vl_reranker_2b_xh2a_w4a16.yaml"


def resolve_config(*, recipe, variant, model_dir, sources, workflow, quant_checkpoint, precision):
    del variant, model_dir, sources, quant_checkpoint
    _validate_visual_workflow(recipe, workflow)
    if precision.weight != "w4":
        # The shared source YAML carries W4 calibration and its two sensitive
        # language nodes. Native input keeps the established base A8 policy.
        scheme = deepcopy(workflow["export"]["model"]["quant_scheme"])
        for name in ("model_language_model_layers_26_mlp_up_proj", "model_language_model_layers_27_mlp_gate_proj"):
            scheme["nodes"].pop(name, None)
        return {"quant": None, "export.model.quant_scheme": scheme}
    config_path = W4_A16_WORKFLOW if precision.activation == "a16" else W4_WORKFLOW
    config = yaml.safe_load(resolve_workflow_config(config_path).read_text())
    _validate_visual_workflow(recipe, config)
    return {"quant": config["quant"], "export.model.quant_scheme": config["export"]["model"]["quant_scheme"]}


def _validate_visual_workflow(recipe, workflow):
    # Installed Recipe replacements define the effective geometry (including
    # the real tiny-model CI profile), and are applied by the build executor.
    effective = deepcopy(workflow)
    for path, value in recipe.config_replacements:
        if path.startswith("export.model.visual_config."):
            cursor = effective
            parts = path.split(".")
            for part in parts[:-1]:
                cursor = cursor.setdefault(part, {})
            cursor[parts[-1]] = value
    validate_visual_workflow(recipe, effective)
