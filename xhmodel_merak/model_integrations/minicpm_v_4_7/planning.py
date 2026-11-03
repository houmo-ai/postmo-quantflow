"""Bind and validate MiniCPM's paired draft without importing model runtimes."""

from xhmodel_merak.xh_llm.models.minicpm_v_4_7.draft_checkpoint import validate_draft_target
from xhmodel_merak.errors import invalid_parameter


def resolve_config(*, recipe, variant, model_dir, sources, workflow, quant_checkpoint=None, precision=None):
    if variant.speculative_mode != "dspark":
        return {}
    source = sources["dspark"]
    try:
        draft = validate_draft_target(source.path, model_dir)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise invalid_parameter(recipe.model_id, "sources.dspark", source.path, str(error)) from error
    return {
        "export.model.dspark_config.hf_model": source.path,
        "export.model.dspark_config.noise_token_id": draft["dflash_config"]["mask_token_id"],
    }
