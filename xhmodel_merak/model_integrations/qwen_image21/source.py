"""Check Diffusers source identity without importing Torch or reading weights."""

import json

from xhmodel_merak.errors import invalid_parameter


def resolve_config(*, recipe, model_dir, **_):
    from xhmodel_merak.xh_other_model.models.qwen_image21.sampling import read_source_sampling

    expected = {
        "transformer/config.json": ("_class_name", "QwenImage21Transformer2DModel"),
        "vae/config.json": ("_class_name", "AutoencoderKLQwenImage21"),
        "text_encoder/config.json": ("model_type", "qwen3_vl"),
        "scheduler/scheduler_config.json": ("_class_name", "FlowMatchEulerDiscreteScheduler"),
    }
    for relative, (key, value) in expected.items():
        try:
            config = json.loads((model_dir / relative).read_text())
            if config.get(key) != value:
                raise ValueError(f"requires {key}={value}")
        except (OSError, ValueError, AttributeError) as error:
            raise invalid_parameter(
                recipe.model_id, "model_dir", str(model_dir), f"invalid {relative}: {error}"
            ) from error
    try:
        read_source_sampling(model_dir, model_id=recipe.model_id)
    except (OSError, ValueError, TypeError) as error:
        raise invalid_parameter(recipe.model_id, "model_dir", str(model_dir), str(error)) from error
    return {}
