"""Check the VLA source rather than accepting a generic Qwen2.5-VL checkpoint."""

import json

from xhmodel_merak.errors import invalid_parameter


def resolve_config(*, recipe, model_dir, **_):
    try:
        cfg = json.loads((model_dir / "config.json").read_text())
        from xhmodel_merak.xh_other_model.models.wall_oss.config import validate_source_config

        validate_source_config(cfg)
        for name in ("model.safetensors", "config.yml", "tokenizer.json"):
            if not (model_dir / name).is_file():
                raise ValueError(f"missing {name}")
        if not any((model_dir / name).is_file() for name in ("preprocessor_config.json", "processor_config.json")):
            raise ValueError("missing processor configuration")
    except (ValueError, OSError) as error:
        raise invalid_parameter(recipe.model_id, "model_dir", str(model_dir), str(error)) from error
    return {}
