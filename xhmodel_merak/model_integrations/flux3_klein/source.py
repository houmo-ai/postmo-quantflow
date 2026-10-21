"""Source checks run before expensive checkpoint hashing and model construction."""

from xhmodel_merak.errors import invalid_parameter


def resolve_config(*, recipe, model_dir, sources=None, **_):
    from xhmodel_merak.xh_other_model.models.flux3_klein.source import read_source

    try:
        read_source(model_dir)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise invalid_parameter(recipe.model_id, "model_dir", str(model_dir), str(error)) from error
    sources = sources or {}
    if "prompt_upsampler" in sources and "upsampler_bundle" in sources:
        raise invalid_parameter(
            recipe.model_id, "sources", list(sources), "choose raw prompt_upsampler or upsampler_bundle"
        )
    replacements = {}
    for name, source in sources.items():
        replacements[f"export.{name}"] = source.path
    if "prompt_upsampler" in sources:
        from xhmodel_merak.xh_other_model.models.flux3_klein.upsampler_build import validate_source

        try:
            validate_source(sources["prompt_upsampler"].path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise invalid_parameter(
                recipe.model_id, "sources.prompt_upsampler", sources["prompt_upsampler"].path, str(error)
            ) from error
    return replacements
