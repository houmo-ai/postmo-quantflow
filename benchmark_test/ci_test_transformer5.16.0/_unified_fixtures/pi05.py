# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
from dataclasses import replace

from _unified_preparation import prepare_context


def prepare(case, root, monkeypatch):
    catalog, recipe, source, calibration, image, kwargs, inputs, replacements, family = prepare_context(case, root)
    from _pi05_small_fixture import write_pi05

    tokenizer = root / "tokenizer"
    write_pi05(source, tokenizer, monkeypatch)
    monkeypatch.setitem(
        catalog._recipes,
        recipe.model_id,
        replace(
            recipe,
            config_replacements=(*recipe.config_replacements, ("export.other.hidden_size", 64)),
        ),
    )
    kwargs["resources"] = {"tokenizer": str(tokenizer)}
    inputs = {
        "task": "pick up the object",
        "observation.images.image": str(image),
        "observation.images.image2": str(image),
        "observation.state": [0.0] * 8,
        "seed": 1,
    }
    return kwargs, inputs
