# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Lightweight native H3 source selection; no LTX or author code dependency."""

from xhmodel_merak.errors import invalid_parameter


def resolve_config(*, recipe, model_dir, precision=None, **_):
    from xhmodel_merak.model_integrations.sol_h3.source import validate_checkpoint_source_config
    from xhmodel_merak.xh_other_model.models.sol_h3.native_assets import NativeSourceAssets

    try:
        assets = NativeSourceAssets.read(model_dir)
        if precision is not None and precision.weight != "w16":
            validate_checkpoint_source_config(assets)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise invalid_parameter(recipe.model_id, "model_dir", str(model_dir), str(error)) from error
    return {"export.conditioner_quant_type": None} if precision is not None and precision.weight == "w16" else {}
