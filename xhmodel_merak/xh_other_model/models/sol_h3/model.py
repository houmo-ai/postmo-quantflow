# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
from ...builder import register_other_model


@register_other_model("XHSolH3Model")
class XHSolH3Model:
    """Four-update H3 draft, learned latent bridge and three-update LTX refiner."""

    WORKFLOW_CLS = "xhmodel_merak.xh_other_model.models.sol_h3.workflow:SolH3Workflow"


@register_other_model("XHNativeH3Model")
class XHNativeH3Model:
    """Native H3 stages with shared Sol-H3 graph/runtime components."""

    WORKFLOW_CLS = "xhmodel_merak.xh_other_model.models.sol_h3.native_workflow:NativeH3Workflow"
