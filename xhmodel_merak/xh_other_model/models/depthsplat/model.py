# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
from ...builder import register_other_model


@register_other_model("XHDepthSplatModel")
class XHDepthSplatModel:
    WORKFLOW_CLS = "xhmodel_merak.xh_other_model.models.depthsplat.workflow:DepthSplatWorkflow"
