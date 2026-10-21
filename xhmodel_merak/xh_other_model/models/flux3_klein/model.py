# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
from ...builder import register_other_model


@register_other_model("XHFlux3KleinModel")
class XHFlux3KleinModel:
    """Complete distilled text-to-image and reference-editing pipeline."""

    WORKFLOW_CLS = "xhmodel_merak.xh_other_model.models.flux3_klein.workflow:Flux3KleinWorkflow"
