# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
from ...builder import register_other_model


@register_other_model("XHQwenImage21Model")
class XHQwenImage21Model:
    """Registry entry for the encoder, cached DiT and RGBA VAE workflow."""

    WORKFLOW_CLS = "xhmodel_merak.xh_other_model.models.qwen_image21.workflow:QwenImage21Workflow"
