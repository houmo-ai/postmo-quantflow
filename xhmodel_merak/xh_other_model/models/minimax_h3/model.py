from xhmodel_merak.xh_other_model.builder import register_other_model


@register_other_model("XHMiniMaxH3Model")
class XHMiniMaxH3Model:
    """MiniMax-H3 multi-component export workflow binding model."""

    WORKFLOW_CLS = "xhmodel_merak.xh_other_model.models.minimax_h3.workflow:MiniMaxH3Workflow"


__all__ = ["XHMiniMaxH3Model"]
