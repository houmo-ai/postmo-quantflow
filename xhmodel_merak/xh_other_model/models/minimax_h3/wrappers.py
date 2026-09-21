from xhmodel_merak.xh_other_model.models.minimax_h3 import (
    MiniMaxH3AudioVAEDecoderWrapper,
    MiniMaxH3AudioVAEEncoderWrapper,
    MiniMaxH3TextEncoderWrapper,
    MiniMaxH3TransformerWrapper,
    MiniMaxH3VAEDecoderWrapper,
    MiniMaxH3VAEEncoderWrapper,
)


MiniMaxH3VAEEncoderExportWrapper = MiniMaxH3VAEEncoderWrapper
MiniMaxH3VAEDecoderExportWrapper = MiniMaxH3VAEDecoderWrapper
MiniMaxH3AudioVAEEncoderExportWrapper = MiniMaxH3AudioVAEEncoderWrapper
MiniMaxH3AudioVAEDecoderExportWrapper = MiniMaxH3AudioVAEDecoderWrapper


__all__ = [
    "MiniMaxH3AudioVAEDecoderWrapper",
    "MiniMaxH3AudioVAEEncoderExportWrapper",
    "MiniMaxH3AudioVAEDecoderExportWrapper",
    "MiniMaxH3TextEncoderWrapper",
    "MiniMaxH3TransformerWrapper",
    "MiniMaxH3VAEDecoderExportWrapper",
    "MiniMaxH3VAEEncoderExportWrapper",
]
