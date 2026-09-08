"""Repository-local quantization recipes."""

from .paddleocr_vl import (
    PaddleOCRVLCombinedQModel,
    PaddleOCRVLQModel,
    PaddleOCRVLQuantRecipeResult,
    PaddleOCRVLVisionQModel,
    quantize_paddleocr_vl,
    register_paddleocr_vl_gptqmodel,
)


__all__ = [
    "PaddleOCRVLCombinedQModel",
    "PaddleOCRVLQModel",
    "PaddleOCRVLQuantRecipeResult",
    "PaddleOCRVLVisionQModel",
    "quantize_paddleocr_vl",
    "register_paddleocr_vl_gptqmodel",
]
