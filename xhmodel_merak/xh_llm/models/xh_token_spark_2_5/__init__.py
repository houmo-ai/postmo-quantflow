from .spark2_5_hmonnx_inference import XHSpark2_5HMONNXModel
from .spark2_5_model import XHSpark2_5Model, XHSpark2_5ModelConfig


def __getattr__(name: str):
    if name in ("Spark2_5QModel", "register_spark2_5_gptqmodel"):
        from .gptqmodel_wrap import Spark2_5QModel, register_spark2_5_gptqmodel

        return Spark2_5QModel if name == "Spark2_5QModel" else register_spark2_5_gptqmodel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Spark2_5QModel",
    "register_spark2_5_gptqmodel",
    "XHSpark2_5Model",
    "XHSpark2_5ModelConfig",
    "XHSpark2_5HMONNXModel",
]
