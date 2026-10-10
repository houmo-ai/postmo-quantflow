from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_spark import Spark2_5Config
from .modeling_spark import Spark2_5ForCausalLM, Spark2_5Model


def register_spark2_5() -> None:
    AutoConfig.register(Spark2_5Config.model_type, Spark2_5Config, exist_ok=True)
    AutoModel.register(Spark2_5Config, Spark2_5Model, exist_ok=True)
    AutoModelForCausalLM.register(Spark2_5Config, Spark2_5ForCausalLM, exist_ok=True)


__all__ = [
    "Spark2_5Config",
    "Spark2_5Model",
    "Spark2_5ForCausalLM",
    "register_spark2_5",
]
