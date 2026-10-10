"""Register the Spark X2.5 checkpoint layout with GPTQModel for loading."""

from typing import Any

from gptqmodel.models import auto
from gptqmodel.models.base import BaseQModel


class Spark2_5QModel(BaseQModel):  # noqa: N801
    """GPTQModel checkpoint reader for Spark2_5ForCausalLM (fused QKV)."""

    require_trust_remote_code = False
    layer_modules_strict = False
    pre_lm_head_norm_module = "model.norm"

    module_tree: list[Any] = [
        "model",
        "layers",
        "#",
        {
            "input_layernorm": ("input_layernorm:!",),
            "self_attn": (
                "g_proj:!",
                "q_k_v_proj:0",
                "out_proj:1",
            ),
            "post_attention_layernorm": ("post_attention_layernorm:!",),
            "mlp": ("gate_proj:0", "up_proj:0", "down_proj:1"),
        },
    ]


def register_spark2_5_gptqmodel() -> None:
    auto.MODEL_MAP["spark2_5"] = Spark2_5QModel
    auto.SUPPORTED_MODELS = list(auto.MODEL_MAP.keys())


register_spark2_5_gptqmodel()
