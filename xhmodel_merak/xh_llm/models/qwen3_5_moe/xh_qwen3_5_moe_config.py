from ..qwen3_5.xh_qwen3_5_config import XHQwen3_5_VisualConfig, XHQwen3_5ModelConfig
from .dynamic_prune import normalize_dynamic_prune


class XHQwen3_5Moe_VisualConfig(XHQwen3_5_VisualConfig):  # noqa: N801
    pass


class XHQwen3_5MoeModelConfig(XHQwen3_5ModelConfig):  # noqa: N801
    def __init__(self, *, dynamic_prune=None, **kwargs):
        super().__init__(**kwargs)
        self.dynamic_prune = normalize_dynamic_prune(dynamic_prune)
        if self.dynamic_prune["mode"] == "decode" and self.spec_decode_mode is not None:
            raise ValueError("Decode ACEP currently supports ordinary decode, not MTP/DFlash verification")
