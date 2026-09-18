from ..qwen3_5_moe.xh_qwen3_5_moe_config import XHQwen3_5MoeModelConfig


class XHQwen3_5MoeDynamicPruneModelConfig(XHQwen3_5MoeModelConfig):  # noqa: N801
    """Retain flat legacy settings while using canonical decode-only pruning."""

    def __init__(
        self,
        *,
        dynamic_prune_threshold: float = 0.85,
        dynamic_prune_s_scalar_path: str | None = None,
        dynamic_prune=None,
        **kwargs,
    ):
        prune = dict(dynamic_prune or {})
        prune.setdefault("mode", "decode")
        prune.setdefault("threshold", dynamic_prune_threshold)
        prune.setdefault("s_scalar_path", dynamic_prune_s_scalar_path)
        super().__init__(dynamic_prune=prune, **kwargs)
        self.dynamic_prune_threshold = self.dynamic_prune["threshold"]
        self.dynamic_prune_s_scalar_path = self.dynamic_prune.get("s_scalar_path")
