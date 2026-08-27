from ..qwen3_5_moe.xh_qwen3_5_moe_config import XHQwen3_5MoeModelConfig


class XHQwen3_5MoeDynamicPruneModelConfig(XHQwen3_5MoeModelConfig):  # noqa: N801
    """Qwen3.5-MoE config with converter-independent prune controls."""

    def __init__(
        self,
        *,
        dynamic_prune_threshold: float = 0.85,
        dynamic_prune_s_scalar_path: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dynamic_prune_threshold = float(dynamic_prune_threshold)
        if not 0.0 < self.dynamic_prune_threshold <= 1.0:
            raise ValueError(
                "dynamic_prune_threshold is the ACEP cumulative coverage and must be in (0, 1], "
                f"got {self.dynamic_prune_threshold}"
            )
        self.dynamic_prune_s_scalar_path = dynamic_prune_s_scalar_path
