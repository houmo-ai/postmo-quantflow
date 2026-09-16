"""Weight-free GDN leaf ABI shared by dense and MoE low-memory exports."""

from ._hybrid_gated_delta_net import _split_conv_cache_layout


class HybridGatedDeltaNetPlaceholderMixin:
    def _update_cfg(self, cfg):
        self.split_conv_cache = cfg.get("split_conv_cache", True)
        self.linear_attention_mode = cfg.get("linear_attention_mode", "chunk")
        self.input_sequence_length = cfg.get("input_sequence_length", 256)
        self._verify_output_intermediates = cfg.get("verify_output_intermediates", False)
        self.suppress_recurrent_state_outputs = cfg.get("suppress_recurrent_state_outputs", False)

    @property
    def split_conv_cache_output_layout(self):
        return _split_conv_cache_layout(self)

    def forward(
        self, hidden_states, conv_cache=None, recurrent_state=None,
        linear_attn_mask=None, current_input_length=None,
    ):
        if not self.split_conv_cache:
            return hidden_states, conv_cache, recurrent_state
        if not isinstance(conv_cache, (tuple, list)) or len(conv_cache) != 3:
            raise ValueError("Split GDN placeholder requires three Q/K/V cache inputs")
        conv_count, state_count = self.split_conv_cache_output_layout
        # These values only carry shape/dtype through main-graph tracing;
        # independently exported real subgraphs replace every leaf later.
        conv_outputs = tuple(cache for cache in conv_cache for _ in range(conv_count // 3))
        return hidden_states, *conv_outputs, *((recurrent_state,) * state_count)
