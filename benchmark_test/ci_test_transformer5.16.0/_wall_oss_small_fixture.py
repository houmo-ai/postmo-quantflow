"""Minimal real routed Wall layers for CPU and SDK regressions."""

from xhmodel_merak.xh_other_model.models.wall_oss.source import Source


def make_small_source():
    return Source(
        dict(
            hidden_size=128,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_hidden_layers=2,
            rms_norm_eps=1e-6,
            experts=[dict(hidden_size=128, intermediate_size=192), dict(hidden_size=64, intermediate_size=128)],
        ),
        6,
    ).eval()
