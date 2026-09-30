# SPDX-License-Identifier: Apache-2.0
"""Entry points for the ``vllm_merak`` plugin package."""


def register_plugins() -> None:
    """Register Merak config, loader, and architecture hooks early."""
    import vllm_merak.config  # noqa: F401

    # import vllm_merak.model_load  # noqa: F401
    from vllm_merak.platform import register_architecture

    register_architecture()


__all__ = ["register_plugins"]
