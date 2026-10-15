# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Freeze model-specific norm shapes; Conv3d lowering belongs to the SDK."""


def prepare_static_shapes(module, inputs):
    """Use fake execution for shape discovery, with no real large-tensor convolution."""
    from torch._subclasses.fake_tensor import FakeTensorMode

    hooks = []
    for layer in module.modules():
        if hasattr(layer, "set_static_shape"):

            def freeze(layer, values):
                layer.set_static_shape(tuple(int(x) for x in values[0].shape))

            hooks.append(layer.register_forward_pre_hook(freeze))
    if not hooks:
        return module
    try:
        with FakeTensorMode(allow_non_fake_inputs=True) as mode:
            module(*(mode.from_tensor(value) for value in inputs))
    finally:
        for hook in hooks:
            hook.remove()
    return module
