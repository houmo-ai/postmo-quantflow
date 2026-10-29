# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Isolated CPU graph conversion; CUDA is reserved for SDK weight quantization."""

from __future__ import annotations

import gc
from copy import deepcopy
from pathlib import Path

import torch


def validate_conv3d_mode(mode, *, fastpath=False):
    if mode not in ("slice", "gather"):
        raise ValueError("upscaler_conv3d_mode must be 'slice' or 'gather'")
    if type(fastpath) is not bool:
        raise ValueError("upscaler_conv3d_fastpath must be a boolean")
    from inspect import signature

    from xhquant.frontend.convert_impl import build_fx_transformer
    from xhquant.frontend.fx_transform.conv3d_to_conv2d import Conv3dToConv2d

    parameters = signature(Conv3dToConv2d).parameters
    if (
        (mode == "gather" and "mode" not in parameters)
        or "enable_fastpath" not in parameters
        or "quant_config" not in signature(build_fx_transformer).parameters
    ):
        raise ValueError("upscaler lowering requires an SDK with configurable Conv3dToConv2d fastpath")
    return mode


def export_graph(
    module,
    inputs,
    path,
    input_names,
    output_names,
    *,
    quant_scheme,
    dynamo=False,
    norm_fp16=False,
    conv3d_mode=None,
    conv3d_fastpath=None,
):
    from torch.utils._pytree import tree_leaves

    from xhquant.api import (
        ConfigDict,
        DeviceType,
        QuantScheme,
        convert_dynamo_model_to_quanted_model,
        convert_fx_model_to_quanted_model,
        convert_quanted_model_to_hmonnx,
        create_quant_config,
    )

    flat_inputs = tuple(tree_leaves(inputs))
    if any(v.device.type != "cpu" for v in (*module.parameters(), *module.buffers(), *flat_inputs)):
        raise ValueError("H3 source graphs and calibration tensors must reside on CPU")
    if len(flat_inputs) != len(input_names):
        raise ValueError("each static input requires a published name")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scheme = deepcopy(quant_scheme)
    scheme["target_device"] = DeviceType[scheme["target_device"]]
    config = ConfigDict(create_quant_config(QuantScheme(**scheme)))
    if conv3d_mode is not None or conv3d_fastpath is not None:
        mode = validate_conv3d_mode(
            "slice" if conv3d_mode is None else conv3d_mode,
            fastpath=False if conv3d_fastpath is None else conv3d_fastpath,
        )
        if conv3d_mode is not None:
            config.conv3d_to_conv2d_mode = mode
        if conv3d_fastpath is not None:
            config.conv3d_to_conv2d_fastpath = conv3d_fastpath
    if type(norm_fp16) is not bool:
        raise ValueError("norm_fp16 must be a boolean")
    # Use the SDK's exponent-scaled FP16 RMS reduction. Only DiT callers opt
    # in: scaled VAE/Qwen epsilon values need their existing wide path.
    rms = dict(config.ops_cfg.get("RMSNorm", {}))
    rms.update(
        force_fp32=not norm_fp16,
        sum_dtype="float16" if norm_fp16 else "float32",
        keep_fp32_variance=not norm_fp16,
    )
    config.ops_cfg["RMSNorm"] = rms
    # This SDK's ConvTranspose SEFP quantizer retains its packed coefficient
    # buffer under qweight, while its eager release hook checks HMFP buffers.
    # Keep source coefficients only for this small audio stage until export;
    # the serialized graph still contains the requested SEFP coefficients.
    config.auto_release_unused_parameters = not any(
        isinstance(layer, (torch.nn.ConvTranspose1d, torch.nn.ConvTranspose2d, torch.nn.ConvTranspose3d))
        for layer in module.modules()
    )
    convert = convert_dynamo_model_to_quanted_model if dynamo else convert_fx_model_to_quanted_model
    quanted = None
    try:
        from .conv_graph import prepare_static_shapes

        current = prepare_static_shapes(deepcopy(module).eval(), inputs)
        if scheme["quant_type"] == "w16a16_sefp":
            from xhquant.nn import FlashAttention

            for attention in current.modules():
                if isinstance(attention, (FlashAttention)):
                    for field in ("q_bits", "k_bits", "v_bits", "s_bits", "p_bits"):
                        setattr(attention, field, 16)
        quanted = convert(current, inputs, quant_scheme["target_device"], config)
        quanted.cpu()
        convert_quanted_model_to_hmonnx(quanted, flat_inputs, str(destination), input_names, output_names)
        from .graph_audit import audit_graph

        audit_graph(
            destination,
            quant_type=scheme["quant_type"],
            inputs=input_names,
            outputs=output_names,
            norm_fp16=norm_fp16,
        )
    finally:
        del quanted
        gc.collect()
        torch.cuda.empty_cache()
    return destination
