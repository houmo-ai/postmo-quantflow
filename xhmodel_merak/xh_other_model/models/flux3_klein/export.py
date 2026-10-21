# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""CPU graph conversion with bounded GPU residency for SDK weight PTQ."""

import gc
from copy import deepcopy

import torch


def export_graph(module, inputs, path, names, quant_config, *, device, dynamo=False):
    from xhquant.api import (
        ConfigDict,
        DeviceType,
        convert_dynamo_model_to_quanted_model,
        convert_fx_model_to_quanted_model,
        convert_quanted_model_to_hmonnx,
    )

    if any(value.device.type != "cpu" for value in (*module.parameters(), *module.buffers())):
        raise ValueError("source weights and graph calibration must remain on CPU")
    if any(value.device.type != "cpu" for value in inputs):
        raise ValueError("graph calibration inputs must remain on CPU")
    selected = torch.device(device)
    if selected.type != "cuda":
        raise ValueError("FLUX graph export requires CUDA for SDK weight PTQ")
    if selected.index is None:
        selected = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(selected)
    config = ConfigDict(deepcopy(quant_config))
    config.auto_release_unused_parameters = True
    converter = convert_dynamo_model_to_quanted_model if dynamo else convert_fx_model_to_quanted_model
    quanted = None
    try:
        quanted = converter(deepcopy(module).eval(), inputs, DeviceType.XH2a, config)
        quanted.cpu()
        convert_quanted_model_to_hmonnx(quanted, inputs, str(path), names, ["output"])
    finally:
        del quanted
        gc.collect()
        torch.cuda.empty_cache()
