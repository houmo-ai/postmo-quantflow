"""Consume Bonsai Phase A artifacts without changing generic HF loaders."""

import json
from pathlib import Path

import torch
import torch.nn as nn


def checkpoint_paths(directory, checkpoint=None):
    directory = Path(directory)
    metadata = json.loads((directory / "phase_a_metadata.json").read_text())
    checkpoint = Path(checkpoint) if checkpoint else directory / metadata.get("checkpoint_file", "quant_weight.pt")
    aux = directory / metadata.get("aux_file", "phase_a_aux.pt")
    return checkpoint, aux, metadata


def load_phase_a_checkpoint(model, checkpoint, *, allow_partial=False):
    checkpoint, aux, metadata = checkpoint_paths(Path(checkpoint).parent, checkpoint)
    if metadata.get("format") != "xh2_pq2_via_ssfp4_compat" or int(metadata.get("version", 1)) < 2:
        raise ValueError("Bonsai requires a v2 Phase A checkpoint; regenerate the converted artifact")
    if not allow_partial and not all(metadata.get(key) for key in (
        "complete_language_model", "float_tensors_converted", "embedding_inverse_applied",
    )):
        raise ValueError("Full Bonsai export requires inverse embedding and all converted float tensors")
    if not aux.is_file():
        raise FileNotFoundError(f"Bonsai online rotation requires {aux}")
    bits = int(metadata["ssfp"]["man_bit"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    payload = {key[:-len('.quant_weight')]: value for key, value in state.items() if key.endswith('.quant_weight')}
    floats = {key: value for key, value in state.items() if not key.endswith('.quant_weight')}
    if not allow_partial:
        expected = {key for key in model.state_dict() if key.startswith(("model.language_model.", "lm_head."))}
        missing = sorted(expected - floats.keys())
        if missing:
            raise ValueError(f"Bonsai language parameters missing ({len(missing)}): {missing[:8]}")
    incompatible = model.load_state_dict(floats, strict=False, assign=True)
    for name, value in payload.items():
        module = model.get_submodule(name)
        if not isinstance(module, nn.Linear):
            raise ValueError(f"Bonsai integer payload must belong to a Linear: {name}")
        module.register_buffer("quant_weight", value.to(module.weight.device), persistent=False)
        module.logical_bits = bits
    model._bonsai_phase_a_loaded = True
    return {"registered_quant_weight": len(payload), "missing_keys": len(incompatible.missing_keys)}
