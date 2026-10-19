"""Generate SAFE priors and directed ExFold tables from the consumed HF model.

The calibration protocol follows moe_prune_v2's AMP/AIMER builders and
moe_dynamic_topk's co-routed scalar regression. No research checkout is needed
at runtime; only sufficient statistics are retained, never a full output cache.
"""

from __future__ import annotations

import hashlib
import inspect
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from xhquant.api import get_xhquant_logger

from .dynamic_prune import _source_fingerprint, compute_layer_scalars
from .safe_calibration_stats import SafePairwiseCalibrator, compute_aimer_keep


CALIBRATION_DEFAULTS = {
    "dataset": "Salesforce/wikitext",
    "subset": "wikitext-2-raw-v1",
    "split": "train",
    "text_path": None,
    "nsamples": 128,
    "sequence_length": 2048,
    "batch_size": 1,
}


def normalize_calibration(config):
    if config is None:
        config = {}
    if not isinstance(config, dict) or set(config) - set(CALIBRATION_DEFAULTS):
        raise ValueError("Unknown or invalid SAFE calibration configuration")
    result = {**CALIBRATION_DEFAULTS, **config}
    for key in ("nsamples", "sequence_length", "batch_size"):
        if isinstance(result[key], bool) or not isinstance(result[key], int) or result[key] <= 0:
            raise ValueError(f"SAFE calibration.{key} must be a positive integer")
    if result["split"] != "train":
        raise ValueError("SAFE calibration must use the training split")
    return result


def calibration_input_ids(tokenizer, config):
    """Take one continuous training segment, without chat templating or shuffle."""
    if config["text_path"]:
        text = Path(config["text_path"]).read_text(encoding="utf-8")
    else:
        from datasets import load_dataset

        dataset = load_dataset(config["dataset"], config["subset"], split=config["split"])
        text = "\n".join(row["text"] for row in dataset)
    tokenizer.model_max_length = 2**31 - 1
    needed = config["nsamples"] * config["sequence_length"]
    # Only the resulting fixed-length windows enter the model.
    tokens = tokenizer(text, truncation=False, return_tensors="pt").input_ids.reshape(-1)
    if tokens.numel() < needed:
        raise ValueError(f"SAFE training text has {tokens.numel()} tokens; {needed} required")
    return tokens[:needed].reshape(config["nsamples"], config["sequence_length"]).contiguous()


@torch.no_grad()
def selected_expert_outputs(experts, hidden, ids):
    """Materialize unweighted routed outputs for this batch only."""
    outputs = hidden.new_empty((*ids.shape, hidden.shape[-1]))
    for expert_id in ids.unique():
        token, slot = (ids == expert_id).nonzero(as_tuple=True)
        gate, up = F.linear(hidden[token], experts.gate_up_proj[expert_id]).chunk(2, dim=-1)
        outputs[token, slot] = F.linear(experts.act_fn(gate) * up, experts.down_proj[expert_id])
    return outputs


def _pytorch_reference(function, name):
    """Find the pure PyTorch function behind a kernel-dispatch wrapper.

    Hub decorators copy the original name onto every wrapper and onto an
    installed FLA implementation. The reference is therefore the innermost
    function with that name outside the accelerated packages.
    """

    def module_name(candidate):
        try:
            return inspect.getmodule(candidate).__name__ or ""
        except Exception:
            return ""

    def is_reference(candidate):
        # functools.wraps copies the original name and source location onto every
        # wrapper, so only a function with no further __wrapped__ link is real.
        if not inspect.isfunction(candidate) or getattr(candidate, "__wrapped__", None) is not None:
            return False
        if getattr(candidate, "__name__", None) != name:
            return False
        return not module_name(candidate).startswith(("fla", "causal_conv1d"))

    seen = set()
    pending = [function]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen or not callable(current):
            continue
        seen.add(id(current))
        if is_reference(current):
            return current
        for cell in getattr(current, "__closure__", None) or ():
            try:
                pending.append(cell.cell_contents)
            except ValueError:
                continue
        pending.append(getattr(current, "__wrapped__", None))
    return None


@contextmanager
def portable_linear_attention(device):
    """Keep CPU calibration off Triton kernels shipped with flash-linear-attention.

    Transformers prefers an installed FLA package even when the activation is on
    CPU, and those kernels reject CPU pointers. CUDA calibration keeps the fast
    path. The replacement is confined to this call and restored on every exit.
    """
    if torch.device(device).type == "cuda":
        yield
        return
    try:
        from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as modeling
    except ImportError:
        yield
        return
    names = (
        "torch_chunk_gated_delta_rule",
        "torch_recurrent_gated_delta_rule",
        "causal_conv1d_fn",
        "causal_conv1d_update",
    )
    replacements = {}
    for attribute in names:
        reference_name = attribute
        current = getattr(modeling, attribute, None)
        reference = _pytorch_reference(current, reference_name) if current is not None else None
        if reference is not None and reference is not current:
            replacements[attribute] = current
            setattr(modeling, attribute, reference)
    try:
        yield
    finally:
        for name, current in replacements.items():
            setattr(modeling, name, current)


@torch.no_grad()
def calibrate_native_model(text_model, input_ids, device, batch_size=1, eps=1e-8):
    """Observe native routing before graph wrapping; restore hooks/device on exit."""
    logger = get_xhquant_logger()
    layers = list(text_model.layers)
    if batch_size <= 0 or input_ids.ndim != 2 or not len(input_ids):
        raise ValueError("SAFE calibration requires nonempty token windows and positive batch_size")
    device = torch.device(device)
    original_device = next(text_model.parameters()).device
    if original_device.type == "meta" or any(p.device != original_device for p in text_model.parameters()):
        raise ValueError("Automatic SAFE calibration requires a materialized model on one device")
    training = text_model.training
    amp, aimer, stats, hooks = {}, {}, [], []
    try:
        text_model.eval()
        # Stream expert matrices for the weights-only priors before moving the
        # full text model: AMP's FP32 intermediates never duplicate the model.
        for index, layer in enumerate(layers):
            experts = layer.mlp.experts
            gate, up = experts.gate_up_proj.chunk(2, dim=1)
            block = SimpleNamespace(
                expert_gate_proj_weight=gate,
                expert_up_proj_weight=up,
                expert_down_proj_weight=experts.down_proj,
            )
            amp[str(index)] = compute_layer_scalars(
                1 + layer.post_attention_layernorm.weight.float(), block, device, eps
            )
            aimer[str(index)] = compute_aimer_keep(experts.gate_up_proj, experts.down_proj, device=device, eps=eps)
            logger.info(f"SAFE AMP/AIMER: body layer {index + 1}/{len(layers)}")
        text_model.to(device)
        with portable_linear_attention(device):
            for layer in layers:
                accumulator = SafePairwiseCalibrator(layer.mlp.experts.num_experts, device=device)
                stats.append(accumulator)

                def observe(experts, inputs, accumulator=accumulator):
                    hidden, ids, _weights = inputs
                    accumulator.update(ids, selected_expert_outputs(experts, hidden, ids))

                hooks.append(layer.mlp.experts.register_forward_pre_hook(observe))
            for start in range(0, len(input_ids), batch_size):
                batch = input_ids[start : start + batch_size].to(device)
                text_model(input_ids=batch, use_cache=False)
                logger.info(
                    f"SAFE ExFold calibration: {min(start + batch_size, len(input_ids))}/{len(input_ids)} windows"
                )
        tables = [accumulator.finalize() for accumulator in stats]
        return (
            {"amp": amp, "aimer": aimer},
            {
                key: torch.stack([table[index] for table in tables])
                for index, key in enumerate(("coeff", "loss", "expert_norms"))
            },
        )
    finally:
        for hook in hooks:
            hook.remove()
        text_model.to(original_device)
        text_model.train(training)
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def prepare_automatic_safe_inputs(hf_model, prune, model_dir, work_dir, device="auto"):
    """Explicit inputs win; otherwise generate/reuse checkpoint-bound artifacts."""
    if prune.get("mode") != "decode" or prune.get("algorithm") != "safe":
        return
    if any(prune.get(key) for key in ("s_scalar_path", "ace_priors_path", "exfold_artifact_path")):
        return
    if not prune.get("auto_s_scalar", True):
        raise ValueError("SAFE requires calibration files or auto_s_scalar=true")
    from transformers import AutoTokenizer

    config = normalize_calibration(prune.get("calibration"))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    input_ids = calibration_input_ids(tokenizer, config)
    text_model = hf_model.model.language_model
    identity = {
        "method": "safe_corouted_calibration_v1",
        "source": _source_fingerprint(model_dir),
        "calibration": config,
        "tokens_sha256": hashlib.sha256(input_ids.numpy().tobytes()).hexdigest(),
        "dtype": str(next(text_model.parameters()).dtype),
        "layers": len(text_model.layers),
        "ridge": 1e-3,
        "coefficient_clip": 4.0,
    }
    root = Path(work_dir) / "safe_calibration"
    priors_path, exfold_path = root / "priors.pt", root / "exfold.pt"
    reuse = priors_path.is_file() and exfold_path.is_file() and not prune.get("overwrite_s_scalar", False)
    if reuse:
        for path in (priors_path, exfold_path):
            if torch.load(path, weights_only=True, map_location="cpu").get("identity") != identity:
                raise ValueError(
                    "SAFE calibration cache disagrees with checkpoint/data/settings; set overwrite_s_scalar=true"
                )
    else:
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        priors, exfold = calibrate_native_model(text_model, input_ids, device, config["batch_size"])
        _save({**priors, "identity": identity}, priors_path)
        _save({**exfold, "identity": identity}, exfold_path)
    prune["ace_priors_path"] = str(priors_path)
    prune["exfold_artifact_path"] = str(exfold_path)
    get_xhquant_logger().info(f"SAFE automatic calibration artifacts: {root} (reused={reuse})")
