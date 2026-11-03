"""Thin GPTQModel/AutoRound adapter for MiniCPM-V-4.7 text backbones.

Workflow YAML stays in ModelZoo. GPTQ/AutoRound producer logic lives in
``gptqmodel.recipes.minicpm_v_4_7``. This module only:

* resolves repository-owned calibration resources
* translates ``quant_cfg`` into stable recipe kwargs
* keeps ``apply_causal_dequant_to_minicpm_host`` as host consumption glue
"""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xhmodel_merak.xh_llm.models.minicpm_o_4_5.resource_path import resolve_repo_resource


_DEFAULT_AUTOROUND_DATASET = "xh2modelzoo://data/calib_data/NeelNanda-pile-10k.jsonl"
_LEGACY_AUTOROUND_DATASETS = {"NeelNanda/pile-10k", "pile-10k"}


def quantize_minicpm_v47_llm_gptq(
    *,
    model_dir: str,
    output_dir: str,
    quant_cfg: Mapping[str, Any],
    device: str,
) -> str:
    """Resolve Dense GPTQ inputs and call the stable MiniCPM-V-4.7 recipe."""

    if not isinstance(quant_cfg, Mapping):
        raise TypeError(f"quant_cfg must be a mapping, got {type(quant_cfg)!r}")

    _ensure_gptqmodel_on_path(quant_cfg)
    calibration_jsonl = str(quant_cfg.get("calibration_jsonl") or "")
    if not calibration_jsonl:
        raise ValueError("MiniCPM-V-4.7 GPTQ requires quant.calibration_jsonl (text-only corpus)")
    calibration_path = resolve_repo_resource(calibration_jsonl, description="LLM calibration JSONL")

    sampling_seed = quant_cfg.get("sampling_seed")
    llm_backbone_dir = quant_cfg.get("llm_backbone_dir")
    from gptqmodel.recipes.minicpm_v_4_7 import quantize_minicpm_v_4_7_gptq

    result = quantize_minicpm_v_4_7_gptq(
        model_dir=model_dir,
        output_dir=output_dir,
        method=str(quant_cfg.get("method", "gptq")),
        artifact_format="float_causal_bin",
        bits=int(quant_cfg.get("bits", 4)),
        group_size=int(quant_cfg.get("group_size", 64)),
        damp_percent=float(quant_cfg.get("damp_percent", 0.01)),
        nsamples=int(quant_cfg.get("nsamples", 128)),
        seqlen=int(quant_cfg.get("seqlen", 1024)),
        sampling_seed=None if sampling_seed is None else int(sampling_seed),
        calibration_jsonl=str(calibration_path),
        llm_backbone_dir=None if llm_backbone_dir in (None, "") else str(llm_backbone_dir),
        device=device,
        trust_remote_code=_as_bool(quant_cfg.get("trust_remote_code", True)),
        dry_run=_as_bool(quant_cfg.get("dry_run", False)),
        **({"head": dict(quant_cfg["head"])} if quant_cfg.get("head") is not None else {}),
    )
    return str(result.output_dir)


def quantize_minicpm_v47_llm_autoround(
    *,
    model_dir: str,
    output_dir: str,
    quant_cfg: Mapping[str, Any],
    device: str,
) -> str:
    """Resolve MoE AutoRound inputs and call the stable MiniCPM-V-4.7 recipe."""

    if not isinstance(quant_cfg, Mapping):
        raise TypeError(f"quant_cfg must be a mapping, got {type(quant_cfg)!r}")

    _ensure_gptqmodel_on_path(quant_cfg)
    _ensure_autoround_on_path(quant_cfg)

    calibration_cfg = _section(quant_cfg, "calibration")
    runtime_cfg = _section(quant_cfg, "runtime")
    moe_cfg = _section(quant_cfg, "moe")
    bits = int(quant_cfg.get("bits", 4))
    dataset = _resolve_autoround_dataset(
        calibration_cfg.get(
            "dataset",
            calibration_cfg.get(
                "jsonl",
                quant_cfg.get("dataset", quant_cfg.get("calibration_jsonl", _DEFAULT_AUTOROUND_DATASET)),
            ),
        )
    )

    from gptqmodel.recipes.minicpm_v_4_7 import quantize_minicpm_v_4_7_autoround

    result = quantize_minicpm_v_4_7_autoround(
        model_dir=model_dir,
        output_dir=output_dir,
        method=str(quant_cfg.get("method", "autoround")),
        artifact_format="float_causal_bin",
        bits=bits,
        group_size=int(quant_cfg.get("group_size", 64)),
        sym=_as_bool(quant_cfg.get("sym", True)),
        data_type=str(quant_cfg.get("data_type", "int_sym_xh")),
        iters=int(quant_cfg.get("iters", 400)),
        nsamples=int(calibration_cfg.get("nsamples", quant_cfg.get("nsamples", 256))),
        seqlen=int(calibration_cfg.get("seqlen", quant_cfg.get("seqlen", 2048))),
        batch_size=int(runtime_cfg.get("batch_size", quant_cfg.get("batch_size", 8))),
        gradient_accumulate_steps=int(
            runtime_cfg.get(
                "gradient_accumulate_steps",
                quant_cfg.get("gradient_accumulate_steps", 1),
            )
        ),
        seed=int(runtime_cfg.get("seed", quant_cfg.get("seed", 42))),
        dataset=dataset,
        device=device,
        device_map=str(runtime_cfg.get("device_map", quant_cfg.get("device_map", "0"))),
        low_gpu_mem_usage=_as_bool(
            runtime_cfg.get("low_gpu_mem_usage", quant_cfg.get("low_gpu_mem_usage", True))
        ),
        save_quantized=_as_bool(quant_cfg.get("save_quantized", False)),
        export_format=str(quant_cfg.get("format", quant_cfg.get("autoround_format", "auto_round:gptqmodel"))),
        attn_bits=_optional_int(moe_cfg.get("attn_bits", moe_cfg.get("self_attn_bits", 8))),
        shared_expert_bits=_optional_int(moe_cfg.get("shared_expert_bits", 8)),
        expert_bits=_optional_int(moe_cfg.get("expert_bits", bits)),
        expert_up_gate_bits=_optional_int(moe_cfg.get("expert_up_gate_bits")),
        expert_down_bits=_optional_int(moe_cfg.get("expert_down_bits")),
        dry_run=_as_bool(quant_cfg.get("dry_run", False)),
        **({"head": dict(quant_cfg["head"])} if quant_cfg.get("head") is not None else {}),
    )
    return str(result.output_dir)


def apply_causal_dequant_to_minicpm_host(full_model: Any, quant_weight: str) -> None:
    """Load CausalLM QDQ weights and, when present, its calibrated packed head."""

    import torch

    root = Path(quant_weight)
    if root.is_dir():
        _apply_causal_head_checkpoint(full_model, root)
        return
    if not os.path.isfile(quant_weight):
        raise FileNotFoundError(f"quant_weight backbone not found: {quant_weight}")
    backbone_state = torch.load(quant_weight, weights_only=True, map_location="cpu")
    backbone_state = {
        key: value for key, value in backbone_state.items() if not str(key).endswith(".bias")
    }

    language_state: dict[str, Any] = {}
    lm_head_weight = None
    for key, value in backbone_state.items():
        if key == "lm_head.weight":
            lm_head_weight = value
            continue
        if key.startswith("model."):
            language_state[key[len("model.") :]] = value
            continue
        raise KeyError(f"unexpected CausalLM dequant key for MiniCPM-V-4.7: {key}")

    missing, unexpected = full_model.model.language_model.load_state_dict(language_state, strict=True)
    if missing or unexpected:
        raise RuntimeError(
            "Failed to apply MiniCPM-V-4.7 CausalLM dequant to language_model: "
            f"missing={missing}, unexpected={unexpected}"
        )
    if lm_head_weight is None:
        raise KeyError("CausalLM dequant is missing lm_head.weight")
    with torch.no_grad():
        full_model.lm_head.weight.copy_(lm_head_weight)


def validate_head_quant_config(quant_cfg, model_cfg) -> None:
    """Reject mismatched head requests before starting backbone calibration."""
    from xhmodel_merak.xh_llm.shared_head import target_head_bits

    bits = target_head_bits(model_cfg.get("quant_scheme"))
    head = quant_cfg.get("head")
    if head is None:
        if bits is not None and bits < 8:
            raise ValueError(f"MiniCPM export head W{bits} requires matching quant.head.bits={bits}")
        return
    if not isinstance(head, Mapping) or type(head.get("bits")) is not int or head["bits"] not in (4, 5, 8):
        raise ValueError("quant.head.bits must be 4, 5 or 8")
    if head["bits"] != bits:
        raise ValueError(f"quant.head.bits={head['bits']} conflicts with export lm_head W{bits}")


def prepare_causal_head_checkpoint(source, *, bits, raw_model_dir, quant_cfg, device):
    """Reuse a calibrated head or complete an existing QDQ backbone via GPTQModel."""
    import json

    root = Path(source).expanduser().resolve()
    if not (root / "config.json").is_file():
        raise FileNotFoundError(
            f"MiniCPM checkpoint config does not exist: {root / 'config.json'}. "
            "--quanted-model-dir must point to a saved checkpoint, not an HMONNX export directory."
        )
    config = json.loads((root / "config.json").read_text())
    if ((config.get("quantization_config") or {}).get("meta") or {}).get("lm_head_quantization"):
        validate_causal_head_checkpoint(root, bits)
        return str(root)
    if not isinstance(quant_cfg, Mapping) or not quant_cfg.get("head"):
        raise ValueError(f"A floating MiniCPM head requires quant.head.bits={bits} for calibration")
    _ensure_gptqmodel_on_path(quant_cfg)
    from gptqmodel.recipes.minicpm_v47_qdq_head import quantize_minicpm_v47_head

    calibration = _section(quant_cfg, "calibration")
    runtime = _section(quant_cfg, "runtime")
    dataset = calibration.get("jsonl", calibration.get("dataset", quant_cfg.get("calibration_jsonl")))
    if dataset is None:
        raise ValueError("MiniCPM head calibration requires calibration.jsonl or calibration.dataset")
    dataset = _resolve_autoround_dataset(dataset)
    calibration_key = "calibration_jsonl" if Path(str(dataset)).is_file() else "calibration_dataset"
    result = quantize_minicpm_v47_head(
        model_dir=str(root), tokenizer_dir=raw_model_dir, head=dict(quant_cfg["head"]),
        nsamples=int(calibration.get("nsamples", quant_cfg.get("nsamples", 256))),
        seqlen=int(calibration.get("seqlen", quant_cfg.get("seqlen", 2048))),
        device=device, seed=int(runtime.get("seed", quant_cfg.get("sampling_seed", 42))),
        **{calibration_key: str(dataset)},
    )
    validate_causal_head_checkpoint(result.output_dir, bits)
    return result.output_dir


def validate_causal_head_checkpoint(source: str | Path, bits: int) -> None:
    """Require the producer's float-QDQ backbone and matching calibrated head."""
    import json

    from gptqmodel.quantization.lm_head import HeadQuantizationConfig, inspect_checkpoint_head

    root = Path(source)
    config = json.loads((root / "config.json").read_text())
    quant = config.get("quantization_config") or {}
    producer = (quant.get("meta") or {}).get("float_qdq_backbone") or {}
    if (producer.get("model_family") != "minicpm_v_4_7"
            or producer.get("artifact_format") != "float_causal_bin"
            or config.get("model_type") not in {"qwen3_5_text", "qwen3_5_moe_text"}):
        raise ValueError("Expected a MiniCPM-V-4.7 float-QDQ backbone with calibrated head")
    if quant.get("dynamic", {}).get("-:^(?!lm_head$).*") != {}:
        raise ValueError("Packed-head checkpoint must retain the float-QDQ backbone")
    probe = inspect_checkpoint_head(root, HeadQuantizationConfig(bits=bits))
    if probe is None:
        raise ValueError(f"MiniCPM lm_head W{bits} requires a calibrated packed head")
    actual_bits = int(probe.get("bits", bits))
    if actual_bits != bits:
        raise ValueError(f"MiniCPM lm_head W{bits} is already packed as W{actual_bits}; already packed checkpoint cannot be re-packed")


def _apply_causal_head_checkpoint(full_model: Any, root: Path) -> None:
    import json

    from safetensors import safe_open
    from xhmodel_merak.xh_llm.shared_head import load_checkpoint_head

    config = json.loads((root / "config.json").read_text())
    bits = config["quantization_config"]["meta"]["lm_head_quantization"]["bits"]
    validate_causal_head_checkpoint(root, bits)
    index = json.loads((root / "model.safetensors.index.json").read_text())
    language_state = {}
    for filename in sorted(set(index["weight_map"].values())):
        with safe_open(root / filename, framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                if key.startswith("model."):
                    language_state[key[len("model."):]] = tensors.get_tensor(key)
                elif not key.startswith("lm_head."):
                    raise KeyError(f"Unexpected MiniCPM causal checkpoint tensor: {key}")
    language_model = full_model.model.language_model
    # GPTQModel's endpoint writer retains its per-expert Linear layout. Native
    # MiniCPM uses fused gate/up and down parameters; restore that layout with
    # exact stacking before loading into the host.
    import torch

    for name, parameter in language_model.state_dict().items():
        if not name.endswith(".mlp.experts.gate_up_proj") or name in language_state:
            continue
        prefix = name.removesuffix("gate_up_proj")
        gates, ups, downs = [], [], []
        for expert in range(parameter.shape[0]):
            gates.append(language_state.pop(prefix + f"{expert}.gate_proj.weight"))
            ups.append(language_state.pop(prefix + f"{expert}.up_proj.weight"))
            downs.append(language_state.pop(prefix + f"{expert}.down_proj.weight"))
        language_state[name] = torch.stack([torch.cat((gate, up), dim=0) for gate, up in zip(gates, ups)])
        language_state[prefix + "down_proj"] = torch.stack(downs)
    language_model.load_state_dict(language_state, strict=True)
    dtype = full_model.lm_head.weight.dtype
    full_model.lm_head, actual_bits = load_checkpoint_head(root, dtype=dtype)
    if actual_bits != bits:
        raise ValueError("Calibrated head payload and metadata precision differ")


def _ensure_gptqmodel_on_path(quant_cfg: Mapping[str, Any]) -> None:
    gptqmodel_source = quant_cfg.get("gptqmodel_source") or os.environ.get("GPTQMODEL_SOURCE")
    if gptqmodel_source and str(gptqmodel_source) not in sys.path:
        sys.path.insert(0, str(gptqmodel_source))


def _ensure_autoround_on_path(quant_cfg: Mapping[str, Any]) -> None:
    autoround_source = (
        quant_cfg.get("autoround_source")
        or os.environ.get("AUTOROUND_SOURCE")
        or (Path(os.environ.get("GPTQMODEL_SOURCE", "")) / "third_party" / "auto-round")
    )
    if autoround_source and Path(autoround_source).is_dir():
        autoround_source = str(Path(autoround_source).resolve())
        if autoround_source not in sys.path:
            sys.path.insert(0, autoround_source)


def _resolve_autoround_dataset(dataset: Any) -> str:
    value = str(dataset or _DEFAULT_AUTOROUND_DATASET).strip()
    if value in _LEGACY_AUTOROUND_DATASETS:
        value = _DEFAULT_AUTOROUND_DATASET
    if (
        value.startswith(("xh2modelzoo://", "repo://"))
        or value.endswith(".jsonl")
        or os.path.isfile(value)
    ):
        return str(resolve_repo_resource(value, description="AutoRound calibration dataset"))
    return value


def _section(cfg: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = cfg.get(key) or {}
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"quant.{key} must be a mapping when provided")
    return value


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


__all__ = [
    "apply_causal_dequant_to_minicpm_host",
    "quantize_minicpm_v47_llm_autoround",
    "quantize_minicpm_v47_llm_gptq",
]
