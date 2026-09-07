"""ModelZoo parameter adapter for GPTQModel's public Ling recipe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class Ling3FlashQuantSpec:
    model_dir: str
    output_dir: str
    method: str
    dataset: str = "NeelNanda/pile-10k"
    calibration_jsonl: str | None = None
    text_key: str = "text"
    nsamples: int | None = None
    seqlen: int = 2048
    batch_size: int = 1
    moe_batch_size: int | None = None
    iters: int = 400
    group_size: int = 64
    base_bits: int = 8
    expert_bits: int = 4
    device: str = "cuda:0"
    device_map: str = "0"
    offload_to_disk: bool = False
    offload_dir: str = "work_dirs/ling_3_flash_offload"
    wait_for_submodule_finalizers: bool = True
    hessian_mse: bool = True
    max_quant_layers: int | None = None
    seed: int = 42
    dry_run: bool = False
    data_type: str = "int_sym_xh"
    mla_weight_layout: str = "xh2_absorbed_v1"

    def validate(self) -> None:
        self.method = str(self.method).lower().replace("-", "_")
        if self.method == "auto_round":
            self.method = "autoround"
        if self.nsamples is None:
            self.nsamples = 256 if self.method == "autoround" else 128
        if self.method not in {"gptq", "autoround"}:
            raise ValueError(f"Unsupported Ling quantization method: {self.method!r}")
        if self.group_size != 64:
            raise ValueError("Ling-3-Flash deployment profile requires group/block size 64")
        if self.base_bits != 8 or self.expert_bits != 4:
            raise ValueError("Ling-3-Flash profile is fixed to routed experts W4 and all other quantized linears W8")
        if self.nsamples <= 0 or self.seqlen <= 0 or self.batch_size <= 0:
            raise ValueError("nsamples, seqlen and batch_size must be positive")
        if self.moe_batch_size is not None and self.moe_batch_size <= 0:
            raise ValueError("moe_batch_size must be positive when specified")
        if self.iters < 0:
            raise ValueError("iters must be non-negative")
        if self.mla_weight_layout not in ("xh2_absorbed_v1", "hf_unabsorbed"):
            raise ValueError(f"Unsupported Ling MLA layout: {self.mla_weight_layout!r}")


def _recipe_kwargs(spec: Ling3FlashQuantSpec, calibration_texts=None) -> dict:
    spec.validate()
    kwargs = {
        "model_dir": spec.model_dir,
        "output_dir": spec.output_dir,
        "method": spec.method,
        "mla_weight_layout": spec.mla_weight_layout,
        "artifact_format": "gptqmodel_hf",
        "bits": spec.base_bits,
        "expert_bits": spec.expert_bits,
        "group_size": spec.group_size,
        "sym": True,
        "batch_size": spec.batch_size,
        "nsamples": spec.nsamples,
        "seqlen": spec.seqlen,
        "calibration_jsonl": spec.calibration_jsonl,
        "calibration_text_key": spec.text_key,
        "calibration_dataset": spec.dataset,
        "calibration_data": calibration_texts,
        "device": spec.device,
        "trust_remote_code": True,
        "offload_to_disk": spec.offload_to_disk,
        "offload_path": spec.offload_dir if spec.offload_to_disk else None,
        "hessian_mse": spec.hessian_mse,
        "wait_for_submodule_finalizers": spec.wait_for_submodule_finalizers,
        "moe_routing_batch_size": spec.moe_batch_size,
        "max_quant_layers": spec.max_quant_layers,
        "seed": spec.seed,
        "dry_run": spec.dry_run,
    }
    if spec.method == "autoround":
        kwargs.update(data_type=spec.data_type, iters=spec.iters, device_map=spec.device_map)
    return kwargs


def _recipe():
    from ..qwen3_5.quant_adapter import _load_callable

    return _load_callable(
        "gptqmodel.recipes.ling3_flash",
        "quantize_ling3_flash",
        "Ling quantization requires GPTQModel's Ling GPTQ/AutoRound recipe API.",
    )


def quantization_plan(spec: Ling3FlashQuantSpec) -> dict:
    kwargs = _recipe_kwargs(spec)
    result = _recipe()(**{**kwargs, "dry_run": True})
    return {
        "method": result.method,
        **({"data_type": spec.data_type} if spec.method == "autoround" else {}),
        "model_dir": str(Path(spec.model_dir).expanduser().resolve()),
        "output_dir": result.output_dir,
        "precision": result.precision,
        "group_size": spec.group_size,
        "symmetric": True,
        "calibration": result.calibration,
        "attention_backend": result.provenance["attention_backend"],
        "device": spec.device if spec.method == "gptq" else spec.device_map,
        # The recipe is queried without executing it here; report the caller's
        # run mode, not the dry-run used to resolve these parameters.
        "provenance": {**result.provenance, "dry_run": spec.dry_run},
    }


def quantize_ling3_flash(
    spec: Ling3FlashQuantSpec,
    *,
    calibration_texts: Iterable[str] | None = None,
) -> Path:
    result = _recipe()(**_recipe_kwargs(spec, calibration_texts))
    return Path(result.output_dir)


# Keep the old helper imports working while their implementations live solely
# in GPTQModel. New callers should use the public recipe instead.
def __getattr__(name: str):
    if name in {
        "build_autoround_block_names",
        "build_autoround_layer_config",
        "canonicalize_autoround_gptqmodel_config",
        "_install_autoround_nonfinite_loss_guard",
    }:
        from gptqmodel.recipes import ling3_flash_autoround

        return getattr(ling3_flash_autoround, name)
    raise AttributeError(name)


def autoround_calibration_source(spec: Ling3FlashQuantSpec, calibration_texts=None):
    from gptqmodel.recipes.ling3_flash import autoround_calibration_source as source

    return source(
        calibration_data=calibration_texts,
        calibration_jsonl=spec.calibration_jsonl,
        calibration_dataset=spec.dataset,
        calibration_text_key=spec.text_key,
    )


__all__ = ["Ling3FlashQuantSpec", "quantization_plan", "quantize_ling3_flash"]
