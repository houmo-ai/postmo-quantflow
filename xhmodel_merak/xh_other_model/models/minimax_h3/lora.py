from __future__ import annotations

import gc
from collections.abc import Mapping
from pathlib import Path

import torch


DEFAULT_LORA_ALPHA = 8
DEFAULT_LORA_SCALE = 1.0
DEFAULT_LORA_FILENAME = "minimax_h3_fl2v_turbo_8step_v1.0_bf16.safetensors"
LORA_TARGET_MODULES = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
)
_LORA_A_SUFFIX = ".lora_A.default.weight"
_LORA_B_SUFFIX = ".lora_B.default.weight"


def default_lora_path(repo_root: str | Path) -> Path:
    return Path(repo_root).expanduser().resolve() / "data" / "models" / DEFAULT_LORA_FILENAME


def _load_lora_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    from safetensors.torch import load_file as load_safetensors_file

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"LoRA checkpoint does not exist: {path}")
    checkpoint = load_safetensors_file(path, device="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Expected a state-dict mapping in {path}, got {type(checkpoint).__name__}.")
    if not all(isinstance(key, str) for key in checkpoint):
        raise TypeError(f"LoRA checkpoint contains non-string keys: {path}")
    if not all(isinstance(value, torch.Tensor) for value in checkpoint.values()):
        raise TypeError(f"LoRA checkpoint contains non-tensor values: {path}")
    return checkpoint


def _validate_lora_state_dict(state_dict: Mapping[str, torch.Tensor], path: Path) -> int:
    lora_a = {}
    lora_b = {}
    unsupported_keys = []
    for key, tensor in state_dict.items():
        if key.endswith(_LORA_A_SUFFIX):
            lora_a[key[: -len(_LORA_A_SUFFIX)]] = tensor
        elif key.endswith(_LORA_B_SUFFIX):
            lora_b[key[: -len(_LORA_B_SUFFIX)]] = tensor
        else:
            unsupported_keys.append(key)

    if unsupported_keys:
        preview = ", ".join(unsupported_keys[:3])
        raise ValueError(f"{path} is not a pure PEFT LoRA state dict; unsupported keys: {preview}")
    if not lora_a:
        raise ValueError(f"No {_LORA_A_SUFFIX} tensors found in {path}.")

    missing_a = sorted(lora_b.keys() - lora_a.keys())
    missing_b = sorted(lora_a.keys() - lora_b.keys())
    if missing_a or missing_b:
        raise ValueError(f"Unpaired LoRA tensors in checkpoint: missing A={missing_a[:3]}, missing B={missing_b[:3]}.")

    ranks = set()
    for module_name in lora_a:
        a_tensor = lora_a[module_name]
        b_tensor = lora_b[module_name]
        if a_tensor.ndim != 2 or b_tensor.ndim != 2:
            raise ValueError(
                f"LoRA tensors for {module_name} must be matrices, got "
                f"A{tuple(a_tensor.shape)} and B{tuple(b_tensor.shape)}."
            )
        if a_tensor.shape[0] != b_tensor.shape[1]:
            raise ValueError(
                f"LoRA rank mismatch for {module_name}: A{tuple(a_tensor.shape)} and B{tuple(b_tensor.shape)}."
            )
        if not module_name.endswith(LORA_TARGET_MODULES):
            raise ValueError(f"Unsupported LoRA target module {module_name!r} in {path}.")
        ranks.add(a_tensor.shape[0])

    if len(ranks) != 1:
        raise ValueError(f"Mixed LoRA ranks are unsupported, found {sorted(ranks)}.")
    return ranks.pop()


def _resolve_linear_module(transformer: torch.nn.Module, module_name: str) -> torch.nn.Linear:
    module: torch.nn.Module = transformer
    for part in module_name.split("."):
        if part.isdigit():
            module = module[int(part)]  # type: ignore[index]
        else:
            module = getattr(module, part)
    if not isinstance(module, torch.nn.Linear):
        raise TypeError(f"LoRA target {module_name!r} is {type(module).__name__}, expected nn.Linear.")
    return module


def load_minimax_h3_lora(
    transformer: torch.nn.Module,
    lora_path: str | Path,
    *,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_scale: float = DEFAULT_LORA_SCALE,
) -> None:
    """Fuse a pure PEFT-format LoRA into transformer Linear weights.

    Avoids importing ``peft`` so export works even when peft/gptqmodel versions conflict.
    Effective update matches the previous peft path:
    ``W <- W + (lora_scale * lora_alpha / rank) * (B @ A)``.
    """

    path = Path(lora_path).expanduser().resolve()
    state_dict = _load_lora_state_dict(path)
    rank = _validate_lora_state_dict(state_dict, path)
    effective_scale = float(lora_scale) * float(lora_alpha) / float(rank)

    module_names = sorted({key[: -len(_LORA_A_SUFFIX)] for key in state_dict if key.endswith(_LORA_A_SUFFIX)})
    fused_modules = 0
    with torch.no_grad():
        for module_name in module_names:
            linear = _resolve_linear_module(transformer, module_name)
            lora_a = state_dict[f"{module_name}{_LORA_A_SUFFIX}"].to(
                device=linear.weight.device,
                dtype=torch.float32,
            )
            lora_b = state_dict[f"{module_name}{_LORA_B_SUFFIX}"].to(
                device=linear.weight.device,
                dtype=torch.float32,
            )
            delta = torch.matmul(lora_b, lora_a)
            if tuple(delta.shape) != tuple(linear.weight.shape):
                raise ValueError(
                    f"LoRA delta shape mismatch for {module_name}: "
                    f"delta{tuple(delta.shape)} vs weight{tuple(linear.weight.shape)}."
                )
            linear.weight.add_(delta, alpha=effective_scale)
            fused_modules += 1

    transformer.requires_grad_(False)
    transformer.eval()
    tensor_count = len(state_dict)
    del state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        f"Loaded LoRA: path={path} tensors={tensor_count} modules={fused_modules} "
        f"rank={rank} alpha={lora_alpha} scale={lora_scale} "
        f"effective_scale={effective_scale:.8g} fused=True peft_free=True",
        flush=True,
    )


__all__ = [
    "DEFAULT_LORA_ALPHA",
    "DEFAULT_LORA_FILENAME",
    "DEFAULT_LORA_SCALE",
    "LORA_TARGET_MODULES",
    "default_lora_path",
    "load_minimax_h3_lora",
]
