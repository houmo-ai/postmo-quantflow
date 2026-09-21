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


def load_minimax_h3_lora(
    transformer: torch.nn.Module,
    lora_path: str | Path,
    *,
    lora_alpha: int = DEFAULT_LORA_ALPHA,
    lora_scale: float = DEFAULT_LORA_SCALE,
) -> None:
    from peft import LoraConfig

    path = Path(lora_path).expanduser().resolve()
    state_dict = _load_lora_state_dict(path)
    rank = _validate_lora_state_dict(state_dict, path)
    transformer.add_adapter(
        LoraConfig(
            r=rank,
            lora_alpha=lora_alpha,
            init_lora_weights=False,
            target_modules=list(LORA_TARGET_MODULES),
            use_rslora=False,
        )
    )

    adapter_parameters = {
        name: parameter
        for name, parameter in transformer.named_parameters()
        if ".lora_A." in name or ".lora_B." in name
    }
    missing = sorted(adapter_parameters.keys() - state_dict.keys())
    unexpected = sorted(state_dict.keys() - adapter_parameters.keys())
    shape_mismatches = [
        (name, tuple(state_dict[name].shape), tuple(parameter.shape))
        for name, parameter in adapter_parameters.items()
        if name in state_dict and state_dict[name].shape != parameter.shape
    ]
    if missing or unexpected or shape_mismatches:
        raise ValueError(
            "LoRA checkpoint is incompatible with the loaded H3 transformer: "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}, shape_mismatches={shape_mismatches[:3]}."
        )

    incompatible = transformer.load_state_dict(state_dict, strict=False)
    missing_lora = [key for key in incompatible.missing_keys if ".lora_A." in key or ".lora_B." in key]
    if incompatible.unexpected_keys or missing_lora:
        raise RuntimeError(
            "LoRA loading did not consume the expected adapter tensors: "
            f"missing={missing_lora[:3]}, unexpected={incompatible.unexpected_keys[:3]}."
        )

    transformer.set_adapters("default", weights=lora_scale)
    transformer.fuse_lora(lora_scale=1.0, safe_fusing=True, adapter_names=["default"])
    transformer.unload_lora()
    transformer.requires_grad_(False)
    transformer.eval()
    tensor_count = len(state_dict)
    del state_dict
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        f"Loaded LoRA: path={path} tensors={tensor_count} rank={rank} alpha={lora_alpha} "
        f"scale={lora_scale} effective_scale={lora_scale * lora_alpha / rank:.8g} fused=True",
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
