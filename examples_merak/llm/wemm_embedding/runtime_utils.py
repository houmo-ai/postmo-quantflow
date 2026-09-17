"""Shared WeMM-Embedding HMONNX runtime helpers."""

from __future__ import annotations

from pathlib import Path


def resolve_hmonnx_meta(export_dir: str | Path) -> Path:
    root = Path(export_dir)
    if root.is_file():
        return root
    for name in ("golden_meta_info.json", "export_meta_info.json"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    nested = sorted(root.glob("**/golden_meta_info.json"))
    if nested:
        return nested[0]
    raise FileNotFoundError(f"No WeMM HMONNX metadata found under {export_dir}")


def load_hmonnx_embedder(export_dir: str | Path, device: str):
    import torch

    from xhmodel_merak.xh_llm import AutoLLMHONNXModel

    meta_file = resolve_hmonnx_meta(export_dir)
    model = AutoLLMHONNXModel.from_pretrained(str(meta_file), device_map=[device])
    model.to(torch.device(device))
    return model, meta_file


__all__ = ["load_hmonnx_embedder", "resolve_hmonnx_meta"]
