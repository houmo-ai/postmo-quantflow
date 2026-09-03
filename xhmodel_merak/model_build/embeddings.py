"""Canonical, round-trip-checked runtime embedding assets."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_artifact.files import file_sha256
from xhmodel_merak.model_catalog.registry import get_adapter_spec

from .contracts import BuildPlan


def write_embedding_bins(stage_path: Path, plan: BuildPlan | None = None) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    embedding_paths = set(stage_path.rglob("quant_embedding.pt")) | set(stage_path.rglob("token_embedding.pt"))
    for pt_path in sorted(embedding_paths):
        temporary = pt_path.with_name(f".{pt_path.name}.normalize-{uuid.uuid4().hex}.tmp")
        try:
            import numpy as np
            import torch

            payload = torch.load(pt_path, map_location="cpu", weights_only=True)
            source_tensor = _embedding_tensor(payload)
            tensor = source_tensor.detach().to(dtype=torch.float16, device="cpu").contiguous()
            if tensor.ndim != 2:
                raise ValueError(f"embedding tensor must be two-dimensional, got shape={tuple(tensor.shape)}")
            if any(dimension <= 0 for dimension in tensor.shape):
                raise ValueError(f"embedding tensor dimensions must be positive, got shape={tuple(tensor.shape)}")
            expected_hidden_size = _expected_embedding_hidden_size(pt_path, stage_path, plan)
            if expected_hidden_size is not None and int(tensor.shape[1]) != expected_hidden_size:
                raise ValueError(
                    f"embedding hidden dimension {int(tensor.shape[1])} does not match model config "
                    f"hidden_size={expected_hidden_size}"
                )
            if not torch.isfinite(tensor).all():
                raise ValueError("embedding tensor contains non-finite values")

            normalized_payload = _replace_embedding_tensor(payload, source_tensor, tensor)
            torch.save(normalized_payload, temporary)
            temporary.replace(pt_path)

            persisted = _embedding_tensor(torch.load(pt_path, map_location="cpu", weights_only=True))
            if (
                persisted.dtype != torch.float16
                or persisted.device.type != "cpu"
                or not persisted.is_contiguous()
                or tuple(persisted.shape) != tuple(tensor.shape)
                or not torch.equal(persisted, tensor)
            ):
                raise ValueError("normalized embedding .pt did not round-trip as contiguous CPU float16")

            bin_path = pt_path.with_suffix(".bin")
            array = tensor.numpy().astype("<f2", copy=False)
            bin_path.write_bytes(array.tobytes(order="C"))
            persisted_bin = np.frombuffer(bin_path.read_bytes(), dtype="<f2").reshape(array.shape)
            if not np.array_equal(persisted_bin, array):
                raise ValueError("normalized embedding .bin did not round-trip exactly")
            metadata.append(
                {
                    "pt": pt_path.relative_to(stage_path).as_posix(),
                    "bin": bin_path.relative_to(stage_path).as_posix(),
                    "shape": list(tensor.shape),
                    "dtype": "float16",
                    "byte_order": "little",
                    "pt_sha256": file_sha256(pt_path),
                    "bin_sha256": file_sha256(bin_path),
                }
            )
        except Exception as exc:
            raise ModelError(
                ErrorCode.ARTIFACT_VALIDATION_FAILED,
                f"Failed to normalize token embedding {pt_path}: {exc}",
                embedding=str(pt_path),
            ) from exc
        finally:
            temporary.unlink(missing_ok=True)
    return metadata


def _expected_embedding_hidden_size(pt_path: Path, stage_path: Path, plan: BuildPlan | None) -> int | None:
    documents: list[tuple[Path, bool]] = [
        (pt_path.parent / "export_meta_info.json", True),
        (pt_path.parent / "golden_meta_info.json", True),
        (pt_path.parent / "config.json", False),
        (pt_path.parent / "hf_config" / "config.json", False),
        (stage_path / "hf_config" / "config.json", False),
    ]
    if (
        plan is not None
        and get_adapter_spec(
            "build",
            plan.build_adapter,
        ).source_config_for_embeddings
    ):
        documents.append((Path(plan.model_dir) / "config.json", False))

    candidates: dict[Path, int] = {}
    for path, allow_direct_metadata in documents:
        if path in candidates or not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        hidden_size = _config_hidden_size(payload, allow_direct_metadata=allow_direct_metadata)
        if hidden_size is not None:
            candidates[path] = hidden_size
    sizes = set(candidates.values())
    if len(sizes) > 1:
        raise ValueError(
            "packaged/source model configs declare conflicting embedding hidden sizes: "
            + ", ".join(f"{path}={size}" for path, size in candidates.items())
        )
    return next(iter(sizes), None)


def _config_hidden_size(value: Any, *, allow_direct_metadata: bool = False) -> int | None:
    if not isinstance(value, Mapping):
        return None

    text_sizes: set[int] = set()

    def collect_text_configs(item: Any) -> None:
        if not isinstance(item, Mapping):
            return
        for key, child in item.items():
            if key == "text_config" and isinstance(child, Mapping):
                hidden_size = child.get("hidden_size")
                if isinstance(hidden_size, int) and not isinstance(hidden_size, bool) and hidden_size > 0:
                    text_sizes.add(hidden_size)
            collect_text_configs(child)

    collect_text_configs(value)
    if len(text_sizes) == 1:
        return next(iter(text_sizes))
    if text_sizes:
        return None

    direct = value.get("hidden_size")
    if isinstance(direct, int) and not isinstance(direct, bool) and direct > 0:
        return direct
    if allow_direct_metadata:
        model_config = value.get("model_config")
        if isinstance(model_config, Mapping):
            return _config_hidden_size(model_config, allow_direct_metadata=True)
    return None


def _embedding_tensor(payload: Any) -> Any:
    if hasattr(payload, "detach"):
        return payload
    if isinstance(payload, Mapping):
        for key in ("weight", "embed_tokens.weight", "model.embed_tokens.weight"):
            value = payload.get(key)
            if hasattr(value, "detach"):
                return value
        tensors = [value for value in payload.values() if hasattr(value, "detach")]
        if len(tensors) == 1:
            return tensors[0]
    raise TypeError("embedding file must contain one tensor or a state dict with a weight tensor")


def _replace_embedding_tensor(payload: Any, source_tensor: Any, normalized_tensor: Any) -> Any:
    if payload is source_tensor:
        return normalized_tensor
    if isinstance(payload, Mapping):
        result = dict(payload)
        matches = [key for key, value in payload.items() if value is source_tensor]
        if len(matches) != 1:
            raise TypeError("embedding state dict must contain exactly one selected weight tensor")
        result[matches[0]] = normalized_tensor
        return result
    raise TypeError("embedding file must contain one tensor or a state dict with a weight tensor")
