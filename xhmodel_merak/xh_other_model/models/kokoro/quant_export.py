from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from .assets import sha256
from .graph import GraphArtifact
from .host import save_voice_pack_numpy


XHQUANT_TORCH_ONNX_INTERNAL_OPTIMIZE_ENV = "XHQUANT_TORCH_ONNX_INTERNAL_OPTIMIZE"


def _convert_hmonnx(
    artifact: GraphArtifact,
    feed: Mapping[str, Any],
    output_path: Path,
    *,
    target: str,
    quant_type: str,
    debug: bool,
) -> None:
    import torch

    from xhquant.api import (
        DeviceType,
        QuantScheme,
        convert_onnx_to_hmonnx,
        create_quant_config,
        xhquant_init,
    )

    try:
        device_type = getattr(DeviceType, target)
    except AttributeError as error:
        raise ValueError(f"unsupported xhquant target device: {target}") from error
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xhquant_init(str(output_path.parent / f"{output_path.stem}.log"), debug)
    quant_config = create_quant_config(QuantScheme(target_device=device_type, quant_type=quant_type))
    inputs = []
    for name in artifact.input_names:
        value = feed[name]
        if torch.is_tensor(value):
            inputs.append(value.detach().cpu())
        else:
            inputs.append(torch.from_numpy(np.asarray(value)))
    with _xhquant_torch_onnx_optimize_context(True):
        convert_onnx_to_hmonnx(
            str(artifact.path),
            inputs,
            device_type,
            str(output_path),
            quant_config=quant_config,
            input_names=list(artifact.input_names),
            output_names=list(artifact.output_names),
            # Use xhquant's normal frontend constant folding so static Slice and
            # Reshape parameters become graph constants.  No Kokoro-specific ONNX
            # optimization is applied before this conversion.
            simplify=True,
            decompose_lstm=True,
        )


@contextmanager
def _xhquant_torch_onnx_optimize_context(enabled: bool):
    """Scope exporter normalization without leaking environment state."""

    previous = os.environ.get(XHQUANT_TORCH_ONNX_INTERNAL_OPTIMIZE_ENV)
    os.environ[XHQUANT_TORCH_ONNX_INTERNAL_OPTIMIZE_ENV] = "1" if enabled else "0"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(XHQUANT_TORCH_ONNX_INTERNAL_OPTIMIZE_ENV, None)
        else:
            os.environ[XHQUANT_TORCH_ONNX_INTERNAL_OPTIMIZE_ENV] = previous


def _hmonnx_artifact_metadata(path: Path, work_dir: Path) -> dict[str, Any]:
    import onnx

    proto = onnx.load(str(path), load_external_data=False)
    op_counts: dict[str, int] = {}
    domain_counts: dict[str, int] = {}
    for node in proto.graph.node:
        if node.op_type == "LSTM":
            raise RuntimeError(f"target HMONNX still contains LSTM: {path}: {node.name}")
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
        domain = node.domain or "ai.onnx"
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    return {
        "hmonnx_file": str(path.relative_to(work_dir)),
        "hmonnx_sha256": sha256(path),
        "hmonnx_size_bytes": path.stat().st_size,
        "hmonnx_external_data": _external_data_metadata(proto, path, work_dir),
        "hmonnx_node_count": len(proto.graph.node),
        "hmonnx_op_counts": dict(sorted(op_counts.items())),
        "hmonnx_domain_counts": dict(sorted(domain_counts.items())),
    }


def _external_data_metadata(
    proto: Any,
    model_path: Path,
    work_dir: Path,
) -> list[dict[str, Any]]:
    import onnx

    locations: set[str] = set()
    for initializer in proto.graph.initializer:
        if initializer.data_location != onnx.TensorProto.EXTERNAL:
            continue
        fields = {item.key: item.value for item in initializer.external_data}
        location = fields.get("location")
        if location:
            locations.add(location)
    result: list[dict[str, Any]] = []
    for location in sorted(locations):
        external_path = model_path.parent / location
        if not external_path.is_file():
            raise FileNotFoundError(f"HMONNX external data referenced by {model_path} is missing: {external_path}")
        result.append(
            {
                "file": str(external_path.relative_to(work_dir)),
                "sha256": sha256(external_path),
                "size_bytes": external_path.stat().st_size,
            }
        )
    return result


def _export_runtime_assets(voice_path: str | Path, work_dir: Path) -> dict[str, Any]:
    """Export Host-only constants required by the self-contained demo."""

    source = Path(voice_path).expanduser().resolve()
    destination = work_dir / "assets" / "voices" / f"{source.stem}.npy"
    save_voice_pack_numpy(source, destination)
    voice = np.load(destination, mmap_mode="r", allow_pickle=False)
    return {
        "voice_pack": {
            "name": source.stem,
            "file": str(destination.relative_to(work_dir)),
            "sha256": sha256(destination),
            "dtype": str(voice.dtype),
            "shape": list(voice.shape),
            "size_bytes": destination.stat().st_size,
            "format": "numpy_npy_allow_pickle_false",
        }
    }
