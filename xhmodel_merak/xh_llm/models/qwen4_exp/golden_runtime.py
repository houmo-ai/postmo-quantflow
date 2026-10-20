"""Capture real SDK execution, including externally owned PageKV state."""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def _is_verify_phase(phase: str) -> bool:
    return re.fullmatch(r"decode_m(?:[2-9]|[1-9][0-9]+)", phase) is not None


class GoldenCapture:
    """A request-scoped SDK Golden recorder; existing captures are never overwritten.

    Wrap the actual component invocation in ``step``. The SDK writes graph inputs,
    intermediate tensors and outputs. External mutable state is saved before and
    after that invocation because it is absent from the ONNX argument list.
    ``finish`` refuses to report completion if a requested component never ran.
    """

    def __init__(self, output_dir: str | Path, components: Sequence[str]):
        self.root = Path(output_dir).resolve()
        self.components = tuple(dict.fromkeys(components))
        if not self.components or set(self.components) - {"prefill", "decode", "visual", "mtp"}:
            raise ValueError("Golden components must select prefill, decode, visual or mtp")
        self.manifest_path = self.root / "qwen4_golden_capture.json"
        if self.manifest_path.exists():
            raise FileExistsError(f"Golden capture already exists: {self.manifest_path}")
        self.root.mkdir(parents=True, exist_ok=True)
        self.steps: list[dict[str, Any]] = []
        self._counts: dict[str, int] = {}
        self._failed = False
        self.required_graphs: set[str] = set()
        self._ple_manifests: dict[str, dict] = {}
        self._write("running")

    def add_ple_resources(self, phases: Mapping[str, Any]) -> None:
        """Deliver original packed bytes and a NumPy-only lookup demonstration."""
        if self._ple_manifests:
            raise ValueError("External PLE resources have already been registered")
        self._ple_manifests = copy_ple_resources(
            self.root, {
                phase: host for phase, host in phases.items()
                if phase in self.components or (_is_verify_phase(phase) and "mtp" in self.components)
            },
        )
        if self._ple_manifests:
            self._write("running")

    def _write(self, status: str, error: str | None = None) -> None:
        payload = {
            "format": "qwen4_exp_sdk_golden_v1",
            "status": status,
            "requested_components": list(self.components),
            "steps": self.steps,
        }
        if error is not None:
            payload["error"] = error
        if self._ple_manifests:
            payload["external_ple"] = {
                "demo": "ple_demo.py",
                "manifests": {phase: item["path"] for phase, item in self._ple_manifests.items()},
            }
        self.manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @contextmanager
    def step(self, component: str, session: Any, *, graph_name: str | None = None,
             runtime_state: Mapping[str, Any] | None = None,
             ple_indices: Mapping[str, Any] | None = None):
        if component not in self.components:
            yield
            return
        name = graph_name or component
        if not name or Path(name).name != name or name in {".", ".."} or "\\" in name:
            raise ValueError(f"Unsafe Golden graph name: {name!r}")
        index = self._counts.get(name, 0)
        destination = self.root / name / f"step_{index}"
        if destination.exists():
            raise FileExistsError(f"Golden step already exists: {destination}")
        destination.mkdir(parents=True)
        interpreter = session.interpreter
        if not hasattr(interpreter, "_enable_golden"):
            raise ValueError("SDK Golden requires the default or auto-offload interpreter; disable CUDA Graph")
        old = (session.save_golden, session.save_golden_dir, session.step, interpreter._enable_golden)
        # V2 owner.save_golden alone does not activate the interpreter hooks.
        interpreter._enable_golden = True
        session.save_golden_dir = str(destination.parent)
        session.step = index
        session.save_golden = True
        try:
            before = _save_state(destination, "runtime_before", runtime_state or {})
            yield
            after = _save_state(destination, "runtime_after", runtime_state or {})
            inputs = _inventory(destination, session.get_input_names(), inputs=True)
            outputs = _inventory(destination, session.get_output_names(), inputs=False)
            if ple_indices:
                ple_phase = name if name in self._ple_manifests else component
                if ple_phase not in self._ple_manifests:
                    raise ValueError("Register compressed PLE resources before capturing external PLE inputs")
                record_ple_sample(
                    self.root, self._ple_manifests[ple_phase], ple_phase, index, destination, ple_indices,
                )
            self.steps.append({
                "component": component,
                "graph": name,
                "step": index,
                "directory": destination.relative_to(self.root).as_posix(),
                "inputs": inputs,
                "outputs": outputs,
                "runtime_before": before,
                "runtime_after": after,
            })
            self._counts[name] = index + 1
            self._write("running")
        except BaseException as exc:
            self._failed = True
            self._write("failed", str(exc))
            raise
        finally:
            session.save_golden, session.save_golden_dir, session.step, interpreter._enable_golden = old

    def finish(self) -> dict[str, tuple[str, ...]]:
        if self._failed:
            raise ValueError("Golden capture contains a failed execution")
        missing = set(self.components) - {step["component"] for step in self.steps}
        missing_graphs = self.required_graphs - {step["graph"] for step in self.steps}
        if missing or missing_graphs:
            missing.update(missing_graphs)
            message = f"Requested Golden components did not execute: {sorted(missing)}"
            self._write("failed", message)
            raise ValueError(message)
        self._write("complete")
        return {
            component: tuple(str(self.root / step["directory"]) for step in self.steps
                             if step["component"] == component)
            for component in self.components
        }


def copy_ple_resources(root: Path, phases: Mapping[str, Any]) -> dict:
    """Package unmodified compressed rows for both serving and direct Golden."""
    from . import ple_demo

    entries = {}
    for phase, host in phases.items():
        if host is None:
            continue
        if phase not in {"prefill", "decode"} and not _is_verify_phase(phase):
            raise ValueError(f"Unsupported external PLE phase: {phase}")
        manifest = copy.deepcopy(host.manifest)
        manifest["samples"] = []
        for spec in manifest["layers"]:
            for kind in ("packed", "exponent"):
                source = host.root / spec[kind]["path"]
                relative = Path(phase) / f"layer_{spec['layer_idx']}.{kind}.bin"
                _link_or_copy(source, root / "ple" / relative)
                spec[kind]["path"] = relative.as_posix()
        filename = "manifest.json" if not entries else f"{phase}_manifest.json"
        entries[phase] = {"path": f"ple/{filename}", "manifest": manifest}
        (root / entries[phase]["path"]).write_text(json.dumps(manifest, indent=2) + "\n")
    if entries:
        shutil.copyfile(ple_demo.__file__, root / "ple_demo.py")
    return entries


def record_ple_sample(root: Path, entry: dict, component: str, index: int, destination: Path, indices) -> None:
    """Link actual SDK graph inputs to their compressed-lookup replay samples."""
    import numpy as np

    manifest = entry["manifest"]
    for spec in manifest["layers"]:
        input_name = spec["input_name"]
        stem = f"{component}_{index}_{input_name}"
        indices_file = Path("samples") / f"{stem}.indices.npy"
        expected_file = Path("samples") / f"{stem}.embedding.npy"
        (root / "ple" / indices_file).parent.mkdir(exist_ok=True)
        np.save(root / "ple" / indices_file, indices[input_name], allow_pickle=False)
        _link_or_copy(destination / f"{input_name}.npy", root / "ple" / expected_file)
        manifest["samples"].append({
            "layer_idx": spec["layer_idx"], "indices": indices_file.as_posix(),
            "expected": expected_file.as_posix(), "component": component, "step": index,
        })
    (root / entry["path"]).write_text(json.dumps(manifest, indent=2) + "\n")


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite Golden resource: {destination}")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def _inventory(directory: Path, names: Sequence[str], *, inputs: bool) -> list[dict[str, Any]]:
    import numpy as np

    result = []
    for name in names:
        filename = name if inputs else name.replace("/", "_")
        path = directory / f"{filename}.npy"
        if not path.resolve().is_relative_to(directory.resolve()) or not path.is_file():
            raise ValueError(f"SDK did not capture graph tensor {name!r} in {directory}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        result.append({"name": name, "file": path.relative_to(directory).as_posix(),
                       "shape": list(array.shape), "dtype": str(array.dtype)})
    if not result:
        raise ValueError("Golden graph must have named inputs and outputs")
    return result


def _save_state(directory: Path, prefix: str, values: Mapping[str, Any]) -> list[dict[str, Any]]:
    import numpy as np
    import torch

    result = []
    for index, (name, tensor) in enumerate(values.items()):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Golden runtime state {name!r} must be a tensor")
        # Snapshot before execution: PageKV/GDR operations can mutate the storage.
        array = tensor.detach().cpu().numpy()
        filename = f"{prefix}_{index}.npy"
        np.save(directory / filename, array, allow_pickle=False)
        result.append({"name": name, "file": filename, "shape": list(array.shape), "dtype": str(array.dtype)})
    return result
