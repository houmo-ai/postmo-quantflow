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

import torch


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
                if phase in self.components or (
                    (phase == "decode" or _is_verify_phase(phase)) and "mtp" in self.components
                )
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

    entries, copied = {}, set()
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
                relative = Path(f"layer_{spec['layer_idx']}.{kind}.bin")
                if relative not in copied:
                    _link_or_copy(source, root / "ple" / relative)
                    copied.add(relative)
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


@torch.no_grad()
def run_qwen4_exp_hmonnx_golden(
    hmonnx_path: str | Path,
    sample_inputs: tuple[torch.Tensor, ...],
    golden_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    step: int = 0,
) -> dict[str, torch.Tensor]:
    """Execute one cached decode turn and persist full HMONNX Golden tensors."""

    from .xhquant_extensions import qwen4_exp_xhquant_extension_scope

    with qwen4_exp_xhquant_extension_scope():
        from xhquant.api import HMONNXGoldenInference
        from xhquant.core.cache_tensor import CacheTensor

        session = HMONNXGoldenInference(str(hmonnx_path))
        session.exec_device = torch.device(device)
        session.save_golden = True
        session.golden_dir = str(golden_dir)
        session.step = int(step)
        session.initialize()
        input_names = session.get_input_names()
        from .host_ple_inputs import HostPLEInputs
        from .qsa_host_controls import QSA_HOST_CONTROL_NAMES, build_qsa_host_controls, qsa_host_control_spec

        host_ple = HostPLEInputs.from_graph(hmonnx_path) if Path(hmonnx_path).is_file() else None
        host_names = [spec["input_name"] for spec in host_ple.manifest["layers"]] if host_ple else []
        base_names = [name for name in input_names if name not in host_names]
        if len(sample_inputs) == len(input_names):
            values = dict(zip(input_names, sample_inputs, strict=True))
        elif host_ple and len(sample_inputs) == len(base_names):
            values = dict(zip(base_names, sample_inputs, strict=True))
        elif (qsa_host_control_spec(session) is not None
              and len(sample_inputs) == len(base_names) - len(QSA_HOST_CONTROL_NAMES)):
            names = [name for name in base_names if name not in QSA_HOST_CONTROL_NAMES]
            values = dict(zip(names, sample_inputs, strict=True))
        else:
            raise ValueError(
                f"Qwen4-Exp HMONNX input count mismatch: expected {len(input_names)}, got {len(sample_inputs)}"
            )
        qsa_spec = qsa_host_control_spec(session)
        if qsa_spec is not None and "qsa_attention_mask" not in values:
            topk, mask_dtype = qsa_spec
            pool_cache = next(value for name, value in values.items() if name.startswith("qsa_indexer_key_cache_"))
            controls = build_qsa_host_controls(
                int(values["past_seq_length"].item()), int(values["current_input_length"].item()),
                int(values["inputs_embeds"].shape[1]), int(pool_cache.shape[2]) * 4,
                topk, device=device, dtype=mask_dtype,
            )
            values.update(zip(QSA_HOST_CONTROL_NAMES, controls, strict=True))
        if host_ple:
            embeddings = host_ple.prepare(
                values["ple_input_ids"], values, int(values["current_input_length"].item()), device=device,
            )
            for name, embedding in zip(host_names, embeddings, strict=True):
                if name in values and not torch.equal(values[name].to(embedding), embedding):
                    raise ValueError(f"Provided PLE embeddings disagree with host lookup: {name}")
                values[name] = embedding
        input_feed = {
            name: (
                values[name].to(session.get_input(name).dtype)
                if isinstance(values[name], CacheTensor)
                else values[name].detach().to(session.get_input(name).dtype)
            )
            for name in input_names
        }
        if host_ple and any((Path(golden_dir) / f"step_{step}" / f"{name}.npy").exists() for name in host_names):
            raise FileExistsError("External PLE Golden step already exists")
        outputs = session.run(input_feed)
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)
        output_map = dict(zip(session.get_output_names(), outputs, strict=True))
    if host_ple:
        import numpy as np

        from .golden_runtime import copy_ple_resources, record_ple_sample

        root = Path(golden_dir)
        manifest_path = root / "ple/manifest.json"
        if manifest_path.exists():
            entry = {"path": "ple/manifest.json", "manifest": json.loads(manifest_path.read_text())}
            saved_layers = entry["manifest"]["layers"]
            if len(saved_layers) != len(host_ple.manifest["layers"]):
                raise ValueError("Existing Golden PLE layer contract differs")
            for saved, original in zip(saved_layers, host_ple.manifest["layers"], strict=True):
                if ({k: v for k, v in saved.items() if k not in {"packed", "exponent"}}
                        != {k: v for k, v in original.items() if k not in {"packed", "exponent"}}):
                    raise ValueError("Existing Golden PLE hash contract differs")
                for kind in ("packed", "exponent"):
                    archived = (root / "ple" / saved[kind]["path"]).resolve()
                    original_path = host_ple.root / original[kind]["path"]
                    if not archived.is_relative_to((root / "ple").resolve()):
                        raise ValueError("Unsafe existing Golden PLE asset path")
                    if not os.path.samefile(archived, original_path):
                        if archived.stat().st_size != original_path.stat().st_size:
                            raise ValueError("Existing Golden PLE payload size differs")
                        with archived.open("rb") as saved_file, original_path.open("rb") as source_file:
                            while block := source_file.read(16 * 1024 * 1024):
                                if saved_file.read(len(block)) != block:
                                    raise ValueError("Existing Golden PLE payload bytes differ")
        else:
            entry = copy_ple_resources(root, {"decode": host_ple})["decode"]
        destination = root / f"step_{step}"
        for name in host_names:
            value = input_feed[name].detach().cpu()
            if value.dtype == torch.bfloat16:
                value = value.float()
            np.save(destination / f"{name}.npy", value.numpy(), allow_pickle=False)
        record_ple_sample(root, entry, "decode", int(step), destination, host_ple.last_indices)
    summary_path = Path(golden_dir) / f"step_{step}" / "qwen4_exp_runtime_summary.json"
    _write_json(
        summary_path,
        {
            "hmonnx_path": str(Path(hmonnx_path).resolve()),
            "inputs": [_tensor_manifest(name, value) for name, value in input_feed.items()],
            "outputs": [_tensor_manifest(name, value) for name, value in output_map.items()],
        },
    )
    return output_map

def _tensor_manifest(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
    }

def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(data), indent=2, sort_keys=True) + "\n", encoding="utf-8")



@torch.no_grad()
def _validate_hmonnx_runtime_parity(
    hmonnx_path: str | Path,
    sample_inputs: tuple[torch.Tensor, ...],
    reference_model: torch.nn.Module,
    *,
    device: str | torch.device,
    reference_device: str | torch.device,
) -> list[dict[str, Any]]:
    """Prove that the serialized HMONNX graph matches its XH2 export graph."""

    from xhquant.api import HMONNXGoldenInference
    from xhquant.core.cache_tensor import CacheTensor

    session = HMONNXGoldenInference(str(hmonnx_path))
    session.exec_device = torch.device(device)
    session.save_golden = False
    session.initialize()
    input_names = session.get_input_names()
    if len(input_names) != len(sample_inputs):
        raise RuntimeError(
            f"Qwen4-Exp HMONNX parity input count mismatch: expected {len(input_names)}, got {len(sample_inputs)}"
        )
    runtime_feed = {
        name: (
            CacheTensor(value.detach().clone()).to(device=session.exec_device, dtype=session.get_input(name).dtype)
            if isinstance(value, CacheTensor)
            else value.detach().to(device=session.exec_device, dtype=session.get_input(name).dtype)
        )
        for name, value in zip(input_names, sample_inputs, strict=True)
    }
    runtime_outputs = session.run(runtime_feed)
    if not isinstance(runtime_outputs, (tuple, list)):
        runtime_outputs = (runtime_outputs,)
    reference_inputs = tuple(
        CacheTensor(value.detach().clone()).to(device=reference_device, dtype=session.get_input(name).dtype)
        if isinstance(value, CacheTensor)
        else value.detach().to(device=reference_device, dtype=session.get_input(name).dtype)
        for name, value in zip(input_names, sample_inputs, strict=True)
    )
    reference_outputs = reference_model(*reference_inputs)
    if not isinstance(reference_outputs, (tuple, list)):
        reference_outputs = (reference_outputs,)
    output_names = session.get_output_names()
    if len(runtime_outputs) != len(reference_outputs) or len(output_names) != len(runtime_outputs):
        raise RuntimeError(
            "Qwen4-Exp HMONNX parity output count mismatch: "
            f"runtime={len(runtime_outputs)}, reference={len(reference_outputs)}, names={len(output_names)}"
        )

    parity: list[dict[str, Any]] = []
    for name, actual, expected in zip(output_names, runtime_outputs, reference_outputs, strict=True):
        if actual.shape != expected.shape:
            raise RuntimeError(
                f"Qwen4-Exp HMONNX output {name!r} shape mismatch: "
                f"runtime={tuple(actual.shape)}, reference={tuple(expected.shape)}"
            )
        if actual.is_floating_point() or expected.is_floating_point():
            actual_float = actual.float()
            expected_float = expected.float().to(actual_float.device)
            if not torch.isfinite(actual_float).all() or not torch.isfinite(expected_float).all():
                raise RuntimeError(f"Qwen4-Exp HMONNX output {name!r} contains non-finite values")
            max_abs_error = float((actual_float - expected_float).abs().max()) if actual.numel() else 0.0
            torch.testing.assert_close(actual_float, expected_float, atol=1e-3, rtol=1e-3)
        else:
            expected = expected.to(actual.device)
            max_abs_error = (
                float((actual.to(torch.int64) - expected.to(torch.int64)).abs().max()) if actual.numel() else 0.0
            )
            torch.testing.assert_close(actual, expected)
        parity.append(
            {
                "name": name,
                "shape": list(actual.shape),
                "dtype": str(actual.dtype).removeprefix("torch."),
                "max_abs_error": max_abs_error,
            }
        )
    return parity
