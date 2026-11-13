"""Bounded checkpoint reads for Host PLE inputs during streamed export."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import torch
from torch import nn


class HostPLEInput(nn.Module):
    """Consume a Host lookup result while retaining the graph's history update."""

    def __init__(self, layer_idx: int, context_len: int):
        super().__init__()
        from xhquant.nn import DynamicSlice

        self.layer_idx = layer_idx
        self.history_slice = DynamicSlice([context_len], [1], [1])
        self.embeddings = None

    def forward(self, input_ids, past_key_values):
        supplied = getattr(past_key_values, "_streamed_ple_embeddings", {})
        embeddings = supplied.get(self.layer_idx, self.embeddings)
        if embeddings is None:
            raise ValueError("Streamed PLE requires an explicit Host embedding input")
        cache = past_key_values.layers[self.layer_idx]
        history = torch.cat((cache.conv_states[2], input_ids.to(torch.int32)), dim=-1)
        cache.conv_states[2] = self.history_slice(history, past_key_values.current_input_length)
        return embeddings.flatten(-2)


class PLEExportResources:
    """One package-owned set of immutable PLE tables, shared by every phase."""

    def __init__(self, root, *, resume=False):
        self.root = Path(root)
        self.directory = self.root / "ple"
        self.manifest_path = self.directory / "manifest.json"
        manifest = json.loads(self.manifest_path.read_text()) if resume and self.manifest_path.exists() else {}
        self.layers = {spec["layer_idx"]: spec for spec in manifest.get("layers", [])}

    def publish(self):
        if self.layers:
            self.directory.mkdir(parents=True, exist_ok=True)
            self.manifest_path.write_text(json.dumps(
                {"version": 1, "layers": [self.layers[key] for key in sorted(self.layers)]}, indent=2,
            ) + "\n")


def streamed_ple_tensor_names(model, prefix):
    """Identify only the actual native PLE tables, before loading a layer."""
    return tuple(
        f"{prefix}.{path}.ngram_embedding.weight"
        for path, module in model.named_modules()
        if type(module).__name__ == "Qwen4ExpTextNGramEmbedding"
    )


def prepare_streamed_ple(loader, layer, layer_prefix, output_dir, *, quant_device="cpu", chunk_rows=4096,
                         resources=None):
    """Encode checkpoint rows while the complete lookup table remains on meta."""
    for native in layer.modules():
        if type(native).__name__ == "Qwen4ExpTextNGramEmbedding" and not native.ngram_embedding.weight.is_meta:
            raise ValueError("Streamed PLE table must stay on meta until replaced by a Host input")
    return _prepare_ple(
        layer, layer_prefix, output_dir, dtype=loader.dtype, tensor_info=loader.tensor_info,
        iter_rows=lambda key, count: loader.iter_tensor_rows(key, chunk_rows=count),
        quant_device=quant_device, chunk_rows=chunk_rows,
        resources=resources,
    )


def prepare_native_ple(model, output_dir, *, quant_device="cpu", chunk_rows=4096, resources=None):
    """Use the same Host PLE encoding before ordinary frontend capture."""
    tables = {name: value.detach() for name, value in model.named_parameters()
              if name.endswith("ngram_embedding.weight")}
    if not tables:
        return []
    dtype = next(iter(tables.values())).dtype
    return _prepare_ple(
        model, "", output_dir, dtype=dtype,
        tensor_info=lambda key: {
            "shape": tuple(tables[key].shape),
            "dtype": {torch.float16: "F16", torch.bfloat16: "BF16", torch.float32: "F32"}[tables[key].dtype],
        },
        iter_rows=lambda key, count: tables[key].split(count),
        quant_device=quant_device, chunk_rows=chunk_rows,
        resources=resources,
    )


def _prepare_ple(layer, layer_prefix, output_dir, *, dtype, tensor_info, iter_rows, quant_device, chunk_rows,
                 resources=None):
    """Encode bounded native/checkpoint rows and replace tables with Host inputs."""
    from .ple_export import _validate_hash_spec, extract_qwen4_exp_ple_specs
    from .ple_sefp4 import quantize_ple_sefp4_rows

    if isinstance(chunk_rows, bool) or not isinstance(chunk_rows, int) or chunk_rows < 1:
        raise ValueError("PLE chunk_rows must be a positive integer")
    if dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("Host PLE inputs require a float16 or bfloat16 export dtype")
    resources = resources if resources is not None else PLEExportResources(output_dir)
    root = resources.directory
    root.mkdir(parents=True, exist_ok=True)
    specs_by_layer = {s["layer_idx"]: s for s in extract_qwen4_exp_ple_specs(layer)}
    prepared = []
    created = []
    replacements = []
    try:
        for path, native in tuple(layer.named_modules()):
            if type(native).__name__ != "Qwen4ExpTextNGramEmbedding":
                continue
            if native.layer_idx in resources.layers:
                prepared.append(resources.layers[native.layer_idx])
                replacements.append((path, HostPLEInput(int(native.layer_idx), int(native.context_len))))
                continue
            key = ".".join(part for part in (layer_prefix, path, "ngram_embedding.weight") if part)
            info = tensor_info(key)
            shape = tuple(info["shape"])
            if len(shape) != 2 or min(shape) < 1 or info["dtype"] not in {"F16", "BF16", "F32"}:
                raise ValueError(f"Invalid floating-point PLE checkpoint table: {key}")
            rows, width = shape
            spec = dict(specs_by_layer[int(native.layer_idx)])
            _validate_hash_spec(spec, rows)
            token = uuid.uuid4().hex
            packed_path = root / f"ple{native.layer_idx}.{token}.packed.bin"
            exp_path = root / f"ple{native.layer_idx}.{token}.exponent.bin"
            groups = (width + 63) // 64
            count = 0
            with packed_path.open("xb") as packed_file:
                created.append(packed_path)
                with exp_path.open("xb") as exp_file:
                    created.append(exp_path)
                    for chunk in iter_rows(key, chunk_rows):
                        if chunk.ndim != 2 or tuple(chunk.shape[1:]) != (width,) or not 0 < len(chunk) <= chunk_rows:
                            raise ValueError(f"Invalid PLE row chunk for {key}")
                        # Match the export loader's execution dtype before SEFP4 encoding.
                        chunk = chunk.to(device=quant_device, dtype=dtype)
                        packed, exponent = quantize_ple_sefp4_rows(chunk)
                        packed_file.write(packed.detach().cpu().numpy().tobytes())
                        exp_file.write(exponent.detach().cpu().numpy().astype("<i2", copy=False).tobytes())
                        count += len(chunk)
                    if count != rows:
                        raise ValueError(f"Incomplete PLE checkpoint read for {key}: {count} != {rows}")
                    for stream in (packed_file, exp_file):
                        stream.flush()
                        os.fsync(stream.fileno())
            spec.update(
                input_name=f"ple_embeddings_{native.layer_idx}", embedding_dim=width,
                output_dtype=str(dtype).removeprefix("torch."),
                packed={"path": packed_path.relative_to(resources.root).as_posix(),
                        "shape": [rows, groups, 32], "dtype": "int8"},
                exponent={"path": exp_path.relative_to(resources.root).as_posix(),
                          "shape": [rows, groups, 1], "dtype": "int16"},
            )
            prepared.append(spec)
            resources.layers[int(native.layer_idx)] = spec
            replacements.append((path, HostPLEInput(int(native.layer_idx), int(native.context_len))))
        # Change module ownership only after every real sidecar is complete.
        for path, replacement in replacements:
            layer.set_submodule(path, replacement)
        resources.publish()
        return prepared
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise


def set_streamed_ple_inputs(layer, specs, values, *, cache=None):
    """Bind explicit part-graph inputs; no lookup table is retained in this module."""
    if len(specs) != len(values):
        raise ValueError("Streamed PLE input count does not match the resource contract")
    inputs = {int(spec["layer_idx"]): value for spec, value in zip(specs, values, strict=True)}
    if cache is not None:
        # Cache is created inside the captured forward. Keeping tensor bindings
        # there keeps streamed inputs explicit in the static wrapper boundary.
        cache._streamed_ple_embeddings = inputs
        return
    for module in layer.modules():
        if isinstance(module, HostPLEInput):
            module.embeddings = inputs[module.layer_idx]


__all__ = ["HostPLEInput", "prepare_streamed_ple", "set_streamed_ple_inputs", "streamed_ple_tensor_names"]
