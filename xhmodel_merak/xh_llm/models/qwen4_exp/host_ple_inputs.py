"""Prepare external PLE graph inputs without placing lookup operators in HMONNX."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import torch

from .ple_demo import compute_ngram_ids, dequantize_ple_rows, gather_ple_rows, load_tables


class HostPLEInputs:
    """Memory-map compressed tables and assemble only the current request's rows."""

    def __init__(self, graph_path: str | Path, manifest: dict):
        self.root = Path(graph_path).resolve().parent
        if manifest.get("version") != 1 or not isinstance(manifest.get("layers"), list):
            raise ValueError("Unsupported Qwen4 external PLE manifest")
        self.manifest = manifest
        self.tables = [load_tables(self.root, spec) for spec in manifest["layers"]]
        self.last_indices: dict[str, np.ndarray] = {}

    @classmethod
    def from_graph(cls, graph_path: str | Path) -> "HostPLEInputs | None":
        graph = onnx.load(str(graph_path), load_external_data=False)
        metadata = {item.key: item.value for item in graph.metadata_props}
        if "qwen4_exp_ple" not in metadata:
            return None
        manifest = json.loads(metadata["qwen4_exp_ple"])
        names = [spec["input_name"] for spec in manifest["layers"]]
        if len(names) != len(set(names)) or (names and [i.name for i in graph.graph.input][-len(names):] != names):
            raise ValueError("External PLE inputs must appear once at the end of the graph input list")
        return cls(graph_path, manifest)

    def prepare(self, token_ids, state, current_input_length: int, *, device):
        ids = token_ids.detach().cpu().numpy()
        result = []
        self.last_indices = {}
        for spec, (packed, exponent) in zip(self.manifest["layers"], self.tables, strict=True):
            history = state[f"ple_ngram_history_{spec['layer_idx']}"].detach().cpu().numpy()
            indices = compute_ngram_ids(ids, history, spec, current_input_length)
            # These are host operations. The graph receives only their result.
            selected, selected_exp = gather_ple_rows(packed, exponent, indices)
            embedding = dequantize_ple_rows(
                selected, selected_exp, spec["embedding_dim"], spec["output_dtype"],
            )
            dtype = torch.bfloat16 if spec["output_dtype"] == "bfloat16" else torch.float16
            result.append(torch.from_numpy(embedding).to(device=device, dtype=dtype))
            self.last_indices[spec["input_name"]] = indices
        return tuple(result)
