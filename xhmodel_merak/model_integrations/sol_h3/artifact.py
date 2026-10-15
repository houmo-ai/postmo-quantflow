# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
from .plugin import COMPONENTS


def validate_precision(meta, *, deployment, checkpoint_weight):
    """Keep publication and loading on the same component precision contract."""
    if deployment != {"video": meta["quant_type"]}:
        raise ValueError("Sol-H3 graph precision disagrees with the declared deployment precision")
    feature_precision = "w4a8h0_ssfp" if checkpoint_weight == "w4" else meta["quant_type"]
    if any(
        meta["graphs"][name].get("quant_type", meta["quant_type"]) != feature_precision
        for name in meta["feature_graphs"]
    ):
        raise ValueError("Sol-H3 Qwen feature precision disagrees with the declared checkpoint precision")
    for name, graph in meta["graphs"].items():
        if name not in meta["feature_graphs"]:
            expected = "w16a16_sefp" if name == "ltx_image_encode" else meta["quant_type"]
            if graph.get("quant_type", meta["quant_type"]) != expected:
                raise ValueError(f"Sol-H3 {name} precision disagrees with its component contract")


def validate_runtime(root, components, *, resolve_file=None):
    from xhmodel_merak.xh_other_model.models.sol_h3.artifacts import export_entries, read_metadata

    if tuple(components) != COMPONENTS:
        raise ValueError("Sol-H3 requires the complete video pipeline")
    meta = read_metadata(root)
    entries = export_entries(meta)
    if resolve_file is not None:
        from xhmodel_merak.model_build.packaging import validate_onnx_external_locations

        for name, entry in entries.items():
            if entry["type"] != "tokenizer":
                resolve_file(entry["path"], name)
        for value in meta["resources"].values():
            resolve_file(value, "resources")
        for value in meta["tokenizer_files"]:
            resolve_file(value, "tokenizer_files")
        graphs = [root / entry["path"] for entry in entries.values() if entry["type"] == "hmonnx"]
        for value in validate_onnx_external_locations(root, graphs=graphs):
            resolve_file(value.relative_to(root).as_posix(), "graph.external_data")
    return meta, entries
