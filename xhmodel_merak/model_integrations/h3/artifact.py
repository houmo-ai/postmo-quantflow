# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
from xhmodel_merak.model_integrations.sol_h3.artifact import validate_precision as validate_precision


def validate_runtime(root, components, *, resolve_file=None):
    from xhmodel_merak.model_build.packaging import validate_onnx_external_locations
    from xhmodel_merak.xh_other_model.models.sol_h3.native_artifacts import export_entries, read_metadata

    if tuple(components) != ("video",):
        raise ValueError("Native H3 requires the complete audio/video pipeline")
    meta = read_metadata(root)
    entries = export_entries(meta)
    if resolve_file is not None:
        for graph in meta["graphs"].values():
            resolve_file(graph["path"], "graph")
        for path in (*meta["resources"].values(), *meta["tokenizer_files"], *meta.get("processor_files", [])):
            resolve_file(path, "resource")
        for path in validate_onnx_external_locations(root, graphs=[root / g["path"] for g in meta["graphs"].values()]):
            resolve_file(path.relative_to(root).as_posix(), "graph.external_data")
    return meta, entries
