"""Shared complete-bundle validation for build and inference."""

from .plugin import COMPONENTS


def validate_runtime(root, components, *, resolve_file=None):
    from xhmodel_merak.xh_other_model.models.flux3_klein.bundle import export_entries, read_metadata

    if tuple(components) != COMPONENTS:
        raise ValueError("FLUX 3 requires the complete image component")
    meta = read_metadata(root)
    entries = export_entries(meta)
    if resolve_file is not None:
        from xhmodel_merak.model_build.packaging import validate_onnx_external_locations

        for name, entry in entries.items():
            if entry["type"] != "tokenizer":
                resolve_file(entry["path"], name)
        for path in meta["processor_files"]:
            resolve_file(path, "processor_files")
        graphs = [
            root / entry["path"] for entry in entries.values() if entry["type"] in ("hmonnx", "prefill", "decode")
        ]
        for path in validate_onnx_external_locations(root, graphs=graphs):
            resolve_file(path.relative_to(root).as_posix(), "graph.external_data")
        for path in meta.get("prompt_upsampler", {}).get("files", []):
            resolve_file(path, "prompt_upsampler.files")
    return meta, entries
