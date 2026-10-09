"""One build/load contract for every graph and runtime resource."""

from .plugin import COMPONENTS


def validate_runtime(root, components, *, resolve_file=None):
    from xhmodel_merak.xh_other_model.models.qwen_image21.artifacts import read_dit_metadata
    from xhmodel_merak.xh_other_model.models.qwen_image21.build import export_entries
    from xhmodel_merak.xh_other_model.models.qwen_image21.bundle import read_bundle_metadata

    if tuple(components) != COMPONENTS:
        raise ValueError("Qwen-Image requires the complete image component")
    meta = read_bundle_metadata(root)
    _, gear = read_dit_metadata(root, meta["dit_metadata"])
    reference_tokens = meta["vae_encode"]["token_capacity"]
    if gear.prefix_tokens <= reference_tokens or gear.image_tokens < reference_tokens:
        raise ValueError("DiT capacities must cover the configured reference size")
    if meta["feature"]["capacity"] != gear.encoder_tokens:
        raise ValueError("feature and DiT encoder capacities must agree")
    entries = export_entries(root, meta)
    if resolve_file is not None:
        from xhmodel_merak.model_build.packaging import validate_onnx_external_locations

        for name, entry in entries.items():
            if entry["type"] != "tokenizer":
                resolve_file(entry["path"], name)
        for path in meta["processor_files"]:
            resolve_file(path, "processor_files")
        graphs = [root / entry["path"] for entry in entries.values() if entry["type"] == "hmonnx"]
        for path in validate_onnx_external_locations(root, graphs=graphs):
            resolve_file(path.relative_to(root).as_posix(), "graph.external_data")
    return meta, entries
