"""Remove operator attributes named ``saturate`` from an ONNX model.

Usage: python remove_staure.py model.onnx
The original is moved to model.onnx.bak; the cleaned model keeps the input path.
"""

import argparse
from pathlib import Path

import onnx


def remove_saturate(nodes):
    """Remove attributes in place, including nested graphs; return their count."""
    removed = 0
    for node in nodes:
        for index in range(len(node.attribute) - 1, -1, -1):
            attribute = node.attribute[index]
            if attribute.name == "saturate":
                del node.attribute[index]
                removed += 1
            elif attribute.type == onnx.AttributeProto.GRAPH:
                removed += remove_saturate(attribute.g.node)
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                for graph in attribute.graphs:
                    removed += remove_saturate(graph.node)
    return removed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input ONNX file")
    args = parser.parse_args()
    source = args.input.resolve()
    backup = source.with_name(f"{source.name}.bak")
    if backup.exists() or backup.is_symlink():
        parser.error(f"Backup already exists: {backup}")

    # Only edit graph metadata; large external weights need not be loaded or copied.
    model = onnx.load(source, load_external_data=False)
    removed = remove_saturate(model.graph.node)
    for function in model.functions:
        removed += remove_saturate(function.node)
    for training_info in model.training_info:
        removed += remove_saturate(training_info.initialization.node)
        removed += remove_saturate(training_info.algorithm.node)
    source.rename(backup)
    try:
        onnx.save(model, source)
    except Exception:
        backup.replace(source)
        raise
    print(f"Removed {removed} saturate attribute(s). Saved to {source}; backup: {backup}")


if __name__ == "__main__":
    main()
