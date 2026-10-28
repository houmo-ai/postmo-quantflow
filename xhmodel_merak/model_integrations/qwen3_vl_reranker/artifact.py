"""Validate runtime identity and the score ABI for build and load."""

import json

from xhmodel_merak.model_integrations.autoregressive.artifact import validate_artifact as validate_llm_artifact


def validate_score_contract(root, metadata):
    import onnx

    from xhmodel_merak.model_artifact.files import confined_file

    expected = {
        "version": 1,
        "output": "score_logits",
        "score_index": 0,
        "head_width": 64,
        "activation": "sigmoid",
        "projection": "yes_minus_no",
    }
    if metadata.get("reranker") != expected or metadata.get("decode_hmonnx"):
        raise ValueError("Artifact does not implement the Prefill-only Reranker v1 score contract")
    for field in ("quant_embedding", "prefill_hmonnx"):
        confined_file(root.resolve(), metadata[field], field)
    confined_file(root.resolve(), metadata["visual_config"]["hmonnx"], "visual.hmonnx")
    graph = onnx.load(str(root / metadata["prefill_hmonnx"]), load_external_data=False).graph
    if len(graph.output) != 1 or graph.output[0].name != "score_logits":
        raise ValueError("Reranker graph must expose one score_logits output")
    shape = tuple(dim.dim_value for dim in graph.output[0].type.tensor_type.shape.dim)
    if shape != (1, 1, 64):
        raise ValueError(f"Reranker score_logits ABI must be [1, 1, 64], got {shape}")
    config = metadata.get("model_config", {})
    flash = config.get("flash_attention") or {}
    if flash.get("enable"):
        hf = json.loads((root / metadata["hf_config"] / "config.json").read_text())
        nodes = [node for node in graph.node if node.op_type == "FlashAttention"]
        if len(nodes) != hf["text_config"]["num_hidden_layers"]:
            raise ValueError("Reranker FlashAttention export must include every decoder layer")
        for node in nodes:
            attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
            if any(attrs.get(key) != flash.get(key, 8) for key in ("q_bits", "k_bits", "v_bits", "p_bits")):
                raise ValueError("Reranker FlashAttention graph precision differs from metadata")


def validate_artifact(root, manifest, meta_file, recipe):
    validate_llm_artifact(root, manifest, meta_file, recipe)
    validate_score_contract(root, json.loads(meta_file.read_text(encoding="utf-8")))
