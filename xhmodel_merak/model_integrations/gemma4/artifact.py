"""Build/load checks for Gemma4 assistant ownership and graph ABI."""

from collections.abc import Mapping

from xhmodel_merak.model_artifact.files import confined_file


def validate_mtp(root, meta, plan=None):
    expected = "mtp" in plan.components if plan is not None else True
    if not expected:
        return
    spec = meta.get("spec_decode", {})
    if not isinstance(spec, Mapping):
        raise ValueError("Gemma4 MTP requires a spec_decode mapping")
    draft = spec.get("draft", {})
    if not isinstance(draft, Mapping) or spec.get("mode") != "mtp" or spec.get("runtime_contract_version") != 2:
        raise ValueError("Gemma4 MTP requires a version-2 draft contract; re-export the model")
    allowed = {
        "gemma4_mtp_dense_shared_kv_v1",
        "gemma4_mtp_readonly_page_attention_v2",
        "gemma4_mtp_readonly_mixed_attention_v3",
    }
    if draft.get("abi") not in allowed or draft.get("cache_mutation") != "read_only":
        raise ValueError("Gemma4 MTP requires a read-only target-KV assistant ABI")
    if spec.get("shared_head") is not None:
        raise ValueError("Gemma4 assistant has its own head; target-owned shared_head is invalid")
    count = spec.get("num_draft_tokens")
    if type(count) is not int or count <= 0 or spec.get("verify_length") != count + 1:
        raise ValueError("Gemma4 verify_length must equal num_draft_tokens + 1")
    if plan is not None:
        requested = plan.config_overrides.get("export.model.num_draft_tokens")
        if requested is not None and requested != count:
            raise ValueError("Gemma4 artifact draft count differs from the build request")
    if draft.get("prefill_hmonnx") or spec.get("draft_prefill_onnx"):
        raise ValueError("Gemma4 MTP has no independent draft prefill graph")
    path = confined_file(root.resolve(), draft.get("decode_hmonnx"), "mtp.decode_hmonnx")
    for key in ("draft_decode_onnx", "draft_onnx"):
        if confined_file(root.resolve(), spec.get(key), key) != path:
            raise ValueError("Gemma4 draft graph aliases disagree")
    standalone = confined_file(root.resolve(), spec.get("standalone_draft_decode_onnx"), "mtp.standalone")
    if confined_file(root.resolve(), draft.get("standalone_decode_hmonnx"), "draft.standalone") != standalone:
        raise ValueError("Gemma4 standalone draft graph aliases disagree")
    import onnx

    for phase in ("prefill", "decode"):
        target_path = confined_file(root.resolve(), meta.get(f"{phase}_hmonnx"), f"target.{phase}")
        target = onnx.load(target_path, load_external_data=False).graph
        if [item.name for item in target.output] != ["logits", "target_hidden_state"]:
            raise ValueError("Gemma4 MTP target must export logits and target_hidden_state")
        if phase == "decode":
            logits_shape = target.output[0].type.tensor_type.shape.dim
            if len(logits_shape) != 3 or logits_shape[1].dim_value != count + 1:
                raise ValueError("Gemma4 target verify graph width differs from num_draft_tokens + 1")
    graph = onnx.load(standalone, load_external_data=False).graph
    inputs = {item.name for item in graph.input}
    kv_names = {f"shared_{kind}_cache_{mode}" for kind in ("key", "value") for mode in ("sliding", "full")}
    if not kv_names <= inputs:
        raise ValueError("Gemma4 standalone draft must consume all four target KV inputs")
    if [item.name for item in graph.output] != ["logits", "assistant_hidden_state"]:
        raise ValueError("Gemma4 draft must output logits and projected assistant hidden state")
    for node in graph.node:
        if node.op_type == "KVcache":
            attrs = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
            if attrs.get("only_handle_old_cache") != 1:
                raise ValueError("Gemma4 draft KVcache must be read-only")
