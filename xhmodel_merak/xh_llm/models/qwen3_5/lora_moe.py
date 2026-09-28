"""PEFT packed expert parameters mapped to semantic Qwen MoeBlock modules."""

from dataclasses import replace


EXPERT_PARAMETERS = {"experts.gate_up_proj", "experts.down_proj"}


def expert_target(path, config):
    if not config.get("target_parameters"):
        return None
    for suffix, kind in (
        (".experts.base_layer", "gate_up"),
        (".experts.gate_up_proj", "gate_up"),
        (".experts.down_proj", "down"),
        (".experts", "down"),
    ):
        if path.endswith(suffix):
            return kind
    return None


def canonical_path(path):
    for prefix in ("base_model.model.", "base_model.", "model.", "language_model."):
        if path.startswith(prefix):
            path = path[len(prefix) :]
    return path


def select_adapter(adapter, prefix="", *, exclude=()):
    """Keep exactly the targets belonging to a loaded placeholder or main graph."""

    def belongs(path, parent):
        path, parent = canonical_path(path), canonical_path(parent)
        return not parent or path == parent or path.startswith(parent + ".")

    pairs = tuple(
        p
        for p in adapter.pairs
        if belongs(p.module_path, prefix) and not any(belongs(p.module_path, item) for item in exclude)
    )
    if not pairs:
        return None
    keys = {key for pair in pairs for key in (pair.a_key, pair.b_key)}
    return replace(adapter, pairs=pairs, weight_map={k: v for k, v in adapter.weight_map.items() if k in keys})


def attach_expert_lora(model, adapter, tensors, *, module_prefix=""):
    from xhquant.nn.modules.moeblock import MoeBlock

    blocks = {
        canonical_path(".".join(p for p in (module_prefix, name) if p)): module
        for name, module in model.named_modules()
        if isinstance(module, MoeBlock)
    }
    grouped = {}
    for pair in adapter.pairs:
        if pair.target_parameter is None:
            continue
        path = canonical_path(pair.module_path)
        parent = path.split(".experts", 1)[0]
        target = parent + ".moeblock"
        if target not in blocks:
            raise ValueError(f"Expert LoRA target {pair.module_path!r} has no MoeBlock {target!r}")
        projections = grouped.setdefault(target, {})
        if pair.target_parameter in projections:
            raise ValueError(f"Duplicate expert LoRA target {target}/{pair.target_parameter}")
        projections[pair.target_parameter] = pair
    for target, pairs in grouped.items():
        if set(pairs) != {"gate_up", "down"}:
            raise ValueError(f"Expert LoRA {target} requires both gate_up_proj and down_proj")
        block = blocks[target]
        weight = block.expert_gate_proj_weight
        if weight is None:
            packed = block.expert_gate_proj_packed
            experts, intermediate, hidden = len(packed), packed[0].out_features, packed[0].in_features
        else:
            experts, intermediate, hidden = weight.shape
        result = {}
        for kind, inputs, outputs in (("gate_up", hidden, 2 * intermediate), ("down", intermediate, hidden)):
            pair = pairs[kind]
            a, b = tensors[pair.a_key], tensors[pair.b_key]
            if tuple(a.shape) != (experts * adapter.rank, inputs) or tuple(b.shape) != (
                outputs,
                experts * adapter.rank,
            ):
                raise ValueError(f"Invalid packed expert LoRA shapes for {pair.module_path}: {a.shape}, {b.shape}")
            a = a.reshape(experts, adapter.rank, inputs)
            # PEFT target_parameters flattens B with expert as the innermost axis.
            b = b.reshape(outputs, adapter.rank, experts).permute(2, 0, 1).contiguous()
            if kind == "gate_up":
                result["gate"] = (a, b[:, :intermediate].contiguous())
                result["up"] = (a, b[:, intermediate:].contiguous())
            else:
                result["down"] = (a, b)
        block.set_lora(result, scaling=adapter.scale)
    return sum(len(pairs) for pairs in grouped.values())


class QwenLoRAPlaceholderMixin:
    def prepare_wrapped_placeholder_module(self, module, module_target, wrap_cfg):
        from .lora import attach_lora_buffers, inspect_lora_adapter

        path = wrap_cfg.get("_lora_adapter_path")
        if path:
            adapter = select_adapter(inspect_lora_adapter(path), module_target)
            if adapter:
                attach_lora_buffers(module, adapter, module_prefix=module_target)
        return module

    def prepare_placeholder_frontend(self, frontend, module_target, wrap_cfg):
        from .lora import apply_lora_graph, inspect_lora_adapter

        path = wrap_cfg.get("_lora_adapter_path")
        if path:
            adapter = select_adapter(inspect_lora_adapter(path), module_target)
            if adapter:
                apply_lora_graph(frontend, adapter, wrap_cfg.get("_lora_w_schema"), wrap_cfg.get("_lora_a_schema"))
        return frontend
