"""Bind validated target-owned output heads across HMONNX sessions."""

from __future__ import annotations

import torch

_PAYLOAD = ("qweight", "scale_or_exp", "weight", "bias")
_SCHEMA = (
    "in_features",
    "out_features",
    "fp_mode",
    "hmfp_weight_exp_bit",
    "hmfp_weight_man_bit",
    "hmfp_weight_nshare",
    "hmfp_weight_rounding",
    "hmfp_weight_hidden_bit",
    "_qweight_packed_w4",
    "_tp_shard_start",
    "_tp_shard_end",
)


def shared_head_bits(contract):
    """Read the versioned declaration; absence preserves legacy artifacts."""
    if "shared_head" not in contract:
        return None
    value = contract["shared_head"]
    if not isinstance(value, dict) or value.get("version") != 1 or value.get("owner") != "target":
        raise ValueError("Unsupported target-owned shared_head contract")
    bits = value.get("logical_bits")
    if type(bits) is not int or bits not in (4, 5, 8):
        raise ValueError("shared_head.logical_bits must be 4, 5 or 8")
    if contract.get("draft_head_weight_bits", bits) != bits:
        raise ValueError("Draft head bits conflict with target-owned shared_head")
    return bits


def session_head(session):
    heads = [
        module
        for module in session.node_modules
        if str(getattr(module, "node_name", "")).replace(".", "_").endswith("lm_head")
    ]
    if not heads:
        # The dynamo exporter may use node_linear instead of the FX module
        # name. Parameter inputs still identify the owning lm_head unambiguously.
        import onnx

        graph = onnx.load(session._onnx_file, load_external_data=False).graph
        names = {
            node.name
            for node in graph.node
            if node.op_type == "Linear"
            and any(name.replace(".", "_").endswith("lm_head_qweight") for name in node.input)
        }
        heads = [module for module in session.node_modules if module.node_name in names]
    if len(heads) != 1:
        raise ValueError(f"Shared head contract requires one lm_head, found {len(heads)}")
    return heads[0]


def assert_shared_head(source, consumer):
    """Check storage after runtime placement, including before CUDA capture."""
    for name in _PAYLOAD:
        owner = getattr(source, name, None)
        value = getattr(consumer, name, None)
        if owner is None and value is None:
            continue
        if not isinstance(owner, torch.Tensor) or not isinstance(value, torch.Tensor):
            raise RuntimeError(f"Shared head payload {name} is missing")
        if owner.device != value.device or owner.shape != value.shape or owner.dtype != value.dtype:
            raise RuntimeError(f"Shared head payload {name} placement differs")
        if owner.is_meta or value.is_meta or owner.data_ptr() != value.data_ptr():
            raise RuntimeError(f"Shared head payload {name} does not reuse target storage")


def _equal_payload(left, right):
    if left.device == right.device:
        return torch.equal(left, right)
    # Head matrices are large. Bound the cross-device validation copy instead
    # of materializing two full vocabulary matrices on CPU.
    if left.ndim == 0:
        return torch.equal(left.cpu(), right.cpu())
    for start in range(0, left.shape[0], 256):
        if not torch.equal(left[start : start + 256].cpu(), right[start : start + 256].cpu()):
            return False
    return True


def bind_session_heads(target_session, consumer_sessions, *, logical_bits):
    """Validate the exported payload before aliasing; never overwrite a mismatch.

    The HMONNX Linear ABI stores symmetric zero-point/format information in
    attributes and integer codes/scales in buffers. Both are checked. Sharing
    uses the loader's SharedTensor objects so subsequent device moves retain
    its deduplicated storage semantics.
    """
    source = session_head(target_session)
    if source.hmfp_weight_man_bit != logical_bits or source.qweight is None:
        raise ValueError("Target head does not match shared-head logical precision")
    consumers = [session_head(session) for session in consumer_sessions]
    for consumer in consumers:
        for name in _SCHEMA:
            if getattr(source, name, None) != getattr(consumer, name, None):
                raise ValueError(f"Shared head schema mismatch: {name}")
        for name in _PAYLOAD:
            owner, value = getattr(source, name, None), getattr(consumer, name, None)
            if owner is None and value is None:
                continue
            if not isinstance(owner, torch.Tensor) or not isinstance(value, torch.Tensor):
                raise ValueError(f"Shared head payload mismatch: {name}")
            if owner.is_meta or value.is_meta or owner.shape != value.shape or owner.dtype != value.dtype:
                raise ValueError(f"Shared head tensor schema mismatch: {name}")
            if not _equal_payload(owner, value):
                raise ValueError(f"Shared head tensor values differ: {name}")
    for consumer in consumers:
        for name in _PAYLOAD:
            setattr(consumer, name, getattr(source, name, None))
        assert_shared_head(source, consumer)
        # The check runs before execution/capture, detecting later replacement
        # by placement, tensor-parallel conversion or another runtime consumer.
        consumer.register_forward_pre_hook(lambda module, args: assert_shared_head(source, module))
    return source
