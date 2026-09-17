from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper
from safetensors.torch import save_file

from xhmodel_merak.xh_llm.models.macaron_v1_tall.lora_constant_bundle import (
    ConstantONNXWriter,
    MacaronLoRAConstantBundle,
    append_canonical_lora_to_stage,
    load_initializer_references,
    read_manifest,
)
from xhquant.export.onnx.onnx_opset._impl.schemas.opset1.linear_schema import linear_schema


class _StaticAdapters:
    """Small checkpoint substitute for graph-injection unit tests."""

    scaling = 2.0

    def __init__(
        self,
        weights: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    ) -> None:
        self._weights = weights

    def pair(
        self,
        adapter_name: str,
        module_key: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a_weight, b_weight = self._weights[module_key][adapter_name]
        return torch.from_numpy(a_weight), torch.from_numpy(b_weight)


class _TargetOnlyAdapters:
    """Expose checkpoint names while failing if a target map reads weights."""

    model_shape = SimpleNamespace(hidden_size=4)

    def module_keys(self) -> tuple[str, ...]:
        return (
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.mlp.experts",
        )

    def pair(self, *_: object) -> tuple[torch.Tensor, torch.Tensor]:
        raise AssertionError("Building targets must not read LoRA tensors")


def _load_module():
    return importlib.import_module("xhmodel_merak.xh_llm.models.macaron_v1_tall." "non_routed_lora_injection")


def _load_expert_module():
    return importlib.import_module("xhmodel_merak.xh_llm.models.macaron_v1_tall." "routed_expert_lora_injection")


def _load_weights_module():
    return importlib.import_module("xhmodel_merak.xh_llm.models.macaron_v1_tall.lora_weights")


@pytest.fixture(params=["installed", "legacy", "saturate"])
def xh2a_add_schema(request):
    """Check real graphs against both SDK schema generations and the local SDK."""
    importlib.import_module("xhquant.export.onnx.onnx_opset")
    original = onnx.defs.get_schema("Add", 1, "ai.houmo.xh2a")
    if request.param == "installed":
        yield original
        return

    schema = onnx.defs.OpSchema(
        "Add",
        original.domain,
        original.since_version,
        inputs=original.inputs,
        outputs=original.outputs,
        type_constraints=[
            (item.type_param_str, item.allowed_type_strs, item.description)
            for item in original.type_constraints
        ],
        attributes=(
            [onnx.defs.OpSchema.Attribute("saturate", onnx.defs.OpSchema.AttrType.INT)]
            if request.param == "saturate" else []
        ),
    )
    onnx.defs.deregister_schema(original.name, original.since_version, original.domain)
    try:
        onnx.defs.register_schema(schema)
        yield schema
    finally:
        if onnx.defs.has(original.name, original.since_version, original.domain):
            onnx.defs.deregister_schema(original.name, original.since_version, original.domain)
        onnx.defs.register_schema(original)


def _single_linear_graph(qweight: str, graph_name: str):
    # The imported SDK schema also validates this base op in focused runs.
    scale_name = qweight.removesuffix(".qweight") + ".scale"
    packed_weight, packed_scale = _load_module().quantize_non_routed_lora_weight(
        np.ones((4, 4), dtype=np.float16), dim=0, mode="ssfp", nshare=64,
        man_bit=8, hidden_bit=False, rounding="RNE",
    )
    return helper.make_graph(
        [
            helper.make_node(
                linear_schema.name,
                ["hidden", qweight, scale_name],
                ["output"],
                domain=linear_schema.domain,
                have_bias=0,
                in_features=4,
                out_features=4,
                mode="ssfp",
                hmfp_act_exp_bit=5,
                hmfp_act_man_bit=8,
                hmfp_act_nshare=64,
                hmfp_act_rounding="RNE",
                hmfp_act_hidden_bit=1,
                hmfp_act_max_exp_boost=0,
                hmfp_weight_exp_bit=5,
                hmfp_weight_man_bit=8,
                hmfp_weight_nshare=64,
                hmfp_weight_rounding="RNE",
                hmfp_weight_hidden_bit=0,
                hmfp_psum_dtype="fp24",
                hmfp_psum_round_mode="trunc",
                output_dtype="float16",
            )
        ],
        graph_name,
        [helper.make_tensor_value_info("hidden", TensorProto.FLOAT16, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 4])],
        initializer=[
            numpy_helper.from_array(packed_weight, qweight),
            numpy_helper.from_array(packed_scale, scale_name),
        ],
    )


def _load_pipeline_module():
    return importlib.import_module("xhmodel_merak.xh_llm.models.macaron_v1_tall." "lora_export_pipeline")


def _load_overlay_module():
    return importlib.import_module(
        "xhmodel_merak.xh_llm.models.macaron_v1_tall."
        "lora_adapter_overlay"
    )


def _bundle_values(artifact) -> dict[str, dict[str, np.ndarray]]:
    return {
        adapter_name: {
            initializer.name: numpy_helper.to_array(initializer)
            for initializer in onnx.load(
                artifact.adapter_onnx(adapter_name),
                load_external_data=True,
            ).graph.initializer
        }
        for adapter_name in ("L0", "L1", "L2", "L3")
    }


def _write_non_routed_bundle(
    module,
    tmp_path: Path,
    graph,
    targets,
    adapters,
    quant_override=None,
):
    bindings = module.build_non_routed_lora_bindings(
        graph,
        targets,
        rank=2,
        quant_override=quant_override,
    )
    with MacaronLoRAConstantBundle(tmp_path) as bundle:
        module.write_non_routed_lora_constants(
            bundle,
            adapters,
            bindings,
        )
        artifact = bundle.close()
    references = load_initializer_references(
        artifact.adapter_onnx("L0"),
        relative_data_location="stage_external_data",
    )
    (tmp_path / "stage_external_data").write_bytes(artifact.adapter_onnx("L0").with_suffix(".onnx.data").read_bytes())
    return bindings, artifact, references


def test_export_lora_converts_checkpoint_to_deployment_dtype(
    tmp_path: Path,
) -> None:
    module = _load_weights_module()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "num_hidden_layers": 1,
                    "num_experts": 1,
                    "hidden_size": 4,
                    "moe_intermediate_size": 2,
                }
            }
        )
    )
    key_prefix = "base_model.model.model.layers.0.self_attn.q_proj"
    for name in module.ADAPTER_NAMES:
        checkpoint = tmp_path / "loras" / name / "adapter_model.safetensors"
        checkpoint.parent.mkdir(parents=True)
        (checkpoint.parent / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "bias": "none",
                    "r": 2,
                    "lora_alpha": 4,
                    "target_parameters": [
                        "experts.gate_up_proj",
                        "experts.down_proj",
                    ],
                }
            )
        )
        save_file(
            {
                f"{key_prefix}.lora_A.weight": torch.ones((2, 4), dtype=torch.float32),
                f"{key_prefix}.lora_B.weight": torch.ones((4, 2), dtype=torch.float32),
            },
            checkpoint,
        )

    with module.open_macaron_lora_adapters(tmp_path) as adapters:
        lora_a, lora_b = adapters.pair(
            "L0",
            "model.layers.0.self_attn.q_proj",
        )
        assert adapters.rank == 2
        assert adapters.scaling == 2.0
        assert adapters.model_shape.hidden_size == 4

    assert lora_a.dtype == torch.float16
    assert lora_b.dtype == torch.float16


def test_non_routed_lora_targets_do_not_materialize_checkpoints() -> None:
    module = _load_module()

    targets = module.build_non_routed_lora_targets(_TargetOnlyAdapters())

    assert [(target.qweight_name, target.lora_key, target.b_rows) for target in targets] == [
        (
            "model.language_model.layers.0.self_attn.q_proj.qweight",
            "model.layers.0.self_attn.q_proj",
            None,
        )
    ]


def test_fused_non_routed_lora_targets_use_model_hidden_size() -> None:
    module = _load_module()

    class _FusedAdapters(_TargetOnlyAdapters):
        def module_keys(self) -> tuple[str, ...]:
            return ("model.layers.0.linear_attn.in_proj_qkv",)

    targets = module.build_non_routed_lora_targets(_FusedAdapters())

    assert {target.qweight_name: target.b_rows for target in targets} == {
        "model.language_model.layers.0.linear_attn.in_proj_q.qweight": (
            0,
            4,
        ),
        "model.language_model.layers.0.linear_attn.in_proj_k.qweight": (
            4,
            8,
        ),
        "model.language_model.layers.0.linear_attn.in_proj_v.qweight": (
            8,
            16,
        ),
    }


def test_packed_routed_expert_lora_matches_peft_param_wrapper_layout() -> None:
    """PEFT ParamWrapper packs B as [out, rank, expert]."""

    module = _load_weights_module()
    experts, rank, hidden, intermediate = 3, 2, 4, 5
    shape = module.MacaronModelShape(
        num_layers=1,
        num_experts=experts,
        hidden_size=hidden,
        moe_intermediate_size=intermediate,
    )

    gate_up_a = torch.arange(experts * rank * hidden, dtype=torch.float32).reshape(experts * rank, hidden)
    gate_up_b = torch.arange(2 * intermediate * rank * experts, dtype=torch.float32).reshape(
        2 * intermediate, rank * experts
    )
    down_a = torch.arange(experts * rank * intermediate, dtype=torch.float32).reshape(experts * rank, intermediate)
    down_b = torch.arange(hidden * rank * experts, dtype=torch.float32).reshape(hidden, rank * experts)

    gate_a, gate_b, up_a, up_b = module._reshape_gate_up(gate_up_a, gate_up_b, shape=shape, rank=rank)
    packed_down_a, packed_down_b = module._reshape_down(down_a, down_b, shape=shape, rank=rank)

    expected_gate_up_b = gate_up_b.reshape(2 * intermediate, rank, experts).permute(2, 0, 1)
    expected_down_b = down_b.reshape(hidden, rank, experts).permute(2, 0, 1)
    torch.testing.assert_close(gate_a, gate_up_a.reshape(experts, rank, hidden))
    torch.testing.assert_close(up_a, gate_a)
    torch.testing.assert_close(gate_b, expected_gate_up_b[:, :intermediate, :])
    torch.testing.assert_close(up_b, expected_gate_up_b[:, intermediate:, :])
    torch.testing.assert_close(packed_down_a, down_a.reshape(experts, rank, intermediate))
    torch.testing.assert_close(packed_down_b, expected_down_b)

    # Verify the packed tensors preserve PEFT's per-expert delta equation,
    # including the Macaron adapter scaling alpha / r = 128 / 64 = 2.
    scaling = 2.0
    x_hidden = torch.arange(2 * hidden, dtype=torch.float32).reshape(2, hidden) / 10
    x_intermediate = torch.arange(2 * intermediate, dtype=torch.float32).reshape(2, intermediate) / 10
    for expert in range(experts):
        peft_gate_b = gate_up_b.reshape(2 * intermediate, rank, experts)[:, :, expert]
        peft_down_b = down_b.reshape(hidden, rank, experts)[:, :, expert]
        peft_gate_delta = (x_hidden @ gate_up_a.reshape(experts, rank, hidden)[expert].T) @ peft_gate_b.T
        packed_gate_delta = (x_hidden @ gate_a[expert].T) @ torch.cat((gate_b[expert], up_b[expert])).T
        peft_down_delta = (x_intermediate @ down_a.reshape(experts, rank, intermediate)[expert].T) @ peft_down_b.T
        packed_down_delta = (x_intermediate @ packed_down_a[expert].T) @ packed_down_b[expert].T
        torch.testing.assert_close(packed_gate_delta * scaling, peft_gate_delta * scaling)
        torch.testing.assert_close(packed_down_delta * scaling, peft_down_delta * scaling)


def test_attention_lora_uses_canonical_initializer_and_adapter_constant_packs(
    tmp_path: Path,
    xh2a_add_schema,
) -> None:
    module = _load_module()
    qweight = "model.language_model.layers.0.self_attn.q_proj.qweight"
    graph = _single_linear_graph(qweight, "lora")
    adapter_weights = {
        name: (
            np.full((2, 4), index + 1, dtype=np.float16),
            np.full((4, 2), index + 1, dtype=np.float16),
        )
        for index, name in enumerate(module.ADAPTER_NAMES)
    }

    target = module.NonRoutedLoRATarget(
        qweight_name=qweight,
        lora_key="layer.q_proj",
        b_rows=None,
    )
    bindings, artifact, references = _write_non_routed_bundle(
        module,
        tmp_path,
        graph,
        (target,),
        _StaticAdapters({"layer.q_proj": adapter_weights}),
    )
    modified = module.bind_non_routed_lora_graph(graph, bindings, references)
    runtime_input_order = list(artifact.input_order)
    runtime_packs = _bundle_values(artifact)

    assert modified == 1
    assert not [value.name for value in graph.input if "lora" in value.name]
    residual_add = next(node for node in graph.node if node.op_type == "Add")
    assert residual_add.domain == xh2a_add_schema.domain
    assert {attr.name: helper.get_attribute_value(attr) for attr in residual_add.attribute} == (
        {"saturate": 0} if "saturate" in xh2a_add_schema.attributes else {}
    )
    assert len(runtime_input_order) == 4
    assert all(name == name.lower() for name in runtime_input_order)
    assert not any(node.op_type == "Mul" for node in graph.node)
    assert not any(node.op_type == "Gather" for node in graph.node)
    lora_linear_nodes = [node for node in graph.node if node.op_type == "Linear" and node.output[0].startswith("lora_")]
    assert len(lora_linear_nodes) == 2
    assert not any(node.op_type == "MatMul" for node in graph.node)
    initializers = {initializer.name: initializer for initializer in graph.initializer}
    lora_initializer_names = [
        initializer.name for initializer in graph.initializer if initializer.name in runtime_input_order
    ]
    assert lora_initializer_names == runtime_input_order
    assert all(name in initializers for name in lora_linear_nodes[0].input[1:3])
    assert all(name in initializers for name in lora_linear_nodes[1].input[1:3])
    assert list(runtime_packs["L0"]) == runtime_input_order
    assert any(not np.array_equal(runtime_packs["L0"][name], runtime_packs["L1"][name]) for name in runtime_input_order)
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid("ai.houmo.xh2a", 1),
        ],
    )
    model_path = tmp_path / "attention_lora.onnx"
    onnx.save(model, model_path)
    onnx.checker.check_model(model_path)


def test_decode_reuses_prefill_non_routed_lora_constants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _load_module()
    qweight = "model.language_model.layers.0.self_attn.q_proj.qweight"
    adapter_weights = {
        name: (
            np.full((2, 4), index + 1, dtype=np.float16),
            np.full((4, 2), index + 1, dtype=np.float16),
        )
        for index, name in enumerate(module.ADAPTER_NAMES)
    }
    adapters = _StaticAdapters({"layer.q_proj": adapter_weights})
    target = module.NonRoutedLoRATarget(
        qweight_name=qweight,
        lora_key="layer.q_proj",
        b_rows=None,
    )
    quantization_calls = 0
    quantize = module.quantize_non_routed_lora_weight

    def count_quantization(*args, **kwargs):
        nonlocal quantization_calls
        quantization_calls += 1
        return quantize(*args, **kwargs)

    prefill = _single_linear_graph(qweight, "prefill")
    decode = _single_linear_graph(qweight, "decode")
    monkeypatch.setattr(module, "quantize_non_routed_lora_weight", count_quantization)
    with MacaronLoRAConstantBundle(tmp_path) as bundle:
        prefill_bindings = module.build_non_routed_lora_bindings(prefill, (target,), rank=2)
        decode_bindings = module.build_non_routed_lora_bindings(decode, (target,), rank=2)
        assert prefill_bindings == decode_bindings
        module.write_non_routed_lora_constants(bundle, adapters, prefill_bindings)
        assert quantization_calls == 2 * len(module.ADAPTER_NAMES)
        artifact = bundle.close()
        references = load_initializer_references(
            artifact.adapter_onnx("L0"),
            relative_data_location="stage_external_data",
        )
        assert module.bind_non_routed_lora_graph(prefill, prefill_bindings, references) == 1
        assert module.bind_non_routed_lora_graph(decode, decode_bindings, references) == 1
        assert quantization_calls == 2 * len(module.ADAPTER_NAMES)
        assert [
            initializer.name for initializer in decode.initializer if initializer.name in artifact.input_order
        ] == list(artifact.input_order)
        assert all(
            initializer.data_location == TensorProto.EXTERNAL
            for initializer in decode.initializer
            if initializer.name in artifact.input_order
        )
        phase_dir = tmp_path / "prefill"
        phase_dir.mkdir()
        prefill_path = phase_dir / "model_prefill_with_act.onnx"
        onnx.save(
            helper.make_model(
                prefill,
                opset_imports=[
                    helper.make_opsetid("", 18),
                    helper.make_opsetid("ai.houmo.xh2a", 1),
                ],
            ),
            prefill_path,
        )
    assert artifact.manifest_path.is_file()
    loaded_prefill = onnx.load(prefill_path, load_external_data=False)
    assert {
        initializer.name for initializer in loaded_prefill.graph.initializer if initializer.name in artifact.input_order
    } == set(artifact.input_order)
    assert {
        entry.value
        for initializer in loaded_prefill.graph.initializer
        if initializer.name in artifact.input_order
        for entry in initializer.external_data
        if entry.key == "location"
    } == {"stage_external_data"}


def test_shared_expert_lora_uses_canonical_initializer_and_adapter_constant_packs(
    tmp_path: Path,
    xh2a_add_schema,
) -> None:
    module = _load_module()
    qweight = "model.language_model.layers.0.mlp.shared_expert.gate_proj.qweight"
    graph = _single_linear_graph(qweight, "shared_expert_runtime_lora")
    adapter_weights = {
        name: (
            np.full((2, 4), index + 1, dtype=np.float16),
            np.full((4, 2), index + 1, dtype=np.float16),
        )
        for index, name in enumerate(module.ADAPTER_NAMES)
    }
    lora_key = "model.layers.0.mlp.shared_expert.gate_proj"
    target = module.NonRoutedLoRATarget(
        qweight_name=qweight,
        lora_key=lora_key,
        b_rows=None,
    )
    bindings, artifact, references = _write_non_routed_bundle(
        module,
        tmp_path,
        graph,
        (target,),
        _StaticAdapters({lora_key: adapter_weights}),
    )
    modified = module.bind_non_routed_lora_graph(graph, bindings, references)
    runtime_input_order = list(artifact.input_order)
    runtime_packs = _bundle_values(artifact)

    assert modified == 1
    assert "lora_adapter_idx" not in {value.name for value in graph.input}
    residual_add = next(node for node in graph.node if node.op_type == "Add")
    assert residual_add.domain == xh2a_add_schema.domain
    assert {attr.name: helper.get_attribute_value(attr) for attr in residual_add.attribute} == (
        {"saturate": 0} if "saturate" in xh2a_add_schema.attributes else {}
    )
    assert len(runtime_input_order) == 4
    assert all(name == name.lower() for name in runtime_input_order)
    assert not [value.name for value in graph.input if value.name in runtime_input_order]
    assert not any(node.op_type == "Gather" for node in graph.node)
    lora_initializer_names = [
        initializer.name for initializer in graph.initializer if initializer.name in runtime_input_order
    ]
    assert lora_initializer_names == runtime_input_order

    lora_linears = [node for node in graph.node if node.op_type == "Linear" and node.output[0].startswith("lora_")]
    assert len(lora_linears) == 2
    assert lora_linears[0].input[1:3] == runtime_input_order[:2]
    assert lora_linears[1].input[1:3] == runtime_input_order[2:]

    expected_a_qweight, expected_a_scale = module.quantize_non_routed_lora_weight(
        adapter_weights["L0"][0].T,
        dim=0,
        mode="ssfp",
        nshare=64,
        man_bit=8,
        hidden_bit=False,
        rounding="RNE",
    )
    expected_b_qweight, expected_b_scale = module.quantize_non_routed_lora_weight(
        adapter_weights["L0"][1].T * 2.0,
        dim=0,
        mode="ssfp",
        nshare=64,
        man_bit=8,
        hidden_bit=False,
        rounding="RNE",
    )
    expected_shapes = [
        expected_a_qweight.shape,
        expected_a_scale.shape,
        expected_b_qweight.shape,
        expected_b_scale.shape,
    ]
    for adapter_name in module.ADAPTER_NAMES:
        assert list(runtime_packs[adapter_name]) == runtime_input_order
        assert [tensor.shape for tensor in runtime_packs[adapter_name].values()] == expected_shapes
        assert all(tensor.ndim == 3 for tensor in runtime_packs[adapter_name].values())
    assert any(not np.array_equal(runtime_packs["L0"][name], runtime_packs["L1"][name]) for name in runtime_input_order)

    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 18),
            helper.make_opsetid("ai.houmo.xh2a", 1),
        ],
    )
    model_path = tmp_path / "shared_expert_lora.onnx"
    onnx.save(model, model_path)
    onnx.checker.check_model(model_path)


def test_runtime_lora_bundle_writes_one_weight_pack_per_adapter(
    tmp_path: Path,
) -> None:
    module = _load_module()
    input_order = ["layer0_a_qweight", "layer0_a_scale"]
    packs = {
        name: {
            input_order[0]: np.full((1, 64, 2), index + 1, dtype=np.int8),
            input_order[1]: np.full((1, 1, 2), index + 1, dtype=np.int16),
        }
        for index, name in enumerate(module.ADAPTER_NAMES)
    }

    with MacaronLoRAConstantBundle(tmp_path) as bundle:
        for name in input_order:
            bundle.add(
                name,
                {adapter_name: packs[adapter_name][name] for adapter_name in module.ADAPTER_NAMES},
            )
        artifact = bundle.close()

    manifest = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "onnx_initializer_bundle"
    assert manifest["input_order"] == input_order
    assert set(manifest["adapters"]) == set(module.ADAPTER_NAMES)
    schemas = []
    for adapter_name in module.ADAPTER_NAMES:
        entry = manifest["adapters"][adapter_name]
        pack_path = artifact.manifest_path.parent / entry["onnx"]
        model = onnx.load(pack_path, load_external_data=True)
        onnx.checker.check_model(model)
        tensors = {initializer.name: numpy_helper.to_array(initializer) for initializer in model.graph.initializer}
        assert list(tensors) == input_order
        assert (artifact.manifest_path.parent / entry["external_data"]).is_file()
        schemas.append([(name, value.dtype, value.shape) for name, value in tensors.items()])
    assert all(schema == schemas[0] for schema in schemas[1:])
    assert not np.array_equal(
        numpy_helper.to_array(
            onnx.load(artifact.manifest_path.parent / manifest["adapters"]["L0"]["onnx"]).graph.initializer[0]
        ),
        numpy_helper.to_array(
            onnx.load(artifact.manifest_path.parent / manifest["adapters"]["L1"]["onnx"]).graph.initializer[0]
        ),
    )


def test_hmonnx_linear_accepts_canonical_lora_initializers(
    tmp_path: Path,
) -> None:
    from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference

    module = _load_module()
    features = 64
    weight_l0 = np.eye(features, dtype=np.float16)
    qweight_l0, scale_l0 = module.quantize_non_routed_lora_weight(
        weight_l0,
        dim=0,
        mode="ssfp",
        nshare=64,
        man_bit=8,
        hidden_bit=False,
        rounding="RNE",
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "Linear",
                [
                    "hidden",
                    "canonical_qweight",
                    "canonical_scale",
                    "",
                    "",
                ],
                ["output"],
                domain="ai.houmo.xh2a",
                have_bias=0,
                in_features=features,
                out_features=features,
                mode="ssfp",
                hmfp_act_exp_bit=5,
                hmfp_act_man_bit=8,
                hmfp_act_nshare=64,
                hmfp_act_rounding="RNE",
                hmfp_act_hidden_bit=1,
                hmfp_act_max_exp_boost=0,
                hmfp_weight_exp_bit=5,
                hmfp_weight_man_bit=8,
                hmfp_weight_nshare=64,
                hmfp_weight_rounding="RNE",
                hmfp_weight_hidden_bit=0,
                hmfp_psum_dtype="fp24",
                hmfp_psum_round_mode="trunc",
                output_dtype="float16",
            )
        ],
        "runtime_lora_linear",
        [helper.make_tensor_value_info("hidden", TensorProto.FLOAT16, [1, features])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, features])],
        initializer=[
            numpy_helper.from_array(qweight_l0, "canonical_qweight"),
            numpy_helper.from_array(scale_l0, "canonical_scale"),
        ],
    )
    model_path = tmp_path / "runtime_lora_linear.onnx"
    onnx.save(
        helper.make_model(
            graph,
            opset_imports=[
                helper.make_opsetid("", 18),
                helper.make_opsetid("ai.houmo.xh2a", 1),
            ],
        ),
        model_path,
    )
    session = HMONNXInference(str(model_path))
    hidden = torch.ones((1, features), dtype=torch.float16)

    output_l0 = session.run({"hidden": hidden})

    assert output_l0.shape == (1, features)


def test_fused_qkv_reuses_runtime_a_without_duplicate_producer(
    tmp_path: Path,
) -> None:
    module = _load_module()
    qweight_q = "model.language_model.layers.0.linear_attn.in_proj_q.qweight"
    qweight_k = "model.language_model.layers.0.linear_attn.in_proj_k.qweight"
    graph = helper.make_graph(
        [
            helper.make_node(
                "Linear",
                ["hidden", qweight_q],
                ["q"],
                domain="ai.houmo.xh2a",
                in_features=4,
                out_features=4,
            ),
            helper.make_node(
                "Linear",
                ["hidden", qweight_k],
                ["k"],
                domain="ai.houmo.xh2a",
                in_features=4,
                out_features=4,
            ),
        ],
        "fused_qkv",
        [helper.make_tensor_value_info("hidden", TensorProto.FLOAT16, [1, 4])],
        [
            helper.make_tensor_value_info("q", TensorProto.FLOAT16, [1, 4]),
            helper.make_tensor_value_info("k", TensorProto.FLOAT16, [1, 4]),
        ],
        initializer=[
            numpy_helper.from_array(np.ones((4, 4), dtype=np.float16), qweight_q),
            numpy_helper.from_array(np.ones((4, 4), dtype=np.float16), qweight_k),
        ],
    )
    shared_a = {
        name: (
            np.full((2, 4), index + 1, dtype=np.float16),
            np.full((4, 2), index + 1, dtype=np.float16),
        )
        for index, name in enumerate(module.ADAPTER_NAMES)
    }
    lora_key = "layer.linear_attn.in_proj_qkv"
    targets = tuple(
        module.NonRoutedLoRATarget(
            qweight_name=qweight,
            lora_key=lora_key,
            b_rows=None,
        )
        for qweight in (qweight_q, qweight_k)
    )

    decode_graph = onnx.GraphProto()
    decode_graph.CopyFrom(graph)
    with MacaronLoRAConstantBundle(tmp_path) as bundle:
        bindings = module.build_non_routed_lora_bindings(graph, targets, rank=2)
        decode_bindings = module.build_non_routed_lora_bindings(decode_graph, targets, rank=2)
        module.write_non_routed_lora_constants(
            bundle,
            _StaticAdapters({lora_key: shared_a}),
            bindings,
        )
        artifact = bundle.close()
    references = load_initializer_references(
        artifact.adapter_onnx("L0"),
        relative_data_location="stage_external_data",
    )
    assert module.bind_non_routed_lora_graph(graph, bindings, references) == 2
    assert module.bind_non_routed_lora_graph(decode_graph, decode_bindings, references) == 2

    a_qweight = "lora_layer_linear_attn_in_proj_qkv_a_qweight"
    a_scale = "lora_layer_linear_attn_in_proj_qkv_a_scale"
    assert [initializer.name for initializer in graph.initializer].count(a_qweight) == 1
    assert [initializer.name for initializer in graph.initializer].count(a_scale) == 1
    assert len({initializer.name for initializer in decode_graph.initializer}) == len(decode_graph.initializer)
    assert not any(node.op_type == "Gather" for node in graph.node)
    assert len([node for node in graph.node if node.output[0] == "lora_layer_linear_attn_in_proj_qkv_linear_a"]) == 1


def test_lora_weight_quantization_uses_base_aligned_layout() -> None:
    module = _load_module()
    weights = np.arange(4 * 65 * 3, dtype=np.float32).reshape(4, 65, 3) / 100

    qweight, scale = module.quantize_non_routed_lora_weight(
        weights,
        mode="ssfp",
        nshare=64,
        man_bit=8,
        hidden_bit=False,
        rounding="RNE",
    )

    assert qweight.dtype == np.int8
    assert scale.dtype == np.float16
    assert qweight.shape == (4, 2, 64, 3)
    assert scale.shape == (4, 2, 1, 3)
    assert np.all(qweight[:, 1, 1:, :] == 0)


def test_lora_weight_quantization_uses_base_sefp_exponents() -> None:
    module = _load_module()
    weights = (np.arange(4 * 65 * 3, dtype=np.float32).reshape(4, 65, 3) / 100).astype(np.float16)

    qweight, exponent = module.quantize_non_routed_lora_weight(
        weights,
        mode="sefp",
        nshare=64,
        man_bit=8,
        hidden_bit=True,
        rounding="RNE",
    )

    assert qweight.dtype == np.int8
    assert exponent.dtype == np.int16
    assert qweight.shape == (4, 2, 64, 3)
    assert exponent.shape == (4, 2, 1, 3)


def test_injected_lora_copies_base_linear_quantization_attrs(
    tmp_path: Path,
) -> None:
    module = _load_module()
    qweight = "model.language_model.layers.0.linear_attn.in_proj_q.qweight"
    base_attrs = {
        "mode": "sefp",
        "hmfp_act_exp_bit": 5,
        "hmfp_act_man_bit": 8,
        "hmfp_act_nshare": 64,
        "hmfp_act_rounding": "RNE",
        "hmfp_act_hidden_bit": 1,
        "hmfp_weight_exp_bit": 5,
        "hmfp_weight_man_bit": 8,
        "hmfp_weight_nshare": 64,
        "hmfp_weight_rounding": "RNE",
        "hmfp_weight_hidden_bit": 1,
        "hmfp_psum_dtype": "fp24",
        "hmfp_psum_round_mode": "trunc",
        "output_dtype": "float16",
    }
    graph = helper.make_graph(
        [
            helper.make_node(
                "Linear",
                ["hidden", qweight],
                ["output"],
                domain="ai.houmo.xh2a",
                in_features=4,
                out_features=4,
                have_bias=0,
                **base_attrs,
            )
        ],
        "lora_attrs",
        [helper.make_tensor_value_info("hidden", TensorProto.FLOAT16, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 4])],
        initializer=[numpy_helper.from_array(np.ones((4, 4), dtype=np.float16), qweight)],
    )
    adapter_weights = {
        name: (
            np.ones((2, 4), dtype=np.float16),
            np.ones((4, 2), dtype=np.float16),
        )
        for name in module.ADAPTER_NAMES
    }

    target = module.NonRoutedLoRATarget(
        qweight_name=qweight,
        lora_key="layer.in_proj_q",
        b_rows=None,
    )
    bindings, artifact, references = _write_non_routed_bundle(
        module,
        tmp_path,
        graph,
        (target,),
        _StaticAdapters({"layer.in_proj_q": adapter_weights}),
    )
    module.bind_non_routed_lora_graph(graph, bindings, references)
    runtime_input_order = list(artifact.input_order)
    runtime_packs = _bundle_values(artifact)

    injected = [node for node in graph.node if node.op_type == "Linear" and node.output[0].startswith("lora_")]
    assert len(injected) == 2
    for node in injected:
        attrs = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
        for name, value in base_attrs.items():
            expected = value.encode() if isinstance(value, str) else value
            assert attrs[name] == expected
    scales = [name for name in runtime_input_order if name.endswith("_scale")]
    assert scales == [
        "lora_layer_in_proj_q_a_scale",
        "lora_model_language_model_layers_0_linear_attn_in_proj_q_qweight_b_scale",
    ]
    assert all(runtime_packs["L0"][name].dtype == np.int16 for name in scales)


def test_non_routed_lora_can_use_w8_with_w4_base(tmp_path: Path) -> None:
    module = _load_module()
    qweight = "model.language_model.layers.0.self_attn.q_proj.qweight"
    graph = _single_linear_graph(qweight, "mixed_precision_lora")
    graph.node[0].attribute.extend(
        [
            helper.make_attribute("mode", "ssfp"),
            helper.make_attribute("hmfp_weight_man_bit", 4),
            helper.make_attribute("hmfp_weight_nshare", 64),
            helper.make_attribute("hmfp_weight_hidden_bit", 0),
            helper.make_attribute("hmfp_weight_rounding", "RNE"),
        ]
    )
    adapter_weights = {
        name: (
            np.ones((2, 4), dtype=np.float16),
            np.ones((4, 2), dtype=np.float16),
        )
        for name in module.ADAPTER_NAMES
    }
    target = module.NonRoutedLoRATarget(
        qweight_name=qweight,
        lora_key="layer.q_proj",
        b_rows=None,
    )
    bindings, _, references = _write_non_routed_bundle(
        module,
        tmp_path,
        graph,
        (target,),
        _StaticAdapters({"layer.q_proj": adapter_weights}),
        quant_override={
            "mode": "ssfp",
            "nshare": 64,
            "man_bit": 8,
            "hidden_bit": False,
        },
    )

    module.bind_non_routed_lora_graph(graph, bindings, references)

    base = next(node for node in graph.node if node.output[0].endswith("_pre_lora"))
    lora = [
        node
        for node in graph.node
        if node.op_type == "Linear" and node.output[0].startswith("lora_")
    ]
    base_attrs = {
        attr.name: helper.get_attribute_value(attr) for attr in base.attribute
    }
    assert base_attrs["hmfp_weight_man_bit"] == 4
    assert len(lora) == 2
    for node in lora:
        attrs = {
            attr.name: helper.get_attribute_value(attr)
            for attr in node.attribute
        }
        assert attrs["mode"] == b"ssfp"
        assert attrs["hmfp_weight_man_bit"] == 8
        assert attrs["hmfp_weight_nshare"] == 64
        assert attrs["hmfp_weight_hidden_bit"] == 0


def test_expert_lora_quantization_uses_linear_aligned_layout() -> None:
    module = _load_expert_module()
    weight = np.arange(3 * 5 * 65, dtype=np.float32).reshape(3, 5, 65) / 100

    qweight, scale = module.quantize_routed_expert_lora_weight(weight)

    assert qweight.dtype == np.int8
    assert scale.dtype == np.float16
    assert qweight.shape == (3, 2, 64, 5)
    assert scale.shape == (3, 2, 1, 5)
    assert np.all(qweight[:, 1, 1:, :] == 0)


def test_expert_lora_quantization_supports_sefp() -> None:
    module = _load_expert_module()
    weight = np.arange(3 * 5 * 65, dtype=np.float32).reshape(3, 5, 65) / 100

    qweight, exponent = module.quantize_routed_expert_lora_weight(
        weight,
        quant_spec=module.AlignedLoRAQuantSpec(
            mode="sefp",
            nshare=64,
            man_bit=8,
            hidden_bit=True,
            rounding="RNE",
        ),
    )

    assert qweight.dtype == np.int8
    assert exponent.dtype == np.int16
    assert qweight.shape == (3, 2, 64, 5)
    assert exponent.shape == (3, 2, 1, 5)


def test_routed_lora_quant_specs_follow_moeblock_weight_attrs() -> None:
    module = _load_expert_module()
    node = helper.make_node(
        "MoeBlock",
        [f"input_{index}" for index in range(15)],
        ["output"],
        name="node_moe_block",
        domain="ai.houmo.xh2a",
        mode="sefp",
        hmfp_weight_man_bit=7,
        hmfp_weight_nshare=32,
        hmfp_weight_rounding="RNE",
        hmfp_weight_hidden_bit=1,
        hmfp_weight_man_bit_down=6,
    )

    specs = module.routed_lora_quant_specs(node)

    assert specs["gate"] == module.AlignedLoRAQuantSpec("sefp", 32, 7, True, "RNE")
    assert specs["up"] == specs["gate"]
    assert specs["down"] == module.AlignedLoRAQuantSpec("sefp", 32, 6, True, "RNE")


def test_routed_lora_binding_overrides_w4_base_with_w8_weights() -> None:
    module = _load_expert_module()
    config_module = importlib.import_module(
        "xhmodel_merak.xh_llm.models.macaron_v1_tall."
        "lora_export_config"
    )
    node = helper.make_node(
        "MoeBlock",
        [f"input_{index}" for index in range(15)],
        ["output"],
        name="node_moe_block",
        domain="ai.houmo.xh2a",
        mode="ssfp",
        hmfp_act_man_bit=8,
        hmfp_act_hidden_bit=1,
        hmfp_weight_exp_bit=16,
        hmfp_weight_man_bit=4,
        hmfp_weight_nshare=64,
        hmfp_weight_rounding="RNE",
        hmfp_weight_hidden_bit=0,
    )
    graph = helper.make_graph([node], "expert_lora", [], [])
    quantization = config_module.LoRAQuantization(
        quant_type="w8a8h0_ssfp",
        hmfp_spec={
            "mode": "ssfp",
            "man_bit": 8,
            "hidden_bit": False,
        },
        act_spec={
            "mode": "sefp",
            "man_bit": 8,
            "hidden_bit": True,
        },
    )

    bindings = module.build_routed_expert_lora_bindings(
        graph,
        1,
        quantization,
    )

    assert len(bindings) == 1
    expected = module.AlignedLoRAQuantSpec(
        mode="ssfp",
        nshare=64,
        man_bit=8,
        hidden_bit=False,
        rounding="RNE",
    )
    assert bindings[0].quant_override == expected
    assert set(bindings[0].quant_specs.values()) == {expected}


def test_expert_lora_injection_uses_28_canonical_initializer_inputs() -> None:
    module = _load_expert_module()
    node = helper.make_node(
        "MoeBlock",
        [f"input_{index}" for index in range(15)],
        ["output"],
        name="node_moe_block",
        domain="ai.houmo.xh2a",
        mode="ssfp",
        hmfp_weight_man_bit=4,
        hmfp_weight_nshare=64,
        hmfp_weight_rounding="RNE",
        hmfp_weight_hidden_bit=0,
    )
    graph = helper.make_graph([node], "expert_lora", [], [])
    qweight, scale = module.quantize_routed_expert_lora_weight(np.ones((3, 5, 65), dtype=np.float32))
    canonical_references = {}
    for names in module.routed_expert_lora_input_names(0).values():
        canonical_references[names[0]] = numpy_helper.from_array(qweight, names[0])
        canonical_references[names[1]] = numpy_helper.from_array(scale, names[1])

    modified = module.inject_routed_expert_lora_constants(
        graph,
        0,
        canonical_references,
        scaling=2.0,
    )

    assert modified == 1
    injected_node = graph.node[0]
    assert injected_node.op_type == "MoeBlock"
    assert len(injected_node.input) == 28
    assert injected_node.input[15] == ""
    assert injected_node.input[16:] == [
        name for pair in module.routed_expert_lora_input_names(0).values() for name in pair
    ]
    assert all(name == name.lower() for name in injected_node.input[16:])
    assert not graph.input
    assert {initializer.name for initializer in graph.initializer} == set(
        injected_node.input[16:]
    )
    attributes = {attribute.name: helper.get_attribute_value(attribute) for attribute in injected_node.attribute}
    assert attributes["lora_scaling"] == pytest.approx(2.0)
    assert "lora_weight_man_bit" not in attributes


def test_expert_lora_injection_writes_complete_quant_override() -> None:
    module = _load_expert_module()
    node = helper.make_node(
        "MoeBlock",
        [f"input_{index}" for index in range(15)],
        ["output"],
        name="node_moe_block",
        domain="ai.houmo.xh2a",
        mode="ssfp",
        hmfp_weight_man_bit=4,
        hmfp_weight_nshare=64,
        hmfp_weight_rounding="RNE",
        hmfp_weight_hidden_bit=0,
    )
    graph = helper.make_graph([node], "expert_lora", [], [])
    qweight, scale = module.quantize_routed_expert_lora_weight(
        np.ones((3, 5, 65), dtype=np.float32)
    )
    canonical_references = {}
    for names in module.routed_expert_lora_input_names(0).values():
        canonical_references[names[0]] = numpy_helper.from_array(
            qweight,
            names[0],
        )
        canonical_references[names[1]] = numpy_helper.from_array(
            scale,
            names[1],
        )
    quant_spec = module.AlignedLoRAQuantSpec(
        mode="sefp",
        nshare=64,
        man_bit=8,
        hidden_bit=True,
        rounding="RNE",
    )

    module.inject_routed_expert_lora_constants(
        graph,
        0,
        canonical_references,
        scaling=2.0,
        quant_spec=quant_spec,
    )

    attributes = {
        attribute.name: helper.get_attribute_value(attribute)
        for attribute in graph.node[0].attribute
    }
    assert {
        name: attributes[name]
        for name in module.LORA_QUANT_ATTRIBUTES
    } == {
        "lora_mode": b"sefp",
        "lora_weight_exp_bit": 16,
        "lora_weight_man_bit": 8,
        "lora_weight_nshare": 64,
        "lora_weight_rounding": b"RNE",
        "lora_weight_hidden_bit": 1,
    }


def test_final_text_graph_hashes_are_refreshed_after_lora_injection(
    tmp_path: Path,
) -> None:
    module = _load_pipeline_module()
    metadata_path = tmp_path / "golden_meta_info.json"
    metadata_path.write_text(
        json.dumps(
            {
                "prefill_hmonnx_md5": "stale",
                "decode_hmonnx_md5": "stale",
            }
        ),
        encoding="utf-8",
    )
    graph_bytes = {
        "prefill": b"final-prefill",
        "decode": b"final-decode",
    }
    for phase, content in graph_bytes.items():
        phase_dir = tmp_path / phase
        phase_dir.mkdir()
        (phase_dir / f"model_{phase}_with_act.onnx").write_bytes(content)

    config_module = importlib.import_module("xhmodel_merak.xh_llm.models.macaron_v1_tall." "lora_export_config")
    export_config = config_module.load_macaron_lora_export_config(
        {
            "macaron_lora": {
                "adapters": {"available": ["L0", "L1", "L2", "L3"]},
                "quantization": {
                    "non_routed": {"quant_scheme": "inherit"},
                    "routed_expert": {
                        "quant_scheme": "inherit",
                    },
                },
            }
        }
    )
    module._refresh_export_metadata(
        tmp_path,
        tmp_path / "runtime_lora" / "manifest.json",
        export_config,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for phase, content in graph_bytes.items():
        import hashlib

        expected = hashlib.md5(content, usedforsecurity=False).hexdigest()
        assert metadata[f"{phase}_hmonnx_md5"] == expected


def test_canonical_lora_is_appended_to_export_stage_external_data(
    tmp_path: Path,
) -> None:
    export_dir = tmp_path / "hmquant_xh2_macaron_v1_tall_w8a8"
    stage_dir = export_dir / "prefill"
    stage_dir.mkdir(parents=True)
    stage_data = stage_dir / f"{export_dir.name}_prefill_external_data"
    stage_data.write_bytes(b"base")
    bundle_dir = export_dir / "runtime_lora"
    bundle_dir.mkdir()
    writer = ConstantONNXWriter(bundle_dir / "L0.onnx")
    writer.add("lora_weight", np.array([1, 2, 3], dtype=np.int8))
    writer.close()

    references = append_canonical_lora_to_stage(
        export_dir,
        "prefill",
        bundle_dir / "L0.onnx",
    )

    metadata = {entry.key: entry.value for entry in references["lora_weight"].external_data}
    assert metadata["location"] == stage_data.name
    assert metadata["offset"] == "4"
    assert stage_data.read_bytes() == b"base\x01\x02\x03"
    assert not (stage_dir / "L0.onnx.data").exists()

    assert metadata == {
        "location": stage_data.name,
        "offset": "4",
        "length": "3",
    }


def test_expert_bundle_merges_into_four_same_schema_lora_onnx_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    non_expert_module = _load_module()
    expert_module = _load_expert_module()
    output_dir = tmp_path / "export"
    input_order = ["shared_a_qweight"]
    shared_packs = {
        adapter_name: {
            input_order[0]: np.full((1, 64, 2), adapter_index + 1, dtype=np.int8),
        }
        for adapter_index, adapter_name in enumerate(non_expert_module.ADAPTER_NAMES)
    }
    packed = {
        key: torch.full((4, 2, 3, 5), key_index + 1, dtype=torch.float16)
        for key_index, key in enumerate(expert_module.ROUTED_EXPERT_LORA_WEIGHT_KEYS)
    }
    monkeypatch.setattr(
        expert_module,
        "pack_expert_lora_layer",
        lambda adapters, layer_idx: packed,
    )

    with MacaronLoRAConstantBundle(output_dir) as bundle:
        bundle.add(
            input_order[0],
            {
                adapter_name: shared_packs[adapter_name][input_order[0]]
                for adapter_name in non_expert_module.ADAPTER_NAMES
            },
        )
        binding = expert_module.RoutedExpertLoRABinding(
            layer_idx=0,
            projection_specs=tuple(
                (
                    projection,
                    expert_module.DEFAULT_ROUTED_EXPERT_LORA_QUANT_SPEC,
                )
                for projection in ("gate", "up", "down")
            ),
        )
        expert_module.write_routed_expert_lora_constants(
            bundle,
            object(),
            (binding,),
        )
        artifact = bundle.close()

    manifest = read_manifest(artifact.manifest_path)
    assert len(manifest["input_order"]) == 13
    schemas = []
    for adapter_name in expert_module.ADAPTER_NAMES:
        adapter_path = artifact.manifest_path.parent / manifest["adapters"][adapter_name]["onnx"]
        model = onnx.load(adapter_path, load_external_data=True)
        schemas.append(
            [
                (
                    initializer.name,
                    initializer.data_type,
                    tuple(initializer.dims),
                )
                for initializer in model.graph.initializer
            ]
        )
        assert all(
            "L0" not in initializer.name and "L1" not in initializer.name for initializer in model.graph.initializer
        )
        expert_qweight = next(
            initializer for initializer in model.graph.initializer if initializer.name == "layer0_lora_gate_a_qweight"
        )
        assert len(expert_qweight.dims) == 4
    assert all(schema == schemas[0] for schema in schemas[1:])


def _minimal_macaron_export_inputs(tmp_path: Path):
    model_dir = tmp_path / "hf_model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "num_hidden_layers": 1,
                    "num_experts": 1,
                    "hidden_size": 4,
                    "moe_intermediate_size": 2,
                }
            }
        ),
        encoding="utf-8",
    )
    prefixes = {
        "q": "base_model.model.model.layers.0.self_attn.q_proj",
        "gate_up": ("base_model.model.model.layers.0.mlp.experts.base_layer"),
        "down": "base_model.model.model.layers.0.mlp.experts",
    }
    for index, adapter_name in enumerate(("L0", "L1", "L2", "L3")):
        adapter_dir = model_dir / "loras" / adapter_name
        adapter_dir.mkdir(parents=True)
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(
                {
                    "peft_type": "LORA",
                    "bias": "none",
                    "r": 2,
                    "lora_alpha": 4,
                    "target_parameters": [
                        "experts.gate_up_proj",
                        "experts.down_proj",
                    ],
                }
            ),
            encoding="utf-8",
        )
        value = float(index + 1)
        save_file(
            {
                f"{prefixes['q']}.lora_A.weight": torch.full((2, 4), value),
                f"{prefixes['q']}.lora_B.weight": torch.full((4, 2), value),
                f"{prefixes['gate_up']}.lora_A.weight": torch.full((2, 4), value),
                f"{prefixes['gate_up']}.lora_B.weight": torch.full((4, 2), value),
                f"{prefixes['down']}.lora_A.weight": torch.full((2, 2), value),
                f"{prefixes['down']}.lora_B.weight": torch.full((4, 2), value),
            },
            adapter_dir / "adapter_model.safetensors",
        )

    base_export_dir = tmp_path / "base_work"
    base_model_dir = base_export_dir / "hmquant_test"
    base_model_dir.mkdir(parents=True)
    (base_export_dir / "workflow.yaml").write_text("export: {}\n")
    (base_model_dir / "golden_meta_info.json").write_text(
        json.dumps(
            {
                "prefill_hmonnx_md5": "base",
                "decode_hmonnx_md5": "base",
            }
        )
    )
    qweight = "model.language_model.layers.0.self_attn.q_proj.qweight"
    for phase in ("prefill", "decode"):
        phase_dir = base_model_dir / phase
        phase_dir.mkdir()
        graph = helper.make_graph(
            [
                helper.make_node(
                    "Linear",
                    ["hidden", qweight],
                    ["linear_output"],
                    domain="ai.houmo.xh2a",
                    in_features=4,
                    out_features=4,
                    mode="ssfp",
                    hmfp_weight_nshare=64,
                    hmfp_weight_man_bit=8,
                    hmfp_weight_hidden_bit=0,
                ),
                helper.make_node(
                    "MoeBlock",
                    [f"moe_input_{item}" for item in range(15)],
                    ["moe_output"],
                    name="node_moe_block",
                    domain="ai.houmo.xh2a",
                    mode="ssfp",
                    hmfp_weight_nshare=64,
                    hmfp_weight_man_bit=8,
                    hmfp_weight_hidden_bit=0,
                ),
            ],
            phase,
            [],
            [],
            initializer=[
                numpy_helper.from_array(
                    np.ones((4, 4), dtype=np.float16),
                    qweight,
                )
            ],
        )
        onnx.save(
            helper.make_model(
                graph,
                opset_imports=[
                    helper.make_opsetid("", 18),
                    helper.make_opsetid("ai.houmo.xh2a", 1),
                ],
            ),
            phase_dir / f"hmquant_test_{phase}_with_act.onnx",
        )
        (phase_dir / f"{base_model_dir.name}_{phase}_external_data").write_bytes(b"base")

    config_module = importlib.import_module("xhmodel_merak.xh_llm.models.macaron_v1_tall." "lora_export_config")
    export_config = config_module.load_macaron_lora_export_config(
        {
            "macaron_lora": {
                "adapters": {"available": ["L0", "L1", "L2", "L3"]},
                "quantization": {
                    "non_routed": {"quant_scheme": "inherit"},
                    "routed_expert": {
                        "quant_scheme": "inherit",
                    },
                },
            }
        }
    )
    return model_dir, base_export_dir, base_model_dir, export_config


def test_macaron_pipeline_atomically_builds_complete_export(
    tmp_path: Path,
) -> None:
    pipeline = _load_pipeline_module()
    model_dir, base_work, base_model, export_config = _minimal_macaron_export_inputs(tmp_path)
    final_work = tmp_path / "final_work"

    final_model = pipeline.inject_macaron_lora_export(
        model_dir=model_dir,
        base_export_dir=base_work,
        base_model_export_dir=base_model,
        final_export_dir=final_work,
        export_config=export_config,
    )

    assert final_model == final_work / base_model.name
    assert final_model.is_dir()
    assert not (tmp_path / ".final_work.tmp").exists()
    assert base_model.is_dir()
    manifest = read_manifest(final_model / "runtime_lora" / "manifest.json")
    assert len(manifest["input_order"]) == 16
    for phase in ("prefill", "decode"):
        graph_path = next((final_model / phase).glob("*_with_act.onnx"))
        graph = onnx.load(graph_path, load_external_data=False).graph
        lora_names = [
            initializer.name for initializer in graph.initializer if initializer.name in manifest["input_order"]
        ]
        assert lora_names == manifest["input_order"]
        moe = _load_expert_module().moeblock_node_for_layer(graph, 0)
        assert moe is not None and len(moe.input) == 28
        assert (
            len([node for node in graph.node if node.op_type == "Linear" and node.output[0].startswith("lora_")]) == 2
        )


def test_macaron_pipeline_removes_staging_export_on_failure(
    tmp_path: Path,
) -> None:
    pipeline = _load_pipeline_module()
    model_dir, base_work, base_model, export_config = _minimal_macaron_export_inputs(tmp_path)
    final_work = tmp_path / "final_work"

    decode_graph_path = next(
        (base_model / "decode").glob("*_with_act.onnx")
    )
    decode_model = onnx.load(
        decode_graph_path,
        load_external_data=False,
    )
    linear = next(
        node
        for node in decode_model.graph.node
        if node.op_type == "Linear"
    )
    for attribute in linear.attribute:
        if attribute.name == "mode":
            attribute.s = b"sefp"
        elif attribute.name == "hmfp_weight_hidden_bit":
            attribute.i = 1
    onnx.save(decode_model, decode_graph_path)

    with pytest.raises(
        ValueError,
        match="Prefill and Decode non-routed LoRA bindings differ",
    ):
        pipeline.inject_macaron_lora_export(
            model_dir=model_dir,
            base_export_dir=base_work,
            base_model_export_dir=base_model,
            final_export_dir=final_work,
            export_config=export_config,
        )

    assert not final_work.exists()
    assert not (tmp_path / ".final_work.tmp").exists()
    assert base_model.is_dir()


def test_python_hmonnx_overlay_binds_one_quantized_adapter(
    tmp_path: Path,
) -> None:
    pipeline = _load_pipeline_module()
    overlay_module = _load_overlay_module()
    model_dir, base_work, base_model, export_config = _minimal_macaron_export_inputs(
        tmp_path
    )
    final_model = pipeline.inject_macaron_lora_export(
        model_dir=model_dir,
        base_export_dir=base_work,
        base_model_export_dir=base_model,
        final_export_dir=tmp_path / "final_work",
        export_config=export_config,
    )
    manifest = read_manifest(final_model / "runtime_lora" / "manifest.json")

    overlay = overlay_module.create_macaron_lora_overlay(
        final_model,
        "L3",
        tmp_path / "l3_overlay",
    )

    assert overlay_module.create_macaron_lora_overlay(
        final_model,
        "L3",
        overlay,
    ) == overlay
    assert (overlay / "runtime_lora").is_symlink()
    for phase in ("prefill", "decode"):
        graph_path = next((overlay / phase).glob("*_with_act.onnx"))
        graph = onnx.load(graph_path, load_external_data=False).graph
        initializers = {item.name: item for item in graph.initializer}
        for name in manifest["input_order"]:
            locations = {
                item.value
                for item in initializers[name].external_data
                if item.key == "location"
            }
            assert locations == {"../runtime_lora/L3.onnx.data"}
