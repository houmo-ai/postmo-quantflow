from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import xhquant.nn as xhnn
from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import (
    FULL_ATTENTION,
    LINEAR_ATTENTION,
)
from xhmodel_merak.xh_llm.models.qwen4_exp.checkpoint import _build_qwen4_exp_gptq_export_device_map
from xhmodel_merak.xh_llm.models.qwen4_exp.golden_runtime import run_qwen4_exp_hmonnx_golden
from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import (
    Qwen4ExpExportableGatedDeltaNet,
    inspect_qwen4_exp_export_capability,
)
from xhmodel_merak.xh_llm.models.qwen4_exp.ple_sharding import (
    group_qwen4_exp_ple_shard_placement_tags,
    shard_qwen4_exp_ple_gather,
)
from xhmodel_merak.xh_llm.models.qwen4_exp.static_export import (
    build_qwen4_exp_serving_sample_inputs,
    export_qwen4_exp_hmonnx,
)


def _flash_next_text_config() -> SimpleNamespace:
    return SimpleNamespace(
        model_type="qwen4_exp_text",
        num_hidden_layers=2,
        layer_types=[LINEAR_ATTENTION, FULL_ATTENTION],
        hidden_size=32,
        hc_count=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        indexer_head_dim=4,
        indexer_budget=8,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_conv_kernel_dim=4,
        ple_conv_kernel_size=4,
        ngram_size=3,
        ple_layer_ids=[1],
    )


def test_hmonnx_preflight_rejects_sdk_without_public_export_api(monkeypatch):
    import xhquant.api
    from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import inspect_qwen4_exp_export_capability

    monkeypatch.delattr(xhquant.api, "to_export_hmonnx_v2")
    capability = inspect_qwen4_exp_export_capability(_flash_next_text_config(), require_hmonnx_runtime=True)
    assert not capability.supported
    assert any("public v2 export API" in reason for reason in capability.reasons)


def test_ple_gather_sharding_reuses_external_bytes_and_preserves_outputs(tmp_path: Path) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    from onnx.reference import ReferenceEvaluator

    weights = np.arange(16, dtype=np.float16).reshape(8, 2)
    weight_name = "model.layers.1.ple.ple_embedding.ngram_embedding_weight"
    graph = helper.make_graph(
        [helper.make_node("Gather", [weight_name, "indices"], ["output"], name="ple_gather", axis=0)],
        "ple_gather",
        [helper.make_tensor_value_info("indices", TensorProto.INT32, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 4, 2])],
        [numpy_helper.from_array(weights, name=weight_name)],
    )
    source_path = tmp_path / "source.onnx"
    onnx.save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]),
        source_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="source.onnx.data",
        size_threshold=0,
    )
    source_data_path = tmp_path / "source.onnx.data"
    source_stat = source_data_path.stat()
    output_path = tmp_path / "sharded.onnx"

    result = shard_qwen4_exp_ple_gather(source_path, output_path, shard_count=4)

    assert result.shard_rows == (2, 2, 2, 2)
    assert result.shard_nbytes == (8, 8, 8, 8)
    assert source_data_path.stat().st_ino == source_stat.st_ino
    assert source_data_path.stat().st_size == source_stat.st_size

    metadata = onnx.load(output_path, load_external_data=False)
    initializers = {initializer.name: initializer for initializer in metadata.graph.initializer}
    assert weight_name not in initializers
    shard_initializers = [
        initializer for name, initializer in initializers.items() if name.startswith(f"{weight_name}.row_shard_")
    ]
    assert [list(initializer.dims) for initializer in shard_initializers] == [[2, 2]] * 4
    assert [node.op_type for node in metadata.graph.node].count("Gather") == 4
    assert [node.op_type for node in metadata.graph.node].count("Clip") == 0
    assert [node.op_type for node in metadata.graph.node].count("Where") == 7
    tags = [node for node in metadata.graph.node if node.op_type == "Tag"]
    assert len(tags) == 4
    assert [node.domain for node in tags] == ["ai.houmo.xh2a"] * 4

    original = onnx.load(source_path, load_external_data=True)
    sharded = onnx.load(output_path, load_external_data=True)
    for node in sharded.graph.node:
        if node.op_type == "Tag":
            node.op_type = "Identity"
            node.domain = ""
            del node.attribute[:]
    indices = np.asarray([[0, 2, 5, 7]], dtype=np.int32)
    expected = ReferenceEvaluator(original).run(None, {"indices": indices})[0]
    actual = ReferenceEvaluator(sharded).run(None, {"indices": indices})[0]
    np.testing.assert_array_equal(actual, expected)


def test_ple_shard_placement_tags_group_adjacent_shards(tmp_path: Path) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    weight_name = "model.layers.1.ple.ple_embedding.ngram_embedding_weight"
    source_path = tmp_path / "source.onnx"
    sharded_path = tmp_path / "sharded.onnx"
    grouped_path = tmp_path / "grouped.onnx"
    graph = helper.make_graph(
        [helper.make_node("Gather", [weight_name, "indices"], ["output"], name="ple_gather", axis=0)],
        "ple_gather",
        [helper.make_tensor_value_info("indices", TensorProto.INT32, [1, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 4, 2])],
        [numpy_helper.from_array(np.arange(16, dtype=np.float16).reshape(8, 2), name=weight_name)],
    )
    onnx.save_model(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]),
        source_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="group_source.onnx.data",
        size_threshold=0,
    )
    shard_qwen4_exp_ple_gather(source_path, sharded_path, shard_count=4)

    result = group_qwen4_exp_ple_shard_placement_tags(
        sharded_path,
        grouped_path,
        placement_group_size=2,
    )
    grouped = onnx.load(grouped_path, load_external_data=False)
    tag_types = []
    for node in grouped.graph.node:
        if node.op_type == "Tag":
            attributes = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
            tag_types.append(attributes["tag_type"])

    assert result.shard_count == 4
    assert result.llm_boundary_shards == (1, 3)
    assert tag_types == [b"PLE", b"LLM", b"PLE", b"LLM"]


def test_gptq_export_device_map_keeps_oversized_ple_table_on_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    config = SimpleNamespace(
        text_config=SimpleNamespace(num_hidden_layers=5, ple_layer_ids=[2]),
    )

    device_map = _build_qwen4_exp_gptq_export_device_map(config)

    assert device_map[""] == "cpu"
    assert device_map["model.language_model.layers.0"] == "cuda:0"
    assert device_map["model.language_model.layers.1"] == "cpu"
    assert device_map["model.language_model.layers.2"] == "cuda:2"
    assert "model.language_model.layers.1.ple.ple_embedding.ngram_embedding" not in device_map
    assert device_map["model.visual"] == "cuda:2"


def test_preflight_accepts_ple_qsa_and_hyperconnection_architecture() -> None:
    capability = inspect_qwen4_exp_export_capability(_flash_next_text_config(), require_hmonnx_runtime=False)

    assert capability.supported is True
    assert capability.has_transformers_qwen4_exp is True
    assert capability.reasons == ()
    assert capability.details["has_ple_layers"] is True
    assert capability.details["has_qsa_indexer_layers"] is True
    assert capability.cache_contract is not None


def test_gated_delta_net_uses_runtime_valid_length(monkeypatch) -> None:
    captured = {}

    def fake_forward(
        _module,
        hidden_states,
        conv_cache, recurrent_state, linear_attn_mask, current_input_length,
    ):
        captured["current_input_length"] = current_input_length
        return hidden_states, conv_cache, recurrent_state

    monkeypatch.setattr(Qwen4ExpExportableGatedDeltaNet, "_forward_merged", fake_forward)
    module = Qwen4ExpExportableGatedDeltaNet.__new__(Qwen4ExpExportableGatedDeltaNet)
    torch.nn.Module.__init__(module)
    module.layer_idx = 0
    conv_cache = torch.zeros((1, 4, 3))
    recurrent_state = torch.zeros((1, 2, 4, 4))
    cache = SimpleNamespace(
        current_input_length=torch.tensor([3], dtype=torch.int32),
        layers=[SimpleNamespace(conv_states=[conv_cache], recurrent_states=[recurrent_state])],
    )

    Qwen4ExpExportableGatedDeltaNet.forward(
        module,
        torch.zeros((1, 64, 8)),
        cache_params=cache,
    )

    assert captured["current_input_length"] is cache.current_input_length


def test_golden_runner_casts_inputs_and_writes_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeInfo:
        def __init__(self, dtype: torch.dtype):
            self.dtype = dtype

    class FakeSession:
        def __init__(self, path: str):
            self.path = path
            self.exec_device = None
            self.save_golden = False
            self.golden_dir = None
            self.step = None

        def initialize(self) -> None:
            pass

        def get_input_names(self) -> list[str]:
            return ["tokens", "state"]

        def get_input(self, name: str) -> FakeInfo:
            return FakeInfo(torch.int32 if name == "tokens" else torch.float16)

        def get_output_names(self) -> list[str]:
            return ["hidden", "present_state"]

        def run(self, feed):
            assert feed["tokens"].dtype == torch.int32
            assert feed["state"].dtype == torch.float16
            return feed["state"] + 1, feed["state"]

    import xhquant.api

    monkeypatch.setattr(xhquant.api, "HMONNXGoldenInference", FakeSession)
    outputs = run_qwen4_exp_hmonnx_golden(
        tmp_path / "model.onnx",
        (torch.tensor([1], dtype=torch.int64), torch.tensor([2.0], dtype=torch.float32)),
        tmp_path / "golden",
    )

    assert tuple(outputs) == ("hidden", "present_state")
    summary = json.loads((tmp_path / "golden/step_0/qwen4_exp_runtime_summary.json").read_text())
    assert [item["dtype"] for item in summary["inputs"]] == ["int32", "float16"]


@pytest.mark.parametrize(
    "verify_width,with_ple",
    [(2, False), (4, True), (4, "packed")],
    ids=["recurrent2_gdr", "padded4_gdr_and_ple", "padded4_packed_gdr"],
)
def test_verify_recurrent_graph_matches_sdk_decode_steps(tmp_path, monkeypatch, with_ple, verify_width):
    import numpy as np
    import onnx
    from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

    from xhmodel_merak.exec_backends import HMONNXInferenceConfig
    from xhmodel_merak.xh_llm.models.qwen4_exp.host_ple_runtime import Qwen4ExpHostPLEInferenceV2
    from xhmodel_merak.xh_llm.models.qwen4_exp.xhquant_extensions import qwen4_exp_xhquant_extension_scope
    from xhmodel_merak.xh_llm.utils import unfold_args

    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "0")
    monkeypatch.setattr(torch._dynamo.config, "disable", False)
    packed_projection = with_ple == "packed"
    with_ple = with_ple is True
    torch.manual_seed(53)
    config = Qwen4ExpTextConfig(
        max_position_embeddings=16,
        vocab_size=32, hidden_size=64 if packed_projection else 16, num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, indexer_head_dim=8, linear_key_head_dim=4, linear_value_head_dim=4,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_conv_kernel_dim=4,
        moe_intermediate_size=8, shared_expert_intermediate_size=8, num_experts=2,
        num_experts_per_tok=1, hc_count=2, hc_lowrank=4, layer_types=["linear_attention"],
        indexer_n_heads=2, indexer_kv_heads=1, indexer_budget=4, indexer_compress_ratio=2,
        ple_layer_ids=[], rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.5},
    )
    if with_ple:
        config.ple_layer_ids = [1]
        config.ple_embed_dim = 64
        config.ple_conv_kernel_size = 4
        config.ngram_size = 3
        config.heads_per_ngram = 1
        config.ngram_vocab_size_base = 5
        config.make_ngram_vocab_size_divisible_by = 8
        config.split_ngram_parts = 1
        config.eos_token_id = 0
    native = Qwen4ExpForCausalLM(config).half().eval()
    if packed_projection:
        native.model.layers[0].linear_attn.in_proj_qkv = xhnn.GPTQPackedLinear(xhnn.GPTQPackedWeight(
            qweight=torch.full((8, 24), 0x11111111, dtype=torch.int32),
            qzeros=torch.zeros(1, 3, dtype=torch.int32), scales=torch.full((1, 24), 0.125).half(),
            g_idx=torch.zeros(64, dtype=torch.int32), bits=4, group_size=64,
            in_features=64, out_features=24, qzero_format=2,
        ))
    retained = {name: (parameter, parameter.data_ptr(), parameter.detach().clone())
                for name, parameter in native.named_parameters()
                if "hyper_connection" in name or name == "lm_head.weight" or "ngram_embedding" in name}
    retained_classes = {name: type(module) for name, module in native.named_modules()
                        if "hyper_connection" in name and isinstance(module, torch.nn.Linear)}
    sessions, samples = {}, {}
    for width in (verify_width, 1):
        model = native
        _, _, sample = build_qwen4_exp_serving_sample_inputs(model, input_sequence_length=width, context_length=8)
        exported = export_qwen4_exp_hmonnx(
            model, tmp_path / f"seq{width}", text_config=config, sample_inputs=sample,
            input_sequence_length=width, cache_length=8, quant_device="cpu",
            quant_config={"quant_type": "w16a16h1_sefp", "preserve_model_dtype": True},
            use_fused_gdr=True, linear_attention_mode="recurrent", return_hidden_states=True, ple_quant_type="fp16",
        )
        current = dict(native.named_parameters())
        for name, (owner, pointer, original) in retained.items():
            assert current[name] is owner and owner.device.type == "cpu" and owner.data_ptr() == pointer
            torch.testing.assert_close(owner, original, rtol=0, atol=0)
        assert type(native.lm_head) is torch.nn.Linear
        current_modules = dict(native.named_modules())
        for name, cls in retained_classes.items():
            assert type(current_modules[name]) is cls
        graph = onnx.load(exported.hmonnx_path, load_external_data=False)
        assert not any(node.op_type == "Conv1d" for node in graph.graph.node)
        convolutions = [node for node in graph.graph.node if node.op_type == "Conv2d"]
        assert len(convolutions) == 1 + int(with_ple)
        geometry = [{attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
                    for node in convolutions]
        assert any(item["dilations"] == [1, 1] and item["pads"] == [3, 0] for item in geometry)
        if with_ple:
            # The SDK's initial frontend pass may expand dilation into zero
            # taps while preserving the convolution geometry.
            ple_node = next(node for node, item in zip(convolutions, geometry, strict=True)
                            if item["pads"] == [0, 0])
            ple_geometry = {attr.name: onnx.helper.get_attribute_value(attr) for attr in ple_node.attribute}
            # SDK versions may retain native dilation or expand it into
            # zero taps. Both must describe the same ten-position PLE window.
            dilation, width_dilation = ple_geometry["dilations"]
            kernel, kernel_width = ple_geometry["kernel_shape"]
            assert width_dilation == kernel_width == 1
            assert (kernel - 1) * dilation + 1 == 10
            weight = next(value for value in graph.graph.initializer if value.name == ple_node.input[3])
            taps = onnx.numpy_helper.to_array(weight, base_dir=str(Path(exported.hmonnx_path).parent))
            expanded = np.zeros((*taps.shape[:2], 10, 1), dtype=taps.dtype)
            expanded[:, :, ::dilation, :] = taps
            assert not expanded[:, :, [1, 2, 4, 5, 7, 8], :].any()
        assert sum(node.op_type == "GDRRecurrentScan" for node in graph.graph.node) == width
        assert not any(node.op_type in {"GDRChunkScan", "GDRBlockTriInverse"} for node in graph.graph.node)
        with qwen4_exp_xhquant_extension_scope():
            session = Qwen4ExpHostPLEInferenceV2(str(exported.hmonnx_path), HMONNXInferenceConfig(exec_devices=["cpu"]))
            session.initialize()
        sessions[width] = session
        samples[width] = dict(zip(session.get_input_names(), unfold_args(sample), strict=True))
    embeds = torch.randn(1, verify_width + 1, config.hidden_size).half()
    token_ids = torch.arange(3, verify_width + 4, dtype=torch.int32).unsqueeze(0)
    from xhmodel_merak.xh_llm.mrope import MROPE_CURRENT_POSITION_NAMES, MROPE_POOL_POSITION_NAMES

    data_names = {"inputs_embeds", "past_seq_length", "current_input_length", "ple_input_ids",
                  *MROPE_CURRENT_POSITION_NAMES, *MROPE_POOL_POSITION_NAMES}
    state_names = [name for name in samples[1] if name not in data_names]
    initial = {name: value.clone() for name, value in samples[1].items() if name in state_names}
    initial['delta_recurrent_state_0'] = torch.randn_like(initial['delta_recurrent_state_0']) * 0.1
    def execute(width, offset, states, valid_length=None):
        valid_length = width if valid_length is None else valid_length
        data = dict(samples[width], **states)
        data.update(inputs_embeds=embeds[:, offset:offset + width],
                    past_seq_length=torch.tensor([offset], dtype=torch.int32),
                    current_input_length=torch.tensor([valid_length], dtype=torch.int32),
                    ple_input_ids=token_ids[:, offset:offset + width])
        for name in MROPE_CURRENT_POSITION_NAMES:
            data[name] = torch.arange(offset, offset + width, dtype=torch.int32).unsqueeze(0)
        for name in MROPE_POOL_POSITION_NAMES:
            data[name] = (torch.arange((width + 3) // 4, dtype=torch.int32) * 4 + offset // 4 * 4).unsqueeze(0)
        # This regression compares SDK
        # arithmetic/state semantics rather than Inductor's C++ code generation.
        # Scope the runtime-only disable so machines without g++ remain usable.
        with torch._dynamo.config.patch(disable=True):
            result = dict(zip(sessions[width].get_output_names(), sessions[width].run(data), strict=True))
        # Match serving commit_outputs' copy into the original cache dtype.
        committed = {name: result[f'present_{name}'].to(initial[name].dtype).clone() for name in state_names}
        return result, committed

    def assert_states(actual, expected):
        for name in state_names:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    # A width-four verify graph still receives two real positions for the
    # current single-candidate policy. Padding must leave recurrent/conv/PLE
    # state unchanged, including when the next single-token decode consumes it.
    for valid_length in sorted({2, verify_width}):
        batch, batch_state = execute(
            verify_width, 0, {name: value.clone() for name, value in initial.items()}, valid_length,
        )
        sequential_state = {name: value.clone() for name, value in initial.items()}
        sequential = []
        for offset in range(valid_length):
            result, sequential_state = execute(1, offset, sequential_state)
            sequential.append(result)
        for name in ('logits', 'target_hidden_states'):
            torch.testing.assert_close(
                batch[name][:, :valid_length], torch.cat([result[name] for result in sequential], dim=1),
                rtol=0, atol=0,
            )
        assert_states(batch_state, sequential_state)
        after_batch, batch_state = execute(1, valid_length, batch_state)
        after_sequential, sequential_state = execute(1, valid_length, sequential_state)
        for name in ('logits', 'target_hidden_states'):
            torch.testing.assert_close(after_batch[name], after_sequential[name], rtol=0, atol=0)
        assert_states(batch_state, sequential_state)
    sessions.clear()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width,group_size", [(16, None), (10240, 2560)])
def test_qwen4_rmsnorm_preserves_native_grouped_fp32_affine(dtype, width, group_size):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRMSNorm

    from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import Qwen4ExpExportableRMSNorm

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(34)
        native = Qwen4ExpTextRMSNorm(width, group_size=group_size, eps=1e-5).to(dtype).eval()
        native.weight.data.uniform_(-0.9, 0.9)
        value = torch.randn(1, 2, width, dtype=dtype)
        if group_size is not None:
            # Distinct HC stream magnitudes reject a whole-10240 reduction.
            value = (value.reshape(1, 2, 4, group_size)
                     * torch.tensor([0.03125, 0.5, 8, 128], dtype=dtype).reshape(1, 1, 4, 1)).flatten(-2)
    expected = native(value)
    adapted = Qwen4ExpExportableRMSNorm(native).eval()
    assert adapted.weight is native.weight
    assert adapted.norm.weight.shape == (group_size or width,)
    assert torch.equal(adapted.norm.weight, torch.ones_like(adapted.norm.weight))
    assert dict(adapted.named_parameters()) == {"weight": native.weight}
    assert set(adapted.state_dict()) == {"weight"}
    torch.testing.assert_close(adapted(value), expected, rtol=0, atol=0)
    assert adapted(value).dtype == dtype
    # Reapplying dtype placement must keep the unit scale harmless too.
    torch.testing.assert_close(adapted.to(dtype)(value), expected, rtol=0, atol=0)


def test_qwen4_rmsnorm_adapter_is_idempotent_and_preserves_checkpoint_layout():
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextRMSNorm,
        Qwen4ExpTextRMSNormGated,
    )

    from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import (
        Qwen4ExpExportableRMSNorm,
        Qwen4ExpExportableRMSNormGated,
        adapt_qwen4_exp_rms_norms,
    )

    model = torch.nn.ModuleDict({
        "plain": Qwen4ExpTextRMSNorm(8),
        "nested": torch.nn.ModuleList([Qwen4ExpTextRMSNorm(16, group_size=4)]),
        "gated": Qwen4ExpTextRMSNormGated(8),
    }).half().eval()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    parameters = dict(model.named_parameters())
    assert adapt_qwen4_exp_rms_norms(model) == ("plain", "nested.0", "gated")
    assert isinstance(model["plain"], Qwen4ExpExportableRMSNorm)
    assert isinstance(model["gated"], Qwen4ExpExportableRMSNormGated)
    assert adapt_qwen4_exp_rms_norms(model) == ()
    assert set(model.state_dict()) == set(state)
    assert all(dict(model.named_parameters())[name] is value for name, value in parameters.items())
    model.load_state_dict(state, strict=True)
    model.float()
    assert model["gated"].norm.weight is model["gated"].weight
    assert set(model.state_dict()) == set(state)


@pytest.mark.parametrize("group_size,gate", [(None, None), (4, None), (None, "sigmoid"), (None, "silu")])
def test_qwen4_rmsnorm_real_sdk_export_and_replay(tmp_path, group_size, gate):
    import onnx
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRMSNorm, Qwen4ExpTextRMSNormGated

    from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import (
        Qwen4ExpExportableRMSNorm,
        Qwen4ExpExportableRMSNormGated,
    )
    from xhquant.api import (
        ConfigDict,
        FrontendType,
        PrecisionMode,
        ptq_quantize,
        to_export_graph,
        to_export_hmonnx_v2,
        to_frontend_graph,
        to_quant_graph,
    )
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    torch.manual_seed(34)
    native = (Qwen4ExpTextRMSNorm(16, group_size=group_size) if gate is None
              else Qwen4ExpTextRMSNormGated(16, activation=gate)).half().eval()
    native.weight.data.uniform_(-0.75, 0.75)
    adapted = (Qwen4ExpExportableRMSNorm(native) if gate is None
               else Qwen4ExpExportableRMSNormGated(native)).eval()
    sample = (torch.randn(1, 2, 16).half(),)
    if gate is not None:
        sample += (torch.randn_like(sample[0]),)
    frontend = to_frontend_graph(adapted, FrontendType.TorchFX, sample)
    norms = [module for module in frontend.modules() if isinstance(module, xhnn.RMSNorm)]
    assert len(norms) == 1 and norms[0].weight.shape == (group_size or 16,)
    quant = to_quant_graph(frontend, "xh2a", ConfigDict({"quant_type": "w16a16h1_sefp", "preserve_model_dtype": True}))
    ptq_quantize(quant, [list(sample)], PrecisionMode.ALIGNED, ["cpu"], infer_shape=False)
    quant.fixed()
    exported = to_export_graph(quant, list(sample))
    path = tmp_path / "norm.onnx"
    to_export_hmonnx_v2(
        exported,
        sample,
        str(path),
        cfg=ConfigDict(input_names=("values",) if gate is None else ("values", "gate"), output_names=("normalized",)),
        save_as_external_data=True,
    )
    graph = onnx.load(path, load_external_data=False)
    norm, = [node for node in graph.graph.node if node.op_type == "RMSNorm"]
    attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in norm.attribute}
    assert attrs["axis"] == -1 and attrs["force_fp32"] == 1
    infos = {value.name: value for value in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
    # XH2 QCast lowers the FP32 reference boundary to its FP16 deployment
    # dtype. The fused operator retains FP32 variance accumulation.
    assert infos[norm.input[0]].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    assert infos[norm.output[0]].type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
    weight = next(item for item in graph.graph.initializer if item.name == norm.input[1])
    assert list(weight.dims) == [group_size or 16]
    assert not any(node.op_type in {"Pow", "ReduceMean", "Sqrt"} for node in graph.graph.node)
    session = HMONNXInferenceV2(str(path), HMONNXInferenceConfig(exec_devices=["cpu"]))
    # Replay an independent input after tracing, including distinct HC groups.
    value = sample[0] * torch.linspace(0.125, 4, 16).half()
    with torch._dynamo.config.patch(disable=True):
        result = session.run({"values": value, **({"gate": sample[1]} if gate is not None else {})})
    actual = result[0] if isinstance(result, (tuple, list)) else result
    assert actual.dtype == torch.float16
    assert actual.shape == value.shape and torch.isfinite(actual).all()


@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_qwen4_gated_rmsnorm_preserves_native_scale_gate_and_checkpoint(dtype, activation):
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRMSNormGated

    from xhmodel_merak.xh_llm.models.qwen3_5._hybrid_gated_delta_net import HybridRMSNormGatedMixin
    from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import Qwen4ExpExportableRMSNormGated

    torch.manual_seed(68)
    native = Qwen4ExpTextRMSNormGated(16, activation=activation).to(dtype).eval()
    native.weight.data.uniform_(-1.5, 1.5)
    value = torch.randn(1, 3, 2, 16, dtype=dtype)
    value[:, :, 0] *= 0.03125
    value[:, :, 1] *= 16
    gate = torch.randn_like(value) * 2
    adapted = Qwen4ExpExportableRMSNormGated(native)
    assert isinstance(adapted, HybridRMSNormGatedMixin)
    assert adapted.weight is native.weight and adapted.norm.weight is native.weight
    assert set(adapted.state_dict()) == {"weight"}
    torch.testing.assert_close(adapted(value, gate), native(value, gate), rtol=0, atol=0)


# RoPE deployment contracts: the SDK owns half-split rotation, while Qwen
# retains its partial rotary prefix and the four-dimensional table layout.
_QWEN4_ROPE_CASES = [
    ((1, 24, 2, 256), (1, 2, 64), 1),
    ((1, 2, 2, 256), (1, 2, 64), 1),
    ((1, 2, 4, 128), (1, 2, 64), 2),
    ((1, 1, 3, 128), (1, 3, 64), 1),
    ((1, 3, 2, 64), (1, 2, 64), 1),
    ((1, 2, 3, 128), (1, 2, 3, 64), None),
]
_QWEN4_ROPE_IDS = ["main_query", "main_key", "index_query", "block_key", "dense", "query_specific_blocks"]


class _Qwen4PartialRopeProbe(torch.nn.Module):
    def __init__(self, table_axis):
        super().__init__()
        self.table_axis = table_axis
        self.rope = xhnn.Rope()

    def forward(self, values, cos, sin):
        from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import _apply_rotary

        return _apply_rotary(values, cos, sin, unsqueeze_dim=self.table_axis, rope=self.rope)


@pytest.mark.parametrize("shape,table_shape,table_axis", _QWEN4_ROPE_CASES, ids=_QWEN4_ROPE_IDS)
def test_qwen4_partial_rope_matches_native_half_split(shape, table_shape, table_axis):
    torch.manual_seed(83)
    values, cos, sin = torch.randn(shape).half(), torch.randn(table_shape).half(), torch.randn(table_shape).half()
    tables = (cos, sin) if table_axis is None else (cos.unsqueeze(table_axis), sin.unsqueeze(table_axis))
    width = table_shape[-1]
    rotary, tail = values[..., :width], values[..., width:]
    expected = torch.cat((rotary * tables[0]
                          + torch.cat((-rotary[..., width // 2:], rotary[..., :width // 2]), -1) * tables[1],
                          tail), -1)
    model = _Qwen4PartialRopeProbe(table_axis)
    torch.testing.assert_close(model(values, cos, sin), expected, rtol=0, atol=0)
    torch.testing.assert_close(model(values, cos, sin)[..., width:], tail, rtol=0, atol=0)
    assert not model.state_dict()


@pytest.mark.parametrize(
    "shape,table_shape,table_axis",
    [
        pytest.param(*_QWEN4_ROPE_CASES[0], id="main_query"),
        pytest.param(*_QWEN4_ROPE_CASES[2], id="index_query"),
        pytest.param(*_QWEN4_ROPE_CASES[4], id="dense"),
        pytest.param(*_QWEN4_ROPE_CASES[5], id="query_specific_blocks"),
    ],
)
def test_qwen4_partial_rope_real_sdk_hmonnx_replay(tmp_path, shape, table_shape, table_axis):
    import onnx

    from xhquant.api import (
        ConfigDict,
        FrontendType,
        PrecisionMode,
        ptq_quantize,
        to_export_graph,
        to_export_hmonnx_v2,
        to_frontend_graph,
        to_quant_graph,
    )
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    torch.manual_seed(83)
    sample = (torch.randn(shape).half(), torch.randn(table_shape).half(), torch.randn(table_shape).half())
    model = _Qwen4PartialRopeProbe(table_axis).eval()
    frontend = to_frontend_graph(model, FrontendType.TorchFX, sample)
    assert sum(isinstance(module, xhnn.Rope) for module in frontend.modules()) == 1
    quant = to_quant_graph(frontend, "xh2a", ConfigDict({"quant_type": "w16a16h1_sefp", "preserve_model_dtype": True}))
    ptq_quantize(quant, [list(sample)], PrecisionMode.ALIGNED, ["cpu"], infer_shape=False)
    quant.fixed()
    exported = to_export_graph(quant, list(sample))
    path = tmp_path / "partial_rope.onnx"
    to_export_hmonnx_v2(
        exported,
        sample,
        str(path),
        cfg=ConfigDict(input_names=("values", "cos", "sin"), output_names=("rotated",)),
        save_as_external_data=True,
    )
    graph = onnx.load(path, load_external_data=False)
    rope, = [node for node in graph.graph.node if node.op_type == "Rope"]
    infos = {value.name: value for value in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
    assert rope.domain == "ai.houmo.xh2a"
    assert not any(node.op_type in {"Neg", "Mul", "Add"} for node in graph.graph.node)
    for name in rope.input:
        tensor = infos[name].type.tensor_type
        assert tensor.elem_type == onnx.TensorProto.FLOAT16 and len(tensor.shape.dim) == 4
        assert tensor.shape.dim[-1].dim_value == table_shape[-1]
    session = HMONNXInferenceV2(str(path), HMONNXInferenceConfig(exec_devices=["cpu"]))
    replay = (sample[0] * 0.5, sample[1] * 0.25, sample[2] * 0.75)
    with torch._dynamo.config.patch(disable=True):
        expected = exported(*replay)
        actual = session.run(dict(zip(("values", "cos", "sin"), replay, strict=True)))
    expected = expected[0] if isinstance(expected, (tuple, list)) else expected
    actual = actual[0] if isinstance(actual, (tuple, list)) else actual
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[..., table_shape[-1]:], replay[0][..., table_shape[-1]:], rtol=0, atol=0)


def test_static_nested_boundary_runs_real_ptq_export_and_runtime(tmp_path):
    class StateStep(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4, bias=False)

        def forward(self, values, states):
            return self.linear(values) + states[0], states[0] + 1

    module = StateStep().half().eval()
    inputs = torch.linspace(-1, 1, 8).view(1, 2, 4).half()
    state = torch.full_like(inputs, 0.25)
    original = {name: value.detach().clone() for name, value in module.state_dict().items()}
    result = export_qwen4_exp_hmonnx(
        module, tmp_path, export_wrapper=module, sample_inputs=(inputs, (state,)),
        export_input_names=("values", "state"), export_output_names=("logits", "present_state"),
        ple_quant_type="fp16", quant_config={"quant_type": "w16a16h1_sefp", "preserve_model_dtype": True},
        validate_export_graph_parity=True,
    )
    info = json.loads(Path(result.io_manifest_path).read_text())
    assert [value["name"] for value in info["inputs"]] == ["values", "state"]
    assert info["outputs"] == ["logits", "present_state"]
    assert len(info["hmonnx_runtime_parity"]) == 2
    assert all(value["max_abs_error"] == 0 for value in info["hmonnx_runtime_parity"])
    for name, value in module.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_native_checkpoint_loader_reloads_real_tiny_parameters(tmp_path):
    from test_qwen4_exp_workflow_quant_adapter import _tiny_stateful_model

    from xhmodel_merak.xh_llm.models.qwen4_exp.checkpoint import load_qwen4_exp_model_for_export

    native = _tiny_stateful_model().half().eval()
    native.save_pretrained(tmp_path)
    loaded = load_qwen4_exp_model_for_export(tmp_path, device="cpu", dtype=torch.float16)
    assert loaded.state_dict().keys() == native.state_dict().keys()
    for name, value in loaded.state_dict().items():
        torch.testing.assert_close(value, native.state_dict()[name], rtol=0, atol=0)
