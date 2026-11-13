"""PLE is an explicit host input; graphs contain no lookup or unpack operation."""
import importlib
import json
from pathlib import Path

import numpy as np
import onnx
import pytest
import torch
from onnx import TensorProto, helper, numpy_helper

from xhmodel_merak.xh_llm.models.qwen4_exp.ple_export import (
    export_ple_sefp4_hmonnx,
    extract_qwen4_exp_ple_specs,
    read_qwen4_exp_ple_specs,
)
from xhmodel_merak.xh_llm.models.qwen4_exp.ple_sefp4 import lookup_ple_sefp4, quantize_ple_sefp4_rows


# Run every fixture against the actual deployed SDK schemas, including when collected alone.
importlib.import_module("xhquant.export.onnx.onnx_opset._impl.xh2a_opset1")

def _spec():
    return {"layer_idx": 1, "ngram_size": 2, "heads_per_ngram": 1, "eos_token_id": 0,
            "multipliers": [3, 5], "vocab_sizes": [7], "offsets": [0]}


def _fixture(root, *, external=True, axis=0, ple=True, domain=""):
    root.mkdir()
    weight = np.random.default_rng(4).normal(size=(7, 65)).astype(np.float16)
    name = "model.layers.1.ple.ngram_embedding_weight" if ple else "ordinary_embedding"
    keep = np.arange(65, dtype=np.float16)
    initializers = [numpy_helper.from_array(weight, name), numpy_helper.from_array(keep, "keep"),
                    numpy_helper.from_array(np.array([2], dtype=np.int64), "axes")]
    gather = helper.make_node("Gather", [name, "hashed_ids"], ["lookup"], axis=axis,
                              domain=domain, name="ple_lookup")
    graph = helper.make_graph(
        [helper.make_node("Unsqueeze", ["ids", "axes"], ["hashed_ids"], name="dead_hash"), gather,
         helper.make_node("Add", ["lookup", "keep"], ["embedding"], name="add_weight"),
         helper.make_node("Identity", ["history"], ["present_history"], name="history_update")], "ple",
        [helper.make_tensor_value_info("ids", TensorProto.INT64, [1, 2]),
         helper.make_tensor_value_info("history", TensorProto.INT64, [1, 1])],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT16, [1, 2, 1, 65]),
         helper.make_tensor_value_info("present_history", TensorProto.INT64, [1, 1])],
        initializers,
        value_info=[helper.make_tensor_value_info("lookup", TensorProto.FLOAT16, [1, 2, 1, 65]),
                    helper.make_tensor_value_info("hashed_ids", TensorProto.INT64, [1, 2, 1])],
    )
    model = helper.make_model(graph, opset_imports=[
        helper.make_opsetid("", 17), helper.make_opsetid("ai.houmo.xh2a", 1),
    ])
    path = root / "model.onnx"
    onnx.save_model(model, path, save_as_external_data=external, all_tensors_to_one_file=True,
                    location="weights.data", size_threshold=0)
    return path, weight, keep


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("domain", ["", "ai.houmo.xh2a"])
def test_streamed_host_input_export_preserves_state_and_matches_sdk(tmp_path, external, domain):
    source, weight, keep = _fixture(tmp_path / "source", external=external, domain=domain)
    original = source.read_bytes()
    output = tmp_path / "destination" / "model.onnx"
    report = export_ple_sefp4_hmonnx(source, output, ple_specs=[_spec()], chunk_rows=3)
    assert source.read_bytes() == original
    assert report["status"] == "externalized"
    assert not report["tables"][0]["reused_packed_bytes"]
    model = onnx.load(output)
    assert [v.name for v in model.graph.input] == ["ids", "history", "ple_embeddings_1"]
    assert [v.name for v in model.graph.output] == ["embedding", "present_history"]
    assert [n.op_type for n in model.graph.node] == ["Add", "Identity"]
    assert [t.name for t in model.graph.initializer] == ["keep"]
    np.testing.assert_array_equal(numpy_helper.to_array(model.graph.initializer[0]), keep)
    assert report["external_bytes"] == keep.nbytes
    spec, = read_qwen4_exp_ple_specs(output)
    packed, exp = quantize_ple_sefp4_rows(torch.from_numpy(weight))
    np.testing.assert_array_equal(np.fromfile(output.parent / spec["packed"]["path"], dtype=np.int8)
                                  .reshape(spec["packed"]["shape"]), packed.numpy())
    np.testing.assert_array_equal(np.fromfile(output.parent / spec["exponent"]["path"], dtype="<i2")
                                  .reshape(spec["exponent"]["shape"]), exp.numpy())
    before = output.read_bytes()
    assert export_ple_sefp4_hmonnx(output, output)["status"] == "already_external_ple"
    assert output.read_bytes() == before
    onnx.checker.check_model(str(output))

    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    session = HMONNXInferenceV2(str(output), HMONNXInferenceConfig(exec_devices=["cpu"]))
    ids = torch.tensor([[0, 6]], dtype=torch.int64)
    ple = lookup_ple_sefp4(packed, exp, ids.unsqueeze(-1), embedding_dim=65)
    actual = session.run({"ids": ids, "history": torch.tensor([[3]]), "ple_embeddings_1": ple})
    torch.testing.assert_close(actual[0], ple + torch.from_numpy(keep), rtol=0, atol=0)
    torch.testing.assert_close(actual[1], torch.tensor([[3]]), rtol=0, atol=0)


def test_no_ple_in_place_is_unchanged(tmp_path):
    source, _, _ = _fixture(tmp_path / "source", ple=False)
    before = {p.name: p.read_bytes() for p in source.parent.iterdir()}
    assert export_ple_sefp4_hmonnx(source, source)["status"] == "no_ple"
    assert before == {p.name: p.read_bytes() for p in source.parent.iterdir()}


@pytest.mark.parametrize("corruption", ["escape", "length", "axis", "missing_spec"])
def test_rejects_unsafe_incomplete_or_incompatible_input(tmp_path, corruption):
    source, _, _ = _fixture(tmp_path / "source", axis=1 if corruption == "axis" else 0)
    model = onnx.load(source, load_external_data=False)
    if corruption in ("escape", "length"):
        for entry in model.graph.initializer[0].external_data:
            if entry.key == ("location" if corruption == "escape" else "length"):
                entry.value = "../outside.data" if corruption == "escape" else "9999999999"
        source.write_bytes(model.SerializeToString())
    output = tmp_path / "output.onnx"
    with pytest.raises(ValueError):
        export_ple_sefp4_hmonnx(source, output, ple_specs=[] if corruption == "missing_spec" else [_spec()])
    assert not output.exists()
    assert not list(tmp_path.glob("output.ple*"))


def test_bfloat16_host_input_and_corrupt_sidecar_rejection(tmp_path):
    source, weight, _ = _fixture(tmp_path / "source", external=False)
    model = onnx.load(source)
    bf16 = torch.from_numpy(weight).bfloat16()
    tensor = model.graph.initializer[0]
    tensor.data_type = TensorProto.BFLOAT16
    tensor.raw_data = bf16.view(torch.int16).numpy().tobytes()
    for value in (model.graph.output[0], model.graph.value_info[0]):
        value.type.tensor_type.elem_type = TensorProto.BFLOAT16
    source.write_bytes(model.SerializeToString())
    output = tmp_path / "bf16.onnx"
    export_ple_sefp4_hmonnx(source, output, ple_specs=[_spec()], chunk_rows=2)
    loaded = onnx.load(output)
    spec, = read_qwen4_exp_ple_specs(output)
    assert spec["output_dtype"] == "bfloat16"
    assert loaded.graph.input[-1].type.tensor_type.elem_type == TensorProto.BFLOAT16
    onnx.checker.check_model(str(output))
    (output.parent / spec["packed"]["path"]).write_bytes(b"broken")
    with pytest.raises(ValueError, match="byte length"):
        export_ple_sefp4_hmonnx(output, output)


def test_native_ngram_module_exposes_real_hash_contract():
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextNGramEmbedding

    config = Qwen4ExpTextConfig(ngram_size=2, heads_per_ngram=1, ngram_vocab_size_base=5, eos_token_id=0,
                               make_ngram_vocab_size_divisible_by=1)
    module = Qwen4ExpTextNGramEmbedding(config, embedding_dim=64, layer_idx=1)
    spec, = extract_qwen4_exp_ple_specs(module)
    assert spec["layer_idx"] == 1
    assert spec["multipliers"] == module.layer_multipliers.tolist()
    assert spec["vocab_sizes"] == module.ngram_heads_vocab_sizes.tolist()
    assert spec["offsets"] == module.ngram_heads_offsets.tolist()


def test_fresh_export_golden_helper_prepares_host_inputs_from_original_samples(tmp_path):
    from xhmodel_merak.xh_llm.models.qwen4_exp.golden_runtime import run_qwen4_exp_hmonnx_golden
    from xhmodel_merak.xh_llm.models.qwen4_exp.host_ple_inputs import HostPLEInputs

    source, _, keep = _fixture(tmp_path / "source")
    model = onnx.load(source, load_external_data=False)
    names = {"ids": "ple_input_ids", "history": "ple_ngram_history_1"}
    for node in model.graph.node:
        for index, name in enumerate(node.input):
            node.input[index] = names.get(name, name)
    for value in model.graph.input:
        value.name = names.get(value.name, value.name)
    model.graph.input.append(helper.make_tensor_value_info("current_input_length", TensorProto.INT32, [1]))
    source.write_bytes(model.SerializeToString())
    output = tmp_path / "exported.onnx"
    export_ple_sefp4_hmonnx(source, output, ple_specs=[_spec()])
    ids, history, length = torch.tensor([[1, 6]]), torch.tensor([[3]]), torch.tensor([2], dtype=torch.int32)
    host = HostPLEInputs.from_graph(output)
    expected, = host.prepare(ids, {"ple_ngram_history_1": history}, 2, device="cpu")
    outputs = run_qwen4_exp_hmonnx_golden(output, (ids, history, length), tmp_path / "golden", device="cpu")
    torch.testing.assert_close(outputs["embedding"], expected + torch.from_numpy(keep), rtol=0, atol=0)
    torch.testing.assert_close(outputs["present_history"], history, rtol=0, atol=0)

    # A second step reuses immutable resources and appends an independent sample.
    run_qwen4_exp_hmonnx_golden(output, (ids, history, length, expected), tmp_path / "golden", device="cpu", step=1)
    golden = tmp_path / "golden"
    manifest = json.loads((golden / "ple/manifest.json").read_text())
    assert len(manifest["samples"]) == 2
    assert (golden / "ple_demo.py").is_file()
    for layer in manifest["layers"]:
        for kind in ("packed", "exponent"):
            assert (golden / "ple" / layer[kind]["path"]).is_file()
    import shutil
    import subprocess
    import sys

    portable = tmp_path / "portable_golden"
    shutil.move(str(golden), portable)
    shutil.rmtree(source.parent)
    output.unlink()
    for layer in host.manifest["layers"]:
        for kind in ("packed", "exponent"):
            (host.root / layer[kind]["path"]).unlink()
    replay = subprocess.run([sys.executable, str(portable / "ple_demo.py"), "--sample", "1"],
                            capture_output=True, text=True, check=True)
    assert json.loads(replay.stdout)["matched_expected"] is True


def test_pruning_retains_cache_writer_after_hash_consumer_disappears(tmp_path):
    source, _, _ = _fixture(tmp_path / "source")
    model = onnx.load(source, load_external_data=False)
    for name in ("past_seq_length", "current_input_length"):
        model.graph.input.append(helper.make_tensor_value_info(name, TensorProto.INT32, [1]))
    for name in ("token_states", "token_state_cache"):
        model.graph.input.append(helper.make_tensor_value_info(name, TensorProto.FLOAT16, [1, 2]))
    cache = helper.make_node(
        "LLMCache", ["token_states", "past_seq_length", "current_input_length", "token_state_cache"],
                             ["cached_token_states"], domain="ai.houmo.xh2a", name="token_history_writer",
                             axis=1, attention_max_length=2)
    model.graph.node[0].input[0] = "cached_ids"
    model.graph.node.insert(0, cache)
    model.graph.node.insert(1, helper.make_node("Cast", ["cached_token_states"], ["cached_ids"],
                                              to=TensorProto.INT64, name="cached_ids_cast"))
    source.write_bytes(model.SerializeToString())
    output = tmp_path / "external.onnx"
    export_ple_sefp4_hmonnx(source, output, ple_specs=[_spec()])
    graph = onnx.load(output, load_external_data=False).graph
    assert "token_history_writer" in {n.name for n in graph.node}
    assert "dead_hash" not in {n.name for n in graph.node}
    onnx.checker.check_model(str(output))


@pytest.mark.parametrize("fused", [False, True])
def test_sdk_graph_guard_retains_native_gdn_cache_argument_contract(fused):
    from transformers import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedDeltaNet

    from xhmodel_merak.xh_llm.models.qwen4_exp.modeling import Qwen4ExpExportableGatedDeltaNet
    from xhquant.frontend.utils import module_graph_forward_guard

    config = Qwen4ExpTextConfig(
        hidden_size=16, num_hidden_layers=1, layer_types=["linear_attention"], ple_layer_ids=[],
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=4,
        linear_value_head_dim=4, linear_conv_kernel_dim=2,
    )
    module = Qwen4ExpExportableGatedDeltaNet.from_native(
        Qwen4ExpTextGatedDeltaNet(config, layer_idx=0), batch_size=1, sequence_length=1,
    )
    if fused:
        module.enable_fused_gdr()
    with module_graph_forward_guard(torch.nn.Sequential(module)):
        # The flat shared graph ABI rejects these keywords before reaching the
        # family adapter. Its native-cache error proves the intended entry ran.
        with pytest.raises(ValueError, match="requires explicit cache state"):
            module(torch.zeros(1, 1, 16), cache_params=None, attention_mask=torch.ones(1, 1))


def test_modulelist_export_weight_name_retains_native_layer_identity(tmp_path):
    source, _, _ = _fixture(tmp_path / "source")
    model = onnx.load(source, load_external_data=False)
    name = ("self___native_text_model_layers_slice_None__2__None____modules__1___ple_ple_embedding"
            "__modules__ngram_embedding.weight")
    model.graph.node[1].input[0] = name
    model.graph.initializer[0].name = name
    source.write_bytes(model.SerializeToString())
    target = tmp_path / "host.onnx"
    export_ple_sefp4_hmonnx(source, target, ple_specs=[_spec()])
    assert onnx.load(target).graph.input[-1].name == "ple_embeddings_1"


@pytest.mark.parametrize("sequence_length", [1, 4], ids=["decode", "prefill"])
def test_native_text_export_externalizes_ple_after_real_sdk_tracing(tmp_path, sequence_length):
    from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

    from xhmodel_merak.xh_llm.models.qwen4_exp.static_export import (
        build_qwen4_exp_serving_sample_inputs,
        export_qwen4_exp_hmonnx,
    )

    config = Qwen4ExpTextConfig(
        max_position_embeddings=16,
        vocab_size=32, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        num_key_value_heads=1, head_dim=8, linear_key_head_dim=4, linear_value_head_dim=4,
        linear_num_key_heads=2, linear_num_value_heads=2, moe_intermediate_size=8,
        shared_expert_intermediate_size=8, num_experts=2, num_experts_per_tok=1,
        hc_count=2, hc_lowrank=4, layer_types=["linear_attention", "qwen_sparse_attention"],
        ple_layer_ids=[1], ple_embed_dim=64, ple_conv_kernel_size=2, ngram_size=2,
        heads_per_ngram=1, ngram_vocab_size_base=5, make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=1, indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8,
        indexer_budget=4, indexer_compress_ratio=4,
        rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.5}, eos_token_id=0,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(8)
        native = Qwen4ExpForCausalLM(config).half().eval()
    _, _, sample = build_qwen4_exp_serving_sample_inputs(
        native, input_sequence_length=sequence_length, context_length=8,
    )
    result = export_qwen4_exp_hmonnx(
        native, tmp_path, sample_inputs=sample, text_config=config, input_sequence_length=sequence_length,
        cache_length=8, quant_config={"quant_type": "w16a16h1_sefp", "preserve_model_dtype": True},
        quant_device="cpu", use_fused_gdr=True,
        return_hidden_states=True,
    )
    graph = onnx.load(result.onnx_path, load_external_data=False)
    onnx.checker.check_model(str(result.onnx_path))
    assert any(node.op_type == ("GDRChunkScan" if sequence_length > 1 else "GDRRecurrentScan")
               for node in graph.graph.node)
    rms_norms = [node for node in graph.graph.node if node.op_type == "RMSNorm"]
    assert rms_norms, "Native Qwen4 RMSNorm must use the SDK fused operator"
    assert all(next(attr.i for attr in node.attribute if attr.name == "axis") == -1 for node in rms_norms)
    public_outputs = {value.name for value in graph.graph.output}
    assert all(
        set(node.output) <= public_outputs
        for node in graph.graph.node if node.op_type in {"Identity", "HMIdentity"}
    )
    assert not any(node.op_type == "Qwen4PLESEFP4Gather" for node in graph.graph.node)
    assert not any(node.op_type == "BitShift" for node in graph.graph.node)
    assert not any(node.op_type == "Cast" and any(attr.name == "to" and attr.i == onnx.TensorProto.UINT32
                                                  for attr in node.attribute) for node in graph.graph.node)
    assert not any(node.op_type in {"Div", "Floor"} and node.domain in {"", "ai.onnx"}
                   for node in graph.graph.node)
    assert not any("ngram_embedding" in tensor.name for tensor in graph.graph.initializer)
    inputs = json.loads(Path(result.io_manifest_path).read_text())["inputs"]
    assert inputs[-1] == {"name": "ple_embeddings_0", "dtype": "float16", "shape": [1, sequence_length, 1, 64]}
    assert {"ple_input_ids", "ple_ngram_history_0", "ple_conv_state_0"} <= {item["name"] for item in inputs}
    assert {"qsa_pool_offset", "qsa_pool_new_count", "qsa_pool_write_start",
            "qsa_attention_mask", "qsa_query_valid"} <= {
        item["name"] for item in inputs
    }
    layer = read_qwen4_exp_ple_specs(result.onnx_path)[0]
    assert layer["packed"]["shape"] == [8, 1, 32]
    assert layer["exponent"]["shape"] == [8, 1, 1]
