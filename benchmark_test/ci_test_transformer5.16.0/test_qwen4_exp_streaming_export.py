"""Actual small Qwen4 streamed capture, assembly and SDK runtime contracts."""

import copy
import json
from pathlib import Path

import onnx
import pytest
import torch
from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

from xhmodel_merak.xh_llm.models.qwen4_exp.serving_graph import convert_qwen4_exp_kv_to_llm_cache
from xhmodel_merak.xh_llm.models.qwen4_exp.static_export import (
    build_qwen4_exp_serving_sample_inputs,
    export_qwen4_exp_hmonnx,
)
from xhmodel_merak.xh_llm.models.qwen4_exp.streaming_export import export_qwen4_exp_streamed_phase
from xhmodel_merak.xh_llm.models.qwen4_exp.streaming_loader import Qwen4ExpStreamingLoader


def _config(ple=False):
    return Qwen4ExpTextConfig(
        max_position_embeddings=32,
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        linear_key_head_dim=4,
        linear_value_head_dim=4,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=2,
        num_experts_per_tok=1,
        hc_count=2,
        hc_lowrank=4,
        layer_types=["linear_attention", "qwen_sparse_attention"],
        ple_layer_ids=[1] if ple else [],
        ple_embed_dim=64,
        ple_conv_kernel_size=2,
        ngram_size=2,
        heads_per_ngram=1,
        ngram_vocab_size_base=5,
        make_ngram_vocab_size_divisible_by=8,
        split_ngram_parts=1,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=4,
        indexer_compress_ratio=4,
        rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.5},
        eos_token_id=0,
    )


def test_static_model_publishes_streamed_phases_without_capture_files(tmp_path, monkeypatch):
    from xhmodel_merak.xh_llm.models.qwen4_exp.configuration_qwen4_exp import XHQwen4ExpModelConfig
    from xhmodel_merak.xh_llm.models.qwen4_exp.qwen4_exp_model import XHQwen4ExpModel

    monkeypatch.setenv("HUGE_MODEL_EXPORT_ENABLED", "1")
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "0")
    checkpoint = tmp_path / "checkpoint"
    Qwen4ExpForCausalLM(_config(ple=True)).half().eval().save_pretrained(checkpoint)
    quant = {"quant_type": "w16a16h1_sefp", "preserve_model_dtype": True}
    model = XHQwen4ExpModel(
        XHQwen4ExpModelConfig(
            hf_model=str(checkpoint),
            model_name="qwen4_tiny",
            context_max_length=8,
            prefill_chunk_length=3,
            quant_scheme=quant,
            quant_device="cpu",
            use_fused_gdr=True,
            visual_config=None,
        )
    )
    root = tmp_path / "package"
    meta = model.export_hmonnx(str(root))
    saved = json.loads((root / "golden_meta_info.json").read_text())
    assert saved["model_config"]["quant_scheme"] == {
        "quant_type": "w16a16h1_sefp",
        "preserve_model_dtype": True,
    }
    assert "spec_decode" not in saved
    assert set(saved["qwen4_exp_ple"]) == {"prefill", "decode"}
    for phase in ("prefill", "decode"):
        relative = saved[phase + "_hmonnx"]
        assert getattr(meta, phase + "_hmonnx") == relative
        expected_name = f"qwen4_exp_{phase}_llm_cache.onnx"
        assert relative == f"{phase}/{expected_name}"
        onnx.checker.check_model(str(root / relative))
        graph = onnx.load(str(root / relative), load_external_data=False)
        assert not any(node.op_type == "PlaceHolder" for node in graph.graph.node)
        assert {layer["input_name"] for layer in saved["qwen4_exp_ple"][phase]["layers"]} == {
            value.name for value in graph.graph.input if value.name.startswith("ple_embeddings_")
        }
        assert not any(value.name.startswith("present_past_") for value in graph.graph.output)
        assert not any(value.name.startswith("_streamed_input_") for value in graph.graph.output)
        files = [path for path in (root / phase).rglob("*") if path.is_file()]
        assert all(path.suffix != ".json" for path in files)
        assert [path.relative_to(root / phase).as_posix() for path in files if path.suffix == ".onnx"] == [
            Path(relative).name,
        ]
        assert not any(path.name == "qwen4_exp_text_external_data" and path.parent.name != phase for path in files)


@pytest.mark.parametrize(
    "sequence_length,mode,ple",
    [
        (1, None, False),
        (2, "recurrent", False),
        (3, None, True),
    ],
)
def test_real_streamed_graph_matches_standard_export(tmp_path, monkeypatch, sequence_length, mode, ple):
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "0")
    torch.manual_seed(41)
    config = _config(ple)
    native = Qwen4ExpForCausalLM(config).half().eval()
    checkpoint = tmp_path / "checkpoint"
    native.save_pretrained(checkpoint)
    loader = Qwen4ExpStreamingLoader(checkpoint)
    quant = {"quant_type": "w16a16h1_sefp", "preserve_model_dtype": True}
    streamed = export_qwen4_exp_streamed_phase(
        loader,
        tmp_path / "streamed",
        input_sequence_length=sequence_length,
        cache_length=8,
        quant_config=quant,
        quant_device="cpu",
        use_fused_gdr=True,
        linear_attention_mode=mode,
        return_hidden_states=True,
    )
    assert all(value.is_meta for value in loader.model.parameters())
    _, _, sample = build_qwen4_exp_serving_sample_inputs(
        native,
        input_sequence_length=sequence_length,
        context_length=8,
    )
    standard = export_qwen4_exp_hmonnx(
        native,
        tmp_path / "standard",
        sample_inputs=copy.deepcopy(sample),
        text_config=config,
        input_sequence_length=sequence_length,
        cache_length=8,
        quant_config=quant,
        quant_device="cpu",
        use_fused_gdr=True,
        linear_attention_mode=mode,
        return_hidden_states=True,
    )
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "1")
    for label, result in (("streamed", streamed), ("standard", standard)):
        path = Path(
            convert_qwen4_exp_kv_to_llm_cache(
                result.onnx_path,
                tmp_path / f"{label}_published" / "model.onnx",
                context_length=8,
                num_kv_heads=config.num_key_value_heads,
                head_dim=config.head_dim,
            )
        )
        graph = onnx.load(path, load_external_data=False)
        assert not any(node.op_type == "PlaceHolder" for node in graph.graph.node)
        cache_internal_names = {
            name
            for node in graph.graph.node
            if node.op_type == "SparseAttentionSelectKV"
            for name in node.input
            if name.startswith(("present_past_", "present_qsa_indexer_"))
        }
        assert cache_internal_names
        assert cache_internal_names <= {value.name for value in graph.graph.value_info}
        assert not cache_internal_names & {value.name for value in graph.graph.output}
        types = {value.name: value.type for value in (*graph.graph.input, *graph.graph.value_info)}
        for name in cache_internal_names:
            writer = next(node for node in graph.graph.node if name in node.output)
            assert writer.op_type == "KVcache"
            assert types[name] == types[writer.input[3]]
        onnx.checker.check_model(str(path))
        actual = _replay_graph(path, native, config, sample, sequence_length)
        if label == "streamed":
            expected = actual
        else:
            for actual_step, expected_step in zip(actual, expected, strict=True):
                for group in ("outputs", "state"):
                    assert set(actual_step[group]) == set(expected_step[group])
                    for name in actual_step[group]:
                        torch.testing.assert_close(
                            actual_step[group][name],
                            expected_step[group][name],
                            atol=0.004,
                            rtol=0.03,
                        )
    info = json.loads(Path(streamed.io_manifest_path).read_text())
    assert info["streamed_blocks"] == 2
    assert [item["name"] for item in info["inputs"]] == [
        value.name for value in onnx.load(streamed.onnx_path, load_external_data=False).graph.input
    ]
    if ple:
        assert info["ple_quantization"]["status"] == "externalized"
    assert len(info["block_load_audits"]) == 2
    assert all(
        item["loaded_parameter_bytes"] > 0 and item["remaining_parameter_bytes_after_release"] == 0
        for item in info["block_load_audits"]
    )


def _replay_graph(path, native, config, sample, sequence_length):
    from xhmodel_merak.exec_backends import HMONNXInferenceConfig
    from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import (
        Qwen4ExpCacheContract,
        qwen4_exp_cache_input_specs,
    )
    from xhmodel_merak.xh_llm.models.qwen4_exp.host_ple_inputs import HostPLEInputs
    from xhmodel_merak.xh_llm.models.qwen4_exp.host_ple_runtime import (
        Qwen4ExpHostPLEInferenceV2,
        enable_qwen4_exp_inplace_kv,
    )
    from xhquant.core.cache_tensor import CacheTensor

    graph = onnx.load(path, load_external_data=False).graph
    session = Qwen4ExpHostPLEInferenceV2(str(path), HMONNXInferenceConfig(exec_devices=["cpu"]))
    state_specs = qwen4_exp_cache_input_specs(Qwen4ExpCacheContract.from_text_config(config))
    paired = {name for name, spec in state_specs.items() if spec.state_kind in {"kv_key", "kv_value"}}
    enable_qwen4_exp_inplace_kv(session, paired)
    state = {}
    for (name, spec), value in zip(state_specs.items(), sample[10][: len(state_specs)], strict=True):
        state[name] = value.clone() if spec.dtype == "integer" else CacheTensor(value.clone())
    host = HostPLEInputs.from_graph(path)
    from xhmodel_merak.xh_llm.models.qwen4_exp.qsa_host_controls import QSA_HOST_CONTROL_NAMES, build_qsa_host_controls
    from xhmodel_merak.xh_llm.mrope import (
        MROPE_CURRENT_POSITION_NAMES,
        MROPE_POOL_POSITION_NAMES,
        prepare_mrope_position_inputs,
    )

    group_start = torch.zeros((3, 1, 1), dtype=torch.int32)
    snapshots = []
    for step in range(2):
        past = step * sequence_length
        ids = torch.arange(sequence_length).reshape(1, -1).to(torch.int32) + 3 + 5 * step
        positions = torch.arange(past, past + sequence_length).reshape(1, 1, -1).to(torch.int32)
        positions = torch.cat((positions + 1, positions * 2 + 1, positions * 3 + 1), dim=0)
        current_pos, pool_pos, group_start = prepare_mrope_position_inputs(
            group_start,
            positions,
            past,
            sequence_length,
            sequence_length,
        )
        controls = build_qsa_host_controls(
            past, sequence_length, sequence_length, 8, 1, device="cpu", dtype=torch.float16
        )
        feeds = {
            "inputs_embeds": native.model.embed_tokens(ids.long()).detach(),
            "past_seq_length": torch.tensor([past], dtype=torch.int32),
            "current_input_length": torch.tensor([sequence_length], dtype=torch.int32),
            "ple_input_ids": ids,
            **dict(zip(MROPE_CURRENT_POSITION_NAMES, current_pos.unbind(0), strict=True)),
            **dict(zip(MROPE_POOL_POSITION_NAMES, pool_pos.unbind(0), strict=True)),
            **dict(zip(QSA_HOST_CONTROL_NAMES, controls, strict=True)),
            **state,
        }
        if host is not None:
            values = host.prepare(ids, state, sequence_length, device="cpu")
            feeds.update(zip((spec["input_name"] for spec in host.manifest["layers"]), values, strict=True))
        with torch._dynamo.config.patch(disable=True):
            raw = session.run({value.name: feeds[value.name] for value in graph.input})
        outputs = dict(
            zip((value.name for value in graph.output), raw if isinstance(raw, (list, tuple)) else (raw,), strict=True)
        )
        for name in state:
            updated = outputs.get("present_" + name)
            if updated is not None:
                state[name].copy_(updated)
        for name, spec in state_specs.items():
            if spec.state_kind in {"kv_key", "kv_value", "delta_recurrent"} or (
                spec.state_kind == "qsa_indexer_key" and (past + sequence_length) // 4
            ):
                assert torch.count_nonzero(state[name]) > 0, name
        snapshots.append(
            {
                "outputs": {name: value.detach().clone() for name, value in outputs.items()},
                "state": {name: value.detach().clone() for name, value in state.items()},
            }
        )
    return snapshots
