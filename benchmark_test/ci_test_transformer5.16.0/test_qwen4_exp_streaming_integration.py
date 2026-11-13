"""Streamed serving publication and real target/draft SDK execution."""

import json
from pathlib import Path

import onnx
import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig

from xhmodel_merak.xh_llm.models.qwen4_exp.package_export import export_qwen4_exp_serving_package


def _resume_child_code(checkpoint, output, export_config, report_path):
    return (
        "import runpy,sys;"
        "sys.path.insert(0,str(__import__('pathlib').Path.cwd()));"
        f"runpy.run_path({str(Path(__file__).resolve())!r})['_check_resume_provenance_in_fresh_process']"
        f"({str(checkpoint)!r},{str(output)!r},{export_config!r},{str(report_path)!r})"
    )


def test_resume_child_bootstrap_needs_no_local_backports():
    import subprocess
    import sys

    probe = (
        "import builtins,runpy;original=builtins.__import__\n"
        "def without_backports(name,*args,**kwargs):\n"
        " if name in ('aenum','typing_extensions'): raise ModuleNotFoundError(name)\n"
        " return original(name,*args,**kwargs)\n"
        "builtins.__import__=without_backports\n"
        "def check(*args):\n"
        " assert args == ('checkpoint','output',{'setting':4},'report')\n"
        "runpy.run_path=lambda _: {'_check_resume_provenance_in_fresh_process':check}\n"
    )
    child = subprocess.run(
        [sys.executable, "-c", probe + _resume_child_code("checkpoint", "output", {"setting": 4}, "report")],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode == 0, child.stderr


def _check_resume_provenance_in_fresh_process(checkpoint, output, export_config, report_path):
    """Run against the same real package in a distinct interpreter, without PTQ."""
    import hashlib

    from onnx import numpy_helper
    from safetensors.torch import load_file

    from xhmodel_merak.xh_llm import base_model

    checkpoint, output = Path(checkpoint), Path(output)
    from xhmodel_merak.xh_llm.models.qwen4_exp import mtp_export

    def unexpected(*args, **kwargs):
        raise AssertionError("Completed resume must not repeat target/head/draft PTQ")

    base_model.ptq_quantize = unexpected
    mtp_export.export_qwen4_exp_mtp = unexpected
    manifest_path = output / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    resources = {name for record in manifest["mtp_graphs"].values() for name in record["resources"]}
    resources.update(record["hmonnx"] for record in manifest["graphs"].values())
    target_roles = {Path(record["hmonnx"]).relative_to(output).parts[0] for record in manifest["graphs"].values()}
    resources.update(
        str(path.relative_to(output))
        for path in output.rglob("*")
        if path.is_file() and path.relative_to(output).parts[0] in target_roles | {"ple"}
    )
    resources.update({"golden_meta_info.json", ".qwen4_fixed_head.pt"})

    def hashes():
        return {name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in resources}

    def resume():
        return export_qwen4_exp_serving_package(
            model_dir=checkpoint, output_dir=output, device="cpu", resume=True, model_config=export_config
        )

    before = hashes()
    assert resume()["status"] == "complete"
    assert hashes() == before
    results = {"normal": "zero PTQ; graph/resources/metadata/head bytes unchanged"}
    graph = output / "mtp_prefill/model.onnx"
    cases = [
        ("source_head", checkpoint / "model.safetensors"),
        ("source_assistant", checkpoint / "mtp.safetensors"),
        ("graph_same_shape", graph),
        ("missing_completion", manifest_path),
    ]
    sidecars = [
        output / name for name in manifest["mtp_graphs"]["prefill"]["resources"] if name != "mtp_prefill/model.onnx"
    ]
    if sidecars:
        cases.append(("sidecar", sidecars[0]))
    for case, path in cases:
        original = path.read_bytes()
        try:
            if case.startswith("source_"):
                tensors = load_file(path)
                name = "lm_head.weight" if case == "source_head" else next(iter(tensors))
                tensors[name].view(-1)[0] += 1
                save_file(tensors, path)
            elif case == "graph_same_shape":
                model = onnx.load(path)
                tensor = next(t for t in model.graph.initializer if "lm_head.qweight" in t.name)
                values = numpy_helper.to_array(tensor).copy()
                values.flat[0] = int(values.flat[0]) + 1 if int(values.flat[0]) < 127 else 126
                tensor.CopyFrom(numpy_helper.from_array(values, name=tensor.name))
                onnx.save(model, path)
                onnx.checker.check_model(str(path))
            elif case == "sidecar":
                contents = bytearray(original)
                contents[0] ^= 1
                path.write_bytes(contents)
            else:
                data = json.loads(original)
                del data["mtp_graphs"]["prefill"]
                path.write_text(json.dumps(data))
            try:
                resume()
            except ValueError as exc:
                results[case] = str(exc)
            else:
                raise AssertionError(f"Resume accepted changed {case}")
        finally:
            path.write_bytes(original)
        assert hashes() == before
    Path(report_path).write_text(json.dumps(results, indent=2) + "\n")


def test_streamed_package_exports_and_executes_target_and_draft_graphs(tmp_path, monkeypatch):
    from xhmodel_merak.exec_backends import HMONNXInferenceConfig
    from xhmodel_merak.xh_llm import base_model
    from xhmodel_merak.xh_llm.models.qwen4_exp.host_ple_runtime import (
        Qwen4ExpHostPLEInferenceV2,
        enable_qwen4_exp_inplace_kv,
    )
    from xhmodel_merak.xh_llm.models.qwen4_exp.mtp_model import Qwen4ExpMTPModel
    from xhmodel_merak.xh_llm.models.qwen4_exp.mtp_runtime import MTPGraphRuntime

    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "0")
    monkeypatch.delenv("HUGE_MODEL_EXPORT_ENABLED", raising=False)
    fixed_before_ptq, head_quantizations, head_ids, head_refs, patched_classes = [], [], set(), [], set()
    original_ptq = base_model.ptq_quantize

    def tracked_ptq(graph, *args, **kwargs):
        heads = [
            graph.get_submodule(node.target)
            for node in graph.graph.nodes
            if node.op == "call_module" and str(node.target).replace(".", "_").endswith("lm_head")
        ]
        head_refs.extend(heads)
        for head in heads:
            fixed_before_ptq.append(head.is_quant_fixed)
            head_ids.add(id(head))
            cls = type(head)
            if cls not in patched_classes:
                original_quantize = cls.weight_static_quant

                def counted(module, *values, _original=original_quantize, **options):
                    if id(module) in head_ids:
                        head_quantizations.append(id(module))
                    return _original(module, *values, **options)

                monkeypatch.setattr(cls, "weight_static_quant", counted)
                patched_classes.add(cls)
        return original_ptq(graph, *args, **kwargs)

    monkeypatch.setattr(base_model, "ptq_quantize", tracked_ptq)
    torch.manual_seed(103)
    config = Qwen4ExpTextConfig(
        max_position_embeddings=16,
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
        ple_layer_ids=[1],
        ple_embed_dim=64, ple_conv_kernel_size=2, ngram_size=2, heads_per_ngram=1,
        ngram_vocab_size_base=5, make_ngram_vocab_size_divisible_by=8, split_ngram_parts=1,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=8,
        indexer_budget=4,
        indexer_compress_ratio=4,
        rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.5},
        eos_token_id=0,
        mtp={"num_hidden_layers": 1, "layer_types": ["full_attention"]},
    )
    native = Qwen4ExpForCausalLM(config).half().eval()
    checkpoint = tmp_path / "source"
    native.save_pretrained(checkpoint)
    assistant = Qwen4ExpMTPModel(config).half().eval()
    save_file(
        {f"mtp.{name}": value.contiguous() for name, value in assistant.state_dict().items()},
        checkpoint / "mtp.safetensors",
    )
    output = tmp_path / "artifact"
    export_config = {
        "context_max_length": 8,
        "prefill_chunk_length": 3,
        "spec_decode_mode": "mtp",
        "quant_scheme": {"quant_type": "w8a8h1_sefp", "ops": {"MoeBlock": {"quant_type": "w4a8h0_ssfp"}}},
        "use_fused_gdr": True,
        "cache_abi": "llm_cache_inplace_v1",
    }
    from xhmodel_merak.xh_llm.models.qwen4_exp import ple_sefp4

    encoded_chunks = []
    encode_rows = ple_sefp4.quantize_ple_sefp4_rows

    def record_ple_encoding(rows):
        encoded_chunks.append(len(rows))
        return encode_rows(rows)

    monkeypatch.setattr(ple_sefp4, "quantize_ple_sefp4_rows", record_ple_encoding)
    report = export_qwen4_exp_serving_package(
        model_dir=checkpoint,
        output_dir=output,
        device="cpu",
        model_config=export_config,
    )
    assert report["status"] == "complete" and report["export_mode"] == "low_memory"
    assert len(encoded_chunks) == 1
    assert len(list((output / "ple").glob("*.packed.bin"))) == 1
    assert len(list((output / "ple").glob("*.exponent.bin"))) == 1
    assert not list(output.glob("*/ple.json"))
    assert not list(output.glob("decode_m*"))
    assert len(list(output.rglob("*.onnx"))) == 4
    assert not list(output.rglob("*_raw")) and not list(output.rglob("raw"))
    assert not list(output.rglob("placeholders"))
    assert not list(output.rglob("block_*.onnx"))
    meta = json.loads((output / "golden_meta_info.json").read_text())
    assert meta["spec_decode"]["verify_length"] == 5
    assert meta["spec_decode"]["target_verify_graph"] == "decode"
    assert meta["spec_decode"]["num_draft_tokens"] == 4
    assert fixed_before_ptq == [False, True, True, True]
    assert len(head_quantizations) == 1
    paths = [
        meta["prefill_hmonnx"],
        meta["decode_hmonnx"],
        meta["spec_decode"]["draft_prefill_onnx"],
        meta["spec_decode"]["draft_decode_onnx"],
    ]
    monkeypatch.setenv("TORCH_COMPILE_DISABLE", "1")
    head_payloads = []
    for relative in paths:
        path = output / relative
        onnx.checker.check_model(str(path))
        graph = onnx.load(path, load_external_data=False)
        if relative == meta["decode_hmonnx"]:
            inputs = {value.name: value for value in graph.graph.input}
            assert inputs["inputs_embeds"].type.tensor_type.shape.dim[1].dim_value == 5
            assert sum(node.op_type == "GDRRecurrentScan" for node in graph.graph.node) == 5
            assert not any(node.op_type == "GDRChunkScan" for node in graph.graph.node)
        assert not any(node.op_type == "PlaceHolder" for node in graph.graph.node)
        (head,) = [
            node
            for node in graph.graph.node
            if node.op_type == "Linear" and any("lm_head.qweight" in name for name in node.input)
        ]
        initializers = {value.name: value for value in graph.graph.initializer}
        from onnx import numpy_helper

        head_payloads.append(
            tuple(
                numpy_helper.to_array(initializers[name], base_dir=str(path.parent)).tobytes()
                for name in head.input[1:]
            )
        )
        for node in graph.graph.node:
            if node.op_type == "MoeBlock":
                attributes = {item.name: onnx.helper.get_attribute_value(item) for item in node.attribute}
                assert attributes["hmfp_weight_man_bit"] == 4
        assert not any(
            value.name.startswith(
                ("present_past_key_cache_", "present_past_value_cache_", "present_qsa_indexer_key_cache_")
            )
            for value in graph.graph.output
        )
        session = Qwen4ExpHostPLEInferenceV2(str(path), HMONNXInferenceConfig(exec_devices=["cpu"]))
        cache_names = {
            name
            for name in session.get_input_names()
            if name.startswith(("past_key_cache_", "past_value_cache_", "qsa_indexer_key_cache_"))
        }
        assert any(name.startswith("qsa_indexer_key_cache_") for name in cache_names)
        paired = {name for name in cache_names if name.startswith(("past_key_cache_", "past_value_cache_"))}
        enable_qwen4_exp_inplace_kv(session, paired)
        feeds = {}
        for name in session.get_input_names():
            info = session.get_input(name)
            shape = tuple(info.shape)
            value = torch.zeros(shape, dtype=info.dtype)
            if info.name == "current_input_length":
                value.fill_(1)
            if info.name in cache_names or info.name.startswith("delta_recurrent_state_"):
                from xhquant.core.cache_tensor import CacheTensor

                value = CacheTensor(value)
            feeds[info.name] = value
        from xhmodel_merak.xh_llm.models.qwen4_exp.host_ple_inputs import HostPLEInputs

        host = HostPLEInputs.from_graph(path)
        if host is not None:
            rows = host.prepare(feeds["ple_input_ids"], feeds, 1, device="cpu")
            feeds.update(zip((spec["input_name"] for spec in host.manifest["layers"]), rows, strict=True))
        raw = session.run(feeds)
        outputs = dict(
            zip(
                (value.name for value in graph.graph.output),
                raw if isinstance(raw, (tuple, list)) else (raw,),
                strict=True,
            )
        )
        assert outputs["logits"].shape[-1] == config.vocab_size
        assert all(torch.isfinite(value).all() for value in outputs.values())
    assert all(value == head_payloads[0] for value in head_payloads)
    import hashlib

    def graph_hashes():
        roles = {Path(relative).parts[0] for relative in paths} | {"ple"}
        return {
            str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in output.rglob("*")
            if path.is_file()
            and path.relative_to(output).parts[0] in roles
            and "raw" not in path.relative_to(output).parts
        }

    before_resume = graph_hashes()
    resumed = export_qwen4_exp_serving_package(
        model_dir=checkpoint,
        output_dir=output,
        device="cpu",
        resume=True,
        model_config=export_config,
    )
    assert resumed["status"] == "complete"
    assert fixed_before_ptq == [False, True, True, True]
    assert len(head_quantizations) == 1
    assert graph_hashes() == before_resume
    import subprocess
    import sys

    child_code = _resume_child_code(checkpoint, output, export_config, tmp_path / "resume_provenance.json")
    with (tmp_path / "resume_provenance.log").open("w") as child_log:
        child = subprocess.run(
            [sys.executable, "-c", child_code], stdout=child_log, stderr=subprocess.STDOUT, timeout=180
        )
    assert child.returncode == 0, (tmp_path / "resume_provenance.log").read_text()
    assert {"normal", "source_head", "source_assistant", "graph_same_shape", "missing_completion"} <= set(
        json.loads((tmp_path / "resume_provenance.json").read_text())
    )
    from xhmodel_merak.xh_llm.models.qwen4_exp.package_export import _load_fixed_head_checkpoint

    record = report["fixed_quantized_head"]
    head = _load_fixed_head_checkpoint(output, record, record["bindings"])
    assert head.is_quant_fixed
    assert head.qweight.detach().cpu().numpy().tobytes() == head_payloads[0][0]
    with pytest.raises(ValueError, match="same source"):
        _load_fixed_head_checkpoint(output, record, {**record["bindings"], "source_config_sha256": "bad"})
    private_head = output / record["path"]
    with private_head.open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        _load_fixed_head_checkpoint(output, record, record["bindings"])
    draft = meta["spec_decode"]["draft"]
    runtime = MTPGraphRuntime(
        str(output / draft["decode_hmonnx"]),
        cache_contract=draft["cache_contract"],
        context_length=8,
        device="cpu",
        cache_abi="llm_cache_inplace_v1",
        prefill_path=str(output / draft["prefill_hmonnx"]),
        prefill_chunk_length=3,
    )
    assert runtime.prefill_session is not None
    assert torch.load(output / "quant_embedding.pt", weights_only=True)["weight"].shape == (32, 16)
    from types import SimpleNamespace

    from xhmodel_merak.model_integrations.autoregressive.build import sanitize_artifact

    sanitize_artifact(output, SimpleNamespace())
    assert not private_head.exists()
    assert all((output / relative).is_file() for relative in paths)
