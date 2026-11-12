"""Unified routing/publication contracts; heavy weight export is the only build seam.

The transaction uses real checkpoint inspection, tokenizer serialization, ONNX
files, Qwen4 metadata, embedding conversion, Manifest and relocation validation.
It runs in the explicit Qwen4 checkpoint scope and does not claim full-checkpoint
export or model quality.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml
from safetensors.torch import save_file

from xhmodel_merak.errors import ModelError
from xhmodel_merak.model_build import describe_model
from xhmodel_merak.model_build.resolver import resolve_build_plan
from xhmodel_merak.model_build.workflow import materialize_workflow_config
from xhmodel_merak.model_inference import inference_model
from xhmodel_merak.xh_llm.models.qwen4_exp.build_entry import build_qwen4_exp_model as build_model
from xhmodel_merak.xh_llm.mrope import MROPE_CURRENT_POSITION_NAMES, MROPE_POOL_POSITION_NAMES


MODEL_ID = "qwen3.8-flash-next"
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def qwen4_script_checkpoint_scope():
    # Match the explicit Qwen4 build entry. Generic callers remain unpatched.
    from xhmodel_merak.xh_llm.models.qwen4_exp.checkpoint_patch import qwen4_exp_checkpoint_scope

    with qwen4_exp_checkpoint_scope():
        yield


def _config():
    return {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 2560,
            "num_hidden_layers": 48,
            "hc_count": 4,
            "hc_lowrank": 4,
            "vocab_size": 4,
            "layer_types": ["linear_attention"] * 3 + ["qwen_sparse_attention"],
            "num_attention_heads": 24,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "rope_parameters": {"rope_type": "default", "partial_rotary_factor": 0.25},
            "indexer_head_dim": 128,
            "indexer_n_heads": 4,
            "indexer_kv_heads": 1,
            "indexer_compress_ratio": 4,
            "indexer_budget": 2048,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 48,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "ple_conv_kernel_size": 4,
            "ngram_size": 3,
            "ple_layer_ids": [2],
            "eos_token_id": 1,
        },
    }


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source"
    path.mkdir()
    config = _config()
    config["text_config"]["layer_types"] *= 12
    (path / "config.json").write_text(json.dumps(config))
    return path


def test_registry_is_lightweight_and_exposes_only_supported_scope():
    script = (
        "import sys\nfrom xhmodel_merak.model_build import list_models, describe_model\n"
        f"assert any(m['model_id'] == {MODEL_ID!r} for m in list_models())\n"
        f"d=describe_model({MODEL_ID!r}, full=True)\n"
        "assert d['components'] == ['visual', 'prefill', 'decode']\n"
        "assert not {'torch','transformers','xhquant'} & sys.modules.keys()\n"
    )
    subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True, capture_output=True, text=True)
    description = describe_model(MODEL_ID, full=True)
    assert description["variants"] == ["full", "mtp"]
    assert description["modalities"] == ["text", "image", "video"]
    assert description["golden_components"] == ["visual", "prefill", "decode", "mtp"]
    assert description["quant_methods"] == ["rtn", "autoround"]


def test_resolver_routes_lengths_precision_and_existing_checkpoint(source, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "quantization_config.json").write_text(json.dumps({"bits": 8, "quant_method": "autoround"}))
    plan = resolve_build_plan(
        model_id=MODEL_ID,
        model_dir=str(source),
        output_dir=str(tmp_path / "out"),
        quant_checkpoint=str(checkpoint),
        context_length=4096,
        prefill_chunk_length=128,
    )
    assert plan.components == ("visual", "prefill", "decode")
    assert plan.precision.weight == "wmix"
    assert not any(r.startswith("auto-round") for r in plan.worker_requirements)
    assert plan.worker_capabilities == ()
    path = tmp_path / "effective.yaml"
    materialize_workflow_config(plan, path)
    effective = yaml.safe_load(path.read_text())
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    model = WorkflowConfig.from_file(str(path)).with_overrides(plan.config_overrides).export["model"]
    assert model.get("export_mode", "static") == "static"
    assert model["context_max_length"] == 4096
    assert model["prefill_chunk_length"] == 128
    assert model["max_pe_length"] == 262144
    assert effective["export"]["model"]["quant_scheme"]["ops"]["GPTQPackedMoeBlock"]["quant_type"] == "w4a8h0_ssfp"


@pytest.mark.parametrize("count", [1, 2, 4])
def test_mtp_variant_routes_assistant_and_preserves_visual_components(source, tmp_path, count):
    plan = resolve_build_plan(
        model_id=MODEL_ID, model_dir=str(source), output_dir=str(tmp_path / "mtp"),
        speculative={"mode": "mtp", "num_draft_tokens": count},
    )
    assert plan.variant == "mtp"
    assert plan.components == ("visual", "prefill", "decode", "mtp")
    path = tmp_path / "mtp.yaml"
    materialize_workflow_config(plan, path)
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    model = WorkflowConfig.from_file(str(path)).with_overrides(plan.config_overrides).export["model"]
    assert model["spec_decode_mode"] == "mtp"
    assert model["return_hidden_states"] is True
    assert model["visual_config"]["enable"] is True
    assert model["num_draft_tokens"] == count
    from xhmodel_merak.xh_llm.models.qwen4_exp.configuration_qwen4_exp import XHQwen4ExpModelConfig

    runtime_config = XHQwen4ExpModelConfig(**model)
    assert runtime_config.num_draft_tokens == count
    assert runtime_config.verify_length == count + 1
    assert plan.context_length == 262144
    assert model["context_max_length"] == 262144
    assert model["prefill_chunk_length"] == 256


@pytest.mark.parametrize(
    "options",
    [
        {"speculative": {"mode": "dflash"}},
        {"lora": {"path": "/not-supported"}},
        {"precision": {"weight": "w4", "activation": "a8"}},
        {"quant_method": "gptq"},
        {"components": ["visual"]},
        {"context_length": 128, "prefill_chunk_length": 256},
    ],
)
def test_unsupported_build_options_fail_before_execution(source, tmp_path, options):
    with pytest.raises(ModelError):
        resolve_build_plan(model_id=MODEL_ID, model_dir=str(source), output_dir=str(tmp_path / "out"), **options)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("low_memory", [None, False, True])
def test_streamed_export_defaults_to_low_memory_and_allows_explicit_standard(source, tmp_path, mtp, low_memory):
    options = {"speculative": {"mode": "mtp", "num_draft_tokens": 1}} if mtp else {}
    plan = resolve_build_plan(
        model_id=MODEL_ID, model_dir=str(source), output_dir=str(tmp_path / "out"),
        low_memory=low_memory, **options,
    )
    assert plan.low_memory is (low_memory is not False)
    assert plan.environment["HUGE_MODEL_EXPORT_ENABLED"] == ("0" if low_memory is False else "1")
    assert plan.variant == ("mtp" if mtp else "full")
    assert not (tmp_path / "out").exists()


def test_rejects_other_qwen_identity(source, tmp_path):
    config = json.loads((source / "config.json").read_text())
    config["model_type"] = "qwen3_next"
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ModelError):
        resolve_build_plan(model_id=MODEL_ID, model_dir=str(source), output_dir=str(tmp_path / "out"))


def _tokenizer(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "hello": 2, "test": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[EOS]"
    )
    tokenizer.chat_template = "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    tokenizer.save_pretrained(path)


def _checkpoint(source, path):
    shutil.copytree(source, path)
    config = json.loads((path / "config.json").read_text())
    config["quantization_config"] = {
        "quant_method": "autoround",
        "bits": 8,
        "group_size": 64,
        "dynamic": {"model.layers.*.mlp.experts.*": {"bits": 4}},
    }
    (path / "config.json").write_text(json.dumps(config))
    _tokenizer(path)
    tensors = {
        "model.language_model.embed_tokens.weight": torch.ones(4, 2560, dtype=torch.float16),
        "model.language_model.hyper_connection_mixer.hc_norm.weight": torch.ones(10240, dtype=torch.float16),
        "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight": torch.ones(
            4, 10240, dtype=torch.float16,
        ),
        "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight": torch.ones(
            10240, 4, dtype=torch.float16,
        ),
    }
    for layer in range(48):
        for suffix in ["self_attn.qweight", "mlp.down_proj.qweight"]:
            tensors[f"model.language_model.layers.{layer}.{suffix}"] = torch.ones(2, 2, dtype=torch.int32)
        tensors[f"model.language_model.layers.{layer}.input_layernorm.weight"] = torch.ones(2560, dtype=torch.float16)
    save_file(tensors, str(path / "model.safetensors"))


def _write_graph(path):
    import onnx
    from onnx import TensorProto, helper

    path.parent.mkdir(exist_ok=True)
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "publication_fixture",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)


def _visual_metadata(root):
    gears = []
    for capacity in (96, 196, 384, 704, 1536):
        relative = f"visual_m{capacity}/qwen38_visual.onnx"
        _write_graph(root / relative)
        gears.append({"image_token_capacity": capacity, "patch_token_capacity": capacity * 4, "hmonnx": relative})
    manifest = {
        "schema_version": 1, "visual_input_mode": "patches", "token_unit": "post_spatial_merge",
        "routing_policy": "smallest_fit", "overflow_policy": "reject", "gears": gears,
    }
    (root / "visual_gears.json").write_text(json.dumps(manifest))
    return {
        **manifest, "image_token_gears": [96, 196, 384, 704, 1536], "image_token_capacity": 1536,
        "spatial_merge_size": 2, "gear_manifest": "visual_gears.json", "hmonnx": gears[-1]["hmonnx"],
    }


def _add_publication_cache_inputs(path, contract, context_length, input_length):
    """Give the publication-only fixture the serving cache wiring it declares."""
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(path)
    graph = model.graph
    cache_inputs = []
    for name in ("past_seq_length", "current_input_length"):
        if name not in {value.name for value in graph.input}:
            cache_inputs.append(helper.make_tensor_value_info(name, TensorProto.INT32, [1]))
    cache_inputs.extend(helper.make_tensor_value_info(name, TensorProto.INT32, [1, input_length])
                        for name in MROPE_CURRENT_POSITION_NAMES)
    cache_inputs.extend(helper.make_tensor_value_info(name, TensorProto.INT32, [1, (input_length + 3) // 4])
                        for name in MROPE_POOL_POSITION_NAMES)
    ordinal = 0
    for layer in contract.layers:
        tensors = [tensor for tensor in layer.tensors if tensor.state_kind in {"kv_key", "kv_value"}]
        if not tensors:
            continue
        for tensor in tensors:
            kind = "key" if tensor.state_kind == "kv_key" else "value"
            name = f"past_{kind}_cache_{ordinal}"
            shape = [{"batch": 1, "cache_length": context_length}.get(dim, dim) for dim in tensor.shape]
            cache_inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT16, shape))
            current = f"current_{kind}_{ordinal}"
            current_shape = [*shape[:2], 1, shape[3]]
            graph.node.append(helper.make_node("Constant", [], [current], value=helper.make_tensor(
                current, TensorProto.FLOAT16, current_shape, [0] * (shape[0] * shape[1] * shape[3]),
            )))
            graph.node.append(helper.make_node(
                "KVcache", [current, "past_seq_length", "current_input_length", name],
                [f"present_{name}"], domain="ai.houmo.xh2a", axis=2, attention_max_length=-1,
            ))
        ordinal += 1
    # Host PLE inputs must remain the final graph inputs.
    existing_inputs = list(graph.input)
    del graph.input[:]
    graph.input.extend(cache_inputs + existing_inputs)
    graph_domains = {entry.domain for entry in model.opset_import}
    if "ai.houmo.xh2a" not in graph_domains:
        model.opset_import.append(helper.make_opsetid("ai.houmo.xh2a", 1))
    onnx.save(model, path)


def _prepare_publication(source, tmp_path, monkeypatch, *, variant, external_ple=False):
    from xhmodel_merak.xh_llm.models.qwen4_exp.serving_export import build_qwen4_exp_serving_metadata

    checkpoint = tmp_path / "checkpoint"
    _checkpoint(source, checkpoint)
    _tokenizer(source)
    for module in ["xhmodel_merak.model_build.executor", "xhmodel_merak.model_inference.api"]:
        monkeypatch.setattr(module + ".validate_worker_requirements", lambda *a, **kw: None)
        monkeypatch.setattr(module + ".validate_worker_capabilities", lambda *a, **kw: None)

    class ExportBoundary:
        def __init__(self, model_dir, config_path):
            self.config_path = config_path

        def quant(self, **kwargs):
            raise AssertionError("reusing a checkpoint must not run quantization")

        def export(self, *, quant_result, output_dir, device, config_overrides):
            from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

            root = Path(output_dir)
            config = WorkflowConfig.from_file(self.config_path).with_overrides(config_overrides)
            saved = root / "workflow.yaml"
            config.dump(str(saved))
            assert config.export["model"].get("export_mode", "static") == ("serving" if variant == "mtp" else "static")
            assert Path(quant_result.quanted_model_dir) == checkpoint
            for phase in ["prefill", "decode"]:
                _write_graph(root / phase / f"qwen38_{phase}.onnx")
            text_config = json.loads((checkpoint / "config.json").read_text())["text_config"]
            mtp_metadata = None
            if variant == "mtp":
                from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import Qwen4ExpCacheContract

                draft = root / "mtp/draft.onnx"
                _write_graph(draft)
                mtp_metadata = {
                    "onnx_path": str(draft),
                    "cache_contract": Qwen4ExpCacheContract.from_text_config(text_config).to_dict(),
                    "hidden_size": 2560,
                    "hc_count": 4,
                }
            build_qwen4_exp_serving_metadata(
                output_dir=root,
                model_name="qwen3_8_flash_next",
                hf_model_dir=checkpoint,
                text_config=SimpleNamespace(**text_config),
                model_config=config.export["model"],
                prefill_hmonnx=root / "prefill/qwen38_prefill.onnx",
                decode_hmonnx=root / "decode/qwen38_decode.onnx",
                embedding_state={"weight": torch.ones(4, 2560, dtype=torch.float16)},
                mtp_metadata=mtp_metadata,
            )
            metadata_file = root / "golden_meta_info.json"
            metadata = json.loads(metadata_file.read_text())
            metadata["visual_config"] = _visual_metadata(root)
            if external_ple:
                metadata.update(_external_ple_artifact(root))
            import hashlib

            from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import Qwen4ExpCacheContract

            contract = Qwen4ExpCacheContract.from_text_config(text_config)
            for phase in ("prefill", "decode"):
                path = root / metadata[f"{phase}_hmonnx"]
                length = metadata["model_config"]["prefill_chunk_length"] if phase == "prefill" else 1
                _add_publication_cache_inputs(path, contract, metadata["model_config"]["context_max_length"], length)
                metadata[f"{phase}_hmonnx_md5"] = hashlib.md5(path.read_bytes()).hexdigest()
            metadata_file.write_text(json.dumps(metadata))
            (root / "temporary_trace.txt").write_text("not a runtime asset")
            return SimpleNamespace(work_dir=str(root), config_file=str(saved), meta={})

    monkeypatch.setattr("xhmodel_merak.model_build.workflow.create_workflow", ExportBoundary)
    return {
        "model_id": MODEL_ID,
        "model_dir": str(source),
        "quant_checkpoint": str(checkpoint),
        "output_dir": str(tmp_path / "artifact"),
        "device": "cpu",
        "speculative": {"mode": "mtp", "num_draft_tokens": 1} if variant == "mtp" else None,
    }, {"prompt": "hello", "max_new_tokens": 1}


def prepare(case, root, monkeypatch):
    """Prepare checkpoint/artifact contract data at explicit export boundaries.

    This fixture is for publication contracts, not native model execution.
    """
    model_dir = root / "source"
    model_dir.mkdir()
    config = _config()
    config["text_config"]["layer_types"] *= 12
    (model_dir / "config.json").write_text(json.dumps(config))
    return _prepare_publication(model_dir, root, monkeypatch, variant=case.variant)


def check_build_publish_relocate_and_dispatch(source, tmp_path, monkeypatch, *, variant="full"):
    from xhmodel_merak.xh_llm.models.qwen4_exp import generation

    kwargs, inputs = _prepare_publication(source, tmp_path, monkeypatch, variant=variant)
    output = Path(kwargs["output_dir"])
    checkpoint = Path(kwargs["quant_checkpoint"])
    result = build_model(**kwargs)
    assert result.loadable is True
    assert json.loads((output / "environment.json").read_text())["variant"] == variant
    assert not (output / "temporary_trace.txt").exists()
    entries = json.loads((output / "export_meta_info.json").read_text())
    assert entries["prefill"]["type"] == "prefill"
    assert entries["decode"]["type"] == "decode"
    assert entries["visual_m96"]["type"] == "hmonnx"
    assert entries["quant_embedding_bin"]["path"] == "quant_embedding.bin"
    assert (output / "quant_embedding.bin").stat().st_size == 4 * 2560 * 2
    relocated = tmp_path / "relocated"
    output.rename(relocated)
    shutil.rmtree(source)
    shutil.rmtree(checkpoint)
    calls = []

    def generate(meta_file, **kwargs):
        assert Path(meta_file).parent == relocated
        assert os.environ["ENABLE_HMINFERENCE_V2"] == "1"
        assert os.environ["XHQUANT_PACKED_W4"] == "1"
        calls.append(kwargs)
        return "hello", 1, 0.1, 10.0, {"prompt_tokens": 1}

    monkeypatch.setattr(generation, "run_text_generation", generate)
    before = {key: os.environ.get(key) for key in ["ENABLE_HMINFERENCE_V2", "XHQUANT_PACKED_W4"]}
    answer = inference_model(
        artifact_dir=str(relocated),
        expected_model_id=MODEL_ID,
        inputs=inputs,
        device="cpu",
    )
    assert answer.output_text == "hello"
    assert calls[0]["prompt"] == "hello"
    assert calls[0]["max_new_tokens"] == 1
    assert calls[0]["device"] == "cpu"
    assert before == {key: os.environ.get(key) for key in before}
    with pytest.raises(ModelError):
        inference_model(artifact_dir=str(relocated), inputs={"prompt": "hello", "image": "x.png"}, device="cpu")
    if variant == "full":
        with pytest.raises(ModelError):
            inference_model(artifact_dir=str(relocated), inputs={"prompt": "hello"}, golden=["mtp"], device="cpu")
    (relocated / entries["decode"]["path"]).write_bytes(b"corrupt")
    with pytest.raises(ModelError):
        inference_model(artifact_dir=str(relocated), inputs={"prompt": "hello"}, device="cpu")
    assert len(calls) == 1


def test_build_publish_relocate_and_dispatch(source, tmp_path, monkeypatch):
    check_build_publish_relocate_and_dispatch(source, tmp_path, monkeypatch)


def test_build_cli_forwards_selected_mtp_mode(monkeypatch, capsys):
    from examples_merak.llm.qwen4_exp import build
    from xhmodel_merak.xh_llm.models.qwen4_exp import build_entry

    calls = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(to_dict=lambda: {"speculative": kwargs["speculative"]})

    monkeypatch.setattr(build_entry, "build_qwen4_exp_model", fake_build)
    build.main(["--speculative", '{"mode":"mtp","num_draft_tokens":1}',
                "--model-dir", "/model"])
    assert calls[0]["speculative"] == {"mode": "mtp", "num_draft_tokens": 1}
    assert json.loads(capsys.readouterr().out) == {"speculative": {"mode": "mtp", "num_draft_tokens": 1}}


def test_visual_export_and_sdk_inference_share_the_tested_torch_version():
    profile = describe_model(MODEL_ID, full=True)["dependencies"]["profiles"]["mtp"]
    assert "torch==2.8.0" in profile["build"]["pip"]
    assert "torch==2.11.0" not in profile["build"]["pip"]
    assert "torch==2.8.0" in profile["inference"]["pip"]
    assert "av==15.1.0" in profile["inference"]["pip"]
    assert profile["validation"]["clean_install"] is False


def test_checkpoint_projection_matches_native_hf_saved_aliases(tmp_path):
    from transformers import Qwen4ExpConfig

    from xhmodel_merak.model_integrations.qwen4_exp.build import project_checkpoint_config

    original = _config()
    original["text_config"]["num_hidden_layers"] = 4
    original["text_config"]["layer_types"][-1] = "full_attention"
    original["vision_config"] = {"model_type": "qwen4_exp", "hidden_size": 16}
    native = Qwen4ExpConfig.from_dict(json.loads(json.dumps(original)))
    native.use_cache = False
    native.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    expected = project_checkpoint_config(original, source=True)
    actual = project_checkpoint_config(saved, source=False)
    assert expected["text_config"]["layer_types"] == actual["text_config"]["layer_types"]
    assert expected["vision_config"]["model_type"] == actual["vision_config"]["model_type"]
    assert actual["vision_config"]["hidden_size"] == 16
    assert "use_cache" not in expected and "use_cache" not in actual
    assert saved["use_cache"] is False
    assert original["text_config"]["layer_types"][-1] == "full_attention"
    assert original["vision_config"]["model_type"] == "qwen4_exp"


@pytest.mark.parametrize(
    "damage", [None, "layer_order", "vision_size", "vision_type", "text_cache", "rope", "bad_cache"]
)
def test_checkpoint_identity_accepts_only_native_qwen4_serialization_aliases(source, tmp_path, damage):
    from xhmodel_merak.model_build.checkpoint import verify_quant_model_identity

    source_path = source / "config.json"
    original = json.loads(source_path.read_text())
    original["vision_config"] = {"model_type": "qwen4_exp", "hidden_size": 16}
    original["text_config"]["use_cache"] = True
    original["text_config"]["layer_types"] = [
        "full_attention" if layer == "qwen_sparse_attention" else layer
        for layer in original["text_config"]["layer_types"]
    ]
    source_path.write_text(json.dumps(original))
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    saved = json.loads(json.dumps(original))
    saved["text_config"]["layer_types"] = [
        "qwen_sparse_attention" if layer == "full_attention" else layer
        for layer in saved["text_config"]["layer_types"]
    ]
    saved["vision_config"]["model_type"] = "qwen4_exp_vision"
    saved["use_cache"] = False
    saved["quantization_config"] = {"bits": 8, "quant_method": "gptq", "auto_round_enabled": True}
    if damage == "layer_order":
        saved["text_config"]["layer_types"][0], saved["text_config"]["layer_types"][3] = (
            saved["text_config"]["layer_types"][3], saved["text_config"]["layer_types"][0]
        )
    elif damage == "vision_size":
        saved["vision_config"]["hidden_size"] = 32
    elif damage == "vision_type":
        saved["vision_config"]["model_type"] = "different_vision"
    elif damage == "text_cache":
        saved["text_config"]["use_cache"] = False
    elif damage == "rope":
        saved["text_config"]["rope_parameters"]["partial_rotary_factor"] = 0.5
    elif damage == "bad_cache":
        saved["use_cache"] = "false"
    config_path = checkpoint / "config.json"
    config_path.write_text(json.dumps(saved))
    plan = resolve_build_plan(
        model_id=MODEL_ID, model_dir=str(source), quant_checkpoint=str(checkpoint),
        output_dir=str(tmp_path / "artifact"),
    )
    if damage is None:
        verify_quant_model_identity(checkpoint, [(config_path, saved)], plan)
    else:
        with pytest.raises(ModelError, match="model config does not match"):
            verify_quant_model_identity(checkpoint, [(config_path, saved)], plan)
    assert json.loads(source_path.read_text()) == original
    assert json.loads(config_path.read_text()) == saved


@pytest.fixture
def native_text_checkpoint(tmp_path):
    """Save the actual HF layout; no synthetic final norm or hand-authored weights."""
    from transformers import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

    config = Qwen4ExpTextConfig(
        max_position_embeddings=16,
        vocab_size=16, hidden_size=16, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        linear_key_head_dim=4, linear_value_head_dim=4,
        linear_num_key_heads=2, linear_num_value_heads=2,
        moe_intermediate_size=8, shared_expert_intermediate_size=8,
        num_experts=2, num_experts_per_tok=1, hc_count=2, hc_lowrank=4,
        layer_types=["linear_attention", "full_attention"], ple_layer_ids=[],
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=8,
        indexer_budget=4, indexer_compress_ratio=2,
        rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.5},
    )
    path = tmp_path / "native"
    Qwen4ExpForCausalLM(config).save_pretrained(path)
    return path


def _inspect_native_checkpoint(path):
    from xhmodel_merak.model_build.checkpoint import _inspect_quant_weight_files
    from xhmodel_merak.model_catalog.registry import get_recipe

    return _inspect_quant_weight_files(path, SimpleNamespace(
        model_id=MODEL_ID, model_dir=str(path), build_adapter=get_recipe(MODEL_ID).build_adapter,
    ))


def test_native_text_saved_checkpoint_requires_hyper_connection_terminal(native_text_checkpoint):
    from safetensors.torch import load_file

    weights = load_file(native_text_checkpoint / "model.safetensors")
    assert "model.norm.weight" not in weights
    assert "model.hyper_connection_mixer.hc_norm.weight" in weights
    assert _inspect_native_checkpoint(native_text_checkpoint)["weights"]


@pytest.mark.parametrize("tensor", [
    "hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight",
])
@pytest.mark.parametrize("damage", ["missing", "truncated", "misplaced"])
def test_native_checkpoint_rejects_incomplete_terminal(native_text_checkpoint, tensor, damage):
    from safetensors.torch import load_file

    weight_file = native_text_checkpoint / "model.safetensors"
    weights = load_file(weight_file)
    name = "model.hyper_connection_mixer." + tensor
    value = weights.pop(name)
    if damage == "truncated":
        weights[name] = value[:-1].contiguous()
    elif damage == "misplaced":
        weights["model.layers.0.model.hyper_connection_mixer." + tensor] = value
    save_file(weights, str(weight_file))
    with pytest.raises(ModelError, match="verifiably complete") as error:
        _inspect_native_checkpoint(native_text_checkpoint)
    assert any(item["tensor"].endswith(tensor) for item in error.value.details["terminal_mismatches"])


@pytest.mark.parametrize("value", [None, 0, -1, True, "2", "missing"])
def test_native_checkpoint_rejects_invalid_terminal_dimensions(native_text_checkpoint, value):
    config_file = native_text_checkpoint / "config.json"
    config = json.loads(config_file.read_text())
    if value == "missing":
        config.pop("hc_count")
    else:
        config["hc_count"] = value
    config_file.write_text(json.dumps(config))
    with pytest.raises(ModelError, match="verifiably complete") as error:
        _inspect_native_checkpoint(native_text_checkpoint)
    assert any(
        "hc_count" in item.get("invalid_config_fields", [])
        for item in error.value.details["terminal_mismatches"]
    )


def test_standard_checkpoint_still_requires_final_norm(native_text_checkpoint):
    from xhmodel_merak.model_build.checkpoint import _inspect_quant_weight_files
    from xhmodel_merak.model_catalog.registry import get_recipe

    plan = SimpleNamespace(model_id="qwen3.5-9b", model_dir=str(native_text_checkpoint))
    with pytest.raises(ModelError, match="verifiably complete") as error:
        _inspect_quant_weight_files(native_text_checkpoint, plan)
    assert error.value.details["has_final_norm"] is False
    assert "terminal_mismatches" not in error.value.details
    assert not hasattr(get_recipe(plan.model_id), "checkpoint_terminal_tensors")


@pytest.mark.parametrize("terminal", [
    (),
    (("norm.weight", ()),),
    (("norm.weight", ("hidden_size+1",)),),
    (("norm.weight", ("hidden_size",)), ("norm.weight", ("hidden_size",))),
])
def test_recipe_requires_explicit_valid_terminal_contract(terminal):
    from dataclasses import replace

    from xhmodel_merak.model_catalog.registry import get_recipe

    with pytest.raises(ValueError, match="terminal tensors"):
        replace(get_recipe(MODEL_ID), checkpoint_terminal_tensors=terminal)



def test_native_checkpoint_rejects_truncated_container(native_text_checkpoint):
    weight_file = native_text_checkpoint / "model.safetensors"
    with weight_file.open("r+b") as handle:
        handle.truncate(weight_file.stat().st_size - 1)
    with pytest.raises(ModelError, match="weight container is invalid"):
        _inspect_native_checkpoint(native_text_checkpoint)



def test_native_vlm_saved_layout_reaches_checkpoint_validation_with_real_plan(
    native_text_checkpoint, tmp_path, monkeypatch,
):
    from dataclasses import replace

    from transformers import Qwen4ExpConfig, Qwen4ExpVisionConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForConditionalGeneration

    from xhmodel_merak.model_build.checkpoint import _inspect_quant_weight_files
    from xhmodel_merak.model_catalog.registry import get_catalog, get_recipe

    text_config = json.loads((native_text_checkpoint / "config.json").read_text())
    config = Qwen4ExpConfig(
        text_config=text_config,
        vision_config=Qwen4ExpVisionConfig(
            depth=1, hidden_size=16, num_heads=2, intermediate_size=32,
            out_hidden_size=16, patch_size=2, temporal_patch_size=2,
            spatial_merge_size=2, num_position_embeddings=16,
        ).to_dict(),
    )
    path = tmp_path / "native-vlm"
    Qwen4ExpForConditionalGeneration(config).save_pretrained(path)
    # Only scale the product-size constraints; architecture, terminal contract,
    # resolver and checkpoint inspection remain the actual registered behavior.
    recipe = get_recipe(MODEL_ID)
    tiny_constraints = tuple(
        (field, {"text_config.hidden_size": 16, "text_config.num_hidden_layers": 2}.get(field, value))
        for field, value in recipe.source_constraints
    )
    monkeypatch.setitem(get_catalog()._recipes, MODEL_ID, replace(recipe, source_constraints=tiny_constraints))
    plan = resolve_build_plan(
        model_id=MODEL_ID, model_dir=str(path), output_dir=str(tmp_path / "planned"),
    )
    assert plan.model_dir == str(path)
    assert _inspect_quant_weight_files(path, plan)["weights"]


def _external_ple_artifact(root):
    import copy

    import onnx
    from onnx import TensorProto, helper

    metadata = {"qwen4_exp_ple": {}}
    for phase in ("prefill", "decode"):
        directory = root / phase
        directory.mkdir(exist_ok=True)
        (directory / "packed.raw").write_bytes(bytes(7 * 2 * 32))
        (directory / "exponent.raw").write_bytes(bytes(7 * 2 * 2))
        spec = {
            "layer_idx": 1, "input_name": "ple_embeddings_1", "embedding_dim": 65,
            "output_dtype": "float16", "ngram_size": 3, "heads_per_ngram": 1,
            "eos_token_id": 0, "multipliers": [1, 3, 5], "vocab_sizes": [3, 4], "offsets": [0, 3],
            "packed": {"path": "packed.raw", "shape": [7, 2, 32], "dtype": "int8"},
            "exponent": {"path": "exponent.raw", "shape": [7, 2, 1], "dtype": "int16"},
        }
        manifest = {"version": 1, "layers": [spec]}
        model = helper.make_model(helper.make_graph(
            [helper.make_node("Identity", ["ple_embeddings_1"], ["output"])], "external_ple",
            [helper.make_tensor_value_info("input_ids", TensorProto.INT64, [1, 4]),
             helper.make_tensor_value_info("ple_embeddings_1", TensorProto.FLOAT16, [1, 4, 2, 65])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [1, 4, 2, 65])],
        ), opset_imports=[helper.make_opsetid("", 17)])
        model.metadata_props.add(key="qwen4_exp_ple", value=json.dumps(manifest))
        onnx.save(model, directory / "model.onnx")
        metadata[phase + "_hmonnx"] = f"{phase}/model.onnx"
        rebased = copy.deepcopy(manifest)
        for kind in ("packed", "exponent"):
            rebased["layers"][0][kind]["path"] = f"{phase}/{spec[kind]['path']}"
        metadata["qwen4_exp_ple"][phase] = rebased
    return metadata


def test_external_ple_validates_real_graph_and_inventory_after_relocation(tmp_path):
    from xhmodel_merak.model_integrations.qwen4_exp.artifact import validate_external_ple_contract

    root = tmp_path / "artifact"
    root.mkdir()
    metadata = _external_ple_artifact(root)
    moved = tmp_path / "moved"
    root.rename(moved)
    entries = validate_external_ple_contract(moved, metadata)
    assert entries == {
        f"ple_{phase}_1_{kind}": {"path": f"{phase}/{kind}.raw", "type": "embedding"}
        for phase in ("prefill", "decode") for kind in ("packed", "exponent")
    }


@pytest.mark.parametrize("damage", [
    "missing", "size", "escape", "symlink", "shape", "dtype", "root_mismatch", "hash_offsets",
    "custom_op", "undeclared_input", "missing_root", "missing_graph", "duplicate_graph_metadata",
    "retained_table", "missing_both",
    "batch_shape", "non_trailing",
])
def test_external_ple_rejects_broken_files_or_graph_contract(tmp_path, damage):
    import onnx

    from xhmodel_merak.model_integrations.qwen4_exp.artifact import validate_external_ple_contract

    metadata = _external_ple_artifact(tmp_path)
    graph_path = tmp_path / "prefill/model.onnx"
    model = onnx.load(graph_path)
    manifest = json.loads(model.metadata_props[0].value)
    spec = manifest["layers"][0]
    packed = tmp_path / "prefill/packed.raw"
    if damage == "missing":
        packed.unlink()
    elif damage == "size":
        packed.write_bytes(b"truncated")
    elif damage == "escape":
        spec["packed"]["path"] = "../prefill/packed.raw"
    elif damage == "symlink":
        packed.rename(packed.with_suffix(".original"))
        packed.symlink_to("packed.original")
    elif damage == "shape":
        model.graph.input[-1].type.tensor_type.shape.dim[-1].dim_value = 66
    elif damage == "dtype":
        model.graph.input[-1].type.tensor_type.elem_type = onnx.TensorProto.FLOAT
    elif damage == "root_mismatch":
        metadata["qwen4_exp_ple"]["prefill"]["layers"][0]["multipliers"][0] = 7
    elif damage == "hash_offsets":
        spec["offsets"] = [0, 2]
    elif damage == "custom_op":
        model.graph.node[0].op_type = "Qwen4PLESEFP4Gather"
    elif damage == "undeclared_input":
        model.graph.input.add().CopyFrom(onnx.helper.make_tensor_value_info(
            "ple_embeddings_2", onnx.TensorProto.FLOAT16, [1, 4, 2, 65],
        ))
    elif damage == "retained_table":
        model.graph.initializer.add(name="model.ple.ngram_embedding_weight", data_type=onnx.TensorProto.FLOAT16,
                                    dims=[7, 65], raw_data=bytes(7 * 65 * 2))
    elif damage == "batch_shape":
        model.graph.input[-1].type.tensor_type.shape.dim[0].dim_value = 2
    elif damage == "non_trailing":
        first = model.graph.input.pop(0)
        model.graph.input.append(first)
    elif damage in {"missing_root", "missing_both"}:
        metadata.pop("qwen4_exp_ple")
    if damage in {"missing_graph", "missing_both"}:
        del model.metadata_props[:]
    else:
        model.metadata_props[0].value = json.dumps(manifest)
        if damage == "duplicate_graph_metadata":
            model.metadata_props.add(key="qwen4_exp_ple", value=json.dumps(manifest))
    onnx.save(model, graph_path)
    with pytest.raises((ModelError, ValueError)):
        validate_external_ple_contract(tmp_path, metadata)


def test_public_external_ple_resources_are_published_and_required_in_manifest(source, tmp_path, monkeypatch):
    from xhmodel_merak.model_artifact.validation import read_manifest
    from xhmodel_merak.model_catalog.registry import get_recipe
    from xhmodel_merak.model_integrations.qwen4_exp.artifact import validate_artifact

    kwargs, _ = _prepare_publication(source, tmp_path, monkeypatch, variant="full", external_ple=True)
    result = build_model(**kwargs)
    relocated = tmp_path / "relocated_external_ple"
    Path(result.output_dir).rename(relocated)
    shutil.rmtree(source)
    shutil.rmtree(tmp_path / "checkpoint")
    root, manifest = read_manifest(str(relocated))
    entries = json.loads((root / "export_meta_info.json").read_text())
    for phase in ("prefill", "decode"):
        for kind in ("packed", "exponent"):
            entry = entries[f"ple_{phase}_1_{kind}"]
            assert entry == {"path": f"{phase}/{kind}.raw", "type": "embedding"}
            assert any(item["path"] == entry["path"] for item in manifest["artifact"]["files"])
    meta_file = root / "golden_meta_info.json"
    validate_artifact(root, manifest, meta_file, get_recipe(MODEL_ID))
    manifest["artifact"]["files"] = [
        item for item in manifest["artifact"]["files"] if item["path"] != "prefill/packed.raw"
    ]
    with pytest.raises(ModelError, match="Manifest"):
        validate_artifact(root, manifest, meta_file, get_recipe(MODEL_ID))



def test_external_ple_preserves_native_table_padding(tmp_path):
    import onnx

    from xhmodel_merak.model_integrations.qwen4_exp.artifact import validate_external_ple_contract

    metadata = _external_ple_artifact(tmp_path)
    for phase in ("prefill", "decode"):
        path = tmp_path / phase / "model.onnx"
        model = onnx.load(path)
        graph_manifest = json.loads(model.metadata_props[0].value)
        for kind, nbytes in (("packed", 2 * 32), ("exponent", 2 * 2)):
            graph_manifest["layers"][0][kind]["shape"][0] += 90
            metadata["qwen4_exp_ple"][phase]["layers"][0][kind]["shape"][0] += 90
            (tmp_path / phase / f"{kind}.raw").write_bytes(bytes(97 * nbytes))
        model.metadata_props[0].value = json.dumps(graph_manifest)
        onnx.save(model, path)
    assert len(validate_external_ple_contract(tmp_path, metadata)) == 4


def _mtp_v3_contract_artifact(root, *, explicit_cache=False, cache_op="KVcache", verify_length=2):
    """Small real ONNX files exercise metadata/graph checks, not model numerics."""
    import hashlib

    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import Qwen4ExpCacheContract

    text = dict(_config()["text_config"], hidden_size=16, hc_count=2, num_hidden_layers=2,
                layer_types=["linear_attention", "full_attention"], ple_layer_ids=[],
                mtp={"num_hidden_layers": 1, "layer_types": ["full_attention"]})
    target = Qwen4ExpCacheContract.from_text_config(text)
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

    assistant = Qwen4ExpCacheContract.from_text_config(Qwen4ExpTextConfig(
        **dict(text, num_hidden_layers=1, layer_types=["full_attention"]),
    ))
    context, chunk = 8, 4
    verify_role = f"decode_m{verify_length}"
    for role, width, contract in (("mtp_prefill", chunk, assistant), ("mtp_decode", 1, assistant),
                                  (verify_role, verify_length, target)):
        directory = root / role
        directory.mkdir()
        inputs = [helper.make_tensor_value_info("inputs_embeds" if role == verify_role else "next_token_embedding",
                                               TensorProto.FLOAT16, [1, width, 16])]
        if role == verify_role:
            inputs.append(helper.make_tensor_value_info("ple_input_ids", TensorProto.INT32, [1, width]))
        else:
            inputs.append(helper.make_tensor_value_info("target_hidden", TensorProto.FLOAT16, [1, width, 32]))
        inputs.extend(helper.make_tensor_value_info(name, TensorProto.INT32, [1])
                      for name in ("past_seq_length", "current_input_length"))
        inputs.extend(helper.make_tensor_value_info(name, TensorProto.INT32, [1, width])
                      for name in MROPE_CURRENT_POSITION_NAMES)
        inputs.extend(helper.make_tensor_value_info(name, TensorProto.INT32, [1, (width + 3) // 4])
                      for name in MROPE_POOL_POSITION_NAMES)
        selected_width = min(int(text["indexer_budget"]) // 4, context // 4) * 4 + 3
        for name, dtype, shape in (
            ("qsa_pool_offset", TensorProto.INT32, [1]),
            ("qsa_pool_new_count", TensorProto.INT32, [1]),
            ("qsa_pool_write_start", TensorProto.INT32, [1]),
            ("qsa_query_valid", TensorProto.FLOAT16, [1, width]),
            ("qsa_attention_mask", TensorProto.FLOAT16, [1, width, 1, selected_width]),
        ):
            inputs.append(helper.make_tensor_value_info(name, dtype, shape))
        cache_names = {}
        attention_layers = [layer.layer_idx for layer in contract.layers
                            if layer.layer_type in {"full_attention", "qwen_sparse_attention"}]
        for tensor in contract.tensors:
            if tensor.state_kind in {"kv_key", "kv_value"}:
                if not explicit_cache:
                    continue
                kind = "key" if tensor.state_kind == "kv_key" else "value"
                cache_names[tensor.name] = f"past_{kind}_cache_{attention_layers.index(tensor.layer_idx)}"
            inputs.append(helper.make_tensor_value_info(
                cache_names.get(tensor.name, tensor.name),
                TensorProto.INT32 if tensor.dtype == "integer" else TensorProto.FLOAT16,
                [{"batch": 1, "cache_length": context, "qsa_pool_length": context // 4}.get(dim, dim)
                 for dim in tensor.shape],
            ))
        hidden = "target_hidden_states" if role == verify_role else "mtp_hidden_states"
        nodes = [helper.make_node("Constant", [], [name], value=numpy_helper.from_array(
            np.zeros((1, width, channels), dtype=np.float16))) for name, channels in (("logits", 4), (hidden, 32))]
        outputs = [helper.make_tensor_value_info("logits", TensorProto.FLOAT16, [1, width, 4]),
                   helper.make_tensor_value_info(hidden, TensorProto.FLOAT16, [1, width, 32])]
        for tensor in contract.tensors:
            if tensor.state_kind in {"qsa_indexer_tail", "qsa_key_tail", "qsa_value_tail"}:
                nodes.append(helper.make_node("Identity", [tensor.name], [f"present_{tensor.name}"]))
                outputs.append(helper.make_tensor_value_info(
                    f"present_{tensor.name}", TensorProto.FLOAT16,
                    [{"batch": 1}.get(dim, dim) for dim in tensor.shape],
                ))
        if role == verify_role:
            for tensor in contract.tensors:
                if tensor.state_kind == "delta_recurrent":
                    nodes.append(helper.make_node("Identity", [tensor.name], [f"present_{tensor.name}"]))
                    outputs.append(helper.make_tensor_value_info(
                        f"present_{tensor.name}", TensorProto.FLOAT16,
                        [{"batch": 1}.get(dim, dim) for dim in tensor.shape],
                    ))
        if explicit_cache:
            for tensor in contract.tensors:
                if tensor.name not in cache_names:
                    continue
                cache_name = cache_names[tensor.name]
                current_name = f"current_{cache_name}"
                shape = [{"batch": 1, "cache_length": width}.get(dim, dim) for dim in tensor.shape]
                nodes.append(helper.make_node("Constant", [], [current_name], value=numpy_helper.from_array(
                    np.zeros(shape, dtype=np.float16),
                )))
                nodes.append(helper.make_node(
                    cache_op, [current_name, "past_seq_length", "current_input_length", cache_name],
                    [f"present_{cache_name}"], domain="ai.houmo.xh2a", axis=2, attention_max_length=-1,
                ))
        else:
            nodes.extend(helper.make_node("PageKVCache", [], [f"page_{index}"], domain="ai.houmo.xh2a")
                         for index, layer in enumerate(contract.layers)
                         if layer.layer_type in {"full_attention", "qwen_sparse_attention"})
        model = helper.make_model(helper.make_graph(nodes, role, inputs, outputs),
                                  opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("ai.houmo.xh2a", 1)])
        onnx.save(model, directory / "model.onnx")
    for phase in ("prefill", "decode"):
        _write_graph(root / phase / "model.onnx")
    metadata = {
        "prefill_hmonnx": "prefill/model.onnx", "decode_hmonnx": "decode/model.onnx",
        "model_config": {"context_max_length": context, "prefill_chunk_length": chunk},
        f"{verify_role}_hmonnx": f"{verify_role}/model.onnx",
        f"{verify_role}_hmonnx_md5": hashlib.md5((root / verify_role / "model.onnx").read_bytes()).hexdigest(),
        "spec_decode": {
            "mode": "mtp", "runtime_contract_version": 3, "num_draft_tokens": 1, "verify_length": verify_length,
            "target_verify_graph": verify_role, "target_verify_padding": verify_length,
            "target": {"verify_hmonnx": f"{verify_role}/model.onnx", "verify_chunk_length": verify_length},
            "draft_prefill_onnx": "mtp_prefill/model.onnx", "draft_decode_onnx": "mtp_decode/model.onnx",
            "draft_onnx": "mtp_decode/model.onnx",
            "draft": {"prefill_hmonnx": "mtp_prefill/model.onnx", "decode_hmonnx": "mtp_decode/model.onnx",
                      "prefill_chunk_length": chunk, "decode_chunk_length": 1,
                      "cache_contract": assistant.to_dict(), "hidden_size": 16, "hc_count": 2},
        },
    }
    if explicit_cache:
        metadata["qwen4_exp_cache_abi"] = "llm_cache_inplace_v1"
        metadata["spec_decode"]["draft"]["cache_abi"] = "llm_cache_inplace_v1"
    return metadata, text, target


@pytest.mark.parametrize("verify_length,cache_op,damage", [
    (2, "KVcache", None), (2, "LLMCache", None),
    (4, "KVcache", None), (4, "LLMCache", None),
    *((4, "KVcache", damage) for damage in ("missing_input", "missing_writer", "axis", "page", "draft_abi")),
])
def test_mtp_v3_explicit_kv_cache_contract(tmp_path, damage, cache_op, verify_length):
    import hashlib

    import onnx

    from xhmodel_merak.model_integrations.qwen4_exp.artifact import validate_mtp_v3_contract

    metadata, text, target = _mtp_v3_contract_artifact(
        tmp_path, explicit_cache=True, cache_op=cache_op, verify_length=verify_length,
    )
    path = tmp_path / f"decode_m{verify_length}/model.onnx"
    graph = onnx.load(path)
    if damage == "missing_input":
        kept = [item for item in graph.graph.input if item.name != "past_key_cache_0"]
        del graph.graph.input[:]
        graph.graph.input.extend(kept)
    elif damage == "missing_writer":
        writer = next(node for node in graph.graph.node if node.op_type == cache_op)
        graph.graph.node.remove(writer)
    elif damage == "axis":
        writer = next(node for node in graph.graph.node if node.op_type == cache_op)
        next(attr for attr in writer.attribute if attr.name == "axis").i = 1
    elif damage == "page":
        graph.graph.node.add().CopyFrom(onnx.helper.make_node("PageKVCache", [], ["page"]))
    elif damage == "draft_abi":
        metadata["spec_decode"]["draft"]["cache_abi"] = "page_kv_v1"
    onnx.save(graph, path)
    metadata[f"decode_m{verify_length}_hmonnx_md5"] = hashlib.md5(path.read_bytes()).hexdigest()
    if damage is None:
        validate_mtp_v3_contract(tmp_path, metadata, text, target)
    else:
        with pytest.raises(ValueError):
            validate_mtp_v3_contract(tmp_path, metadata, text, target)


@pytest.mark.parametrize("verify_length", [2, 4])
def test_mtp_v3_requires_independent_graphs_and_registers_verify_role(tmp_path, verify_length):
    from xhmodel_merak.model_integrations.qwen4_exp.artifact import (
        mtp_v3_graph_entries,
        validate_mtp_v3_contract,
    )

    metadata, text, target = _mtp_v3_contract_artifact(tmp_path, verify_length=verify_length)
    validate_mtp_v3_contract(tmp_path, metadata, text, target)
    assert mtp_v3_graph_entries(metadata) == {
        "target_verify": {"path": f"decode_m{verify_length}/model.onnx", "type": "decode"},
    }


@pytest.mark.parametrize("damage", ["alias", "verify_alias", "chunk", "hc", "cache_contract", "cache_input",
                                    "sequence", "hidden_output", "page_cache", "digest", "missing_graph", "ple",
                                    "extra_input", "chunk_gdr", "recurrent_output", "recurrent_count",
                                    "partial_recurrent"])
def test_mtp_v3_rejects_incomplete_actual_graph_contract(tmp_path, damage):
    import hashlib

    import onnx

    from xhmodel_merak.model_integrations.qwen4_exp.artifact import validate_mtp_v3_contract
    from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import Qwen4ExpCacheContract

    verify_length = 4
    metadata, text, target = _mtp_v3_contract_artifact(tmp_path, verify_length=verify_length)
    spec = metadata["spec_decode"]
    path = tmp_path / f"decode_m{verify_length}/model.onnx"
    graph = onnx.load(path)
    if damage == "alias":
        spec["draft"]["prefill_hmonnx"] = spec["draft"]["decode_hmonnx"]
    elif damage == "verify_alias":
        spec["target"]["verify_hmonnx"] = metadata["decode_hmonnx"]
    elif damage == "chunk":
        spec["draft"]["prefill_chunk_length"] = 1
    elif damage == "hc":
        spec["draft"]["hc_count"] = 1
    elif damage == "cache_contract":
        spec["draft"]["cache_contract"]["layers"][0]["tensors"] = []
    elif damage == "cache_input":
        del graph.graph.input[-1]
    elif damage == "extra_input":
        graph.graph.input.add().CopyFrom(onnx.helper.make_tensor_value_info(
            "unowned_state", onnx.TensorProto.FLOAT16, [1],
        ))
    elif damage == "sequence":
        graph.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 1
    elif damage == "hidden_output":
        del graph.graph.output[1]
    elif damage == "recurrent_output":
        del graph.graph.output[-1]
    elif damage == "recurrent_count":
        metadata["model_config"]["use_fused_gdr"] = True
    elif damage == "partial_recurrent":
        graph.graph.node.add().CopyFrom(onnx.helper.make_node(
            "GDRRecurrentScan", [], ["incomplete_recurrent_state"], domain="ai.houmo.xh2a",
        ))
    elif damage == "chunk_gdr":
        graph.graph.node.add().CopyFrom(onnx.helper.make_node(
            "GDRChunkScan", [], ["invalid_chunk_state"], domain="ai.houmo.xh2a",
        ))
    elif damage == "page_cache":
        del graph.graph.node[-1]
    elif damage == "digest":
        metadata[f"decode_m{verify_length}_hmonnx_md5"] = "0" * 32
    elif damage == "missing_graph":
        (tmp_path / "mtp_decode/model.onnx").unlink()
    elif damage == "ple":
        text["ple_layer_ids"] = [1]
        target = Qwen4ExpCacheContract.from_text_config(text)
    if damage in {
        "cache_input", "sequence", "hidden_output", "page_cache", "extra_input", "chunk_gdr", "recurrent_output",
        "partial_recurrent",
    }:
        onnx.save(graph, path)
        metadata[f"decode_m{verify_length}_hmonnx_md5"] = hashlib.md5(path.read_bytes()).hexdigest()
    with pytest.raises((ModelError, ValueError)):
        validate_mtp_v3_contract(tmp_path, metadata, text, target)



@pytest.mark.parametrize("count,damage", [
    (1, None), (2, None), (4, None),
    *((4, damage) for damage in ("candidate", "width", "owner", "bits", "bool_version")),
])
def test_mtp_v4_candidate_and_fixed_head_contract(tmp_path, count, damage):
    from xhmodel_merak.model_integrations.qwen4_exp.artifact import validate_mtp_v3_contract

    metadata, text, target = _mtp_v3_contract_artifact(tmp_path, explicit_cache=True, verify_length=count + 1)
    metadata["model_config"]["quant_scheme"] = {"quant_type": "w8a8h1_sefp"}
    spec = metadata["spec_decode"]
    spec.update(runtime_contract_version=4, num_draft_tokens=count, draft_head_weight_bits=8,
                shared_head={"version": 1, "owner": "target", "logical_bits": 8})
    if damage == "candidate":
        spec["num_draft_tokens"] = True
    elif damage == "width":
        spec["num_draft_tokens"] = spec["verify_length"]
    elif damage == "owner":
        spec["shared_head"]["owner"] = "draft"
    elif damage == "bits":
        spec["shared_head"]["logical_bits"] = 4
    elif damage == "bool_version":
        spec["shared_head"]["version"] = True
    if damage:
        with pytest.raises(ValueError):
            validate_mtp_v3_contract(tmp_path, metadata, text, target)
    else:
        validate_mtp_v3_contract(tmp_path, metadata, text, target)


@pytest.mark.parametrize("damage", ["role", "width", "padding", "capacity", "bool_width"])
def test_mtp_v3_rejects_mismatched_verify4_dimensions(tmp_path, damage):
    from xhmodel_merak.model_integrations.qwen4_exp.artifact import validate_mtp_v3_contract

    metadata, text, target = _mtp_v3_contract_artifact(tmp_path, verify_length=4)
    spec = metadata["spec_decode"]
    if damage == "role":
        spec["target_verify_graph"] = "decode_m2"
    elif damage == "width":
        spec["target"]["verify_chunk_length"] = 2
    elif damage == "padding":
        spec["target_verify_padding"] = 2
    elif damage == "capacity":
        metadata["model_config"]["context_max_length"] = 3
    else:
        spec["verify_length"] = True
    with pytest.raises(ValueError):
        validate_mtp_v3_contract(tmp_path, metadata, text, target)


@pytest.mark.parametrize("include_verify", [False, True])
def test_mtp_v3_external_ple_requires_verify_resources(tmp_path, include_verify):
    import copy

    from xhmodel_merak.model_integrations.qwen4_exp.artifact import validate_external_ple_contract

    metadata = _external_ple_artifact(tmp_path)
    shutil.copytree(tmp_path / "prefill", tmp_path / "decode_m2")
    metadata.update(decode_m2_hmonnx="decode_m2/model.onnx", spec_decode={"runtime_contract_version": 3})
    if include_verify:
        verify = copy.deepcopy(metadata["qwen4_exp_ple"]["prefill"])
        for kind in ("packed", "exponent"):
            verify["layers"][0][kind]["path"] = f"decode_m2/{kind}.raw"
        metadata["qwen4_exp_ple"]["decode_m2"] = verify
        entries = validate_external_ple_contract(tmp_path, metadata)
        assert entries["ple_decode_m2_1_packed"] == {"path": "decode_m2/packed.raw", "type": "embedding"}
        assert len(entries) == 6
    else:
        with pytest.raises(ValueError, match="every target phase"):
            validate_external_ple_contract(tmp_path, metadata)


def test_raw_checkpoint_uses_graph_quant_scheme_without_gptq_producer(source, tmp_path):
    plan = resolve_build_plan(
        model_id=MODEL_ID, model_dir=str(source), output_dir=str(tmp_path / "out"), quant_method=None,
    )
    assert plan.quant_method is None and plan.precision.weight == "w16"
    assert plan.config_overrides["quant"] is None
    assert plan.context_length == 262144 and plan.prefill_chunk_length == 256
    effective_path = tmp_path / "effective.yaml"
    materialize_workflow_config(plan, effective_path)
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    model = WorkflowConfig.from_file(str(effective_path)).with_overrides(plan.config_overrides).export["model"]
    assert model["quant_scheme"]["quant_type"] == "w8a8h1_sefp"
    assert model["visual_config"]["quant_scheme"]["quant_type"] == "w8a8h1_sefp"
    assert model["ple_quant_type"] == "sefp4"
    assert not any("gptqmodel" in requirement.lower() for requirement in plan.worker_requirements)
