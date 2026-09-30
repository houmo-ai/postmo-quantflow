import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


HARNESS = Path("tools/gemma4_merak_hardware_validation.py")
TRACE_MODULE = Path("vllm_merak/gemma4_trace.py")


def _load_harness():
    spec = importlib.util.spec_from_file_location("gemma4_hardware_validation", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_hardware_validation_surfaces_exist() -> None:
    assert HARNESS.is_file()
    assert TRACE_MODULE.is_file()


def test_trace_is_disabled_without_an_explicit_sink(monkeypatch, tmp_path) -> None:
    from vllm_merak.gemma4_trace import emit_trace, trace_enabled

    sink = tmp_path / "unexpected.jsonl"
    monkeypatch.delenv("VLLM_MERAK_GEMMA4_TRACE_JSONL", raising=False)

    assert not trace_enabled("VLLM_MERAK_GEMMA4_TRACE_JSONL")
    assert not emit_trace(
        "VLLM_MERAK_GEMMA4_TRACE_JSONL",
        component="model",
        event="chunk",
        payload={"sink": str(sink)},
    )
    assert not sink.exists()


def test_trace_jsonl_has_a_stable_envelope(monkeypatch, tmp_path) -> None:
    from vllm_merak.gemma4_trace import emit_trace, flush_traces

    sink = tmp_path / "gemma.jsonl"
    monkeypatch.setenv("VLLM_MERAK_GEMMA4_TRACE_JSONL", str(sink))
    monkeypatch.setenv("VLLM_MERAK_TRACE_RUN_ID", "run-7")
    monkeypatch.setenv("RANK", "3")
    monkeypatch.setenv("LOCAL_RANK", "1")

    assert emit_trace(
        "VLLM_MERAK_GEMMA4_TRACE_JSONL",
        component="model",
        event="chunk",
        payload={"request_id": "req-7", "phase": "decode", "P": 3134, "Q": 1},
    )
    assert not sink.exists(), "enabled tracing must buffer the timed request path"
    flush_traces()

    row = json.loads(sink.read_text())
    assert row["schema"] == "vllm-merak.gemma4.trace/v1"
    assert row["component"] == "model"
    assert row["event"] == "chunk"
    assert row["request_id"] == "req-7"
    assert row["phase"] == "decode"
    assert row["P"] == 3134
    assert row["Q"] == 1
    assert isinstance(row["timestamp_ns"], int)
    assert isinstance(row["pid"], int)
    assert row["run_id"] == "run-7"
    assert row["rank"] == 3
    assert row["local_rank"] == 1


def test_trace_flush_retries_short_writes(monkeypatch, tmp_path) -> None:
    import os

    from vllm_merak import gemma4_trace

    sink = tmp_path / "short-write.jsonl"
    monkeypatch.setenv(gemma4_trace.GEMMA4_TRACE_ENV, str(sink))
    monkeypatch.setenv(gemma4_trace.TRACE_RUN_ID_ENV, "short-write-run")
    real_write = os.write
    write_sizes = []

    def short_write(fd, data):
        written = real_write(fd, data[: max(1, len(data) // 3)])
        write_sizes.append(written)
        return written

    monkeypatch.setattr(gemma4_trace.os, "write", short_write)
    assert gemma4_trace.emit_trace(
        gemma4_trace.GEMMA4_TRACE_ENV,
        component="model",
        event="short_write",
        payload={"request_id": "req"},
    )
    gemma4_trace.flush_traces()

    assert len(write_sizes) > 1
    assert json.loads(sink.read_text())["run_id"] == "short-write-run"


def test_trace_requires_run_identity_when_enabled(monkeypatch, tmp_path) -> None:
    from vllm_merak.gemma4_trace import GEMMA4_TRACE_ENV, emit_trace

    monkeypatch.setenv(GEMMA4_TRACE_ENV, str(tmp_path / "trace.jsonl"))
    monkeypatch.delenv("VLLM_MERAK_TRACE_RUN_ID", raising=False)
    with pytest.raises(RuntimeError, match="run ID"):
        emit_trace(
            GEMMA4_TRACE_ENV,
            component="model",
            event="chunk",
            payload={},
        )


def test_harness_trace_sinks_are_unique_per_run(monkeypatch, tmp_path) -> None:
    harness = _load_harness()
    args = SimpleNamespace(
        trace_dir=tmp_path / "traces",
        output=tmp_path / "result.jsonl",
    )

    first = harness._configure_trace_sinks(args)
    second = harness._configure_trace_sinks(args)

    assert first.run_id != second.run_id
    assert first.model_trace != second.model_trace
    assert first.kv_trace != second.kv_trace
    assert first.model_trace.read_bytes() == b""
    assert second.model_trace.read_bytes() == b""
    assert __import__("os").environ["VLLM_MERAK_TRACE_RUN_ID"] == second.run_id


def test_trace_validation_rejects_rows_from_another_run(tmp_path) -> None:
    harness = _load_harness()
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps({"run_id": "current"}) + "\n",
        encoding="utf-8",
    )
    assert harness.validate_trace_run(trace, "current") == 1

    with trace.open("a", encoding="utf-8") as sink:
        sink.write(json.dumps({"run_id": "stale"}) + "\n")
    with pytest.raises(RuntimeError, match="stale"):
        harness.validate_trace_run(trace, "current")


def test_harness_expands_boundary_and_three_request_matrix(tmp_path) -> None:
    harness = _load_harness()

    args = harness.build_parser().parse_args(
        [
            "--model-dir",
            str(tmp_path / "model"),
            "--mode",
            "hmonnx-cuda-graph",
            "--window",
            "1024",
            "--boundaries",
            "--repeat",
            "3",
            "--three-requests",
            "--output",
            str(tmp_path / "result.jsonl"),
        ]
    )

    assert args.reasoning_parser is None
    assert args.repeat == 3
    assert args.request_count == 3
    cases = harness.build_validation_cases(args)
    assert [case.target_prompt_tokens for case in cases] == [
        1023,
        1024,
        1025,
        1344,
        4096,
    ]
    assert {case.mode for case in cases} == {"hmonnx-cuda-graph"}
    assert {case.request_count for case in cases} == {3}
    assert {case.repeat for case in cases} == {3}


def test_harness_accepts_text_image_and_explicit_lengths(tmp_path) -> None:
    harness = _load_harness()

    image = tmp_path / "image.jpg"
    image.write_bytes(b"not-decoded-during-argument-planning")
    args = harness.build_parser().parse_args(
        [
            "--model-dir",
            str(tmp_path / "model"),
            "--mode",
            "eager",
            "--prompt-token-lengths",
            "511",
            "512",
            "832",
            "--image",
            str(image),
            "--three-requests",
            "--output",
            str(tmp_path / "result.jsonl"),
        ]
    )

    cases = harness.build_validation_cases(args)
    assert [case.target_prompt_tokens for case in cases] == [511, 512, 832]
    assert {case.input_kind for case in cases} == {"image"}
    assert all(case.image == image for case in cases)


@pytest.mark.parametrize("mode", ["eager", "hmonnx-cuda-graph"])
def test_harness_runtime_environment_is_mode_specific(monkeypatch, mode) -> None:
    harness = _load_harness()

    monkeypatch.delenv("VLLM_MERAK_CUDA_GRAPH", raising=False)
    harness.configure_runtime_environment(mode)

    assert ("VLLM_MERAK_CUDA_GRAPH" in __import__("os").environ) is (
        mode == "hmonnx-cuda-graph"
    )
    assert __import__("os").environ["VLLM_MERAK_USE_V2"] == "1"
    assert __import__("os").environ["VLLM_ENABLE_V1_MULTIPROCESSING"] == "0"


def test_harness_rejects_conflicting_length_modes(tmp_path) -> None:
    harness = _load_harness()

    args = harness.build_parser().parse_args(
        [
            "--model-dir",
            str(tmp_path / "model"),
            "--mode",
            "eager",
            "--window",
            "1024",
            "--boundaries",
            "--prompt-token-lengths",
            "1024",
            "--output",
            str(tmp_path / "result.jsonl"),
        ]
    )
    with pytest.raises(
        ValueError, match="either --boundaries or --prompt-token-lengths"
    ):
        harness.build_validation_cases(args)


def test_hmonnx_graph_matrix_requires_a_replay_attempt(tmp_path) -> None:
    harness = _load_harness()
    args = harness.build_parser().parse_args(
        [
            "--model-dir",
            str(tmp_path / "model"),
            "--mode",
            "hmonnx-cuda-graph",
            "--prompt-token-lengths",
            "320",
            "--repeat",
            "1",
            "--output",
            str(tmp_path / "result.jsonl"),
        ]
    )

    with pytest.raises(ValueError, match="repeat.*2"):
        harness.build_validation_cases(args)


def test_hardware_matrix_requires_three_requests_for_recycle_evidence(tmp_path) -> None:
    harness = _load_harness()
    args = harness.build_parser().parse_args(
        [
            "--model-dir",
            str(tmp_path / "model"),
            "--mode",
            "eager",
            "--prompt-token-lengths",
            "320",
            "--request-count",
            "2",
            "--output",
            str(tmp_path / "result.jsonl"),
        ]
    )

    with pytest.raises(ValueError, match="request-count.*3.*barrier"):
        harness.build_validation_cases(args)


def test_hardware_engine_disables_prefix_cache_for_recycle_evidence(tmp_path) -> None:
    harness = _load_harness()
    args = SimpleNamespace(
        model_dir=tmp_path / "model",
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        request_count=3,
    )
    artifact = harness.Gemma4ArtifactContract(
        meta_path=tmp_path / "meta.json",
        meta_sha256="0" * 64,
        attention_contract_version=2,
        prefill_chunk_size=320,
        visual_token_count=280,
    )

    kwargs = harness.build_llm_kwargs(args, artifact, object(), None)

    assert kwargs["enable_prefix_caching"] is False
    assert kwargs["max_num_seqs"] == 1
    assert kwargs["limit_mm_per_prompt"] == {
        "image": 0,
        "video": 0,
        "audio": 0,
    }


def test_hardware_engine_does_not_override_unified_visual_contract(tmp_path) -> None:
    harness = _load_harness()
    model_dir = tmp_path / "model"
    hf_config = model_dir / "hf_config"
    hf_config.mkdir(parents=True)
    (hf_config / "config.json").write_text(
        json.dumps({"model_type": "gemma4_unified"}),
        encoding="utf-8",
    )
    args = SimpleNamespace(
        model_dir=model_dir,
        max_model_len=8192,
        gpu_memory_utilization=0.9,
        image=tmp_path / "image.jpg",
    )
    artifact = harness.Gemma4ArtifactContract(
        meta_path=tmp_path / "meta.json",
        meta_sha256="0" * 64,
        attention_contract_version=2,
        prefill_chunk_size=320,
        visual_token_count=280,
    )

    kwargs = harness.build_llm_kwargs(args, artifact, object(), None)

    assert kwargs["limit_mm_per_prompt"] == {
        "image": 1,
        "video": 0,
        "audio": 0,
    }
    assert "mm_processor_kwargs" not in kwargs


def test_qwen_contract_v1_does_not_emit_gemma_trace(monkeypatch, tmp_path) -> None:
    from vllm_merak.model_executor.model_loader.merak_for_causal_lm import (
        MerakForCausalLM,
    )

    trace_path = tmp_path / "qwen-must-not-exist.jsonl"
    monkeypatch.setenv("VLLM_MERAK_GEMMA4_TRACE_JSONL", str(trace_path))
    model = MerakForCausalLM.__new__(MerakForCausalLM)
    model.inference_device = torch.device("cpu")
    model.input_embeddings = type("Embeddings", (), {"weight": torch.empty(1)})()
    model.logits_processor = type("Logits", (), {"vocab_size": 4})()
    model._prefer_input_ids_runtime = True
    model._ensure_runtime_cache_tensors = lambda: None

    class Runtime:
        def set_prefill(self):
            pass

        def set_decode(self):
            pass

        def get_input_sequence_length(self):
            return 4

        def get_data_preprocessor(self):
            return lambda data: (data["input_ids"],)

        def forward(self, input_ids):
            del input_ids
            return torch.arange(4, dtype=torch.float32)

    model.inference_model = Runtime()
    output = model._run_request(0, input_ids=torch.tensor([1, 2]))

    assert output.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert not trace_path.exists()


def test_qwen_contract_v1_does_not_emit_kv_trace(monkeypatch, tmp_path) -> None:
    from vllm_merak.worker_v1.merak_model_runner_v1 import MerakModelRunnerV1

    trace_path = tmp_path / "qwen-kv-must-not-exist.jsonl"
    monkeypatch.setenv("VLLM_MERAK_KV_TRACE_JSONL", str(trace_path))
    runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
    runner.model = SimpleNamespace(
        _uses_gemma4_page_attention_contract=lambda: False,
        placeholder_attn_layers=[],
    )
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    runner._kv_block_zeroer = SimpleNamespace(zero_block_ids=lambda _ids: None)

    runner._zero_block_ids([3])

    assert not trace_path.exists()


def test_image_prompt_builder_targets_expanded_visual_length(tmp_path) -> None:
    harness = _load_harness()
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fixture")
    case = harness.ValidationCase(12, "eager", "image", image, 1, 1)
    args = SimpleNamespace(unit="x", prompt_text="p")
    artifact = harness.Gemma4ArtifactContract(
        meta_path=tmp_path / "golden_meta_info.json",
        meta_sha256="0" * 64,
        attention_contract_version=2,
        prefill_chunk_size=320,
        visual_token_count=3,
    )

    class Tokenizer:
        @staticmethod
        def apply_chat_template(messages, *, tokenize, add_generation_prompt):
            assert tokenize and add_generation_prompt
            text = messages[0]["content"][0]["text"]
            # Two fixed chat-template tokens plus one image placeholder. The
            # runtime expands that placeholder from one token to three and
            # adds the two Gemma image-boundary tokens.
            return list(range(len(text) + 3))

    messages = harness.build_exact_image_messages(Tokenizer(), case, args, artifact)
    rendered = Tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    assert len(rendered) - 1 + artifact.visual_token_count + 2 == 12
    assert messages[0]["content"][0]["text"].endswith("p")


def test_image_prompt_builder_uses_batch_encoding_input_ids(tmp_path) -> None:
    harness = _load_harness()
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fixture")
    case = harness.ValidationCase(12, "eager", "image", image, 1, 1)
    args = SimpleNamespace(unit="x", prompt_text="p")
    artifact = harness.Gemma4ArtifactContract(
        meta_path=tmp_path / "golden_meta_info.json",
        meta_sha256="0" * 64,
        attention_contract_version=2,
        prefill_chunk_size=320,
        visual_token_count=3,
    )

    class Tokenizer:
        @staticmethod
        def apply_chat_template(messages, *, tokenize, add_generation_prompt):
            assert tokenize and add_generation_prompt
            text = messages[0]["content"][0]["text"]
            token_ids = list(range(len(text) + 3))
            return {
                "input_ids": token_ids,
                "attention_mask": [1] * len(token_ids),
            }

    messages = harness.build_exact_image_messages(Tokenizer(), case, args, artifact)
    rendered = Tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    assert len(rendered["input_ids"]) - 1 + artifact.visual_token_count + 2 == 12


def _write_gemma4_artifact(
    model_dir: Path,
    *,
    visual_token_count: int = 280,
    prefill_chunk_size: int = 320,
) -> Path:
    model_dir.mkdir()
    (model_dir / "merak_config.json").write_text(
        json.dumps(
            {
                "xh_model": {
                    "model_type": "hmonnx",
                    "meta_info": "golden_meta_info.json",
                }
            }
        ),
        encoding="utf-8",
    )
    meta_path = model_dir / "golden_meta_info.json"
    meta_path.write_text(
        json.dumps(
            {
                "attention_contract_version": 2,
                "model_config": {"prefill_chunk_length": prefill_chunk_size},
                "visual_config": {
                    "hmonnx": "visual/visual.hmonnx",
                    "num_image_tokens": visual_token_count,
                },
            }
        ),
        encoding="utf-8",
    )
    return meta_path


def test_harness_derives_visual_span_and_atomic_chunk_from_artifact(tmp_path) -> None:
    harness = _load_harness()
    model_dir = tmp_path / "model"
    meta_path = _write_gemma4_artifact(model_dir)

    artifact = harness.load_gemma4_artifact_contract(model_dir)

    assert artifact.meta_path == meta_path
    assert len(artifact.meta_sha256) == 64
    assert artifact.attention_contract_version == 2
    assert artifact.visual_token_count == 280
    assert artifact.prefill_chunk_size == 320


def test_harness_fingerprints_referenced_hmonnx_graphs(tmp_path) -> None:
    harness = _load_harness()
    model_dir = tmp_path / "model"
    _write_gemma4_artifact(model_dir)
    graph = model_dir / "visual" / "visual.hmonnx"
    graph.parent.mkdir()
    graph.write_bytes(b"graph-v1")
    artifact = harness.load_gemma4_artifact_contract(model_dir)

    fingerprints = harness._referenced_graph_fingerprints(artifact)

    assert fingerprints == [
        {
            "path": str(graph.resolve()),
            "bytes": len(b"graph-v1"),
            "sha256": __import__("hashlib").sha256(b"graph-v1").hexdigest(),
        }
    ]


@pytest.mark.parametrize(
    ("visual_token_count", "prefill_chunk_size", "message"),
    [(281, 320, "visual.*maximum.*280"), (280, 319, "chunk.*320")],
)
def test_harness_rejects_artifact_outside_visual_atomic_contract(
    tmp_path, visual_token_count, prefill_chunk_size, message
) -> None:
    harness = _load_harness()
    model_dir = tmp_path / "model"
    _write_gemma4_artifact(
        model_dir,
        visual_token_count=visual_token_count,
        prefill_chunk_size=prefill_chunk_size,
    )

    with pytest.raises(ValueError, match=message):
        harness.load_gemma4_artifact_contract(model_dir)


def test_gemma4_loader_is_routable_through_real_vllm_multimodal_registry() -> None:
    from vllm.model_executor.models.gemma4_mm import Gemma4MultiModalProcessor

    from vllm_merak.model_executor.model_loader.gemma4 import (
        Gemma4MerakForCausalLM,
    )

    assert Gemma4MerakForCausalLM.supports_multimodal is True
    assert Gemma4MerakForCausalLM.requires_raw_input_tokens is True
    factories = Gemma4MerakForCausalLM._processor_factory
    assert issubclass(factories.processor, Gemma4MultiModalProcessor)


def test_gemma4_loader_exposes_upstream_multimodal_placeholders() -> None:
    from vllm_merak.model_executor.model_loader.gemma4 import (
        Gemma4MerakForCausalLM,
    )

    assert Gemma4MerakForCausalLM.get_placeholder_str("image", 0) == (
        "<image_soft_token>"
    )
    assert Gemma4MerakForCausalLM.get_placeholder_str("audio", 0) == (
        "<audio_soft_token>"
    )
    assert Gemma4MerakForCausalLM.get_placeholder_str("video", 0) == "<|video|>"
    with pytest.raises(ValueError, match="Unsupported modality: depth"):
        Gemma4MerakForCausalLM.get_placeholder_str("depth", 0)


def test_real_gemma4_prompt_replacement_marks_280_contiguous_visual_tokens() -> None:
    from vllm.multimodal.processing.processor import PromptUpdateDetails

    from vllm_merak.model_executor.model_loader.gemma4 import (
        MerakGemma4MultiModalProcessor,
    )

    image_token_id = 99
    processor = object.__new__(MerakGemma4MultiModalProcessor)
    processor.info = SimpleNamespace(
        get_hf_processor=lambda **_kwargs: SimpleNamespace(image_token="<image>"),
        ctx=SimpleNamespace(
            get_merged_mm_kwargs=lambda _kwargs: {"max_soft_tokens": 280}
        ),
        get_image_repl=lambda **_kwargs: PromptUpdateDetails.select_token_id(
            [7, *([image_token_id] * 280), 8], image_token_id
        ),
    )

    class Items:
        def __contains__(self, modality):
            return modality == "image"

        @staticmethod
        def get_items(_modality, _item_type):
            return SimpleNamespace(
                get_image_size=lambda _index: SimpleNamespace(width=448, height=448)
            )

    (update,) = processor._get_prompt_updates(Items(), {}, SimpleNamespace())
    details = update.replacement(0)
    mask = details.is_embed(None, details.full)

    assert details.full == [7, *([image_token_id] * 280), 8]
    assert mask.tolist() == [False, *([True] * 280), False]


def test_gemma4_encoder_budget_uses_vit_output_tokens_not_patch_inputs(
    monkeypatch,
) -> None:
    from vllm_merak.model_executor.model_loader import gemma4

    source_config = SimpleNamespace(
        vision_config=SimpleNamespace(default_output_length=2496),
        audio_config=None,
    )
    model_config = SimpleNamespace(
        model="/fake/gemma4",
        get_multimodal_config=lambda: SimpleNamespace(limit_per_prompt={}),
    )
    info = gemma4.MerakGemma4ProcessingInfo(
        SimpleNamespace(
            model_config=model_config,
            get_merged_mm_kwargs=lambda _kwargs: {},
        )
    )
    monkeypatch.setattr(
        gemma4,
        "_load_merak_gemma4_source_config",
        lambda _model_id: source_config,
    )
    monkeypatch.setattr(
        gemma4,
        "_load_merak_gemma4_meta",
        lambda _model_id: {
            "visual_config": {
                "max_patches": 280 * 9,
                "num_image_tokens": 280,
                "pooling_kernel_size": 3,
            }
        },
    )

    budget = info.get_mm_max_tokens_per_item(262144, {"image": 1})

    assert budget is not None
    assert budget["image"] == 280
    assert "video" not in budget
    assert info.get_supported_mm_limits() == {"image": None}


def test_gemma4_processor_builds_fixed_hmonnx_visual_inputs() -> None:
    from transformers import BatchFeature

    from vllm_merak.model_executor.model_loader.gemma4 import (
        _augment_hmonnx_visual_inputs,
    )

    positions = torch.tensor(
        [[[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2]]]
    )
    outputs = BatchFeature(
        {
            "pixel_values": torch.ones((1, 9, 4)),
            "pixel_position_ids": positions,
        }
    )

    actual = _augment_hmonnx_visual_inputs(
        outputs,
        visual_token_count=2,
        pooling_kernel_size=3,
    )

    assert actual["pixel_values"].shape == (1, 18, 4)
    assert actual["pixel_position_ids"].shape == (1, 18, 2)
    assert actual["pooling_matrix"].shape == (1, 2, 18)
    assert actual["visual_attention_mask"].shape == (1, 1, 1, 18)
    assert actual["image_soft_token_count"].tolist() == [1]
    assert torch.count_nonzero(actual["pooling_matrix"][0, 0]) == 9
    assert torch.all(actual["pixel_position_ids"][0, 9:] == 0)
    assert torch.all(actual["visual_attention_mask"][0, 0, 0, 9:] < 0)


def test_gemma4_model_routes_all_four_exported_visual_inputs() -> None:
    from vllm_merak.model_executor.model_loader.gemma4 import (
        Gemma4MerakForCausalLM,
    )

    calls = []

    class Visual:
        device = torch.device("cpu")
        dtype = torch.float16

        @staticmethod
        def forward(*args):
            calls.append(args)
            return torch.arange(12, dtype=torch.float16).reshape(1, 3, 4)

    model = Gemma4MerakForCausalLM.__new__(Gemma4MerakForCausalLM)
    model.inference_model = SimpleNamespace(visual=Visual())
    model._runner_device = torch.device("cpu")
    model.input_embeddings = SimpleNamespace(weight=torch.empty(1, dtype=torch.float16))
    embeddings = model.embed_multimodal(
        pixel_values=torch.ones((1, 18, 4)),
        pixel_position_ids=torch.zeros((1, 18, 2), dtype=torch.int32),
        pooling_matrix=torch.ones((1, 3, 18), dtype=torch.float16),
        visual_attention_mask=torch.zeros((1, 1, 1, 18), dtype=torch.float16),
        image_soft_token_count=torch.tensor([2]),
    )

    assert len(calls) == 1 and len(calls[0]) == 4
    assert isinstance(embeddings, tuple)
    assert embeddings[0].shape == (2, 4)


def test_visual_span_validation_requires_one_contiguous_artifact_sized_span() -> None:
    harness = _load_harness()
    artifact = harness.Gemma4ArtifactContract(
        meta_path=Path("golden_meta_info.json"),
        meta_sha256="0" * 64,
        attention_contract_version=2,
        prefill_chunk_size=320,
        visual_token_count=280,
    )

    prompt_ids = [7, 8, *([99] * 280), 10]
    assert harness.validate_processed_visual_span(prompt_ids, 99, artifact) == (
        2,
        281,
    )

    with pytest.raises(ValueError, match="contiguous"):
        harness.validate_processed_visual_span(
            [7, *([99] * 140), 8, *([99] * 140)], 99, artifact
        )
    with pytest.raises(ValueError, match="artifact.*280"):
        harness.validate_processed_visual_span([7, *([99] * 279)], 99, artifact)


def test_harness_result_row_has_hardware_evidence_schema() -> None:
    harness = _load_harness()
    case = harness.ValidationCase(5, "eager", "text", None, 1, 2)
    output = SimpleNamespace(
        request_id="req-2",
        prompt_token_ids=[1, 2, 3, 4, 5],
        outputs=[
            SimpleNamespace(
                text="ok",
                token_ids=[7, 8],
                finish_reason="length",
            )
        ],
    )

    row = harness._result_row(
        case=case,
        output=output,
        elapsed_s=1.25,
        repeat_index=1,
        request_index=2,
    )

    assert row["schema"] == "vllm-merak.gemma4.hardware-validation/v1"
    assert row["request_id"] == "req-2"
    assert row["target_prompt_tokens"] == row["actual_prompt_tokens"] == 5
    assert row["request_index"] == 2
    assert row["mode"] == "eager"
    assert row["hardware_claim"] is False


def _model_trace_rows(*, start: int = 477, seq_len: int = 1820, graph=None):
    execution_mode = "cuda_graph" if graph is not None else "eager"
    common = {
        "run_id": "run-1",
        "request_id": "req-1",
        "phase": "prefill",
        "P": 1500,
        "Q": 320,
    }
    return [
        {
            **common,
            "event": "request_plan",
            "chunk_plan": [[0, 320]],
            "inclusive_mm_ranges": [[1500, 1779]],
        },
        {
            **common,
            "event": "chunk_context",
            "cache_group_idx": 0,
            "cache_indices": [0],
            "attention_kind": "full_attention",
            "sliding_window": None,
            "kv_window_start_abs": None,
            "seq_lens": [seq_len],
            "mm_prefix_ranges": [[[1500, 1779]]],
            "execution_mode": execution_mode,
            "metadata_ptrs": {
                "block_ids": 11,
                "slot_mapping": 12,
                "seq_lens": 13,
                "kv_window_start_abs": None,
                "mm_prefix_ranges": 14,
            },
            "backing_ptrs": {"0": 101},
            "backing_bytes": {"0": 4096},
            "graph_runtime": graph,
        },
        {
            **common,
            "event": "chunk_context",
            "cache_group_idx": 1,
            "cache_indices": [1],
            "attention_kind": "sliding_attention",
            "sliding_window": 1024,
            "kv_window_start_abs": [start],
            "seq_lens": [seq_len],
            "mm_prefix_ranges": [[[1500, 1779]]],
            "execution_mode": execution_mode,
            "metadata_ptrs": {
                "block_ids": 21,
                "slot_mapping": 22,
                "seq_lens": 23,
                "kv_window_start_abs": 24,
                "mm_prefix_ranges": 25,
            },
            "backing_ptrs": {"1": 201},
            "backing_bytes": {"1": 2048},
            "graph_runtime": graph,
        },
    ]


def _kv_recycle_rows():
    backing = {"ptr": 201, "bytes": 2048, "cache_group_indices": [1]}
    record = {
        "request_id": "req-1",
        "group_idx": 1,
        "backing_ptr": 201,
        "block_ids": [3],
    }
    return [
        {
            "run_id": "run-1",
            "event": "ownership",
            "request_id": "req-1",
            "owned_block_ids_by_group": {"1": [3]},
            "unique_backings": [backing],
        },
        {
            "run_id": "run-1",
            "event": "release",
            "request_id": "req-1",
            "released_block_ids": [3],
            "release_records": [record],
            "unique_backings": [backing],
        },
        {
            "run_id": "run-1",
            "event": "zero_complete",
            "ordering": "before_model_execute",
            "zeroed_block_ids": [3],
            "recycled_block_ids": [3],
            "zero_records": [record],
            "unique_backings": [backing],
        },
    ]


def test_hardware_evidence_validates_visibility_formula_and_cache_proof() -> None:
    harness = _load_harness()
    kv_rows = _kv_recycle_rows()

    summary = harness.validate_hardware_evidence(
        model_rows=_model_trace_rows(),
        kv_rows=kv_rows,
        mode="eager",
    )

    assert summary["chunk_context_rows"] == 2
    assert summary["zero_complete_rows"] == 1
    assert summary["hardware_claim"] is True


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"start": 478}, "kv_window_start_abs"),
        ({"seq_len": 1819}, "seq_lens"),
    ],
)
def test_hardware_evidence_rejects_incorrect_visibility_formula(
    kwargs, message
) -> None:
    harness = _load_harness()
    with pytest.raises(RuntimeError, match=message):
        harness.validate_hardware_evidence(
            model_rows=_model_trace_rows(**kwargs),
            kv_rows=_kv_recycle_rows(),
            mode="eager",
        )


def test_hmonnx_graph_evidence_requires_observed_replay() -> None:
    harness = _load_harness()
    kv_rows = _kv_recycle_rows()

    with pytest.raises(RuntimeError, match="replay"):
        harness.validate_hardware_evidence(
            model_rows=_model_trace_rows(graph={"replay_count": 0}),
            kv_rows=kv_rows,
            mode="hmonnx-cuda-graph",
        )

    summary = harness.validate_hardware_evidence(
        model_rows=_model_trace_rows(graph={"replay_count": 1}),
        kv_rows=kv_rows,
        mode="hmonnx-cuda-graph",
    )
    assert summary["graph_replay_observed"] is True


def test_hardware_evidence_rejects_missing_recycle_proof() -> None:
    harness = _load_harness()
    kv_rows = _kv_recycle_rows()
    kv_rows[-1]["recycled_block_ids"] = []
    kv_rows[-1]["zero_records"] = []
    with pytest.raises(RuntimeError, match="recycl"):
        harness.validate_hardware_evidence(
            model_rows=_model_trace_rows(),
            kv_rows=kv_rows,
            mode="eager",
        )


def test_hardware_evidence_matches_visual_ranges_to_atomic_request_plan() -> None:
    harness = _load_harness()
    model_rows = _model_trace_rows()
    for row in model_rows:
        if row["event"] == "chunk_context":
            row["mm_prefix_ranges"] = [[[1499, 1778]]]

    with pytest.raises(RuntimeError, match="visual.*plan"):
        harness.validate_hardware_evidence(
            model_rows=model_rows,
            kv_rows=_kv_recycle_rows(),
            mode="eager",
        )


def test_hardware_claim_is_promoted_only_after_evidence_validation(tmp_path) -> None:
    harness = _load_harness()
    output = tmp_path / "pending.jsonl"
    output.write_text(
        json.dumps({"event": "manifest", "hardware_claim": False})
        + "\n"
        + json.dumps({"event": "request_result", "hardware_claim": False})
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "hardware_claim": True,
        "chunk_context_rows": 2,
        "zero_complete_rows": 1,
    }

    harness.promote_hardware_claim(output, summary)

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert all(row["hardware_claim"] is True for row in rows)
    assert rows[0]["evidence_summary"] == summary
