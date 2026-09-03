import copy
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
from safetensors.numpy import save_file as save_safetensors

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_build.fingerprints import resource_sha256 as _resource_sha256
from xhmodel_merak.model_catalog.loading import load_adapter
from xhmodel_merak.model_integrations.autoregressive import build as llm
from xhmodel_merak.model_integrations.pi05 import build as pi05


@pytest.mark.parametrize("module", ["llm", "autoregressive"])
def test_removed_build_facades_are_not_importable(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(f"xhmodel_merak.model_build.adapters.{module}")


@pytest.mark.parametrize(
    ("name", "canonical"),
    (
        ("autoregressive.build", "xhmodel_merak.model_integrations.autoregressive.build"),
        ("pi05.build", "xhmodel_merak.model_integrations.pi05.build"),
        ("kokoro.build", "xhmodel_merak.model_integrations.kokoro.build"),
    ),
)
def test_registered_build_adapter_loads_family_implementation(name: str, canonical: str) -> None:
    implementation = load_adapter("build", name)
    canonical_module = importlib.import_module(canonical)
    for hook in ("normalize_export", "package_runtime", "sanitize_artifact", "validate_artifact"):
        assert getattr(implementation, hook) is getattr(canonical_module, hook)


def _touch(root: Path, relative: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _patch_auto_tokenizer(monkeypatch: pytest.MonkeyPatch, loader) -> None:
    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=loader)
    monkeypatch.setitem(sys.modules, "transformers", transformers)


def _write_pi05_policy(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "chunk_size": 50,
        "input_features": {"observation.state": {"shape": [8]}},
        "output_features": {"action": {"shape": [7]}},
        "max_action_dim": 32,
        "num_inference_steps": 10,
        "image_resolution": [224, 224],
        "normalization_mapping": {"STATE": "QUANTILES", "ACTION": "QUANTILES"},
    }
    for name in pi05._POLICY_FILES:
        if name.endswith(".json"):
            (root / name).write_text(json.dumps(config if name == "config.json" else {}), encoding="utf-8")
    save_safetensors(
        {
            "observation.state.q01": np.zeros(8, dtype=np.float32),
            "observation.state.q99": np.ones(8, dtype=np.float32),
        },
        root / "policy_preprocessor_step_2_normalizer_processor.safetensors",
    )
    save_safetensors(
        {
            "action.q01": np.zeros(7, dtype=np.float32),
            "action.q99": np.ones(7, dtype=np.float32),
        },
        root / "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    )


def test_pi05_adapter_packages_only_runtime_assets_and_validates_component_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_pi05_policy(source)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.model").write_bytes(b"tokenizer")
    (tokenizer / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tokenizer / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    stage = tmp_path / "artifact"
    stage.mkdir()

    components = {
        "vision": {
            "component_dir": "Vision",
            "hmonnx_file": "hmonnx/vision.onnx",
            "input_shape": [1, 3, 224, 224],
        },
        "other": {
            "component_dir": "Other",
            "graphs": [
                {
                    "name": "action_in_proj",
                    "hmonnx_file": "hmonnx/action_in_proj.onnx",
                    "input_shape": [1, 50, 32],
                },
                {
                    "name": "action_out_proj",
                    "hmonnx_file": "hmonnx/action_out_proj.onnx",
                    "input_shape": [1, 50, 1024],
                },
                {
                    "name": "time_mlp",
                    "hmonnx_file": "hmonnx/time_mlp.onnx",
                    "input_shape": [1, 1024],
                },
            ],
        },
        "gemma": {"component_dir": "Gemma2B", "meta_file": "export_meta_info.json"},
        "expert": {"component_dir": "GemmaExpert", "meta_file": "export_meta_info.json"},
    }
    (stage / "export_meta_info.json").write_text(
        json.dumps(
            {
                "hf_model": str(source),
                "compact_prefix": {
                    "selected_image_indices": [0, 1],
                    "action_horizon": 50,
                    "text_max_length": 200,
                    "prefix_sequence_length": 712,
                    "cache_length": 1024,
                },
                "components": components,
            }
        ),
        encoding="utf-8",
    )

    class FakeTokenizer:
        pad_token_id = 0

        def __call__(self, _text, *, max_length, **_kwargs):
            return {
                "input_ids": SimpleNamespace(shape=(1, max_length)),
                "attention_mask": SimpleNamespace(shape=(1, max_length)),
            }

    _patch_auto_tokenizer(monkeypatch, lambda *_args, **_kwargs: FakeTokenizer())
    _touch(stage, "Vision/hmonnx/vision.onnx")
    for graph in components["other"]["graphs"]:
        _touch(stage, f"Other/{graph['hmonnx_file']}")
    for name, component in (("gemma", components["gemma"]), ("expert", components["expert"])):
        component_root = stage / component["component_dir"]
        component_root.mkdir()
        nested = {
            "hf_model": str(source),
            "prefill_onnx_file": "prefill/model.onnx",
            "decode_onnx_file": "decode/model.onnx",
            "kv_cache_shape": [1, 2, 1024, 64],
            "num_hidden_layers": 18,
            "hidden_size": 2048 if name == "gemma" else 1024,
            "prefill_input_sequence_length": 712 if name == "gemma" else 50,
            "decode_input_sequence_length": 1 if name == "gemma" else 50,
            "prefill_attention_mask_shape": [1, 1, 1, 1024],
            "decode_attention_mask_shape": [1, 1, 1, 1024],
            "selected_image_indices": [0, 1],
            "text_max_length": 200,
            "prefix_sequence_length": 712,
            "action_horizon": 50,
            "cache_length": 1024,
        }
        if name == "gemma":
            nested["token_embedding_file"] = "token_embedding.pt"
            _touch(component_root, "token_embedding.pt")
        (component_root / "export_meta_info.json").write_text(json.dumps(nested), encoding="utf-8")
        _touch(component_root, "prefill/model.onnx")
        _touch(component_root, "decode/model.onnx")

    plan = SimpleNamespace(
        model_id="pi0.5-libero",
        model_dir=str(source),
        components=("vision", "gemma", "expert", "other"),
        resources={"tokenizer": SimpleNamespace(path=str(tokenizer), sha256=_resource_sha256(tokenizer))},
    )
    pi05.package_runtime(stage, plan)
    pi05.sanitize_artifact(stage, plan)
    metadata = pi05.validate_artifact(stage, plan)

    assert metadata["loadable"] is True
    assert metadata["components"] == list(plan.components)
    assert not (stage / "runtime/policy/model.safetensors").exists()
    assert json.loads((stage / "export_meta_info.json").read_text(encoding="utf-8"))["hf_model"] == "runtime/policy"

    valid_meta = json.loads((stage / "export_meta_info.json").read_text(encoding="utf-8"))
    invalid_metadata = (
        (("components", "other", "graphs", 0, "name"), "arbitrary", "graph set"),
        (("compact_prefix", "selected_image_indices"), [1, 0], "selected_image_indices"),
        (("compact_prefix", "action_horizon"), 49, "action_horizon"),
        (("compact_prefix", "cache_length"), 700, "capacity"),
        (("components", "vision", "input_shape"), [1, 3, 256, 256], "image_resolution"),
        (("components", "other", "graphs", 0, "input_shape"), [1, 50, 16], "action_in_proj"),
    )
    for path, value, message in invalid_metadata:
        candidate = copy.deepcopy(valid_meta)
        target = candidate
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        (stage / "export_meta_info.json").write_text(json.dumps(candidate), encoding="utf-8")
        with pytest.raises(ModelError, match=message):
            pi05.validate_artifact(stage, plan)

    (stage / "export_meta_info.json").write_text(json.dumps(valid_meta), encoding="utf-8")
    expert_meta_path = stage / "GemmaExpert/export_meta_info.json"
    valid_expert_meta = json.loads(expert_meta_path.read_text(encoding="utf-8"))
    invalid_expert_meta = copy.deepcopy(valid_expert_meta)
    invalid_expert_meta["kv_cache_shape"][2] = 512
    expert_meta_path.write_text(json.dumps(invalid_expert_meta), encoding="utf-8")
    with pytest.raises(ModelError, match="KV cache"):
        pi05.validate_artifact(stage, plan)
    expert_meta_path.write_text(json.dumps(valid_expert_meta), encoding="utf-8")

    policy_path = stage / "runtime/policy/config.json"
    valid_policy = json.loads(policy_path.read_text(encoding="utf-8"))
    for field, value, message in (
        ("max_action_dim", 6, "max_action_dim"),
        ("num_inference_steps", 0, "num_inference_steps"),
        ("image_resolution", [224, 0], "image_resolution"),
    ):
        invalid_policy = copy.deepcopy(valid_policy)
        invalid_policy[field] = value
        policy_path.write_text(json.dumps(invalid_policy), encoding="utf-8")
        with pytest.raises(ModelError, match=message):
            pi05.validate_artifact(stage, plan)
    invalid_policy = copy.deepcopy(valid_policy)
    invalid_policy["input_features"]["observation.state"]["shape"] = [7]
    policy_path.write_text(json.dumps(invalid_policy), encoding="utf-8")
    with pytest.raises(ModelError, match="policy dimensions"):
        pi05.validate_artifact(stage, plan)
    invalid_policy = copy.deepcopy(valid_policy)
    invalid_policy["output_features"]["action"]["shape"] = [6]
    policy_path.write_text(json.dumps(invalid_policy), encoding="utf-8")
    with pytest.raises(ModelError, match="policy dimensions"):
        pi05.validate_artifact(stage, plan)
    policy_path.write_text(json.dumps(valid_policy), encoding="utf-8")


def test_pi05_adapter_rejects_component_metadata_path_escape(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    _write_pi05_policy(source)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.model").write_bytes(b"tokenizer")
    (tokenizer / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tokenizer / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    stage = tmp_path / "artifact"
    stage.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (stage / "export_meta_info.json").write_text(
        json.dumps(
            {
                "compact_prefix": {"text_max_length": 48},
                "components": {
                    "gemma": {"component_dir": "..", "meta_file": "outside.json"},
                    "expert": {"component_dir": "missing", "meta_file": "meta.json"},
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeTokenizer:
        pad_token_id = 0

        def __call__(self, _text, *, max_length, **_kwargs):
            return {
                "input_ids": SimpleNamespace(shape=(1, max_length)),
                "attention_mask": SimpleNamespace(shape=(1, max_length)),
            }

    _patch_auto_tokenizer(monkeypatch, lambda *_args, **_kwargs: FakeTokenizer())
    plan = SimpleNamespace(
        model_id="pi0.5-libero",
        model_dir=str(source),
        resources={"tokenizer": SimpleNamespace(path=str(tokenizer), sha256=_resource_sha256(tokenizer))},
    )

    with pytest.raises(ModelError) as exc_info:
        pi05.package_runtime(stage, plan)

    assert exc_info.value.code == ErrorCode.ARTIFACT_VALIDATION_FAILED
    assert json.loads(outside.read_text(encoding="utf-8")) == {}


def test_pi05_adapter_rejects_sentencepiece_only_tokenizer(tmp_path: Path) -> None:
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    (tokenizer / "tokenizer.model").write_bytes(b"tokenizer")
    plan = SimpleNamespace(model_id="pi0.5-libero")

    with pytest.raises(ModelError, match="complete HF snapshot"):
        pi05._validate_tokenizer_snapshot(tokenizer, 48, plan)


def test_pi05_adapter_rejects_symlinked_source_root_and_policy_child(tmp_path: Path) -> None:
    outside = tmp_path / "outside-policy"
    _write_pi05_policy(outside)
    source_link = tmp_path / "source-link"
    source_link.symlink_to(outside, target_is_directory=True)
    stage = tmp_path / "artifact"
    stage.mkdir()

    with pytest.raises(ModelError, match="root must not be a symlink"):
        pi05.package_runtime(
            stage,
            SimpleNamespace(model_id="pi0.5-libero", model_dir=str(source_link)),
        )

    source = tmp_path / "source"
    _write_pi05_policy(source)
    external_config = tmp_path / "config.json"
    external_config.write_text((source / "config.json").read_text(encoding="utf-8"), encoding="utf-8")
    (source / "config.json").unlink()
    (source / "config.json").symlink_to(external_config)
    with pytest.raises(ModelError, match="must not contain symlinks"):
        pi05.package_runtime(
            stage,
            SimpleNamespace(model_id="pi0.5-libero", model_dir=str(source)),
        )


def test_pi05_adapter_rejects_symlinked_tokenizer_child(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_pi05_policy(source)
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    external_model = tmp_path / "external-tokenizer.model"
    external_model.write_bytes(b"external")
    (tokenizer / "tokenizer.model").symlink_to(external_model)
    (tokenizer / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (tokenizer / "special_tokens_map.json").write_text("{}", encoding="utf-8")
    stage = tmp_path / "artifact"
    stage.mkdir()
    plan = SimpleNamespace(
        model_id="pi0.5-libero",
        model_dir=str(source),
        resources={"tokenizer": SimpleNamespace(path=str(tokenizer), sha256="unused")},
    )

    with pytest.raises(ModelError, match="tokenizer source must not contain symlinks"):
        pi05.package_runtime(stage, plan)

    assert not (stage / "runtime/tokenizer").exists()


@pytest.mark.parametrize("failure", ["malformed", "empty", "missing", "nonfinite"])
def test_pi05_adapter_rejects_malformed_policy_safetensors(tmp_path: Path, failure: str) -> None:
    source = tmp_path / "source"
    _write_pi05_policy(source)
    stats_path = source / "policy_preprocessor_step_2_normalizer_processor.safetensors"
    if failure == "malformed":
        stats_path.write_bytes(b"{}")
    elif failure == "empty":
        save_safetensors({}, stats_path)
    elif failure == "missing":
        save_safetensors({"unrelated": np.zeros(1, dtype=np.float32)}, stats_path)
    else:
        save_safetensors(
            {
                "observation.state.q01": np.zeros(8, dtype=np.float32),
                "observation.state.q99": np.full(8, np.nan, dtype=np.float32),
            },
            stats_path,
        )

    with pytest.raises(ModelError, match="safetensors|tensors|normalization"):
        pi05._validate_policy_snapshot(source, SimpleNamespace(model_id="pi0.5-libero"))


def test_llm_adapter_rejects_runtime_length_mismatch_and_unusable_tokenizer(monkeypatch, tmp_path):
    plan = SimpleNamespace(
        model_id="qwen3.5-4b",
        context_length=2048,
        prefill_chunk_length=256,
        max_pe_length=262144,
    )
    with pytest.raises(ModelError, match="runtime lengths"):
        llm._validate_runtime_lengths(
            {
                "model_config": {
                    "context_max_length": 1,
                    "prefill_chunk_length": 256,
                    "max_pe_length": 262144,
                }
            },
            plan,
        )

    class EmptyTokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return "prompt"

        def __call__(self, *_args, **_kwargs):
            return {"input_ids": []}

    _patch_auto_tokenizer(monkeypatch, lambda *_args, **_kwargs: EmptyTokenizer())
    with pytest.raises(ModelError, match="usable local tokenizer"):
        llm._validate_tokenizer_snapshot(tmp_path, plan)


def test_llm_adapter_validates_builtin_tokenizer_without_executing_auto_map_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    marker = tmp_path / "artifact-code-executed"
    (tokenizer / "malicious.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (tokenizer / "vocab.txt").write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nruntime\nvalidation\n",
        encoding="utf-8",
    )
    (tokenizer / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "BertTokenizer",
                "auto_map": {"AutoTokenizer": ["malicious.MaliciousTokenizer", None]},
                "chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}",
            }
        ),
        encoding="utf-8",
    )

    class BuiltinTokenizer:
        def apply_chat_template(self, *_args, **_kwargs):
            return "runtime validation"

        def __call__(self, *_args, **_kwargs):
            return {"input_ids": [1, 2]}

    def load_tokenizer(path, *, local_files_only, trust_remote_code):
        assert local_files_only is True
        config = json.loads((Path(path) / "tokenizer_config.json").read_text(encoding="utf-8"))
        assert config["tokenizer_class"] == "BertTokenizer"
        if trust_remote_code:
            source = Path(path) / "malicious.py"
            exec(compile(source.read_bytes(), str(source), "exec"), {})
        return BuiltinTokenizer()

    _patch_auto_tokenizer(monkeypatch, load_tokenizer)

    llm._validate_tokenizer_snapshot(tokenizer, SimpleNamespace(model_id="qwen"))

    assert not marker.exists()


def test_llm_adapter_normalizes_transformers_5_tokenizer_backend_for_457(tmp_path):
    stage = tmp_path / "artifact"
    hf_config = stage / "hf_config"
    hf_config.mkdir(parents=True)
    (hf_config / "tokenizer.json").write_text(
        json.dumps({"added_tokens": [{"content": "/"}]}),
        encoding="utf-8",
    )
    (hf_config / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "TokenizersBackend"}),
        encoding="utf-8",
    )
    (stage / "golden_meta_info.json").write_text(
        json.dumps({"hf_config": "hf_config"}),
        encoding="utf-8",
    )
    plan = SimpleNamespace(model_dir=str(tmp_path / "source"), quant_checkpoint=None, sources={}, resources={})

    llm.sanitize_artifact(stage, plan)

    tokenizer_config = json.loads((hf_config / "tokenizer_config.json").read_text(encoding="utf-8"))
    assert tokenizer_config["tokenizer_class"] == "PreTrainedTokenizerFast"
    tokenizer_data = json.loads((hf_config / "tokenizer.json").read_text(encoding="utf-8"))
    assert tokenizer_data["added_tokens"][0]["content"] == "/"


def test_llm_adapter_binds_reused_checkpoint_tokenizer_processor_and_quant_config(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    checkpoint_config = {
        "model_type": "qwen3_5",
        "text_config": {"hidden_size": 8, "num_hidden_layers": 2, "rope_theta": 1_000_000.0},
        "quantization_config": {"bits": 4, "quant_method": "gptq"},
    }
    (checkpoint / "config.json").write_text(json.dumps(checkpoint_config), encoding="utf-8")
    (checkpoint / "tokenizer.model").write_bytes(b"checkpoint-tokenizer")
    (checkpoint / "tokenizer_config.json").write_text(
        json.dumps({"asset_source": "checkpoint"}),
        encoding="utf-8",
    )
    (checkpoint / "processor_config.json").write_text(
        json.dumps({"processor_class": "CheckpointProcessor"}),
        encoding="utf-8",
    )
    (checkpoint / "quantize_config.json").write_text(json.dumps({"bits": 4}), encoding="utf-8")
    templates = checkpoint / "chat_templates"
    templates.mkdir()
    (templates / "default.jinja").write_text("{{ messages }}", encoding="utf-8")
    audio_processor = checkpoint / "audio_processor"
    audio_processor.mkdir()
    (audio_processor / "mel_filters.npz").write_bytes(b"processor-data")

    stage = tmp_path / "artifact"
    hf_config = stage / "hf_config"
    hf_config.mkdir(parents=True)
    exported_config = copy.deepcopy(checkpoint_config)
    exported_config["quantization_config"]["desc_act"] = False
    (hf_config / "config.json").write_text(json.dumps(exported_config), encoding="utf-8")
    (hf_config / "tokenizer_config.json").write_text(
        json.dumps({"asset_source": "raw-model"}),
        encoding="utf-8",
    )

    llm.package_runtime(
        stage,
        SimpleNamespace(model_id="qwen", quant_checkpoint=str(checkpoint)),
    )

    packaged_config = json.loads((hf_config / "config.json").read_text(encoding="utf-8"))
    assert packaged_config["quantization_config"] == {
        "bits": 4,
        "quant_method": "gptq",
        "desc_act": False,
    }
    assert (hf_config / "tokenizer.model").read_bytes() == b"checkpoint-tokenizer"
    assert json.loads((hf_config / "tokenizer_config.json").read_text(encoding="utf-8")) == {
        "asset_source": "checkpoint"
    }
    assert (hf_config / "processor_config.json").is_file()
    assert (hf_config / "quantize_config.json").is_file()
    assert (hf_config / "chat_templates/default.jinja").is_file()
    assert (hf_config / "audio_processor/mel_filters.npz").read_bytes() == b"processor-data"


@pytest.mark.parametrize(
    ("mode", "graph_aliases"),
    [
        (
            "mtp",
            (
                ("prefill_hmonnx", "draft_prefill_onnx"),
                ("decode_hmonnx", "draft_decode_onnx"),
            ),
        ),
        (
            "dflash",
            (
                ("context_hmonnx", "draft_context_onnx"),
                ("decode_hmonnx", "draft_decode_onnx"),
            ),
        ),
    ],
)
def test_llm_adapter_requires_consistent_runtime_and_contract_v2_draft_graphs(
    tmp_path: Path,
    mode: str,
    graph_aliases: tuple[tuple[str, str], ...],
) -> None:
    config_overrides = {"export.model.num_draft_tokens": 4}
    if mode == "dflash":
        config_overrides["export.model.dflash_config.noise_token_id"] = 248070
    plan = SimpleNamespace(
        model_id="qwen",
        components=("prefill", "decode", mode),
        config_overrides=config_overrides,
    )

    with pytest.raises(ModelError, match="draft runtime contract"):
        llm._validate_speculative_assets(tmp_path, {}, plan)

    draft = {}
    top_level = {}
    for nested_field, top_field in graph_aliases:
        path = tmp_path / f"{nested_field}.hmonnx"
        path.write_bytes(b"graph")
        draft[nested_field] = path.name
        top_level[top_field] = path.name
    if mode == "dflash":
        draft["noise_token_id"] = 248070
    metadata = {
        "spec_decode": {
            "runtime_contract_version": 2,
            "mode": mode,
            "num_draft_tokens": 4,
            "verify_length": 5,
            "draft": draft,
            **top_level,
        }
    }
    llm._validate_speculative_assets(tmp_path, metadata, plan)

    alternate = tmp_path / "alternate.hmonnx"
    alternate.write_bytes(b"graph")
    inconsistent = copy.deepcopy(metadata)
    inconsistent["spec_decode"][graph_aliases[0][1]] = alternate.name
    with pytest.raises(ModelError, match="incomplete draft graph set"):
        llm._validate_speculative_assets(tmp_path, inconsistent, plan)

    mismatched_plan = SimpleNamespace(
        model_id="qwen",
        components=("prefill", "decode", mode),
        config_overrides={**config_overrides, "export.model.num_draft_tokens": 3},
    )
    with pytest.raises(ModelError, match="incomplete draft graph set"):
        llm._validate_speculative_assets(tmp_path, metadata, mismatched_plan)

    if mode == "dflash":
        missing_noise_token = copy.deepcopy(metadata)
        del missing_noise_token["spec_decode"]["draft"]["noise_token_id"]
        with pytest.raises(ModelError, match="incomplete draft graph set"):
            llm._validate_speculative_assets(tmp_path, missing_noise_token, plan)

        wrong_noise_token = copy.deepcopy(metadata)
        wrong_noise_token["spec_decode"]["draft"]["noise_token_id"] += 1
        with pytest.raises(ModelError, match="incomplete draft graph set") as exc_info:
            llm._validate_speculative_assets(tmp_path, wrong_noise_token, plan)
        assert any(
            item.get("expected") == 248070
            for item in exc_info.value.details["invalid_runtime_assets"]
            if item["field"] == "draft.noise_token_id"
        )

        context_decode = tmp_path / "context_decode_hmonnx.hmonnx"
        context_decode.write_bytes(b"graph")
        with_context_decode = copy.deepcopy(metadata)
        with_context_decode["spec_decode"]["draft"]["context_decode_hmonnx"] = context_decode.name
        with_context_decode["spec_decode"]["draft_context_decode_onnx"] = context_decode.name
        llm._validate_speculative_assets(tmp_path, with_context_decode, plan)

    (tmp_path / f"{graph_aliases[-1][0]}.hmonnx").unlink()
    with pytest.raises(ModelError, match="incomplete draft graph set"):
        llm._validate_speculative_assets(tmp_path, metadata, plan)
