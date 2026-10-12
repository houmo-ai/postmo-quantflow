from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_ROOT = REPO_ROOT / "vllm_merak"
PACKAGE_ROOT = PLUGIN_ROOT / "vllm_merak"
requires_vllm = pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="vLLM behavior checks run in the independently installed plugin environment",
)


def _plugin_metadata() -> dict:
    return tomllib.loads((PLUGIN_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_vllm_merak_remains_an_independently_installable_vllm_plugin():
    metadata = _plugin_metadata()

    assert metadata["project"]["name"] == "vllm-merak"
    assert metadata["project"]["dependencies"] == []
    assert metadata["project"]["entry-points"] == {
        "vllm.general_plugins": {
            "merak": "vllm_merak:register_plugins",
            "wemm": "vllm_merak.native_wemm:register",
        },
        "vllm.platform_plugins": {"merak": "vllm_merak.platform:register"},
    }
    assert metadata["tool"]["setuptools"]["packages"]["find"]["include"] == ["vllm_merak*"]


def test_vllm_merak_entry_points_resolve_to_declared_source_symbols():
    init_tree = ast.parse((PACKAGE_ROOT / "__init__.py").read_text(encoding="utf-8"))
    native_wemm_tree = ast.parse((PACKAGE_ROOT / "native_wemm.py").read_text(encoding="utf-8"))
    platform_tree = ast.parse((PACKAGE_ROOT / "platform.py").read_text(encoding="utf-8"))

    init_functions = {node.name for node in init_tree.body if isinstance(node, ast.FunctionDef)}
    native_wemm_functions = {node.name for node in native_wemm_tree.body if isinstance(node, ast.FunctionDef)}
    platform_functions = {node.name for node in platform_tree.body if isinstance(node, ast.FunctionDef)}

    assert "register_plugins" in init_functions
    assert "register" in native_wemm_functions
    assert "register" in platform_functions


@requires_vllm
def test_native_wemm_registration_is_lazy_and_only_overrides_qwen_when_selected(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm import ModelRegistry

    from vllm_merak import native_wemm

    registrations: list[tuple[str, object]] = []
    monkeypatch.setattr(ModelRegistry, "register_model", lambda arch, cls: registrations.append((arch, cls)))
    monkeypatch.setattr(ModelRegistry, "get_supported_archs", lambda: set())

    monkeypatch.delenv("VLLM_PLUGINS", raising=False)
    native_wemm.register()
    assert registrations == [
        (
            "WeMMEmbedding",
            "vllm_merak.model_executor.model_loader.wemm_embedding_native:WeMMEmbedding",
        )
    ]

    registrations.clear()
    monkeypatch.setenv("VLLM_PLUGINS", "wemm")
    native_wemm.register()
    assert registrations == [
        (
            "WeMMEmbedding",
            "vllm_merak.model_executor.model_loader.wemm_embedding_native:WeMMEmbedding",
        ),
        (
            "Qwen3_5ForConditionalGeneration",
            "vllm_merak.model_executor.model_loader.wemm_embedding_native:WeMMEmbedding",
        ),
    ]


@requires_vllm
def test_merak_loader_discovery_excludes_native_vllm_models():
    from vllm_merak.model_executor.register import _should_import_module_file

    assert _should_import_module_file(Path("qwen3_5.py")) is True
    assert _should_import_module_file(Path("wemm_embedding.py")) is True
    assert _should_import_module_file(Path("wemm_embedding_native.py")) is False


@requires_vllm
def test_merak_registration_and_selected_loader_do_not_import_other_models():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import importlib.abc, sys",
                    "class BlockUnusedModels(importlib.abc.MetaPathFinder):",
                    "    def find_spec(self, fullname, path=None, target=None):",
                    "        if fullname in ('vllm_merak.model_executor.model_loader.glm53', 'vllm_merak.model_executor.model_loader.gemma4'):",
                    "            raise AssertionError('Imported an unselected model: ' + fullname)",
                    "sys.meta_path.insert(0, BlockUnusedModels())",
                    "from vllm_merak import register_plugins",
                    "register_plugins()",
                    "from vllm import ModelRegistry",
                    "assert 'Glm53MerakForCausalLM' in ModelRegistry.get_supported_archs()",
                    "assert 'Gemma4UnifiedMerakForCausalLM' in ModelRegistry.get_supported_archs()",
                    "from vllm_merak.model_executor.register import get_registered_model_cls, get_model_registration_paths",
                    "paths = get_model_registration_paths()",
                    "assert paths['DeepseekV4MerakForCausalLM'].endswith('deepseek_v4:DeepseekV4MerakForCausalLM')",
                    "cls = get_registered_model_cls('DeepseekV4ForCausalLM')",
                    "assert cls.__name__ == 'DeepseekV4MerakForCausalLM'",
                    "assert get_registered_model_cls(cls.__name__) is cls",
                    "assert get_registered_model_cls('UnknownArchitecture') is None",
                    "assert 'vllm_merak.model_executor.model_loader.glm53' not in sys.modules",
                    "assert 'vllm_merak.model_executor.model_loader.gemma4' not in sys.modules",
                ]
            ),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


@requires_vllm
def test_merak_loader_discovery_preserves_runtime_class_overrides(monkeypatch):
    from vllm_merak.model_executor.register import (
        MERAK_MODEL_REGISTRY,
        get_model_registration_paths,
        get_registered_model_cls,
    )

    replacement = type("DeepseekV4MerakForCausalLM", (torch.nn.Module,), {})
    monkeypatch.setitem(
        MERAK_MODEL_REGISTRY.module_dict, "DeepseekV4ForCausalLM", replacement
    )

    assert get_registered_model_cls("DeepseekV4ForCausalLM") is replacement
    assert get_registered_model_cls("DeepseekV4MerakForCausalLM") is replacement
    assert get_model_registration_paths()["DeepseekV4MerakForCausalLM"] is replacement


class _WeMMTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if text == "<embedding>":
            return [99]
        if text == "already-anchored":
            return [1, 2, 99]
        return [1, 2, 3]


class _WeMMProcessingInfo:
    def get_tokenizer(self) -> _WeMMTokenizer:
        return _WeMMTokenizer()

    def get_hf_config(self):
        return type(
            "Config",
            (),
            {"vision_start_token_id": 10, "vision_end_token_id": 11},
        )()


@requires_vllm
def test_native_wemm_prompt_keeps_exactly_one_final_pooling_anchor():
    from vllm_merak.model_executor.model_loader.wemm_embedding_native import (
        ensure_wemm_embedding_token,
        prepare_wemm_prompt_token_ids,
    )

    tokenizer = _WeMMTokenizer()
    original = [99, 1, 99, 2]

    assert ensure_wemm_embedding_token(tokenizer, original) == [1, 2, 99]
    assert original == [99, 1, 99, 2]
    assert prepare_wemm_prompt_token_ids(tokenizer, "plain") == [1, 2, 3, 99]
    assert prepare_wemm_prompt_token_ids(tokenizer, "already-anchored") == [1, 2, 99]
    assert prepare_wemm_prompt_token_ids(tokenizer, "plain", max_input_tokens=3) == [
        1,
        2,
        99,
    ]
    with pytest.raises(ValueError, match="max_input_tokens must be positive"):
        prepare_wemm_prompt_token_ids(tokenizer, "plain", max_input_tokens=0)


@requires_vllm
def test_native_wemm_processor_preserves_video_vision_boundaries(
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.model_executor.models.qwen3_vl import Qwen3VLMultiModalProcessor
    from vllm.multimodal.processing import PromptReplacement, PromptUpdateDetails
    from vllm_merak.model_executor.model_loader.wemm_embedding_native import (
        WeMMMultiModalProcessor,
    )

    base_update = PromptReplacement(
        modality="video",
        target="wrapped-video-placeholder",
        replacement=lambda _: PromptUpdateDetails.select_token_id(
            [20, 99, 21],
            99,
        ),
    )
    monkeypatch.setattr(
        Qwen3VLMultiModalProcessor,
        "_get_prompt_updates",
        lambda *_args, **_kwargs: [base_update],
    )
    processor = object.__new__(WeMMMultiModalProcessor)
    processor.info = _WeMMProcessingInfo()

    updates = processor._get_prompt_updates(None, {}, None)
    resolved = updates[0].resolve(0)

    assert resolved.content.full == [10, 20, 99, 21, 11]
    assert resolved.content.is_embed is not None
    assert resolved.content.is_embed(None, resolved.content.full).tolist() == [
        False,
        False,
        True,
        False,
        False,
    ]


@requires_vllm
def test_native_wemm_video_grid_expands_into_official_frame_bundles():
    from vllm_merak.model_executor.model_loader.wemm_embedding_native import (
        _expand_video_grid_thw,
    )

    grid_thw = torch.tensor([[2, 4, 6], [1, 8, 10]])
    expanded = _expand_video_grid_thw(grid_thw)

    assert expanded.tolist() == [[1, 4, 6], [1, 4, 6], [1, 8, 10]]
    assert grid_thw.tolist() == [[2, 4, 6], [1, 8, 10]]

    with pytest.raises(ValueError, match="shape"):
        _expand_video_grid_thw(torch.tensor([[1, 2]]))
    with pytest.raises(ValueError, match="positive"):
        _expand_video_grid_thw(torch.tensor([[0, 2, 2]]))


def test_native_wemm_evaluator_selects_only_one_plugin_backend():
    adapter = (REPO_ROOT / "examples_merak/llm/wemm_embedding/mmeb_v3_vllm_merak.py").read_text(encoding="utf-8")

    assert 'os.environ["VLLM_PLUGINS"] = "merak" if merak else "wemm"' in adapter
    assert 'kwargs["hf_overrides"]' not in adapter
    assert "vllm_merak.model_executor.model_loader.wemm_embedding_native" in adapter


def test_native_wemm_provenance_hashes_the_plugin_model_source():
    source_dir = REPO_ROOT / "examples_merak/llm/wemm_embedding"
    for name in ("mmeb_v3_eval.py", "run_mmeb_v3_vllm_merak_full.py"):
        source = (source_dir / name).read_text(encoding="utf-8")
        assert "vllm_merak.model_executor.model_loader" in source
        assert "wemm_embedding_native" in source
        assert "from vllm.model_executor.models import wemm_embedding" not in source


def test_vllm_merak_import_from_modelzoo_root_resolves_inner_package():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import vllm_merak; print(Path(vllm_merak.__file__).resolve())",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert Path(completed.stdout.strip()) == PACKAGE_ROOT / "__init__.py"


def test_vllm_merak_snapshot_excludes_repository_and_build_state():
    forbidden_names = {
        ".git",
        "build",
        "dist",
    }

    offenders = sorted(
        path.relative_to(PLUGIN_ROOT).as_posix()
        for path in PLUGIN_ROOT.rglob("*")
        if path.name in forbidden_names or path.name.endswith(".egg-info")
    )
    assert offenders == []

    gitignore = (PLUGIN_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert {"__pycache__/", ".pytest_cache/", ".ruff_cache"} <= set(gitignore)
    assert {"!/data/", "!/data/images/", "!/data/images/qwen2_vl_demo.jpeg"} <= set(gitignore)
    assert (PLUGIN_ROOT / "data/images/qwen2_vl_demo.jpeg").is_file()


def test_vllm_merak_runtime_delivery_files_define_modelzoo_as_source_of_truth():
    runtime_profiles = tomllib.loads((PLUGIN_ROOT / "runtime_profiles.toml").read_text(encoding="utf-8"))
    local_profile = runtime_profiles["profiles"]["vllm_023_local"]

    assert local_profile["python"] == "3.12"
    assert local_profile["packages"]["vllm-merak"] == "0.20.0"
    assert local_profile["package_specifiers"]["vllm"] == ">=0.23,<0.24"
    assert local_profile["packages"]["torch"] == "2.11.0+cu129"
    assert local_profile["packages"]["transformers"] == "5.13.0"
    assert local_profile["runtime"] == {
        "target": "cuda",
        "cuda_required": True,
        "required_modules": ["vllm._C_stable_libtorch"],
        "required_torch_ops": ["_C.apply_repetition_penalties_"],
    }

    checker = PLUGIN_ROOT / "tools" / "check_runtime_environment.py"
    installer = PLUGIN_ROOT / "tools" / "install_from_modelzoo.sh"
    assert checker.is_file()
    assert installer.is_file()

    checker_tree = ast.parse(checker.read_text(encoding="utf-8"))
    checker_functions = {node.name for node in checker_tree.body if isinstance(node, ast.FunctionDef)}
    assert {"load_profile", "check_environment", "_pip_check", "main"} <= checker_functions
    checker_text = checker.read_text(encoding="utf-8")
    assert '("vllm.general_plugins", "wemm")' in checker_text
    assert '"vllm_merak.native_wemm:register"' in checker_text

    installer_text = installer.read_text(encoding="utf-8")
    assert 'pip install -v -e "$PLUGIN_ROOT"' in installer_text
    assert "--no-deps" in installer_text
    assert "check_runtime_environment.py" in installer_text


def test_modelzoo_service_requirements_keep_worker_transformers_contract():
    metadata = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["dependencies"] == ["transformers (>=4.57.0,<5.17.0)"]
    export_dependencies = metadata["project"]["optional-dependencies"]["export"]
    assert any(requirement.startswith("torch ") for requirement in export_dependencies)
    assert any(requirement.startswith("onnx ") for requirement in export_dependencies)
    assert any(requirement.startswith("transformers ") for requirement in export_dependencies)


def test_cuda_vllm_build_helper_requires_source_and_sampler_abi():
    helper = (PLUGIN_ROOT / "tools" / "install_vllm_cuda_xhquant.sh").read_text(encoding="utf-8")

    assert 'VLLM_SOURCE="${VLLM_SOURCE:?' in helper
    assert 'cd "$VLLM_SOURCE"' in helper
    assert "VLLM_TARGET_DEVICE=cuda" in helper
    assert 'SOURCE_COMMIT="$(git -C "$VLLM_SOURCE" rev-parse HEAD)"' in helper
    assert '"VLLM_USE_PRECOMPILED=1"' in helper
    assert '"VLLM_PRECOMPILED_WHEEL_COMMIT=' in helper
    assert '"VLLM_PRECOMPILED_WHEEL_VARIANT=$CUDA_VARIANT"' in helper
    assert 'conda run -n "$CONDA_ENV" python -c' in helper
    assert 'importlib.import_module("vllm._C_stable_libtorch")' in helper
    assert 'hasattr(torch.ops._C, "apply_repetition_penalties_")' in helper
    assert "torch.ops._C.apply_repetition_penalties_(" in helper
    assert 'print("CUDA-target vLLM sampler ABI: PASS")' in helper


def test_offline_smoke_forwards_context_and_concurrency_limits():
    example = (PLUGIN_ROOT / "examples" / "vllm_merak_generate.py").read_text(encoding="utf-8")

    assert "max_model_len=args.max_model_len" in example
    assert "max_num_seqs=args.max_num_seqs" in example
    assert "parser.set_defaults(max_num_seqs=1)" in example


def test_native_wemm_smoke_uses_plugin_without_hf_overrides():
    example = (PLUGIN_ROOT / "examples" / "vllm_wemm_embedding.py").read_text(encoding="utf-8")

    assert 'os.environ["VLLM_PLUGINS"] = "wemm"' in example
    assert '"runner": "pooling"' in example
    assert "prepare_wemm_prompt_token_ids" in example
    assert "hf_overrides" not in example


def test_qwen35_release_smoke_profile_is_small_and_uses_flash_attention():
    profile_path = PLUGIN_ROOT / "test_profiles" / "qwen3_5_9b_flash_attention_smoke.yaml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))

    assert profile["quant"]["group_size"] == 64
    assert profile["quant"]["calibration"] == {
        "dataset": "NeelNanda/pile-10k",
        "nsamples": 2,
        "seqlen": 128,
    }
    assert profile["quant"]["iters"] == 2
    model = profile["export"]["model"]
    assert model["flash_attention"]["enable"] is True
    assert model["prefill_chunk_length"] == 256
    assert model["visual_config"]["enable"] is True
    assert model["visual_config"]["image_token_gears"] == [96, 196, 384, 704, 1536]
    assert model["visual_config"]["image_token_capacity"] == 1536
    assert model["only_first_block"] is False


def test_release_smoke_covers_generation_and_penalty_matrix():
    smoke = (PLUGIN_ROOT / "tools" / "run_hmonnx_release_smoke.py").read_text(encoding="utf-8")

    assert 'config_format="merak_llm"' in smoke
    assert 'load_format="merak_llm"' in smoke
    assert '"greedy"' in smoke
    assert '"temperature_topk_topp"' in smoke
    assert '"penalties"' in smoke
    assert '"image_greedy"' in smoke
    assert "llm.chat" in smoke
    assert "repetition_penalty=1.1" in smoke
    assert "presence_penalty=0.1" in smoke
    assert "frequency_penalty=0.1" in smoke

    acceptance = (PLUGIN_ROOT / "tools" / "run_release_acceptance.sh").read_text(encoding="utf-8")
    assert "check_runtime_environment.py" in acceptance
    assert "check_hmonnx_attention_abi.py" in acceptance
    assert "run_hmonnx_release_smoke.py" in acceptance
