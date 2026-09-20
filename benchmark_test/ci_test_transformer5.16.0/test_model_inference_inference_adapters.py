import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from xhmodel_merak.errors import ErrorCode, ModelError
from xhmodel_merak.model_catalog import get_recipe
from xhmodel_merak.model_catalog.loading import load_adapter
from xhmodel_merak.model_inference.inputs import validate_inputs
from xhmodel_merak.model_integrations.autoregressive import inference as _autoregressive
from xhmodel_merak.model_integrations.gemma4 import inference as gemma4
from xhmodel_merak.model_integrations.ling import inference as ling
from xhmodel_merak.model_integrations.paddleocr_vl import inference as paddleocr_vl
from xhmodel_merak.model_integrations.pi05 import inference as pi05
from xhmodel_merak.xh_other_model.models.pi05 import compact_runtime


@pytest.mark.parametrize(
    ("name", "canonical"),
    (
        ("qwen.inference", "xhmodel_merak.model_integrations.qwen.inference"),
        ("gemma4.inference", "xhmodel_merak.model_integrations.gemma4.inference"),
        ("ling.inference", "xhmodel_merak.model_integrations.ling.inference"),
        ("paddleocr-vl.inference", "xhmodel_merak.model_integrations.paddleocr_vl.inference"),
        ("pi05.inference", "xhmodel_merak.model_integrations.pi05.inference"),
        ("kokoro.inference", "xhmodel_merak.model_integrations.kokoro.inference"),
        (
            "moss_transcribe_diarize.inference",
            "xhmodel_merak.model_integrations.moss_transcribe_diarize.inference",
        ),
    ),
)
def test_registered_inference_adapter_loads_family_implementation(name: str, canonical: str) -> None:
    assert load_adapter("inference", name) is importlib.import_module(canonical).adapter


def test_moss_adapter_rejects_hmonnx_path_outside_artifact(tmp_path: Path) -> None:
    from xhmodel_merak.model_integrations.moss_transcribe_diarize import inference as moss

    geometry = {
        "prefill_length": 512,
        "cache_length": 4096,
        "rope_max_positions": 4096,
        "rope_mode": "fp32_table_integer_index",
        "onnx_file": "graph.onnx",
        "hmonnx_file": "graph.onnx",
    }
    for name in (
        "graph.onnx",
        "weights.pt",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "tokenizer_config.json",
        "preprocessor_config.json",
        "configuration_moss_transcribe_diarize.py",
        "modeling_moss_transcribe_diarize.py",
        "processing_moss_transcribe_diarize.py",
    ):
        (tmp_path / name).write_text("fixture", encoding="utf-8")
    meta = tmp_path / "meta.json"
    meta.write_text(
        json.dumps(
            {
                "quant_type": "w8a8_sefp",
                "weights_file": "weights.pt",
                "lm_head_file": "weights.pt",
                "components": {
                    "encode": {"onnx_file": "graph.onnx", "hmonnx_file": "../outside.onnx"},
                    "prefill_decode": {"prefill": geometry, "decode": geometry},
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ModelError, match="inside the Artifact root"):
        moss.adapter.validate_artifact(
            recipe=get_recipe("moss-transcribe-diarize-0.9b"),
            root=tmp_path,
            manifest={},
            meta_file=meta,
        )


def test_autoregressive_runtime_has_one_canonical_implementation() -> None:
    assert _autoregressive.run_text_generation.__module__ == (
        "xhmodel_merak.model_integrations.autoregressive.inference"
    )


def _adapter_kwargs(tmp_path: Path, model: str, inputs: dict):
    meta = tmp_path / "meta.json"
    meta.write_text("{}", encoding="utf-8")
    recipe = get_recipe(model)
    return {
        "recipe": recipe,
        "root": tmp_path,
        "manifest": {
            "model_id": recipe.model_id,
            "artifact": {
                "files": [
                    {"path": path.relative_to(tmp_path).as_posix()} for path in tmp_path.rglob("*") if path.is_file()
                ]
            },
        },
        "meta_file": meta,
        "inputs": inputs,
        "device": "cpu",
        "components": recipe.components,
        "selected_golden": (),
        "golden_artifacts": {},
    }


def test_ling_adapter_returns_text_result(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def run_text_generation(*args, **kwargs):
        captured.update(kwargs)
        return "answer", 2, 0.5, 4.0, {"prompt_tokens": 3}

    monkeypatch.setattr(ling, "run_text_generation", run_text_generation)

    result = ling.adapter.run(**_adapter_kwargs(tmp_path, "ling-3.0-flash", {"prompt": "question"}))

    assert result.output_type == "text"
    assert result.output_text == "answer"
    assert result.output_tokens == 2
    assert captured.get("cuda_graph", False) is False


def test_paddleocr_vl_adapter_requires_image_and_returns_text(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "document.png"
    image.touch()
    captured = {}

    def run_paddleocr_generation(*args, **kwargs):
        captured.update(kwargs)
        return "recognized", 3, 0.5, 6.0, {"prompt_tokens": 4}

    monkeypatch.setattr(paddleocr_vl, "run_paddleocr_generation", run_paddleocr_generation)
    kwargs = _adapter_kwargs(
        tmp_path,
        "paddleocr-vl-1.6",
        {"prompt": "OCR:", "image": str(image)},
    )

    result = paddleocr_vl.adapter.run(**kwargs)

    assert result.output_type == "text"
    assert result.output_text == "recognized"
    assert result.output_tokens == 3
    assert captured["image_path"] == image.resolve()
    assert captured["max_new_tokens"] == 1024
    assert captured["enable_golden"] is False

    with pytest.raises(ModelError) as caught:
        paddleocr_vl.adapter.run(
            **_adapter_kwargs(tmp_path, "paddleocr-vl-1.6", {"prompt": "OCR:"})
        )
    assert caught.value.code == ErrorCode.INVALID_PARAMETER


def test_autoregressive_runtime_disables_token_type_ids(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    class FakeInputs(dict):
        def __init__(self):
            self.input_ids = torch.tensor([[1, 2]])
            super().__init__(input_ids=self.input_ids)

        def to(self, device):
            assert device == torch.device("cpu")
            return self

    class FakeTokenizer:
        pad_token_id = 0
        eos_token_id = 1

        def apply_chat_template(self, messages, **kwargs):
            return "prompt"

        def __call__(self, texts, **kwargs):
            captured["tokenizer_kwargs"] = kwargs
            return FakeInputs()

        def decode(self, values, **kwargs):
            return "answer"

    class FakeModel:
        def get_tokenizer(self, **kwargs):
            return FakeTokenizer()

        def to(self, device):
            assert device == torch.device("cpu")

        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3]])

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            return FakeModel()

    class FakeContext:
        def __init__(self, model, devices=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr("xhmodel_merak.xh_llm.AutoLLMHONNXModel", FakeAutoModel)
    monkeypatch.setattr("xhmodel_merak.xh_llm.LLMInferenceContextManager", FakeContext)

    result = _autoregressive.run_text_generation(
        tmp_path / "meta.json",
        prompt="question",
        max_new_tokens=1,
        device="cpu",
        device_map=False,
        pack_w4=False,
    )

    assert result[0] == "answer"
    assert captured["tokenizer_kwargs"] == {
        "return_tensors": "pt",
        "return_token_type_ids": False,
    }


def test_gemma_adapter_rejects_multiple_media_inputs_before_runtime(tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    audio = tmp_path / "audio.wav"
    image.touch()
    audio.touch()

    with pytest.raises(ModelError) as exc_info:
        gemma4.adapter.run(
            **_adapter_kwargs(
                tmp_path,
                "gemma4-e2b",
                {"prompt": "question", "image": str(image), "audio": str(audio)},
            )
        )

    assert exc_info.value.code == ErrorCode.INVALID_PARAMETER


def test_gemma_video_frame_directory_has_deterministic_processor_metadata(tmp_path: Path) -> None:
    from PIL import Image

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    Image.new("RGB", (12, 8), color="red").save(frames_dir / "02.png")
    Image.new("RGB", (12, 8), color="blue").save(frames_dir / "01.png")

    frames, metadata = gemma4._load_video(frames_dir)

    assert [frame.getpixel((0, 0)) for frame in frames] == [(0, 0, 255), (255, 0, 0)]
    assert metadata == {
        "fps": 2.0,
        "width": 12,
        "height": 8,
        "duration": 1.0,
        "total_num_frames": 2,
        "frames_indices": [0, 1],
        "video_backend": "image_directory",
    }


def test_gemma_empty_frame_directory_is_rejected_before_model_loading(monkeypatch, tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.AutoLLMHONNXModel.from_pretrained",
        lambda *_args, **_kwargs: pytest.fail("model runtime must not load for an invalid frame directory"),
    )

    with pytest.raises(ValueError, match="video frame directory contains no supported images"):
        gemma4._run_multimodal(
            tmp_path / "meta.json",
            recipe=get_recipe("gemma4-e2b"),
            prompt="question",
            modality="video",
            media_path=frames_dir,
            max_new_tokens=1,
            device="cpu",
        )


def test_gemma_multimodal_runtime_scopes_cache_to_requested_device(monkeypatch, tmp_path: Path) -> None:
    from PIL import Image

    image = tmp_path / "image.png"
    Image.new("RGB", (16, 16)).save(image)
    captured = {}

    class FakeInputs(dict):
        def __init__(self):
            self.input_ids = torch.tensor([[1, 2]])
            super().__init__(input_ids=self.input_ids)

        def to(self, device):
            assert device == "cpu"
            return self

    class FakeProcessor:
        tokenizer = SimpleNamespace(
            pad_token_id=0,
            eos_token_id=1,
            decode=lambda values, **kwargs: "ok",
        )

        def apply_chat_template(self, messages):
            return FakeInputs()

    class FakeModel:
        def get_tf_processor(self):
            return FakeProcessor()

        def to(self, device):
            assert device == "cpu"

        def generate(self, **kwargs):
            return torch.tensor([[1, 2, 3]])

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(path, **kwargs):
            captured["loader_kwargs"] = kwargs
            return FakeModel()

    class FakeContext:
        def __init__(self, model, devices=None):
            captured["devices"] = devices

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(gemma4, "load_json", lambda path: {"model_config": {"context_max_length": 128}})
    monkeypatch.setattr("xhmodel_merak.xh_llm.AutoLLMHONNXModel", FakeAutoModel)
    monkeypatch.setattr("xhmodel_merak.xh_llm.LLMInferenceContextManager", FakeContext)

    result = gemma4._run_multimodal(
        tmp_path / "meta.json",
        recipe=get_recipe("gemma4-e2b"),
        prompt="question",
        modality="image",
        media_path=image,
        max_new_tokens=1,
        device="cpu",
    )

    assert captured["devices"] == ["cpu"]
    assert captured["loader_kwargs"] == {}
    assert result[0] == "ok"


def test_pi05_adapter_returns_horizon_50_actions(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "image.png"
    image2 = tmp_path / "image2.png"
    image.touch()
    image2.touch()

    class FakeRuntime:
        def __init__(self, root, *, device, manifest):
            assert Path(root) == tmp_path
            assert device == "cpu"
            assert manifest["model_id"] == "pi0.5-libero"

        def predict(self, **kwargs):
            assert kwargs["state"] == [0.0] * 8
            return np.zeros((50, 7), dtype=np.float32), {"valid_prefix_length": 540}

    monkeypatch.setattr(
        "xhmodel_merak.xh_other_model.models.pi05.compact_runtime.PI05CompactRuntime",
        FakeRuntime,
    )
    result = pi05.adapter.run(
        **_adapter_kwargs(
            tmp_path,
            "pi0.5-libero",
            {
                "task": "pick up the cup",
                "observation.images.image": str(image),
                "observation.images.image2": str(image2),
                "observation.state": [0.0] * 8,
                "seed": 9,
            },
        )
    )

    assert result.output_type == "actions"
    assert result.horizon == 50
    assert result.action_dim == 7
    assert result.seed == 9


def test_pi05_quantile_stats_apply_forward_and_inverse(tmp_path: Path) -> None:
    stats_path = tmp_path / "stats.safetensors"
    save_file(
        {
            "observation.state.q01": torch.tensor([0.0, 10.0]),
            "observation.state.q99": torch.tensor([2.0, 14.0]),
        },
        stats_path,
    )
    stats = compact_runtime._load_stats(stats_path, "observation.state", "quantiles")

    normalized = compact_runtime._apply_normalization(torch.tensor([1.0, 12.0]), stats, inverse=False)
    restored = compact_runtime._apply_normalization(normalized, stats, inverse=True)

    torch.testing.assert_close(normalized, torch.zeros(2))
    torch.testing.assert_close(restored, torch.tensor([1.0, 12.0]))


def test_pi05_compact_runtime_passes_key_and_value_cache_per_layer(monkeypatch) -> None:
    calls = {}

    class Session:
        def __init__(self, name):
            self.name = name

        def __call__(self, *args):
            calls.setdefault(self.name, []).append(args)
            if self.name == "vision":
                return torch.ones(1, 2, 4)
            if self.name in {"action_in", "expert"}:
                return torch.ones(1, 2, 4)
            if self.name == "time_mlp":
                return torch.ones(1, 4)
            if self.name == "action_out":
                return torch.zeros(1, 2, 3)
            return None

    runtime = compact_runtime.PI05CompactRuntime.__new__(compact_runtime.PI05CompactRuntime)
    runtime.device = torch.device("cpu")
    runtime.selected_image_indices = (0, 1)
    runtime.image_resolution = (16, 16)
    runtime.horizon = 2
    runtime.cache_length = 8
    runtime.max_action_dim = 3
    runtime.action_dim = 2
    runtime.num_inference_steps = 1
    runtime.contract = {"prefix_sequence_length": 5}
    runtime.token_embedding = torch.nn.Embedding(4, 4)
    runtime.embedding_scale = 1.0
    runtime.cache_shape = (1, 1, 8, 1)
    runtime.num_layers = 2
    runtime.CacheTensor = lambda value: value
    runtime.action_stats = {
        "mode": "quantiles",
        "low": torch.zeros(2),
        "high": torch.full((2,), 2.0),
    }
    runtime.sessions = {
        name: Session(name) for name in ("vision", "gemma", "action_in", "time_mlp", "expert", "action_out")
    }
    runtime._tokenize = lambda task, state: (
        torch.zeros((1, 1), dtype=torch.long),
        torch.ones((1, 1), dtype=torch.bool),
        torch.zeros(8),
    )
    runtime._graph_input_shape = lambda name: [1, 4]
    monkeypatch.setattr(compact_runtime, "_load_image", lambda *args: torch.zeros(1, 3, 16, 16))

    actions, _ = runtime.predict(task="pick", image="a", image2="b", state=[0.0] * 8, seed=7)

    assert actions.shape == (2, 2)
    assert len(calls["gemma"][0]) == 4 + 2 * runtime.num_layers
    assert len(calls["expert"][0]) == 5 + 2 * runtime.num_layers
    for prefix_cache, expert_cache in zip(calls["gemma"][0][4:], calls["expert"][0][5:], strict=True):
        assert prefix_cache is expert_cache


def test_runtime_input_validation_rejects_non_finite_numbers() -> None:
    with pytest.raises(ModelError) as exc_info:
        validate_inputs(get_recipe("kokoro-82m-v1.1-zh"), {"text": "hello", "speed": float("nan")})
    assert exc_info.value.code == ErrorCode.INVALID_PARAMETER

    with pytest.raises(ModelError) as exc_info:
        validate_inputs(
            get_recipe("pi0.5-libero"),
            {
                "task": "pick",
                "observation.images.image": "a",
                "observation.images.image2": "b",
                "observation.state": [0.0] * 7 + [float("inf")],
            },
        )
    assert exc_info.value.code == ErrorCode.INVALID_PARAMETER
