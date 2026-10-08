"""GLM uses its own cache geometry with the shared single-request lifecycle."""

import importlib.util
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("vllm") is None,
    reason="vLLM adapter checks require the independently installed plugin environment",
)


@pytest.fixture
def glm53():
    from vllm_merak.model_executor.model_loader import glm53 as module

    return module


class _Runtime:
    device = torch.device("cpu")

    def __init__(self):
        self.embed_tokens = nn.Embedding(16, 4, dtype=torch.float16)
        self.resets = 0
        self.calls = []
        self.to_calls = []
        self.load_kwargs = None
        self.cache_mixin = SimpleNamespace(clear_kv_cache=self.clear_kv_cache)

    def to(self, device):
        self.to_calls.append(torch.device(device))
        return self

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_kvcache_mixin(self):
        return self.cache_mixin

    def clear_kv_cache(self):
        self.resets += 1

    def get_data_preprocessor(self):
        def process(data):
            self.calls.append((data["input_ids"].numel(), data["past_seq_length"]))
            return (data["input_ids"],)

        return process

    def set_prefill(self):
        pass

    def set_decode(self):
        pass

    def forward(self, *args):
        return torch.zeros(1, 16)


class _GlmRuntime(_Runtime):
    def __init__(self, chunk_length):
        from xhmodel_merak.xh_llm.models.glm5_3_flash.contract import DSA, LINEAR, Glm53CacheSpec

        super().__init__()
        self.meta_info = SimpleNamespace(model_config=SimpleNamespace(prefill_chunk_length=chunk_length))
        self.cache_spec = Glm53CacheSpec((LINEAR, DSA), 4096, 2, 8, 4, 16, 8, index_topk=16)


class _DeepseekRuntime(_Runtime):
    def __init__(self, chunk_length):
        from xhmodel_merak.xh_llm.models.deepseek_v4.cache_abi import CSA, HCA, SLIDING, DeepSeekV4CacheABI
        from xhmodel_merak.xh_llm.models.deepseek_v4.static_cache import DeepSeekV4StaticCacheSpec

        super().__init__()
        self.cache_abi = DeepSeekV4CacheABI(
            DeepSeekV4StaticCacheSpec(
                max_context_length=4096,
                prefill_chunk_length=chunk_length,
                sliding_window=16,
                latent_head_dim=32,
                index_head_dim=8,
                index_topk=16,
            ),
            (SLIDING, CSA, HCA),
        )


def test_real_image_features_are_merged_into_token_embeddings(glm53):
    from test_glm53_multimodal import tiny_multimodal_config
    from transformers.models.glm5_next import Glm5NextForConditionalGeneration
    from xhmodel_merak.xh_llm.models.glm5_3_flash.vision import Glm53VisionEncoder, StaticVisionModel

    torch.manual_seed(541)
    config = tiny_multimodal_config()
    source = Glm5NextForConditionalGeneration(config).eval()
    wrapped = StaticVisionModel(source.model.visual, 32).eval()
    encoder = Glm53VisionEncoder(wrapped.stem, wrapped.blocks, wrapped.merger, config.vision_config, 32, dtype=torch.float32)
    adapter = object.__new__(glm53.Glm53MerakForCausalLM)
    nn.Module.__init__(adapter)
    adapter.inference_model = SimpleNamespace(vision_encoder=encoder)
    adapter.input_embeddings = source.get_input_embeddings()
    grid = torch.tensor([[1, 4, 4], [1, 2, 4]])
    pixels = torch.randn(24, 24)
    with torch.no_grad():
        expected = source.get_image_features(pixels, grid).pooler_output
        actual = adapter.embed_multimodal(pixel_values=pixels, image_grid_thw=grid)
        assert [value.shape[0] for value in actual] == [4, 2]
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=2e-4, atol=1e-5)
        ids = torch.tensor([2, 50, 50, 50, 50, 3, 50, 50, 4])
        mask = ids == 50
        merged = adapter.embed_input_ids(ids, actual, is_multimodal=mask)
        torch.testing.assert_close(merged[mask], torch.cat(expected), rtol=2e-4, atol=1e-5)
        torch.testing.assert_close(merged[~mask], source.get_input_embeddings()(ids[~mask]))
    with pytest.raises(ValueError, match="placeholders"):
        adapter.embed_input_ids(ids, actual, is_multimodal=mask & (torch.arange(ids.numel()) != 1))


@pytest.mark.parametrize("size", [(8, 12), (12, 8), (20, 20)])
def test_real_glm_processor_budget_matches_packed_image(tmp_path, glm53, size):
    from PIL import Image
    from test_glm53_multimodal import save_tiny_processor, tiny_multimodal_config
    from vllm.multimodal.processing.context import InputProcessingContext

    hf = tmp_path / "hf_config"
    hf.mkdir()
    tiny_multimodal_config().save_pretrained(hf)
    save_tiny_processor(hf)
    (tmp_path / "golden_meta_info.json").write_text(json.dumps({"glm53_vision": {
        "max_patches_per_frame": 32, "vision_config": {"spatial_merge_size": 2},
    }}))
    config = SimpleNamespace(model=str(tmp_path), trust_remote_code=False)
    info = glm53.Glm53MerakProcessingInfo(InputProcessingContext(config, None))
    processor = info.get_hf_processor()
    batch = processor(text="<|image|>", images=[Image.new("RGB", size)], return_tensors="pt")
    count = int(batch["image_grid_thw"].prod()) // 4
    assert info.get_num_image_tokens(image_width=size[0], image_height=size[1]) == count
    assert int((batch["input_ids"] == processor.image_token_id).sum()) == count
    assert count <= info.get_max_image_tokens() == 8
    assert info.get_supported_mm_limits() == {"image": None}
    assert info.get_num_frames_with_most_features(262144, {"image": 1, "video": 0}) == 1
    with pytest.raises(ValueError, match="capacity"):
        info.get_hf_processor(max_image_tokens=9)


def _config(tmp_path, *, max_num_seqs=1, max_model_len=4096, architecture="Glm5NextForConditionalGeneration"):
    from vllm_merak.config import MerakConfig

    (tmp_path / "hf_config").mkdir()
    (tmp_path / "hf_config" / "config.json").write_text(json.dumps({"architectures": [architecture]}))
    config = MerakConfig(
        xh_model={"model_type": "hmonnx"},
        enable_page_attention=False,
        use_v2=False,
        auto_offload=False,
        cuda_graph=False,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_config=config,
            model=str(tmp_path),
            max_model_len=max_model_len,
            trust_remote_code=False,
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        speculative_config=None,
        cache_config=SimpleNamespace(),
    )


def _patch_runtime(monkeypatch, glm53, *, chunk_length=256, runtime_class="XHGlm53HMONNXModel"):
    runtime_type = _GlmRuntime if runtime_class == "XHGlm53HMONNXModel" else _DeepseekRuntime
    runtime = runtime_type(chunk_length)
    monkeypatch.setattr(glm53, runtime_class, runtime_type)

    def load(*args, **kwargs):
        runtime.load_kwargs = kwargs
        return runtime

    monkeypatch.setattr(glm53.AutoLLMHONNXModel, "from_pretrained", load)
    monkeypatch.setattr(glm53, "MerakAttention", lambda **kw: nn.Identity())
    monkeypatch.setattr(
        glm53,
        "load_hf_config",
        lambda *a, **kw: SimpleNamespace(vocab_size=16, get_text_config=lambda: SimpleNamespace(vocab_size=16)),
    )
    return runtime


def test_glm53_discovered_through_public_registry_in_fresh_process():
    # Exercise the production discovery entry, without manually importing glm53.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "from vllm_merak.model_executor.register import get_registered_model_cls",
                    "cls = get_registered_model_cls('Glm5NextForConditionalGeneration')",
                    "assert cls.__name__ == 'Glm53MerakForCausalLM'",
                    "assert cls.__module__ == 'vllm_merak.model_executor.model_loader.glm53'",
                ]
            ),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_glm53_registered_and_uses_glm_cache_geometry(tmp_path, monkeypatch, glm53):
    from vllm_merak.model_executor.register import MERAK_MODEL_REGISTRY

    runtime = _patch_runtime(monkeypatch, glm53)
    adapter = glm53.Glm53MerakForCausalLM(vllm_config=_config(tmp_path))
    assert MERAK_MODEL_REGISTRY.get("Glm5NextForConditionalGeneration") is type(adapter)
    assert adapter.prefill_chunk_length == 256
    assert adapter.logits_processor.vocab_size == 16
    assert adapter.inference_model is runtime
    spec = adapter.inference_model.cache_spec
    assert spec is runtime.cache_spec
    assert spec.layer_types == ("linear_attention", "deepseek_sparse_attention")
    assert spec.cache_shapes("linear_attention") == ((1, 48, 4), (1, 2, 8, 8))
    assert spec.cache_shapes("deepseek_sparse_attention") == ((1, 1, 4096, 16), (1, 1, 1024, 8), (1, 4, 8), (1, 4, 8))


@pytest.fixture(params=["glm53", "deepseek_v4"])
def runtime_adapter(request, tmp_path, monkeypatch):
    from vllm_merak.model_executor.model_loader import deepseek_v4, glm53

    if request.param == "glm53":
        module, cls, architecture, runtime_class, chunk = (
            glm53,
            glm53.Glm53MerakForCausalLM,
            "Glm5NextForConditionalGeneration",
            "XHGlm53HMONNXModel",
            256,
        )
    else:
        module, cls, architecture, runtime_class, chunk = (
            deepseek_v4,
            deepseek_v4.DeepseekV4MerakForCausalLM,
            "DeepseekV4ForCausalLM",
            "XHDeepSeekV4HMONNXModel",
            128,
        )
    runtime = _patch_runtime(monkeypatch, module, chunk_length=chunk, runtime_class=runtime_class)
    adapter = cls(vllm_config=_config(tmp_path, architecture=architecture))
    adapter._extract_request_result = lambda model, output, **kwargs: SimpleNamespace(logits=output)
    assert adapter.prefill_chunk_length == chunk
    assert runtime.load_kwargs == {}
    assert runtime.to_calls == [torch.device("cuda:0" if torch.cuda.is_available() else "cpu")]
    if request.param == "deepseek_v4":
        assert adapter.inference_model.cache_abi is runtime.cache_abi
        assert runtime.cache_abi.spec.latent_head_dim == 32
        assert runtime.cache_abi.spec.sliding_window == 16
        assert runtime.cache_abi.csa_layers == (1,)
        assert runtime.cache_abi.hca_layers == (2,)
    return adapter, runtime


@pytest.mark.parametrize("tail", [0, 44])
def test_shared_lifecycle_chunks_and_switches_requests(runtime_adapter, tail):
    adapter, runtime = runtime_adapter
    chunk = adapter.prefill_chunk_length
    length = 2 * chunk + tail
    adapter._runtime_forward(torch.arange(length), past_len=0, request_id="a")
    adapter.release_internal_cache_owner("unknown")
    assert adapter._cache_owner_request_id == "a"
    assert adapter._cache_owner_next_past_len == length
    adapter._runtime_forward(torch.tensor([1]), past_len=length, request_id="a")
    adapter.release_internal_cache_owner("a")
    assert adapter._cache_owner_request_id is None
    assert adapter._cache_owner_next_past_len == 0
    adapter._runtime_forward(torch.tensor([2, 3]), past_len=0, request_id="b")
    expected = [(chunk, 0), (chunk, chunk)]
    if tail:
        expected.append((tail, 2 * chunk))
    assert runtime.calls == expected + [(1, length), (2, 0)]
    assert runtime.resets == 2


def test_shared_lifecycle_preserves_runtime_ports_and_embeddings(runtime_adapter, monkeypatch):
    adapter, runtime = runtime_adapter
    original_weight = runtime.embed_tokens.weight.detach().clone()
    assert adapter.input_embeddings.weight.data_ptr() == runtime.embed_tokens.weight.data_ptr()
    ports = []
    forwarded = []
    logits = torch.arange(16, dtype=torch.float16).reshape(1, 16)

    def preprocess(data):
        assert data["input_ids"].dtype == torch.int32
        assert data["input_ids"].shape[0] == 1
        values = (data["input_ids"], torch.tensor([data["past_seq_length"]]), object())
        ports.append(values)
        return values

    def forward(*values):
        forwarded.append(values)
        return logits

    monkeypatch.setattr(runtime, "get_data_preprocessor", lambda: preprocess)
    monkeypatch.setattr(runtime, "forward", forward)
    output = adapter._runtime_forward(
        torch.arange(adapter.prefill_chunk_length + 1), past_len=0, request_id="ports"
    )
    assert [values[1].item() for values in ports] == [0, adapter.prefill_chunk_length]
    assert all(
        all(actual is expected for actual, expected in zip(values, recorded, strict=True))
        for values, recorded in zip(forwarded, ports, strict=True)
    )
    assert output.data_ptr() == logits.data_ptr()
    assert torch.equal(runtime.embed_tokens.weight, original_weight)


@pytest.mark.parametrize("dtype", [None, "float16"])
def test_glm53_accepts_default_and_explicit_fp16_cache(tmp_path, monkeypatch, glm53, dtype):
    _patch_runtime(monkeypatch, glm53)
    kwargs = {} if dtype is None else {"kv_cache_dtype": dtype}
    adapter = glm53.Glm53MerakForCausalLM(vllm_config=_config(tmp_path), **kwargs)
    assert adapter.cache_dtype == "float16"
    assert adapter.cache_torch_dtype == torch.float16


@pytest.mark.parametrize("auto_offload", [False, True])
def test_glm53_preserves_runtime_device_map(tmp_path, monkeypatch, glm53, auto_offload):
    runtime = _patch_runtime(monkeypatch, glm53)
    monkeypatch.setenv("ENABLE_HMINFERENCE_V2", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    config = _config(tmp_path)
    config.model_config.hf_config.use_v2 = True
    config.model_config.hf_config.auto_offload = auto_offload
    config.model_config.hf_config.cuda_graph = False
    adapter = glm53.Glm53MerakForCausalLM(vllm_config=config)
    assert runtime.load_kwargs["device_map"] == ([0, 1, 2, 3] if auto_offload else [0])
    assert runtime.to_calls == ([] if auto_offload else [torch.device("cuda:0")])
    assert adapter.inference_device == runtime.device


@pytest.mark.parametrize(
    "max_num_seqs,max_model_len,error",
    [
        (2, 4096, "max_num_seqs=1"),
        (1, 8192, "exceeds"),
    ],
)
def test_glm53_rejects_unsupported_scheduler_contracts(
    tmp_path, monkeypatch, max_num_seqs, max_model_len, error, glm53
):
    _patch_runtime(monkeypatch, glm53)
    with pytest.raises(ValueError, match=error):
        glm53.Glm53MerakForCausalLM(
            vllm_config=_config(tmp_path, max_num_seqs=max_num_seqs, max_model_len=max_model_len)
        )


@pytest.mark.parametrize("contract", ["page_attention", "speculative", "cache_dtype"])
def test_glm53_rejects_incompatible_state_contract(tmp_path, monkeypatch, glm53, contract):
    _patch_runtime(monkeypatch, glm53)
    config = _config(tmp_path)
    kwargs = {}
    if contract == "page_attention":
        config.model_config.hf_config.enable_page_attention = True
        error = "PageAttention"
    elif contract == "speculative":
        config.speculative_config = object()
        error = "speculative"
    else:
        kwargs["kv_cache_dtype"] = "float32"
        error = "float16"
    with pytest.raises(ValueError, match=error):
        glm53.Glm53MerakForCausalLM(vllm_config=config, **kwargs)


def test_shared_lifecycle_rejects_wrong_owner_and_clears_failed_request(runtime_adapter, monkeypatch):
    adapter, runtime = runtime_adapter
    adapter._runtime_forward(torch.tensor([1, 2]), past_len=0, request_id="a")
    with pytest.raises(RuntimeError, match="matching resident owner"):
        adapter._runtime_forward(torch.tensor([3]), past_len=2, request_id="b")
    with pytest.raises(RuntimeError, match="not monotonic"):
        adapter._runtime_forward(torch.tensor([3]), past_len=1, request_id="a")

    def fail(*args):
        raise RuntimeError("synthetic runtime failure")

    successful_forward = runtime.forward
    monkeypatch.setattr(runtime, "forward", fail)
    with pytest.raises(RuntimeError, match="synthetic runtime failure"):
        adapter._runtime_forward(torch.tensor([3]), past_len=2, request_id="a")
    assert adapter._cache_owner_request_id is None
    assert adapter._cache_owner_next_past_len == 0
    # Failed cache contents cannot be resumed; the next request clears them.
    assert runtime.resets == 1
    monkeypatch.setattr(runtime, "forward", successful_forward)
    adapter._runtime_forward(torch.tensor([4, 5]), past_len=0, request_id="new")
    assert runtime.resets == 2
    assert runtime.calls == [(2, 0), (1, 2), (2, 0)]
