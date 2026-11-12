"""Small tensor checks for Qwen4 generation's physical cache and step ABI."""

from types import SimpleNamespace

import pytest
import torch

from xhmodel_merak.xh_llm.models.qwen4_exp.cache_contract import Qwen4ExpCacheContract
from xhmodel_merak.xh_llm.models.qwen4_exp.generation import _allocate_page_caches, _page_step
from xhmodel_merak.xh_llm.models.qwen4_exp.qwen4_exp_hmonnx_inference import (
    Qwen4ExpServingDataPreprocess,
    Qwen4ExpServingState,
    XHQwen4ExpHMONNXModel,
)
from xhquant.xhonnxruntime.parsers.page_attention import PageKVCache


def _writer(heads=1):
    return PageKVCache(
        layer_idx=0, cache_index=0, sliding_window=128, output_sequence_length=128, num_kv_heads=heads, head_size=64
    )


def test_allocate_bind_write_and_read_physical_page_cache():
    prefill, decode = object(), object()
    writer = _writer()
    model = SimpleNamespace(
        prefill_model=prefill,
        decode_model=decode,
        device="cpu",
        dtype=torch.float32,
        kvcache_config=SimpleNamespace(num_layers=1),
        qwen4_cache_contract=SimpleNamespace(head_dim=64),
        meta_info=SimpleNamespace(model_config=SimpleNamespace(context_max_length=65)),
        _get_page_attention_modules=lambda stage: [writer],
        is_prefill=lambda: True,
    )
    caches = _allocate_page_caches(model)
    assert len(caches) == 1 and caches[0].shape == (2, 2, 1, 64, 64)
    assert caches[0].dtype == torch.float32
    assert torch.count_nonzero(caches[0]) == 0
    XHQwen4ExpHMONNXModel.set_page_attention_context(
        model,
        caches,
        torch.tensor([0, 1]),
        torch.tensor([63, 64, -1, -1]),
        64,
        65,
    )
    key = torch.arange(4 * 64, dtype=torch.float32).reshape(1, 1, 4, 64)
    value = key + 1000
    writer._write_raw_cache(key, value)
    torch.testing.assert_close(caches[0][0, 0, 0, 63], key[0, 0, 0])
    torch.testing.assert_close(caches[0][1, 1, 0, 0], value[0, 0, 1])
    assert torch.count_nonzero(caches[0][:, 1, :, 1:]) == 0
    gathered_key, gathered_value = writer._gather_raw_cache(torch.tensor([63]), torch.tensor([2]))
    torch.testing.assert_close(gathered_key[0, 0, 63:65], key[0, 0, :2])
    torch.testing.assert_close(gathered_value[0, 0, 63:65], value[0, 0, :2])
    model._get_page_attention_modules = lambda stage: [_writer(1 if stage is prefill else 2)]
    with pytest.raises(ValueError, match="layouts differ"):
        _allocate_page_caches(model)


def test_partial_prefill_uses_valid_logits_and_padding_has_no_cache_slots():
    config = SimpleNamespace(
        layer_types=["full_attention"],
        num_hidden_layers=1,
        hidden_size=64,
        num_key_value_heads=1,
        num_attention_heads=1,
        head_dim=64,
        hc_count=1,
        indexer_head_dim=64,
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=2,
        ple_conv_kernel_size=2,
        ngram_size=2,
        ple_layer_ids=[],
        ple_hidden_size=64,
    )
    contract = Qwen4ExpCacheContract.from_text_config(config)
    state = Qwen4ExpServingState(contract, batch_size=1, context_length=128, dtype=torch.float32, device="cpu")
    embedding = torch.nn.Embedding(8, 64)
    with torch.no_grad():
        embedding.weight.copy_(torch.arange(8 * 64).reshape(8, 64))
    processor = Qwen4ExpServingDataPreprocess(
        token_embedding=embedding,
        input_sequence_length=4,
        state=state,
        past_key_caches=None,
        past_value_caches=None,
        enable_page_attention=True,
        pad_token_id=0,
    )

    class StepGraph:
        """A numeric head fixture; production preprocessing is exercised intact."""

        context = None

        def get_input_sequence_length(self):
            return 4

        def get_data_preprocessor(self):
            return processor

        def set_page_attention_context(self, *args, **kwargs):
            self.context = args, kwargs

        def forward(self, embeds, past, valid, ple_ids, *states):
            assert past.item() == 63 and valid.item() == 2
            assert ple_ids.tolist() == [[2, 3, 0, 0]]
            return embeds

    graph = StepGraph()
    assert not callable(graph)  # HMONNX runtime wrappers expose forward(), not __call__.
    logits = _page_step(graph, [], torch.tensor([[2, 3]]), 63)
    torch.testing.assert_close(logits, embedding(torch.tensor([3])))
    args, kwargs = graph.context
    assert args[1].tolist() == [0, 1]
    assert args[2].tolist() == [63, 64, -1, -1]
    assert kwargs["effective_kv_length"] == 65


class _GenerationGraph:
    """Boundary graph executor with real PageKV binding and numeric next-token head."""

    generate = XHQwen4ExpHMONNXModel.generate
    set_page_attention_context = XHQwen4ExpHMONNXModel.set_page_attention_context
    device = "cpu"
    dtype = torch.float32
    hf_model_dir = ""

    def __init__(self):
        self.prefill_model, self.decode_model = object(), object()
        self.prefill_writer, self.decode_writer = _writer(), _writer()
        self.meta_info = SimpleNamespace(model_config=SimpleNamespace(context_max_length=16, prefill_chunk_length=4))
        self.kvcache_config = SimpleNamespace(num_layers=1)
        self.qwen4_cache_contract = SimpleNamespace(head_dim=64)
        self.steps = []
        self.resets = 0
        self.prefill = True
        self.cache_ids = []

    def _get_page_attention_modules(self, graph):
        return [self.prefill_writer if graph is self.prefill_model else self.decode_writer]

    def get_tokenizer(self):
        return SimpleNamespace(eos_token_id=None)

    def reset_internal_state(self):
        self.resets += 1
        self.steps.clear()

    def is_prefill(self):
        return self.prefill

    def set_prefill(self):
        self.prefill = True

    def set_decode(self):
        self.prefill = False

    def get_input_sequence_length(self):
        return 4 if self.prefill else 1

    def get_data_preprocessor(self):
        # Only the graph execution boundary is synthetic. Preprocessing already
        # has a separate real-processor numeric test above.
        return lambda data: (data,)

    def forward(self, data):
        ids = data["input_ids"]
        valid = ids.shape[1]
        width = self.get_input_sequence_length()
        padded = torch.nn.functional.pad(ids, (0, width - valid))
        writer = self.prefill_writer if self.prefill else self.decode_writer
        context = writer._get_context()
        if data["past_seq_length"] == 0:
            assert torch.count_nonzero(context.paged_kv_cache) == 0
            self.cache_ids.append(context.paged_kv_cache)
        self.steps.append((self.prefill, data["past_seq_length"], valid, context.slot_mapping.tolist()))
        key = torch.nn.functional.one_hot(padded, num_classes=64).float().unsqueeze(1)
        writer._write_raw_cache(key, key)
        return torch.nn.functional.one_hot((padded + 1) % 64, num_classes=64).float()

    def to(self, *, device):
        assert device == "cpu"
        return self


def test_generation_multiple_prefill_chunks_decode_eos_and_new_request():
    from xhmodel_merak.xh_llm.models.qwen4_exp.generation import generate_token_ids

    model = _GenerationGraph()
    prompt = torch.tensor([[1, 2, 3, 4, 5]])
    output = generate_token_ids(model, prompt, max_new_tokens=5, eos_token_ids=[7, 9])
    assert output == [6, 7]
    assert model.steps == [(True, 0, 4, [0, 1, 2, 3]), (True, 4, 1, [4, -1, -1, -1]), (False, 5, 1, [5])]
    assert model.resets == 1
    output = generate_token_ids(model, prompt, max_new_tokens=3)
    assert output == [6, 7, 8]
    assert model.steps[-1] == (False, 6, 1, [6])
    assert model.resets == 2
    assert model.cache_ids[0] is not model.cache_ids[1]


@pytest.mark.parametrize(
    "tokens,limit",
    [
        (torch.ones((1, 15), dtype=torch.long), 2),
        (torch.ones((1, 1), dtype=torch.long), 0),
        (torch.ones((2, 1), dtype=torch.long), 2),
    ],
)
def test_generation_rejects_bad_request_before_reset_or_allocation(tokens, limit):
    from xhmodel_merak.xh_llm.models.qwen4_exp.generation import generate_token_ids

    model = _GenerationGraph()
    with pytest.raises(ValueError):
        generate_token_ids(model, tokens, max_new_tokens=limit)
    assert model.resets == 0 and model.steps == []


@pytest.mark.parametrize(
    "chat,eos,device_map,page_attention",
    [
        (False, 7, None, False),
        (True, [7, 9], None, True),
        (False, [7, 9], ["cpu"], False),
        (True, 7, ["cpu"], True),
    ],
)
def test_text_generation_tokenizer_boundary_and_eos_config(
    tmp_path, monkeypatch, eos, chat, device_map, page_attention
):
    import json

    from xhmodel_merak.xh_llm.models.qwen4_exp import generation

    class Tokenizer:
        chat_template = "template" if chat else None
        eos_token_id = 63

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "hello"}]
            assert not tokenize and add_generation_prompt
            return "formatted hello"

        def __call__(self, text, **kwargs):
            assert text == ("formatted hello" if chat else "hello")
            if chat:
                assert kwargs["add_special_tokens"] is False
            return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}

        def decode(self, tokens, *, skip_special_tokens):
            assert tokens == [6, 7] and skip_special_tokens
            return "result"

    model = _GenerationGraph()
    model.enable_page_attention = page_attention
    model.hf_model_dir = str(tmp_path)
    model.get_tokenizer = Tokenizer
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": eos}))

    def load(path, **kwargs):
        assert path == "runtime.json"
        assert kwargs == {
            "enable_page_attention": True,
            "enable_cuda_graph": False,
            "device_map": "cpu" if device_map is None else device_map,
        }
        return model

    monkeypatch.setattr(generation.AutoLLMHONNXModel, "from_pretrained", load)
    text, count, seconds, rate, stats = generation.run_text_generation(
        "runtime.json",
        prompt="hello",
        max_new_tokens=5,
        device="cpu",
        device_map=device_map,
    )
    assert text == "result" and count == 2
    assert seconds > 0 and rate == 2 / seconds
    assert stats == {"input_tokens": 5, "decoding": "greedy", "page_attention": page_attention}


@pytest.mark.parametrize(
    "prompt_length,max_new_tokens,expected_steps,expected_output",
    [
        (9, 1, [(True, 0, 4, [0, 1, 2, 3]), (True, 4, 4, [4, 5, 6, 7]), (False, 8, 1, [8])], [10]),
        (8, 2, [(True, 0, 4, [0, 1, 2, 3]), (True, 4, 4, [4, 5, 6, 7]), (False, 8, 1, [8])], [9, 10]),
    ],
)
def test_unaligned_context_tail_uses_decode_without_padding_overflow(
    prompt_length,
    max_new_tokens,
    expected_steps,
    expected_output,
):
    from xhmodel_merak.xh_llm.models.qwen4_exp.generation import generate_token_ids

    class CapacityCheckedGraph(_GenerationGraph):
        def forward(self, data):
            # Exercise the actual fixed-width index scatter that made the
            # padded prefill tail fail, independently of PageKV's pad mask.
            past = data["past_seq_length"]
            positions = torch.arange(past, past + self.get_input_sequence_length())
            scratch = torch.zeros(self.meta_info.model_config.context_max_length)
            scratch.scatter_(0, positions, torch.ones_like(positions, dtype=scratch.dtype))
            return super().forward(data)

    model = CapacityCheckedGraph()
    model.meta_info.model_config.context_max_length = 10
    output = generate_token_ids(
        model,
        torch.arange(1, prompt_length + 1).unsqueeze(0),
        max_new_tokens=max_new_tokens,
    )
    assert output == expected_output
    assert model.steps == expected_steps
    assert not model.is_prefill()


@pytest.mark.parametrize("device_map", [[], "cpu", ["cuda:0"], ["cpu", "cuda:0"]])
def test_generation_rejects_invalid_execution_device_map_before_loading(device_map):
    from xhmodel_merak.xh_llm.models.qwen4_exp.generation import run_text_generation

    with pytest.raises(ValueError):
        run_text_generation("missing.json", prompt="hello", max_new_tokens=1, device="cpu", device_map=device_map)


@pytest.mark.parametrize(
    ("device", "device_map", "expected_device", "expected_map"),
    [
        ("cuda", ["cuda:3"], "cuda:3", ["cuda:3"]),
        ("cuda:3", ["cuda"], "cuda:3", ["cuda:3"]),
        ("cuda", ["cuda", "cuda:7"], "cuda:3", ["cuda:3", "cuda:7"]),
        ("cuda:7", ["cuda:7"], "cuda:7", ["cuda:7"]),
    ],
)
def test_generation_resolves_cuda_alias_before_device_map_membership(
    monkeypatch, device, device_map, expected_device, expected_map
):
    from contextlib import nullcontext

    from xhmodel_merak.xh_llm.models.qwen4_exp import generation

    contexts = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(torch.cuda, "device", lambda value: contexts.append(value) or nullcontext())
    monkeypatch.setattr(generation, "_run_text_generation", lambda meta_file, **kwargs: kwargs)
    result = generation.run_text_generation(
        "unused.json", prompt="hello", max_new_tokens=1, device=device, device_map=device_map
    )
    assert result["device"] == expected_device
    assert result["device_map"] == expected_map
    assert contexts == [expected_device]


@pytest.mark.parametrize("eos", [7, [7, 9]])
def test_model_generate_runs_owned_loop_preserves_ids_and_finishes_stream(eos):
    model = _GenerationGraph()
    prompt = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int32)
    events = []
    streamer = SimpleNamespace(put=lambda value: events.append(value.clone()), end=lambda: events.append("end"))
    result = XHQwen4ExpHMONNXModel.generate(
        model,
        prompt,
        max_new_tokens=4,
        eos_token_id=eos,
        pad_token_id=0,
        attention_mask=torch.ones_like(prompt),
        streamer=streamer,
    )
    assert result.tolist() == [[1, 2, 3, 4, 5, 6, 7]]
    assert result.dtype == prompt.dtype and result.device == prompt.device
    assert events[0].tolist() == prompt.tolist()
    assert events[1].tolist() == [[6, 7]] and events[2] == "end"
    assert model.steps[-1] == (False, 5, 1, [5])


@pytest.mark.parametrize(
    "options",
    [
        {"do_sample": True},
        {"attention_mask": torch.tensor([[1, 0]])},
        {"inputs_embeds": torch.zeros(1, 3, 4)},
        {"num_beams": 2},
    ],
)
def test_model_generate_rejects_unsupported_request_and_ends_stream(options):
    events = []
    model = _GenerationGraph()
    streamer = SimpleNamespace(end=lambda: events.append("end"))
    with pytest.raises(ValueError):
        XHQwen4ExpHMONNXModel.generate(model, torch.tensor([[1, 2]]), streamer=streamer, **options)
    assert events == ["end"] and model.resets == 0


def test_model_generate_resets_persistent_draft_and_ends_streamer_on_execution_error(monkeypatch):
    from xhmodel_merak.xh_llm.models.qwen4_exp import generation

    events = []
    monkeypatch.setattr(
        generation, "create_mtp_runtime", lambda model, **kwargs: SimpleNamespace(reset=lambda: events.append("reset"))
    )

    def fail(*args, **kwargs):
        raise RuntimeError("graph failed")

    monkeypatch.setattr(generation, "generate_token_ids", fail)
    streamer = SimpleNamespace(put=lambda value: None, end=lambda: events.append("end"))
    with pytest.raises(RuntimeError, match="graph failed"):
        XHQwen4ExpHMONNXModel.generate(_GenerationGraph(), torch.tensor([[1]]), streamer=streamer)
    assert events == ["reset", "end"]


def test_verify_page_layout_must_match_target():
    model = _GenerationGraph()
    model.verify_model = object()
    model._get_page_attention_modules = lambda graph: [_writer(2 if graph is model.verify_model else 1)]
    with pytest.raises(ValueError, match="prefill/verify PageKV"):
        _allocate_page_caches(model)


@pytest.mark.parametrize("version", [2, 3, 4])
def test_shared_mtp_factory_resolves_graphs_and_execution_devices(monkeypatch, tmp_path, version):
    from addict import Dict

    from xhmodel_merak.xh_llm.models.qwen4_exp import mtp_runtime
    from xhmodel_merak.xh_llm.models.qwen4_exp.generation import create_mtp_runtime

    captured = {}
    builds = []

    def factory(path, **kwargs):
        builds.append(path)
        captured.update(path=path, **kwargs)
        return captured

    monkeypatch.setattr(mtp_runtime, "MTPGraphRuntime", factory)
    spec = Dict(
        mode="mtp",
        runtime_contract_version=version,
        draft={
            "decode_hmonnx": "mtp_decode/model.hmonnx",
            "prefill_hmonnx": "mtp_prefill/model.hmonnx",
            "prefill_chunk_length": 4,
            "cache_contract": {"marker": 1},
        },
    )
    model = SimpleNamespace(
        meta_info=SimpleNamespace(
            spec_decode=spec,
            prefill_hmonnx=str(tmp_path / "prefill/model.hmonnx"),
            model_config=SimpleNamespace(context_max_length=16),
        ),
        device="cuda:7",
        _valid_devices=["cuda:2", "cuda:7"],
        _qwen4_mtp_graph=object(),
        _qwen4_mtp_prefill_graph=object(),
    )
    first = create_mtp_runtime(model)
    assert create_mtp_runtime(model) is first
    assert len(builds) == 1
    if version >= 3:
        model._qwen4_verify_length = 5
        capture = SimpleNamespace(components=("mtp",), required_graphs=set())
        assert create_mtp_runtime(model, capture=capture) is first
        assert capture.required_graphs == {"mtp_prefill", "mtp_decode", "decode_m5"}
        assert len(builds) == 1
    assert captured["path"] == tmp_path / "mtp_decode/model.hmonnx"
    assert captured["device_map"] == ["cuda:2", "cuda:7"]
    assert captured["onnx_graph"] is model._qwen4_mtp_graph
    assert ("prefill_path" in captured) == (version >= 3)
    if version >= 3:
        assert captured["prefill_graph"] is model._qwen4_mtp_prefill_graph
        assert captured["prefill_path"] == tmp_path / "mtp_prefill/model.hmonnx"
    model._qwen4_mtp_runtime = SimpleNamespace(session=None)
    create_mtp_runtime(model)
    assert len(builds) == 2


def test_model_generate_passes_multimodal_inputs_and_resets_persistent_mtp(monkeypatch):
    from xhmodel_merak.xh_llm.models.qwen4_exp import generation

    events = []
    draft = SimpleNamespace(reset=lambda: events.append("reset"))
    monkeypatch.setattr(generation, "create_mtp_runtime", lambda model, **kwargs: draft)
    embeds = torch.randn(1, 2, 64)
    positions = torch.arange(2).reshape(1, 1, 2).expand(3, 1, 2)

    def execute(model, tokens, **kwargs):
        assert tokens.dtype == torch.int64
        torch.testing.assert_close(kwargs["inputs_embeds"], embeds)
        torch.testing.assert_close(kwargs["position_ids"], positions)
        assert kwargs["mtp_runtime"] is draft
        return [3]

    monkeypatch.setattr(generation, "generate_token_ids", execute)
    output = XHQwen4ExpHMONNXModel.generate(
        _GenerationGraph(),
        torch.tensor([[1, 2]], dtype=torch.int32),
        inputs_embeds=embeds,
        position_ids=positions,
        max_new_tokens=1,
    )
    assert output.tolist() == [[1, 2, 3]] and output.dtype == torch.int32
    assert events == ["reset"]


@pytest.mark.parametrize(
    "configured_eos,tokenizer_eos,explicit_eos,expected",
    [
        ([7, 9], 8, None, [6, 7]),
        (7, 8, None, [6, 7]),
        (None, 7, None, [6, 7]),
        ([9], 8, 6, [6]),
    ],
)
def test_model_generate_resolves_packaged_eos_without_runtime_metadata(
    tmp_path,
    configured_eos,
    tokenizer_eos,
    explicit_eos,
    expected,
):
    import json

    model = _GenerationGraph()
    assert not hasattr(model.meta_info, "eos_token_id")
    model.hf_model_dir = str(tmp_path)
    model.get_tokenizer = lambda: SimpleNamespace(eos_token_id=tokenizer_eos)
    if configured_eos is not None:
        (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": configured_eos}))
    result = XHQwen4ExpHMONNXModel.generate(
        model,
        torch.tensor([[4, 5]]),
        max_new_tokens=4,
        eos_token_id=explicit_eos,
    )
    assert result.tolist() == [[4, 5] + expected]


@pytest.mark.parametrize(
    "metadata,expected",
    [
        ({"runtime_contract_version": 3}, 1),
        ({"runtime_contract_version": 4}, 4),
        ({"runtime_contract_version": 4, "num_draft_tokens": 2}, 2),
    ],
)
def test_mtp_candidate_defaults_preserve_legacy_and_use_new_v4(metadata, expected):
    from xhmodel_merak.xh_llm.models.qwen4_exp.generation import _mtp_candidate_count

    model = SimpleNamespace(meta_info=SimpleNamespace(spec_decode=metadata))
    assert _mtp_candidate_count(model) == expected


def test_mtp_verify_width_rejected_before_request_state_reset():
    from xhmodel_merak.xh_llm.models.qwen4_exp.generation import generate_token_ids

    model = SimpleNamespace(
        _qwen4_verify_length=5,
        meta_info=SimpleNamespace(
            spec_decode={"runtime_contract_version": 4, "num_draft_tokens": 5},
            model_config=SimpleNamespace(context_max_length=32, prefill_chunk_length=4),
        ),
    )
    with pytest.raises(ValueError, match="verify length"):
        generate_token_ids(model, torch.tensor([[1]]), max_new_tokens=2, mtp_runtime=object())


@pytest.mark.parametrize("fail", [False, True])
def test_text_helper_exposes_mtp_stats_and_releases_model(monkeypatch, fail):
    from xhmodel_merak.xh_llm.models.qwen4_exp import generation

    events = []

    class Tokenizer:
        chat_template = None

        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, tokens, **kwargs):
            assert tokens == [3]
            return "answer"

    stats = {"num_rounds": 1, "accepted_drafts_total": 2}
    model = SimpleNamespace(
        enable_page_attention=False,
        device="cpu",
        last_mtp_stats=stats,
        get_tokenizer=Tokenizer,
        close=lambda: events.append("close"),
    )
    model.to = lambda **kwargs: model

    def generate(**kwargs):
        if fail:
            raise RuntimeError("request failed")
        return torch.tensor([[1, 2, 3]])

    model.generate = generate
    monkeypatch.setattr(generation.AutoLLMHONNXModel, "from_pretrained", lambda *args, **kwargs: model)
    if fail:
        with pytest.raises(RuntimeError, match="request failed"):
            generation.run_text_generation("meta.json", prompt="prompt", max_new_tokens=1, device="cpu")
    else:
        result = generation.run_text_generation("meta.json", prompt="prompt", max_new_tokens=1, device="cpu")
        assert result[-1]["mtp"] is stats
    assert events == ["close"]
