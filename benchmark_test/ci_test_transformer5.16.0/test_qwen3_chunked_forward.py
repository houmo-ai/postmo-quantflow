"""CPU generation regressions using a one-layer Qwen3 and real SDK KV caches.

No checkpoint, tokenizer download, HMONNX session or quantization is required.
These cover WRAP scheduling/cache semantics, not QUANTED_ALIGNED numerics.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from examples_merak.llm.qwen3.debug_scripts import qwen3_xh2a_forward as script

pytestmark = pytest.mark.transformer516


@pytest.fixture
def tiny_models(monkeypatch):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from xhmodel_merak.xh_llm.models.qwen3 import XHQwen3Model, XHQwen3ModelConfig
    from xhquant.core.context import get_current_context

    monkeypatch.setattr(get_current_context().kvcache, "fast_mode", False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(519)
        native = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, max_position_embeddings=32, eos_token_id=None, pad_token_id=31,
            attention_dropout=0.0,
        )).half().eval()
    native.config._attn_implementation = "eager"
    model = XHQwen3Model(XHQwen3ModelConfig(
        model_name="tiny-qwen3", model_type="Qwen3ForCausalLM",
        prefill_chunk_length=4, context_max_length=32, max_pe_length=32,
    ))
    model.to_wrap(deepcopy(native))
    model.pad_token_id = 31
    model.to(device="cpu", dtype=torch.float16)
    model.eval()
    return native, model


def run_chunked(model, ids):
    from xhmodel_merak.xh_llm import LLMInferenceContextManager

    calls, logits = [], []
    def capture(_module, inputs, output):
        calls.append((inputs[0].shape[1], int(inputs[1].item()), int(inputs[2].item())))
        logits.append(output.detach().clone())
    handle = model.get_inference_model().register_forward_hook(capture)
    try:
        with LLMInferenceContextManager(model, devices=["cpu"]), torch.no_grad():
            tokens = script._generate_wrap_step_by_step(
                model, ids, max_new_tokens=3, eos_token_ids=set(),
            )
            caches = [cache.detach().clone() for cache in (*model.past_key_caches, *model.past_value_caches)]
    finally:
        handle.remove()
    return tokens, calls, logits, caches


def full_prefix_greedy(native, ids, count=3):
    """Independent reference: recompute the full prefix without a KV cache."""
    tokens = []
    with torch.no_grad():
        for _ in range(count):
            token = int(native(input_ids=ids, use_cache=False).logits[:, -1].argmax(-1).item())
            tokens.append(token)
            ids = torch.cat((ids, torch.tensor([[token]])), dim=1)
    return tokens


@pytest.mark.parametrize("prompt_length", [3, 4, 6], ids=["short", "exact", "partial-tail"])
@pytest.mark.parametrize("num_logits", [0, 1], ids=["full-logits", "gathered-logits"])
def test_chunked_prefill_matches_full_prefix(tiny_models, prompt_length, num_logits):
    native, model = tiny_models
    model.config.num_logits_to_keep = num_logits
    model.wrap_cfg.num_logits_to_keep = num_logits
    ids = torch.arange(1, prompt_length + 1).unsqueeze(0)
    tokens, calls, logits, _ = run_chunked(model, ids)
    assert tokens == full_prefix_greedy(native, ids)
    expected_prefill = [(4, start, min(4, prompt_length - start)) for start in range(0, prompt_length, 4)]
    assert calls == expected_prefill + [(1, prompt_length, 1), (1, prompt_length + 1, 1)]
    assert [output.shape[1] for output in logits] == [4 if num_logits == 0 else 1] * len(expected_prefill) + [1, 1]


def test_cache_fast_modes_preserve_logits_tokens_and_valid_cache(tiny_models, monkeypatch):
    from xhquant.core.context import get_current_context

    native, model = tiny_models
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    slow = run_chunked(model, ids)
    monkeypatch.setattr(get_current_context().kvcache, "fast_mode", True)
    fast = run_chunked(model, ids)
    assert slow[0] == fast[0] == full_prefix_greedy(native, ids)
    assert slow[1] == fast[1]
    for left, right in zip(slow[2], fast[2], strict=True):
        torch.testing.assert_close(left, right, rtol=2e-3, atol=2e-3)
    for left, right in zip(slow[3], fast[3], strict=True):
        torch.testing.assert_close(left[:, :, :8], right[:, :, :8], rtol=2e-3, atol=2e-3)


def test_forward_main_disables_fast_cache_before_execution(tiny_models, monkeypatch, tmp_path):
    from xhquant.core.context import get_current_context

    native, model = tiny_models
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    expected = full_prefix_greedy(native, ids)
    tokenizer = SimpleNamespace(eos_token_id=None, decode=lambda ids, **kwargs: str(ids))
    processor = SimpleNamespace(tokenizer=tokenizer, apply_chat_template=lambda *args, **kwargs: SimpleNamespace(input_ids=ids))
    monkeypatch.setattr(model, "get_tf_processor", lambda: processor)
    model.config.hf_model = str(tmp_path)
    config_path = tmp_path / "config.py"
    config_path.touch()
    monkeypatch.setattr(script, "parse_args", lambda: SimpleNamespace(
        max_new_tokens=3, config_path=str(config_path), device="cpu", dtype="fp16",
        work_dir=str(tmp_path / "work"), debug=False, seed=519, model_dir=None,
        prompt="test", think=False, eval_type="wrap",
    ))
    monkeypatch.setattr(script, "_load_merak_config", lambda *args: (SimpleNamespace(pretty_text="tiny"), model.config))
    monkeypatch.setattr(script, "_build_xh_model", lambda *args: model)
    steps, modes = [], []
    monkeypatch.setattr(script, "_make_step_reporter", lambda *args: steps.append)
    context = get_current_context()
    monkeypatch.setattr(context.kvcache, "fast_mode", True)
    handle = model.get_inference_model().register_forward_pre_hook(lambda *_: modes.append(context.kvcache.fast_mode))
    try:
        assert script.main() == 0
    finally:
        handle.remove()
    assert modes == [False] * 4
    assert [step.output_token_id for step in steps] == expected


def test_greedy_generation_matches_full_prefix_and_ignores_rng(tiny_models):
    native, _ = tiny_models
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    expected = full_prefix_greedy(native, ids)
    # Seeded model initialization is not a cross-version token golden. Compare
    # against independent, cache-free argmax decoding of these same weights.
    assert len(expected) == 3
    for seed in (1, 519):
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(seed)
            generated = native.generate(input_ids=ids, max_new_tokens=3, do_sample=False)
        assert generated[0, ids.shape[1]:].tolist() == expected
