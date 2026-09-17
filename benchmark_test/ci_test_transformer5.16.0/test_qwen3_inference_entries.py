"""Exercise native entrypoints with a tiny Qwen3; test HMONNX host scheduling.

Only file/tokenizer loading and streaming UI are replaced for native tests.
HMONNX scheduling uses a session double, not a numerical graph equivalence test.
"""
import importlib
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.transformer516
PREFIX = "examples_merak.llm.qwen3.debug_scripts."


@pytest.mark.parametrize("entry", ["native_qwen3_forward", "native_qwen3_generate", "qwen3_native_generate_workflow"])
def test_native_entry_runs_real_greedy_model(monkeypatch, tmp_path, entry):
    from transformers import BatchEncoding, Qwen3Config, Qwen3ForCausalLM
    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow
    from xhquant.core.context import get_current_context

    module = importlib.import_module(PREFIX + entry)
    ids = torch.tensor([[1, 2, 3]])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(519)
        model = Qwen3ForCausalLM(Qwen3Config(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
            num_attention_heads=2, num_key_value_heads=1, head_dim=8,
            max_position_embeddings=32, pad_token_id=31,
        )).half().eval()
    # Every possible token is EOS, bounding the scripts' 1024-token budget to one real step.
    model.generation_config.eos_token_id = list(range(32))
    model.config._attn_implementation = "eager"
    with torch.no_grad():
        expected = int(model(input_ids=ids, use_cache=False).logits[:, -1].argmax(-1).item())
    decoded, templates, generate_calls, forward_calls = [], [], [], []

    class Tokenizer:
        pad_token_id = 31
        eos_token_id = None
        def apply_chat_template(self, messages, **kwargs):
            templates.append((messages, kwargs))
            return "prompt"
        def __call__(self, *args, **kwargs):
            return BatchEncoding({"input_ids": ids.clone()})
        def encode(self, text):
            return [99]
        def decode(self, token_ids, **kwargs):
            tokens = list(token_ids)
            decoded.append(tokens)
            return str(tokens)

    tokenizer = Tokenizer()
    def apply_template(messages, **kwargs):
        templates.append((messages, kwargs))
        return BatchEncoding({"input_ids": ids.clone()})
    processor = SimpleNamespace(tokenizer=tokenizer, apply_chat_template=apply_template)
    xh_model = SimpleNamespace(config=SimpleNamespace(enable_auto_offload=False),
                               get_tf_processor=lambda: processor, get_native_model=lambda: model)
    real_generate = model.generate
    def generate(**kwargs):
        generate_calls.append(kwargs.copy())
        return real_generate(**kwargs)
    monkeypatch.setattr(model, "generate", generate)
    handle = model.register_forward_pre_hook(lambda *args: forward_calls.append(True))
    monkeypatch.chdir(tmp_path)
    prompt = tmp_path / "prompt.json"
    prompt.write_text('{"query": "test prompt"}')
    monkeypatch.setattr(get_current_context().kvcache, "fast_mode", False)
    try:
        if entry == "native_qwen3_generate":
            monkeypatch.setattr(module.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: tokenizer)
            monkeypatch.setattr(module.AutoModelForCausalLM, "from_pretrained", lambda *args, **kwargs: model)
            module.main(SimpleNamespace(model_dir=str(tmp_path), prompt=str(prompt)))
        elif entry == "native_qwen3_forward":
            cfg_path = tmp_path / "config.py"
            cfg_path.touch()
            cfg = SimpleNamespace(hf_model=str(tmp_path), context_max_length=32, to_json_string=lambda: "{}")
            monkeypatch.setattr(module, "_load_merak_config", lambda *args: (SimpleNamespace(pretty_text="tiny"), cfg))
            monkeypatch.setattr(module, "_build_xh_model", lambda *args: xh_model)
            monkeypatch.setattr(module, "parse_args", lambda: SimpleNamespace(
                config_path=str(cfg_path), model_dir=None, prompt=str(prompt), device="cpu",
                dtype="fp16", max_new_tokens=3, seed=519, think=False,
                work_dir=str(tmp_path / "work"), debug=False,
            ))
            assert module.main() == 0
        else:
            workflow = SimpleNamespace(workflow_config=SimpleNamespace(export={"model": {"chip_arch": "XH2a"}}),
                                       get_model=lambda **kwargs: xh_model)
            monkeypatch.setattr(AutoLLMWorkflow, "from_config", lambda **kwargs: workflow)
            monkeypatch.setattr(module, "parse_args", lambda: SimpleNamespace(
                model_dir=str(tmp_path), config_path="tiny.yaml", device="cpu", prompt=str(prompt),
                think=False, debug=False, auto_offload=False,
            ))
            class Streamer:
                def __init__(self, **kwargs):
                    pass
                def put(self, value):
                    pass
                def end(self):
                    pass
            monkeypatch.setattr(module, "TextStreamer", Streamer)
            module.main()
    finally:
        handle.remove()
    assert forward_calls == [True]
    assert decoded[-1] == [expected]
    assert templates[0][0] == [{"role": "user", "content": "test prompt"}]
    if entry != "native_qwen3_forward":
        assert len(generate_calls) == 1
        assert generate_calls[0]["do_sample"] is False
        assert generate_calls[0]["pad_token_id"] == 31


@pytest.mark.parametrize("prompt_length", [2, 3, 5])
@pytest.mark.parametrize("stop_at_prefill", [False, True])
def test_hmonnx_host_chunks_feedback_and_eos(prompt_length, stop_at_prefill):
    from examples_merak.llm.qwen3 import qwen3_xh_hmonnx_forward as module

    class SessionDouble:
        def __init__(self):
            self.calls, self.transitions = [], []
        def set_prefill(self):
            self.phase = "prefill"
            self.transitions.append(self.phase)
        def set_decode(self):
            self.phase = "decode"
            self.transitions.append(self.phase)
        def get_input_sequence_length(self):
            return 3
        def get_data_preprocessor(self):
            return lambda data: (data,)
        def forward(self, data):
            self.calls.append((self.phase, data["input_ids"].tolist(), data["past_seq_length"]))
            token = 4 if self.phase == "prefill" else 7
            logits = torch.zeros(1, 1, 8)
            logits[..., token] = 1
            return logits
    model = SessionDouble()
    ids = torch.arange(prompt_length).unsqueeze(0)
    steps = []
    result = module._generate_hmonnx_step_by_step(model, ids, max_new_tokens=5,
        eos_token_ids={4 if stop_at_prefill else 7}, step_callback=steps.append)
    expected = [("prefill", ids[:, start:start + 3].tolist(), start) for start in range(0, prompt_length, 3)]
    if not stop_at_prefill:
        expected.append(("decode", [[4]], prompt_length))
    assert model.calls == expected
    assert result == ([4] if stop_at_prefill else [4, 7])
    assert [step.output_token_id for step in steps] == result
    assert model.transitions == (["prefill"] if stop_at_prefill else ["prefill", "decode"])
    model.calls.clear()
    assert module._generate_hmonnx_step_by_step(model, ids, max_new_tokens=0, eos_token_ids=set()) == []
    assert model.calls == []
    with pytest.raises(ValueError, match="at least one"):
        module._generate_hmonnx_step_by_step(model, ids[:, :0], max_new_tokens=1, eos_token_ids=set())
