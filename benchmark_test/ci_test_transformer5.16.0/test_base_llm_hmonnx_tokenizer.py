import pytest

from xhmodel_merak.xh_llm.hmonnx import base_llm_hmonnx_model


def _tokenizer_owner():
    owner = object.__new__(base_llm_hmonnx_model.BaseLLMHMONNXModel)
    owner.hf_model_dir = "/tmp/exported-hf-config"
    return owner


def test_exported_tokenizer_trusts_packaged_remote_code_by_default(monkeypatch):
    calls = []
    sentinel = object()
    monkeypatch.setattr(
        base_llm_hmonnx_model.AutoTokenizer,
        "from_pretrained",
        lambda path, **kwargs: calls.append((path, kwargs)) or sentinel,
    )

    assert _tokenizer_owner().get_tokenizer() is sentinel
    assert calls == [
        ("/tmp/exported-hf-config", {"trust_remote_code": True}),
    ]


def test_exported_tokenizer_preserves_explicit_remote_code_opt_out(monkeypatch):
    calls = []
    monkeypatch.setattr(
        base_llm_hmonnx_model.AutoTokenizer,
        "from_pretrained",
        lambda path, **kwargs: calls.append((path, kwargs)),
    )

    _tokenizer_owner().get_tokenizer(trust_remote_code=False)
    assert calls == [
        ("/tmp/exported-hf-config", {"trust_remote_code": False}),
    ]


def test_exported_tokenizer_falls_back_for_new_generic_backend(monkeypatch):
    calls = []
    sentinel = object()

    def missing_backend(*args, **kwargs):
        raise ValueError("Tokenizer class TokenizersBackend does not exist or is not currently imported.")

    monkeypatch.setattr(base_llm_hmonnx_model.AutoTokenizer, "from_pretrained", missing_backend)
    monkeypatch.setattr(
        base_llm_hmonnx_model.PreTrainedTokenizerFast,
        "from_pretrained",
        lambda path, **kwargs: calls.append((path, kwargs)) or sentinel,
    )

    assert _tokenizer_owner().get_tokenizer() is sentinel
    assert calls == [
        ("/tmp/exported-hf-config", {"trust_remote_code": True}),
    ]


def test_exported_tokenizer_does_not_mask_other_loader_errors(monkeypatch):
    def invalid_tokenizer(*args, **kwargs):
        raise ValueError("invalid tokenizer data")

    monkeypatch.setattr(base_llm_hmonnx_model.AutoTokenizer, "from_pretrained", invalid_tokenizer)

    with pytest.raises(ValueError, match="invalid tokenizer data"):
        _tokenizer_owner().get_tokenizer()
