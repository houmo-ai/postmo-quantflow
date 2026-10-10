"""Checkpoint remote-code defaults and explicit opt-out across loading paths."""

from types import SimpleNamespace

import pytest
import torch

from xhmodel_merak.xh_llm.base_model import XHBaseModel


@pytest.mark.parametrize("trust", [None, False, True])
def test_remote_code_setting_reaches_config_and_model(monkeypatch, trust):
    # Only replace the HF I/O boundary, exercising both ModelZoo loaders.
    from xhmodel_merak.xh_llm import base_model

    calls = []

    def config_loader(path, **kwargs):
        calls.append(kwargs["trust_remote_code"])
        return SimpleNamespace(quantization_config=None)

    def model_loader(path, **kwargs):
        calls.append(kwargs["trust_remote_code"])
        model = torch.nn.Linear(2, 2)
        model.config = SimpleNamespace(tie_word_embeddings=False)
        return model

    class Model(XHBaseModel):
        HF_AUTO_MODEL_CLS = SimpleNamespace(from_pretrained=model_loader)

        @classmethod
        def untied_weights(cls, model):
            return model

    monkeypatch.setattr(base_model.AutoConfig, "from_pretrained", config_loader)
    kwargs = {} if trust is None else {"trust_remote_code": trust}
    Model.get_hf_model("unused", **kwargs)
    assert calls == [trust is not False, trust is not False]



@pytest.mark.parametrize("trust", [None, False, True])
def test_remote_code_setting_reaches_gptqmodel(monkeypatch, trust):
    from gptqmodel import GPTQModel

    calls = []
    model = torch.nn.Linear(2, 2)
    model.config = SimpleNamespace(quantization_config=None)

    def loader(path, **kwargs):
        calls.append(kwargs["trust_remote_code"])
        return SimpleNamespace(model=model)

    # Replace only the external weight-loading boundary.
    monkeypatch.setattr(GPTQModel, "load", loader)
    kwargs = {} if trust is None else {"trust_remote_code": trust}
    assert XHBaseModel._load_gptqmodel("unused", **kwargs) is model
    assert calls == [trust is not False]



@pytest.mark.parametrize("trust", [None, False, True])
@pytest.mark.parametrize("entry", ["native", "empty"])
def test_checkpoint_auto_map_is_trusted_by_default(tmp_path, trust, entry):
    from transformers import AutoModelForCausalLM, GPT2Config, GPT2LMHeadModel

    config = GPT2Config(n_layer=1, n_head=1, n_embd=8, vocab_size=16, n_positions=8, bos_token_id=0, eos_token_id=1)
    model = GPT2LMHeadModel(config)
    model.config.auto_map = {
        "AutoConfig": "custom_config.CustomConfig",
        "AutoModelForCausalLM": "custom_model.CustomModel",
    }
    model.save_pretrained(tmp_path)
    marker = tmp_path / "remote_config_executed"
    (tmp_path / "custom_config.py").write_text(
        "from pathlib import Path\n"
        "from transformers import GPT2Config\n"
        f"Path({str(marker)!r}).touch()\n"
        "class CustomConfig(GPT2Config):\n"
        "    pass\n"
    )
    (tmp_path / "custom_model.py").write_text(
        "from transformers import GPT2LMHeadModel\n"
        "from .custom_config import CustomConfig\n"
        "class CustomModel(GPT2LMHeadModel):\n"
        "    config_class = CustomConfig\n"
        "    remote_implementation = True\n"
    )

    class Model(XHBaseModel):
        HF_AUTO_MODEL_CLS = AutoModelForCausalLM

    kwargs = {} if trust is None else {"trust_remote_code": trust}
    loader = Model.get_hf_model if entry == "native" else Model.get_empty_hf_model
    loaded = loader(str(tmp_path), dtype=torch.float32, **kwargs)
    assert marker.exists() is (trust is not False)
    assert getattr(loaded, "remote_implementation", False) is (trust is not False)
    if entry == "empty":
        assert next(loaded.parameters()).device.type == "meta"
