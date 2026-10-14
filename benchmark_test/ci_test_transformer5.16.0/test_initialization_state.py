"""Checkpoint loading must not disable later native-model initialization."""

import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest
import torch
import transformers.modeling_utils as modeling_utils


def test_autouse_fixture_isolates_success_and_failure(tmp_path):
    source = Path(__file__).resolve().parent
    (tmp_path / "conftest.py").write_bytes((source / "conftest.py").read_bytes())
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_sequence.py").write_text('''
import torch
import transformers.modeling_utils as modeling_utils

NAMES = ("kaiming_uniform_", "uniform_", "normal_")
ORIGINAL = {name: getattr(torch.nn.init, name) for name in NAMES}
MISSING = object()
FLAG = getattr(modeling_utils, "_init_weights", MISSING)

def assert_restored():
    assert all(getattr(torch.nn.init, name) is value for name, value in ORIGINAL.items())
    assert getattr(modeling_utils, "_init_weights", MISSING) is FLAG

def disable():
    for name in NAMES:
        setattr(torch.nn.init, name, lambda *args, **kwargs: None)
    modeling_utils._init_weights = False

def test_01_success():
    disable()

def test_02_failure():
    assert_restored()
    disable()
    raise RuntimeError("intentional loader failure")

def test_03_next_native_model():
    assert_restored()
    model = torch.nn.Linear(64, 192)
    assert torch.isfinite(model.weight).all()
    assert torch.isfinite(model.bias).all()
''')
    env = os.environ.copy()
    # The copied conftest imports shared CI helpers from its original directory.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(source), *filter(None, env.get("PYTHONPATH", "").split(os.pathsep))]
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-c", str(tmp_path / "pytest.ini"),
         "--confcutdir", str(tmp_path), str(tmp_path / "test_sequence.py"),
         "--junitxml", str(tmp_path / "result.xml"), "-q"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    cases = list(ET.parse(tmp_path / "result.xml").iter("testcase"))
    assert len(cases) == 3, result.stdout + result.stderr
    assert [case.get("name") for case in cases if case.find("failure") is not None] == ["test_02_failure"]
    assert all(case.find("error") is None and case.find("skipped") is None for case in cases)
    failure = cases[1].find("failure")
    assert "intentional loader failure" in failure.get("message", "")


@pytest.mark.parametrize("flag", [None, True, False])
@pytest.mark.parametrize("fails", [False, True])
def test_initializers_restored_on_exit(monkeypatch, flag, fails, restore_model_initialization_state):
    if flag is None:
        monkeypatch.delattr(modeling_utils, "_init_weights", raising=False)
    else:
        monkeypatch.setattr(modeling_utils, "_init_weights", flag, raising=False)
    originals = {name: getattr(torch.nn.init, name)
                 for name in ("kaiming_uniform_", "uniform_", "normal_")}

    def skip(*args, **kwargs):
        pass

    class LoadFailure(Exception):
        pass

    def load():
        with restore_model_initialization_state():
            for name in originals:
                setattr(torch.nn.init, name, skip)
            modeling_utils._init_weights = False
            # The loader still gets its no-initialization behavior in scope.
            value = torch.full((4,), 17.0)
            torch.nn.init.uniform_(value)
            torch.testing.assert_close(value, torch.full((4,), 17.0))
            if fails:
                raise LoadFailure("checkpoint rejected")

    if fails:
        with pytest.raises(LoadFailure, match="checkpoint rejected"):
            load()
    else:
        load()
    for name, original in originals.items():
        assert getattr(torch.nn.init, name) is original
    if flag is None:
        assert not hasattr(modeling_utils, "_init_weights")
    else:
        assert modeling_utils._init_weights is flag


def test_real_gptq_load_then_native_model(tmp_path, restore_model_initialization_state):
    from gptqmodel import GPTQModel, QuantizeConfig
    from transformers import Qwen2Config, Qwen2ForCausalLM, Qwen2Tokenizer

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(4891)
        expected = torch.nn.Linear(64, 192)
        config = Qwen2Config(
            vocab_size=128, hidden_size=64, intermediate_size=128,
            num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=128,
        )
        Qwen2ForCausalLM(config).save_pretrained(tmp_path)
        Qwen2Tokenizer(vocab={"<|endoftext|>": 0, "t": 1, "Ġ": 2}, merges=[]).save_pretrained(tmp_path)
        with restore_model_initialization_state():
            loaded = GPTQModel.load(str(tmp_path), QuantizeConfig(
                bits=4, group_size=64, sym=True, desc_act=False,
                device="cpu", offload_to_disk=False,
            ))
            assert loaded.model.config.hidden_size == 64
        torch.manual_seed(4891)
        actual = torch.nn.Linear(64, 192)
        torch.testing.assert_close(actual.weight, expected.weight, rtol=0, atol=0)
        torch.testing.assert_close(actual.bias, expected.bias, rtol=0, atol=0)
