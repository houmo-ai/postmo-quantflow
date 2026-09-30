import json
from types import SimpleNamespace

import pytest

from vllm_merak.model_executor.model_loader.hf_assets import (
    get_model_trust_remote_code,
    load_hf_config,
    load_hf_pretrained,
)


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_load_hf_pretrained_forwards_remote_code_policy(
    tmp_path,
    trust_remote_code: bool,
) -> None:
    hf_assets_dir = tmp_path / "hf_config"
    hf_assets_dir.mkdir()
    calls = []

    class FakeLoader:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            calls.append((model_id, kwargs))
            return cls()

    loaded = load_hf_pretrained(
        FakeLoader,
        tmp_path,
        trust_remote_code=trust_remote_code,
        use_fast=False,
    )

    assert isinstance(loaded, FakeLoader)
    assert calls == [
        (
            str(hf_assets_dir),
            {
                "trust_remote_code": trust_remote_code,
                "use_fast": False,
            },
        )
    ]


@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_model_config_remote_code_policy_is_not_reinterpreted(
    trust_remote_code: bool,
) -> None:
    model_config = SimpleNamespace(trust_remote_code=trust_remote_code)

    assert get_model_trust_remote_code(model_config) is trust_remote_code


def test_model_config_remote_code_policy_must_be_boolean() -> None:
    with pytest.raises(TypeError, match="must be bool"):
        get_model_trust_remote_code(SimpleNamespace(trust_remote_code=1))


def _write_remote_code_config(model_dir) -> None:
    (model_dir / "configuration_merak_remote.py").write_text(
        """
from transformers import PretrainedConfig


class MerakRemotePolicyConfig(PretrainedConfig):
    model_type = "merak_remote_policy_test"

    def __init__(self, marker="unset", **kwargs):
        super().__init__(**kwargs)
        self.marker = marker
""".lstrip(),
        encoding="utf-8",
    )
    (model_dir / "config.json").write_text(
        json.dumps(
            {
                "auto_map": {
                    "AutoConfig": ("configuration_merak_remote.MerakRemotePolicyConfig")
                },
                "model_type": "merak_remote_policy_test",
                "marker": "loaded-from-custom-code",
            }
        ),
        encoding="utf-8",
    )


def test_custom_remote_config_respects_false_without_prompt(tmp_path) -> None:
    _write_remote_code_config(tmp_path)

    with pytest.raises(ValueError, match="trust_remote_code=True"):
        load_hf_config(tmp_path, trust_remote_code=False)


def test_custom_remote_config_loads_when_explicitly_trusted(tmp_path) -> None:
    _write_remote_code_config(tmp_path)

    config = load_hf_config(tmp_path, trust_remote_code=True)

    assert type(config).__name__ == "MerakRemotePolicyConfig"
    assert config.marker == "loaded-from-custom-code"
