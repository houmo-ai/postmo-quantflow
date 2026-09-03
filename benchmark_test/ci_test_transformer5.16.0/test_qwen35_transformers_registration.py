import json

from transformers import AutoConfig


def test_qwen35_package_registers_vendored_config(tmp_path) -> None:
    from xhmodel_merak.xh_llm.models.qwen3_5 import Qwen3_5Config

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": {"vocab_size": 32, "hidden_size": 16},
                "vision_config": {},
            }
        ),
        encoding="utf-8",
    )

    config = AutoConfig.from_pretrained(tmp_path)

    assert config.model_type == Qwen3_5Config.model_type
    assert config.text_config.vocab_size == 32
    assert config.text_config.hidden_size == 16
