import pytest
import torch

from xhmodel_merak.xh_llm.models.wemm_embedding.processor import (
    validate_embedding_pooling_anchor,
)


class _Tokenizer:
    def encode(self, text, *, add_special_tokens):
        assert text == "<embedding>"
        assert add_special_tokens is False
        return [99]


def test_processor_contract_accepts_embedding_as_last_valid_token() -> None:
    inputs = {
        "input_ids": torch.tensor([[0, 1, 99, 0], [4, 5, 6, 99]]),
        "attention_mask": torch.tensor([[0, 1, 1, 0], [1, 1, 1, 1]]),
    }

    validate_embedding_pooling_anchor(inputs, _Tokenizer())


def test_processor_contract_rejects_pooling_on_im_end() -> None:
    inputs = {
        "input_ids": torch.tensor([[1, 2, 98]]),
        "attention_mask": torch.ones((1, 3), dtype=torch.long),
    }

    with pytest.raises(ValueError, match="must end with the <embedding>"):
        validate_embedding_pooling_anchor(inputs, _Tokenizer())


def test_processor_contract_rejects_duplicate_embedding_tokens() -> None:
    inputs = {
        "input_ids": torch.tensor([[1, 99, 2, 99]]),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
    }

    with pytest.raises(ValueError, match="exactly one <embedding>"):
        validate_embedding_pooling_anchor(inputs, _Tokenizer())
