from __future__ import annotations

from types import SimpleNamespace

import onnx
import pytest
from onnx import TensorProto, helper

from vllm_merak.model_executor.model_loader.qwen_hybrid_cache_contract import (
    QwenHybridCacheContract,
)


class _Session:
    def __init__(self, inputs: list[str], outputs: list[str]) -> None:
        self._inputs = inputs
        self._outputs = outputs

    def get_input_names(self) -> list[str]:
        return list(self._inputs)

    def get_output_names(self) -> list[str]:
        return list(self._outputs)


class _DeferredGraph:
    def __init__(self, onnx_file: str) -> None:
        self._onnx_file = onnx_file
        self._session = None

    def get_input_names(self) -> list[str]:
        return self._session.get_input_names()

    def get_output_names(self) -> list[str]:
        return self._session.get_output_names()


def _save_io_only_graph(path, inputs: list[str], outputs: list[str]) -> None:
    def value(name: str):
        return helper.make_tensor_value_info(name, TensorProto.FLOAT, [1])

    graph = helper.make_graph(
        [],
        path.stem,
        [value(name) for name in inputs],
        [value(name) for name in outputs],
    )
    onnx.save(helper.make_model(graph), path)


def _metadata(
    *,
    model_type: str = "Qwen3_5ForConditionalGeneration",
    mode: str | None = None,
    num_draft_tokens: int = 2,
) -> dict:
    metadata: dict = {
        "model_config": {
            "model_type": model_type,
            "fuse_gdr_ops": True,
            "prefill_recurrent_state_uses_cache": True,
        },
        "kv_cache": {
            "linear_kv_cache_config": {
                "num_layers": 2,
            }
        },
    }
    if mode is not None:
        metadata["spec_decode"] = {
            "mode": mode,
            "num_draft_tokens": num_draft_tokens,
        }
    return metadata


def _runtime(*, verify_steps: int) -> SimpleNamespace:
    recurrent_inputs = [f"past_recurrent_state_{layer}" for layer in range(2)]
    recurrent_outputs = [
        (
            f"recurrent_state_out_{layer}"
            if verify_steps == 1
            else f"recurrent_state_out_{layer}_{step}"
        )
        for layer in range(2)
        for step in range(verify_steps)
    ]
    return SimpleNamespace(
        prefill_model=SimpleNamespace(
            hmonnx_session=_Session(
                ["input_embeddings", *recurrent_inputs],
                ["logits", "conv_cache_out_q_0"],
            )
        ),
        decode_model=SimpleNamespace(
            hmonnx_session=_Session(
                ["input_embeddings", *recurrent_inputs],
                ["logits", *recurrent_outputs],
            )
        ),
        _prefill_recurrent_state_uses_cache=lambda: True,
        get_kvcache_mixin=lambda: SimpleNamespace(
            past_recurrent_states=[object(), object()]
        ),
    )


@pytest.mark.parametrize(
    ("model_type", "mode", "verify_steps"),
    [
        ("Qwen3_5ForConditionalGeneration", None, 1),
        ("Qwen3_5MoeForConditionalGeneration", "mtp", 3),
        ("Qwen3_6ForConditionalGeneration", "dflash", 3),
        ("Qwen3_6MoeForConditionalGeneration", "mtp", 3),
    ],
)
def test_fused_recurrent_contract_accepts_base_mtp_and_dflash_target_graphs(
    model_type: str,
    mode: str | None,
    verify_steps: int,
) -> None:
    contract = QwenHybridCacheContract.from_metadata(
        _metadata(model_type=model_type, mode=mode)
    )
    runtime = _runtime(verify_steps=verify_steps)

    contract.validate_graphs(runtime)
    contract.validate_prepared_runtime_state(runtime)

    assert contract.prefill_recurrent_state_uses_cache is True
    assert contract.linear_layers == 2
    assert contract.verify_steps == verify_steps


def test_fused_recurrent_contract_reads_lazy_hmonnx_graph_io(tmp_path) -> None:
    recurrent_inputs = ["past_recurrent_state_0", "past_recurrent_state_1"]
    recurrent_outputs = [
        f"recurrent_state_out_{layer}_{step}" for layer in range(2) for step in range(3)
    ]
    prefill_path = tmp_path / "prefill.onnx"
    decode_path = tmp_path / "decode.onnx"
    _save_io_only_graph(
        prefill_path,
        ["input_embeddings", *recurrent_inputs],
        ["logits"],
    )
    _save_io_only_graph(
        decode_path,
        ["input_embeddings", *recurrent_inputs],
        ["logits", *recurrent_outputs],
    )
    runtime = SimpleNamespace(
        prefill_model=SimpleNamespace(hmonnx_session=_DeferredGraph(str(prefill_path))),
        decode_model=SimpleNamespace(hmonnx_session=_DeferredGraph(str(decode_path))),
        _prefill_recurrent_state_uses_cache=lambda: True,
    )

    contract = QwenHybridCacheContract.from_metadata(_metadata(mode="dflash"))
    contract.validate_graphs(runtime)


def test_legacy_output_returning_contract_does_not_require_runtime_capability() -> None:
    metadata = _metadata()
    del metadata["model_config"]["prefill_recurrent_state_uses_cache"]
    contract = QwenHybridCacheContract.from_metadata(metadata)

    contract.validate_graphs(object())
    contract.validate_prepared_runtime_state(object())

    assert contract.prefill_recurrent_state_uses_cache is False


@pytest.mark.parametrize("invalid", ["true", 1, None])
def test_fused_recurrent_contract_rejects_non_boolean_capability(
    invalid: object,
) -> None:
    metadata = _metadata()
    metadata["model_config"]["prefill_recurrent_state_uses_cache"] = invalid

    with pytest.raises(ValueError, match="must be a boolean"):
        QwenHybridCacheContract.from_metadata(metadata)


def test_fused_recurrent_contract_rejects_prefill_state_output() -> None:
    contract = QwenHybridCacheContract.from_metadata(_metadata())
    runtime = _runtime(verify_steps=1)
    runtime.prefill_model.hmonnx_session._outputs.append("recurrent_state_out_0")

    with pytest.raises(RuntimeError, match="must not export"):
        contract.validate_graphs(runtime)


@pytest.mark.parametrize("mode", ["mtp", "dflash"])
def test_fused_recurrent_contract_requires_every_verify_snapshot(
    mode: str,
) -> None:
    contract = QwenHybridCacheContract.from_metadata(_metadata(mode=mode))
    runtime = _runtime(verify_steps=3)
    runtime.decode_model.hmonnx_session._outputs.remove("recurrent_state_out_1_2")

    with pytest.raises(
        RuntimeError,
        match=r"missing=\[\(1, 2\)\]",
    ):
        contract.validate_graphs(runtime)


def test_fused_recurrent_contract_requires_prepared_state_per_layer() -> None:
    contract = QwenHybridCacheContract.from_metadata(_metadata())
    runtime = _runtime(verify_steps=1)
    runtime.get_kvcache_mixin = lambda: SimpleNamespace(
        past_recurrent_states=[object()]
    )

    with pytest.raises(RuntimeError, match="state count mismatch"):
        contract.validate_prepared_runtime_state(runtime)
