from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from vllm_merak.spec_decode.artifact import (
    HMONNXSpecDecodeArtifact,
    SpecDecodeDraftGraphs,
)
from vllm_merak.spec_decode.controller import HMONNXSpecDecodeController


class FakeBackend:
    def __init__(self) -> None:
        self.mtp_appends: list[dict[str, Any]] = []
        self.dflash_appends: list[dict[str, Any]] = []
        self.mtp_drafts: list[dict[str, Any]] = []
        self.mtp_commits: list[dict[str, Any]] = []
        self.dflash_drafts: list[dict[str, Any]] = []
        self.gemma_drafts: list[dict[str, Any]] = []
        self.released: list[str] = []
        self.dflash_can_draft = True

    def append_mtp_context(self, **kwargs: Any) -> None:
        self.mtp_appends.append(kwargs)

    def append_dflash_context(self, **kwargs: Any) -> None:
        self.dflash_appends.append(kwargs)

    def draft_mtp(self, **kwargs: Any) -> list[int]:
        self.mtp_drafts.append(kwargs)
        return [101, 102, 103, 104][: kwargs["num_draft_tokens"]]

    def commit_mtp_draft(self, **kwargs: Any) -> None:
        self.mtp_commits.append(kwargs)

    def draft_dflash(self, **kwargs: Any) -> list[int]:
        self.dflash_drafts.append(kwargs)
        return list(range(201, 201 + kwargs["num_draft_tokens"]))

    def can_draft_dflash(self, **kwargs: Any) -> bool:
        return self.dflash_can_draft

    def draft_gemma4(self, **kwargs: Any) -> list[int]:
        self.gemma_drafts.append(kwargs)
        return [301, 302, 303, 304][: kwargs["num_draft_tokens"]]

    def release_request(self, req_id: str) -> None:
        self.released.append(req_id)


def _artifact(
    *,
    mode: str,
    abi: str,
    num_drafts: int,
) -> HMONNXSpecDecodeArtifact:
    return HMONNXSpecDecodeArtifact(
        artifact_dir=Path("/artifact"),
        metadata_path=Path("/artifact/golden_meta_info.json"),
        runtime_contract_version=2,
        mode=mode,  # type: ignore[arg-type]
        num_draft_tokens=num_drafts,
        verify_length=num_drafts + 1,
        target_hidden_output_name="target_hidden",
        target_linear_state_outputs=None,
        draft_abi=abi,
        draft_graphs=SpecDecodeDraftGraphs(),
        draft_cache_mutation=("read_only" if abi.startswith("gemma4_") else "in_place"),
        draft_cache_binding=(
            "target_attention_type_owner" if abi.startswith("gemma4_") else None
        ),
        draft_constant_positions=abi.startswith("gemma4_"),
        draft_position_semantics=(
            "target_kv_valid_length" if abi.startswith("gemma4_") else None
        ),
        draft_layer_attention_types=(
            ("sliding_attention", "full_attention") if abi.startswith("gemma4_") else ()
        ),
        minimum_sliding_cache_slack=(num_drafts if abi.startswith("gemma4_") else None),
        dflash_noise_token_id=248070 if mode == "dflash" else None,
    )


def _result(hidden: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(hidden_states=hidden)


def test_qwen_mtp_shifted_context_and_reject_rollback() -> None:
    backend = FakeBackend()
    controller = HMONNXSpecDecodeController(
        _artifact(mode="mtp", abi="qwen_mtp_v1", num_drafts=4),
        backend,
    )
    prompt_hidden = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
    prompt_embeddings = torch.tensor([[101.0, 1001.0], [202.0, 2002.0]])
    controller.on_target_forward(
        req_id="r",
        phase="prefill",
        past_len=0,
        input_ids=torch.tensor([10, 11]),
        input_embeddings=prompt_embeddings,
        result=_result(prompt_hidden),
    )

    assert len(backend.mtp_appends) == 1
    append = backend.mtp_appends[0]
    assert append["past_length"] == 0
    torch.testing.assert_close(append["hidden_states"], prompt_hidden[:1])
    torch.testing.assert_close(append["next_token_ids"], torch.tensor([11]))
    torch.testing.assert_close(
        append["next_token_embeddings"],
        prompt_embeddings[1:],
    )

    assert controller.propose_one(req_id="r", current_token_id=12) == [
        101,
        102,
        103,
        104,
    ]
    first_draft = backend.mtp_drafts[-1]
    assert first_draft["past_length"] == 1
    torch.testing.assert_close(
        first_draft["target_hidden"],
        prompt_hidden[1:],
    )

    verify_hidden = torch.tensor(
        [
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
            [6.0, 60.0],
            [7.0, 70.0],
        ]
    )
    controller.on_target_forward(
        req_id="r",
        phase="verify",
        past_len=2,
        input_ids=torch.tensor([12, 101, 102, 103, 104]),
        result=_result(verify_hidden),
    )
    # One accepted draft plus the recovery token means two target steps are
    # committed. The selected hidden is verify index accepted_steps - 1.
    controller.on_target_accept(
        req_id="r",
        accepted_steps=2,
        sampled_token_ids=torch.tensor([101, 999, -1, -1, -1]),
    )
    assert backend.mtp_commits == [{"req_id": "r", "accepted_steps": 2}]

    controller.propose_one(req_id="r", current_token_id=999)
    second_draft = backend.mtp_drafts[-1]
    assert second_draft["past_length"] == 3
    torch.testing.assert_close(
        second_draft["target_hidden"],
        verify_hidden[1:2],
    )


@pytest.mark.parametrize(
    ("mode", "abi", "expected_kind"),
    [
        ("mtp", "qwen_mtp_paged_v2", "mtp"),
        ("dflash", "qwen_dflash_paged_v2", "dflash"),
    ],
)
def test_qwen_paged_abis_use_same_controller_semantics(
    mode: str,
    abi: str,
    expected_kind: str,
) -> None:
    backend = FakeBackend()
    num_drafts = 4 if mode == "mtp" else 9
    controller = HMONNXSpecDecodeController(
        _artifact(mode=mode, abi=abi, num_drafts=num_drafts),
        backend,
    )
    controller.on_target_forward(
        req_id="r",
        phase="prefill",
        past_len=0,
        input_ids=torch.tensor([10, 11]),
        result=_result(torch.ones(2, 3)),
    )
    drafts = controller.propose_one(req_id="r", current_token_id=12)

    if expected_kind == "mtp":
        assert len(drafts) == 4
        assert backend.mtp_appends
        assert backend.mtp_drafts[-1]["past_length"] == 1
        assert not backend.dflash_drafts
    else:
        assert len(drafts) == 9
        assert backend.dflash_appends
        assert backend.dflash_drafts[-1]["past_length"] == 2
        assert not backend.mtp_drafts


def test_qwen_mtp_zero_draft_acceptance_commits_recovery_step() -> None:
    backend = FakeBackend()
    controller = HMONNXSpecDecodeController(
        _artifact(mode="mtp", abi="qwen_mtp_v1", num_drafts=4),
        backend,
    )
    controller.on_target_forward(
        req_id="r",
        phase="prefill",
        past_len=0,
        input_ids=torch.tensor([10, 11]),
        result=_result(torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
    )
    drafts = controller.propose_one(req_id="r", current_token_id=12)
    verify_hidden = torch.arange(10, dtype=torch.float32).reshape(5, 2)
    controller.on_target_forward(
        req_id="r",
        phase="verify",
        past_len=2,
        input_ids=torch.tensor([12, *drafts]),
        result=_result(verify_hidden),
    )
    controller.on_target_accept(
        req_id="r",
        accepted_steps=1,
        sampled_token_ids=torch.tensor([999, -1, -1, -1, -1]),
    )

    # The next draft overwrites the first rejected speculative slot with the
    # target recovery pair hidden(12) -> 999.  No full-accept tail append is
    # needed for this path.
    controller.propose_one(req_id="r", current_token_id=999)
    assert len(backend.mtp_appends) == 1
    assert backend.mtp_drafts[-1]["past_length"] == 2
    torch.testing.assert_close(
        backend.mtp_drafts[-1]["target_hidden"],
        verify_hidden[:1],
    )


def test_qwen_mtp_prefill_chunk_boundary_pairs_pending_hidden() -> None:
    backend = FakeBackend()
    controller = HMONNXSpecDecodeController(
        _artifact(mode="mtp", abi="qwen_mtp_v1", num_drafts=4),
        backend,
    )
    controller.on_target_forward(
        req_id="r",
        phase="prefill",
        past_len=0,
        input_ids=torch.tensor([10]),
        result=_result(torch.tensor([[1.0, 2.0]])),
    )
    assert backend.mtp_appends == []

    controller.on_target_forward(
        req_id="r",
        phase="prefill",
        past_len=1,
        input_ids=torch.tensor([11, 12]),
        result=_result(torch.tensor([[3.0, 4.0], [5.0, 6.0]])),
    )
    append = backend.mtp_appends[-1]
    assert append["past_length"] == 0
    torch.testing.assert_close(
        append["hidden_states"],
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    torch.testing.assert_close(
        append["next_token_ids"],
        torch.tensor([11, 12]),
    )


def test_qwen_mtp_all_accepted_materializes_missing_shifted_pair() -> None:
    backend = FakeBackend()
    controller = HMONNXSpecDecodeController(
        _artifact(mode="mtp", abi="qwen_mtp_v1", num_drafts=4),
        backend,
    )
    controller.on_target_forward(
        req_id="r",
        phase="prefill",
        past_len=0,
        input_ids=torch.tensor([10, 11]),
        result=_result(torch.tensor([[1.0, 2.0], [3.0, 4.0]])),
    )
    drafts = controller.propose_one(req_id="r", current_token_id=12)
    verify_hidden = torch.tensor(
        [
            [10.0, 11.0],
            [20.0, 21.0],
            [30.0, 31.0],
            [40.0, 41.0],
            [50.0, 51.0],
        ]
    )
    controller.on_target_forward(
        req_id="r",
        phase="verify",
        past_len=2,
        input_ids=torch.tensor([12, *drafts]),
        result=_result(verify_hidden),
    )
    controller.on_target_accept(
        req_id="r",
        accepted_steps=5,
        sampled_token_ids=torch.tensor([*drafts, 999]),
    )
    assert backend.mtp_commits == [{"req_id": "r", "accepted_steps": 5}]

    missing_pair = backend.mtp_appends[-1]
    assert missing_pair["past_length"] == 5
    torch.testing.assert_close(
        missing_pair["hidden_states"],
        verify_hidden[3:4],
    )
    torch.testing.assert_close(
        missing_pair["next_token_ids"],
        torch.tensor([104]),
    )
    controller.propose_one(req_id="r", current_token_id=999)
    assert backend.mtp_drafts[-1]["past_length"] == 6
    torch.testing.assert_close(
        backend.mtp_drafts[-1]["target_hidden"],
        verify_hidden[4:5],
    )


def test_qwen_dflash_appends_only_accepted_target_hidden_in_place() -> None:
    backend = FakeBackend()
    controller = HMONNXSpecDecodeController(
        _artifact(mode="dflash", abi="qwen_dflash_v1", num_drafts=9),
        backend,
    )
    prompt_hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    controller.on_target_forward(
        req_id="r",
        phase="prefill",
        past_len=0,
        input_ids=torch.tensor([1, 2, 3]),
        result=_result(prompt_hidden),
    )
    assert backend.dflash_appends[-1]["past_length"] == 0
    assert backend.dflash_appends[-1]["decode_step"] is False
    torch.testing.assert_close(
        backend.dflash_appends[-1]["hidden_states"],
        prompt_hidden,
    )

    drafts = controller.propose_one(req_id="r", current_token_id=4)
    assert drafts == list(range(201, 210))
    assert backend.dflash_drafts[-1]["past_length"] == 3

    verify_hidden = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    controller.on_target_forward(
        req_id="r",
        phase="verify",
        past_len=3,
        input_ids=torch.tensor([4, *drafts]),
        result=_result(verify_hidden),
    )
    controller.on_target_accept(
        req_id="r",
        accepted_steps=3,
        sampled_token_ids=torch.tensor([201, 202, 999, *([-1] * 7)]),
    )
    accepted_append = backend.dflash_appends[-1]
    assert accepted_append["past_length"] == 3
    assert accepted_append["decode_step"] is True
    torch.testing.assert_close(
        accepted_append["hidden_states"],
        verify_hidden[:3],
    )

    controller.propose_one(req_id="r", current_token_id=999)
    assert backend.dflash_drafts[-1]["past_length"] == 6


def test_qwen_dflash_zero_draft_acceptance_appends_only_recovery_hidden() -> None:
    backend = FakeBackend()
    controller = HMONNXSpecDecodeController(
        _artifact(mode="dflash", abi="qwen_dflash_v1", num_drafts=9),
        backend,
    )
    controller.on_target_forward(
        req_id="r",
        phase="prefill",
        past_len=0,
        input_ids=torch.tensor([1, 2, 3]),
        result=_result(torch.arange(12, dtype=torch.float32).reshape(3, 4)),
    )
    drafts = controller.propose_one(req_id="r", current_token_id=4)
    verify_hidden = torch.arange(40, dtype=torch.float32).reshape(10, 4)
    controller.on_target_forward(
        req_id="r",
        phase="verify",
        past_len=3,
        input_ids=torch.tensor([4, *drafts]),
        result=_result(verify_hidden),
    )
    controller.on_target_accept(
        req_id="r",
        accepted_steps=1,
        sampled_token_ids=torch.tensor([999, *([-1] * 9)]),
    )

    accepted_append = backend.dflash_appends[-1]
    assert accepted_append["past_length"] == 3
    assert accepted_append["decode_step"] is True
    torch.testing.assert_close(
        accepted_append["hidden_states"],
        verify_hidden[:1],
    )
    controller.propose_one(req_id="r", current_token_id=999)
    assert backend.dflash_drafts[-1]["past_length"] == 4


def test_qwen_dflash_runtime_width_can_use_exported_width_prefix() -> None:
    backend = FakeBackend()
    artifact = _artifact(
        mode="dflash",
        abi="qwen_dflash_paged_shared_v3",
        num_drafts=15,
    )
    controller = HMONNXSpecDecodeController(
        artifact,
        backend,
        num_draft_tokens=9,
    )
    controller.on_target_forward(
        req_id="r",
        phase="prefill",
        past_len=0,
        input_ids=torch.tensor([1, 2]),
        result=_result(torch.ones(2, 4)),
    )

    assert controller.propose_one(req_id="r", current_token_id=3) == list(
        range(201, 210)
    )
    assert backend.dflash_drafts[-1]["num_draft_tokens"] == 9

    with pytest.raises(ValueError, match="within the exported HMONNX capacity"):
        HMONNXSpecDecodeController(
            artifact,
            backend,
            num_draft_tokens=16,
        )


def test_qwen_dflash_falls_back_when_full_physical_block_does_not_fit() -> None:
    backend = FakeBackend()
    controller = HMONNXSpecDecodeController(
        _artifact(mode="dflash", abi="qwen_dflash_paged_shared_v3", num_drafts=9),
        backend,
    )
    controller.on_target_forward(
        req_id="r",
        phase="prefill",
        past_len=0,
        input_ids=torch.tensor([1, 2]),
        result=_result(torch.ones(2, 4)),
    )
    backend.dflash_can_draft = False

    assert controller.propose_one(req_id="r", current_token_id=3) == []
    assert backend.dflash_drafts == []

    # Empty proposal leaves no transaction behind. The following ordinary
    # target decode can append its state and try drafting again.
    controller.on_target_forward(
        req_id="r",
        phase="decode",
        past_len=2,
        input_ids=torch.tensor([3]),
        result=_result(torch.ones(1, 4)),
    )
    backend.dflash_can_draft = True
    assert controller.propose_one(req_id="r", current_token_id=4) == list(
        range(201, 210)
    )


@pytest.mark.parametrize(
    "abi",
    [
        "gemma4_mtp_readonly_page_attention_v1",
        "gemma4_mtp_readonly_page_attention_v2",
    ],
)
def test_gemma4_uses_accepted_target_position_and_page_contexts(
    abi: str,
) -> None:
    backend = FakeBackend()
    controller = HMONNXSpecDecodeController(
        _artifact(
            mode="mtp",
            abi=abi,
            num_drafts=4,
        ),
        backend,
    )
    initial_contexts = {0: object(), 1: object()}
    controller.on_target_forward(
        req_id="r",
        phase="prefill",
        past_len=0,
        input_ids=torch.tensor([10, 11, 12]),
        # Gemma4 target prefill exports only the final target hidden state;
        # unlike Qwen, the draft reads the target PageAttention cache directly.
        result=_result(torch.arange(4).reshape(1, 4)),
        page_attention_group_contexts=initial_contexts,
    )
    drafts = controller.propose_one(req_id="r", current_token_id=13)
    first = backend.gemma_drafts[-1]
    assert first["target_visible_length"] == 3
    assert first["page_attention_group_contexts"] is initial_contexts
    torch.testing.assert_close(
        first["target_hidden"],
        torch.arange(4).reshape(1, 4),
    )

    accepted_contexts = {0: object(), 1: object()}
    verify_hidden = torch.arange(20).reshape(5, 4)
    controller.on_target_forward(
        req_id="r",
        phase="verify",
        past_len=3,
        input_ids=torch.tensor([13, *drafts]),
        result=_result(verify_hidden),
        page_attention_group_contexts=accepted_contexts,
    )
    controller.on_target_accept(
        req_id="r",
        accepted_steps=3,
        sampled_token_ids=torch.tensor([301, 302, 999, -1, -1]),
    )
    controller.propose_one(req_id="r", current_token_id=999)
    second = backend.gemma_drafts[-1]
    assert second["target_visible_length"] == 6
    assert second["page_attention_group_contexts"] is accepted_contexts
    torch.testing.assert_close(second["target_hidden"], verify_hidden[2:3])


@pytest.mark.parametrize(
    "abi",
    [
        "gemma4_mtp_readonly_page_attention_v1",
        "gemma4_mtp_readonly_page_attention_v2",
    ],
)
def test_gemma4_zero_draft_acceptance_hides_rejected_cache_tail(
    abi: str,
) -> None:
    backend = FakeBackend()
    controller = HMONNXSpecDecodeController(
        _artifact(
            mode="mtp",
            abi=abi,
            num_drafts=4,
        ),
        backend,
    )
    initial_contexts = {0: object(), 1: object()}
    controller.on_target_forward(
        req_id="r",
        phase="prefill",
        past_len=0,
        input_ids=torch.tensor([10, 11, 12]),
        result=_result(torch.arange(4).reshape(1, 4)),
        page_attention_group_contexts=initial_contexts,
    )
    drafts = controller.propose_one(req_id="r", current_token_id=13)
    verify_contexts = {0: object(), 1: object()}
    verify_hidden = torch.arange(20).reshape(5, 4)
    controller.on_target_forward(
        req_id="r",
        phase="verify",
        past_len=3,
        input_ids=torch.tensor([13, *drafts]),
        result=_result(verify_hidden),
        page_attention_group_contexts=verify_contexts,
    )
    controller.on_target_accept(
        req_id="r",
        accepted_steps=1,
        sampled_token_ids=torch.tensor([999, -1, -1, -1, -1]),
    )

    controller.propose_one(req_id="r", current_token_id=999)
    next_draft = backend.gemma_drafts[-1]
    assert next_draft["target_visible_length"] == 4
    assert next_draft["page_attention_group_contexts"] is verify_contexts
    torch.testing.assert_close(
        next_draft["target_hidden"],
        verify_hidden[:1],
    )


def test_verify_mismatch_and_release_fail_closed() -> None:
    backend = FakeBackend()
    controller = HMONNXSpecDecodeController(
        _artifact(mode="mtp", abi="qwen_mtp_v1", num_drafts=4),
        backend,
    )
    controller.on_target_forward(
        req_id="r",
        phase="prefill",
        past_len=0,
        input_ids=torch.tensor([10, 11]),
        result=_result(torch.ones(2, 3)),
    )
    controller.propose_one(req_id="r", current_token_id=12)
    with pytest.raises(RuntimeError, match="do not match"):
        controller.on_target_forward(
            req_id="r",
            phase="verify",
            past_len=2,
            input_ids=torch.tensor([12, 101, 999, 103, 104]),
            result=_result(torch.ones(5, 3)),
        )

    controller.release_stale_requests([])
    assert controller.active_request_ids == frozenset()
    assert backend.released == ["r"]


def test_custom_proposer_validates_config_and_uses_last_sampled_token(
    monkeypatch,
) -> None:
    from vllm_merak.spec_decode import proposer as proposer_module

    monkeypatch.delenv("ENABLE_HMINFERENCE_V2", raising=False)

    artifact = _artifact(
        mode="mtp",
        abi="qwen_mtp_v1",
        num_drafts=4,
    )
    monkeypatch.setattr(
        proposer_module,
        "load_hmonnx_spec_decode_artifact",
        lambda *_args, **_kwargs: artifact,
    )

    class RuntimeBackend(FakeBackend):
        def __init__(self, *, runner, artifact) -> None:
            super().__init__()
            self.runner = runner
            self.artifact = artifact

    monkeypatch.setattr(
        proposer_module,
        "HMONNXDraftRuntimeBackend",
        RuntimeBackend,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(model="/artifact"),
        speculative_config=SimpleNamespace(num_speculative_tokens=4),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False,
            mamba_cache_mode="none",
        ),
    )
    proposer = proposer_module.HMONNXSpecDecodeProposer(config)
    assert os.environ["ENABLE_HMINFERENCE_V2"] == "1"
    runner = SimpleNamespace(
        model=object(),
        input_batch=SimpleNamespace(req_ids=["r"]),
    )
    proposer.attach_merak_runner(runner)
    proposer.on_target_forward(
        req_id="r",
        phase="prefill",
        past_len=0,
        input_ids=torch.tensor([10, 11]),
        result=_result(torch.ones(2, 3)),
        page_attention_group_contexts=None,
    )

    assert proposer.propose(
        [[12, 13]],
        num_tokens_no_spec=None,
        token_ids_cpu=None,
    ) == [[101, 102, 103, 104]]
    assert proposer.backend.mtp_drafts[-1]["current_token_id"] == 13

    config.cache_config.enable_prefix_caching = True
    with pytest.raises(ValueError, match="enable_prefix_caching=False"):
        proposer_module.HMONNXSpecDecodeProposer(config)
    config.cache_config.enable_prefix_caching = False
    config.cache_config.mamba_cache_mode = "align"
    with pytest.raises(ValueError, match="mamba_cache_mode='none'"):
        proposer_module.HMONNXSpecDecodeProposer(config)

    # A max-width DFlash export is one reusable artifact for every runtime
    # verify width up to that capacity.  This lets deployment tune K without
    # re-exporting target/draft graphs.
    artifact = _artifact(
        mode="dflash",
        abi="qwen_dflash_paged_shared_v3",
        num_drafts=15,
    )
    config.speculative_config.num_speculative_tokens = 9
    config.cache_config.mamba_cache_mode = "none"
    proposer = proposer_module.HMONNXSpecDecodeProposer(config)
    assert proposer.num_draft_tokens == 9

    config.speculative_config.num_speculative_tokens = 16
    with pytest.raises(ValueError, match="exceeds the HMONNX export capacity"):
        proposer_module.HMONNXSpecDecodeProposer(config)


def test_dflash2_custom_proposer_publishes_standard_rejection_probs(
    monkeypatch,
) -> None:
    from vllm_merak.spec_decode import proposer as proposer_module

    artifact = _artifact(
        mode="dflash",
        abi="qwen_dflash2_v1",
        num_drafts=4,
    )
    monkeypatch.setattr(
        proposer_module,
        "load_hmonnx_spec_decode_artifact",
        lambda *_args, **_kwargs: artifact,
    )

    class RuntimeBackend(FakeBackend):
        def __init__(self, *, runner, artifact) -> None:
            super().__init__()
            self.runner = runner
            self.artifact = artifact

        def take_last_dflash_draft_probs(self) -> torch.Tensor:
            return torch.full((1, 4, 32), 1.0 / 32)

    monkeypatch.setattr(
        proposer_module,
        "HMONNXDraftRuntimeBackend",
        RuntimeBackend,
    )
    config = SimpleNamespace(
        model_config=SimpleNamespace(model="/artifact"),
        speculative_config=SimpleNamespace(
            num_speculative_tokens=4,
            draft_sample_method="probabilistic",
            rejection_sample_method="standard",
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=1),
        cache_config=SimpleNamespace(
            enable_prefix_caching=False,
            mamba_cache_mode="none",
        ),
    )
    proposer = proposer_module.HMONNXSpecDecodeProposer(config)
    request_generator = torch.Generator(device="cpu").manual_seed(123)
    runner = SimpleNamespace(
        model=object(),
        input_batch=SimpleNamespace(
            req_ids=["r"],
            req_id_to_index={"r": 0},
            temperature_cpu=[1.0],
            greedy_reqs=set(),
            generators={0: request_generator},
        ),
        _draft_probs=None,
        _draft_prob_req_ids=None,
    )
    proposer.attach_merak_runner(runner)
    proposer.on_target_forward(
        req_id="r",
        phase="prefill",
        past_len=0,
        input_ids=torch.tensor([10, 11]),
        result=_result(torch.ones(2, 3)),
        page_attention_group_contexts=None,
    )

    proposals = proposer.propose(
        [[12]],
        num_tokens_no_spec=None,
        token_ids_cpu=None,
    )

    assert proposals == [[201, 202, 203, 204]]
    assert proposer.backend.dflash_drafts[-1]["temperature"] == 1.0
    assert proposer.backend.dflash_drafts[-1]["generator"] is request_generator
    assert runner._draft_prob_req_ids == ["r"]
    assert runner._draft_probs.shape == (1, 4, 32)
    assert proposer.take_last_draft_probs() is runner._draft_probs

    config.speculative_config.draft_sample_method = "greedy"
    greedy_proposer = proposer_module.HMONNXSpecDecodeProposer(config)
    greedy_proposer.runner = runner
    assert greedy_proposer._draft_temperature("r") is None
    assert greedy_proposer._draft_generator("r") is None
