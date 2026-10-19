"""Native hybrid Qwen MoE calibration preserves inference and model state."""

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration

from xhmodel_merak.xh_llm.models.qwen3_5_moe.safe_calibration import (
    calibrate_native_model,
    calibration_input_ids,
    normalize_calibration,
    portable_linear_attention,
    prepare_automatic_safe_inputs,
    selected_expert_outputs,
)
from xhmodel_merak.xh_llm.models.qwen3_5_moe.safe_prune import pack_safe_data


@pytest.fixture
def native_model():
    torch.manual_seed(4821)
    config = Qwen3_5MoeConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 8,
            "num_experts": 3,
            "num_experts_per_tok": 2,
            "moe_intermediate_size": 16,
            "shared_expert_intermediate_size": 16,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 2,
            "linear_key_head_dim": 8,
            "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4,
            "pad_token_id": 0,
            "eos_token_id": 31,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 8,
            "intermediate_size": 16,
            "out_hidden_size": 16,
            "num_heads": 2,
            "patch_size": 2,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "num_position_embeddings": 16,
        },
    )
    return Qwen3_5MoeForConditionalGeneration(config)


@pytest.fixture
def text_model(native_model):
    return native_model.model.language_model


def test_native_calibration_preserves_outputs_and_produces_packable_statistics(text_model):
    ids = torch.arange(1, 17).reshape(2, 8)
    text_model.eval()
    # The installed FLA kernels reject the CPU activations used by this test.
    with portable_linear_attention("cpu"), torch.no_grad():
        before = text_model(input_ids=ids, use_cache=False).last_hidden_state.clone()
    text_model.train()
    state = {name: value.clone() for name, value in text_model.state_dict().items()}
    experts = [layer.mlp.experts for layer in text_model.layers]
    hooks = [tuple(module._forward_pre_hooks) for module in experts]
    observed = []
    observer = text_model.register_forward_hook(
        lambda _module, _inputs, output: observed.append(output.last_hidden_state.clone())
    )
    try:
        priors, tables = calibrate_native_model(text_model, ids, "cpu", batch_size=1)
        assert observer.id in text_model._forward_hooks
    finally:
        observer.remove()
    torch.testing.assert_close(torch.cat(observed), before)
    assert text_model.training
    assert all(parameter.device.type == "cpu" for parameter in text_model.parameters())
    assert [tuple(module._forward_pre_hooks) for module in experts] == hooks
    for name, value in text_model.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    text_model.eval()
    with portable_linear_attention("cpu"), torch.no_grad():
        after = text_model(input_ids=ids, use_cache=False).last_hidden_state
    torch.testing.assert_close(after, before, rtol=0, atol=0)
    for name, shape in (("coeff", (2, 3, 3)), ("loss", (2, 3, 3)), ("expert_norms", (2, 3))):
        assert tables[name].shape == shape and tables[name].dtype == torch.float32
        assert torch.isfinite(tables[name]).all()
    assert (tables["expert_norms"] > 0).all()
    for layer in range(2):
        for kind in ("amp", "aimer"):
            prior = priors[kind][str(layer)]
            assert prior.shape == (3,) and prior.dtype == torch.float32
            assert torch.isfinite(prior).all() and (prior > 0).all()
            assert not torch.equal(prior, torch.ones_like(prior))
        packed = pack_safe_data(
            priors["amp"][str(layer)], priors["aimer"][str(layer)], tables["coeff"][layer], tables["loss"][layer]
        )
        assert packed.shape == (3, 7) and packed.dtype == torch.float16
        assert packed.is_contiguous() and torch.isfinite(packed).all()


@pytest.mark.parametrize("training", [False, True])
def test_calibration_failure_restores_existing_hooks_and_training(text_model, training):
    text_model.train(training)
    experts = text_model.layers[0].mlp.experts
    calls = []
    handle = experts.register_forward_pre_hook(lambda *_: calls.append(True))
    before = tuple(experts._forward_pre_hooks)
    try:
        with pytest.raises(IndexError):
            calibrate_native_model(text_model, torch.tensor([[text_model.config.vocab_size]]), "cpu")
        assert text_model.training == training
        assert tuple(experts._forward_pre_hooks) == before
        assert all(parameter.device.type == "cpu" for parameter in text_model.parameters())
        assert all(not layer.mlp.experts._forward_pre_hooks for layer in text_model.layers[1:])
        assert not calls
    finally:
        handle.remove()


def test_selected_outputs_reconstruct_native_expert_mixture(text_model):
    experts = text_model.layers[0].mlp.experts
    hidden = torch.randn(4, 16)
    ids = torch.tensor([[0, 2], [2, 1], [1, 0], [0, 1]])
    weights = torch.tensor([[0.2, 0.8], [0.4, 0.6], [0.7, 0.3], [0.1, 0.9]])
    outputs = selected_expert_outputs(experts, hidden, ids)
    assert outputs.shape == (4, 2, 16) and outputs.dtype == hidden.dtype
    torch.testing.assert_close((outputs * weights[..., None]).sum(1), experts(hidden, ids, weights))


def tiny_tokenizer():
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "a": 1, "b": 2, "c": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]")


def test_training_tokens_are_one_continuous_segment(tmp_path):
    tokenizer = tiny_tokenizer()
    path = tmp_path / "train.txt"
    path.write_text("a b c a b c a", encoding="utf-8")
    config = normalize_calibration({"text_path": str(path), "nsamples": 2, "sequence_length": 3})
    torch.testing.assert_close(calibration_input_ids(tokenizer, config), torch.tensor([[1, 2, 3], [1, 2, 3]]))
    with pytest.raises(ValueError, match="7 tokens; 9 required"):
        calibration_input_ids(tokenizer, {**config, "nsamples": 3})


@pytest.mark.parametrize("config", [{"split": "test"}, {"nsamples": 0}, {"batch_size": True}, {"unknown": 1}])
def test_calibration_rejects_invalid_protocol(config):
    with pytest.raises(ValueError):
        normalize_calibration(config)


def test_automatic_artifacts_are_checkpoint_bound_reusable_and_reject_stale_tokens(native_model, tmp_path):
    model_dir, work_dir, text_path = tmp_path / "model", tmp_path / "work", tmp_path / "train.txt"
    native_model.save_pretrained(model_dir)
    tiny_tokenizer().save_pretrained(model_dir)
    text_path.write_text("a b c a b c a b", encoding="utf-8")
    config = {
        "mode": "decode",
        "algorithm": "safe",
        "calibration": {
            "text_path": str(text_path),
            "nsamples": 2,
            "sequence_length": 4,
        },
    }
    first = dict(config)
    prepare_automatic_safe_inputs(native_model, first, model_dir, work_dir, device="cpu")
    paths = [work_dir / "safe_calibration" / name for name in ("priors.pt", "exfold.pt")]
    payloads = [torch.load(path, weights_only=True) for path in paths]
    assert [first[key] for key in ("ace_priors_path", "exfold_artifact_path")] == [str(path) for path in paths]
    identity = payloads[0]["identity"]
    assert identity == payloads[1]["identity"] and identity["layers"] == 2
    assert len(identity["source"]) == len(identity["tokens_sha256"]) == 64
    assert identity["calibration"] == normalize_calibration(config["calibration"])
    timestamps = [path.stat().st_mtime_ns for path in paths]
    second = dict(config)
    prepare_automatic_safe_inputs(native_model, second, model_dir, work_dir, device="cpu")
    assert first == second and [path.stat().st_mtime_ns for path in paths] == timestamps
    torch.testing.assert_close(torch.load(paths[1], weights_only=True)["coeff"], payloads[1]["coeff"])
    text_path.write_text("b a c a b c a b", encoding="utf-8")
    with pytest.raises(ValueError, match="cache disagrees"):
        prepare_automatic_safe_inputs(native_model, dict(config), model_dir, work_dir, device="cpu")
    assert [path.stat().st_mtime_ns for path in paths] == timestamps


@pytest.mark.parametrize("key", ["ace_priors_path", "exfold_artifact_path", "s_scalar_path"])
def test_explicit_calibration_input_is_never_automatically_supplemented(native_model, tmp_path, key):
    prune = {"mode": "decode", "algorithm": "safe", key: "explicit.pt"}
    prepare_automatic_safe_inputs(native_model, prune, tmp_path / "absent-model", tmp_path / "work", device="cpu")
    assert prune == {"mode": "decode", "algorithm": "safe", key: "explicit.pt"}
    assert not (tmp_path / "work").exists()
