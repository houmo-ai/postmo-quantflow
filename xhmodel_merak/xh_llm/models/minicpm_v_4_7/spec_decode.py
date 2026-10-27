"""MiniCPM multimodal inputs over the shared DFlash verification state machine."""

from pathlib import Path
from types import SimpleNamespace

import torch

from xhmodel_merak.xh_llm.hmonnx.base_llm_hmonnx_model import BaseLLMHMONNXModel
from xhmodel_merak.xh_llm.models.qwen3_5._spec_decode_shared import SpecDecodeVerifyResult
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_onnx_model import (
    _alloc_cache_inputs, _ensure_logits_shape, _resolve_input_name,
)
from xhmodel_merak.xh_llm.models.qwen3_5.spec_decode_v2 import Qwen3_5SpecDecodeV2Model

from .positions import position_chunk
from .spec_decode_loop import run_spec_decode_loop


class MiniCPMV47SpecDecodeModel(Qwen3_5SpecDecodeV2Model):
    prepared = None
    image_token_id = -1

    def __init__(self, *, enable_golden=False, enable_confidence_truncation=False, confidence_threshold=None, verify_graphs=None, **kwargs):
        self.enable_golden = bool(enable_golden)
        self.enable_confidence_truncation = bool(enable_confidence_truncation)
        self.confidence_threshold = confidence_threshold
        self._verify_graph_configs = {int(length): config for length, config in dict(verify_graphs or {}).items()}
        self._verify_session_states = {}
        self.last_spec_decode_stats = None
        super().__init__(**kwargs)
        from xhmodel_merak.exec_backends import HMONNXInferenceConfig, HMONNXInferenceV2, MultiHMONNXLoader

        if self._verify_graph_configs:
            paths = {
                f"verify_{length}": config["onnx"] if isinstance(config, dict) else config.onnx
                for length, config in self._verify_graph_configs.items()
            }
            loaded = MultiHMONNXLoader(paths)
            for length, path in ((int(name.removeprefix("verify_")), value) for name, value in paths.items()):
                session = HMONNXInferenceV2.from_onnx_graph(
                    path, loaded.graphs[f"verify_{length}"],
                    HMONNXInferenceConfig(
                        exec_devices=[self._session_device], layers=self._v2_target_layers,
                        enable_golden=self.enable_golden,
                    ),
                )
                self._set_hmonnx_session_golden_dir(session, Path(path).parent)
                if self.enable_golden and hasattr(session, "reset_step"):
                    session.reset_step()
                info = session.get_input(session.get_input_names()[0])
                self._verify_session_states[length] = SimpleNamespace(session=session, inputs_info=info)

    def _apply_positions(self, feed, session, *, offset: int, length: int):
        if self.prepared is None or self.prepared.position_ids is None:
            return feed
        positions = position_chunk(self.prepared.position_ids, self.prepared.rope_deltas, offset=offset, length=length)
        for axis, names in enumerate(
            (("time_position_ids",), ("hight_position_ids", "height_position_ids"), ("width_position_ids",))
        ):
            for name in names:
                if name in feed:
                    info = session.get_input(name)
                    feed[name] = positions[axis, 0].to(device=self.device, dtype=info.dtype).reshape(info.shape)
        return feed

    def _build_prefill_feed(self, chunk_input_ids, valid_len, past_seq_len, cache_state):
        feed = super()._build_prefill_feed(chunk_input_ids, valid_len, past_seq_len, cache_state)
        if self.prepared is not None and self.prepared.image_embeds is not None:
            features = self.prepared.image_embeds
            prior = int((self.prepared.input_ids[:, :past_seq_len] == self.image_token_id).sum())
            mask = chunk_input_ids.to(self.device) == self.image_token_id
            count = int(mask.sum())
            if count:
                embeds = feed[self._prefill_inputs_name]
                embeds[:, :valid_len][mask] = features[prior : prior + count].to(embeds)
        return self._apply_positions(
            feed, self.prefill_session, offset=past_seq_len, length=int(self._prefill_inputs_info.shape[1])
        )

    def _build_decode_feed(self, token_ids, past_seq_len, cache_state, current_input_length=None):
        feed = super()._build_decode_feed(token_ids, past_seq_len, cache_state, current_input_length)
        return self._apply_positions(
            feed, self.decode_session, offset=past_seq_len, length=int(self._decode_inputs_info.shape[1])
        )

    def _run_draft_dflash(self, next_tok, past_seq_len):
        cache_state = self._ensure_dflash_cache_state()
        block_len = int(self.draft_decode_session.get_input("noise_embedding").shape[1])
        block_ids = torch.full((1, block_len), self._dflash_noise_mask_token_id, dtype=torch.long, device=self.device)
        block_ids[0, 0] = next_tok[0, 0]
        output = self._run_draft_session(
            self.draft_decode_session,
            self._build_dflash_decode_feed(self._embed_token_ids(block_ids), past_seq_len, cache_state, next_tok),
        )
        self._last_draft_output = output
        if "draft_token_ids" in output:
            return [output["draft_token_ids"][:, index : index + 1] for index in range(output["draft_token_ids"].shape[1])]
        return super()._run_draft_dflash(next_tok, past_seq_len)

    def _build_verify_feed(self, state, token_ids, past_seq_len, cache_state):
        saved = (self.decode_session, self._decode_inputs_info, self._decode_mask_info, self._decode_past_seq_info, self._decode_current_seq_info, self._decode_inputs_name, self._decode_past_seq_name, self._decode_current_seq_name, self._decode_mask_name)
        self.decode_session = state.session
        self._decode_inputs_info = state.inputs_info
        names = state.session.get_input_names()
        self._decode_inputs_name = names[0]
        self._decode_past_seq_name = _resolve_input_name(state.session, ("past_seq_length", "valid_length"))
        self._decode_current_seq_name = _resolve_input_name(state.session, ("current_input_length", "current_length"))
        self._decode_mask_name = "linear_attn_mask" if "linear_attn_mask" in names else None
        self._decode_past_seq_info = state.session.get_input(self._decode_past_seq_name)
        self._decode_current_seq_info = state.session.get_input(self._decode_current_seq_name)
        self._decode_mask_info = state.session.get_input(self._decode_mask_name) if self._decode_mask_name else saved[2]
        try:
            feed = self._build_decode_feed(token_ids, past_seq_len, cache_state, current_input_length=int(token_ids.shape[1]))
        finally:
            (self.decode_session, self._decode_inputs_info, self._decode_mask_info, self._decode_past_seq_info, self._decode_current_seq_info, self._decode_inputs_name, self._decode_past_seq_name, self._decode_current_seq_name, self._decode_mask_name) = saved
        return self._apply_positions(feed, state.session, offset=past_seq_len, length=int(state.inputs_info.shape[1]))

    def generate(self, input_ids, tokenizer, max_new_tokens=256, return_stats=False, **kwargs):
        if not self.enable_confidence_truncation or not self._verify_session_states:
            return super().generate(input_ids, tokenizer, max_new_tokens=max_new_tokens, return_stats=return_stats, **kwargs)
        if max_new_tokens <= 0:
            return ""
        if input_ids.dim() != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must be a nonempty [1, seq] tensor")
        if self.token_embedding is None:
            raise ValueError("token_embedding is not set, call set_input_embeddings first")
        prompt = input_ids[:, -self.max_context_tokens:] if self.max_context_tokens else input_ids
        self._ensure_prefill_session()
        self._ensure_decode_session()
        cache = _alloc_cache_inputs(self.prefill_session, self.device)
        chunk_length = int(self._prefill_inputs_info.shape[1])
        for past in range(0, int(prompt.shape[1]), chunk_length):
            chunk = prompt[:, past:past + chunk_length]
            valid = int(chunk.shape[1])
            feed = self._build_prefill_feed(chunk, valid, past, cache)
            _, outputs = self._run_hmonnx(self.prefill_session, feed)
            logits = _ensure_logits_shape(self._extract_logits(outputs))
            hidden = self._extract_hidden(outputs)
            if hidden is not None:
                self._append_dflash_context(hidden[:, :valid], past)
            self._update_linear_cache(cache, outputs)
        token = int(torch.argmax(logits[:, valid - 1], dim=-1).item())
        decode_cache = _alloc_cache_inputs(self.decode_session, self.device)
        for name, value in cache.items():
            if name in decode_cache:
                decode_cache[name] = value

        self._verify_lengths = []
        truncated = 0

        def draft(current, _hidden, past, *_args):
            nonlocal truncated
            tokens = self._run_draft_dflash(torch.tensor([[current]], device=self.device), past)
            ids = [int(item.reshape(-1)[0].item()) for item in tokens]
            output = getattr(self, "_last_draft_output", {})
            scores = output.get("confidence_logits", output.get("confidence"))
            if self.enable_confidence_truncation and scores is not None:
                kept = 0
                product = 1.0
                probabilities = scores.float().sigmoid() if "confidence_logits" in output else scores.float()
                for score in probabilities.reshape(-1).tolist():
                    product *= score
                    if product < float(self.confidence_threshold):
                        break
                    kept += 1
                truncated += max(0, len(ids) - kept)
                ids = ids[:max(kept, 0)]
            return ids

        def verify(current, drafted, past):
            length = len(drafted) + 1
            self._verify_lengths.append(length)
            state = self._verify_session_states[length]
            ids = torch.tensor([[current, *drafted]], dtype=torch.long, device=self.device)
            feed = self._build_verify_feed(state, ids, past, decode_cache)
            _, output = self._run_hmonnx(state.session, feed)
            values = _ensure_logits_shape(self._extract_logits(output))
            predicted = [int(torch.argmax(values[:, index], dim=-1).item()) for index in range(length)]
            return SpecDecodeVerifyResult(past, [current, *drafted], predicted, self._extract_hidden(output), output)

        def commit(result, steps):
            self._apply_verify_linear_cache_outputs(decode_cache, result.raw_result, accepted_steps=steps)

        def post_verify(result, accepted, steps, _next, mtp_past):
            hidden = result.verify_hidden
            if hidden is not None:
                self._append_dflash_context(hidden[:, :steps], result.initial_seq_len, decode_step=True)
            return self._select_hidden_step(hidden, accepted), mtp_past

        generated, stats = run_spec_decode_loop(
            max_new_tokens=max_new_tokens, initial_token_id=token, initial_past_seq_len=int(prompt.shape[1]),
            initial_last_hidden=None, initial_mtp_past_seq_len=0, eos_token_id=tokenizer.eos_token_id,
            num_drafts=self.block_size - 1, run_draft=draft, verify_round=verify,
            apply_verify_result=commit,
            post_verify=post_verify,
        )
        stats["verify_lengths_per_round"] = self._verify_lengths
        stats["confidence_truncated_total"] = truncated
        self.last_spec_decode_stats = stats
        text = tokenizer.decode(generated)
        return (text, stats) if return_stats else text


def load_speculative_runtime(root: Path, llm: dict, meta, device, enable_golden: bool):
    contract = llm["spec_decode"]
    draft = contract["draft"]
    model = MiniCPMV47SpecDecodeModel(
        prefill={"onnx": str(root / llm["prefill_hmonnx"])},
        decode={"onnx": str(root / llm["decode_hmonnx"])},
        draft={name: {"onnx": str(root / draft[f"{name}_hmonnx"])} for name in ("context", "context_decode", "decode")},
        session_device=device,
        enable_golden=enable_golden,
        spec_decode_mode="dflash",
        block_size=int(contract["num_draft_tokens"]) + 1,
        hidden_output_name="target_hidden",
        dflash_noise_token_id=int(draft["noise_token_id"]),
        max_context_tokens=int(llm["context_max_length"]),
        pad_token_id=int(getattr(meta.model_config, "pad_token_id", None) or 0),
        auto_offload=device.type == "cuda",
        enable_confidence_truncation=bool(contract.get("enable_confidence_truncation", False)),
        confidence_threshold=contract.get("confidence_threshold", 0.3),
        verify_graphs={length: {"onnx": str(root / path)} for length, path in contract.get("verify_graphs", {}).items()},
    )
    model.set_input_embeddings(BaseLLMHMONNXModel._build_embed_tokens_from_meta(meta))
    model.to(device)
    context_length = int(model.draft_context_decode_session.get_input("target_hidden").shape[1])
    if context_length != int(contract["verify_length"]):
        raise ValueError(
            "MiniCPM draft context_decode must match the complete target verify length; re-export required"
        )
    if enable_golden:
        model.set_spec_draft_golden(True)
        for session in (model.prefill_session, model.decode_session, *(state.session for state in model._verify_session_states.values())):
            model._set_hmonnx_session_golden(session, True)
    return model
