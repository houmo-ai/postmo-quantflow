from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from ...hmonnx.hmonnx_model import HMONNXBaseModel, HMONNXModel
from .minicpmo_token2wav_modules import deterministic_hift_source
from .runtime_token2wav_adapters import (
    Token2WavRuntimeProtocol,
    _HMONNXCampplusAdapter,
    _HMONNXFlowAdapter,
    _HMONNXHiFTAdapter,
    _HMONNXSpeechTokenizerAdapter,
    install_hmonnx_token2wav,
)
from .runtime_token2wav_ops import (
    Token2WavExecutionReport,
    _values,
    crop_hift_waveform,
    prepare_flow_frontend_inputs,
    prepare_hift_input,
    run_cfm,
    run_hift_stft_on_cpu,
)
from .runtime_token2wav_streaming import (
    FLOW_CACHE_AXES,
    FLOW_CACHE_NAMES,
    STREAM_SILENCE_TOKEN,
    StreamState,
    _read_present_valid_length,
    _token2wav_capacity_guide,
    _truncate_streaming_cache,
    apply_streaming_waveform_overlap,
    bound_streaming_attention_cache,
    compact_conformer_attention_cache,
    compact_estimator_attention_cache,
    pack_conformer_attention_cache,
    pack_streaming_attention_cache,
    run_stream_cfm,
)


class MiniCPMO45Token2WavHMONNXRuntime(HMONNXBaseModel):
    """Shared graph lifecycle around MiniCPM's Token2Wav state machine.

    Each Flow/HiFT/speaker graph is a repository ``HMONNXModel`` registered by
    ``HMONNXBaseModel``.  The remaining scheduler is model-specific: its fixed
    streaming cache ABI, CFM iteration and waveform overlap differ from both
    the text decoder and CosyVoice's private Token2Wav implementation, so no
    common repository runtime can replace it without changing exported graphs.
    """

    def __init__(
        self,
        root: Path,
        components: Mapping[str, Mapping[str, object]],
        *,
        enable_cuda_graph: bool = False,
        enable_auto_offload: bool = False,
        enable_golden: bool = False,
        device_map: str | torch.device | list[str | torch.device] | None = None,
        noise_seed: int | None = None,
    ) -> None:
        super().__init__(device_map=device_map)

        def graph(path: Path) -> HMONNXModel:
            return HMONNXModel(
                str(path),
                enable_golden=enable_golden,
                enable_cuda_graph=enable_cuda_graph,
                enable_auto_offload=enable_auto_offload,
                device_map=self._valid_devices,
            )

        frontend_meta = components["token2wav_flow_frontend"]
        decoder_meta = components["token2wav_flow_decoder"]
        hift_meta = components["token2wav_hift"]
        export_mode = str(frontend_meta["export_mode"])
        self.frontend_session = None
        self.decoder_session = None
        self.hift_part1_session = None
        self.hift_part2_session = None
        offline_roles = export_mode == "offline"
        if offline_roles:
            self.frontend_session = graph(root / str(frontend_meta["graphs"]["main"]))
            self.decoder_session = graph(root / str(decoder_meta["graphs"]["main"]))
            self.hift_part1_session = graph(root / str(hift_meta["graphs"]["part1"]))
            self.hift_part2_session = graph(root / str(hift_meta["graphs"]["part2"]))
        self.token_capacity = int(frontend_meta["token_capacity"])
        self.mel_capacity = int(frontend_meta["mel_capacity"])
        self.up_rate = int(frontend_meta["up_rate"])
        self.hift_frame_capacity = int(hift_meta["frame_capacity"])
        stft_meta = hift_meta["stft"]
        self._hift_stft_n_fft = int(stft_meta["n_fft"])
        self._hift_stft_hop_length = int(stft_meta["hop_length"])
        self.hift_source_config = hift_meta["source_generator"]
        self._flow_noise_shape = tuple(int(item) for item in decoder_meta["noise_shape"])
        self._noise_generator = torch.Generator(device="cpu")
        if noise_seed is None:
            self._noise_generator.seed()
        else:
            self._noise_generator.manual_seed(noise_seed)
        self._noise_device = torch.device("cpu")
        self._offline_hift_enabled = offline_roles
        self._hift_harmonic_count = int(hift_meta["harmonic_count"])
        self._hift_source_length = int(hift_meta["source_length"])
        self.rand_noise = None
        self.hift_phase_noise = None
        self.hift_source_noise = None
        self.n_timesteps = int(decoder_meta["n_timesteps"])
        self.cfg_rate = float(decoder_meta["cfg_rate"])
        self.report = Token2WavExecutionReport()
        self._base_flow_cache: dict[str, Tensor] | None = None
        self._stream_state: StreamState | None = None
        self.speech_window = None
        speaker = components["speaker"]
        self.campplus_session = graph(root / str(speaker["graphs"]["campplus"]))
        self.campplus_sequence_length = int(speaker["sequence_length"])
        self.speech_tokenizer_session = graph(root / str(speaker["graphs"]["speech_tokenizer"]))
        self.speech_tokenizer_feats_length = int(speaker["feats_length"])
        self.streaming_enabled = export_mode in {"session", "stream"}
        if self.streaming_enabled:
            self._load_streaming_roles(root, components, graph)
        self._regenerate_noise()

    def _create_hift_noise(self, harmonic_count: int, source_length: int) -> tuple[Tensor, Tensor]:
        phase_noise = torch.rand((1, harmonic_count), generator=self._noise_generator)
        phase_noise[:, 0] = 0
        source_noise = torch.randn((1, source_length, harmonic_count), generator=self._noise_generator)
        return (
            phase_noise.to(device=self._noise_device, dtype=torch.float16),
            source_noise.to(device=self._noise_device, dtype=torch.float16),
        )

    def _regenerate_noise(self) -> None:
        self.rand_noise = torch.randn(self._flow_noise_shape, generator=self._noise_generator).to(
            device=self._noise_device,
            dtype=torch.float16,
        )
        if self._offline_hift_enabled:
            self.hift_phase_noise, self.hift_source_noise = self._create_hift_noise(
                self._hift_harmonic_count,
                self._hift_source_length,
            )
        if self.streaming_enabled:
            self.stream_hift_phase_noise, self.stream_hift_source_noise = self._create_hift_noise(
                self._stream_hift_harmonic_count,
                self._stream_hift_source_length,
            )

    def _load_streaming_roles(
        self,
        root: Path,
        components: Mapping[str, Mapping[str, object]],
        graph: Callable[[Path], HMONNXModel],
    ) -> None:
        frontend = components["token2wav_flow_frontend"]["stream_contract"]
        decoder = components["token2wav_flow_decoder"]["stream_contract"]
        hift = components["token2wav_hift"]["stream_contract"]
        self.flow_frontend_session = graph(
            root / str(components["token2wav_flow_frontend"]["graphs"]["stream_flow_frontend"])
        )
        self.estimator_step_session = graph(
            root / str(components["token2wav_flow_decoder"]["graphs"]["stream_flow_estimator_step"])
        )
        self.hift_stream_part1_session = graph(root / str(components["token2wav_hift"]["graphs"]["stream_hift_part1"]))
        self.hift_stream_part2_session = graph(root / str(components["token2wav_hift"]["graphs"]["stream_hift_part2"]))
        self.stream_flow_meta = frontend
        self.stream_estimator_meta = decoder
        self.stream_hift_meta = hift
        self._stream_hift_harmonic_count = int(hift["harmonic_count"])
        self._stream_hift_source_length = int(hift["source_length"])
        self.stream_hift_phase_noise = None
        self.stream_hift_source_noise = None
        self.pre_lookahead_len = int(frontend["pre_lookahead_len"])
        self.chunk_token_capacity = int(frontend["chunk_token_capacity"])
        self.append_capacity = int(decoder["append_capacity"])
        self.base_conformer_layers = int(frontend["base_conformer_layers"])
        self.stream_tail_frames = int(decoder["prompt_cache_policy"]["recent_mel_frames"])
        self.host_timestep_cache_banks = int(decoder["host_timestep_cache_banks"])
        self.source_cache_length = int(hift["source_cache_length"])
        self.speech_cache_length = int(hift["speech_cache_length"])
        self.hift_hop_length = int(components["token2wav_hift"]["hop_length"])
        self.speech_window = torch.from_numpy(np.hamming(self.speech_cache_length * 2)).float()
        self._base_flow_cache = None

    def _set_device(self, device: torch.device):
        super()._set_device(device)
        self._noise_device = device
        self.rand_noise = self.rand_noise.to(device)
        if self._offline_hift_enabled:
            self.hift_phase_noise = self.hift_phase_noise.to(device)
            self.hift_source_noise = self.hift_source_noise.to(device)
        if self.streaming_enabled:
            self.stream_hift_phase_noise = self.stream_hift_phase_noise.to(device)
            self.stream_hift_source_noise = self.stream_hift_source_noise.to(device)
            self.speech_window = self.speech_window.to(device)
        if self._base_flow_cache is not None:
            self._base_flow_cache = {name: value.to(device) for name, value in self._base_flow_cache.items()}
        if self._stream_state is not None:
            self._stream_state.flow_cache = {
                name: value.to(device) for name, value in self._stream_state.flow_cache.items()
            }
            self._stream_state.estimator_cnn_banks = self._stream_state.estimator_cnn_banks.to(device)
            self._stream_state.estimator_att_banks = self._stream_state.estimator_att_banks.to(device)
            self._stream_state.hift_cache = {
                name: value.to(device) for name, value in self._stream_state.hift_cache.items()
            }
        return self

    def _run_flow_role(self, tokens: Tensor, embedding: Tensor, final: bool) -> Tensor:
        if self._stream_state is None:
            raise RuntimeError("Token2Wav streaming cache is not initialized")
        session = self.flow_frontend_session
        if session is None or self.estimator_step_session is None:
            raise RuntimeError("Token2Wav streaming Flow roles are unavailable")
        meta = self.stream_flow_meta
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        token_valid_length = int(tokens.shape[1])
        if tokens.device != embedding.device:
            raise RuntimeError("Token2Wav Flow inputs are on different devices")
        if any(value.device != tokens.device for value in self._stream_state.flow_cache.values()):
            raise RuntimeError("Token2Wav Flow cache is on a different device than graph inputs")
        capacity = getattr(self, "chunk_token_capacity", None)
        if capacity is not None and token_valid_length > capacity:
            raise RuntimeError(f"Token2Wav chunk token length {token_valid_length} exceeds capacity {capacity}")
        if capacity is not None and tokens.shape[1] < capacity:
            tokens = torch.nn.functional.pad(tokens, (0, capacity - tokens.shape[1]), value=STREAM_SILENCE_TOKEN)
        shapes = {
            name: tuple(int(value) for value in meta["frontend_cache_shapes"][name])
            for name in ("conformer_cnn_cache", "conformer_att_cache")
        }
        # Exported streaming Flow graphs use A16: official fp32 prompt state must be
        # cast to fp16 at the graph boundary, mirroring the non-streaming flow_inference.
        cnn_cache = self._stream_state.flow_cache["conformer_cnn_cache"]
        att_cache = self._stream_state.flow_cache["conformer_att_cache"]
        real_valid = self._stream_state.flow_valid_lengths["conformer_att_cache"]
        conformer_capacity = shapes["conformer_att_cache"][3]
        inputs = (
            tokens.to(torch.int32),
            torch.tensor([token_valid_length], dtype=torch.int32, device=tokens.device),
            torch.tensor([int(final)], dtype=torch.int32, device=tokens.device),
            embedding.to(torch.float16),
            cnn_cache.to(torch.float16),
            pack_conformer_attention_cache(
                att_cache,
                valid_length=real_valid,
                capacity=conformer_capacity,
                base_layer_count=self.base_conformer_layers,
            ).to(torch.float16),
            torch.tensor(
                [real_valid],
                dtype=torch.int32,
                device=tokens.device,
            ),
        )
        values = _values(session(*inputs))
        if len(values) != 5:
            raise RuntimeError(f"Token2Wav stream Flow frontend returned {len(values)} tensors, expected 5")
        mu, spks, present_cnn, present_att, present_valid_length = values
        self._stream_state.flow_cache["conformer_cnn_cache"] = present_cnn.to(cnn_cache.dtype)
        prompt_mel_length = int(self._stream_state.prompt_mel_length)
        current_mel_length = (
            token_valid_length * self.up_rate if final else (token_valid_length - self.pre_lookahead_len) * self.up_rate
        )
        if current_mel_length <= 0:
            raise RuntimeError(f"Token2Wav streaming chunk produced invalid mel length {current_mel_length}")
        graph_present_length = _read_present_valid_length(
            present_valid_length,
            name="present_conformer_cache_valid_length",
            capacity=conformer_capacity + int(self.append_capacity),
        )
        expected_present_length = real_valid + current_mel_length
        if graph_present_length != expected_present_length:
            raise RuntimeError(
                "Token2Wav Conformer valid-length mismatch: "
                f"graph={graph_present_length}, expected={expected_present_length}"
            )
        compact_present_att = compact_conformer_attention_cache(
            present_att.to(att_cache.dtype),
            past_valid_length=real_valid,
            current_valid_length=current_mel_length,
            input_capacity=conformer_capacity,
            base_layer_count=self.base_conformer_layers,
        )
        compact_present_att = _truncate_streaming_cache(
            compact_present_att,
            prompt_length=prompt_mel_length,
            tail=self.stream_tail_frames,
            capacity=conformer_capacity,
        )
        self._stream_state.flow_cache["conformer_att_cache"] = compact_present_att
        self._stream_state.flow_valid_lengths["conformer_att_cache"] = int(compact_present_att.shape[3])
        output_mel_capacity = int(meta["output_mel_capacities"]["final" if final else "normal"])
        if current_mel_length > output_mel_capacity:
            raise RuntimeError(f"Token2Wav mel length {current_mel_length} exceeds role capacity {output_mel_capacity}")
        mu = mu[:, :, :current_mel_length]
        cond = torch.zeros_like(mu)
        offset = self._stream_state.flow_valid_lengths["estimator_att_cache"]
        noise = self.rand_noise[:, :, offset : offset + current_mel_length].to(device=mu.device, dtype=mu.dtype)
        mel, self._stream_state.estimator_cnn_banks, self._stream_state.estimator_att_banks = run_stream_cfm(
            self.estimator_step_session,
            mu=mu,
            spks=spks,
            cond=cond,
            noise=noise,
            n_timesteps=self.n_timesteps,
            cfg_rate=self.cfg_rate,
            estimator_cnn_banks=self._stream_state.estimator_cnn_banks,
            estimator_att_banks=self._stream_state.estimator_att_banks,
            estimator_att_capacity=int(
                getattr(self, "stream_estimator_meta", {})
                .get("estimator_step_cache_shapes", {})
                .get("input_att", (0, 0, 0, 150, 0))[3]
            ),
            frame_capacity=int(
                getattr(self, "stream_estimator_meta", {})
                .get("estimator_step_cache_shapes", {})
                .get("frame_capacity", current_mel_length)
            ),
            on_estimator_call=self.report.record_stream_estimator_step,
        )
        estimator_capacity = int(
            getattr(self, "stream_estimator_meta", {})
            .get("estimator_step_cache_shapes", {})
            .get("input_att", (0, 0, 0, 150, 0))[3]
        )
        present_banks = self._stream_state.estimator_att_banks
        truncated_banks = _truncate_streaming_cache(
            present_banks,
            prompt_length=prompt_mel_length,
            tail=self.stream_tail_frames,
            capacity=estimator_capacity,
        )
        self._stream_state.estimator_att_banks = truncated_banks
        self._stream_state.flow_valid_lengths["estimator_att_cache"] = int(truncated_banks.shape[4])
        estimator_cnn = self._stream_state.flow_cache["estimator_cnn_cache"].clone()
        estimator_cnn[: self.n_timesteps] = self._stream_state.estimator_cnn_banks
        self._stream_state.flow_cache["estimator_cnn_cache"] = estimator_cnn
        total_cache_banks = int(self._stream_state.flow_cache["estimator_att_cache"].shape[0])
        unused_banks = truncated_banks.new_zeros((total_cache_banks - self.n_timesteps, *truncated_banks.shape[1:]))
        self._stream_state.flow_cache["estimator_att_cache"] = torch.cat((truncated_banks, unused_banks), dim=0)
        self._stream_state._check_lengths()
        self.report.record_stream_flow(final)
        return mel

    def _run_hift_graph(self, mel: Tensor, cache_source: Tensor, final: bool) -> tuple[Tensor, Tensor]:
        part1_session = self.hift_stream_part1_session
        part2_session = self.hift_stream_part2_session
        frame_capacity = int(self.stream_hift_meta["frame_capacity"])
        mel_cache_length = int(self.stream_hift_meta["mel_cache_length"])
        source_cache_length = int(self.stream_hift_meta["source_cache_length"])
        source_length = int(cache_source.shape[2])
        if source_length not in (0, source_cache_length):
            raise RuntimeError(f"Token2Wav source cache length must be 0 or {source_cache_length}, got {source_length}")
        past_mel_length = mel_cache_length if source_length else 0
        if mel.shape[2] < past_mel_length:
            raise RuntimeError("Token2Wav combined mel is shorter than its cached mel prefix")
        current_mel = mel[:, :, past_mel_length:]
        current_mel_length = int(current_mel.shape[2])
        if current_mel_length > frame_capacity:
            raise RuntimeError(
                f"Token2Wav current mel length {current_mel_length} exceeds HiFT capacity {frame_capacity}"
            )
        past_mel = mel[:, :, :past_mel_length]
        past_mel = F.pad(past_mel, (0, mel_cache_length - past_mel_length))
        current_mel = prepare_hift_input(current_mel, frame_capacity=frame_capacity)
        graph_dtype = torch.float16
        inputs = (
            current_mel.to(graph_dtype),
            torch.tensor([current_mel_length], dtype=torch.int32, device=mel.device),
            past_mel.to(graph_dtype),
            torch.tensor([past_mel_length], dtype=torch.int32, device=mel.device),
        )
        prepared_speech_feat, f0 = _values(part1_session(*inputs))
        full_source = self._host_hift_source(f0, self.stream_hift_phase_noise, self.stream_hift_source_noise)
        if source_length:
            full_source[:, :, :source_length] = cache_source.float().cpu()
        source_stft = run_hift_stft_on_cpu(
            full_source,
            n_fft=self._hift_stft_n_fft,
            hop_length=self._hift_stft_hop_length,
            target_device=mel.device,
            target_dtype=graph_dtype,
        )
        full_source = full_source.to(mel.device)
        (raw_waveform,) = _values(part2_session(prepared_speech_feat, source_stft))
        logical_mel_length = past_mel_length + current_mel_length
        logical_sample_length = logical_mel_length * self.hift_hop_length
        if logical_sample_length > raw_waveform.shape[1] or logical_sample_length > full_source.shape[2]:
            raise RuntimeError("Token2Wav logical HiFT output exceeds fixed graph capacity")
        self.report.record_stream_hift(final)
        return raw_waveform[:, :logical_sample_length], full_source[:, :, :logical_sample_length]

    def _run_hift_role(self, mel: Tensor, final: bool) -> Tensor:
        if self._stream_state is None:
            raise RuntimeError("Token2Wav streaming cache is not initialized")
        mel_valid_length = self._stream_state.hift_valid_lengths["mel"]
        source_valid_length = self._stream_state.hift_valid_lengths["source"]
        past_mel = self._stream_state.hift_cache["mel"][:, :, :mel_valid_length]
        past_source = self._stream_state.hift_cache["source"][:, :, :source_valid_length]
        combined_mel = torch.cat((past_mel.to(mel), mel), dim=2)
        raw_waveform, full_source = self._run_hift_graph(combined_mel, past_source.to(mel.device), final)
        emitted, present_speech, present_speech_length = apply_streaming_waveform_overlap(
            raw_waveform,
            raw_valid_length=int(raw_waveform.shape[1]),
            past_speech=self._stream_state.hift_cache["speech"],
            past_speech_valid_length=self._stream_state.hift_valid_lengths["speech"],
            speech_window=self.speech_window,
            overlap=self.speech_cache_length,
            final=final,
        )
        mel_tail = combined_mel[:, :, -int(self.stream_hift_meta["mel_cache_length"]) :]
        source_tail = full_source[:, :, -int(self.stream_hift_meta["source_cache_length"]) :]
        self._stream_state.hift_cache = {"mel": mel_tail, "source": source_tail, "speech": present_speech}
        self._stream_state.hift_valid_lengths = {
            "mel": int(mel_tail.shape[2]),
            "source": int(source_tail.shape[2]),
            "speech": present_speech_length,
        }
        self._stream_state._check_lengths()
        return emitted

    def stream_flow(self, tokens: Tensor, embedding: Tensor, last_chunk: bool) -> tuple[Tensor, Mapping[str, Tensor]]:
        self._pending_hift_final = last_chunk
        mel = self._run_flow_role(tokens, embedding, last_chunk)
        if self._stream_state is None:
            raise RuntimeError("Token2Wav streaming cache is not initialized")
        return mel, {name: value.detach().clone() for name, value in self._stream_state.flow_cache.items()}

    def stream_hift(
        self,
        mel: Tensor,
        cache_source: Tensor,
        last_chunk: bool | None = None,
    ) -> tuple[Tensor, Tensor]:
        final = bool(last_chunk) if last_chunk is not None else getattr(self, "_pending_hift_final", False)
        self._pending_hift_final = False
        waveform, source = self._run_hift_graph(mel, cache_source, final)
        return waveform.float(), source

    def reset_stream_cache(self) -> None:
        state = getattr(self, "_stream_state", None)
        base_cache = getattr(self, "_base_flow_cache", None)
        if base_cache is None:
            if state is None:
                return
            raise RuntimeError("immutable Token2Wav stream base cache was never initialized")
        if state is not None:
            state.reset(base_cache)

    def reset_state(self) -> None:
        self.reset_stream_cache()
        self._regenerate_noise()

    def bridge_official_init(self, result: object) -> None:
        if not isinstance(result, Mapping):
            raise RuntimeError("unsupported official Token2Wav cache schema: expected mapping or host token2wav_cache")
        base_cache = result.get("flow_cache_base")
        if not isinstance(base_cache, Mapping):
            raise RuntimeError("unsupported official Token2Wav cache schema: missing flow_cache_base mapping")
        prompt_mel_length = result.get("prompt_mel_length")
        if not isinstance(prompt_mel_length, int):
            prompt_mel_length = self._infer_prompt_mel_length(result)
        if prompt_mel_length is None:
            raise RuntimeError("unsupported official Token2Wav cache schema: missing prompt mel length")
        self.init_stream_cache(base_cache, prompt_mel_length)
        if self.report.host_initialization_count == 0:
            self.report.record_host_initialization()

    def _infer_prompt_mel_length(self, result: Mapping[str, object]) -> int | None:
        # The official hift_cache_base mel is zero-length by design; the real prompt
        # mel length lives in the flow conformer attention cache length axis.
        base_cache = result.get("flow_cache_base")
        if isinstance(base_cache, Mapping):
            conformer_att = base_cache.get("conformer_att_cache")
            if isinstance(conformer_att, Tensor) and conformer_att.dim() >= 4:
                return int(conformer_att.shape[3])
        hift_cache = result.get("hift_cache_base")
        if isinstance(hift_cache, Mapping):
            mel = hift_cache.get("mel")
            if isinstance(mel, Tensor):
                if mel.dim() != 3 or mel.shape[0] != 1 or mel.shape[1] != 80:
                    raise RuntimeError(
                        "unsupported official Token2Wav cache schema: prompt mel must be [1, 80, frames]"
                    )
                if mel.shape[-1] > 0:
                    return int(mel.shape[-1])
        return None

    def _stream_att_capacity(self) -> int:
        """Exported streaming attention-cache capacity (prompt + tail window)."""
        meta = getattr(self, "stream_flow_meta", None)
        if not isinstance(meta, Mapping):
            return 0
        frontend_shapes = meta.get("frontend_cache_shapes")
        if not isinstance(frontend_shapes, Mapping):
            return 0
        att_shape = frontend_shapes.get("conformer_att_cache")
        if not isinstance(att_shape, Sequence) or len(att_shape) < 4:
            return 0
        return int(att_shape[3])

    def _require_prompt_within_stream_capacity(self, base_cache: Mapping[str, Tensor]) -> None:
        meta = getattr(self, "stream_flow_meta", None)
        if not isinstance(meta, Mapping):
            return
        frontend_shapes = meta.get("frontend_cache_shapes", {})
        att_shape = frontend_shapes.get("conformer_att_cache") if isinstance(frontend_shapes, Mapping) else None
        if not isinstance(att_shape, Sequence) or len(att_shape) < 4:
            return
        capacity = int(att_shape[3])
        for name, axis in (("conformer_att_cache", 3), ("estimator_att_cache", 4)):
            value = base_cache.get(name)
            if not isinstance(value, Tensor) or value.dim() <= axis:
                continue
            length = int(value.shape[axis])
            if length > capacity:
                raise RuntimeError(_token2wav_capacity_guide(length, capacity))

    def init_stream_cache(self, base_cache: Mapping[str, Tensor], prompt_mel_length: int) -> StreamState:
        expected_shapes = getattr(self, "stream_flow_meta", {}).get("cache_shapes", {})
        missing = [name for name in FLOW_CACHE_NAMES if name not in base_cache]
        if missing:
            raise RuntimeError(f"missing Flow cache tensors: {', '.join(missing)}")
        for name in FLOW_CACHE_NAMES:
            value = base_cache[name]
            if not isinstance(value, Tensor):
                raise RuntimeError(f"Flow cache {name} is not a tensor")
            expected = tuple(int(item) for item in expected_shapes.get(name, value.shape))
            # The exported cache_shapes are the graph capacity (prompt + tail); the
            # official base_cache carries the full prompt only, so its length axis may
            # be shorter than the capacity. Other axes must match exactly.
            axis = FLOW_CACHE_AXES.get(name)
            if len(expected) != value.dim():
                raise RuntimeError(f"Flow cache {name} rank {value.dim()} != expected {len(expected)}")
            if axis is not None and value.shape[axis] > expected[axis]:
                raise RuntimeError(f"Flow cache {name} length {value.shape[axis]} > capacity {expected[axis]}")
            mismatched = [
                (index, int(actual), int(want))
                for index, (actual, want) in enumerate(zip(value.shape, expected, strict=True))
                if index != axis and int(actual) != int(want)
            ]
            if mismatched:
                raise RuntimeError(f"Flow cache {name} shape {tuple(value.shape)} != expected {expected}")
            if value.dtype != torch.float32:
                raise RuntimeError(f"Flow cache {name} dtype must be float32")
        self._require_prompt_within_stream_capacity(base_cache)
        if prompt_mel_length < 0:
            raise RuntimeError("prompt mel length must be non-negative")
        self._regenerate_noise()
        self._base_flow_cache = {name: value.detach().clone() for name, value in base_cache.items()}
        att_capacity = self._stream_att_capacity()
        self._stream_state = StreamState.from_base_cache(
            self._base_flow_cache,
            prompt_mel_length,
            att_cache_capacity=att_capacity,
            append_capacity=getattr(self, "append_capacity", 56),
            host_timestep_cache_banks=getattr(self, "host_timestep_cache_banks", 10),
        )
        stream_hift_meta = getattr(self, "stream_hift_meta", None)
        if isinstance(stream_hift_meta, Mapping):
            cache_dtype = next(iter(self._base_flow_cache.values())).dtype
            cache_device = next(iter(self._base_flow_cache.values())).device
            self._stream_state.hift_cache = {
                "mel": torch.zeros(
                    (1, 80, int(stream_hift_meta["mel_cache_length"])),
                    dtype=cache_dtype,
                    device=cache_device,
                ),
                "source": torch.zeros(
                    (1, 1, int(stream_hift_meta["source_cache_length"])),
                    dtype=cache_dtype,
                    device=cache_device,
                ),
                "speech": torch.zeros(
                    (1, int(stream_hift_meta["speech_cache_length"])),
                    dtype=cache_dtype,
                    device=cache_device,
                ),
            }
            self._stream_state.hift_valid_lengths = {"mel": 0, "source": 0, "speech": 0}
        return self._stream_state

    def stream(self, tokens: Tensor, embedding: Tensor, last_chunk: bool = False) -> Tensor:
        if self._stream_state is None:
            raise RuntimeError("Token2Wav streaming cache is not initialized")
        chunk_tokens = tokens
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.dim() != 2:
            raise RuntimeError("Token2Wav stream tokens must have shape [batch, tokens]")
        capacity = getattr(self, "chunk_token_capacity", tokens.shape[1])
        if tokens.shape[1] > capacity:
            raise RuntimeError("final Token2Wav stream chunk exceeds exported token capacity")
        mel = self._run_flow_role(chunk_tokens, embedding, last_chunk)
        return self._run_hift_role(mel, last_chunk)

    def flow_inference(
        self,
        token: Tensor,
        token_len: Tensor,
        prompt_token: Tensor,
        prompt_token_len: Tensor,
        prompt_feat: Tensor,
        prompt_feat_len: Tensor,
        embedding: Tensor,
        n_timesteps: int = 10,
    ) -> Tensor:
        del token_len, prompt_token_len, prompt_feat_len
        inputs = prepare_flow_frontend_inputs(
            token,
            prompt_token,
            prompt_feat,
            embedding,
            token_capacity=self.token_capacity,
            mel_capacity=self.mel_capacity,
            up_rate=self.up_rate,
        )
        output = self.frontend_session(
            inputs.tokens,
            inputs.token_length,
            inputs.prompt_feat.to(torch.float16),
            inputs.prompt_feat_length,
            inputs.embedding.to(torch.float16),
        )
        self.report.record_flow_frontend()
        values = (output,) if isinstance(output, Tensor) else tuple(output)
        mu, mask, spks, cond = (value.to(torch.float16) for value in values[:4])
        noise = self.rand_noise[:, :, : mu.shape[2]].to(device=mu.device, dtype=mu.dtype)
        feat = run_cfm(
            self.decoder_session,
            mu=mu,
            mask=mask,
            spks=spks,
            cond=cond,
            noise=noise,
            n_timesteps=n_timesteps or self.n_timesteps,
            cfg_rate=self.cfg_rate,
            on_decoder_call=self.report.record_flow_decoder,
        )
        prompt_length = int(prompt_feat.shape[1])
        return feat[:, :, prompt_length : prompt_length + inputs.output_mel_length].float()

    def _host_hift_source(self, f0: Tensor, phase_noise: Tensor, source_noise: Tensor) -> Tensor:
        config = self.hift_source_config
        f0 = F.interpolate(
            f0.float().cpu().unsqueeze(1),
            scale_factor=int(config["upsample_scale"]),
            mode="nearest",
        ).transpose(1, 2)
        sine_waves, _ = deterministic_hift_source(
            f0,
            phase_noise.float().cpu(),
            source_noise.float().cpu(),
            sampling_rate=int(config["sampling_rate"]),
            upsample_scale=int(config["upsample_scale"]),
            sine_amp=float(config["sine_amp"]),
            noise_std=float(config["noise_std"]),
            voiced_threshold=float(config["voiced_threshold"]),
        )
        weight = torch.tensor(config["linear_weight"], dtype=torch.float32)
        bias = torch.tensor(config["linear_bias"], dtype=torch.float32)
        return torch.tanh(F.linear(sine_waves, weight, bias)).transpose(1, 2)

    def hift_inference(self, speech_feat: Tensor) -> Tensor:
        mel_frames = int(speech_feat.shape[2])
        prepared_speech_feat = prepare_hift_input(
            speech_feat,
            frame_capacity=self.hift_frame_capacity,
        ).to(torch.float16)
        f0 = self.hift_part1_session(prepared_speech_feat)
        f0 = f0 if isinstance(f0, Tensor) else f0[0]
        source = self._host_hift_source(f0, self.hift_phase_noise, self.hift_source_noise)
        source_stft = run_hift_stft_on_cpu(
            source,
            n_fft=self._hift_stft_n_fft,
            hop_length=self._hift_stft_hop_length,
            target_device=speech_feat.device,
            target_dtype=torch.float16,
        )
        output = self.hift_part2_session(prepared_speech_feat, source_stft)
        self.report.record_hift()
        waveform = output if isinstance(output, Tensor) else output[0]
        return crop_hift_waveform(waveform, mel_frames=mel_frames)

    def release_state(self) -> None:
        """Drop model-specific tensors; graph release is centralized."""
        self._base_flow_cache = None
        self._stream_state = None
        for name in (
            "hift_phase_noise",
            "hift_source_noise",
            "stream_hift_phase_noise",
            "stream_hift_source_noise",
            "rand_noise",
            "speech_window",
        ):
            setattr(self, name, None)


__all__ = [
    "MiniCPMO45Token2WavHMONNXRuntime",
    "StreamState",
    "Token2WavExecutionReport",
    "Token2WavRuntimeProtocol",
    "_HMONNXCampplusAdapter",
    "_HMONNXFlowAdapter",
    "_HMONNXHiFTAdapter",
    "_HMONNXSpeechTokenizerAdapter",
    "_read_present_valid_length",
    "_truncate_streaming_cache",
    "apply_streaming_waveform_overlap",
    "bound_streaming_attention_cache",
    "compact_conformer_attention_cache",
    "compact_estimator_attention_cache",
    "crop_hift_waveform",
    "install_hmonnx_token2wav",
    "pack_conformer_attention_cache",
    "pack_streaming_attention_cache",
    "prepare_flow_frontend_inputs",
    "prepare_hift_input",
    "run_cfm",
    "run_stream_cfm",
]
