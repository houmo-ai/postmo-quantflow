"""Model-package transcription: encoder windows, Host fusion, then generate."""

from __future__ import annotations

from collections.abc import Mapping
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from xhmodel_merak.errors import ErrorCode, ModelError

from .hmonnx_runtime import MossHMONNXRuntime
from .host_fusion import transcribe_supported_utterance
from .processing import extract_moss_log_mel, load_audio, SAMPLING_RATE

_TRANSCRIPT_SEGMENT = re.compile(
    r"\[(?P<start>\d+(?:\.\d+)?)\]\[(?P<speaker>S\d{2,})\](?P<body>.*?)\[(?P<end>\d+(?:\.\d+)?)\]",
    re.DOTALL,
)
_TIMESTAMP_GAP = re.compile(r"(?:\[\d+(?:\.\d+)?\]\s*)+")

def transcribe_artifact(
    *,
    root: Path,
    meta: Mapping[str, Any],
    audio: str | Path | np.ndarray,
    device: str,
    prompt: str | None = None,
    max_new_tokens: int | None = None,
    selected_golden: tuple[str, ...] = (),
    runtime: MossHMONNXRuntime | None = None,
) -> str:
    """Run Encoder component, Host fusion, and prefill-decode generate."""
    import torch
    from transformers import AutoTokenizer

    from .prefill_decode import MossLMHead
    from .processing import load_official_processor

    waveform = audio if isinstance(audio, np.ndarray) else load_audio(audio)
    generation_budget = audio_generation_budget(len(waveform) / SAMPLING_RATE, max_new_tokens)
    owns_runtime = runtime is None
    hmonnx = runtime or MossHMONNXRuntime(root, meta, device)
    try:
        if selected_golden:
            if not hmonnx.available() or str(device).lower().startswith("cpu"):
                raise ModelError(
                    ErrorCode.GOLDEN_NOT_SUPPORTED,
                    "MOSS Golden requires an XH2a HMONNX device",
                    requested=list(selected_golden),
                )
            hmonnx.enable_golden(selected_golden)
        lm_file = Path(root) / meta.get("lm_head_file", "")
        if lm_file.is_file():
            lm_state = torch.load(lm_file, map_location="cpu", weights_only=True)
        else:
            lm_state = torch.load(Path(root) / meta["weights_file"], map_location="cpu", weights_only=True)["lm_head"]
        lm_state = normalize_lm_head_state(lm_state)
        lm_weight = lm_state["projection.weight"]
        lm_head = MossLMHead(hidden_size=lm_weight.shape[1], vocab_size=lm_weight.shape[0])
        lm_head.load_state_dict(lm_state)
        tokenizer = AutoTokenizer.from_pretrained(
            root, local_files_only=True, trust_remote_code=True, extra_special_tokens={},
        )

        try:
            processor = load_official_processor(root)
        except Exception as error:
            raise ModelError(
                ErrorCode.INFERENCE_FAILED,
                "MOSS inference requires the Official processor in the artifact",
            ) from error
        if not hmonnx.available() or str(device).lower().startswith("cpu"):
            raise ModelError(
                ErrorCode.INFERENCE_FAILED,
                "MOSS inference requires published HMONNX graphs on a non-CPU device",
            )
        lm_head = lm_head.to(device=getattr(hmonnx, "device", device), dtype=torch.float16)
        return transcribe_supported_utterance(
            waveform,
            prompt,
            tokenizer=processor,
            encode_audio=lambda window: encode_audio_hmonnx(window, root, hmonnx),
            embed_tokens=lambda ids: embed_tokens(ids, lm_head),
            generate=lambda fused, mask: generate_hmonnx(
                fused, mask, hmonnx, lm_head, tokenizer, root, max_new_tokens=generation_budget,
            ),
            validate_prompt=lambda length: validate_prompt_budget(
                root, length, hmonnx.input_shape("prefill", "inputs_embeds")[1],
                hmonnx.decoder_cache_length(), generation_budget,
            ),
        )
    finally:
        if owns_runtime:
            hmonnx.close()

def normalize_lm_head_state(state: Mapping[str, Any]) -> dict[str, Any]:
    """Accept MossLMHead (`projection.weight`) or official Linear (`weight`)."""
    if "projection.weight" in state:
        return dict(state)
    if "weight" in state:
        return {"projection.weight": state["weight"]}
    raise KeyError("lm_head state must contain projection.weight or weight")

def encode_audio_hmonnx(window: np.ndarray, root: Path, runtime: MossHMONNXRuntime) -> np.ndarray:
    import torch

    mel, _ = extract_moss_log_mel(window, root)
    output = runtime.encode(torch.from_numpy(mel[None]).to(runtime.device))
    return first_output(output).detach().cpu().numpy()[0]

def embed_tokens(input_ids: np.ndarray, lm_head: Any) -> np.ndarray:
    import torch

    weight = lm_head.projection.weight.detach()
    indices = torch.as_tensor(input_ids, dtype=torch.long, device=weight.device)
    return weight.index_select(0, indices).cpu().numpy()

def project_lm_head(hidden: Any, lm_head: Any) -> Any:
    weight = lm_head.projection.weight
    return lm_head(hidden.to(device=weight.device, dtype=weight.dtype))

def greedy_token(hidden: Any, lm_head: Any) -> Any:
    import torch

    logits = project_lm_head(first_output(hidden), lm_head)
    return torch.argmax(logits, dim=-1).reshape(-1).to(dtype=torch.long)

def token_embedding(token: Any, lm_head: Any, device: str) -> Any:
    import torch

    weight = lm_head.projection.weight
    token_ids = token.reshape(-1).to(device=weight.device, dtype=torch.int64)
    return weight.index_select(0, token_ids).reshape(1, 1, -1).to(device=device)

def first_output(output: Any) -> Any:
    if isinstance(output, Mapping):
        return output["audio_embeds"] if "audio_embeds" in output else output.get("last_hidden_state", next(iter(output.values())))
    return output[0] if isinstance(output, (tuple, list)) else output

def validate_transcript_text(
    text: str,
    model_id: str,
    *,
    max_duration: float | None = None,
    max_duration_tolerance: float = 0.0,
) -> str:
    try:
        return validate_transcript_structure(
            text, model_id, max_duration=max_duration, max_duration_tolerance=max_duration_tolerance,
        )
    except ModelError as error:
        error.details["raw_transcript"] = str(text)
        raise

def validate_transcript_structure(
    text: str,
    model_id: str,
    *,
    max_duration: float | None = None,
    max_duration_tolerance: float = 0.0,
) -> str:
    normalized = str(text).strip()
    segments = list(_TRANSCRIPT_SEGMENT.finditer(normalized))
    if not segments:
        raise ModelError(
            ErrorCode.INFERENCE_FAILED,
            "MOSS generation did not produce a valid timestamped speaker transcript",
            model=model_id,
        )
    cursor = 0
    previous_start = 0.0
    for index, match in enumerate(segments):
        interstitial = normalized[cursor : match.start()].strip()
        if interstitial and not _TIMESTAMP_GAP.fullmatch(interstitial):
            raise ModelError(
                ErrorCode.INFERENCE_FAILED,
                "MOSS transcript contains unstructured text outside segments",
                model=model_id,
            )
        start = float(match.group("start"))
        end = float(match.group("end"))
        if end < start:
            raise ModelError(
                ErrorCode.INFERENCE_FAILED,
                "MOSS transcript segment ends before it starts",
                model=model_id,
            )
        if max_duration_tolerance < 0:
            raise ValueError("max_duration_tolerance must be non-negative")
        if max_duration is not None and (
            start > max_duration + max_duration_tolerance
            or end > max_duration + max_duration_tolerance
        ):
            overflow = max(start, end) - max_duration
            raise ModelError(
                ErrorCode.INFERENCE_FAILED,
                "MOSS transcript timestamp exceeds supported duration "
                f"{max_duration:g}s (segment={index}, start={start:g}, end={end:g}, "
                f"overflow={overflow:g}s)",
                model=model_id,
                raw_transcript=normalized,
                segment=index,
                start=start,
                end=end,
                overflow_s=overflow,
            )
        if index and start < previous_start:
            raise ModelError(
                ErrorCode.INFERENCE_FAILED,
                "MOSS transcript segment timestamps are out of order",
                model=model_id,
            )
        previous_start = start
        cursor = match.end()
    if normalized[cursor:].strip():
        raise ModelError(
            ErrorCode.INFERENCE_FAILED,
            "MOSS transcript contains trailing unstructured text",
            model=model_id,
        )
    return normalized

def validate_prompt_budget(
    root: Path, prompt_length: int, prefill_length: int, cache_length: int,
    max_new_tokens: int | None,
) -> int:
    if prefill_length <= 0 or cache_length <= 0 or prompt_length <= 0:
        raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "MOSS context dimensions must be positive")
    config = json.loads((Path(root) / "generation_config.json").read_text(encoding="utf-8"))
    limit = generation_limit(config, prompt_length, cache_length=cache_length, override=max_new_tokens)
    if ((prompt_length + prefill_length - 1) // prefill_length) * prefill_length > cache_length:
        raise ModelError(ErrorCode.INFERENCE_FAILED, "MOSS padded prefill would overrun the KV cache")
    return limit

def generate_hmonnx(
    fused: np.ndarray,
    attention_mask: np.ndarray,
    runtime: MossHMONNXRuntime,
    lm_head: Any,
    tokenizer: Any,
    root: Path,
    *,
    max_new_tokens: int | None = None,
) -> np.ndarray:
    import torch

    del attention_mask
    prefill_length = runtime.input_shape("prefill", "inputs_embeds")[1]
    cache_length = runtime.decoder_cache_length()
    max_new_tokens = validate_prompt_budget(root, len(fused), prefill_length, cache_length, max_new_tokens)
    paged = bool(getattr(runtime, "uses_page_attention", lambda: False)())
    uses_inplace_cache = getattr(runtime, "uses_inplace_cache", None)
    inplace_cache = paged or (bool(uses_inplace_cache()) if callable(uses_inplace_cache) else False)
    if paged:
        runtime.ensure_paged_kv_caches()
        key = value = None
    elif inplace_cache:
        key, value = runtime.allocate_decoder_cache()
    else:
        key_shape = runtime.input_shape("prefill", "past_key_cache")
        value_shape = runtime.input_shape("prefill", "past_value_cache")
        if key_shape != value_shape:
            raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "MOSS key/value cache shapes differ")
        for name, shape in (("past_key_cache", key_shape), ("past_value_cache", value_shape)):
            if runtime.input_shape("decode", name) != shape:
                raise ModelError(ErrorCode.ARTIFACT_VALIDATION_FAILED, "MOSS prefill/decode cache shapes differ")
        cache = torch.zeros(key_shape, dtype=torch.float32, device=runtime.device)
        key, value = cache, cache.clone()
    current = torch.tensor([fused.shape[0]], dtype=torch.int32, device=runtime.device)
    outputs = None
    last_valid = 0
    for offset in range(0, len(fused), prefill_length):
        block = fused[offset:offset + prefill_length]
        valid = len(block)
        last_valid = valid
        padded = np.zeros((1, prefill_length, fused.shape[1]), dtype=np.float32)
        padded[0, :valid] = block
        outputs = runtime.prefill(
            torch.from_numpy(padded).to(runtime.device),
            torch.tensor([offset], dtype=torch.int32, device=runtime.device),
            torch.tensor([valid], dtype=torch.int32, device=runtime.device),
            key, value,
        )
        if not isinstance(outputs, Mapping):
            raise ModelError(ErrorCode.INFERENCE_FAILED, "MOSS prefill runtime returned an invalid output object")
        if not inplace_cache:
            key, value = outputs["present_key_cache"], outputs["present_value_cache"]
    if not isinstance(outputs, Mapping):
        raise ModelError(ErrorCode.INFERENCE_FAILED, "MOSS prefill runtime returned an invalid output object")
    try:
        hidden = outputs["last_hidden_state"]
        if not inplace_cache:
            key = outputs["present_key_cache"]
            value = outputs["present_value_cache"]
    except KeyError as error:
        raise ModelError(
            ErrorCode.INFERENCE_FAILED,
            f"MOSS prefill runtime is missing output {error.args[0]!r}",
        ) from error
    generated: list[int] = []
    hidden_states = first_output(hidden)
    if last_valid <= 0:
        raise ModelError(ErrorCode.INFERENCE_FAILED, "MOSS prefill produced no valid tokens")
    # Published graphs keep num_logits_to_keep=1, so last_hidden is already (1, 1, H).
    seq = int(hidden_states.shape[1])
    last = min(last_valid, seq) - 1
    if last < 0:
        raise ModelError(ErrorCode.INFERENCE_FAILED, "MOSS prefill hidden state is empty")
    token = greedy_token(hidden_states[:, last:last + 1, :], lm_head)
    eos_id = int(tokenizer.eos_token_id)
    ones = torch.ones(1, dtype=torch.int32, device=runtime.device)
    for step in range(max_new_tokens):
        token_id = int(token.reshape(-1)[0].item())
        if token_id == eos_id:
            break
        generated.append(token_id)
        outputs = runtime.decode(
            token_embedding(token, lm_head, runtime.device),
            current,
            ones,
            key,
            value,
        )
        current = current + 1
        hidden = outputs["last_hidden_state"]
        if not inplace_cache:
            key = outputs["present_key_cache"]
            value = outputs["present_value_cache"]
        token = greedy_token(hidden, lm_head)
    return completed_generation(generated, max_new_tokens, tokenizer)

def completed_generation(generated: list[int], limit: int, tokenizer: Any) -> np.ndarray:
    if len(generated) >= limit:
        raise ModelError(
            ErrorCode.INFERENCE_FAILED,
            "MOSS generation reached its output budget before EOS",
            stop_reason="max_new_tokens",
            raw_transcript=tokenizer.decode(generated, skip_special_tokens=True).strip(),
            generated_token_ids=generated,
            generated_tokens=len(generated),
            max_new_tokens=limit,
        )
    return np.asarray(generated, dtype=np.int64)

def audio_generation_budget(duration_s: float, requested: int | None) -> int | None:
    """Return an explicit token cap, or scale with audio duration at 10 tokens/s."""
    if requested is not None:
        return int(requested)
    return min(65536, max(512, int(float(duration_s) * 10)))

def generation_limit(
    generation_config: Mapping[str, Any],
    prompt_length: int,
    *,
    cache_length: int,
    override: int | None = None,
) -> int:
    available = int(cache_length) - int(prompt_length)
    if available <= 0:
        raise ModelError(
            ErrorCode.INFERENCE_FAILED,
            "MOSS prompt leaves no room in the decoder KV cache",
        )
    del generation_config
    if override is None:
        return available
    requested = int(override)
    if requested <= 0:
        raise ModelError(
            ErrorCode.ARTIFACT_VALIDATION_FAILED,
            "MOSS generation max_new_tokens must be positive",
        )
    if requested > available:
        raise ModelError(
            ErrorCode.INFERENCE_FAILED,
            "MOSS requested output budget exceeds the remaining KV cache; re-export a larger context",
            prompt_length=prompt_length, requested_tokens=requested, available_tokens=available,
        )
    return requested

