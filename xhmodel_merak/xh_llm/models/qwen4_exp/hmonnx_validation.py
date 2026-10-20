"""Qwen4 text-generation report; SDK Golden tensor capture stays in the runtime."""

import json
from pathlib import Path
from typing import Any

from ..qwen3_5.hmonnx_validation import HMONNXQuickTestResult
from ..qwen3_5.hmonnx_validation import hmonnx_generate as _generate


def hmonnx_generate(*, meta_file: str | Path, golden_output_dir: str | Path | None = None, **kwargs):
    """Reuse shared generation and optionally write Qwen4's JSON/TXT report."""
    result = _generate(meta_file=meta_file, **kwargs)
    if kwargs.get("golden", False) or golden_output_dir is not None:
        flags = ("do_sample", "think", "fast", "golden", "debug", "auto_offload", "cuda_graph", "page_attention")
        _dump_hmonnx_golden_output(
            result=result,
            prompt=kwargs.get("prompt", "用中文一句话说明模型是否可用。"),
            image_path=kwargs.get("image_path"),
            output_dir=golden_output_dir,
            extra={flag: kwargs.get(flag, False) for flag in flags},
        )
    return result


def _dump_hmonnx_golden_output(
    *,
    result: HMONNXQuickTestResult,
    prompt: str,
    image_path: str | None,
    output_dir: str | Path | None,
    extra: dict[str, Any] | None = None,
) -> None:
    target_dir = Path(output_dir) if output_dir is not None else Path(result.meta_file).parent / "golden"
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta_file": result.meta_file,
        "prompt": prompt,
        "image_path": image_path,
        "output_text": result.output_text,
        "output_tokens": result.output_tokens,
        "latency_s": result.latency_s,
        "tokens_per_second": result.tokens_per_second,
        "spec_decode_mode": result.spec_decode_mode,
        "block_size": result.block_size,
        "num_rounds": result.num_rounds,
        "draft_tokens_total": result.draft_tokens_total,
        "accepted_drafts_total": result.accepted_drafts_total,
        "accept_rate": result.accept_rate,
        "avg_accepted_per_round": result.avg_accepted_per_round,
        "stats": result.stats,
    }
    if extra:
        payload.update(extra)
    (target_dir / "golden_output.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (target_dir / "golden_output.txt").write_text(result.output_text + "\n", encoding="utf-8")
