"""对照生产路径，逐步演示已导出的 MOSS HMONNX 图是怎么跑的。

加速点不在「整段 ASR 一张图」。导出产物是三张 XH2a 图；Host 仍做 log-mel、
音/文融合和 greedy ``lm_head``。芯片上只有：

* encode：固定 30s Whisper 窗 → 音频 embedding
* prefill：融合后的 embedding 按块写入共享 KV
* decode ：每次一个 token embedding，复用同一份 KV

只看 metadata、不跑芯片::

    PYTHONPATH=$PWD python examples_merak/asr/moss_transcribe_diarize/moss_transcribe_diarize_hmonnx_infer.py \
        --artifact-dir <export_or_build_dir>

在 XH2a 设备上真正跑图（不要用 CPU）::

    PYTHONPATH=$PWD python examples_merak/asr/moss_transcribe_diarize/moss_transcribe_diarize_hmonnx_infer.py \
        --artifact-dir <export_or_build_dir> \
        --audio <16k_mono_pcm16.wav> \
        --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

META_NAME = "export_meta_info.json"


def resolve_artifact(path: Path) -> tuple[Path, dict[str, Any]]:
    """定位产物根目录。workflow 的 output_dir 和 build_model 包都能用。"""
    root = path.expanduser().resolve()
    direct = root / META_NAME
    if direct.is_file():
        return root, json.loads(direct.read_text(encoding="utf-8"))
    found = [candidate for candidate in root.rglob(META_NAME) if candidate.is_file()]
    if len(found) == 1:
        return found[0].parent, json.loads(found[0].read_text(encoding="utf-8"))
    if len(found) > 1:
        raise FileNotFoundError(
            f"multiple {META_NAME} under {root}; pass the component package directory"
        )
    raise FileNotFoundError(f"{META_NAME} not found under {root}")


def _component_row(spec: dict[str, Any]) -> dict[str, Any]:
    """抽出推理时真正会用到的图路径和几何字段。"""
    return {
        "hmonnx_file": spec.get("hmonnx_file"),
        "onnx_file": spec.get("onnx_file"),
        "cache_mode": spec.get("cache_mode"),
        "prefill_length": spec.get("prefill_length"),
        "cache_length": spec.get("cache_length"),
        "rope_mode": spec.get("rope_mode"),
        "enable_page_attention": spec.get("enable_page_attention"),
        "quant_type": spec.get("quant_type"),
    }


def summarize_hmonnx_plan(meta: dict[str, Any]) -> dict[str, Any]:
    """从 metadata 归纳：三张图怎么调度、KV 归谁、Host 还剩哪些工作。"""
    components = meta.get("components") if isinstance(meta.get("components"), dict) else {}
    encode = components.get("encode") if isinstance(components.get("encode"), dict) else {}
    decoder = (
        components.get("prefill_decode")
        if isinstance(components.get("prefill_decode"), dict)
        else {}
    )
    prefill = decoder.get("prefill") if isinstance(decoder.get("prefill"), dict) else {}
    decode = decoder.get("decode") if isinstance(decoder.get("decode"), dict) else {}
    page = bool(prefill.get("enable_page_attention") and decode.get("enable_page_attention"))
    inplace = prefill.get("cache_mode") == "inplace" and decode.get("cache_mode") == "inplace"
    if page:
        kv_owner = "runtime PageAttention pages (HMFP), shared by prefill and decode"
    elif inplace:
        kv_owner = "runtime CacheTensor, written in place, not copied back to Host each step"
    else:
        kv_owner = "Host tensors: present_key/value_cache returned and fed back"
    return {
        "quant_type": meta.get("quant_type"),
        "encode_source": encode.get("hmonnx_file") or encode.get("onnx_file"),
        "decoder_source": "hmonnx" if prefill.get("hmonnx_file") else "onnx",
        "lm_head_file": meta.get("lm_head_file"),
        "kv_owner": kv_owner,
        "graphs": {
            "encode": _component_row(encode),
            "prefill": _component_row(prefill),
            "decode": _component_row(decode),
        },
        "host_still_runs": [
            "log-mel and 30s Whisper windows",
            "audio/text embedding fusion into <|audio_pad|> slots",
            "greedy lm_head on last hidden state",
        ],
        "hmonnx_runs": [
            "encode window: input_features -> audio_embeds",
            "prefill chunks of fused embeddings into KV",
            "decode one token embedding per step",
        ],
        "session_notes": [
            "CPU cannot execute these XH2a graphs",
            "64k FA graphs convert to PageAttention at load unless --disable-page-attention",
        ],
    }


def format_plan(plan: dict[str, Any]) -> str:
    lines = ["=== published MOSS HMONNX plan ==="]
    lines.append(
        f"quant_type={plan.get('quant_type')} encode_source={plan.get('encode_source')} "
        f"decoder_source={plan.get('decoder_source')}"
    )
    if plan.get("lm_head_file"):
        lines.append(f"lm_head (Host): {plan['lm_head_file']}")
    lines.append(f"KV ownership: {plan.get('kv_owner')}")
    lines.append("graphs:")
    graphs = plan.get("graphs") if isinstance(plan.get("graphs"), dict) else {}
    for component, spec in graphs.items():
        row = spec if isinstance(spec, dict) else {}
        lines.append(
            "  {component}: hmonnx={hmonnx_file} onnx={onnx_file} cache={cache_mode} "
            "prefill_len={prefill_length} cache_len={cache_length} rope={rope_mode} "
            "page_attn={enable_page_attention} quant={quant_type}".format(
                component=component, **{key: row.get(key) for key in (
                    "hmonnx_file", "onnx_file", "cache_mode", "prefill_length",
                    "cache_length", "rope_mode", "enable_page_attention", "quant_type",
                )}
            )
        )
    lines.append("\nHost still runs:")
    lines.extend(f"  - {item}" for item in plan.get("host_still_runs") or ())
    lines.append("\nHMONNX session.run:")
    lines.extend(f"  - {item}" for item in plan.get("hmonnx_runs") or ())
    lines.append("\nRuntime notes:")
    lines.extend(f"  - {item}" for item in plan.get("session_notes") or ())
    return "\n".join(lines)


def _shape_of(value: Any) -> str:
    shape = getattr(value, "shape", None)
    if shape is None:
        return type(value).__name__
    return str(tuple(int(dim) for dim in shape))


def _report_hmonnx_timings(runtime: Any) -> None:
    lines = ["=== HMONNX wall time ==="]
    for name in ("encode", "prefill", "decode"):
        count = int(runtime.counts.get(name, 0))
        total = float(runtime.totals_s.get(name, 0.0))
        if not count:
            continue
        lines.append(f"  {name}: {count} calls, {total:.3f}s total, {1000 * total / count:.1f} ms/call")
    print("\n".join(lines), flush=True)


def _page_attention_flag(args: argparse.Namespace) -> bool | None:
    if args.enable_page_attention:
        return True
    if args.disable_page_attention:
        return False
    return None


def _run_hmonnx(
    root: Path,
    meta: dict[str, Any],
    audio: Path,
    device: str,
    prompt: str | None,
    max_new_tokens: int | None,
    enable_page_attention: bool | None,
) -> str:
    """打开三张 HMONNX session，再走与 ``inference.py`` 相同的 Host 编排。"""
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.hmonnx_runtime import MossHMONNXRuntime
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.transcribe import transcribe_artifact

    class TracingMossHMONNXRuntime(MossHMONNXRuntime):
        """只包一层计时：encode/prefill/decode 都落到 ``run`` → ``session.run``。"""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.totals_s: dict[str, float] = defaultdict(float)
            self.counts: dict[str, int] = defaultdict(int)

        def run(self, component: str, inputs: dict[str, Any]) -> Any:
            import torch

            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = super().run(component, inputs)
            if str(self.device).startswith("cuda") and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            self.totals_s[component] += elapsed
            self.counts[component] += 1
            incoming = ", ".join(f"{key}={_shape_of(value)}" for key, value in inputs.items())
            if isinstance(outputs, dict):
                outgoing = ", ".join(f"{key}={_shape_of(value)}" for key, value in outputs.items())
            else:
                outgoing = type(outputs).__name__
            print(
                f"[hmonnx {component} #{self.counts[component]}] {elapsed * 1000:.1f} ms "
                f"| in {{ {incoming} }} | out {{ {outgoing} }}",
                flush=True,
            )
            return outputs

    runtime = TracingMossHMONNXRuntime(
        root, meta, device, enable_page_attention=enable_page_attention,
    )
    try:
        if not runtime.available():
            raise RuntimeError(
                "HMONNX graphs are not executable on this device. Use a non-CPU XH2a device; "
                "CPU inference is the PyTorch reference graph, not these files."
            )
        print(f"opening sessions on {device} (first call compiles / loads graphs)...", flush=True)
        started = time.perf_counter()
        text = transcribe_artifact(
            root=root,
            meta=meta,
            audio=audio,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            runtime=runtime,
        )
        print(f"sessions ready in {time.perf_counter() - started:.2f}s", flush=True)
        print(
            "prefill inputs_embeds "
            f"{runtime.input_shape('prefill', 'inputs_embeds')} | inplace_cache={runtime.uses_inplace_cache()} "
            f"page_attention={runtime.uses_page_attention()} cache_length={runtime.decoder_cache_length()}",
            flush=True,
        )
        _report_hmonnx_timings(runtime)
        return text
    finally:
        runtime.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--artifact-dir", required=True, help="Export dir or unified build_model output.")
    parser.add_argument("--audio", default="", help="16 kHz mono 16-bit PCM WAV/FLAC. Omit to only inspect metadata.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument(
        "--enable-page-attention",
        action="store_true",
        help="Force load-time PageAttention convert (64k FA greedy).",
    )
    parser.add_argument(
        "--disable-page-attention",
        action="store_true",
        help="Keep the exported FA/inplace graphs; do not convert to PA.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.enable_page_attention and args.disable_page_attention:
        raise SystemExit("use only one of --enable-page-attention / --disable-page-attention")
    root, meta = resolve_artifact(Path(args.artifact_dir))
    print(f"artifact root: {root}", flush=True)
    print(format_plan(summarize_hmonnx_plan(meta)), flush=True)
    if not args.audio:
        print(
            "\nNo --audio given; stopping after inspect. Pass --audio and --device to execute the graphs.",
            flush=True,
        )
        return
    text = _run_hmonnx(
        root,
        meta,
        Path(args.audio),
        args.device,
        args.prompt or None,
        args.max_new_tokens,
        _page_attention_flag(args),
    )
    print("=== transcript ===", flush=True)
    print(text, flush=True)


if __name__ == "__main__":
    main()
