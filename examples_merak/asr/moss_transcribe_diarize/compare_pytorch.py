"""Compare official HF greedy against a published MOSS HMONNX artifact.

This is a 30-second quality smoke check, not AliMeeting scoring. It prints
prefix agreement between the two transcripts. Load-bearing CER stays in the
eval harness under work_dirs/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def transcribe_pytorch(
    audio: Path,
    model_dir: Path,
    device: str,
    prompt: str | None,
    max_new_tokens: int | None,
) -> str:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor

    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.processing import DEFAULT_PROMPT, load_audio

    waveform = load_audio(audio)
    instruction = (prompt or DEFAULT_PROMPT).strip() or DEFAULT_PROMPT
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device=device, dtype=torch.bfloat16).eval()
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    messages = [{"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": instruction}]}]
    chat = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=chat, audio=np.asarray(waveform), return_tensors="pt")
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    prompt_len = int(inputs["input_ids"].shape[1])
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens or 512, do_sample=False
        )
    return processor.tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True).strip()


def transcribe_hmonnx(
    audio: Path,
    artifact_dir: Path,
    device: str,
    prompt: str | None,
    enable_page_attention: bool | None,
    max_new_tokens: int | None = None,
) -> str:
    import json as json_lib

    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.hmonnx_runtime import MossHMONNXRuntime
    from xhmodel_merak.xh_llm.models.moss_transcribe_diarize.transcribe import transcribe_artifact

    meta = json_lib.loads((artifact_dir / "export_meta_info.json").read_text(encoding="utf-8"))
    runtime = MossHMONNXRuntime(
        artifact_dir, meta, device, enable_page_attention=enable_page_attention,
    )
    try:
        return transcribe_artifact(
            root=artifact_dir,
            meta=meta,
            audio=audio,
            device=device,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            runtime=runtime,
        )
    finally:
        runtime.close()


def compare_texts(pytorch_text: str, hmonnx_text: str) -> dict[str, Any]:
    prefix = 0
    for left, right in zip(pytorch_text, hmonnx_text):
        if left != right:
            break
        prefix += 1
    overlap = min(len(pytorch_text), len(hmonnx_text))
    agree = sum(left == right for left, right in zip(pytorch_text, hmonnx_text))
    return {
        "pytorch_chars": len(pytorch_text),
        "hmonnx_chars": len(hmonnx_text),
        "prefix_chars": prefix,
        "agree_chars": agree,
        "agree_ratio": None if not overlap else agree / overlap,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="16 kHz mono 16-bit PCM WAV/FLAC, typically a 30s clip.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--model-dir", required=True, help="Official HF MOSS-Transcribe-Diarize directory.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default="")
    parser.add_argument("--enable-page-attention", action="store_true")
    parser.add_argument("--disable-page-attention", action="store_true")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Cap generated tokens. Omit for HMONNX duration scaling; PyTorch defaults to 512.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.enable_page_attention and args.disable_page_attention:
        raise SystemExit("use only one of --enable-page-attention / --disable-page-attention")
    page_attention = True if args.enable_page_attention else False if args.disable_page_attention else None
    audio = Path(args.audio)
    prompt = args.prompt or None
    pytorch_text = transcribe_pytorch(
        audio, Path(args.model_dir), args.device, prompt, args.max_new_tokens,
    )
    hmonnx_text = transcribe_hmonnx(
        audio,
        Path(args.artifact_dir),
        args.device,
        prompt,
        page_attention,
        args.max_new_tokens,
    )
    summary = {
        "pytorch_text": pytorch_text,
        "hmonnx_text": hmonnx_text,
        **compare_texts(pytorch_text, hmonnx_text),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
