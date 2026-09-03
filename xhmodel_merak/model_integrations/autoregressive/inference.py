from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

from xhmodel_merak._environment import scoped_environment


def run_text_generation(
    meta_file: Path,
    *,
    prompt: str,
    max_new_tokens: int,
    device: str,
    device_map: bool,
    pack_w4: bool,
    cuda_graph: bool = False,
) -> tuple[str, int, float, float, dict[str, Any]]:
    import torch

    requested_device = torch.device(device)
    device_context = torch.cuda.device(requested_device) if requested_device.type == "cuda" else nullcontext()
    with (
        device_context,
        scoped_environment(
            {
                "ENABLE_HMINFERENCE_V2": "1",
                "XHQUANT_PACKED_W4": "1" if pack_w4 else "0",
            }
        ),
    ):
        from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager

        kwargs: dict[str, Any] = {}
        if device_map:
            kwargs["device_map"] = [requested_device]
        if cuda_graph and requested_device.type == "cuda":
            kwargs["enable_cuda_graph"] = True
        model = AutoLLMHONNXModel.from_pretrained(str(meta_file), **kwargs)
        tokenizer = model.get_tokenizer(trust_remote_code=True)
        messages = [{"role": "user", "content": prompt}]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer(
            [text],
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(requested_device)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        model.to(requested_device)
        _synchronize(requested_device)
        started = time.perf_counter()
        with LLMInferenceContextManager(model, devices=[requested_device]):
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        _synchronize(requested_device)
        latency_s = time.perf_counter() - started
        prompt_length = int(model_inputs.input_ids.shape[1])
        output_ids = generated_ids[0, prompt_length:]
        output_tokens = int(output_ids.numel())
        output_text = tokenizer.decode(output_ids.tolist(), skip_special_tokens=True).strip()
        return (
            output_text,
            output_tokens,
            latency_s,
            output_tokens / latency_s if latency_s > 0 else 0.0,
            {"prompt_tokens": prompt_length},
        )


def _synchronize(device) -> None:
    import torch

    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


__all__ = ["run_text_generation"]
