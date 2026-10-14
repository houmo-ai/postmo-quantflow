"""Run the Flash target or its DSpark engine through the public text API."""
from pathlib import Path
import time
from dataclasses import asdict

from xhmodel_merak.model_inference.contracts import TextInferenceResult
from xhmodel_merak.model_inference.inputs import validate_inputs


class _DeepSeekV4Adapter:
    def run(self, *, recipe, root, manifest, meta_file, inputs, device, components,
            selected_golden, golden_artifacts):
        normalized = validate_inputs(recipe, inputs)
        variant = manifest.get("recipe", {}).get("variant", "full")
        text, count, latency, speed, stats = run_dspark_generation(
            meta_file, prompt=normalized["prompt"], max_new_tokens=normalized["max_new_tokens"],
            temperature=normalized["temperature"], device=device, devices=normalized.get("devices"),
            use_dspark=variant == "dspark")
        return TextInferenceResult(model_id=recipe.model_id, task=recipe.task, output_type="text",
            artifact_dir=str(root), meta_file=str(meta_file), latency_s=latency, components=components,
            golden_components=selected_golden, golden_artifacts=golden_artifacts,
            output_text=text, output_tokens=count, tokens_per_second=speed, stats=stats,
            spec_decode_mode="dspark" if variant == "dspark" else None)


def run_dspark_generation(meta_file: str | Path, *, prompt: str, max_new_tokens: int,
                          temperature: float, device: str, devices=None, use_dspark=True):
    import torch
    from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
    from xhmodel_merak._environment import scoped_environment

    with scoped_environment({"ENABLE_HMINFERENCE_V2": "1", "XHQUANT_PACKED_W4": "1"}):
        requested = torch.device(device)
        execution_devices = [torch.device(value) for value in devices] if devices else [requested]
        if not devices and requested.type == "cuda":
            first = requested.index or 0
            if first + 1 < torch.cuda.device_count():
                execution_devices.append(torch.device(f"cuda:{first + 1}"))
        model = AutoLLMHONNXModel.from_pretrained(str(meta_file), device_map=execution_devices,
            enable_auto_offload=len(execution_devices) > 1,
            enable_cuda_graph=all(value.type == "cuda" for value in execution_devices),
            enable_prefill_cuda_graph=False,
            enable_decode_cuda_graph=all(value.type == "cuda" for value in execution_devices))
        model.to(execution_devices[0])
        from xhmodel_merak.xh_llm.models.deepseek_v4.chat import (
            configure_flash_chat_tokenizer, decode_flash_thinking_completion,
        )

        tokenizer = configure_flash_chat_tokenizer(model.get_tokenizer())
        prompt_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                     tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=True, reasoning_effort="max")
        ids = tokenizer(prompt_text, return_tensors="pt", return_token_type_ids=False)["input_ids"].to(execution_devices[0])
        if ids.shape[1] + max_new_tokens > model.meta_info.model_config.context_max_length:
            raise ValueError("DSpark prompt plus output budget exceeds the target context")
        engine = model.build_dspark_engine(device=str(execution_devices[-1])) if use_dspark else None
        eos = tokenizer.eos_token_id
        eos_ids = (eos,) if isinstance(eos, int) else tuple(eos or ())
        for value in execution_devices:
            if value.type == "cuda":
                torch.cuda.synchronize(value)
        started = time.perf_counter()
        with LLMInferenceContextManager(model, devices=execution_devices):
            if engine is not None:
                output = engine.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature,
                                          eos_token_ids=eos_ids)
            else:
                generation = {"max_new_tokens": max_new_tokens, "do_sample": temperature > 0,
                              "top_p": 1.0, "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id}
                if temperature > 0:
                    generation["temperature"] = temperature
                output = model.generate(input_ids=ids, **generation)
        for value in execution_devices:
            if value.type == "cuda":
                torch.cuda.synchronize(value)
        latency = time.perf_counter() - started
        generated = output[0, ids.shape[1]:]
        count = generated.numel()
        statistics = asdict(engine.statistics) if engine is not None else {}
        if engine is not None:
            statistics["acceptance_rate"] = engine.statistics.acceptance_rate
        answer, reasoning = decode_flash_thinking_completion(tokenizer, generated.tolist())
        statistics.update(reasoning)
        return answer, count, latency, count / latency, statistics


adapter = _DeepSeekV4Adapter()
