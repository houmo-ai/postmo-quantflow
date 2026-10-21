# Copyright Black Forest Labs
# Copyright 2026 HOUMO AI (modifications)
# SPDX-License-Identifier: Apache-2.0
# Reference image policy adapted from flux3/pup.py at upstream revision
# e53f4ea697fe45db8f38fca91a90f5623f7f8f37; see licenses/flux-3-image-klein/LICENSE.
"""Official prompt protocol and a deployment-only Qwen3.5 generation boundary."""

import io
import json
import math
import time

from PIL import Image, ImageOps
from transformers import LogitsProcessor


MAX_NEW_TOKENS = 3584


def generation_budget(input_length, component):
    """Cap generation by the actual exported context, retaining the protocol ceiling."""
    context, maximum = component["context_length"], component["max_new_tokens"]
    if any(type(value) is not int or value <= 0 for value in (input_length, context, maximum)):
        raise ValueError("prompt upsampler lengths and context must be positive integers")
    remaining = context - input_length
    if remaining <= 0:
        raise ValueError("prompt upsampler input leaves no output space in the exported context")
    return min(maximum, remaining)


def policy_reference(image, count):
    """Match BFL's RGB/white, 32-pixel geometry and JPEG-q92 view."""
    if not 1 <= count <= 10:
        raise ValueError("the prompt upsampler takes one to ten references")
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        image = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
    image = image.convert("RGB")
    width, height = image.size
    if max(width, height) / min(width, height) > 200:
        raise ValueError("upsampler reference aspect ratio exceeds 200:1")
    cap = 1024**2 if count == 1 else 724**2 if count == 2 else 512**2

    def floor(value):
        return max(32, int(math.floor(value / 32)) * 32)

    w, h = max(32, round(width / 32) * 32), max(32, round(height / 32) * 32)
    if w * h > cap:
        beta = math.sqrt(width * height / cap)
        w, h = floor(width / beta), floor(height / beta)
    if w * h > cap:
        if w == 32:
            h = floor(cap / w)
        else:
            w = floor(cap / h)
    if image.size != (w, h):
        image = image.resize((w, h), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    with Image.open(io.BytesIO(buffer.getvalue())) as decoded:
        return decoded.convert("RGB")


def prepare_messages(prompt, images, prompts, target_hw):
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty request")
    if images:
        content = [{"type": "image", "image": policy_reference(image, len(images))} for image in images]
        content.append(
            {
                "type": "text",
                "text": (
                    "Reference images follow in order (the first image, the second image, ...).\n"
                    f"User request: {prompt}\nOutput the caption."
                ),
            }
        )
    else:
        h, w = target_hw
        content = f"TARGET CANVAS: {w}:{h}\n{prompt}"
    return [{"role": "system", "content": prompts["i2i" if images else "t2i"]}, {"role": "user", "content": content}]


class PresencePenalty(LogitsProcessor):
    """vLLM presence penalty counts generated tokens, independently of frequency."""

    def __init__(self, prompt_length, penalty):
        self.prompt_length, self.penalty = prompt_length, penalty

    def __call__(self, input_ids, scores):
        result = scores.clone()
        for row in range(input_ids.shape[0]):
            generated = input_ids[row, self.prompt_length :].unique()
            result[row, generated] -= self.penalty
        return result


class PromptUpsampler:
    """Lazily load only packaged HMONNX forwards; no source/native fallback."""

    def __init__(self, root, component, *, device="cuda:0"):
        from .bundle import bundle_path

        self.root, self.component, self.device = root, component, device
        self.prompts = json.loads(bundle_path(root, component["system_prompts"]).read_text())
        self.model = None
        self.last_stats = {}
        self._golden_originals = {}

    def _load(self):
        if self.model is not None:
            return
        from xhmodel_merak._environment import scoped_environment
        from xhmodel_merak.xh_llm import AutoLLMHONNXModel

        from .bundle import bundle_path

        with scoped_environment({"ENABLE_HMINFERENCE_V2": "1", "XHQUANT_PACKED_W4": "1"}):
            self.model = AutoLLMHONNXModel.from_pretrained(
                str(bundle_path(self.root, self.component["metadata"])),
                device_map=[self.device],
                enable_golden=False,
                enable_cuda_graph=False,
                enable_auto_offload=False,
            )
            self.model.to(self.device)

    def __call__(self, prompt, images, target_hw):
        import torch
        from transformers import AutoProcessor, LogitsProcessorList

        from xhmodel_merak.xh_llm import LLMInferenceContextManager

        from .bundle import bundle_path

        processor = AutoProcessor.from_pretrained(
            bundle_path(self.root, self.component["processor_config"]).parent,
            local_files_only=True,
            trust_remote_code=False,
        )
        messages = prepare_messages(prompt, images, self.prompts, target_hw)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        # Transformers 5.16 emits this auxiliary classification field. The
        # Qwen3.5 deployment wrapper computes M-RoPE from ids/grid_thw itself.
        inputs.pop("mm_token_type_ids", None)
        length = inputs["input_ids"].shape[1]
        budget = generation_budget(length, self.component)
        self._load()
        tokenizer = processor.tokenizer
        stops = [tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
        started = time.perf_counter()
        with torch.inference_mode(), LLMInferenceContextManager(self.model, devices=[self.device]):
            output = self.model.generate(
                **inputs,
                max_new_tokens=budget,
                do_sample=False,
                eos_token_id=stops,
                pad_token_id=tokenizer.pad_token_id,
                logits_processor=LogitsProcessorList([PresencePenalty(length, 0.0 if images else 1.5)]),
            )
        tokens = output[0, length:].tolist()
        if not tokens or tokens[-1] not in stops:
            raise RuntimeError("prompt upsampler did not finish within the generation budget")
        caption = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        if not caption:
            raise RuntimeError("prompt upsampler returned an empty caption")
        grids = inputs.get("image_grid_thw")
        counts = (
            [] if grids is None else [int(grid.prod()) // processor.image_processor.merge_size**2 for grid in grids]
        )
        from xhmodel_merak.xh_llm.models.qwen3_5.visual_token_gears import select_image_token_gear

        self.last_stats = dict(
            input_tokens=length,
            output_tokens=len(tokens),
            latency_s=time.perf_counter() - started,
            context_length=self.component["context_length"],
            max_new_tokens=budget,
            reference_count=len(images),
            image_tokens=counts,
            visual_gears=[select_image_token_gear(count, self.model.visual.gears) for count in counts],
        )
        return caption

    def close(self):
        import gc

        self.cancel_golden()
        if self.model is not None:
            # The HF generation view points back to the deployment model.
            self.model.hf_compatible_model = None
        self.model = None
        gc.collect()

    def cancel_golden(self):
        for model, original in self._golden_originals.values():
            model.forward = original
        self._golden_originals.clear()

    def capture_golden(self, step, *, roles=None):
        """Capture one connected call per graph, rather than every generated token."""
        import gc

        from xhmodel_merak.xh_llm.utils import unfold_args
        from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

        from .bundle import bundle_path
        from .golden import materialize_graph_links
        from .upsampler_build import visual_graphs

        self._load()
        self.cancel_golden()
        meta = json.loads(bundle_path(self.root, self.component["metadata"]).read_text())
        models = {
            "prefill": self.model.prefill_model,
            "decode": self.model.decode_model,
            **{
                role: self.model.visual.models[gear["image_token_capacity"]]
                for role, gear in visual_graphs(self.component, meta).items()
            },
        }
        selected = set(models) if roles is None else set(roles)
        if not selected <= set(models):
            raise ValueError("unknown prompt Golden graph roles")
        for role in selected:
            directory = bundle_path(self.root, self.component["graphs"][role]).parent
            if (directory / f"step_{step}").exists():
                raise FileExistsError(f"prompt upsampler Golden already exists: {directory}")
        for role, model in models.items():
            if role not in selected:
                continue
            original = model.forward
            directory = bundle_path(self.root, self.component["graphs"][role]).parent
            path = str(bundle_path(self.root, self.component["graphs"][role]))
            self._golden_originals[role] = (model, original)

            def forward(*values, model=model, original=original, directory=directory, path=path, role=role):
                # V2 fixes Golden enablement when constructing its interpreter.
                # A one-call diagnostic session uses the exact graph and explicit
                # neural-state inputs, then normal execution resumes on V2.
                session = None
                try:
                    session = HMONNXInferenceV2(
                        path, session_config=HMONNXInferenceConfig(exec_devices=[self.device], enable_golden=True)
                    )
                    session.step = step
                    return session(*unfold_args(values))
                finally:
                    model.forward = original
                    self._golden_originals.pop(role, None)
                    materialize_graph_links(directory / f"step_{step}", self.root)
                    del session
                    gc.collect()

            model.forward = forward
