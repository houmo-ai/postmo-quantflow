# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Native H3 generation with selectable packaged keep-LoRA graph views."""

import time
from pathlib import Path

import torch

from .conditioning import ConditionedH3Schedule
from .host import ConditionedDraft, unpatch_h3
from .native_artifacts import mode_metadata, read_metadata
from .native_video import NativeVideoDecodeAdapter
from .precision import require_finite
from .request import infer_mode
from .runtime import SolH3Runtime, write_video


class NativeH3Runtime(SolH3Runtime):
    def __init__(self, root, *, device="cuda:0"):
        self.root = Path(root).resolve()
        self.bundle_meta = read_metadata(self.root)
        default_mode = "t2va" if self.bundle_meta["mode"] == "all" else self.bundle_meta["mode"]
        self.meta = mode_metadata(self.bundle_meta, default_mode)
        self.device = torch.device(device)
        self.sessions = {}
        self.stats = {}

    def select_adapter(self, name=None, *, mode=None):
        # Validate before dropping a current view; switching cannot retain old KV
        # state or sessions holding the preceding adapter's static weights.
        view = mode_metadata(self.bundle_meta, self.meta["mode"] if mode is None else mode, name)
        self.close()
        self.meta = view

    def draft(self, text, length, seed, *, request=None):
        options = dict(seed=seed, capacities=self.meta["capacity_plan"], rope_config=self.meta["h3_rope_config"])
        draft = (
            ConditionedDraft(request, **options)
            if request is not None
            else ConditionedDraft.from_text(size=self.meta["draft_size"], text_length=length, **options)
        )
        schedule = ConditionedH3Schedule(tuple(self.meta["h3_sigmas"]), tuple(self.meta["audio_sigmas"]))
        video, audio = draft.run(lambda *args: self.invoke("h3_denoiser", *args), text, schedule=schedule)
        self.close()
        return unpatch_h3(video, size=self.meta["draft_size"]), audio.reshape(2, 207, 32).permute(0, 2, 1).contiguous()

    def video_decode(self, latent):
        policy = self.meta["native_video_decoder"]

        def tile(shape, value):
            key = "x".join(str(n) for n in shape)
            if key not in policy["tiles"]:
                raise ValueError(f"Native H3 decoder tile is not exported: {key}")
            return self.invoke(policy["tiles"][key], value)[0]

        decoder = NativeVideoDecodeAdapter(
            self.json_resource(policy["config"]), tile, tile_size=policy["tile_size"], overlap=policy["overlap"]
        )
        pixels = decoder(latent)
        self.close()
        require_finite(pixels, name="native H3 decoded video")
        expected = (1, 3, self.meta["draft_frames"], *self.meta["draft_size"])
        if tuple(pixels.shape) != expected:
            raise ValueError(f"Native H3 video shape differs from metadata: {tuple(pixels.shape)} != {expected}")
        return (pixels[0].permute(1, 2, 3, 0) * 255).round().clamp(0, 255).to(torch.uint8).numpy()

    def generate(
        self, prompt, *, seed=42, output_path="h3.mp4", image=None, last_image=None, references=None, lora_adapter=None
    ):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Native H3 requires a nonempty prompt")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Native H3 seed is outside the supported range")
        if self.device.type != "cuda":
            raise ValueError("Native H3 requires a CUDA HMONNX simulator")
        mode = infer_mode(image=image, last_image=last_image, references=references)
        self.select_adapter(lora_adapter, mode=mode)
        stages = {}

        def stage(name, action):
            started = time.perf_counter()
            result = action()
            stages[name] = time.perf_counter() - started
            return result

        request = None
        if mode == "t2va":
            text, length = stage("text", lambda: self.text(prompt))
        else:
            text, length, request = stage(
                "condition", lambda: self.condition(prompt, image=image, last_image=last_image, references=references)
            )
        video, audio = stage("denoise", lambda: self.draft(text, length, seed, request=request))
        pcm = stage("audio_decode", lambda: self.audio_decode(audio))
        pixels = stage("video_decode", lambda: self.video_decode(video))
        path = Path(output_path).expanduser().resolve()
        stage("mux", lambda: write_video(path, pixels, pcm))
        return dict(
            output_path=path,
            stats=dict(
                stages=stages,
                draft_updates=self.meta["draft_steps"],
                refiner_updates=0,
                text_tokens=length,
                mode=mode,
                lora_adapter=lora_adapter,
                quant_type=self.meta["quant_type"],
                attention="dense",
            ),
        )
