# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Source-free sequential HMONNX runtime for the complete Sol-H3 video pipeline."""

from __future__ import annotations

import gc
import json
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch

from .artifacts import artifact_path, read_metadata, resolve_ltx_audio_mode
from .host import (
    audio_mel,
    h3_rope,
    h3_updates,
    initial_h3,
    initial_ltx,
    ltx_updates,
    unpatch_h3,
    unpatch_ltx,
    upscale_h3,
)
from .precision import require_finite


class SolH3Runtime:
    def __init__(self, root, *, device="cuda:0"):
        self.root = Path(root).resolve()
        self.bundle_meta = read_metadata(self.root)
        self.meta = self.bundle_meta
        self.device = torch.device(device)
        self.sessions = {}
        self.stats = {}

    def json_resource(self, name):
        return json.loads(artifact_path(self.root, self.meta["resources"][name]).read_text())

    def numpy_resource(self, name):
        return np.load(artifact_path(self.root, self.meta["resources"][name]), allow_pickle=False)

    def invoke(self, name, *values):
        from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

        if name not in self.sessions:
            self.sessions[name] = HMONNXInferenceV2(
                str(artifact_path(self.root, self.meta["graphs"][name]["path"])),
                session_config=HMONNXInferenceConfig(exec_devices=[str(self.device)]),
            )
        session = self.sessions[name]
        names = session.get_input_names()
        if tuple(names) != tuple(self.meta["graphs"][name]["input_names"]):
            raise ValueError(f"{name} input ABI differs from the published runtime metadata")
        feed = {
            key: value.to(device=self.device, dtype=session.get_input(key).dtype)
            for key, value in zip(names, values, strict=True)
        }
        require_finite(tuple(feed.values()), name=f"{name} FP16 inputs")
        output = session.run(feed)
        outputs = output if isinstance(output, (list, tuple)) else (output,)
        require_finite(outputs, name=name)
        return tuple(value.detach().cpu() for value in outputs)

    def close(self):
        self.sessions.clear()
        gc.collect()
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                torch.cuda.empty_cache()

    def text(self, prompt):
        from transformers import AutoTokenizer

        from .request import tokenize_prompt

        tokenizer = AutoTokenizer.from_pretrained(self.root / "tokenizer", local_files_only=True)
        capacity = self.meta["text_capacity"]
        maximum = self.meta.get("prompt_max_tokens", capacity)
        ids = tokenize_prompt(prompt, tokenizer, maximum)
        embedding = np.load(
            artifact_path(self.root, self.meta["resources"]["text_embedding"]), mmap_mode="r", allow_pickle=False
        )
        hidden = torch.zeros(1, capacity, embedding.shape[1])
        hidden[:, : len(ids)] = torch.from_numpy(np.array(embedding[ids], copy=True)).float()
        inv = torch.from_numpy(self.numpy_resource("text_rope_inv_freq"))
        positions = torch.ones(capacity)
        positions[: len(ids)] = torch.arange(len(ids))
        phase = positions[:, None] * inv[None]
        phase = torch.cat((phase, phase), -1)[None]
        cos, sin, valid = phase.cos(), phase.sin(), torch.tensor([len(ids)], dtype=torch.int32)
        if "text_prefill_chunk_length" in self.meta:
            hidden, cos, sin = hidden[:, : len(ids)], cos[:, : len(ids)], sin[:, : len(ids)]
            additions = [torch.zeros_like(hidden) for _ in range(3)]
            hidden = self.feature_prefill(hidden, cos, sin, additions)
            hidden = torch.nn.functional.pad(hidden, (0, 0, 0, capacity - len(ids)))
            result = self.refine_text(hidden, len(ids))
            self.close()
            return result, len(ids)
        for name in self.meta["feature_graphs"]:
            hidden = self.invoke(name, hidden, cos, sin, valid)[0]
            # A text prefix component is used once; retain no GPU weights for it.
            self.close()
        result = self.refine_text(hidden, len(ids))
        self.close()
        return result, len(ids)

    def refine_text(self, features, length):
        capacity = self.meta["text_capacity"]
        if type(length) is not int or not 0 < length <= capacity or features.shape[1] != capacity:
            raise ValueError("text refiner requires a nonempty prefix fitting its fixed capacity")
        key_mask = torch.full((1, 1, 1, capacity), -65504.0, dtype=torch.float16)
        key_mask[..., :length] = 0
        return self.invoke("h3_text_refiner", features, key_mask)[0]

    def feature_prefill(self, hidden, cos, sin, additions):
        from .prefill import prefill_feature_graphs

        return prefill_feature_graphs(
            hidden,
            cos,
            sin,
            additions,
            metadata=self.meta,
            invoke=self.invoke,
            close=self.close,
            device=self.device,
        )

    def feature_encoder(self):
        from .conditioner import H3FeatureEncoder

        divisor = self.meta["text_feature_divisor"]
        embedding = torch.from_numpy(self.numpy_resource("text_embedding")).float() * divisor
        return H3FeatureEncoder(
            self.json_resource("qwen_config"),
            embedding,
            self.vision(),
            self.feature_prefill,
            feature_layer=self.meta["text_layers"],
            divisor=divisor,
        )

    def condition(self, prompt, *, image=None, last_image=None, references=None):
        from transformers import AutoProcessor, AutoTokenizer

        from .request import encode_request, prepare_configured_request

        tokenizer = AutoTokenizer.from_pretrained(self.root / "tokenizer", local_files_only=True)
        processor = AutoProcessor.from_pretrained(self.root / "processor", local_files_only=True)
        request = prepare_configured_request(
            prompt,
            config=self.json_resource("input_config"),
            tokenizer=tokenizer,
            image=image,
            last_image=last_image,
            references=references,
        )
        features, _ = encode_request(
            request,
            text_encoder=self.feature_encoder(),
            tokenizer=tokenizer,
            processor=processor,
            vae=self.reference_vae(),
            audio_vae=self.reference_audio() if request.mode == "ref2va" else None,
            text_encoder_layer=self.meta["text_layers"],
        )
        self.close()
        length = features.shape[1]
        features = torch.nn.functional.pad(
            features / self.meta["text_feature_divisor"], (0, 0, 0, self.meta["text_capacity"] - length)
        )
        text = self.refine_text(features, length)
        self.close()
        return text, length, request

    def reference_vae(self):
        from .reference_vae import ReferenceVAEAdapter, invoke_reference_tile

        def invoke(shape, pixels):
            return invoke_reference_tile(self.meta, shape, pixels, lambda name, *args: self.invoke(name, *args)[0])

        return ReferenceVAEAdapter(
            self.json_resource("reference_vae_config"),
            invoke,
            untiled=self.meta.get("mode") in ("i2va", "l2va", "fl2va"),
            **self.meta["reference_vae_policy"],
        )

    def vision(self):
        from .vision import VisionFeatureAdapter

        return VisionFeatureAdapter(
            lambda name, *values: self.invoke(name, *values),
            self.json_resource("qwen_vision_config"),
            gears=self.meta["visual_graphs"],
            device=self.device,
        )

    def reference_audio(self):
        from .reference_audio import ReferenceAudioAdapter

        return ReferenceAudioAdapter(
            self.json_resource("reference_audio_config"),
            lambda *values: self.invoke("h3_reference_audio_encode", *values)[0],
            max_frames=self.meta["reference_audio_frames"],
        )

    def draft(self, text, length, seed, *, request=None):
        from .duration import duration_gear

        gear = duration_gear(self.meta["draft_frames"], self.meta["refiner_frames"], self.meta["fps"])
        size = self.meta["draft_size"]
        if request is not None or self.meta.get("h3_packed_inputs", False):
            from .host import ConditionedDraft, RolePartitionedDraft

            options = dict(seed=seed, capacities=self.meta["capacity_plan"], rope_config=self.meta["h3_rope_config"])
            draft_type = ConditionedDraft
            layout = self.meta.get("h3_role_layout")
            if layout is not None:
                draft_type = RolePartitionedDraft
                options.update(
                    role_capacities=layout["role_capacities"],
                    attention_heads=layout["attention_heads"],
                    attention_block_size=layout["attention_block_size"],
                )
            draft = (
                draft_type(request, **options)
                if request is not None
                else draft_type.from_text(
                    size=size, text_length=length, video_frames=gear.h3_video_frames,
                    audio_frames=gear.h3_audio_frames, **options
                )
            )
            video, audio = draft.run(lambda *args: self.invoke("h3_denoiser", *args), text)
            self.close()
            audio = audio.reshape(2, gear.h3_audio_frames, 32).permute(0, 2, 1).contiguous()
            return unpatch_h3(video, size=size, frames=gear.h3_video_frames), audio
        video, audio = initial_h3(seed, size=size, video_frames=gear.h3_video_frames, audio_frames=gear.h3_audio_frames)
        cos, sin = h3_rope(
            length, text_capacity=self.meta["text_capacity"], size=size,
            video_frames=gear.h3_video_frames, audio_frames=gear.h3_audio_frames,
        )
        video, audio = h3_updates(lambda *args: self.invoke("h3_denoiser", *args), video, audio, text, cos, sin, length)
        self.close()
        audio = audio.reshape(2, gear.h3_audio_frames, 32).permute(0, 2, 1).contiguous()
        return unpatch_h3(video, size=size, frames=gear.h3_video_frames), audio

    def audio_decode(self, latent):
        config = self.json_resource("h3_audio_config")
        mean, std = (torch.tensor(config[key]).reshape(1, 32, 1) for key in ("latents_mean", "latents_std"))
        pcm = self.invoke("h3_audio_decode", latent.float() * std + mean)[0]
        self.close()
        return pcm

    def bridge(self, latent):
        highres = upscale_h3(
            lambda name, x: self.invoke(name, x)[0],
            latent,
            self.meta["upscaler_segments"],
            self.json_resource("upscaler_statistics"),
        )
        self.close()
        result = self.invoke("h3_to_ltx", highres)[0]
        self.close()
        return result

    def audio_encode(self, pcm):
        from .duration import duration_gear

        gear = duration_gear(self.meta["draft_frames"], self.meta["refiner_frames"], self.meta["fps"])
        mel = audio_mel(pcm, source_rate=32000, config=self.json_resource("ltx_audio_frontend"))
        encoded = self.invoke("ltx_audio_encode", mel)[0]
        self.close()
        result = torch.zeros(1, 8, gear.ltx_audio_frames, 16)
        count = min(gear.ltx_audio_frames, encoded.shape[2])
        result[:, :, :count] = encoded[:, :, :count]
        return result

    def ltx_image_encode(self, *, image=None, last_image=None):
        from .ltx_keyframes import prepare_keyframe_pixels

        policy = self.meta["ltx_reference_conditioning"]
        if not policy["enabled"]:
            return []
        self._validate_keyframe_resolution(image, last_image)
        latents = []
        try:
            for value, index in ((image, 0), (last_image, self.meta["refiner_frames"] - 1)):
                if value is not None:
                    pixels = prepare_keyframe_pixels(value, size=self.meta["refiner_size"], crf=policy["image_crf"])
                    latents.append((index, self.invoke("ltx_image_encode", pixels)[0]))
        finally:
            self.close()
        return latents

    def _validate_keyframe_resolution(self, image, last_image):
        policy = self.meta.get("ltx_reference_conditioning", {})
        if not policy.get("require_native_resolution", False):
            return
        from .ltx_keyframes import validate_keyframe_resolution

        for value in (image, last_image):
            if value is not None:
                validate_keyframe_resolution(value, self.meta["refiner_size"])

    def refine(self, video, audio, seed, *, image=None, last_image=None, ltx_audio_mode=None):
        selected_audio_mode = resolve_ltx_audio_mode(self.meta, ltx_audio_mode)
        freeze_audio = selected_audio_mode == "frozen"
        audio_mode_input = ()
        if "ltx_audio_policy" in self.meta:
            audio_mode_input = (
                torch.tensor([self.meta["ltx_audio_policy"]["modes"][selected_audio_mode]], dtype=torch.int32),
            )
        rope = ()
        if self.meta.get("ltx_rope_mode", "inputs") == "inputs":
            with self.numpy_resource("ltx_rope") as rope_data:
                rope = tuple(torch.from_numpy(rope_data[f"rope_{i}"]) for i in range(8))
        policy = self.meta["ltx_reference_conditioning"]
        if policy["graph_mode"] == "static_maximum_guide":
            from .conditioning import REFINER_SIGMAS
            from .ltx_keyframes import clear_static_keyframe_state, prepare_static_keyframe_state, static_keyframe_step

            keyframes = self.ltx_image_encode(image=image, last_image=last_image)
            state = prepare_static_keyframe_state(
                video, audio, keyframe_latents=keyframes, seed=seed, size=self.meta["refiner_size"],
                refiner_frames=self.meta["refiner_frames"], device=self.device, freeze_audio=freeze_audio
            )
            try:
                for step, (sigma, next_sigma) in enumerate(zip(REFINER_SIGMAS[:-1], REFINER_SIGMAS[1:], strict=True)):
                    v, a = self.invoke(
                        "ltx_denoiser",
                        state.video,
                        state.audio,
                        torch.tensor([step], dtype=torch.int32),
                        *rope,
                        state.first_role,
                        state.video_valid_length,
                        *audio_mode_input,
                    )
                    state = static_keyframe_step(state, v.to(self.device), a.to(self.device), sigma, next_sigma)
                return clear_static_keyframe_state(state).cpu()
            finally:
                self.close()
        video, audio = initial_ltx(video, audio, seed, device=self.device, freeze_audio=freeze_audio)
        try:
            video, audio = ltx_updates(
                lambda *args: self.invoke("ltx_denoiser", *args, *audio_mode_input),
                video, audio, rope, freeze_audio=freeze_audio,
            )
            return unpatch_ltx(video, size=self.meta["refiner_size"], frames=(self.meta["refiner_frames"] - 1) // 8 + 1)
        finally:
            self.close()

    def video_decode(self, latent):
        masks = self.numpy_resource("decoder_masks")
        output = torch.zeros(1, 3, self.meta["refiner_frames"], *self.meta["refiner_size"], dtype=torch.float32)
        weights = torch.zeros(1, 1, self.meta["refiner_frames"], *self.meta["refiner_size"], dtype=torch.float32)
        for i, tile in enumerate(self.meta["decoder_tiles"]):
            ic = tuple(slice(*value) for value in tile["input"])
            oc = tuple(slice(*value) for value in tile["output"])
            sample = latent[ic]
            if "video_decoder_mask" in self.meta or "video_decoder_masks" in self.meta:
                from .vae_graph import decoder_mask_inputs

                policy = (
                    self.meta["video_decoder_masks"][tile["graph"]]
                    if "video_decoder_masks" in self.meta else self.meta["video_decoder_mask"]
                )
                pixels = self.invoke(tile["graph"], *decoder_mask_inputs(sample, **policy))[0].float()
                pixels = pixels[..., : oc[3].stop - oc[3].start, : oc[4].stop - oc[4].start]
            else:
                pixels = self.invoke(tile["graph"], sample)[0].float()
            frames = min(pixels.shape[2], oc[2].stop - oc[2].start)
            oc = (*oc[:2], slice(oc[2].start, oc[2].start + frames), *oc[3:])
            pixels = pixels[:, :, :frames]
            mask = torch.ones((1, 1, 1, 1, 1))
            for axis in range(5):
                value = torch.from_numpy(masks[f"tile_{i}_axis_{axis}"])
                if axis == 2:
                    value = value[:frames]
                shape = [1] * 5
                shape[axis] = value.numel()
                mask = mask * value.reshape(shape)
            output[oc] += pixels * mask
            weights[(slice(None), slice(0, 1), *oc[2:])] += mask[:, :1]
        masks.close()
        self.close()
        if bool((weights <= 0).any()):
            raise ValueError("video decoder tile plans leave output pixels uncovered")
        output = output / weights
        require_finite(output, name="decoded video")
        return ((output[0].permute(1, 2, 3, 0) + 1) * 127.5).round().clamp(0, 255).to(torch.uint8).numpy()

    def generate(
        self, prompt, *, seed=42, output_path="sol_h3.mp4", image=None, last_image=None, references=None,
        ltx_audio_mode=None,
    ):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Sol-H3 requires a nonempty prompt")
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("Sol-H3 seed is outside the supported range")
        selected_audio_mode = resolve_ltx_audio_mode(self.bundle_meta, ltx_audio_mode)
        if self.device.type != "cuda":
            raise ValueError("this full Sol-H3 gear requires a CUDA HMONNX simulator")
        if self.bundle_meta.get("mode") == "all":
            from .bundle import mode_metadata
            from .request import infer_mode

            self.close()
            mode = infer_mode(image=image, last_image=last_image, references=references)
            self.meta = mode_metadata(self.bundle_meta, mode)
        if self.meta.get("mode") in ("i2va", "l2va", "fl2va"):
            self._validate_keyframe_resolution(image, last_image)
        stages = {}

        def stage(name, action):
            started = time.perf_counter()
            result = action()
            stages[name] = time.perf_counter() - started
            return result

        request = None
        if self.meta.get("mode", "t2va") == "t2va":
            if image is not None or last_image is not None or references is not None:
                raise ValueError("this T2VA artifact does not accept reference media")
            text, length = stage("text", lambda: self.text(prompt))
        else:
            text, length, request = stage(
                "condition", lambda: self.condition(prompt, image=image, last_image=last_image, references=references)
            )
        video, audio = stage("draft", lambda: self.draft(text, length, seed, request=request))
        pcm = stage("audio_decode", lambda: self.audio_decode(audio))
        video = stage("bridge", lambda: self.bridge(video))
        audio = stage("audio_encode", lambda: self.audio_encode(pcm))
        video = stage(
            "refiner",
            lambda: self.refine(
                video, audio, seed, image=image, last_image=last_image, ltx_audio_mode=selected_audio_mode
            ),
        )
        pixels = stage("video_decode", lambda: self.video_decode(video))
        if self.meta["refiner_frames"] != 121:
            from .host import audio_for_video

            pcm = audio_for_video(pcm, frames=len(pixels), fps=self.meta["fps"])
        path = Path(output_path).expanduser().resolve()
        stage("mux", lambda: write_video(path, pixels, pcm))
        return dict(
            output_path=path,
            stats=dict(
                stages=stages,
                draft_updates=4,
                refiner_updates=3,
                text_tokens=length,
                mode=self.meta.get("mode", "t2va"),
                quant_type=self.meta["quant_type"],
                attention="dense",
                refiner_cache_policy=self.meta["refiner_cache_policy"],
                ltx_audio_mode=selected_audio_mode,
                ltx_audio_frozen=selected_audio_mode == "frozen",
            ),
        )


def write_video(path, pixels, pcm):
    import av

    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = pcm.detach().float().reshape(2, -1).numpy()
    with av.open(str(path), mode="w") as container:
        video = container.add_stream("libx264", rate=24)
        video.width, video.height, video.pix_fmt = pixels.shape[2], pixels.shape[1], "yuv420p"
        video.options = {"crf": "19", "preset": "veryfast"}
        audio = container.add_stream("aac", rate=32000)
        audio.layout = "stereo"
        for index, image in enumerate(pixels):
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 24)
            for packet in video.encode(frame):
                container.mux(packet)
        for start in range(0, waveform.shape[1], 1024):
            frame = av.AudioFrame.from_ndarray(
                np.ascontiguousarray(waveform[:, start : start + 1024]), format="fltp", layout="stereo"
            )
            frame.sample_rate, frame.pts, frame.time_base = 32000, start, Fraction(1, 32000)
            for packet in audio.encode(frame):
                container.mux(packet)
        for stream in (video, audio):
            for packet in stream.encode(None):
                container.mux(packet)
