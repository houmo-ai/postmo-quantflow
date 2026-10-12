# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""DiT component bundle contract; the complete workflow also supplies encoder/VAE."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True)
class DenoiseGear:
    encoder_tokens: int
    image_tokens: int
    prefix_tokens: int
    target_tokens: int
    text_tokens: int | None = None

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "text_tokens" and value is None:
                continue
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.image_tokens < self.target_tokens:
            raise ValueError("image projection capacity must cover the target tokens")

    @property
    def name(self):
        if self.text_tokens is not None:
            return f"s{self.text_tokens}_i{self.image_tokens}_c{self.cache_tokens}"
        return f"p{self.prefix_tokens}_t{self.target_tokens}"

    @property
    def cache_tokens(self):
        return self.prefix_tokens + self.target_tokens


def bundle_path(root, relative):
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ValueError("bundle paths must be nonempty POSIX relative paths")
    path = PurePosixPath(relative)
    if path.is_absolute() or any(part in (".", "..") for part in relative.split("/")):
        raise ValueError("bundle paths cannot escape the bundle root")
    root = Path(root)
    candidate = root
    for part in path.parts:
        candidate = candidate / part
        if candidate.is_symlink():
            raise ValueError("bundle resources cannot be symlinks")
    if not candidate.is_file():
        raise ValueError(f"missing bundle resource: {relative}")
    return candidate


def read_dit_metadata(root, metadata="runtime_meta_info.json"):
    meta = json.loads(bundle_path(root, metadata).read_text())
    if meta.get("schema_version") not in (1, 3, 4) or meta.get("kind") != "qwen_image21_dit":
        raise ValueError("unsupported Qwen-Image-2.1 DiT metadata")
    if meta.get("input_dtype") != "float16":
        raise ValueError("the HMONNX DiT bundle requires float16 graph inputs")
    gear = DenoiseGear(**meta["gear"])
    for key in ("layers", "heads", "head_dim", "context_channels", "latent_channels"):
        value = meta.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"invalid DiT architecture field {key}")
    expected = {"text", "image", "time", "head"}
    expected.update(f"{role}_{i}" for role in ("extract", "cached") for i in range(meta["layers"]))
    if meta["schema_version"] == 3:
        expected = {"extract", "cached"}
        if meta.get("cache_policy") != "replace_target":
            raise ValueError("complete DiT requires replace_target cache policy")
        if (
            meta.get("attention") != "flash_attention"
            or meta.get("attention_metadata") != "query_kv_range_abs"
            or meta.get("target_layout") != "actual_then_padding"
        ):
            raise ValueError("capacity DiT requires FlashAttention ranges and actual_then_padding targets")
    if meta["schema_version"] == 4:
        expected = {"text", "image"}
        if (
            meta.get("cache_policy") != "inplace_replace_target"
            or meta.get("attention") not in ("flash_attention", "causal_flash_noncausal_masked")
            or meta.get("attention_metadata") != "lengths"
            or meta.get("sequence_layout") != "compact_prefix_target"
            or meta.get("cache_tokens") != gear.cache_tokens
            or gear.text_tokens is None
        ):
            raise ValueError("segmented DiT requires in-place cache, compact layout and length metadata")
    if set(meta["graphs"]) != expected:
        raise ValueError("DiT bundle must contain every frontend and paired layer graph")
    for path in meta["graphs"].values():
        bundle_path(root, path)
    return meta, gear


class DenoiseBundle:
    """Load exported DiT graphs without importing diffusers or loading HF weights."""

    def __init__(self, root, *, device="cuda:0", metadata="runtime_meta_info.json"):
        from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

        from .runtime import LayerwiseDenoiser, TokenProjection

        self.meta, self.gear = read_dit_metadata(root, metadata)
        self.device = device
        if self.meta["schema_version"] in (3, 4):
            from xhquant.xhonnxruntime.llm_hmonnx_loader import MultiHMONNXLoader

            paths = {name: str(bundle_path(root, path)) for name, path in self.meta["graphs"].items()}
            # SDK shares only initializers with identical dtype, shape and bytes.
            # Request-owned K/V tensors remain separate from this weight pool.
            self.loader = MultiHMONNXLoader(paths)
            self.sessions = {
                name: HMONNXInferenceV2.from_onnx_graph(
                    path,
                    self.loader.graphs[name],
                    HMONNXInferenceConfig(exec_devices=[device]),
                )
                for name, path in paths.items()
            }
            if self.meta["schema_version"] == 4:
                for session in self.sessions.values():
                    if session.enable_inplace_cache_updates() != 2 * self.meta["layers"]:
                        raise ValueError("segmented DiT must update every layer's K/V in place")
            return
        sessions = {
            name: HMONNXInferenceV2(
                str(bundle_path(root, path)), session_config=HMONNXInferenceConfig(exec_devices=[device])
            )
            for name, path in self.meta["graphs"].items()
        }
        self.text = TokenProjection(
            sessions["text"], capacity=self.gear.encoder_tokens, input_channels=self.meta["context_channels"]
        )
        self.image = TokenProjection(
            sessions["image"], capacity=self.gear.image_tokens, input_channels=self.meta["latent_channels"]
        )
        self.time, self.head = sessions["time"], sessions["head"]
        self.stack = LayerwiseDenoiser(
            [sessions[f"extract_{i}"] for i in range(self.meta["layers"])],
            [sessions[f"cached_{i}"] for i in range(self.meta["layers"])],
            prefix_length=self.gear.prefix_tokens,
            target_length=self.gear.target_tokens,
        )

    def new_request(self, layout, compact_rope, text_features, reference_latents):
        import torch

        from .runtime import DenoiseRequest

        if layout.prefix_capacity != self.gear.prefix_tokens or layout.target_capacity != self.gear.target_tokens:
            raise ValueError("request layout does not match the exported DiT gear")
        if self.meta["schema_version"] == 4:
            from .segment_runtime import SegmentDenoiseRequest

            def graph(role):
                def invoke(tokens, time, cos, sin, past, current, caches):
                    return self.sessions[role](tokens, time, cos, sin, past, current, *caches)

                return invoke

            return SegmentDenoiseRequest(
                layout,
                compact_rope,
                text_features.to(device=self.device, dtype=torch.float16),
                reference_latents.to(device=self.device, dtype=torch.float16),
                text=graph("text"),
                image=graph("image"),
                gear=self.gear,
                layers=self.meta["layers"],
                heads=self.meta["heads"],
                head_dim=self.meta["head_dim"],
            )
        if self.meta["schema_version"] == 3:
            from .full_runtime import FullDenoiseRequest

            def cached(latents, time, cos, sin, mask, caches):
                return self.sessions["cached"](latents, time, cos, sin, mask, *caches)

            return FullDenoiseRequest(
                layout,
                compact_rope,
                text_features.to(device=self.device, dtype=torch.float16),
                reference_latents.to(device=self.device, dtype=torch.float16),
                extract=self.sessions["extract"],
                cached=cached,
                layers=self.meta["layers"],
                heads=self.meta["heads"],
                head_dim=self.meta["head_dim"],
            )
        return DenoiseRequest(
            layout,
            compact_rope,
            text_features.to(device=self.device, dtype=torch.float16),
            reference_latents.to(device=self.device, dtype=torch.float16),
            text_projection=self.text,
            image_projection=self.image,
            time_conditioning=self.time,
            output_head=self.head,
            extract=self.stack.extract,
            cached=self.stack.cached,
            layers=self.meta["layers"],
            heads=self.meta["heads"],
            head_dim=self.meta["head_dim"],
        )
