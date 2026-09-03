from __future__ import annotations

import json
import math
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torch.nn import functional as F


MASK_VALUE = -65_504.0


class PI05CompactRuntime:
    """Self-contained PI0.5 HMONNX runtime backed only by published assets."""

    def __init__(
        self,
        artifact_dir: str | Path,
        *,
        device: str = "cuda:0",
        manifest: Mapping[str, Any] | None = None,
    ) -> None:
        from xhquant.core import CacheTensor

        self.root = Path(artifact_dir).expanduser().resolve()
        self.device = torch.device(device)
        self.manifest = dict(manifest or {})
        self.CacheTensor = CacheTensor
        self.meta = _load_json(_confined_file(self.root, "export_meta_info.json", self.manifest))
        policy_config_path = _confined_file(self.root, "runtime/policy/config.json", self.manifest)
        self.policy_config = _load_json(policy_config_path)
        self.contract = dict(self.meta["compact_prefix"])
        self.selected_image_indices = tuple(int(value) for value in self.contract["selected_image_indices"])
        self.horizon = int(self.contract["action_horizon"])
        self.cache_length = int(self.contract["cache_length"])
        self.text_max_length = int(self.contract["text_max_length"])
        self.action_dim = int(self.policy_config["output_features"]["action"]["shape"][0])
        self.state_dim = int(self.policy_config["input_features"]["observation.state"]["shape"][0])
        self.max_action_dim = int(self.policy_config["max_action_dim"])
        self.num_inference_steps = int(self.policy_config.get("num_inference_steps", 10))
        self.image_resolution = tuple(int(value) for value in self.policy_config["image_resolution"])
        if self.selected_image_indices != (0, 1):
            raise ValueError("PI0.5 LIBERO runtime requires both registered camera inputs in order")
        if self.horizon != 50 or self.action_dim != 7 or self.state_dim != 8:
            raise ValueError(
                "PI0.5 LIBERO runtime dimensions do not match the public contract: "
                f"horizon={self.horizon}, action_dim={self.action_dim}, state_dim={self.state_dim}"
            )
        if self.max_action_dim < self.action_dim or self.num_inference_steps <= 0:
            raise ValueError("PI0.5 runtime max_action_dim or num_inference_steps is invalid")

        tokenizer_path = _confined_dir(self.root, "runtime/tokenizer", self.manifest)
        self.tokenizer = _load_tokenizer(tokenizer_path, self.text_max_length)
        state_mode = _normalization_mode(self.policy_config, "STATE")
        action_mode = _normalization_mode(self.policy_config, "ACTION")
        self.state_stats = _load_stats(
            _confined_file(
                self.root,
                "runtime/policy/policy_preprocessor_step_2_normalizer_processor.safetensors",
                self.manifest,
            ),
            "observation.state",
            state_mode,
        )
        self.action_stats = _load_stats(
            _confined_file(
                self.root,
                "runtime/policy/policy_postprocessor_step_0_unnormalizer_processor.safetensors",
                self.manifest,
            ),
            "action",
            action_mode,
        )
        _validate_stats(self.state_stats, self.state_dim, "observation.state")
        _validate_stats(self.action_stats, self.action_dim, "action")

        paths, gemma_meta = self._runtime_paths()
        token_embedding_state = torch.load(paths.pop("token_embedding"), map_location="cpu", weights_only=True)
        self.token_embedding = nn.Embedding(
            int(token_embedding_state["weight"].shape[0]),
            int(token_embedding_state["weight"].shape[1]),
        )
        self.token_embedding.load_state_dict(token_embedding_state)
        self.token_embedding.to(self.device).eval()
        self.embedding_scale = math.sqrt(float(self.token_embedding.embedding_dim))
        self.cache_shape = tuple(int(value) for value in gemma_meta["kv_cache_shape"])
        self.num_layers = int(gemma_meta["num_hidden_layers"])
        self.sessions = {name: _create_session(path, self.device) for name, path in paths.items()}

    @torch.inference_mode()
    def predict(
        self,
        *,
        task: str,
        image: str | Path,
        image2: str | Path,
        state: list[float] | tuple[float, ...],
        seed: int,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        images = [_load_image(path, self.image_resolution, self.device) for path in (image, image2)]
        tokens, token_masks, normalized_state = self._tokenize(task, state)
        vision_outputs = [self.sessions["vision"](images[index].half()) for index in self.selected_image_indices]
        language_embs = (self.token_embedding(tokens) * self.embedding_scale).half()
        prefix_embs = torch.cat([*vision_outputs, language_embs], dim=1)
        image_masks = [torch.ones(1, dtype=torch.bool, device=self.device) for _ in images]
        prefix_masks = torch.cat(
            [
                image_masks[index][:, None].expand(1, int(output.shape[1]))
                for index, output in zip(self.selected_image_indices, vision_outputs, strict=True)
            ]
            + [token_masks],
            dim=1,
        )
        physical_prefix_length = int(prefix_embs.shape[1])
        expected_prefix_length = int(self.contract["prefix_sequence_length"])
        if physical_prefix_length != expected_prefix_length:
            raise ValueError(
                f"compact prefix length is {physical_prefix_length}, expected {expected_prefix_length}"
            )
        valid_prefix_length = int(prefix_masks.sum().item())
        if valid_prefix_length + self.horizon > self.cache_length:
            raise ValueError(
                "PI0.5 valid prefix and action horizon exceed the exported cache capacity: "
                f"{valid_prefix_length} + {self.horizon} > {self.cache_length}"
            )
        if not bool(prefix_masks[:, :valid_prefix_length].all()) or bool(prefix_masks[:, valid_prefix_length:].any()):
            raise ValueError("compact prefix contains a hole before language padding")
        prefix_attention = torch.full(
            (1, 1, 1, self.cache_length),
            MASK_VALUE,
            dtype=torch.float16,
            device=self.device,
        )
        prefix_attention[..., :valid_prefix_length] = 0
        key_caches = [
            self.CacheTensor(torch.zeros(self.cache_shape, dtype=torch.float16, device=self.device))
            for _ in range(self.num_layers)
        ]
        value_caches = [
            self.CacheTensor(torch.zeros(self.cache_shape, dtype=torch.float16, device=self.device))
            for _ in range(self.num_layers)
        ]
        self.sessions["gemma"](
            prefix_embs,
            torch.zeros(1, dtype=torch.int32, device=self.device),
            torch.tensor([valid_prefix_length], dtype=torch.int32, device=self.device),
            prefix_attention,
            *key_caches,
            *value_caches,
        )

        expert_attention = torch.full_like(prefix_attention, MASK_VALUE)
        expert_attention[..., : valid_prefix_length + self.horizon] = 0
        generator = torch.Generator(device=self.device).manual_seed(seed)
        current = torch.randn(
            (1, self.horizon, self.max_action_dim),
            generator=generator,
            dtype=torch.float32,
            device=self.device,
        )
        time_dimension = int(self._graph_input_shape("time_mlp")[-1])
        for step in range(self.num_inference_steps):
            timestep = torch.tensor(
                [1.0 - step / self.num_inference_steps],
                dtype=torch.float32,
                device=self.device,
            )
            suffix_embs = self.sessions["action_in"](current.half())
            cond = self.sessions["time_mlp"](_time_embedding(timestep, time_dimension).half())
            expert_hidden = self.sessions["expert"](
                suffix_embs,
                torch.tensor([valid_prefix_length], dtype=torch.int32, device=self.device),
                torch.tensor([self.horizon], dtype=torch.int32, device=self.device),
                cond,
                expert_attention,
                *key_caches,
                *value_caches,
            )
            velocity = self.sessions["action_out"](expert_hidden).float()
            current = current - velocity / self.num_inference_steps

        normalized_actions = current[0, :, : self.action_dim].cpu()
        actions = _apply_normalization(normalized_actions, self.action_stats, inverse=True)
        return actions.numpy(), {
            "physical_prefix_length": physical_prefix_length,
            "valid_prefix_length": valid_prefix_length,
            "valid_language_length": int(token_masks.sum().item()),
            "selected_image_indices": list(self.selected_image_indices),
            "normalized_state": normalized_state.tolist(),
            "num_inference_steps": self.num_inference_steps,
        }

    def _tokenize(
        self,
        task: str,
        state: list[float] | tuple[float, ...],
    ) -> tuple[Tensor, Tensor, Tensor]:
        if len(state) != self.state_dim:
            raise ValueError(f"PI0.5 state has {len(state)} values, expected {self.state_dim}")
        state_tensor = torch.tensor(state, dtype=torch.float32)
        normalized = _apply_normalization(state_tensor, self.state_stats, inverse=False)
        bins = np.linspace(-1, 1, 257)[:-1]
        discretized = np.digitize(normalized.numpy(), bins=bins) - 1
        cleaned = task.strip().replace("_", " ").replace("\n", " ")
        prompt = f"Task: {cleaned}, State: {' '.join(map(str, discretized))};\nAction: "
        encoded = self.tokenizer(
            [prompt],
            max_length=self.text_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            encoded["input_ids"].to(self.device),
            encoded["attention_mask"].to(self.device, dtype=torch.bool),
            normalized,
        )

    def _runtime_paths(self) -> tuple[dict[str, Path], dict[str, Any]]:
        components = self.meta["components"]
        paths: dict[str, Path] = {
            "vision": _component_file(
                self.root,
                components["vision"],
                "hmonnx_file",
                manifest=self.manifest,
            ),
        }
        for graph in components["other"]["graphs"]:
            paths[str(graph["name"]).removesuffix("_proj")] = _component_file(
                self.root,
                components["other"],
                "hmonnx_file",
                graph=graph,
                manifest=self.manifest,
            )
        gemma_relative, gemma_meta = _nested_component(
            self.root,
            components["gemma"],
            manifest=self.manifest,
        )
        expert_relative, expert_meta = _nested_component(
            self.root,
            components["expert"],
            manifest=self.manifest,
        )
        paths["gemma"] = _confined_file(
            self.root,
            (gemma_relative / str(gemma_meta["prefill_onnx_file"])).as_posix(),
            self.manifest,
        )
        paths["expert"] = _confined_file(
            self.root,
            (expert_relative / str(expert_meta["decode_onnx_file"])).as_posix(),
            self.manifest,
        )
        paths["token_embedding"] = _confined_file(
            self.root,
            (gemma_relative / str(gemma_meta["token_embedding_file"])).as_posix(),
            self.manifest,
        )
        if int(expert_meta["num_hidden_layers"]) != int(gemma_meta["num_hidden_layers"]):
            raise ValueError("PI0.5 Gemma and expert layer counts do not match")
        if tuple(expert_meta["kv_cache_shape"]) != tuple(gemma_meta["kv_cache_shape"]):
            raise ValueError("PI0.5 Gemma and expert KV cache shapes do not match")
        return paths, gemma_meta

    def _graph_input_shape(self, name: str) -> list[int]:
        for graph in self.meta["components"]["other"]["graphs"]:
            if str(graph["name"]).removesuffix("_proj") == name:
                return [int(value) for value in graph["input_shape"]]
        raise KeyError(name)


def _load_image(path: str | Path, resolution: tuple[int, int], device: torch.device) -> Tensor:
    image = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    height, width = resolution
    ratio = max(tensor.shape[-1] / width, tensor.shape[-2] / height)
    resized_height = int(tensor.shape[-2] / ratio)
    resized_width = int(tensor.shape[-1] / ratio)
    tensor = F.interpolate(tensor, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    tensor = F.pad(tensor, (pad_w0, pad_w0 + remainder_w, pad_h0, pad_h0 + remainder_h))
    return (tensor.clamp(0, 1) * 2.0 - 1.0).to(device)


def _time_embedding(timestep: Tensor, dimension: int) -> Tensor:
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=torch.float64, device=timestep.device)
    period = 0.004 * (4.0 / 0.004) ** fraction
    sin_input = (2 * math.pi / period)[None, :] * timestep.to(torch.float64)[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1).float()


def _create_session(path: Path, device: torch.device):
    if device.type != "cuda":
        from xhquant.api import HMONNXInference

        return HMONNXInference(str(path)).to(device)
    from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceConfig, HMONNXInferenceV2

    config = HMONNXInferenceConfig(enable_cuda_graph=True, exec_devices=[device])
    return HMONNXInferenceV2(str(path), config)


def _load_tokenizer(path: Path, max_length: int):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        raise ValueError("PI0.5 tokenizer must configure a pad_token_id")
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = max_length
    encoded = tokenizer(
        ["Task: validation;\nAction: "],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    if tuple(encoded["input_ids"].shape) != (1, max_length) or tuple(encoded["attention_mask"].shape) != (
        1,
        max_length,
    ):
        raise ValueError("PI0.5 tokenizer does not honor the exported fixed text length")
    return tokenizer


def _normalization_mode(policy_config: Mapping[str, Any], feature_type: str) -> str:
    mapping = policy_config.get("normalization_mapping", {})
    value = mapping.get(feature_type) if isinstance(mapping, Mapping) else None
    mode = str(value or "QUANTILES").rsplit(".", 1)[-1].lower()
    aliases = {"mean_std", "min_max", "quantiles", "quantile10"}
    if mode not in aliases:
        raise ValueError(f"Unsupported PI0.5 normalization mode for {feature_type}: {value!r}")
    return mode


def _load_stats(path: Path, prefix: str, mode: str) -> dict[str, Tensor | str]:
    from safetensors.torch import load_file

    values = load_file(str(path), device="cpu")
    keys = {
        "mean_std": ("mean", "std"),
        "min_max": ("min", "max"),
        "quantiles": ("q01", "q99"),
        "quantile10": ("q10", "q90"),
    }[mode]
    try:
        low, high = (values[f"{prefix}.{key}"].float() for key in keys)
    except KeyError as error:
        raise ValueError(f"PI0.5 {prefix} stats are missing values for {mode}: {keys}") from error
    return {"mode": mode, "low": low, "high": high}


def _validate_stats(stats: Mapping[str, Tensor | str], dimension: int, name: str) -> None:
    for key in ("low", "high"):
        value = stats[key]
        if not isinstance(value, Tensor) or value.numel() < dimension:
            size = value.numel() if isinstance(value, Tensor) else None
            raise ValueError(f"PI0.5 {name} {key} stats have {size} values, expected at least {dimension}")
        selected = value.reshape(-1)[:dimension]
        if not torch.isfinite(selected).all():
            raise ValueError(f"PI0.5 {name} {key} stats contain non-finite values")
    low = stats["low"]
    high = stats["high"]
    if isinstance(low, Tensor) and isinstance(high, Tensor):
        selected_low = low.reshape(-1)[:dimension]
        selected_high = high.reshape(-1)[:dimension]
        if str(stats["mode"]) == "mean_std":
            if bool((selected_high < 0).any()):
                raise ValueError(f"PI0.5 {name} standard deviations must be non-negative")
        elif bool((selected_high < selected_low).any()):
            raise ValueError(f"PI0.5 {name} upper stats must not be below lower stats")


def _apply_normalization(tensor: Tensor, stats: Mapping[str, Tensor | str], *, inverse: bool) -> Tensor:
    dimension = int(tensor.shape[-1])
    low = stats["low"]
    high = stats["high"]
    if not isinstance(low, Tensor) or not isinstance(high, Tensor):
        raise TypeError("PI0.5 normalization stats must be tensors")
    low = low.reshape(-1)[:dimension].to(tensor)
    high = high.reshape(-1)[:dimension].to(tensor)
    mode = str(stats["mode"])
    if mode == "mean_std":
        return tensor * high + low if inverse else (tensor - low) / (high + 1e-8)
    denominator = high - low
    denominator = torch.where(denominator == 0, torch.full_like(denominator, 1e-8), denominator)
    if inverse:
        return (tensor + 1.0) * denominator / 2.0 + low
    return 2.0 * (tensor - low) / denominator - 1.0


def _component_file(
    root: Path,
    component: Mapping[str, Any],
    key: str,
    *,
    graph=None,
    manifest: Mapping[str, Any],
) -> Path:
    relative = Path(str(component["component_dir"])) / str((graph or component)[key])
    return _confined_file(root, relative.as_posix(), manifest)


def _nested_component(
    root: Path,
    component: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    relative_root = Path(str(component["component_dir"]))
    meta_path = _confined_file(root, (relative_root / str(component["meta_file"])).as_posix(), manifest)
    return relative_root, _load_json(meta_path)


def _confined_file(root: Path, value: str, manifest: Mapping[str, Any]) -> Path:
    relative = Path(value)
    if relative.is_absolute() or PureWindowsPath(value).is_absolute() or ".." in relative.parts or "\\" in value:
        raise ValueError(f"PI0.5 runtime path escapes the Artifact: {value}")
    candidate = root / relative
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"PI0.5 runtime paths must not use symlinks: {value}")
    path = candidate.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise FileNotFoundError(f"PI0.5 runtime file is missing or outside the Artifact: {value}")
    _require_manifest_inventory(relative, manifest)
    return path


def _confined_dir(root: Path, value: str, manifest: Mapping[str, Any]) -> Path:
    relative = Path(value)
    if relative.is_absolute() or PureWindowsPath(value).is_absolute() or ".." in relative.parts or "\\" in value:
        raise ValueError(f"PI0.5 runtime path escapes the Artifact: {value}")
    candidate = root / relative
    cursor = root
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise ValueError(f"PI0.5 runtime paths must not use symlinks: {value}")
    path = candidate.resolve()
    if not path.is_relative_to(root) or not path.is_dir():
        raise FileNotFoundError(f"PI0.5 runtime directory is missing or outside the Artifact: {value}")
    if manifest.get("schema_version", 1) == 2:
        inventoried = _manifest_inventory(manifest)
        for child in path.rglob("*"):
            if child.is_symlink():
                raise ValueError(f"PI0.5 runtime paths must not use symlinks: {child}")
            if child.is_file() and child.relative_to(root).as_posix() not in inventoried:
                raise ValueError(f"PI0.5 runtime file is absent from the Manifest inventory: {child}")
    return path


def _require_manifest_inventory(relative: Path, manifest: Mapping[str, Any]) -> None:
    if manifest.get("schema_version", 1) == 2 and relative.as_posix() not in _manifest_inventory(manifest):
        raise ValueError(f"PI0.5 runtime file is absent from the Manifest inventory: {relative}")


def _manifest_inventory(manifest: Mapping[str, Any]) -> set[str]:
    files = manifest.get("artifact", {}).get("files", ())
    return {
        Path(str(entry.get("path"))).as_posix()
        for entry in files
        if isinstance(entry, Mapping) and isinstance(entry.get("path"), str)
    }


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON file must contain an object: {path}")
    return value


__all__ = ["PI05CompactRuntime"]
