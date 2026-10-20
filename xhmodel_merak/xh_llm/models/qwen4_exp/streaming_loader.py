"""Materialize Qwen4 checkpoint modules without loading the complete model."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from ...big_hf_model_helper import BigHFModelExportHelper, WeightMapping


class _ModuleWeightLoader(BigHFModelExportHelper):
    """Reuse the existing strict module loader without its eager main graph."""

    def __init__(self, model_dir: Path) -> None:
        self._hf_model_dir = str(model_dir)
        self._weight_mapping = WeightMapping()
        self._weight_mapping.weight_dir = str(model_dir)
        self._weight_mapping.weight_map = self._read_weight_map()
        self._weight_mapping.tensor_id_to_param_names = {}
        if not self._weight_mapping.weight_map:
            raise ValueError(f"No safetensors checkpoint weights in {model_dir}")


class Qwen4ExpStreamingLoader:
    """Own a meta native model and load independent checkpoint subtrees.

    ``load_module`` returns a separate module by default. Native parameters in
    ``model`` remain meta, so wrapping, PTQ and release cannot corrupt later
    phases. ``clone=False`` explicitly materializes the requested subtree in
    the skeleton, for example the embedding or final HC mixer.

    Packed GPTQ checkpoints use the existing GPTQModel layout preprocessor.
    No complete model load or dense dequantization is performed here.
    """

    def __init__(self, model_dir: str | Path, *, dtype: torch.dtype = torch.float16) -> None:
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText

        try:
            from transformers.modeling_utils import no_init_weights
        except ImportError:
            no_init_weights = init_empty_weights

        self.model_dir = Path(model_dir).expanduser().resolve(strict=True)
        self.dtype = dtype
        self._loader = _ModuleWeightLoader(self.model_dir)
        self.weight_map = self._loader._weight_mapping.weight_map
        config = AutoConfig.from_pretrained(str(self.model_dir), trust_remote_code=False)
        if config.model_type == "qwen4_exp":
            auto_model = AutoModelForImageTextToText
        elif config.model_type == "qwen4_exp_text":
            auto_model = AutoModelForCausalLM
        else:
            raise ValueError(f"Qwen4 streaming requires a native Qwen4 config, got {config.model_type!r}")
        # Keep constructor-initialized non-persistent RoPE buffers. They are
        # small and absent from the checkpoint; only parameters must be meta.
        with no_init_weights(), init_empty_weights():
            self.model = auto_model.from_config(config, trust_remote_code=False, dtype=dtype).eval()
        quant_config = getattr(config, "quantization_config", None)
        if quant_config:
            # The shared GPTQModel preprocessor looks up its model definition
            # with remote-code trust. Native Qwen4 needs no custom code; reject
            # such redirection before delegating to that existing implementation.
            raw_config = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
            if raw_config.get("auto_map"):
                raise ValueError("Qwen4 streaming does not execute checkpoint auto_map code")
            BigHFModelExportHelper._preprocess_quantized_hf_model(self.model, str(self.model_dir))
        self.config = self.model.config
        self.quantization_config = getattr(self.config, "quantization_config", None)
        materialized = [name for name, value in self.model.named_parameters() if not value.is_meta]
        if materialized:
            raise RuntimeError(f"Qwen4 streaming skeleton has materialized parameters: {materialized[:10]}")

    @contextmanager
    def _excluded_meta_tensors(self, module: nn.Module, prefix: str, skip_tensors: tuple[str, ...]):
        if isinstance(skip_tensors, str) or len(set(skip_tensors)) != len(skip_tensors):
            raise ValueError("skip_tensors must contain unique complete checkpoint tensor names")
        excluded = []
        try:
            for name in skip_tensors:
                if not name.startswith(prefix + "."):
                    raise ValueError(f"Excluded tensor is not checkpoint-backed within {prefix!r}: {name!r}")
                try:
                    info = self.tensor_info(name)
                except KeyError as exc:
                    raise ValueError(f"Excluded tensor is not checkpoint-backed within {prefix!r}: {name!r}") from exc
                local = name[len(prefix) + 1:]
                parent_name, _, child = local.rpartition(".")
                owner = module.get_submodule(parent_name) if parent_name else module
                collection = owner._parameters if child in owner._parameters else owner._buffers
                value = collection.get(child)
                if not torch.is_tensor(value) or not value.is_meta:
                    raise ValueError(f"Only an explicit meta checkpoint tensor can be excluded: {name!r}")
                if tuple(value.shape) != info["shape"]:
                    raise ValueError(f"Shape mismatch for excluded checkpoint tensor {name!r}")
                excluded.append((collection, child, collection.pop(child)))
            yield
        finally:
            for collection, child, value in excluded:
                collection[child] = value

    def _converted_parameter(self, name: str, expected: nn.Parameter) -> torch.Tensor | None:
        """Consume the native HF saved expert/PLE layout one module at a time."""
        if name.endswith(".ngram_embedding.weight"):
            return self.read_tensor(name)
        if not name.endswith((".experts.gate_up_proj", ".experts.down_proj")):
            return None
        prefix, projection = name.rsplit(".", 1)
        experts = []
        for index in range(expected.shape[0]):
            suffixes = ("gate_proj.weight", "up_proj.weight") if projection == "gate_up_proj" else ("down_proj.weight",)
            keys = [f"{prefix}.{index}.{suffix}" for suffix in suffixes]
            if not all(key in self.weight_map for key in keys):
                raise RuntimeError(f"Native expert parameters are absent from the safetensors weight map: {keys}")
            values = [self.read_tensor(key) for key in keys]
            experts.append(torch.cat(values, dim=0) if len(values) > 1 else values[0])
        return torch.stack(experts, dim=0)

    @contextmanager
    def _native_converted_parameters(self, module: nn.Module, prefix: str):
        converted = []
        try:
            for local, expected in list(module.named_parameters()):
                name = prefix + "." + local
                if name in self.weight_map:
                    continue
                value = self._converted_parameter(name, expected)
                if value is None:
                    continue
                if tuple(value.shape) != tuple(expected.shape):
                    raise RuntimeError(
                        f"Shape mismatch for converted checkpoint tensor {name}: "
                        f"{tuple(value.shape)} != {tuple(expected.shape)}"
                    )
                parent_name, _, child = local.rpartition(".")
                owner = module.get_submodule(parent_name) if parent_name else module
                owner._parameters.pop(child)
                converted.append((owner, child, nn.Parameter(value.to(dtype=expected.dtype),
                                                           requires_grad=expected.requires_grad)))
            yield len(converted)
        finally:
            for owner, child, value in converted:
                owner._parameters[child] = value

    def load_into(self, module: nn.Module, prefix: str, *, skip_tensors: tuple[str, ...] = ()) -> nn.Module:
        """Load a matching meta module, preserving packed integer tensors.

        This also accepts an independently constructed meta export module with
        the same state layout. Missing persistent state and shape mismatches
        retain the common loader's strict validation.
        """
        if not prefix or prefix.startswith(".") or prefix.endswith(".") or ".." in prefix:
            raise ValueError("A nonempty complete checkpoint module prefix is required")
        try:
            with self._excluded_meta_tensors(module, prefix, skip_tensors), self._native_converted_parameters(
                module, prefix,
            ) as converted:
                audit = self._loader._load_module_from_safetensor(module, prefix)
            if self.quantization_config is not None:
                BigHFModelExportHelper._prepare_loaded_gptqmodel_modules(
                    module, self.quantization_config, model_config=self.config,
                )
            module._streaming_load_audit = dict(audit, checkpoint_prefix=prefix, skipped_tensor_keys=skip_tensors,
                                               converted_parameter_keys=converted)
            return module.eval()
        except BaseException:
            self.release(module)
            raise

    def load_module(
        self, prefix: str, *, clone: bool = True, skip_tensors: tuple[str, ...] = (),
    ) -> nn.Module:
        module = self.model.get_submodule(prefix)
        return self.load_into(copy.deepcopy(module) if clone else module, prefix, skip_tensors=skip_tensors)

    @staticmethod
    def release(module: nn.Module) -> None:
        """Return weights to meta; subsequent loads restore all required state."""
        BigHFModelExportHelper._release_place_holder_module(module)

    @contextmanager
    def materialized(
        self, prefix: str, *, clone: bool = True, skip_tensors: tuple[str, ...] = (),
    ) -> Iterator[nn.Module]:
        module = self.load_module(prefix, clone=clone, skip_tensors=skip_tensors)
        try:
            yield module
        finally:
            self.release(module)

    def _tensor_parts(self, name: str) -> tuple[str, ...]:
        if name in self.weight_map:
            return (name,)
        if name.endswith(".ngram_embedding.weight"):
            prefix = name.removesuffix(".weight")
            pattern = re.compile(re.escape(prefix) + r"\.shard_(\d+)\.weight$")
            matches = sorted((int(match[1]), key) for key in self.weight_map if (match := pattern.fullmatch(key)))
            config = getattr(self.config, "text_config", self.config)
            count = int(getattr(config, "split_ngram_parts", 1))
            if count > 0 and [index for index, _ in matches] == list(range(count)):
                return tuple(key for _, key in matches)
        raise KeyError(f"Checkpoint tensor is missing: {name}")

    def tensor_info(self, name: str) -> dict:
        """Read only the safetensors header; dtype uses its F16/BF16/I32 names."""
        pieces = []
        for part in self._tensor_parts(name):
            with safe_open(self.weight_map[part], framework="pt", device="cpu") as reader:
                view = reader.get_slice(part)
                pieces.append({"shape": tuple(view.get_shape()), "dtype": view.get_dtype()})
        first = pieces[0]
        if len(pieces) == 1:
            return first
        if any(not item["shape"] or item["shape"][1:] != first["shape"][1:]
               or item["dtype"] != first["dtype"] for item in pieces):
            raise ValueError(f"Incompatible native PLE checkpoint shards: {name}")
        return {"shape": (sum(item["shape"][0] for item in pieces), *first["shape"][1:]), "dtype": first["dtype"]}

    def read_tensor(self, name: str, *, rows: slice | None = None) -> torch.Tensor:
        """Read one tensor or a bounded contiguous row range without dtype casts."""
        parts = self._tensor_parts(name)
        if rows is None and len(parts) == 1:
            with safe_open(self.weight_map[parts[0]], framework="pt", device="cpu") as reader:
                return reader.get_tensor(parts[0])
        shape = self.tensor_info(name)["shape"]
        rows = slice(None) if rows is None else rows
        if not shape or rows.step not in (None, 1):
            raise ValueError("Checkpoint row reads require a nonscalar tensor and unit step")
        start = 0 if rows.start is None else rows.start
        stop = shape[0] if rows.stop is None else rows.stop
        if type(start) is not int or type(stop) is not int or not 0 <= start <= stop <= shape[0]:
            raise ValueError(f"Checkpoint row range is outside [0, {shape[0]}]")
        blocks, offset = [], 0
        for part in parts:
            with safe_open(self.weight_map[part], framework="pt", device="cpu") as reader:
                view = reader.get_slice(part)
                count = view.get_shape()[0]
                if start < offset + count and stop > offset:
                    blocks.append(view[max(0, start - offset):min(count, stop - offset)])
                elif start == stop and not blocks:
                    blocks.append(view[0:0])
                offset += count
        return blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=0)

    def iter_tensor_rows(self, name: str, *, chunk_rows: int) -> Iterator[torch.Tensor]:
        """Stream a large PLE table in bounded row chunks."""
        if type(chunk_rows) is not int or chunk_rows < 1:
            raise ValueError("chunk_rows must be a positive integer")
        self.tensor_info(name)  # Validate all shard headers before reading data.
        for part in self._tensor_parts(name):
            with safe_open(self.weight_map[part], framework="pt", device="cpu") as reader:
                view = reader.get_slice(part)
                shape = view.get_shape()
                if not shape:
                    raise ValueError("Checkpoint row iteration requires a nonscalar tensor")
                for start in range(0, shape[0], chunk_rows):
                    yield view[start:min(start + chunk_rows, shape[0])]


__all__ = ["Qwen4ExpStreamingLoader"]
