# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: SIM117
import json
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Optional, cast

import torch
from torch import nn
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.v1.sample.sampler import Sampler
from xhmodel_merak.xh_llm import AutoLLMHONNXModel

from ...config import MerakConfig
from ...hmfp_kv_cache import is_hmfp_cache_dtype
from ...utils import (
    apply_hmonnx_runtime_options,
    get_hmonnx_runtime_kwargs,
    move_hmonnx_auxiliary_tensors,
    resolve_hmonnx_primary_device,
    resolve_torch_dtype,
    should_enable_page_attention,
    should_preserve_hmonnx_runtime_device_map,
)
from ..register import MERAK_MODEL_REGISTRY
from .hf_assets import get_model_trust_remote_code, load_hf_config
from .hybrid_cache_policy import parse_full_attention_cache_shapes
from .merak_for_causal_lm import MerakForCausalLM

logger = init_logger(__name__)


@MERAK_MODEL_REGISTRY.register_module("Qwen3ForCausalLM")
class Qwen3MerakForCausalLM(MerakForCausalLM):
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        kv_cache_dtype: str = "auto",
        *,
        vllm_config: Optional[VllmConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        if vllm_config is not None:
            model_config = vllm_config.model_config
        if model_config is None:
            raise ValueError("MerakForCausalLM requires either model_config or vllm_config.")

        self.vllm_config = vllm_config
        self.prefix = prefix
        config = model_config.hf_config
        if not isinstance(config, MerakConfig):
            raise TypeError(
                "MerakForCausalLM expects a MerakConfig as hf_config, got "
                f"{type(config).__name__}. Did you forget to set "
                "config_format='merak_llm' on the LLM?"
            )
        self.config: MerakConfig = config
        xh_model = self.config.xh_model
        self.enable_page_attention = should_enable_page_attention(self.config)

        config_dir_candidates: list[Path] = []
        model_path = getattr(model_config, "model", None)
        if model_path:
            config_dir_candidates.append(Path(model_path))
        name_or_path = getattr(config, "_name_or_path", None)
        if name_or_path:
            config_dir_candidates.append(Path(name_or_path))

        config_dir = next(
            (candidate for candidate in config_dir_candidates if candidate.is_dir()),
            Path(name_or_path or model_path or "."),
        )
        hf_config_dir = config_dir / "hf_config"
        hf_config_file = hf_config_dir / "config.json"
        hf_config = json.load(hf_config_file.open("r", encoding="utf-8"))
        architecture = hf_config["architectures"][0]
        if architecture not in [
            "Qwen3ForCausalLM",
            "Qwen2ForCausalLM",
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        ]:
            raise ValueError(f"Unsupported architecture {architecture} for Merak model at {config_dir}.")
        logger.info(f"Loading Merak model with architecture {architecture} from {config_dir}.")
        self.device = torch.device("cpu")
        self.inference_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        xh_model_type = xh_model.get("model_type", None)
        if xh_model_type is None:
            raise ValueError(f"Merak model config at {config_dir} is missing 'model_type' field.")

        self.key_cache_config: list[list[int]] = []
        self.value_cache_config: list[list[int]] = []
        cache_dtype = "float16"

        self.num_layers = 0
        self.kv_cache_shape: tuple[int, ...] | None = None
        self.cache_axis = 2
        requested_cache_dtype = kv_cache_dtype.strip().lower() if isinstance(kv_cache_dtype, str) else "auto"
        if xh_model_type.lower() == "hmonnx":
            hmonnx_meta_file = str(config_dir / "golden_meta_info.json")
            hmonnx_model = AutoLLMHONNXModel.from_pretrained(
                hmonnx_meta_file,
                **get_hmonnx_runtime_kwargs(self.config),
            )
            apply_hmonnx_runtime_options(hmonnx_model, self.config)
            preserve_runtime_device_map = should_preserve_hmonnx_runtime_device_map(self.config)
            if not preserve_runtime_device_map:
                hmonnx_model.to(self.inference_device)
            self.inference_device = resolve_hmonnx_primary_device(hmonnx_model, self.inference_device)
            self.inference_model = hmonnx_model
            move_hmonnx_auxiliary_tensors(hmonnx_model, self.inference_device)
            with open(hmonnx_meta_file, "r", encoding="utf-8") as f:
                hmonnx_meta_info = json.load(f)
                kv_cache_meta = hmonnx_meta_info["kv_cache"]
                cache_shapes = parse_full_attention_cache_shapes(kv_cache_meta)
                num_layers = int(kv_cache_meta["num_layers"])
                self.num_layers = num_layers
                self.key_cache_shape = cache_shapes.key
                self.value_cache_shape = cache_shapes.value
                self.kv_cache_shape = cache_shapes.key
                self.cache_axis = cache_shapes.cache_axis
                for _ in range(num_layers):
                    self.key_cache_config.append(list(cache_shapes.key))
                    self.value_cache_config.append(list(cache_shapes.value))
                cache_dtype = kv_cache_meta.get("cache_dtype", "float16")
                if is_hmfp_cache_dtype(requested_cache_dtype):
                    cache_dtype = requested_cache_dtype
        self.cache_dtype = cache_dtype
        self.cache_torch_dtype = resolve_torch_dtype(cache_dtype)
        if self.cache_axis != 2:
            raise NotImplementedError(
                f"Merak HMONNX cache_axis={self.cache_axis} is not supported; expected sequence axis at 2."
            )
        if self.kv_cache_shape is None:
            raise RuntimeError(f"Failed to determine KV cache shape for Merak model at {config_dir}.")

        self.max_cache_tokens = int(self.kv_cache_shape[self.cache_axis])

        hf_model_config = load_hf_config(
            hf_config_dir,
            trust_remote_code=get_model_trust_remote_code(model_config),
        )
        if hasattr(hf_model_config, "text_config"):
            hf_model_config = hf_model_config.text_config
        self._register_placeholder_attention_layers(hf_model_config)
        self.logits_processor = LogitsProcessor(hf_model_config.vocab_size, logits_as_input=True)
        self.input_embeddings = cast(nn.Embedding, self.inference_model.get_input_embeddings())

        self.sampler = Sampler()


def get_model(
    vllm_config: VllmConfig,
    kv_cache_dtype: str,
    **kwargs,
) -> torch.nn.Module:
    del kwargs
    with cast(AbstractContextManager[None], set_current_vllm_config(vllm_config)):
        return Qwen3MerakForCausalLM(vllm_config=vllm_config, kv_cache_dtype=kv_cache_dtype)
