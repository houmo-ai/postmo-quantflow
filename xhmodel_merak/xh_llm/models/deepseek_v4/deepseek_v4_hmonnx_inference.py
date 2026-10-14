"""HMONNX runtime for the fixed-shape DeepSeek-V4 Flash graphs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from ...hmonnx import TextLLMHMONNXModel
from ...types import LLMModelMeta
from .cache_abi import DeepSeekV4CacheABI, default_layer_types
from .runtime import DeepSeekV4CacheMixin, DeepSeekV4DataPreprocess
from .static_cache import DeepSeekV4StaticCacheSpec


def _cache_input_devices_from_layer_infos(
    abi: DeepSeekV4CacheABI,
    layer_infos: Sequence[object],
) -> tuple[tuple[torch.device, ...], ...]:
    """Resolve persistent cache ownership from stable exported layer tags."""

    devices_by_layer: dict[int, torch.device] = {}
    for layer_info in layer_infos:
        device = torch.device(layer_info.device)
        for tag_name in getattr(layer_info, "llm_tags", ()):
            prefix = "layer_"
            if not str(tag_name).startswith(prefix):
                continue
            suffix = str(tag_name)[len(prefix) :]
            if not suffix.isdigit():
                continue
            layer = int(suffix)
            previous = devices_by_layer.setdefault(layer, device)
            if previous != device:
                raise RuntimeError(
                    f"DeepSeek-V4 layer {layer} was assigned to both {previous} and {device}"
                )

    missing = sorted(set(range(len(abi.layer_types))) - devices_by_layer.keys())
    if missing:
        raise RuntimeError(f"missing auto-offload placement for DeepSeek-V4 layers: {missing}")

    return tuple(
        (devices_by_layer[layer],) * len(input_names)
        for layer, input_names in enumerate(abi.graph_cache_input_names_by_layer())
    )


class XHDeepSeekV4HMONNXModel(TextLLMHMONNXModel):
    """Own the heterogeneous SWA/CSA/HCA caches and commit graph states."""

    def __init__(
        self,
        meta_info: LLMModelMeta,
        *,
        enable_auto_offload: bool = False,
        **kwargs,
    ) -> None:
        if getattr(meta_info, "deepseek_v4_state_abi_version", None) != 2:
            raise ValueError("DeepSeek-V4 state ABI expected 2; re-export with fused SparseAttentionSelectKV")
        if kwargs.get("enable_page_attention", False):
            raise ValueError("DeepSeek-V4 uses its dedicated SWA/CSA/HCA cache ABI, not PageAttention")
        super().__init__(
            meta_info,
            enable_auto_offload=enable_auto_offload,
            **kwargs,
        )

        if self.pad_token_id is None:
            raise ValueError("DeepSeek-V4 HMONNX metadata must contain pad_token_id")

        model_config = meta_info.model_config
        configured_layers = getattr(model_config, "max_layers", None)
        configured_layers = 43 if configured_layers is None else int(configured_layers)
        layer_types = tuple(getattr(meta_info, "layer_types", default_layer_types(configured_layers)))
        geometry = {}
        if getattr(meta_info, "dspark", None) or getattr(model_config, "graph_geometry", None):
            from transformers import AutoConfig

            config = AutoConfig.from_pretrained(self.hf_model_dir)
            geometry = {"sliding_window": int(config.sliding_window),
                        "latent_head_dim": int(config.head_dim),
                        "index_head_dim": int(config.index_head_dim),
                        "index_topk": int(config.index_topk)}
        spec = DeepSeekV4StaticCacheSpec(
            max_context_length=int(model_config.context_max_length),
            prefill_chunk_length=int(model_config.prefill_chunk_length),
            decode_sequence_length=int(getattr(model_config, "num_logits_to_keep", 1)),
            **geometry,
        )
        self.cache_abi = DeepSeekV4CacheABI(
            spec=spec,
            layer_types=layer_types,
            batch_size=int(model_config.batch_size),
        )
        input_devices = None
        if enable_auto_offload:
            input_devices = _cache_input_devices_from_layer_infos(
                self.cache_abi,
                self.prefill_model.hmonnx_session.get_layer_infos(),
            )
        self._kvcache_mixin = DeepSeekV4CacheMixin(
            self.cache_abi,
            self.kvcache_config,
            input_devices=input_devices,
        )
        self._kvcache_mixin.to(device=self.device, dtype=self.dtype)
        self._data_processor = None
        expected_inplace_updates = 2 * len(self.cache_abi.csa_layers) + len(self.cache_abi.hca_layers)
        for graph_model in (self.prefill_model, self.decode_model):
            session = graph_model.hmonnx_session
            enable_inplace_updates = getattr(session, "enable_inplace_cache_updates", None)
            if callable(enable_inplace_updates):
                enabled = int(enable_inplace_updates())
                if enabled != expected_inplace_updates:
                    raise RuntimeError(
                        "DeepSeek-V4 non-sliding cache writer count mismatch: "
                        f"expected={expected_inplace_updates}, enabled={enabled}"
                    )
            if graph_model.enable_cuda_graph and not enable_auto_offload:
                enable_borrowed_outputs = getattr(session, "enable_borrowed_cuda_graph_outputs", None)
                if callable(enable_borrowed_outputs):
                    state_outputs = [
                        name for name in session.get_output_names() if str(name).endswith("_state_output")
                    ]
                    if state_outputs:
                        enable_borrowed_outputs(state_outputs)

    def _get_data_preprocessor(self) -> DeepSeekV4DataPreprocess:
        return DeepSeekV4DataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.get_input_sequence_length(),
            cache_mixin=self._kvcache_mixin,
            pad_token_id=self.pad_token_id,
        )

    def get_input_sequence_length(self) -> int:
        config = self.meta_info.model_config
        return int(config.prefill_chunk_length if self._llm_prefill else config.num_logits_to_keep)

    def _load_hmonnx_graphs(self, prefill_hmonnx: str, decode_hmonnx: str):
        dspark = getattr(self.meta_info, "dspark", None)
        if not dspark:
            return super()._load_hmonnx_graphs(prefill_hmonnx, decode_hmonnx)
        from xhquant.xhonnxruntime.llm_hmonnx_loader import MultiHMONNXLoader

        root = Path(self.meta_info.prefill_hmonnx).parent.parent.resolve()
        paths = {"prefill": prefill_hmonnx, "decode": decode_hmonnx}
        expected = {"dspark_context", "dspark_context_decode", "dspark_draft_decode", "dspark_markov"}
        if set(dspark["components"]) != expected:
            raise ValueError("DSpark runtime package must contain all four draft components")
        for name, entry in dspark["components"].items():
            relative = Path(entry["path"])
            path = (root / relative).resolve()
            if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
                raise ValueError("DSpark component path must stay inside the runtime package")
            paths[name] = str(path)
        loader = MultiHMONNXLoader(paths)
        self._llm_graph_loader = loader
        # Prime and drain the requested CUDA contexts before free-memory-based
        # placement. The shared graph weights are still CPU resident here.
        for device in self._valid_devices:
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        return loader.graphs["prefill"], loader.graphs["decode"], loader.initializer_pool.summary()

    def build_dspark_engine(self, *, device: str | None = None):
        from xhquant.xhonnxruntime.hmonnx_inference_v2 import HMONNXInferenceV2, HMONNXInferenceConfig
        from .dspark import DSparkSpec
        from .dspark_engine import DSparkEngine

        metadata = getattr(self.meta_info, "dspark", None)
        if not metadata:
            raise ValueError("DSpark generation requires the complete target/draft runtime package")
        spec = DSparkSpec(**metadata["spec"])
        resolved = torch.device(device or str(self.device))
        loader = self._llm_graph_loader
        graphs = {name: HMONNXInferenceV2.from_onnx_graph(
                      loader.graph_files[name], loader.graphs[name],
                      session_config=HMONNXInferenceConfig(exec_devices=[resolved],
                                                          enable_cuda_graph=self.enable_cuda_graph))
                  for name in metadata["components"]}

        def forward(ids, past, prefill):
            self.set_prefill() if prefill else self.set_decode()
            args = self.get_data_preprocessor()({"input_ids": ids, "past_seq_length": past})
            return self.forward_verify(*args)

        def embedding(ids):
            return self.embed_tokens(ids.to(self.device).long()).to(resolved)

        return DSparkEngine(spec, target_forward=forward, target_cache=self._kvcache_mixin,
                            embedding=embedding, context=graphs["dspark_context"],
                            context_decode=graphs["dspark_context_decode"],
                            draft=graphs["dspark_draft_decode"], markov=graphs["dspark_markov"], device=resolved)

    def forward(self, *args):
        outputs = super().forward(*args)
        has_auxiliary = bool(getattr(self.meta_info.model_config, "dspark", None))
        logits = self._kvcache_mixin.apply_flat_state_outputs(outputs, auxiliary_output=has_auxiliary)
        if has_auxiliary and logits.shape[1] > 1:
            valid = getattr(self._data_processor, "valid_input_length", None)
            last = int(args[4].item()) if valid is None else int(valid) - 1
            return logits[:, last:last + 1]
        return logits

    def forward_verify(self, *args):
        """Run the multi-token decode graph and retain every candidate logit."""
        if not getattr(self.meta_info.model_config, "dspark", None):
            raise ValueError("DSpark verification requires a DSpark target export")
        outputs = super().forward(*args)
        logits = self._kvcache_mixin.apply_flat_state_outputs(outputs, auxiliary_output=True)
        return logits, self._kvcache_mixin.last_target_hidden


__all__ = ["XHDeepSeekV4HMONNXModel"]
