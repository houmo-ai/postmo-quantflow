import copy
from functools import wraps
from typing import Any, Union

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, GenerationConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeCausalLMOutputWithPast,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeForConditionalGeneration,
)

from xhmodel_merak.xh_llm.base_model import get_model_param_buffer_size_gb
from xhmodel_merak.xh_llm.models.qwen3_5_moe.qwen3_5_moe_vision_model import XHQwen3_5MoeVisionModel
from xhquant.api import get_xhquant_logger
from xhquant.nn.modules.moeblock import MoeBlock
from xhquant.utils.registry import _DMRegistryCls

from ...builder import register_llm_model
from ...text_llm_hf_compatible import TextLLMHFCompatible
from ...utils import get_cpu_memory_mb
from ...vision_llm_model import VisionLLMModel
from ..qwen3_5.qwen3_5_llm_model import (
    Qwen3_5_ModelMeta,
    XHQwen3_5Model,
    _enforce_split_conv_cache_wrap_cfg,
)
from ..qwen3_5.qwen3_5_llm_model import (
    _Qwen3_5HFCompatible as _Qwen3_5DenseHFCompatible,
)
from .autoround_checkpoint import checkpoint_overlay_with_untied_lm_head, import_autoround_gptq_v2_weights
from .qwen3_5_moe_hmonnx_inference import XHQwen3_5MoeHMONNXModel
from .xh_qwen3_5_moe_config import XHQwen3_5MoeModelConfig


_AUTOROUND_SOURCE_GROUP_SIZE = 64
_AUTOROUND_TARGET_WEIGHT_NSHARE = 64


try:
    from transformers.modeling_utils import no_init_weights
except ImportError:
    no_init_weights = init_empty_weights


class _Qwen3_5HFCompatible(TextLLMHFCompatible):  # noqa: N801
    def _setup(
        self: Union[Qwen3_5MoeForCausalLM, Qwen3_5MoeForConditionalGeneration],
        text_llm_model: "XHQwen3_5MoeModel",
    ):
        model = super()._setup(text_llm_model)
        if model is not None:
            model_container = getattr(model, "model", None)
            if model_container is not None and hasattr(model_container, "visual"):
                del model_container.visual
            if model_container is not None and hasattr(model_container, "language_model"):
                del model_container.language_model
            elif hasattr(model, "model"):
                del model.model
            if hasattr(model, "lm_head"):
                del model.lm_head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return model

    def set_experts_implementation(self, experts_implementation):  # noqa: D401
        """No-op override for Transformers decode optimization hooks."""
        self.config.experts_implementation = experts_implementation

    def get_correct_experts_implementation(self, experts_implementation):
        return experts_implementation

    def _grouped_mm_can_dispatch(self):
        return False

    @wraps(_Qwen3_5DenseHFCompatible.forward)
    def forward(self, *args, **kwargs) -> Qwen3_5MoeCausalLMOutputWithPast:
        # Dense and MoE share patch-token routing, preprocessing and cache I/O.
        output = _Qwen3_5DenseHFCompatible.forward(self, *args, **kwargs)
        return Qwen3_5MoeCausalLMOutputWithPast(**output)

    @wraps(forward)
    def _sample_forward(self, *args, **kwargs):
        # GenerationMixin validates kwargs after generate() replaces forward.
        # Preserve the multimodal signature while the sampling hook is active.
        return super()._sample_forward(*args, **kwargs)


def build_qwen3_5_moe_hf_compatible_model(
    hf_model: Qwen3_5MoeForConditionalGeneration,
    xh_model: "XHQwen3_5MoeModel",
):
    llm_compatible_modules = _DMRegistryCls("XHCompatible")
    hf_model_cls = type(hf_model)
    if hf_model_cls not in llm_compatible_modules:
        llm_compatible_modules.register_module({hf_model_cls: hf_model_cls.__name__}, _Qwen3_5HFCompatible)
    return llm_compatible_modules.convert(hf_model, text_llm_model=xh_model)


class Qwen3_5Moe_ModelMeta(Qwen3_5_ModelMeta):  # noqa: N801
    pass


@register_llm_model("Qwen3_5MoeForConditionalGeneration")
class XHQwen3_5MoeModel(XHQwen3_5Model):  # noqa: N801
    HF_MODEL_CLS = Qwen3_5MoeForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    META_CLS = Qwen3_5Moe_ModelMeta
    HMONNXINFERENCE_CLS = XHQwen3_5MoeHMONNXModel
    CONFIG_CLS = XHQwen3_5MoeModelConfig
    BUILD_HF_COMPATIBLE_FUNC = staticmethod(build_qwen3_5_moe_hf_compatible_model)
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.qwen3_5.workflow:Qwen35Workflow"

    def __init__(self, config: XHQwen3_5MoeModelConfig):
        super().__init__(config)

        if hasattr(config, "visual_config") and config.visual_config is not None and config.visual_config.enable:
            self.visual = XHQwen3_5MoeVisionModel(config.visual_config)
            self.visual.config.model_name = f"{self.config.model_name}_visual"
        else:
            self.visual = None
        # self.full_attention_layer_indices: list[int] = []
        # self.linear_attention_layer_indices: list[int] = []
        # self._kvcache_config = KVCacheWithLinearConfig()
        # self._kvcache_mixin = KVCacheWithLinearMixin(self.kvcache_config)
        # self.config = cast(XHQwen3_5MoeModelConfig, self.config)
        # self.wrap_cfg["linear_attention_mode"] = "auto"

        # self.wrap_cfg["linear_attention_mode"] = "chunk"  # for prefill
        # self.wrap_cfg["linear_attention_mode"] = "recurrent"  # for decode

    def _get_data_preprocessor(self):
        from .data_preprocess import Qwen3_5_DataPreprocess

        return Qwen3_5_DataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.wrap_cfg.input_sequence_length,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            past_conv_caches=self.past_conv_caches,
            past_recurrent_states=self.past_recurrent_states,
            patch_size=self.config.visual_config.patch_size,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
            vision_end_token_id=self.config.vision_end_token_id,
            spatial_merge_size=self.config.spatial_merge_size,
        )

    def _wraped_pre(self, hf_model):
        super()._wraped_pre(hf_model)
        from .safe_calibration import prepare_automatic_safe_inputs

        prepare_automatic_safe_inputs(
            hf_model, self.config.dynamic_prune, self.hf_model_dir, self.work_dir,
            device=self.config.dynamic_prune.get("s_scalar_compute_device", "auto"),
        )

    def _wraped_post(self, hf_model: Qwen3_5MoeForConditionalGeneration):
        super()._wraped_post(hf_model)

        # The MoE wrapper follows the dense Qwen3.5 cache contract: external
        # export/runtime signatures are flat q/k/v conv-cache tensors, while
        # trace-time linear attention consumes per-layer (q, k, v) tuples.
        # Re-apply the split flag after MoE wrapping so child DynamicModules do
        # not silently trace the merged qkv path when the cache mixin/export
        # side is already split.
        language_model = self._get_language_model(self._wrap_model)
        _enforce_split_conv_cache_wrap_cfg(language_model, self.wrap_cfg)

        split_conv_cache = bool(self.wrap_cfg.get("split_conv_cache", False))
        self._kvcache_mixin.split_conv_cache = split_conv_cache
        if split_conv_cache and self.linear_attention_layer_indices:
            linear_attn = language_model.layers[self.linear_attention_layer_indices[0]].linear_attn
            self._kvcache_mixin._linear_key_dim = linear_attn.key_dim
            self._kvcache_mixin._linear_value_dim = linear_attn.value_dim

        if self.config.dynamic_prune["mode"] == "decode":
            from .dynamic_prune import prepare_scalars

            self._moe_prune_scalars, scalar_path = prepare_scalars(
                language_model, self.config.dynamic_prune, self.hf_model_dir, self.work_dir
            )
            self.config.dynamic_prune["s_scalar_path"] = scalar_path

    def _to_fronted(self, wrap_model):
        frontend_model = super()._to_fronted(wrap_model)
        if self.config.dynamic_prune["mode"] == "decode":
            from .dynamic_prune import configure_decode_graph

            configure_decode_graph(
                frontend_model, self._moe_prune_scalars, self.config.dynamic_prune["threshold"]
            )
        return frontend_model

    def _get_big_language_placeholder_export_components(self):
        if self.config.dynamic_prune["mode"] == "decode":
            raise ValueError("Decode ACEP requires regular export (HUGE_MODEL_EXPORT_ENABLED=0)")
        # Import the real MoE wrappers before entering
        # ``traceable_module_placeholder_context``.  The context temporarily
        # replaces these registrations with lightweight placeholder wrappers;
        # importing ``_moe_model`` for the first time from inside that context
        # would make its decorators register the same HF classes twice.
        from ._moe_model import register_wrap_modules
        from ._qwen3_5_moe_big_export import Qwen3_5_MOE_BigHFModel

        register_wrap_modules()
        return Qwen3_5_MOE_BigHFModel, Qwen3_5_MOE_BigHFModel.PLACEHOLDER_TYPES

    def _check_big_language_placeholder_export_supported(self, empty_hf_model: Any) -> None:
        hf_model_type = str(getattr(empty_hf_model.config, "model_type", "")).lower()
        model_type_name = type(empty_hf_model).__name__.lower()
        if "moe" not in hf_model_type and "moe" not in model_type_name:
            raise NotImplementedError(
                "Qwen3.5 dense big-model placeholder export must use the dense placeholder components."
            )

        language_model = self._get_language_model(empty_hf_model)
        if not hasattr(language_model, "modules"):
            raise NotImplementedError("Qwen3.5 MoE big-model placeholder export requires an nn.Module language model.")

        found_types = {type(module).__name__ for module in language_model.modules()}

        _, placeholder_types = self._get_big_language_placeholder_export_components()
        required_hf_types = {
            "Qwen3_5MoeAttention",
            "Qwen3_5MoeGatedDeltaNet",
            "Qwen3_5MoeSparseMoeBlock",
        }
        missing_types = sorted(required_hf_types - found_types)
        if missing_types:
            raise NotImplementedError(
                "Qwen3.5 MoE big-model placeholder export requires placeholder modules "
                f"{placeholder_types}, but missing {missing_types}."
            )

    def init_wrap_model(self, hf_model: Qwen3_5MoeForConditionalGeneration) -> Any:
        from ._moe_model import register_wrap_modules
        
        register_wrap_modules()
        wrap_model = super(VisionLLMModel, self).init_wrap_model(hf_model)
        return wrap_model


@register_llm_model("Qwen3_5MoeForCausalLM")
class XHQwen3_5MoeCausalLMModel(XHQwen3_5MoeModel):  # noqa: N801
    """Text-only Qwen3.5-MoE model used by AutoRound/GPTQ checkpoints."""

    HF_MODEL_CLS = Qwen3_5MoeForCausalLM
    HF_AUTO_MODEL_CLS = AutoModelForCausalLM
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.qwen3_5_moe.workflow:Qwen35MoeTextWorkflow"

    def __init__(self, config: XHQwen3_5MoeModelConfig):
        super().__init__(config)
        self.visual = None

    def _get_data_preprocessor(self):
        from .data_preprocess import Qwen3_5_DataPreprocess

        return Qwen3_5_DataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.wrap_cfg.input_sequence_length,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            past_conv_caches=self.past_conv_caches,
            past_recurrent_states=self.past_recurrent_states,
            patch_size=1,
            image_token_id=-1,
            video_token_id=-1,
            vision_start_token_id=-1,
            vision_end_token_id=-1,
            spatial_merge_size=1,
        )

    @classmethod
    def get_empty_hf_model(cls, hf_model_dir, **kwargs) -> Any:
        """Build a causal-LM skeleton from a composite checkpoint's text config."""
        with no_init_weights(), init_empty_weights():
            if "dtype" not in kwargs:
                kwargs["dtype"] = cls.HF_MODEL_DTYPE
            kwargs.setdefault("trust_remote_code", True)
            config = AutoConfig.from_pretrained(
                hf_model_dir, trust_remote_code=kwargs["trust_remote_code"]
            ).get_text_config()
            hf_model = cls.HF_AUTO_MODEL_CLS.from_config(config, **kwargs)
            if hf_model.can_generate():
                try:
                    hf_model.generation_config = GenerationConfig.from_pretrained(hf_model_dir)
                except OSError:
                    get_xhquant_logger().info(
                        "Generation config file not found, using a generation config created from the model config."
                    )
        return hf_model

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs) -> Any:
        kwargs.setdefault("trust_remote_code", True)
        config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=kwargs["trust_remote_code"])
        quant_config = getattr(config, "quantization_config", None)
        if not quant_config:
            return super().get_hf_model(hf_model_dir, quant_weight, **kwargs)
        if quant_config.get("checkpoint_format") != "gptq_v2":
            raise ValueError("Qwen3.5 MoE text checkpoint requires AutoRound GPTQ-v2 format")
        if quant_weight:
            raise ValueError("Packed text checkpoint cannot also receive quant_weight")
        # Packed codes are consumed by this model's importer, rather than an
        # HF quantizer. Load the native text skeleton and non-packed weights.
        from ..qwen3_5.qwen3_5_llm_model import qwen3_5_patch

        text_config = copy.deepcopy(config.get_text_config())
        if hasattr(text_config, "quantization_config"):
            del text_config.quantization_config
        kwargs["config"] = text_config
        hf_model = qwen3_5_patch(cls._load_hf_model(hf_model_dir, **kwargs))
        stats = import_autoround_gptq_v2_weights(hf_model, hf_model_dir)
        get_xhquant_logger().info("Imported AutoRound GPTQ-v2 packed weights: %s", stats)
        return hf_model

    def get_native_model(self):
        config = AutoConfig.from_pretrained(self.hf_model_dir, trust_remote_code=True)
        self._autoround_packed_checkpoint = bool(getattr(config, "quantization_config", None))
        if not self._autoround_packed_checkpoint:
            return super().get_native_model()
        lm_head_source = getattr(self.config, "lm_head_source", None)
        kwargs = {"device_map": "auto" if self.config.enable_auto_offload else "cpu"}
        with checkpoint_overlay_with_untied_lm_head(self.hf_model_dir, lm_head_source) as load_dir:
            hf_model = type(self).get_hf_model(str(load_dir), quant_weight=None, **kwargs)
        if not isinstance(hf_model, self.get_hf_model_cls()):
            raise TypeError(
                f"The model is not {self.get_hf_model_cls().__name__}, but {type(hf_model).__name__}"
            )
        logger = get_xhquant_logger()
        if (
            lm_head_source is not None
            and getattr(hf_model.lm_head, "_xhquant_weight_origin", None) != "autoround-gptq-v2"
        ):
            hf_model.lm_head._xhquant_weight_origin = "explicit-base-lm-head"
            logger.info("Loaded untied lm_head.weight during from_pretrained from explicit source: %s", lm_head_source)
        return hf_model

    def get_quant_cfg(self):
        return super().get_quant_cfg()

    @staticmethod
    def _is_lm_head_node(node) -> bool:
        return str(node.target).split(".")[-1] == "lm_head" or node.name == "lm_head"

    def _apply_autoround_linear_w_schemas(self, frontend_model) -> None:
        """Attach strict mixed-precision weight schemas to exact FX nodes.

        ``to_fronted()`` releases ``_wrap_model`` before quantization, and
        xhquant resolves per-node overrides by FX ``node.name``.  Source-model
        path overrides therefore cannot reliably protect float exceptions.
        Node metadata is collected at the start of ``to_quant_graph`` and is
        the authoritative place to encode the checkpoint's mixed precision.
        """

        schemas = {
            "w4": dict(
                bits=4,
                fp_mode="ssfp",
                hidden_bit=False,
                nshare=_AUTOROUND_TARGET_WEIGHT_NSHARE,
            ),
            "w8": dict(
                bits=8,
                fp_mode="ssfp",
                hidden_bit=False,
                nshare=_AUTOROUND_TARGET_WEIGHT_NSHARE,
            ),
        }
        counts = {"w4": 0, "w8": 0, "w8_split": 0, "global": 0, "lm_head": 0, "moe_w4": 0}
        diagnostics = []

        for node in frontend_model.graph.nodes:
            if node.op != "call_module":
                continue
            module = frontend_model.get_submodule(str(node.target))
            if isinstance(module, MoeBlock):
                external_weights = [
                    getattr(module, f"expert_{projection}_quant_weight", None)
                    for projection in ("gate_proj", "up_proj", "down_proj")
                ]
                if any(weight is not None for weight in external_weights):
                    if not all(weight is not None for weight in external_weights):
                        raise RuntimeError(f"Frontend MoeBlock {node.name!r} has incomplete AutoRound expert codes")
                    bits = getattr(module, "_autoround_bits", None)
                    group_size = getattr(module, "_autoround_group_size", None)
                    origin = getattr(module, "_xhquant_weight_origin", None)
                    if (
                        origin != "autoround-gptq-v2"
                        or bits != 4
                        or group_size != _AUTOROUND_SOURCE_GROUP_SIZE
                    ):
                        raise RuntimeError(
                            f"Frontend MoeBlock {node.name!r} expected W4G64 AutoRound codes, "
                            f"got origin={origin!r}, bits={bits!r}, group_size={group_size!r}"
                        )
                    node_quant_cfg = copy.deepcopy(node.meta.get("quant_config", {}))
                    node_quant_cfg["w_schema"] = copy.deepcopy(schemas["w4"])
                    node.meta["quant_config"] = node_quant_cfg
                    counts["moe_w4"] += 1
                    diagnostics.append(
                        (
                            node.name,
                            str(node.target),
                            type(module).__name__,
                            True,
                            bits,
                            origin,
                            schemas["w4"],
                            "w4",
                        )
                    )
                continue
            if not isinstance(module, torch.nn.Linear):
                continue

            quant_weight = getattr(module, "quant_weight", None)
            bits = getattr(module, "_autoround_bits", None)
            group_size = getattr(module, "_autoround_group_size", None)
            origin = getattr(module, "_xhquant_weight_origin", None)
            if quant_weight is not None:
                if origin != "autoround-gptq-v2":
                    raise RuntimeError(
                        f"External quant_weight on frontend node {node.name!r} has unsupported origin {origin!r}"
                    )
                if bits not in (4, 8):
                    raise RuntimeError(
                        f"AutoRound frontend node {node.name!r} has unsupported or missing bit width {bits!r}"
                    )
                if group_size != _AUTOROUND_SOURCE_GROUP_SIZE:
                    raise RuntimeError(
                        f"AutoRound frontend node {node.name!r} expected source group_size="
                        f"{_AUTOROUND_SOURCE_GROUP_SIZE}, got {group_size!r}"
                    )
                # Preserve the producer width even when zero/small codes do
                # not exercise its range. SDK QLinear consumes logical_bits.
                module.logical_bits = bits
                policy = f"w{bits}"
                count_policy = "w8_split" if bits == 8 and hasattr(module, "_autoround_split_from") else policy
            elif self._is_lm_head_node(node):
                policy = "lm_head"
            else:
                if origin == "autoround-gptq-v2" or bits is not None:
                    raise RuntimeError(
                        f"AutoRound metadata reached frontend node {node.name!r} without quant_weight"
                    )
                policy = "global"
                count_policy = policy

            if policy == "lm_head":
                count_policy = policy

            if policy in schemas:
                node_quant_cfg = copy.deepcopy(node.meta.get("quant_config", {}))
                node_quant_cfg["w_schema"] = copy.deepcopy(schemas[policy])
                node.meta["quant_config"] = node_quant_cfg

            counts[count_policy] += 1
            diagnostics.append(
                (
                    node.name,
                    str(node.target),
                    type(module).__name__,
                    quant_weight is not None,
                    bits,
                    origin,
                    schemas.get(policy),
                    policy,
                )
            )

        if counts["w4"] + counts["w8"] + counts["w8_split"] + counts["moe_w4"] == 0:
            raise RuntimeError("No AutoRound packed Linear or MoeBlock nodes were found in the frontend graph")
        if counts["w8_split"] % 3 != 0:
            raise RuntimeError(
                f"Split in_proj_qkv AutoRound nodes must occur in q/k/v triples, got {counts['w8_split']}"
            )
        source_w8_modules = counts["w8"] + counts["w8_split"] // 3

        logger = get_xhquant_logger()
        logger.info(
            "Applied strict AutoRound frontend schemas: w4_linear=%d, w8_linear=%d, global_linear=%d, "
            "lm_head=%d, w4_moe=%d, w8_split_children=%d, source_w8_modules=%d, "
            "source_group_size=%d, target_weight_nshare=%d",
            counts["w4"],
            counts["w8"],
            counts["global"],
            counts["lm_head"],
            counts["moe_w4"],
            counts["w8_split"],
            source_w8_modules,
            _AUTOROUND_SOURCE_GROUP_SIZE,
            _AUTOROUND_TARGET_WEIGHT_NSHARE,
        )
        for node_name, target, module_cls, has_quant_weight, bits, origin, w_schema, policy in diagnostics:
            logger.info(
                "AutoRound frontend schema node=%s target=%s module=%s quant_weight=%s bits=%s origin=%s "
                "effective_w_schema=%s",
                node_name,
                target,
                module_cls,
                has_quant_weight,
                bits,
                origin,
                w_schema if w_schema is not None else f"{policy}-config",
            )

    def _to_quanted(self, frontend_model, state):
        if getattr(self, "_autoround_packed_checkpoint", False):
            self._apply_autoround_linear_w_schemas(frontend_model.prefill)
            self._apply_autoround_linear_w_schemas(frontend_model.decode)
        return super()._to_quanted(frontend_model, state)

    def _get_language_model(self, hf_model: Any) -> Any:
        return hf_model.model

    def _wraped_pre(self, hf_model: Qwen3_5MoeForCausalLM):
        language_model = self._get_language_model(hf_model)
        self.layer_types = list(language_model.config.layer_types)

        if self.config.only_first_block:
            self.linear_attention_layer_indices = []
            self.full_attention_layer_indices = []
            for idx, layer_type in enumerate(self.layer_types):
                if layer_type == "full_attention":
                    self.full_attention_layer_indices.append(idx)
                    break
                self.linear_attention_layer_indices.append(idx)
            self.config.max_layers = len(self.full_attention_layer_indices) + len(
                self.linear_attention_layer_indices
            )
            self.config.only_first_block = False
        else:
            self.full_attention_layer_indices = [
                idx for idx, layer_type in enumerate(self.layer_types) if layer_type == "full_attention"
            ]
            self.linear_attention_layer_indices = [
                idx for idx, layer_type in enumerate(self.layer_types) if layer_type == "linear_attention"
            ]

        logger = get_xhquant_logger()
        language_count, language_bytes = get_model_param_buffer_size_gb(hf_model)
        logger.info(f"_wraped_pre model parameters: {language_count} B ({language_bytes:.2f} GB)")
        logger.info(f"CPU memory usage _wraped_pre: {get_cpu_memory_mb()}")
        return hf_model

    def _wraped_post(self, hf_model: Qwen3_5MoeForCausalLM):
        del hf_model
        language_model = self._get_language_model(self._wrap_model)
        _enforce_split_conv_cache_wrap_cfg(language_model, self.wrap_cfg)
        self.embed_tokens = copy.deepcopy(language_model.get_input_embeddings())
        text_config = language_model.config
        self.pad_token_id = text_config.eos_token_id
        self.layer_types = list(text_config.layer_types)

        self_attn = language_model.layers[self.full_attention_layer_indices[0]].self_attn
        linear_attn = language_model.layers[self.linear_attention_layer_indices[0]].linear_attn
        linear_kv_cache_config = self.kvcache_config.linear_kv_cache_config
        linear_kv_cache_config.conv_dim = linear_attn.conv_dim
        linear_kv_cache_config.conv_kernel_size = linear_attn.conv_kernel_size
        linear_kv_cache_config.num_v_heads = linear_attn.num_v_heads
        linear_kv_cache_config.head_k_dim = linear_attn.head_k_dim
        linear_kv_cache_config.head_v_dim = linear_attn.head_v_dim
        linear_kv_cache_config.num_layers = len(self.linear_attention_layer_indices)

        split_conv_cache = bool(self.wrap_cfg.get("split_conv_cache", False))
        self._kvcache_mixin.split_conv_cache = split_conv_cache
        if split_conv_cache:
            self._kvcache_mixin._linear_key_dim = linear_attn.key_dim
            self._kvcache_mixin._linear_value_dim = linear_attn.value_dim

        if self.use_cache:
            self.kvcache_config.num_layers = len(self.full_attention_layer_indices)
            self.kvcache_config.kv_cache_shape = [
                1,
                text_config.num_key_value_heads,
                self.config.context_max_length,
                self_attn.head_dim,
            ]

        logger = get_xhquant_logger()
        language_count, language_bytes = get_model_param_buffer_size_gb(self._wrap_model)
        logger.info(f"_wraped_post model parameters: {language_count} B ({language_bytes:.2f} GB)")
        logger.info(f"CPU memory usage _wraped_post: {get_cpu_memory_mb()}")
