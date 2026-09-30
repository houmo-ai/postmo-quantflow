from dataclasses import dataclass
from typing import Tuple

import torch
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.cpu_attn import CPUAttentionBackendImpl
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

from ...config import MERAK_PREFERRED_BLOCK_SIZE
from ...hmfp_kv_cache import get_hmfp_cache_shape, is_hmfp_cache_dtype


@dataclass
class MerakAttentionMetadata:
    """Metadata for MerakAttentionBackend.

    Basic terms used below:
    - batch_size_in_sequences - total number of sequences to execute​
    - prompt_lens – per sequence size number of scheduled tokens​
    - batch_size_in_tokens = sum(prompt_lens)​
    - max_context_len = max(context_lens)​
    - max_num_blocks = div_up(max_context_len / BLOCK_SIZE)​
    - num_blocks – total number of blocks in block_indices​
    """

    # Describes past KV cache size for each sequence within a batch
    # Shape: [batch_size_in_sequences]
    # Type: i32​
    past_lens: torch.Tensor

    # Describes start indices of input / speculative tokens from
    # current sequences within a batch sequence​
    # Shape: [batch_size_in_sequences + 1]​
    # Type: i32
    subsequence_begins: torch.Tensor

    # Describes block tables for each sequence within a batch​ -
    # indices along 0th dimension in key_cache and value_cache inputs​
    # Shape: [num_blocks]
    # Type: i32​
    block_indices: torch.Tensor

    # Describes block tables for each sequence within a batch​ -
    # for i-th element, it is an index in block_indices with the
    # first block belonging to i-th sequence​
    # Shape: [batch_size_in_sequences + 1]
    # Type: i32​
    block_indices_begins: torch.Tensor

    # Describes max context length
    # Shape: scalar
    # Type: i32
    max_context_len: torch.Tensor

    # Maps each current query token to its physical KV cache slot.
    # Shape: [batch_size_in_tokens]
    slot_mapping: torch.Tensor

    # Indices of sampled tokens, used in the vLLM V1 API.
    sampled_token_indices: torch.Tensor | None = None

    # Total sequence lengths (past + current) for this exact cache group.
    # Kept optional so contract-v1/Qwen metadata constructed by older callers
    # remains source compatible.
    seq_lens: torch.Tensor | None = None

    # The index maps that relate multi-modal embeddings to the corresponding
    # placeholders.
    #
    # N.B. These aren't really related to attention and don't belong on this
    # type -- this is just a temporary solution to make them available to
    # `model_executable`.
    # multi_modal_placeholder_index_maps: Optional[Dict[
    #     str, MultiModalPlaceholderMap.IndexMap]]

    # # Enable/disable KV scales calculation. This is so that we can disable the
    # # calculation until after prefill and cuda graph capture.
    # enable_kv_scales_calculation: bool

    # # Indices of sampled tokens, used in V1 vLLM API only.
    # sampled_token_indices: ov.Tensor


class MerakAttentionMetadataBuilder(AttentionMetadataBuilder["MerakAttentionMetadata"]):
    _cudagraph_support = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.block_size = kv_cache_spec.block_size
        self._init_reorder_batch_threshold(None, False)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> "MerakAttentionMetadata":
        del common_prefix_len, fast_build

        num_reqs = common_attn_metadata.num_reqs
        seq_lens = common_attn_metadata.seq_lens.to(torch.int32)
        past_lens = common_attn_metadata.compute_num_computed_tokens().to(torch.int32)
        subsequence_begins = common_attn_metadata.query_start_loc.to(torch.int32)
        block_table = common_attn_metadata.block_table_tensor.to(torch.int32)
        slot_mapping = common_attn_metadata.slot_mapping

        block_counts = torch.div(
            seq_lens + self.block_size - 1,
            self.block_size,
            rounding_mode="floor",
        ).to(torch.int32)

        if num_reqs == 0:
            block_indices = torch.empty((0,), dtype=torch.int32, device=self.device)
            block_indices_begins = torch.zeros(1, dtype=torch.int32, device=self.device)
            max_context_len = torch.zeros((), dtype=torch.int32, device=self.device)
        else:
            max_blocks_per_req = block_table.shape[1]
            col_idx = torch.arange(
                max_blocks_per_req, device=self.device, dtype=torch.int32
            )
            mask = col_idx.unsqueeze(0) < block_counts.unsqueeze(1)
            block_indices = block_table[mask]
            zero_prefix = torch.zeros(1, dtype=torch.int32, device=self.device)
            block_indices_begins = torch.cat(
                [zero_prefix, block_counts.cumsum(0).to(torch.int32)]
            )
            max_context_len = seq_lens.max().to(torch.int32)

        sampled_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        return MerakAttentionMetadata(
            past_lens=past_lens,
            subsequence_begins=subsequence_begins,
            block_indices=block_indices,
            block_indices_begins=block_indices_begins,
            max_context_len=max_context_len,
            slot_mapping=slot_mapping,
            sampled_token_indices=sampled_token_indices,
            seq_lens=seq_lens,
        )


@dataclass
class MerakPageAttentionGroupContext:
    """One exact vLLM cache group's PageAttention metadata.

    The tensors are group-owned rather than batch-global: heterogeneous Gemma4
    full/sliding groups may have different block tables and slot mappings.
    """

    block_ids: torch.Tensor
    slot_mapping: torch.Tensor
    seq_lens: torch.Tensor
    kv_window_start_abs: torch.Tensor | None = None
    mm_prefix_ranges: torch.Tensor | None = None


@register_backend(AttentionBackendEnum.CUSTOM)
class MerakAttentionBackend(AttentionBackend):
    @staticmethod
    def indexes_kv_by_block_stride() -> bool:
        """Declare the physical page-stride ABI used by HMONNX cache ops."""

        return True

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MERAK_PREFERRED_BLOCK_SIZE]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        del cls, default_block_size
        return MERAK_PREFERRED_BLOCK_SIZE

    @staticmethod
    def get_impl_cls():
        return CPUAttentionBackendImpl

    @staticmethod
    def get_builder_cls():
        return MerakAttentionMetadataBuilder

    @staticmethod
    def make_merak_metadata(*args, **kwargs) -> "MerakAttentionMetadata":
        return MerakAttentionMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> Tuple[int, ...]:
        if is_hmfp_cache_dtype(cache_dtype_str):
            return get_hmfp_cache_shape(
                num_blocks=num_blocks,
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                cache_dtype=cache_dtype_str,
            )
        return (2, num_blocks, num_kv_heads, block_size, head_size)
