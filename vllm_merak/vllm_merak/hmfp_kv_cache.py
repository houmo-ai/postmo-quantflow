from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator

import torch
from typing_extensions import Self
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from .config import HMFP_NSHARE

SEFP_CACHE_DTYPE = "sefp"

_SUPPORTED_QDATA_DTYPES = {"int8", "i8", "int16", "i16"}
_SUPPORTED_SCALE_DTYPES = {"int8", "i8"}
_DTYPE_BYTES = {
    "int8": 1,
    "i8": 1,
    "int16": 2,
    "i16": 2,
}

_TORCH_DTYPES = {
    "int8": torch.int8,
    "i8": torch.int8,
    "int16": torch.int16,
    "i16": torch.int16,
}


def _normalize_dtype(dtype_name: str) -> str:
    normalized = dtype_name.lower().strip()
    if normalized in {"i8", "int8", "8"}:
        return "int8"
    if normalized in {"i16", "int16", "16"}:
        return "int16"
    return normalized


def _normalize_torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        if dtype in (torch.int8, torch.int16):
            return dtype
        raise ValueError(f"unsupported qdata dtype: {dtype}")

    normalized = _normalize_dtype(dtype)
    torch_dtype = _TORCH_DTYPES.get(normalized)
    if torch_dtype is None:
        raise ValueError(f"unsupported dtype: {dtype}")
    return torch_dtype


def _torch_dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.int8:
        return "int8"
    if dtype is torch.int16:
        return "int16"
    raise ValueError(f"unsupported dtype: {dtype}")


def _qmax_for(dtype: str | torch.dtype) -> int:
    torch_dtype = _normalize_torch_dtype(dtype)
    if torch_dtype is torch.int8:
        return 127
    if torch_dtype is torch.int16:
        return 32767
    raise ValueError(f"unsupported qdata dtype: {dtype}")


def get_hmfp_vllm_backing_cache_dtype(cache_dtype: str | None) -> str:
    if is_hmfp_cache_dtype(cache_dtype):
        return "fp8"
    return "float16" if cache_dtype in (None, "auto") else cache_dtype


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((value + alignment - 1) // alignment) * alignment


def is_hmfp_cache_dtype(cache_dtype: object) -> bool:
    if not isinstance(cache_dtype, str):
        return False
    return cache_dtype.lower().strip().startswith(SEFP_CACHE_DTYPE)


@dataclass(frozen=True)
class HMFPKVCacheConfig:
    nshare: int = HMFP_NSHARE
    k_qdata_dtype: str = "int8"
    v_qdata_dtype: str = "int8"
    scale_exp_dtype: str = "int8"
    slot_alignment: int = 16

    def __post_init__(self) -> None:
        object.__setattr__(self, "k_qdata_dtype", _normalize_dtype(self.k_qdata_dtype))
        object.__setattr__(self, "v_qdata_dtype", _normalize_dtype(self.v_qdata_dtype))
        object.__setattr__(
            self, "scale_exp_dtype", _normalize_dtype(self.scale_exp_dtype)
        )

        if self.nshare != HMFP_NSHARE:
            raise ValueError(f"nshare must be {HMFP_NSHARE}, got {self.nshare}")
        if self.k_qdata_dtype not in _SUPPORTED_QDATA_DTYPES:
            raise ValueError(f"unsupported K qdata dtype: {self.k_qdata_dtype}")
        if self.v_qdata_dtype not in _SUPPORTED_QDATA_DTYPES:
            raise ValueError(f"unsupported V qdata dtype: {self.v_qdata_dtype}")
        if self.scale_exp_dtype not in _SUPPORTED_SCALE_DTYPES:
            raise ValueError(
                f"unsupported scale exponent dtype: {self.scale_exp_dtype}"
            )
        if self.slot_alignment <= 0:
            raise ValueError(
                f"slot_alignment must be positive, got {self.slot_alignment}"
            )

    @property
    def k_qdata_bytes(self) -> int:
        return _DTYPE_BYTES[self.k_qdata_dtype]

    @property
    def v_qdata_bytes(self) -> int:
        return _DTYPE_BYTES[self.v_qdata_dtype]

    @property
    def scale_exp_bytes(self) -> int:
        return _DTYPE_BYTES[self.scale_exp_dtype]

    @property
    def k_torch_dtype(self) -> torch.dtype:
        return _normalize_torch_dtype(self.k_qdata_dtype)

    @property
    def v_torch_dtype(self) -> torch.dtype:
        return _normalize_torch_dtype(self.v_qdata_dtype)

    @property
    def scale_exp_torch_dtype(self) -> torch.dtype:
        return torch.int8

    def compatibility_fields(
        self,
        *,
        num_kv_heads: int,
        head_size: int,
        block_size: int,
    ) -> dict[str, str | int]:
        return {
            "kv_cache_format": SEFP_CACHE_DTYPE,
            "kv_cache_nshare": self.nshare,
            "kv_cache_k_qdata_dtype": self.k_qdata_dtype,
            "kv_cache_v_qdata_dtype": self.v_qdata_dtype,
            "kv_cache_scale_exp_dtype": self.scale_exp_dtype,
            "num_kv_heads": num_kv_heads,
            "head_dim": head_size,
            "block_size": block_size,
            "tensor_layout_order": "k=block,head,dim_group,lane,token;v=block,head,token_group,lane,dim",
        }

    @classmethod
    def from_cache_dtype(cls, cache_dtype: str | None) -> "HMFPKVCacheConfig":
        """Parse a compact cache dtype string.

        Supported spellings intentionally stay simple for the first plugin-only
        path:

        - ``sefp``: int8 K/V qdata, int8 scale exponent, nshare 64.
        - ``sefp_kint16_vint8``: int16 K qdata, int8 V qdata.
        - ``sefp_kint8_vint16``: int8 K qdata, int16 V qdata.
        - ``sefp_kint16_vint16``: int16 K/V qdata.
        """
        if cache_dtype is None:
            return cls()

        normalized = cache_dtype.lower().strip()
        if normalized in {"", SEFP_CACHE_DTYPE}:
            return cls()
        if not is_hmfp_cache_dtype(normalized):
            raise ValueError(f"cache dtype is not SEFP: {cache_dtype}")

        k_qdata_dtype = "int8"
        v_qdata_dtype = "int8"
        for part in normalized.split("_"):
            if part == SEFP_CACHE_DTYPE:
                continue
            if part.startswith("k"):
                k_qdata_dtype = _normalize_dtype(part[1:])
            elif part.startswith("v"):
                v_qdata_dtype = _normalize_dtype(part[1:])
            else:
                raise ValueError(f"unsupported SEFP cache dtype modifier: {part}")

        return cls(k_qdata_dtype=k_qdata_dtype, v_qdata_dtype=v_qdata_dtype)


@dataclass(frozen=True)
class HMFPKVCacheLayout:
    num_kv_heads: int
    head_size: int
    block_size: int
    config: HMFPKVCacheConfig = HMFPKVCacheConfig()
    head_size_v: int | None = None

    # KCache: [num_kv_heads, head_size, block_size]
    # VCache: [num_kv_heads, block_size, head_size_v]

    def __post_init__(self) -> None:
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)
        if self.num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {self.num_kv_heads}")
        if self.head_size <= 0:
            raise ValueError(f"head_size must be positive, got {self.head_size}")
        if self.head_size_v <= 0:
            raise ValueError(f"head_size_v must be positive, got {self.head_size_v}")
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.block_size % self.config.nshare != 0:
            raise ValueError(
                f"block_size={self.block_size} must be divisible by nshare={self.config.nshare}"
            )
        if self.head_size % self.config.nshare != 0:
            raise ValueError(
                f"head_size={self.head_size} must be divisible by nshare={self.config.nshare}"
            )

    @property
    def num_groups(self) -> int:
        return self.head_size // self.config.nshare

    @property
    def num_v_groups(self) -> int:
        return self.block_size // self.config.nshare

    @property
    def k_qdata_bytes(self) -> int:
        return (
            self.num_kv_heads
            * self.num_groups
            * self.config.nshare
            * self.block_size
            * self.config.k_qdata_bytes
        )

    @property
    def k_scale_exp_bytes(self) -> int:
        return (
            self.num_kv_heads
            * self.num_groups
            * self.block_size
            * self.config.scale_exp_bytes
        )

    @property
    def v_qdata_bytes(self) -> int:
        return (
            self.num_kv_heads
            * self.num_v_groups
            * self.config.nshare
            * self.head_size_v
            * self.config.v_qdata_bytes
        )

    @property
    def v_scale_exp_bytes(self) -> int:
        return (
            self.num_kv_heads
            * self.num_v_groups
            * self.head_size_v
            * self.config.scale_exp_bytes
        )

    @property
    def bytes_per_block(self) -> int:
        block_end = self.v_scale_exp_offset + self.v_scale_exp_bytes
        return align_up(
            block_end,
            max(
                self.config.slot_alignment,
                self.config.k_qdata_bytes,
                self.config.v_qdata_bytes,
                self.config.scale_exp_bytes,
            ),
        )

    @property
    def k_qdata_offset(self) -> int:
        return 0

    @property
    def k_scale_exp_offset(self) -> int:
        return align_up(
            self.k_qdata_offset + self.k_qdata_bytes, self.config.scale_exp_bytes
        )

    @property
    def v_qdata_offset(self) -> int:
        return align_up(
            self.k_scale_exp_offset + self.k_scale_exp_bytes, self.config.v_qdata_bytes
        )

    @property
    def v_scale_exp_offset(self) -> int:
        return align_up(
            self.v_qdata_offset + self.v_qdata_bytes, self.config.scale_exp_bytes
        )

    def tensor_offsets(self) -> dict[str, int]:
        return {
            "k_qdata": self.k_qdata_offset,
            "k_scale_exp": self.k_scale_exp_offset,
            "v_qdata": self.v_qdata_offset,
            "v_scale_exp": self.v_scale_exp_offset,
        }

    @property
    def k_qdata_shape(self) -> tuple[int, int, int, int]:
        return (self.num_kv_heads, self.num_groups, self.config.nshare, self.block_size)

    @property
    def k_scale_exp_shape(self) -> tuple[int, int, int, int]:
        return (self.num_kv_heads, self.num_groups, 1, self.block_size)

    @property
    def v_qdata_shape(self) -> tuple[int, int, int, int]:
        return (
            self.num_kv_heads,
            self.num_v_groups,
            self.config.nshare,
            self.head_size_v,
        )

    @property
    def v_scale_exp_shape(self) -> tuple[int, int, int, int]:
        return (self.num_kv_heads, self.num_v_groups, 1, self.head_size_v)

    def tensor_shapes(
        self, num_blocks: int
    ) -> dict[str, tuple[int, int, int, int, int]]:
        if num_blocks < 0:
            raise ValueError(f"num_blocks must be non-negative, got {num_blocks}")
        return {
            "k_qdata": (num_blocks, *self.k_qdata_shape),
            "k_scale_exp": (num_blocks, *self.k_scale_exp_shape),
            "v_qdata": (num_blocks, *self.v_qdata_shape),
            "v_scale_exp": (num_blocks, *self.v_scale_exp_shape),
        }

    def cache_shape(self, num_blocks: int) -> tuple[int, int]:
        if num_blocks < 0:
            raise ValueError(f"num_blocks must be non-negative, got {num_blocks}")
        return (num_blocks, self.bytes_per_block)

    def compatibility_fields(self) -> dict[str, str | int]:
        fields = self.config.compatibility_fields(
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            block_size=self.block_size,
        )
        fields["value_head_dim"] = self.head_size_v
        return fields


@dataclass
class HMFPTensorCache:
    qdata: torch.Tensor
    scale_exp: torch.Tensor


@dataclass
class HMFPKVCache:
    k: HMFPTensorCache
    v: HMFPTensorCache


@dataclass(frozen=True)
class HMFPContiguousKVCacheConfig:
    num_heads: int
    head_dim: int
    max_seq_len: int
    nshare: int = HMFP_NSHARE
    k_qdata_dtype: torch.dtype = torch.int8
    v_qdata_dtype: torch.dtype = torch.int8
    scale_exp_dtype: torch.dtype = torch.int8

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "k_qdata_dtype", _normalize_torch_dtype(self.k_qdata_dtype)
        )
        object.__setattr__(
            self, "v_qdata_dtype", _normalize_torch_dtype(self.v_qdata_dtype)
        )

        if self.num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {self.num_heads}")
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {self.head_dim}")
        if self.max_seq_len < 0:
            raise ValueError(
                f"max_seq_len must be non-negative, got {self.max_seq_len}"
            )
        if self.nshare != HMFP_NSHARE:
            raise ValueError(f"nshare must be {HMFP_NSHARE}, got {self.nshare}")
        if self.head_dim % self.nshare != 0:
            raise ValueError(
                f"head_dim={self.head_dim} must be divisible by nshare={self.nshare}"
            )
        if self.scale_exp_dtype is not torch.int8:
            raise ValueError(
                f"scale_exp_dtype must be torch.int8, got {self.scale_exp_dtype}"
            )

    @property
    def num_groups(self) -> int:
        return self.head_dim // self.nshare


@dataclass
class HMFPPagedKVCache:
    k_qdata: torch.Tensor
    k_scale_exp: torch.Tensor
    v_qdata: torch.Tensor
    v_scale_exp: torch.Tensor
    block_size: int
    num_blocks: int
    num_kv_heads: int
    head_size: int
    nshare: int = HMFP_NSHARE
    head_size_v: int | None = None

    def __post_init__(self) -> None:
        if self.head_size_v is None:
            self.head_size_v = self.head_size
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        if self.num_blocks < 0:
            raise ValueError(f"num_blocks must be non-negative, got {self.num_blocks}")
        if self.num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {self.num_kv_heads}")
        if self.head_size <= 0:
            raise ValueError(f"head_size must be positive, got {self.head_size}")
        if self.head_size_v <= 0:
            raise ValueError(f"head_size_v must be positive, got {self.head_size_v}")
        if self.nshare != HMFP_NSHARE:
            raise ValueError(f"nshare must be {HMFP_NSHARE}, got {self.nshare}")
        if self.block_size % self.nshare != 0:
            raise ValueError(
                f"block_size={self.block_size} must be divisible by nshare={self.nshare}"
            )
        if self.head_size % self.nshare != 0:
            raise ValueError(
                f"head_size={self.head_size} must be divisible by nshare={self.nshare}"
            )

        expected_k_qdata_shape = (
            self.num_blocks,
            self.num_kv_heads,
            self.num_groups,
            self.nshare,
            self.block_size,
        )
        expected_k_scale_shape = (
            self.num_blocks,
            self.num_kv_heads,
            self.num_groups,
            1,
            self.block_size,
        )
        expected_v_qdata_shape = (
            self.num_blocks,
            self.num_kv_heads,
            self.num_v_groups,
            self.nshare,
            self.head_size_v,
        )
        expected_v_scale_shape = (
            self.num_blocks,
            self.num_kv_heads,
            self.num_v_groups,
            1,
            self.head_size_v,
        )
        if tuple(self.k_qdata.shape) != expected_k_qdata_shape:
            raise ValueError(
                f"k_qdata shape must be {expected_k_qdata_shape}, got {tuple(self.k_qdata.shape)}"
            )
        if tuple(self.v_qdata.shape) != expected_v_qdata_shape:
            raise ValueError(
                f"v_qdata shape must be {expected_v_qdata_shape}, got {tuple(self.v_qdata.shape)}"
            )
        if tuple(self.k_scale_exp.shape) != expected_k_scale_shape:
            raise ValueError(
                f"k_scale_exp shape must be {expected_k_scale_shape}, got {tuple(self.k_scale_exp.shape)}"
            )
        if tuple(self.v_scale_exp.shape) != expected_v_scale_shape:
            raise ValueError(
                f"v_scale_exp shape must be {expected_v_scale_shape}, got {tuple(self.v_scale_exp.shape)}"
            )
        if (
            self.k_scale_exp.dtype is not torch.int8
            or self.v_scale_exp.dtype is not torch.int8
        ):
            raise ValueError("SEFP scale_exp tensors must use torch.int8")

    @property
    def num_groups(self) -> int:
        return self.head_size // self.nshare

    @property
    def num_v_groups(self) -> int:
        return self.block_size // self.nshare

    @property
    def group_size(self) -> int:
        return self.nshare

    @property
    def device(self) -> torch.device:
        return self.k_qdata.device

    @property
    def layout(self) -> HMFPKVCacheLayout:
        return HMFPKVCacheLayout(
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            block_size=self.block_size,
            config=HMFPKVCacheConfig(
                nshare=self.nshare,
                k_qdata_dtype=_torch_dtype_name(self.k_qdata.dtype),
                v_qdata_dtype=_torch_dtype_name(self.v_qdata.dtype),
            ),
            head_size_v=self.head_size_v,
        )

    def tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.k_qdata, self.k_scale_exp, self.v_qdata, self.v_scale_exp

    def named_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        yield "k_qdata", self.k_qdata
        yield "k_scale_exp", self.k_scale_exp
        yield "v_qdata", self.v_qdata
        yield "v_scale_exp", self.v_scale_exp

    def numel(self) -> int:
        return sum(tensor.numel() for tensor in self.tensors())

    def copy_block(self, src_block_id: int, dst_block_id: int) -> None:
        for _, tensor in self.named_tensors():
            tensor[dst_block_id].copy_(tensor[src_block_id])

    def swap_blocks(self, block_id_a: int, block_id_b: int) -> None:
        for _, tensor in self.named_tensors():
            tmp = tensor[block_id_a].clone()
            tensor[block_id_a].copy_(tensor[block_id_b])
            tensor[block_id_b].copy_(tmp)

    def clone_empty(self, *, num_blocks: int | None = None) -> "HMFPPagedKVCache":
        return allocate_hmfp_paged_kv_cache(
            num_blocks=self.num_blocks if num_blocks is None else num_blocks,
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            device=self.device,
            nshare=self.nshare,
            k_qdata_dtype=self.k_qdata.dtype,
            v_qdata_dtype=self.v_qdata.dtype,
        )


@dataclass(frozen=True, kw_only=True)
class HMFPFullAttentionSpec(FullAttentionSpec):
    sefp_cache_dtype: str = SEFP_CACHE_DTYPE

    # The SEFP PageAttention backend reads paged KV through a strided view
    # whose block stride equals the raw (physical) per-block byte size, i.e.
    # ``k_qdata[block_id, ...]`` / ``v_qdata[block_id, ...]`` index by the
    # physical page stride (see xhonnxruntime PageAttention._read_*_cache_sefp
    # and hmfp_paged_kv_cache_from_raw_tensor which sets stride[0] to the raw
    # block_stride). This means a page physically padded to the group's max
    # page size is read correctly -- the padding bytes simply live past the
    # logical SEFP sections and are never touched. Opt in to block-stride
    # indexing so vLLM's hybrid page-size unification (unify_kv_cache_spec_page_size)
    # can pad this spec up to the max group page size instead of raising
    # NotImplementedError when it is not an exact divisor of the linear
    # MambaSpec page size.
    indexes_kv_by_block_stride: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        # The inherited AttentionSpec field is still accepted as an __init__
        # keyword, and vLLM code paths can explicitly pass/preserve False while
        # copying generic attention specs.  For HMFP this is not optional: the
        # backend always indexes pages by the physical block stride, so force the
        # flag back on to keep hybrid page-size unification safe.
        object.__setattr__(self, "indexes_kv_by_block_stride", True)

    @property
    def real_page_size_bytes(self) -> int:
        return get_hmfp_cache_block_size_bytes(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            cache_dtype=self.sefp_cache_dtype,
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        merged = super().merge(specs)
        assert all(isinstance(spec, HMFPFullAttentionSpec) for spec in specs), (
            "All SEFP layers in the same KV cache group must use HMFPFullAttentionSpec."
        )
        assert all(
            spec.sefp_cache_dtype == specs[0].sefp_cache_dtype for spec in specs
        ), (
            "All SEFP layers in the same KV cache group must use the same SEFP cache dtype."
        )
        return replace(
            merged,
            sefp_cache_dtype=specs[0].sefp_cache_dtype,
            indexes_kv_by_block_stride=True,
        )


@dataclass(frozen=True, kw_only=True)
class HMFPSlidingWindowSpec(SlidingWindowSpec):
    # vLLM FullAttentionSpec still exposes head_size_v, while newer
    # SlidingWindowSpec versions derive V width from head_size and removed the
    # keyword. Keep it at the Merak compatibility boundary because exported
    # Gemma/Qwen cache contracts describe K and V widths explicitly.
    head_size_v: int | None = None
    sefp_cache_dtype: str = SEFP_CACHE_DTYPE
    indexes_kv_by_block_stride: bool = True

    def __post_init__(self) -> None:
        # vLLM releases that carried head_size_v also defined this validation
        # hook; the newer native SlidingWindowSpec has neither. Preserve both
        # APIs without version-string branching.
        base_post_init = getattr(super(), "__post_init__", None)
        if callable(base_post_init):
            base_post_init()
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)
        object.__setattr__(self, "indexes_kv_by_block_stride", True)

    @property
    def real_page_size_bytes(self) -> int:
        return get_hmfp_cache_block_size_bytes(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            cache_dtype=self.sefp_cache_dtype,
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        assert all(type(spec) is cls for spec in specs), (
            "All SEFP sliding layers in the same KV cache group must use "
            "HMFPSlidingWindowSpec."
        )
        assert all(
            spec.sefp_cache_dtype == specs[0].sefp_cache_dtype for spec in specs
        ), (
            "All SEFP sliding layers in the same KV cache group must use the same SEFP cache dtype."
        )
        merged = super().merge(specs)
        return replace(
            merged,
            sefp_cache_dtype=specs[0].sefp_cache_dtype,
            indexes_kv_by_block_stride=True,
        )


def hmfp_paged_kv_cache_from_raw_tensor(
    raw_cache: torch.Tensor,
    *,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int | None = None,
    cache_dtype: str | None = SEFP_CACHE_DTYPE,
) -> HMFPPagedKVCache:
    if raw_cache.dim() != 2:
        raise ValueError(
            f"SEFP raw cache must have shape [num_blocks, bytes_per_block], got {tuple(raw_cache.shape)}"
        )

    config = HMFPKVCacheConfig.from_cache_dtype(cache_dtype)
    layout = HMFPKVCacheLayout(
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        config=config,
        head_size_v=head_size_v,
    )
    if raw_cache.shape[1] < layout.bytes_per_block:
        raise ValueError(
            f"SEFP raw cache block has {raw_cache.shape[1]} bytes, but layout requires {layout.bytes_per_block}."
        )

    raw_bytes = raw_cache.view(torch.uint8)
    num_blocks = raw_bytes.shape[0]
    if raw_bytes.stride(1) != 1:
        raise ValueError(
            "SEFP raw cache rows must be contiguous in the byte dimension, "
            f"got stride {tuple(raw_bytes.stride())}."
        )
    block_stride = raw_bytes.stride(0)

    def section_view(
        start: int, num_bytes: int, dtype: torch.dtype, shape: tuple[int, ...]
    ) -> torch.Tensor:
        dtype_size = torch.empty((), dtype=dtype).element_size()
        if num_bytes % dtype_size != 0:
            raise ValueError(
                f"SEFP section size {num_bytes} is not divisible by dtype size {dtype_size}"
            )
        end = start + num_bytes
        if end > block_stride:
            raise ValueError(
                f"SEFP section range [{start}, {end}) exceeds raw cache block stride {block_stride}"
            )
        if start % dtype_size != 0 or block_stride % dtype_size != 0:
            raise ValueError(
                f"SEFP section offset/stride is not aligned for dtype {dtype}: "
                f"offset={start}, block_stride={block_stride}"
            )
        section_bytes = torch.as_strided(
            raw_bytes,
            size=(num_blocks, num_bytes),
            stride=(block_stride, 1),
            storage_offset=raw_bytes.storage_offset() + start,
        )
        section = section_bytes.view(dtype)
        strides = list(torch.empty(shape, dtype=dtype).stride())
        strides[0] = block_stride // dtype_size
        return torch.as_strided(section, size=shape, stride=tuple(strides))

    shapes = layout.tensor_shapes(num_blocks)
    offsets = layout.tensor_offsets()
    k_qdata = section_view(
        offsets["k_qdata"],
        layout.k_qdata_bytes,
        config.k_torch_dtype,
        shapes["k_qdata"],
    )
    k_scale_exp = section_view(
        offsets["k_scale_exp"],
        layout.k_scale_exp_bytes,
        torch.int8,
        shapes["k_scale_exp"],
    )
    v_qdata = section_view(
        offsets["v_qdata"],
        layout.v_qdata_bytes,
        config.v_torch_dtype,
        shapes["v_qdata"],
    )
    v_scale_exp = section_view(
        offsets["v_scale_exp"],
        layout.v_scale_exp_bytes,
        torch.int8,
        shapes["v_scale_exp"],
    )

    return HMFPPagedKVCache(
        k_qdata=k_qdata,
        k_scale_exp=k_scale_exp,
        v_qdata=v_qdata,
        v_scale_exp=v_scale_exp,
        block_size=block_size,
        num_blocks=num_blocks,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        nshare=config.nshare,
        head_size_v=layout.head_size_v,
    )


def allocate_hmfp_kv_cache(
    batch_size: int,
    cfg: HMFPContiguousKVCacheConfig,
    device: torch.device | str,
) -> HMFPKVCache:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    qdata_shape = (
        batch_size,
        cfg.num_heads,
        cfg.num_groups,
        cfg.nshare,
        cfg.max_seq_len,
    )
    scale_shape = (
        batch_size,
        cfg.num_heads,
        cfg.num_groups,
        1,
        cfg.max_seq_len,
    )

    def alloc_one(qdata_dtype: torch.dtype) -> HMFPTensorCache:
        return HMFPTensorCache(
            qdata=torch.empty(qdata_shape, dtype=qdata_dtype, device=device),
            scale_exp=torch.empty(
                scale_shape, dtype=cfg.scale_exp_dtype, device=device
            ),
        )

    return HMFPKVCache(
        k=alloc_one(cfg.k_qdata_dtype),
        v=alloc_one(cfg.v_qdata_dtype),
    )


def allocate_hmfp_paged_kv_cache(
    *,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int | None = None,
    device: torch.device | str,
    nshare: int = HMFP_NSHARE,
    k_qdata_dtype: str | torch.dtype = torch.int8,
    v_qdata_dtype: str | torch.dtype = torch.int8,
    group_size: int | None = None,
) -> HMFPPagedKVCache:
    if group_size is not None:
        nshare = group_size
    config = HMFPKVCacheConfig(
        nshare=nshare,
        k_qdata_dtype=_torch_dtype_name(_normalize_torch_dtype(k_qdata_dtype)),
        v_qdata_dtype=_torch_dtype_name(_normalize_torch_dtype(v_qdata_dtype)),
    )
    layout = HMFPKVCacheLayout(
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        config=config,
        head_size_v=head_size_v,
    )
    shapes = layout.tensor_shapes(num_blocks)
    return HMFPPagedKVCache(
        k_qdata=torch.zeros(
            shapes["k_qdata"], dtype=config.k_torch_dtype, device=device
        ),
        k_scale_exp=torch.zeros(shapes["k_scale_exp"], dtype=torch.int8, device=device),
        v_qdata=torch.zeros(
            shapes["v_qdata"], dtype=config.v_torch_dtype, device=device
        ),
        v_scale_exp=torch.zeros(shapes["v_scale_exp"], dtype=torch.int8, device=device),
        block_size=block_size,
        num_blocks=num_blocks,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        nshare=nshare,
        head_size_v=layout.head_size_v,
    )


def allocate_hmfp_paged_kv_caches(
    *,
    num_layers: int,
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    head_size_v: int | None = None,
    device: torch.device | str,
    nshare: int = HMFP_NSHARE,
    k_qdata_dtype: str | torch.dtype = torch.int8,
    v_qdata_dtype: str | torch.dtype = torch.int8,
    group_size: int | None = None,
) -> list[HMFPPagedKVCache]:
    if group_size is not None:
        nshare = group_size
    if num_layers < 0:
        raise ValueError(f"num_layers must be non-negative, got {num_layers}")
    return [
        allocate_hmfp_paged_kv_cache(
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            head_size_v=head_size_v,
            device=device,
            nshare=nshare,
            k_qdata_dtype=k_qdata_dtype,
            v_qdata_dtype=v_qdata_dtype,
        )
        for _ in range(num_layers)
    ]


def quantize_mxint8_like(
    x: torch.Tensor,
    nshare: int = HMFP_NSHARE,
    eps: float = 1e-12,
    qdata_dtype: str | torch.dtype = torch.int8,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dim() != 4:
        raise ValueError(f"x must have shape [B, H, D, S], got {tuple(x.shape)}")

    batch_size, num_heads, head_dim, seq_len = x.shape
    if head_dim % nshare != 0:
        raise ValueError(f"head_dim={head_dim} must be divisible by nshare={nshare}")

    num_groups = head_dim // nshare
    xg = x.reshape(batch_size, num_heads, num_groups, nshare, seq_len)
    qmax = _qmax_for(qdata_dtype)
    qdata_torch_dtype = _normalize_torch_dtype(qdata_dtype)

    amax = xg.abs().amax(dim=3, keepdim=True)
    scale_exp = torch.ceil(torch.log2(torch.clamp(amax / qmax, min=eps)))
    scale_exp = torch.clamp(scale_exp, -128, 127).to(torch.int8)
    scale = torch.pow(2.0, scale_exp.float())

    qdata = torch.round(xg / scale)
    qdata = torch.clamp(qdata, -qmax, qmax).to(qdata_torch_dtype)
    return qdata, scale_exp


def dequantize_mxint8_like(
    qdata: torch.Tensor,
    scale_exp: torch.Tensor,
) -> torch.Tensor:
    if qdata.dim() != 5:
        raise ValueError(
            f"qdata must have shape [B, H, D/G, G, S], got {tuple(qdata.shape)}"
        )
    if (
        scale_exp.shape[:3] != qdata.shape[:3]
        or scale_exp.shape[3] != 1
        or scale_exp.shape[4] != qdata.shape[4]
    ):
        raise ValueError(
            f"scale_exp shape {tuple(scale_exp.shape)} is not compatible with qdata shape {tuple(qdata.shape)}"
        )

    scale = torch.pow(2.0, scale_exp.float())
    xg = qdata.float() * scale.float()
    batch_size, num_heads, num_groups, nshare, seq_len = xg.shape
    return xg.reshape(batch_size, num_heads, num_groups * nshare, seq_len)


def write_hmfp_kv_cache(
    k_fp16: torch.Tensor,
    v_fp16: torch.Tensor,
    cache: HMFPKVCache,
    start_pos: int,
    nshare: int = HMFP_NSHARE,
) -> None:
    if k_fp16.shape != v_fp16.shape:
        raise ValueError(
            f"K and V tensors must have matching shapes, got {tuple(k_fp16.shape)} and {tuple(v_fp16.shape)}"
        )
    if start_pos < 0:
        raise ValueError(f"start_pos must be non-negative, got {start_pos}")

    k_qdata, k_scale_exp = quantize_mxint8_like(
        k_fp16, nshare, qdata_dtype=cache.k.qdata.dtype
    )
    v_qdata, v_scale_exp = quantize_mxint8_like(
        v_fp16, nshare, qdata_dtype=cache.v.qdata.dtype
    )

    end_pos = start_pos + k_fp16.shape[-1]
    if end_pos > cache.k.qdata.shape[-1] or end_pos > cache.v.qdata.shape[-1]:
        raise ValueError(
            f"write range [{start_pos}, {end_pos}) exceeds cache sequence capacity"
        )

    cache.k.qdata[..., start_pos:end_pos] = k_qdata
    cache.k.scale_exp[..., start_pos:end_pos] = k_scale_exp
    cache.v.qdata[..., start_pos:end_pos] = v_qdata
    cache.v.scale_exp[..., start_pos:end_pos] = v_scale_exp


def write_hmfp_kv_cache_by_slot(
    k_token: torch.Tensor,
    v_token: torch.Tensor,
    cache: HMFPPagedKVCache,
    slot_mapping: torch.Tensor,
    nshare: int | None = None,
    k_qdata_dtype: str | torch.dtype | None = None,
    v_qdata_dtype: str | torch.dtype | None = None,
) -> None:
    if k_token.dim() != 3 or v_token.dim() != 3:
        raise ValueError(
            f"k_token/v_token must have shape [N, H, D], got {tuple(k_token.shape)}"
        )
    if k_token.shape[:2] != v_token.shape[:2]:
        raise ValueError("K and V tensors must have matching token and head dimensions")

    num_tokens, num_heads, head_dim = k_token.shape
    if num_heads != cache.num_kv_heads or head_dim != cache.head_size:
        raise ValueError(
            f"token shape [N, H={num_heads}, D={head_dim}] does not match cache H={cache.num_kv_heads}, D={cache.head_size}"
        )
    if v_token.shape[2] != cache.head_size_v:
        raise ValueError(
            f"V token head dimension {v_token.shape[2]} does not match cache {cache.head_size_v}"
        )
    if slot_mapping.numel() != num_tokens:
        raise ValueError(
            f"slot_mapping length must be {num_tokens}, got {slot_mapping.numel()}"
        )

    flat_slot_mapping = slot_mapping.reshape(-1)
    max_slot = cache.num_blocks * cache.block_size
    invalid_slots = (flat_slot_mapping < 0) | (flat_slot_mapping >= max_slot)
    if bool(invalid_slots.any().item()):
        raise ValueError(f"slot_mapping entries must be in [0, {max_slot})")

    actual_nshare = cache.nshare if nshare is None else nshare
    if actual_nshare != cache.nshare:
        raise ValueError(
            f"nshare={actual_nshare} does not match cache.nshare={cache.nshare}"
        )
    if head_dim % actual_nshare != 0:
        raise ValueError(
            f"head_dim={head_dim} must be divisible by nshare={actual_nshare}"
        )

    k_dtype = (
        cache.k_qdata.dtype
        if k_qdata_dtype is None
        else _normalize_torch_dtype(k_qdata_dtype)
    )
    v_dtype = (
        cache.v_qdata.dtype
        if v_qdata_dtype is None
        else _normalize_torch_dtype(v_qdata_dtype)
    )
    if k_dtype != cache.k_qdata.dtype:
        raise ValueError(
            f"k_qdata_dtype={k_dtype} does not match cache.k_qdata.dtype={cache.k_qdata.dtype}"
        )
    if v_dtype != cache.v_qdata.dtype:
        raise ValueError(
            f"v_qdata_dtype={v_dtype} does not match cache.v_qdata.dtype={cache.v_qdata.dtype}"
        )

    num_groups = head_dim // actual_nshare
    k = k_token.reshape(num_tokens, num_heads, num_groups, actual_nshare)

    k_amax = k.abs().amax(dim=-1, keepdim=True)

    k_qmax = _qmax_for(k_dtype)
    v_qmax = _qmax_for(v_dtype)

    k_scale_exp = torch.ceil(torch.log2(torch.clamp(k_amax / k_qmax, min=1e-12)))
    k_scale_exp = torch.clamp(k_scale_exp, -128, 127).to(torch.int8)

    k_scale = torch.pow(2.0, k_scale_exp.float())
    k_qdata = torch.clamp(torch.round(k / k_scale), -k_qmax, k_qmax).to(k_dtype)

    for token_idx in range(num_tokens):
        slot = int(flat_slot_mapping[token_idx])
        block_id = slot // cache.block_size
        offset = slot % cache.block_size
        v_group_id = offset // actual_nshare
        v_lane = offset % actual_nshare

        cache.k_qdata[block_id, :, :, :, offset] = k_qdata[token_idx]
        cache.k_scale_exp[block_id, :, :, :, offset] = k_scale_exp[token_idx]

        v_group = cache.v_qdata[block_id, :, v_group_id].float()
        v_group_scale = torch.pow(
            2.0, cache.v_scale_exp[block_id, :, v_group_id].float()
        )
        v_group = v_group * v_group_scale
        v_group[:, v_lane, :] = v_token[token_idx].float()

        v_amax = v_group.abs().amax(dim=1, keepdim=True)
        v_scale_exp = torch.ceil(torch.log2(torch.clamp(v_amax / v_qmax, min=1e-12)))
        v_scale_exp = torch.clamp(v_scale_exp, -128, 127).to(torch.int8)
        v_scale = torch.pow(2.0, v_scale_exp.float())
        v_qdata = torch.clamp(torch.round(v_group / v_scale), -v_qmax, v_qmax).to(
            v_dtype
        )

        cache.v_qdata[block_id, :, v_group_id] = v_qdata
        cache.v_scale_exp[block_id, :, v_group_id] = v_scale_exp


def copy_hmfp_cache_blocks(
    cache: HMFPPagedKVCache, block_mapping: dict[int, int] | list[tuple[int, int]]
) -> None:
    items = block_mapping.items() if isinstance(block_mapping, dict) else block_mapping
    for src_block_id, dst_block_id in items:
        cache.copy_block(int(src_block_id), int(dst_block_id))


def swap_hmfp_cache_blocks(
    cache: HMFPPagedKVCache, block_id_a: int, block_id_b: int
) -> None:
    cache.swap_blocks(block_id_a, block_id_b)


def get_hmfp_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    cache_dtype: str | None = SEFP_CACHE_DTYPE,
    head_size_v: int | None = None,
) -> tuple[int, int]:
    config = HMFPKVCacheConfig.from_cache_dtype(cache_dtype)
    layout = HMFPKVCacheLayout(
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        config=config,
        head_size_v=head_size_v,
    )
    return layout.cache_shape(num_blocks)


def get_hmfp_cache_block_size_bytes(
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    cache_dtype: str | None = SEFP_CACHE_DTYPE,
    head_size_v: int | None = None,
) -> int:
    config = HMFPKVCacheConfig.from_cache_dtype(cache_dtype)
    layout = HMFPKVCacheLayout(
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        config=config,
        head_size_v=head_size_v,
    )
    return layout.bytes_per_block
