import unittest
from types import SimpleNamespace

import torch

from vllm_merak.hmfp_kv_cache import (
    HMFPFullAttentionSpec,
    HMFPKVCacheConfig,
    HMFPKVCacheLayout,
    HMFPPagedKVCache,
    HMFPSlidingWindowSpec,
    allocate_hmfp_paged_kv_cache,
    hmfp_paged_kv_cache_from_raw_tensor,
    write_hmfp_kv_cache_by_slot,
)
from vllm_merak.worker_v1.merak_model_runner_v1 import MerakModelRunnerV1
from vllm_merak.worker_v1.merak_worker_v1 import _get_cache_block_size_bytes


class SEFPKVCacheTest(unittest.TestCase):
    @staticmethod
    def _full_spec() -> HMFPFullAttentionSpec:
        return HMFPFullAttentionSpec(
            block_size=64,
            num_kv_heads=4,
            head_size=128,
            head_size_v=128,
            dtype=torch.float8_e4m3fn,
            sefp_cache_dtype="sefp_kint8_vint8",
        )

    @staticmethod
    def _sliding_spec(
        window: int = 1024, cache_dtype: str = "sefp_kint8_vint8"
    ) -> HMFPSlidingWindowSpec:
        return HMFPSlidingWindowSpec(
            block_size=64,
            num_kv_heads=4,
            head_size=128,
            head_size_v=128,
            dtype=torch.float8_e4m3fn,
            sefp_cache_dtype=cache_dtype,
            sliding_window=window,
        )

    def test_hmfp_sliding_spec_keeps_native_window_and_hmfp_layout(self) -> None:
        spec = self._sliding_spec()

        self.assertEqual(spec.sliding_window, 1024)
        self.assertEqual(spec.head_size_v, spec.head_size)
        self.assertIs(spec.indexes_kv_by_block_stride, True)
        self.assertEqual(
            spec.real_page_size_bytes, self._full_spec().real_page_size_bytes
        )

    def test_hmfp_sliding_spec_supports_different_kv_head_widths(self) -> None:
        spec = HMFPSlidingWindowSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=576,
            head_size_v=512,
            dtype=torch.float8_e4m3fn,
            sefp_cache_dtype="sefp_kint8_vint8",
            sliding_window=1024,
        )
        layout = HMFPKVCacheLayout(
            num_kv_heads=1,
            head_size=576,
            head_size_v=512,
            block_size=64,
        )

        self.assertEqual(spec.head_size, 576)
        self.assertEqual(spec.head_size_v, 512)
        self.assertEqual(spec.real_page_size_bytes, layout.bytes_per_block)

    def test_hmfp_sliding_spec_forces_block_stride_indexing(self) -> None:
        spec = HMFPSlidingWindowSpec(
            block_size=64,
            num_kv_heads=4,
            head_size=128,
            head_size_v=128,
            dtype=torch.float8_e4m3fn,
            sefp_cache_dtype="sefp_kint8_vint8",
            sliding_window=512,
            indexes_kv_by_block_stride=False,
        )

        self.assertIs(spec.indexes_kv_by_block_stride, True)
        self.assertIs(
            HMFPSlidingWindowSpec.merge([spec]).indexes_kv_by_block_stride, True
        )

    def test_hmfp_sliding_merge_rejects_full_or_different_window(self) -> None:
        merged = HMFPSlidingWindowSpec.merge(
            [self._sliding_spec(), self._sliding_spec()]
        )
        self.assertEqual(merged.sliding_window, 1024)
        self.assertEqual(merged.sefp_cache_dtype, "sefp_kint8_vint8")
        self.assertIs(merged.indexes_kv_by_block_stride, True)

        with self.assertRaises(AssertionError):
            HMFPSlidingWindowSpec.merge([self._sliding_spec(), self._full_spec()])
        with self.assertRaises(AssertionError):
            HMFPSlidingWindowSpec.merge([self._sliding_spec(), self._sliding_spec(512)])
        with self.assertRaises(AssertionError):
            HMFPSlidingWindowSpec.merge(
                [
                    self._sliding_spec(),
                    self._sliding_spec(cache_dtype="sefp_kint16_vint16"),
                ]
            )

    def test_cache_dtype_parser_normalizes_aliases(self) -> None:
        config = HMFPKVCacheConfig.from_cache_dtype("sefp_ki16_vi8")

        self.assertEqual(config.k_qdata_dtype, "int16")
        self.assertEqual(config.v_qdata_dtype, "int8")

    def test_cache_dtype_parser_supports_short_numeric_aliases(self) -> None:
        config = HMFPKVCacheConfig.from_cache_dtype("sefp_k8_v8")

        self.assertEqual(config.k_qdata_dtype, "int8")
        self.assertEqual(config.v_qdata_dtype, "int8")

    def test_cache_dtype_parser_rejects_unknown_modifier(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "unsupported SEFP cache dtype modifier"
        ):
            HMFPKVCacheConfig.from_cache_dtype("sefp_typo")

    def test_config_rejects_non_arch_nshare(self) -> None:
        with self.assertRaisesRegex(ValueError, "nshare must be 64"):
            HMFPKVCacheConfig(nshare=32)

    def test_compatibility_fields_use_nshare_name(self) -> None:
        fields = HMFPKVCacheConfig().compatibility_fields(
            num_kv_heads=1, head_size=64, block_size=64
        )

        self.assertEqual(fields["kv_cache_nshare"], 64)
        self.assertNotIn("kv_cache_group_size", fields)

    def test_full_attention_spec_indexes_kv_by_block_stride(self) -> None:
        spec = HMFPFullAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=64,
            head_size_v=64,
            dtype=torch.uint8,
            sefp_cache_dtype="sefp_k8_v8",
        )

        self.assertIs(spec.indexes_kv_by_block_stride, True)

    def test_full_attention_spec_forces_block_stride_indexing_when_copied_from_generic_spec(
        self,
    ) -> None:
        spec = HMFPFullAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=64,
            head_size_v=64,
            dtype=torch.uint8,
            sefp_cache_dtype="sefp_k8_v8",
            indexes_kv_by_block_stride=False,
        )

        self.assertIs(spec.indexes_kv_by_block_stride, True)
        self.assertIs(
            HMFPFullAttentionSpec.merge([spec]).indexes_kv_by_block_stride, True
        )

    def test_raw_tensor_view_handles_mixed_int16_alignment(self) -> None:
        layout = HMFPKVCacheLayout(
            num_kv_heads=1,
            head_size=64,
            block_size=64,
            config=HMFPKVCacheConfig(k_qdata_dtype="int8", v_qdata_dtype="int16"),
        )
        self.assertEqual(layout.tensor_offsets()["v_qdata"] % 2, 0)

        raw_cache = torch.empty(layout.cache_shape(num_blocks=2), dtype=torch.uint8)
        paged_cache = hmfp_paged_kv_cache_from_raw_tensor(
            raw_cache,
            block_size=64,
            num_kv_heads=1,
            head_size=64,
            cache_dtype="sefp_kint8_vint16",
        )

        self.assertIs(paged_cache.k_qdata.dtype, torch.int8)
        self.assertIs(paged_cache.v_qdata.dtype, torch.int16)
        self.assertEqual(tuple(paged_cache.v_qdata.shape), (2, 1, 1, 64, 64))

    def test_raw_tensor_view_preserves_padded_block_stride(self) -> None:
        layout = HMFPKVCacheLayout(num_kv_heads=1, head_size=64, block_size=64)
        padded_bytes_per_block = layout.bytes_per_block + 128
        raw_cache = torch.empty((2, padded_bytes_per_block), dtype=torch.uint8)

        paged_cache = hmfp_paged_kv_cache_from_raw_tensor(
            raw_cache,
            block_size=64,
            num_kv_heads=1,
            head_size=64,
            cache_dtype="sefp_k8_v8",
        )

        self.assertEqual(paged_cache.k_qdata.stride()[0], padded_bytes_per_block)
        self.assertEqual(paged_cache.k_scale_exp.stride()[0], padded_bytes_per_block)
        self.assertEqual(paged_cache.v_qdata.stride()[0], padded_bytes_per_block)
        self.assertEqual(paged_cache.v_scale_exp.stride()[0], padded_bytes_per_block)

    def test_absorbed_mla_raw_tensor_view_keeps_independent_k_and_v_widths(self) -> None:
        layout = HMFPKVCacheLayout(
            num_kv_heads=1,
            head_size=576,
            head_size_v=512,
            block_size=64,
        )
        raw_cache = torch.empty(layout.cache_shape(num_blocks=2), dtype=torch.uint8)

        cache = hmfp_paged_kv_cache_from_raw_tensor(
            raw_cache,
            block_size=64,
            num_kv_heads=1,
            head_size=576,
            head_size_v=512,
            cache_dtype="sefp_k8_v8",
        )

        self.assertEqual(cache.head_size, 576)
        self.assertEqual(cache.head_size_v, 512)
        self.assertEqual(tuple(cache.k_qdata.shape), (2, 1, 9, 64, 64))
        self.assertEqual(tuple(cache.v_qdata.shape), (2, 1, 1, 64, 512))

        key = torch.zeros((1, 1, 576), dtype=torch.float16)
        value = torch.zeros((1, 1, 512), dtype=torch.float16)
        value[0, 0, -1] = 1.0
        write_hmfp_kv_cache_by_slot(key, value, cache, torch.tensor([0]))
        self.assertNotEqual(int(cache.v_qdata[0, 0, 0, 0, -1]), 0)

    def test_runner_wraps_absorbed_mla_raw_pages_with_asymmetric_layout(self) -> None:
        spec = HMFPFullAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=576,
            head_size_v=512,
            dtype=torch.uint8,
            sefp_cache_dtype="sefp_k8_v8",
        )
        raw_cache = torch.empty((2, spec.real_page_size_bytes), dtype=torch.uint8)
        placeholder = SimpleNamespace(
            layer_name="model.layers.5.attn",
            kv_cache=raw_cache,
            num_kv_heads=1,
            head_size=576,
            head_size_v=512,
            get_kv_cache_spec=lambda _config: spec,
        )
        runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
        runner.cache_dtype = "sefp_k8_v8"
        runner.vllm_config = SimpleNamespace()
        runner.model = SimpleNamespace(
            cache_dtype="sefp_k8_v8",
            placeholder_attn_layers=[placeholder],
        )

        runner._wrap_sefp_kv_cache_handles()

        self.assertIsInstance(placeholder.kv_cache, HMFPPagedKVCache)
        self.assertEqual(tuple(placeholder.kv_cache.k_qdata.shape), (2, 1, 9, 64, 64))
        self.assertEqual(tuple(placeholder.kv_cache.v_qdata.shape), (2, 1, 1, 64, 512))

    def test_runner_wraps_from_finalized_attention_spec_after_hybrid_min_block(self) -> None:
        spec = HMFPFullAttentionSpec(
            block_size=64,
            num_kv_heads=1,
            head_size=64,
            head_size_v=64,
            dtype=torch.uint8,
            sefp_cache_dtype="sefp_k8_v8",
            page_size_padded=8448,
        )
        raw_cache = torch.empty((2, spec.page_size_bytes), dtype=torch.uint8)
        placeholder = SimpleNamespace(
            layer_name="model.layers.3.attn",
            kv_cache=raw_cache,
            num_kv_heads=1,
            head_size=64,
            head_size_v=64,
            get_kv_cache_spec=lambda _config: self.fail(
                "finalized spec must be used instead of the mutated global config"
            ),
        )
        group = SimpleNamespace(
            kv_cache_spec=spec,
            layer_names=[placeholder.layer_name],
        )
        runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
        runner.cache_dtype = "sefp_k8_v8"
        runner.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(block_size=16)
        )
        runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[group])
        runner.model = SimpleNamespace(
            cache_dtype="sefp_k8_v8",
            placeholder_attn_layers=[placeholder],
        )

        runner._publish_finalized_attention_block_size()
        runner._wrap_sefp_kv_cache_handles()

        self.assertIsInstance(placeholder.kv_cache, HMFPPagedKVCache)
        self.assertEqual(placeholder.kv_cache.block_size, 64)
        self.assertEqual(runner.model._merak_attention_block_size, 64)

    def test_worker_sizes_absorbed_mla_pages_from_both_cache_widths(self) -> None:
        layout = HMFPKVCacheLayout(
            num_kv_heads=1,
            head_size=576,
            head_size_v=512,
            block_size=64,
        )

        actual = _get_cache_block_size_bytes(
            "sefp_k8_v8",
            [[1, 1, 262144, 576]],
            [[1, 1, 262144, 512]],
        )

        self.assertEqual(actual, layout.bytes_per_block)

    def test_v_cache_layout_groups_sequence_axis(self) -> None:
        layout = HMFPKVCacheLayout(num_kv_heads=2, head_size=128, block_size=192)

        self.assertEqual(layout.k_qdata_shape, (2, 2, 64, 192))
        self.assertEqual(layout.k_scale_exp_shape, (2, 2, 1, 192))
        self.assertEqual(layout.v_qdata_shape, (2, 3, 64, 128))
        self.assertEqual(layout.v_scale_exp_shape, (2, 3, 1, 128))
        self.assertEqual(layout.v_qdata_bytes, 2 * 3 * 64 * 128)
        self.assertEqual(layout.v_scale_exp_bytes, 2 * 3 * 128)

    def test_layout_rejects_block_size_not_multiple_of_nshare(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "block_size=32 must be divisible by nshare=64"
        ):
            HMFPKVCacheLayout(num_kv_heads=1, head_size=64, block_size=32)

    def test_write_by_slot_groups_value_scale_by_sequence_axis(self) -> None:
        cache = allocate_hmfp_paged_kv_cache(
            num_blocks=1,
            block_size=64,
            num_kv_heads=1,
            head_size=64,
            device="cpu",
        )
        key = torch.zeros((2, 1, 64), dtype=torch.float32)
        value = torch.zeros((2, 1, 64), dtype=torch.float32)
        value[0, 0, 0] = 1.0
        value[1, 0, 0] = 256.0

        write_hmfp_kv_cache_by_slot(key[:1], value[:1], cache, torch.tensor([0]))
        self.assertEqual(int(cache.v_scale_exp[0, 0, 0, 0, 0]), -6)

        write_hmfp_kv_cache_by_slot(key[1:], value[1:], cache, torch.tensor([1]))

        self.assertEqual(tuple(cache.v_qdata.shape), (1, 1, 1, 64, 64))
        self.assertEqual(int(cache.v_scale_exp[0, 0, 0, 0, 0]), 2)
        self.assertEqual(int(cache.v_qdata[0, 0, 0, 0, 0]), 0)
        self.assertEqual(int(cache.v_qdata[0, 0, 0, 1, 0]), 64)

    def test_recycled_v_block_stale_lanes_do_not_change_new_request_value(self) -> None:
        """A recycled V block must behave like a freshly zeroed block.

        V cache lanes share one exponent in groups of 64.  The second cache
        models a physical block recycled from an earlier request: lane 63
        still contains a large, valid SEFP value while the new request writes
        only lane 0.  A block-lifecycle clear must make the new value
        independent of that prior request content.
        """

        def new_a16_cache():
            return allocate_hmfp_paged_kv_cache(
                num_blocks=2,
                block_size=64,
                num_kv_heads=1,
                head_size=64,
                device="cpu",
                k_qdata_dtype="int16",
                v_qdata_dtype="int16",
            )

        clean_cache = new_a16_cache()
        recycled_cache = new_a16_cache()

        # 16384 * 2**1 = 32768 in an untouched lane from the old request.
        recycled_cache.v_qdata[0].fill_(7)
        recycled_cache.v_qdata[1, 0, 0, 63, 0] = 16384
        recycled_cache.v_scale_exp[1, 0, 0, 0, 0] = 1

        key = torch.zeros((1, 1, 64), dtype=torch.float32)
        value = torch.zeros_like(key)
        value[0, 0, 0] = 1.0
        slot_mapping = torch.tensor([64])

        runner = MerakModelRunnerV1.__new__(MerakModelRunnerV1)
        runner.model = type(
            "Model",
            (),
            {
                "placeholder_attn_layers": [
                    type(
                        "Layer",
                        (),
                        {"layer_name": "attn.0", "kv_cache": recycled_cache},
                    )()
                ]
            },
        )()
        runner.kv_cache_config = SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    layer_names=["attn.0"],
                    kv_cache_spec=HMFPFullAttentionSpec(
                        block_size=64,
                        num_kv_heads=1,
                        head_size=64,
                        head_size_v=64,
                        dtype=torch.uint8,
                        sefp_cache_dtype="sefp_kint16_vint16",
                    ),
                )
            ]
        )
        runner._zero_block_ids([1])

        self.assertTrue(torch.all(recycled_cache.v_qdata[0] == 7))

        write_hmfp_kv_cache_by_slot(key, value, clean_cache, slot_mapping)
        write_hmfp_kv_cache_by_slot(key, value, recycled_cache, slot_mapping)

        def decoded_target(cache) -> float:
            qdata = int(cache.v_qdata[1, 0, 0, 0, 0])
            scale_exp = int(cache.v_scale_exp[1, 0, 0, 0, 0])
            return float(qdata * (2.0**scale_exp))

        self.assertEqual(
            decoded_target(recycled_cache),
            decoded_target(clean_cache),
            "recycled block residue changed the newly written V value; "
            "newly allocated FullAttention blocks must be zeroed before RMW",
        )

    def test_write_by_slot_rejects_negative_slot(self) -> None:
        self._assert_write_by_slot_rejects_invalid_slots(torch.tensor([-1]))

    def test_write_by_slot_rejects_too_large_slot(self) -> None:
        self._assert_write_by_slot_rejects_invalid_slots(torch.tensor([64]))

    def _assert_write_by_slot_rejects_invalid_slots(
        self, slot_mapping: torch.Tensor
    ) -> None:
        cache = allocate_hmfp_paged_kv_cache(
            num_blocks=1,
            block_size=64,
            num_kv_heads=1,
            head_size=64,
            device="cpu",
        )
        key = torch.zeros((1, 1, 64), dtype=torch.float16)
        value = torch.zeros((1, 1, 64), dtype=torch.float16)

        with self.assertRaisesRegex(ValueError, "slot_mapping entries"):
            write_hmfp_kv_cache_by_slot(key, value, cache, slot_mapping)


if __name__ == "__main__":
    unittest.main()
