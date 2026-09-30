import os
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import replace
from typing import Any, Callable, Set, cast

import torch
import torch.distributed
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
)
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.profiler.wrapper import CudaProfilerWrapper, TorchProfilerWrapper
from vllm.tasks import SupportedTask
from vllm.utils.torch_utils import set_random_seed
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    DraftTokenIds,
    ModelRunnerOutput,
)
from vllm.v1.utils import compute_iteration_details
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

from ..config import MERAK_PREFERRED_BLOCK_SIZE
from ..hmfp_kv_cache import get_hmfp_cache_block_size_bytes, is_hmfp_cache_dtype
from ..kv_capacity import (
    CapacityGeometry,
    CapacityPlan,
    build_capacity_plan,
    derive_capacity_geometry_from_groups,
    format_capacity_log_lines,
    parse_virtual_gib_budget,
    physical_layers_from_groups,
    project_physical_storage,
)
from ..utils import resolve_torch_dtype
from .merak_model_runner_v1 import MerakModelRunnerV1

# vLLM configures handlers for the ``vllm`` logger tree only. Keep this
# worker's startup diagnostics under that tree so INFO capacity projections
# are visible with the standard vLLM logging configuration.
logger = init_logger(f"vllm.{__name__}")

_SUPPORTED_RUNNER_TYPES = frozenset({"generate", "pooling"})


def _validate_runner_type(runner_type: str) -> None:
    if runner_type not in _SUPPORTED_RUNNER_TYPES:
        raise NotImplementedError(f"Runner type {runner_type} is not supported yet.")


def _get_cache_block_size_bytes(
    cache_dtype: str,
    key_cache_config: list[list[int]],
    value_cache_config: list[list[int]],
) -> int:
    if is_hmfp_cache_dtype(cache_dtype):
        total_bytes = 0
        for key_cache_shape, value_cache_shape in zip(
            key_cache_config, value_cache_config, strict=True
        ):
            if key_cache_shape[:3] != value_cache_shape[:3]:
                raise ValueError(
                    "SEFP K/V cache config expects matching batch/head/context dimensions."
                )
            _, num_kv_heads, _, head_size = key_cache_shape
            head_size_v = value_cache_shape[-1]
            total_bytes += get_hmfp_cache_block_size_bytes(
                block_size=MERAK_PREFERRED_BLOCK_SIZE,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                head_size_v=head_size_v,
                cache_dtype=cache_dtype,
            )
        return total_bytes

    total_elements = 0
    for key_cache_shape, value_cache_shape in zip(
        key_cache_config, value_cache_config, strict=True
    ):
        total_elements += key_cache_shape[1] * key_cache_shape[2] * key_cache_shape[3]
        total_elements += (
            value_cache_shape[1] * value_cache_shape[2] * value_cache_shape[3]
        )

    element_size = torch.empty(
        [], dtype=resolve_torch_dtype(cache_dtype)
    ).element_size()
    return element_size * total_elements


class MerakWorkerV1(WorkerBase):
    """Minimal V1 worker used to load a Merak model under the vLLM V1 engine.

    This worker currently focuses on successful model initialization and KV
    cache sizing. The execution path can be expanded once the load-only path is
    stable under the current vLLM worker APIs.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )
        self.model: nn.Module | None = None
        self.key_cache_config: list[list[int]] = []
        self.value_cache_config: list[list[int]] = []
        self.cache_dtype: str = getattr(
            self.vllm_config.cache_config,
            "merak_logical_kv_cache_dtype",
            self.vllm_config.cache_config.cache_dtype,
        )
        self.kv_cache_config: KVCacheConfig | None = None
        self._capacity_geometry: CapacityGeometry | None = None
        self._capacity_plan: CapacityPlan | None = None
        self.profiler: Any | None = None
        self.profiler_config = vllm_config.profiler_config
        if self.profiler_config.profiler not in ("torch", "cuda", None):
            raise ValueError(f"Unknown profiler type: {self.profiler_config.profiler}")

        self.model_path = vllm_config.model_config.model
        self.model_arch = vllm_config.model_config.architecture
        self.model_runner: MerakModelRunnerV1 | None = None

        # GPUModelRunner initializes the speculative drafter eagerly and
        # queries the TP/PP groups while doing so.  Merak used to construct the
        # runner before initializing Gloo model parallelism; that happened to
        # work only while speculative_config was always None.  Keep the same
        # ordering as upstream GPUWorker: distributed/model-parallel state must
        # exist before any model runner is constructed.
        self.init_distributed_environment()

        model_config = self.model_config
        _validate_runner_type(model_config.runner_type)
        device = torch.device(
            f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
        )
        self.model_runner = MerakModelRunnerV1(
            vllm_config, self.cache_dtype, device=device
        )  # type: ignore

    def init_device(self) -> None:
        set_random_seed(self.model_config.seed)

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        if self.model_runner is None:
            raise RuntimeError("Merak model runner was not initialized.")
        self.model_runner.load_model(load_dummy_weights)
        self.model = self.model_runner.get_model()

        # Pull KV cache shape metadata from the loaded model so
        # get_cache_block_size_bytes() reflects the real export. Until this
        # runs, both lists are empty and the helper would return 0.
        self.key_cache_config = list(getattr(self.model, "key_cache_config", []))
        self.value_cache_config = list(getattr(self.model, "value_cache_config", []))
        model_cache_dtype = getattr(self.model, "cache_dtype", None)
        if isinstance(model_cache_dtype, str):
            self.cache_dtype = model_cache_dtype

    def get_model(self) -> nn.Module:
        if self.model_runner is None:
            raise RuntimeError("Merak model runner was not initialized.")
        return self.model_runner.get_model()

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        if scheduler_output.total_num_scheduled_tokens == 0:
            return EMPTY_MODEL_RUNNER_OUTPUT
        if self.model_runner is None:
            raise RuntimeError("Merak model runner was not initialized.")
        profile_context = nullcontext()
        if self.profiler is not None:
            self.profiler.step()
            details = compute_iteration_details(scheduler_output)
            annotation = (
                f"execute_context_{details.num_ctx_requests}"
                f"({details.num_ctx_tokens})_generation_"
                f"{details.num_generation_requests}"
                f"({details.num_generation_tokens})"
            )
            profile_context = self.profiler.annotate_context_manager(annotation)
        with profile_context:
            output = self.model_runner.execute_model(scheduler_output)
        return cast(ModelRunnerOutput | AsyncModelRunnerOutput | None, output)

    def profile(self, is_start: bool = True, profile_prefix: str | None = None) -> None:
        """Start or stop the vLLM worker profiler for the Merak executor."""
        if self.profiler_config.profiler is None:
            raise RuntimeError(
                "Profiling is not enabled. Configure profiler_config.profiler "
                "and profiler_config.torch_profiler_dir first."
            )

        if is_start:
            if self.profiler is None:
                from vllm.distributed.utils import get_worker_rank_suffix

                rank_suffix = get_worker_rank_suffix(global_rank=self.rank)
                trace_name = (
                    f"{profile_prefix}_{rank_suffix}" if profile_prefix else rank_suffix
                )
                if self.profiler_config.profiler == "torch":
                    self.profiler = TorchProfilerWrapper(
                        self.profiler_config,
                        worker_name=trace_name,
                        local_rank=self.local_rank,
                        activities=["CPU", "CUDA"],
                    )
                else:
                    self.profiler = CudaProfilerWrapper(self.profiler_config)
            self.profiler.start()
            return

        if self.profiler is None:
            logger.warning("Profiler was not started, nothing to stop.")
            return
        self.profiler.stop()

    def sample_tokens(
        self, grammar_output: GrammarOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput:
        if self.model_runner is None:
            raise RuntimeError("Merak model runner was not initialized.")
        sample_tokens_fn = cast(
            Callable[
                [GrammarOutput | None], ModelRunnerOutput | AsyncModelRunnerOutput
            ],
            self.model_runner.sample_tokens,
        )
        return sample_tokens_fn(grammar_output)

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        """Return and clear draft IDs produced by speculative sampling.

        V1 EngineCore retrieves draft tokens in a separate post-step RPC.
        Merak delegates sampling to ``GPUModelRunner`` already, so the worker
        must expose the matching retrieval hook as upstream ``GPUWorker``
        does.
        """
        if self.model_runner is None:
            raise RuntimeError("Merak model runner was not initialized.")
        return self.model_runner.take_draft_token_ids()

    def init_distributed_environment(self) -> None:
        init_distributed_environment(
            world_size=self.parallel_config.world_size,
            rank=self.rank,
            distributed_init_method=self.distributed_init_method,
            backend="gloo",
        )
        torch.distributed.all_reduce(torch.zeros(1).cpu())
        ensure_model_parallel_initialized(
            self.parallel_config.tensor_parallel_size,
            self.parallel_config.pipeline_parallel_size,
        )

    def get_cache_block_size_bytes(self) -> int:
        return _get_cache_block_size_bytes(
            self.cache_dtype,
            self.key_cache_config,
            self.value_cache_config,
        )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        if self.model_runner is None:
            raise RuntimeError("Merak model runner was not initialized.")
        return self.model_runner.get_kv_cache_spec()

    def determine_available_memory(self) -> int:
        # The V1 engine may construct the worker in a child process after the
        # platform ``check_and_update_config`` hook has already returned.  Make
        # the hybrid PageAttention compatibility patch explicit here so the
        # stock vLLM page-size unifier cannot run first.
        from vllm_merak.platform import register_kv_cache_spec

        register_kv_cache_spec()
        kv_cache_specs = self.get_kv_cache_spec()
        if not kv_cache_specs:
            raise RuntimeError(
                "Merak cannot plan KV capacity before the model has emitted its KV cache specs."
            )
        planning_groups = get_kv_cache_groups(
            self.vllm_config,
            deepcopy(kv_cache_specs),
        )
        geometry = derive_capacity_geometry_from_groups(
            planning_groups,
            max_model_len=int(self.model_config.max_model_len),
            max_num_batched_tokens=int(self.scheduler_config.max_num_batched_tokens),
        )

        block_override = getattr(self.cache_config, "num_gpu_blocks_override", None)
        explicit_kv_cache_bytes = getattr(
            self.cache_config, "kv_cache_memory_bytes", None
        )
        env_value = os.environ.get("VLLM_CPU_KVCACHE_SPACE")
        if block_override is not None:
            plan = build_capacity_plan(
                geometry,
                requested_blocks=int(block_override),
                source="num_gpu_blocks_override",
            )
        elif explicit_kv_cache_bytes is not None:
            plan = build_capacity_plan(
                geometry,
                virtual_budget_bytes=int(explicit_kv_cache_bytes),
                source="kv_cache_memory_bytes (virtual scheduler budget)",
            )
        elif env_value is not None:
            plan = build_capacity_plan(
                geometry,
                virtual_budget_bytes=parse_virtual_gib_budget(
                    env_value,
                    variable_name="VLLM_CPU_KVCACHE_SPACE",
                ),
                source="VLLM_CPU_KVCACHE_SPACE",
            )
        else:
            plan = build_capacity_plan(geometry)

        self._capacity_geometry = geometry
        self._capacity_plan = plan
        return plan.virtual_budget_bytes

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        if self.model_runner is None:
            raise RuntimeError("Merak model runner was not initialized.")

        final_geometry = derive_capacity_geometry_from_groups(
            kv_cache_config.kv_cache_groups,
            max_model_len=int(self.model_config.max_model_len),
            max_num_batched_tokens=int(self.scheduler_config.max_num_batched_tokens),
        )
        existing_plan = getattr(self, "_capacity_plan", None)
        validated_plan = build_capacity_plan(
            final_geometry,
            requested_blocks=int(kv_cache_config.num_blocks),
            source="final vLLM KVCacheConfig",
        )
        if (
            existing_plan is not None
            and existing_plan.num_blocks == validated_plan.num_blocks
        ):
            final_plan = replace(
                validated_plan,
                source=existing_plan.source,
                virtual_budget_bytes=existing_plan.virtual_budget_bytes,
            )
        else:
            final_plan = validated_plan

        get_layer_devices = getattr(
            self.model_runner, "_get_kv_cache_layer_devices", None
        )
        layer_devices = get_layer_devices() if callable(get_layer_devices) else {}
        physical_owner_by_layer = None
        physical_pages_by_layer = None
        runner_model = getattr(self.model_runner, "model", None)
        gemma4_contract = getattr(runner_model, "gemma4_cache_contract", None)
        if gemma4_contract is not None:
            from ..model_executor.model_loader.gemma4_cache_contract import (
                Gemma4CacheContract,
            )

            if not isinstance(gemma4_contract, Gemma4CacheContract):
                raise RuntimeError("Invalid Gemma4 physical cache contract type")
            if gemma4_contract.version != 2:
                raise RuntimeError(
                    "Merak physical cache projection requires Gemma4 contract-v2"
                )
            layer_name_by_idx = {
                binding.layer_idx: binding.layer_name
                for binding in gemma4_contract.bindings
            }
            physical_owner_by_layer = {
                binding.layer_name: layer_name_by_idx[binding.cache_owner_layer_idx]
                for binding in gemma4_contract.bindings
                if binding.cache_owner_layer_idx is not None
            }
            physical_pages_by_layer = {
                binding.layer_name: binding.fixed_physical_pages
                for binding in gemma4_contract.bindings
                if binding.cache_owner_layer_idx == binding.layer_idx
                and binding.fixed_physical_pages is not None
            }
        full_layers, linear_layers = physical_layers_from_groups(
            kv_cache_config.kv_cache_groups,
            layer_devices=layer_devices,
            linear_default_device=getattr(
                self.model_runner, "device", torch.device("cpu")
            ),
            kv_cache_tensors=kv_cache_config.kv_cache_tensors or None,
            physical_owner_by_layer=physical_owner_by_layer,
            physical_pages_by_layer=physical_pages_by_layer,
        )
        projection = project_physical_storage(
            num_blocks=kv_cache_config.num_blocks,
            full_layers=full_layers,
            linear_layers=linear_layers,
            linear_slots=int(self.scheduler_config.max_num_seqs),
            virtual_effective_bytes=final_plan.virtual_effective_bytes,
        )
        for line in format_capacity_log_lines(final_plan, projection):
            logger.info("%s", line)

        self._capacity_geometry = final_geometry
        self._capacity_plan = final_plan
        self.kv_cache_config = kv_cache_config
        self.model_runner.initialize_kv_cache(kv_cache_config)

    def compile_or_warm_up_model(self) -> CompilationTimes:
        return CompilationTimes(
            language_model=self.compilation_config.compilation_time,
            encoder=self.compilation_config.encoder_compilation_time,
        )

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        if self.model_runner is None:
            return ()
        return self.model_runner.get_supported_tasks()

    def list_loras(self) -> Set[int]:
        raise NotImplementedError("LoRA is not supported.")

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError("LoRA is not supported.")

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError("LoRA is not supported.")

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError("LoRA is not supported.")
