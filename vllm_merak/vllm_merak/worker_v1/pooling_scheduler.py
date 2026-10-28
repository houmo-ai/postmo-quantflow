# SPDX-License-Identifier: Apache-2.0
"""Allow decoder pooling to consume the final context position on vLLM 0.23."""

from vllm.v1.core.sched.scheduler import Scheduler


class MerakPoolingScheduler(Scheduler):
    def schedule(self):
        # The generation scheduler reserves one position on resumed prefills.
        # Pooling has no generated token: an exact-capacity prompt must still
        # schedule its final input token. Keep request validation and the KV
        # manager at the original capacity, changing only this scheduling bound.
        if self.vllm_config.model_config.runner_type != "pooling":
            raise ValueError("MerakPoolingScheduler requires runner=pooling")
        limit = self.max_model_len
        self.max_model_len = limit + 1
        try:
            return super().schedule()
        finally:
            self.max_model_len = limit
