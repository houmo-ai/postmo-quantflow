"""Streamed Qwen MoE export preserves three independent Macaron adapters."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def test_three_adapters_through_streamed_unified_build(tmp_path):
    script = Path(__file__).with_name("_unified_e2e_worker.py")
    env = os.environ.copy()
    env.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        TOKENIZERS_PARALLELISM="false",
    )
    # This test owns one device; the public matrix schedules its cases separately.
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    log_path = tmp_path / "worker.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, str(script), "macaron-v1-tall/lora", str(tmp_path), "--low-memory"],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=int(env.get("UNIFIED_CI_TIMEOUT", "3600")))
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    assert code == 0, log_path.read_text(errors="replace")[-10000:]
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["passed"] and result["stage"] == "complete", result
    assert set(result["lora_inference_tokens"]) == {"L0", "L1", "L2"}
