# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""All registered variants through the public APIs, one process per GPU.

Ordinary pytest is sufficient: this module schedules only selected parameter
instances on the visible GPUs. Each process owns a GPU until completion, with
its own checkpoint/output directories and persistent failure log.
"""

import concurrent.futures
import json
import os
import queue
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from _unified_cases import UnifiedCase


@pytest.fixture(scope="session")
def unified_runs(request, tmp_path_factory):
    import torch

    selected = selected_cases(request.session.items)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = visible.split(",") if visible is not None else [str(i) for i in range(torch.cuda.device_count())]
    assert devices and all(d.strip() and d != "-1" for d in devices), "Unified integration CI requires a visible CUDA GPU"
    available = queue.Queue()
    for device in devices:
        available.put(device)
    root = tmp_path_factory.mktemp("unified")
    script = Path(__file__).with_name("_unified_e2e_worker.py")

    def execute(case):
        device = available.get()
        folder = root / case.model_id / case.variant
        folder.mkdir(parents=True)
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=device, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   HF_DATASETS_OFFLINE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
                   TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
        env["PYTHONPATH"] = str(script.parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            with (folder / "worker.log").open("w") as log:
                process = subprocess.Popen([sys.executable, str(script), case.id, str(folder)],
                                           env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait(timeout=int(os.environ.get("UNIFIED_CI_TIMEOUT", "3600")))
                except BaseException:
                    # Quantization may spawn descendants. Do not release a GPU
                    # back to the queue while a timed-out child still owns it.
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    raise
            return code, folder
        finally:
            available.put(device)

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(devices)) as pool:
        yield {case.id: pool.submit(execute, case) for case in selected}


def check_case_result(case, unified_runs):
    code, folder = unified_runs[case.id].result()
    log = (folder / "worker.log").read_text(errors="replace")
    assert code == 0, f"{case.id}: exit={code}; full log: {folder / 'worker.log'}\n{log[-10000:]}"
    result = json.loads((folder / "result.json").read_text())
    assert result["passed"] and result["stage"] == "complete", result


def selected_cases(items):
    """Collect the selected model cases once, independent of their test module."""
    cases = [item.callspec.params.get("case") for item in items if hasattr(item, "callspec")]
    cases = [case for case in cases if isinstance(case, UnifiedCase)]
    assert len({case.id for case in cases}) == len(cases), "Duplicate selected unified cases"
    return cases
