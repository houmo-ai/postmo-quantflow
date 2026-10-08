import json
import shutil
from pathlib import Path

import numpy as np


def component_output_dir(model_path: Path) -> Path:
    """Return the published component directory which owns ``model_path``."""
    parent = model_path.parent
    return parent.parent if parent.name.lower() == "hmonnx" else parent


def enable_component_golden(session, model_path: Path) -> None:
    """Enable real-input HMONNX capture and reset numbering for this run."""
    # Ask the backend first so TCIM fails without touching existing artifacts.
    session.save_golden = True
    output_dir = component_output_dir(model_path)
    for step_dir in output_dir.glob("step_*"):
        if step_dir.is_dir() and step_dir.name.removeprefix("step_").isdigit():
            shutil.rmtree(step_dir)
    session.golden_dir = str(output_dir)
    if hasattr(session, "reset_step"):
        session.reset_step()
    else:
        session.step = 0
    # HMONNXGoldenInference.get_input_names() does not initialize its wrapped
    # session lazily, while these demos inspect names before the first run.
    if hasattr(session, "initialize"):
        session.initialize()


class GoldenInputRecorder:
    """Record request-level ASR inputs separately from graph tensor dumps."""

    def __init__(self, work_dir: Path, audio_path: Path, sample_rate: int, sample_count: int):
        self.output_dir = work_dir / "golden_input"
        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True)

        suffix = audio_path.suffix or ".audio"
        copied_audio = self.output_dir / f"input_audio{suffix}"
        shutil.copy2(audio_path, copied_audio)
        self._meta = {
            "audio_file": copied_audio.name,
            "source_audio": str(audio_path.resolve()),
            "sample_rate": int(sample_rate),
            "sample_count": int(sample_count),
            "duration_seconds": float(sample_count / sample_rate),
            "token_snapshot": None,
            "prefill_count": 0,
        }
        self._write_meta()

    def record_tokenizer(self, *, prompt: str, tokenizer_output: str, token_ids) -> None:
        token_array = token_ids.detach().cpu().numpy() if hasattr(token_ids, "detach") else np.asarray(token_ids)
        if token_array.ndim != 2:
            raise ValueError(f"token_ids must be two-dimensional, got shape {token_array.shape}")

        (self.output_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
        (self.output_dir / "tokenizer_output.txt").write_text(tokenizer_output, encoding="utf-8")
        np.save(self.output_dir / "token_ids.npy", token_array)
        self._meta["prefill_count"] += 1
        self._meta["token_snapshot"] = "last_prefill"
        self._meta["token_ids_shape"] = list(token_array.shape)
        self._write_meta()

    def _write_meta(self) -> None:
        (self.output_dir / "input_meta.json").write_text(
            json.dumps(self._meta, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
