"""Configuration, source verification and reproducible data for CMS-909.

This module uses the standard library at import time so --help and input
validation work without loading Torch, CUDA or xhquant.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

EXAMPLE_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLE_DIR.parents[3]
DEFAULT_CUSTOMER_ROOT = EXAMPLE_DIR / "customer"
MANIFEST = EXAMPLE_DIR / "customer_sources.json"
_customer_root = DEFAULT_CUSTOMER_ROOT
_customer_modules = {}


def repo_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=unique_keys)


def parse_training_args(argv=None):
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config")
    known, _ = probe.parse_known_args(argv)
    config = {}
    defaults = {"variant": "B", "bits": 16, "seed": 17, "steps": 300,
                "output": None, "customer_sources": str(DEFAULT_CUSTOMER_ROOT),
                "check_only": False, "relu_path": "nl", "diagnostic_override": None}
    parser = argparse.ArgumentParser(description="Customer model GPU/XH2 ALIGNED numerical training (not NPU execution).")
    if known.config:
        try:
            config = read_json(repo_path(known.config))
            if not isinstance(config, dict):
                raise ValueError("The configuration must be a JSON object")
            unknown = config.keys() - defaults.keys()
            if unknown:
                raise ValueError("Unknown configuration keys: " + ", ".join(sorted(unknown)))
        except (OSError, ValueError) as error:
            parser.error(str(error))
    parser.add_argument("--config", help="JSON configuration; relative paths resolve from repository root")
    parser.add_argument("--variant", choices=("A", "B", "C"))
    parser.add_argument("--bits", type=int, choices=(16, 8))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--output", help="New/empty result directory, relative to repository root")
    parser.add_argument("--customer-sources", help="Root containing the two verified customer files")
    parser.add_argument("--check-only", action="store_true", default=None,
                        help="Validate explicit FP32 backward only; no training/checkpoint")
    parser.add_argument("--relu-path", choices=("nl", "vp"))
    parser.add_argument("--diagnostic-override", choices=("forward_matmul_fp32", "backward_matmul_fp32"))
    supplied = vars(parser.parse_args(argv))
    merged = {**defaults, **config}
    merged.update({k: v for k, v in supplied.items() if k != "config" and v is not None})
    try:
        for key in ("bits", "seed", "steps"):
            if type(merged[key]) is not int:
                raise ValueError(f"{key} must be an integer")
        if merged["bits"] not in (16, 8):
            raise ValueError("bits must be 16 or 8")
        if merged["steps"] < 1 or merged["seed"] < 0:
            raise ValueError("steps must be positive and seed nonnegative")
        if merged["variant"] not in ("A", "B", "C"):
            raise ValueError("variant must be A, B or C")
        if merged["relu_path"] not in ("nl", "vp"):
            raise ValueError("relu_path must be nl or vp")
        if merged["diagnostic_override"] not in (None, "forward_matmul_fp32", "backward_matmul_fp32"):
            raise ValueError("unsupported diagnostic_override")
        if type(merged["check_only"]) is not bool:
            raise ValueError("check_only must be boolean")
        for key in ("customer_sources", "output"):
            if merged[key] is not None and (not isinstance(merged[key], str) or not merged[key].strip()):
                raise ValueError(f"{key} must be a nonempty path string")
        if merged["output"] is None:
            name = "A_fp32" if merged["variant"] == "A" else f'{merged["variant"]}_sefp{merged["bits"]}'
            merged["output"] = f'work_dirs/zte_train/runs/{name}_seed{merged["seed"]}'
        merged["output"] = repo_path(merged["output"])
        merged["customer_sources"] = repo_path(merged["customer_sources"])
    except (TypeError, ValueError) as error:
        parser.error(str(error))
    merged["config"] = repo_path(known.config) if known.config else None
    return argparse.Namespace(**merged)


def verify_customer_sources(root=None):
    root = repo_path(root if root is not None else _customer_root)
    manifest = read_json(MANIFEST)
    verified = {}
    for name, spec in manifest["files"].items():
        candidates = [root / spec["path"], root / spec["archive_layout"]]
        path = next((p for p in candidates if p.is_file()), None)
        if path is None:
            raise FileNotFoundError(f"Missing customer {name}: expected {candidates[0]} (or {candidates[1]}). "
                                    "Restore CMS-909 originals or use --customer-sources.")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != spec["sha256"]:
            raise ValueError(f"Customer {name} SHA256 mismatch: {path}. Expected CMS-909 originals; do not silently use another model version.")
        verified[name] = path.resolve()
    return verified


def configure_customer_sources(root):
    global _customer_root
    files = verify_customer_sources(root)
    _customer_root = repo_path(root)
    return files


def load_customer_modules():
    files = verify_customer_sources()
    key = tuple(files.values())
    if key not in _customer_modules:
        loaded = []
        for name, path in files.items():
            module_name = "_zte_customer_" + name + "_" + hashlib.sha256(str(path).encode()).hexdigest()[:12]
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            loaded.append(module)
        _customer_modules[key] = tuple(loaded)
    return _customer_modules[key]


def prepare_output(path):
    path = repo_path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Output must be a new or empty directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def git_revision(directory):
    """Provenance is optional for wheels; source hashes remain mandatory."""
    import subprocess
    try:
        result = subprocess.run(["git", "-C", str(directory), "rev-parse", "HEAD"],
                                capture_output=True, text=True)
    except OSError as error:
        return {"revision": None, "error": str(error)}
    return {"revision": result.stdout.strip() if result.returncode == 0 else None,
            "error": result.stderr.strip() if result.returncode else None}


def dataset(size, seed):
    import torch
    rng = torch.Generator().manual_seed(seed)
    labels = torch.arange(size) % 9
    x = torch.zeros(size, 9, 13)
    x.scatter_(2, labels[:, None, None].expand(-1, 9, 1), 0.9)
    x.add_(torch.rand(x.shape, generator=rng) * 0.08)
    return x, labels


def rng_state():
    import torch
    return {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}


def restore_rng(state):
    import torch
    torch.set_rng_state(state["cpu"].cpu())
    torch.cuda.set_rng_state(state["cuda"].cpu())
