#!/usr/bin/env python3
"""Validate the independently installed vLLM Merak runtime environment."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
MODELZOO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILES = PLUGIN_ROOT / "runtime_profiles.toml"


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    expected: str
    actual: str


def load_profile(config_path: Path, profile_name: str | None) -> tuple[str, dict]:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    selected = profile_name or config["default_profile"]
    try:
        profile = config["profiles"][selected]
    except KeyError as exc:
        available = ", ".join(sorted(config.get("profiles", {})))
        raise ValueError(f"unknown profile {selected!r}; available: {available}") from exc
    return selected, profile


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "<not installed>"


def _editable_source(name: str) -> Path | None:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return None

    raw = distribution.read_text("direct_url.json")
    if not raw:
        return None
    payload = json.loads(raw)
    if not payload.get("dir_info", {}).get("editable", False):
        return None
    parsed = urlparse(payload.get("url", ""))
    if parsed.scheme != "file":
        return None
    return Path(unquote(parsed.path)).resolve()


def _entry_point_value(group: str, name: str) -> str:
    matches = [entry_point for entry_point in importlib.metadata.entry_points(group=group) if entry_point.name == name]
    if len(matches) != 1:
        return f"<expected one, found {len(matches)}>"
    return matches[0].value


def _pip_check() -> Check:
    completed = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        check=False,
        text=True,
    )
    output = "\n".join(part.strip() for part in (completed.stdout, completed.stderr) if part.strip())
    return Check(
        "pip-check",
        completed.returncode == 0,
        "No broken requirements found.",
        output or f"pip check exited with status {completed.returncode}",
    )


def _runtime_capability_checks(profile: dict) -> list[Check]:
    runtime = profile.get("runtime", {})
    checks: list[Check] = []

    if runtime.get("cuda_required", False):
        import torch

        available = torch.cuda.is_available()
        checks.append(Check("runtime:cuda", available, "available", str(available)))

    for module_name in runtime.get("required_modules", []):
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            checks.append(
                Check(
                    f"runtime:module:{module_name}",
                    False,
                    "importable",
                    f"{type(exc).__name__}: {exc}",
                )
            )
        else:
            checks.append(Check(f"runtime:module:{module_name}", True, "importable", "importable"))

    if runtime.get("required_torch_ops"):
        import torch

        for qualified_name in runtime["required_torch_ops"]:
            value = torch.ops
            try:
                for part in qualified_name.split("."):
                    value = getattr(value, part)
            except AttributeError:
                found = False
            else:
                found = value is not None
            checks.append(
                Check(
                    f"runtime:torch-op:{qualified_name}",
                    found,
                    "registered",
                    "registered" if found else "missing",
                )
            )
            if qualified_name == "_C.apply_repetition_penalties_" and found:
                try:
                    logits = torch.tensor([[2.0, -3.0, 4.0]], device="cuda")
                    prompt_mask = torch.tensor([[True, False, False]], device="cuda")
                    output_mask = torch.tensor([[False, True, False]], device="cuda")
                    penalties = torch.tensor([2.0], device="cuda")
                    value(logits, prompt_mask, output_mask, penalties)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        logits.cpu(),
                        torch.tensor([[1.0, -6.0, 4.0]]),
                        rtol=0.0,
                        atol=0.0,
                    )
                except Exception as exc:
                    checks.append(
                        Check(
                            "runtime:torch-op-call:apply_repetition_penalties_",
                            False,
                            "CUDA kernel returns [1.0, -6.0, 4.0]",
                            f"{type(exc).__name__}: {exc}",
                        )
                    )
                else:
                    checks.append(
                        Check(
                            "runtime:torch-op-call:apply_repetition_penalties_",
                            True,
                            "CUDA kernel returns [1.0, -6.0, 4.0]",
                            "CUDA kernel returned [1.0, -6.0, 4.0]",
                        )
                    )

    return checks


def check_environment(
    profile: dict,
    *,
    plugin_root: Path = PLUGIN_ROOT,
    modelzoo_root: Path = MODELZOO_ROOT,
) -> list[Check]:
    checks: list[Check] = []
    expected_python = profile["python"]
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    checks.append(Check("python", actual_python == expected_python, expected_python, actual_python))

    for package, expected in profile["packages"].items():
        actual = _distribution_version(package)
        checks.append(Check(f"package:{package}", actual == expected, expected, actual))

    for package, expected in profile.get("package_specifiers", {}).items():
        actual = _distribution_version(package)
        try:
            matches = Version(actual) in SpecifierSet(expected)
        except (InvalidSpecifier, InvalidVersion):
            matches = False
        checks.append(Check(f"package:{package}", matches, expected, actual))

    expected_sources = {
        "vllm-merak": plugin_root.resolve(),
        "xhmodel-zoo": modelzoo_root.resolve(),
    }
    for package, expected in expected_sources.items():
        actual_path = _editable_source(package)
        actual = str(actual_path) if actual_path is not None else "<not an editable file install>"
        checks.append(Check(f"source:{package}", actual_path == expected, str(expected), actual))

    entry_points = {
        ("vllm.general_plugins", "merak"): "vllm_merak:register_plugins",
        ("vllm.general_plugins", "wemm"): "vllm_merak.native_wemm:register",
        ("vllm.platform_plugins", "merak"): "vllm_merak.platform:register",
    }
    for (group, name), expected in entry_points.items():
        actual = _entry_point_value(group, name)
        checks.append(
            Check(
                f"entry-point:{group}:{name}",
                actual == expected,
                expected,
                actual,
            )
        )

    try:
        module = importlib.import_module("vllm_merak")
        module_file = getattr(module, "__file__", None)
        module_path = Path(module_file).resolve() if module_file else None
    except ImportError:
        module_path = None
    expected_package = (plugin_root / "vllm_merak").resolve()
    package_is_local = module_path is not None and module_path.parent == expected_package
    checks.append(
        Check(
            "import:vllm_merak",
            package_is_local,
            str(expected_package),
            str(module_path.parent) if module_path is not None else "<not importable>",
        )
    )
    checks.append(_pip_check())
    checks.extend(_runtime_capability_checks(profile))
    return checks


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", help="runtime profile name; defaults to runtime_profiles.toml")
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--json", action="store_true", help="emit machine-readable output")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        profile_name, profile = load_profile(args.profiles.resolve(), args.profile)
    except (OSError, KeyError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"environment profile error: {exc}", file=sys.stderr)
        return 2

    checks = check_environment(profile)
    success = all(check.ok for check in checks)
    if args.json:
        print(
            json.dumps(
                {
                    "profile": profile_name,
                    "profile_status": profile.get("status"),
                    "ok": success,
                    "checks": [check.__dict__ for check in checks],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(f"runtime profile: {profile_name} ({profile.get('status', 'unknown')})")
        for check in checks:
            marker = "PASS" if check.ok else "FAIL"
            print(f"[{marker}] {check.name}: {check.actual}")
            if not check.ok:
                print(f"       expected: {check.expected}")
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
