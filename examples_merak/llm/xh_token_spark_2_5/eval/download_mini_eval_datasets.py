#!/usr/bin/env python3
"""Download and normalize the datasets listed in ``mini_eval_vllm_128k.yaml``.

The generated JSONL files are compatible with ``vllm_spark_tokens_eval.py`` and
are written to ``data/evaluation/spark_x2_5`` by default.  Dataset revisions are
pinned so repeated runs use the same benchmark data.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unicodedata
import zipfile
from pathlib import Path
from typing import Any, Callable

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_SUITE = SCRIPT_DIR / "mini_eval_vllm_128k.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/evaluation/spark_x2_5"

# Revisions were resolved on 2026-09-17.  Updating a revision should be an
# explicit choice because mini_eval.yaml requires the same dataset version for
# every compared model.
SPECS: dict[str, dict[str, Any]] = {
    "ifeval": {
        "repo_id": "google/IFEval",
        "revision": "966cd89545d6b6acfd7638bc708b98261ca58e84",
        "files": ("ifeval_input_data.jsonl",),
        "output": "ifeval.jsonl",
        "loader": "ifeval",
    },
    "gpqa-diamond": {
        "repo_id": "Idavidrein/gpqa",
        "revision": "633f5ee89ab8ad4522a9f850766b73f62147ffdd",
        "files": ("gpqa_diamond.csv",),
        "output": "gpqa_diamond.jsonl",
        "loader": "gpqa",
        "gated": True,
    },
    "aime-2026": {
        "repo_id": "math-ai/aime26",
        "revision": "79037aebdb6580008fb960d17cb21fd3099083e3",
        "files": ("aime2026.jsonl",),
        "output": "aime_2026.jsonl",
        "loader": "aime",
    },
    "aa-lcr": {
        "repo_id": "ArtificialAnalysis/AA-LCR",
        "revision": "9a77ef56b717057ade24ceab4d273712a0b4f19e",
        "files": (
            "AA-LCR_Dataset.csv",
            "extracted_text/AA-LCR_extracted-text.zip",
        ),
        "output": "aa_lcr.jsonl",
        "loader": "aa_lcr",
    },
    "bfcl-v4": {
        "repo_id": "ShishirPatil/gorilla",
        "revision": "6ea57973c7a6097fd7c5915698c54c17c5b1b6c8",
        "output": "bfcl_v4.jsonl",
        "loader": "bfcl_v4",
        "provider": "github",
    },
    "scicode": {
        "repo_id": "SciCode1/SciCode",
        "revision": "4510f6a6aa27c43fad7b43da2c59602a86e88480",
        "files": ("problems_test.jsonl",),
        "output": "scicode.jsonl",
        "loader": "scicode",
    },
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--benchmarks",
        nargs="*",
        help="Only download these benchmark names; accepts spaces or commas.",
    )
    parser.add_argument("--cache-dir", type=Path, help="Hugging Face/download cache directory.")
    parser.add_argument(
        "--token",
        help="Hugging Face token. Prefer HF_TOKEN or `hf auth login` to avoid shell history.",
    )
    parser.add_argument("--force", action="store_true", help="Replace existing normalized JSONL files.")
    parser.add_argument("--dry-run", action="store_true", help="Print the pinned download plan only.")
    return parser


def load_benchmark_names(suite_path: Path) -> list[str]:
    with suite_path.open(encoding="utf-8") as handle:
        suite = yaml.safe_load(handle)
    benchmarks = suite.get("benchmarks") if isinstance(suite, dict) else None
    if not isinstance(benchmarks, list):
        raise ValueError(f"{suite_path} must contain a benchmarks list")
    names = []
    for index, benchmark in enumerate(benchmarks, start=1):
        if not isinstance(benchmark, dict) or not benchmark.get("name"):
            raise ValueError(f"Benchmark entry {index} in {suite_path} needs a name")
        names.append(str(benchmark["name"]))
    return names


def select_benchmarks(suite_names: list[str], requested: list[str] | None) -> list[str]:
    if not requested:
        selected = suite_names
    else:
        wanted = {item.strip().casefold() for value in requested for item in value.split(",") if item.strip()}
        known = {name.casefold() for name in suite_names}
        unknown = sorted(wanted - known)
        if unknown:
            raise ValueError(f"Benchmarks not present in suite: {', '.join(unknown)}")
        selected = [name for name in suite_names if name.casefold() in wanted]
    unsupported = [name for name in selected if name.casefold() not in SPECS]
    if unsupported:
        raise ValueError(f"No download specification for: {', '.join(unsupported)}")
    return selected


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise TypeError(f"{path}:{line_number} is not a JSON object")
        rows.append(row)
    return rows


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".jsonl", prefix=f".{path.stem}_", dir=path.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    os.replace(temporary_path, path)


def download_hf_files(spec: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install huggingface_hub to download mini-eval datasets") from exc

    paths = []
    for filename in spec["files"]:
        try:
            downloaded = hf_hub_download(
                repo_id=spec["repo_id"],
                filename=filename,
                repo_type="dataset",
                revision=spec["revision"],
                cache_dir=str(args.cache_dir) if args.cache_dir else None,
                token=args.token,
            )
        except Exception as exc:
            if spec.get("gated"):
                raise RuntimeError(
                    "GPQA is gated. Accept its terms at "
                    "https://huggingface.co/datasets/Idavidrein/gpqa and authenticate "
                    "with `hf auth login` or HF_TOKEN."
                ) from exc
            raise
        paths.append(Path(downloaded))
    return paths


def load_ifeval(paths: list[Path]) -> list[dict[str, Any]]:
    rows = read_jsonl(paths[0])
    for index, row in enumerate(rows, start=1):
        row.setdefault("id", str(row.get("key", index)))
    return rows


def load_gpqa(paths: list[Path]) -> list[dict[str, Any]]:
    with paths[0].open(encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    for index, row in enumerate(rows, start=1):
        row.setdefault("id", str(index))
    return rows


def load_aime(paths: list[Path]) -> list[dict[str, Any]]:
    rows = read_jsonl(paths[0])
    for index, row in enumerate(rows, start=1):
        row["id"] = str(row.get("id", index))
    return rows


def load_aa_lcr(paths: list[Path]) -> list[dict[str, Any]]:
    csv_path, archive_path = paths
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        metadata_rows = [dict(row) for row in csv.DictReader(handle)]

    rows = []
    with zipfile.ZipFile(archive_path) as archive:
        members = {}
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            members[unicodedata.normalize("NFC", name).casefold()] = name
            # The ZIP creator stored some UTF-8 filenames without the Unicode
            # filename flag, so Python decodes them as CP437 (for example,
            # "ΓÇÖ" instead of a curly apostrophe).  Index the repaired form too.
            try:
                repaired_name = name.encode("cp437").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                repaired_name = name
            members[unicodedata.normalize("NFC", repaired_name).casefold()] = name
        for index, metadata in enumerate(metadata_rows, start=1):
            category = metadata["document_category"]
            document_set = metadata["document_set_id"]
            filenames = [name.strip() for name in metadata["data_source_filenames"].split(";") if name.strip()]
            documents = []
            for document_index, filename in enumerate(filenames, start=1):
                expected = f"lcr/{category}/{document_set}/{filename}"
                member = members.get(unicodedata.normalize("NFC", expected).casefold())
                if member is None:
                    raise FileNotFoundError(f"AA-LCR archive is missing {expected}")
                text = archive.read(member).decode("utf-8", errors="replace")
                documents.append(f"[Document {document_index}: {filename}]\n{text.strip()}")
            prompt = (
                "Answer the question using only the supplied documents.\n\n"
                + "\n\n".join(documents)
                + f"\n\n[Question]\n{metadata['question']}"
            )
            rows.append(
                {
                    "id": str(metadata.get("question_id") or index),
                    "prompt": prompt,
                    "answer": metadata["answer"],
                    "document_category": category,
                    "document_set_id": document_set,
                    "input_tokens": metadata.get("input_tokens"),
                    "data_source_filenames": filenames,
                    "data_source_urls": metadata.get("data_source_urls", "").split(";"),
                }
            )
    return rows


def load_scicode(paths: list[Path]) -> list[dict[str, Any]]:
    rows = read_jsonl(paths[0])
    for index, row in enumerate(rows, start=1):
        sections = [row.get("problem_description_main")]
        if row.get("problem_background_main"):
            sections.append("Background:\n" + str(row["problem_background_main"]))
        if row.get("problem_io"):
            sections.append("Required interface:\n" + str(row["problem_io"]))
        if row.get("required_dependencies"):
            sections.append("Available dependencies:\n" + str(row["required_dependencies"]))
        sections.append("Return the complete Python implementation.")
        row["id"] = str(row.get("problem_id", index))
        row["prompt"] = "\n\n".join(str(section) for section in sections if section)
    return rows


def download_github_files(spec: dict[str, Any], args: argparse.Namespace) -> list[Path]:
    cache_root = args.cache_dir or Path.home() / ".cache" / "mini_eval"
    checkout_root = cache_root / "github" / spec["repo_id"].replace("/", "__") / spec["revision"]
    data_dir = checkout_root / "berkeley-function-call-leaderboard/bfcl_eval/data"
    cached_files = sorted(
        path for path in data_dir.glob("BFCL_v4_*.json") if path.name != "BFCL_v4_format_sensitivity.json"
    )
    if cached_files:
        return cached_files
    if checkout_root.exists():
        raise RuntimeError(f"Incomplete BFCL cache at {checkout_root}; remove it and retry")
    if shutil.which("git") is None:
        raise RuntimeError("git is required to download the official BFCL-V4 dataset")

    checkout_root.parent.mkdir(parents=True, exist_ok=True)
    repository_url = f"https://github.com/{spec['repo_id']}.git"
    sparse_path = "berkeley-function-call-leaderboard/bfcl_eval/data"
    with tempfile.TemporaryDirectory(prefix=".bfcl_", dir=checkout_root.parent) as temporary_dir:
        repository = Path(temporary_dir) / "repository"
        commands = (
            ("init", str(repository)),
            ("-C", str(repository), "remote", "add", "origin", repository_url),
            ("-C", str(repository), "sparse-checkout", "init", "--cone"),
            ("-C", str(repository), "sparse-checkout", "set", sparse_path),
            (
                "-C",
                str(repository),
                "fetch",
                "--depth",
                "1",
                "--filter=blob:none",
                "origin",
                spec["revision"],
            ),
            ("-C", str(repository), "checkout", "--detach", "FETCH_HEAD"),
        )
        for command in commands:
            subprocess.run(["git", *command], check=True)
        os.replace(repository, checkout_root)

    files = sorted(path for path in data_dir.glob("BFCL_v4_*.json") if path.name != "BFCL_v4_format_sensitivity.json")
    if not files:
        raise FileNotFoundError("No BFCL_v4_*.json files found in the pinned Gorilla checkout")
    return files


def _read_json_or_jsonl_bytes(payload: bytes, source: str) -> list[dict[str, Any]]:
    text = payload.decode("utf-8")
    try:
        parsed = json.loads(text)
        values = parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        values = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not all(isinstance(value, dict) for value in values):
        raise TypeError(f"{source} does not contain JSON objects")
    return values


def _flatten_bfcl_messages(question: Any) -> list[dict[str, Any]] | None:
    if not isinstance(question, list):
        return None
    if all(isinstance(message, dict) for message in question):
        return question
    messages = []
    for turn in question:
        if isinstance(turn, list):
            messages.extend(message for message in turn if isinstance(message, dict))
    return messages or None


def load_bfcl_v4(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(paths):
        for record in _read_json_or_jsonl_bytes(path.read_bytes(), str(path)):
            record = dict(record)
            record["source_file"] = path.name
            record["id"] = str(record.get("id", f"bfcl-{len(rows) + 1}"))
            messages = _flatten_bfcl_messages(record.get("question"))
            if messages:
                record["messages"] = messages
            if record.get("function") and not record.get("tools"):
                record["tools"] = record["function"]
            rows.append(record)
    return rows


LOADERS: dict[str, Callable[[list[Path]], list[dict[str, Any]]]] = {
    "ifeval": load_ifeval,
    "gpqa": load_gpqa,
    "aime": load_aime,
    "aa_lcr": load_aa_lcr,
    "bfcl_v4": load_bfcl_v4,
    "scicode": load_scicode,
}


def print_plan(name: str, spec: dict[str, Any], output_path: Path) -> None:
    provider = spec.get("provider", "huggingface")
    print(f"[plan] {name}: {provider}:{spec['repo_id']}@{spec['revision']} -> {output_path}")


def download_benchmark(name: str, spec: dict[str, Any], args: argparse.Namespace) -> tuple[Path, int]:
    output_path = args.output_dir / spec["output"]
    if output_path.is_file() and not args.force:
        count = sum(1 for line in output_path.open(encoding="utf-8") if line.strip())
        print(f"[skip] {name}: {output_path} already exists ({count} records); use --force to replace")
        return output_path, count

    if spec.get("provider") == "github":
        source_paths = download_github_files(spec, args)
    else:
        source_paths = download_hf_files(spec, args)
    rows = LOADERS[spec["loader"]](source_paths)
    if not rows:
        raise ValueError(f"{name} produced no records")
    write_jsonl_atomic(output_path, rows)
    print(f"[done] {name}: {len(rows)} records -> {output_path}")
    return output_path, len(rows)


def main() -> int:
    args = build_parser().parse_args()
    args.suite = args.suite.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    if args.cache_dir:
        args.cache_dir = args.cache_dir.expanduser().resolve()

    suite_names = load_benchmark_names(args.suite)
    selected = select_benchmarks(suite_names, args.benchmarks)
    for name in selected:
        print_plan(name, SPECS[name.casefold()], args.output_dir / SPECS[name.casefold()]["output"])
    if args.dry_run:
        return 0

    failures = []
    manifest = {}
    for name in selected:
        spec = SPECS[name.casefold()]
        try:
            output_path, count = download_benchmark(name, spec, args)
            manifest[name] = {
                "path": str(output_path),
                "records": count,
                "source": spec["repo_id"],
                "revision": spec["revision"],
            }
        except Exception as exc:
            failures.append(name)
            print(f"[error] {name}: {type(exc).__name__}: {exc}", file=sys.stderr)

    if manifest:
        manifest_path = args.output_dir / "download_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"[manifest] {manifest_path}")
    if failures:
        print(f"[summary] failed: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"[summary] downloaded {len(manifest)} benchmark datasets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
