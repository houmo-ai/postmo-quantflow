"""Aggregate official MMEB-v3 score files using the WeMM report protocol."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


EXCLUDED_VISDOC_DIAGNOSTICS = {
    "ViDoSeek-doc",
    "MMLongBench-doc",
    "MMLongBench-page",
}
OFFICIAL_REFERENCE = {
    "V3-All": 55.8,
    "MMEB-v2": 77.9,
    "Image": 79.6,
    "Video": 70.8,
    "VisDoc": 80.7,
    "Text": 45.3,
    "Reasoning Retrieval": 26.3,
    "Instruction Following": 49.3,
    "Long Context": 61.9,
    "Multi Condition": 59.7,
    "General Retrieval": 58.7,
    "Agent": 44.1,
    "Tool": 46.2,
    "GUI": 38.9,
    # Corrected on 2026-09-16 after PeerQA and DeepPlanning switched from
    # per-query local pools to the MMEB-v3 task-wide global retrieval scope.
    # The four official 2B Hit@1 scores average to 35.3754.
    "Memory": 35.4,
    "MCMR": 42.5,
    "Audio": 0.0,
}


def _primary_metric(modality: str, dataset: str) -> str:
    if modality in {"text", "visdoc"}:
        return "ndcg_linear@5"
    return "hit@1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _protocol_errors(
    manifest: dict[str, object],
    job: dict[str, object],
    protocol: dict[str, object],
    adapter_sha256: str,
) -> list[str]:
    errors = []
    expected_backend = manifest.get("backend")
    if protocol.get("backend") != expected_backend:
        errors.append("backend")
    if protocol.get("official_repo_commit") != manifest.get("official_commit"):
        errors.append("official_repo_commit")
    if protocol.get("mmeb_v3_data_revision") != manifest.get("mmeb_v3_data_revision"):
        errors.append("mmeb_v3_data_revision")
    if protocol.get("mmeb_v2_media_revision") != manifest.get("mmeb_v2_data_revision"):
        errors.append("mmeb_v2_media_revision")
    if protocol.get("wemm_vllm_adapter_sha256") != adapter_sha256:
        errors.append("wemm_vllm_adapter_sha256")
    if protocol.get("vllm_version") != manifest.get("vllm_version"):
        errors.append("vllm_version")
    if manifest.get("transformers_version") is not None and protocol.get("transformers_version") != manifest.get(
        "transformers_version"
    ):
        errors.append("transformers_version")

    config_path = Path(str(job["config"]))
    if not config_path.is_file() or protocol.get("dataset_config_sha256") != _sha256(config_path):
        errors.append("dataset_config_sha256")

    if expected_backend == "vllm":
        if protocol.get("vllm_wemm_model_sha256") != manifest.get("vllm_wemm_model_sha256"):
            errors.append("vllm_wemm_model_sha256")
        model_files = manifest.get("model_files") or {}
        if protocol.get("hf_config_sha256") != model_files.get("config.json"):
            errors.append("hf_config_sha256")
    elif expected_backend == "vllm_merak":
        if protocol.get("vllm_merak_source_sha256") != manifest.get("vllm_merak_source_sha256"):
            errors.append("vllm_merak_source_sha256")
        model_files = manifest.get("model_files") or {}
        if protocol.get("hmonnx_meta_sha256") != model_files.get("golden_meta_info.json"):
            errors.append("hmonnx_meta_sha256")
        expected_source_config = manifest.get("source_hf_config_sha256")
        if expected_source_config is not None and protocol.get("hf_config_sha256") != expected_source_config:
            errors.append("hf_config_sha256")

    vllm = protocol.get("vllm") or {}
    image = protocol.get("image_preprocessing") or {}
    video = protocol.get("video_preprocessing") or {}
    if vllm.get("max_model_len") != manifest.get("max_model_len"):
        errors.append("max_model_len")
    if vllm.get("max_num_seqs") != manifest.get("max_num_seqs"):
        errors.append("max_num_seqs")
    if image.get("mode") != "dynamic" or image.get("min_pixels") != 65536 or image.get("max_pixels") != 1572864:
        errors.append("image_preprocessing")
    if (
        video.get("num_frames") != manifest.get("video_num_frames")
        or video.get("frame_size") != 224
        or video.get("min_pixels") != 4 * 32 * 32
        or video.get("max_pixels") != 256 * 32 * 32
        or video.get("total_pixels") != 8192 * 32 * 32
        or video.get("sample_fps") != 1
    ):
        errors.append("video_preprocessing")
    return errors


def _mean(rows: list[dict[str, object]]) -> float | None:
    values = [float(row["score_percent"]) for row in rows if row.get("score_percent") is not None]
    return sum(values) / len(values) if len(values) == len(rows) and values else None


def _text_group(dataset: str) -> str:
    if dataset.startswith("text/BRIGHT/") or dataset.startswith("text/R2MED/"):
        return "Reasoning Retrieval"
    if dataset.startswith("text/FollowIR/") or dataset.startswith("text/InfoSearch/"):
        return "Instruction Following"
    if dataset.startswith("text/LongEmbed/"):
        return "Long Context"
    if dataset.startswith("text/MultiConIR/"):
        return "Multi Condition"
    if dataset.startswith("text/NanoBEIR/"):
        return "General Retrieval"
    raise ValueError(f"Unknown text subgroup for {dataset}")


def _format(value: float | None) -> str:
    return "pending" if value is None else f"{value:.2f}"


def _score_column_label(backend: str) -> str:
    return {
        "vllm": "Float vLLM",
        "vllm_merak": "Quantized vLLM Merak",
    }.get(backend, "Score")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--title", default="WEMM-Embedding MMEB-v3 Report")
    parser.add_argument("--no-official-reference", action="store_true")
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    adapter_sha256 = _sha256(Path(__file__).with_name("mmeb_v3_vllm_merak.py"))

    rows = []
    for job in manifest["jobs"]:
        modality = job["modality"]
        dataset = job["dataset_name"]
        output = Path(job["output"])
        score_path = output / f"{dataset}_score.json"
        protocol_path = output / "protocol.json"
        metric = _primary_metric(modality, dataset)
        score = None
        protocol_errors = ["protocol.json missing"]
        if protocol_path.is_file():
            try:
                protocol = json.loads(protocol_path.read_text())
                protocol_errors = _protocol_errors(manifest, job, protocol, adapter_sha256)
            except (OSError, TypeError, ValueError, json.JSONDecodeError):
                protocol_errors = ["protocol.json invalid"]
        if not protocol_errors and score_path.is_file():
            payload = json.loads(score_path.read_text())
            if metric in payload:
                score = float(payload[metric]) * 100.0
        rows.append(
            {
                "modality": modality,
                "dataset": dataset,
                "metric": metric,
                "score_percent": score,
                "score_file": str(score_path),
                "protocol_file": str(protocol_path),
                "protocol_valid": not protocol_errors,
                "protocol_errors": ",".join(protocol_errors),
                "aggregate_included": not (modality == "visdoc" and dataset in EXCLUDED_VISDOC_DIAGNOSTICS),
            }
        )

    by_modality: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_modality[str(row["modality"])].append(row)
    mcmr = [row for row in by_modality["image"] if row["dataset"] == "MCMR"]
    image = [row for row in by_modality["image"] if row["dataset"] != "MCMR"]
    video = by_modality["video"]
    visdoc = [row for row in by_modality["visdoc"] if row["aggregate_included"]]
    mmeb_v2 = image + video + visdoc
    text = by_modality["text"]
    agent = by_modality["tool"] + by_modality["gui"] + by_modality["memory"]

    text_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in text:
        text_groups[_text_group(str(row["dataset"]))].append(row)

    aggregate_rows = {
        "MMEB-v2": mmeb_v2,
        "Image": image,
        "Video": video,
        "VisDoc": visdoc,
        "Text": text,
        **text_groups,
        "Agent": agent,
        "Tool": by_modality["tool"],
        "GUI": by_modality["gui"],
        "Memory": by_modality["memory"],
        "MCMR": mcmr,
    }
    aggregates = {name: _mean(group) for name, group in aggregate_rows.items()}
    aggregates["Audio"] = 0.0
    v3_rows = mmeb_v2 + text + agent + mcmr
    if len(v3_rows) != 179:
        raise ValueError(f"Expected 179 non-audio aggregate tasks, got {len(v3_rows)}")
    supported_scores = [row["score_percent"] for row in v3_rows]
    if all(value is not None for value in supported_scores):
        aggregates["V3-All"] = sum(float(value) for value in supported_scores) / 190.0
        aggregates["Supported-Only"] = sum(float(value) for value in supported_scores) / 179.0
    else:
        aggregates["V3-All"] = None
        aggregates["Supported-Only"] = None

    use_official_reference = not args.no_official_reference
    summary = {
        "protocol": {
            "total_reported_tasks": 190,
            "executed_aggregate_tasks": 179,
            "audio_zero_tasks": 11,
            "executed_diagnostic_tasks": 3,
            "image_video_agent_metric": "Hit@1",
            "text_visdoc_metric": "NDCG@5",
        },
        "completed_score_files": sum(row["score_percent"] is not None for row in rows),
        "expected_score_files": len(rows),
        "missing": [row["dataset"] for row in rows if row["score_percent"] is None],
        "invalid_protocols": [
            {
                "dataset": row["dataset"],
                "protocol_file": row["protocol_file"],
                "errors": row["protocol_errors"].split(","),
            }
            for row in rows
            if not row["protocol_valid"]
        ],
        "aggregates": {
            name: {
                "score": value,
                "official_wemm_2b": (OFFICIAL_REFERENCE.get(name) if use_official_reference else None),
                "delta": (
                    None
                    if not use_official_reference or value is None or name not in OFFICIAL_REFERENCE
                    else value - OFFICIAL_REFERENCE[name]
                ),
                "num_tasks": (
                    len(aggregate_rows[name])
                    if name in aggregate_rows
                    else {"V3-All": 190, "Supported-Only": 179, "Audio": 11}[name]
                ),
            }
            for name, value in aggregates.items()
        },
        "tasks": rows,
    }
    aggregate_dir = run_dir / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    (aggregate_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    with (aggregate_dir / "per_task.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    order = [
        "V3-All",
        "Supported-Only",
        "MMEB-v2",
        "Image",
        "Video",
        "VisDoc",
        "Text",
        "Reasoning Retrieval",
        "Instruction Following",
        "Long Context",
        "Multi Condition",
        "General Retrieval",
        "Agent",
        "Tool",
        "GUI",
        "Memory",
        "MCMR",
        "Audio",
    ]
    markdown = [
        f"# {args.title}",
        "",
        f"Completed datasets: {summary['completed_score_files']}/{summary['expected_score_files']}",
        "",
        f"| Group | Tasks | {_score_column_label(str(manifest.get('backend', '')))} | Official WeMM-2B | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in order:
        item = summary["aggregates"][name]
        reference = item["official_wemm_2b"]
        markdown.append(
            f"| {name} | {item['num_tasks']} | {_format(item['score'])} | "
            f"{_format(reference)} | {_format(item['delta'])} |"
        )
    if summary["missing"]:
        markdown.extend(("", "## Pending datasets", ""))
        markdown.extend(f"- {name}" for name in summary["missing"])
    (aggregate_dir / "REPORT.md").write_text("\n".join(markdown) + "\n")
    print(json.dumps(summary["aggregates"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
