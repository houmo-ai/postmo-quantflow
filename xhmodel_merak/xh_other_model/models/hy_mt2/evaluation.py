# Copyright 2025 HOUMO AI
#
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


LANGUAGE_NAMES = {
    "eng_Latn": {"en": "English", "zh": "英语"},
    "zho_Hans": {"en": "Chinese", "zh": "中文"},
}

REPRESENTATIVE_SOURCE_TEXT = "今天天气真好。"
REPRESENTATIVE_SOURCE_CODE = "zho_Hans"
REPRESENTATIVE_TARGET_CODE = "eng_Latn"


@dataclass(frozen=True)
class TranslationSample:
    sample_id: str
    source_code: str
    target_code: str
    source: str
    reference: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def build_translation_prompt(source_text: str, source_code: str, target_code: str) -> str:
    """Build the default prompt published with Hy-MT2."""
    if source_code not in LANGUAGE_NAMES:
        raise ValueError(f"unsupported FLORES source language: {source_code}")
    if target_code not in LANGUAGE_NAMES:
        raise ValueError(f"unsupported FLORES target language: {target_code}")
    if source_code == "zho_Hans":
        target_name = LANGUAGE_NAMES[target_code]["zh"]
        return f"将以下文本翻译为 {target_name}，注意只需要输出翻译后的结果，不要额外解释：\n\n{source_text}"
    target_name = LANGUAGE_NAMES[target_code]["en"]
    return (
        f"Translate the following text into {target_name}. Note that you should only output "
        f"the translated result without any additional explanation:\n\n{source_text}"
    )


def build_representative_prompt() -> str:
    """Return the deterministic translation prompt used for export calibration and golden."""
    return build_translation_prompt(
        REPRESENTATIVE_SOURCE_TEXT,
        REPRESENTATIVE_SOURCE_CODE,
        REPRESENTATIVE_TARGET_CODE,
    )


def build_padded_chat_input_ids(tokenizer: Any, prompt: str, sequence_length: int) -> tuple[Any, int]:
    """Tokenize a Hy-MT2 chat prompt and right-pad it to the fixed prefill shape."""
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    import torch

    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"expected input_ids shape [1, sequence], got {tuple(input_ids.shape)}")
    token_count = int(input_ids.shape[1])
    if token_count > sequence_length:
        raise ValueError(f"representative prompt has {token_count} tokens, exceeding prefill length {sequence_length}")
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer must define pad_token_id or eos_token_id")
    if token_count < sequence_length:
        padding = torch.full(
            (1, sequence_length - token_count),
            int(pad_token_id),
            dtype=input_ids.dtype,
        )
        input_ids = torch.cat((input_ids, padding), dim=1)
    return input_ids, token_count


def parse_direction(value: str) -> tuple[str, str]:
    try:
        source_code, target_code = value.split("-", maxsplit=1)
    except ValueError as exc:
        raise ValueError(f"direction must be SOURCE-TARGET, got {value!r}") from exc
    if source_code == target_code:
        raise ValueError(f"source and target languages must differ, got {value!r}")
    for code in (source_code, target_code):
        if code not in LANGUAGE_NAMES:
            raise ValueError(f"unsupported FLORES language code: {code}")
    return source_code, target_code


def load_flores_samples(
    dataset_dir: str | Path,
    directions: Sequence[str],
    *,
    split: str = "devtest",
    limit_per_direction: int | None = None,
    offset: int = 0,
) -> list[TranslationSample]:
    """Load aligned FLORES rows from the consolidated public Parquet mirror."""
    if offset < 0:
        raise ValueError("offset must be non-negative")
    if limit_per_direction is not None and limit_per_direction <= 0:
        raise ValueError("limit_per_direction must be positive")

    parquet_file = Path(dataset_dir) / f"{split}.parquet"
    if not parquet_file.is_file():
        raise FileNotFoundError(f"FLORES parquet split not found: {parquet_file}")

    parsed_directions = [parse_direction(direction) for direction in directions]
    columns = sorted({code for direction in parsed_directions for code in direction})
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_file, columns=columns)
    rows = table.to_pydict()
    row_count = table.num_rows
    stop = row_count if limit_per_direction is None else min(row_count, offset + limit_per_direction)
    if offset >= row_count:
        raise ValueError(f"offset {offset} exceeds FLORES split size {row_count}")

    samples: list[TranslationSample] = []
    for source_code, target_code in parsed_directions:
        for row_index in range(offset, stop):
            samples.append(
                TranslationSample(
                    sample_id=f"{split}:{row_index}:{source_code}-{target_code}",
                    source_code=source_code,
                    target_code=target_code,
                    source=str(rows[source_code][row_index]),
                    reference=str(rows[target_code][row_index]),
                )
            )
    return samples


def compute_translation_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    target_code: str,
) -> dict[str, Any]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    if not predictions:
        raise ValueError("at least one prediction is required")

    from sacrebleu.metrics import BLEU, CHRF

    bleu_tokenizer = "zh" if target_code == "zho_Hans" else "13a"
    references_by_stream = [list(references)]
    chrf_metric = CHRF(word_order=2)
    bleu_metric = BLEU(tokenize=bleu_tokenizer, effective_order=True)
    chrf = chrf_metric.corpus_score(list(predictions), references_by_stream)
    bleu = bleu_metric.corpus_score(
        list(predictions),
        references_by_stream,
    )
    return {
        "samples": len(predictions),
        "chrf_pp": chrf.score,
        "chrf_signature": str(chrf_metric.get_signature()),
        "bleu": bleu.score,
        "bleu_signature": str(bleu_metric.get_signature()),
    }


def summarize_predictions(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        direction = f"{record['source_code']}-{record['target_code']}"
        grouped.setdefault(direction, []).append(record)

    directions: dict[str, Any] = {}
    for direction, direction_records in grouped.items():
        _, target_code = parse_direction(direction)
        directions[direction] = compute_translation_metrics(
            [str(record["prediction"]) for record in direction_records],
            [str(record["reference"]) for record in direction_records],
            target_code=target_code,
        )
    directions["macro_average"] = {
        "chrf_pp": sum(value["chrf_pp"] for value in directions.values()) / len(grouped),
        "bleu": sum(value["bleu"] for value in directions.values()) / len(grouped),
        "directions": len(grouped),
    }
    return directions


def resolve_legacy_meta(export_path: str | Path) -> Path:
    path = Path(export_path)
    if path.is_file() and path.name == "meta.json":
        return path
    if path.is_file() and path.name == "export_meta_info.json":
        root_meta_path = path
    else:
        root_meta_path = path / "export_meta_info.json"
    if not root_meta_path.is_file():
        raise FileNotFoundError(f"Hy-MT2 export metadata not found: {root_meta_path}")
    root_meta = json.loads(root_meta_path.read_text(encoding="utf-8"))
    legacy_meta = root_meta.get("legacy_meta")
    if not legacy_meta:
        raise KeyError(f"legacy_meta is missing from {root_meta_path}")
    resolved = root_meta_path.parent / str(legacy_meta)
    if not resolved.is_file():
        raise FileNotFoundError(f"Hy-MT2 runtime metadata not found: {resolved}")
    return resolved
