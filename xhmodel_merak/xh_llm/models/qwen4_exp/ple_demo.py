"""Standalone NumPy PLE host lookup; copy this file next to a Golden package.

Run ``python ple_demo.py`` to replay the first manifest sample, or supply
``--indices ids.npy --layer 1 --output embeddings.npy --expected golden.npy``.
Only selected compressed rows are read; the full table is never unpacked.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _confined_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"PLE resource escapes package root: {relative}")
    return path


def load_tables(root, spec):
    """Open size-checked raw packed and exponent files without loading them."""
    tables = []
    groups = (int(spec["embedding_dim"]) + 63) // 64
    for key, dtype, tail in (("packed", "int8", (groups, 32)), ("exponent", "int16", (groups, 1))):
        resource = spec[key]
        shape = tuple(int(value) for value in resource["shape"])
        if resource["dtype"] != dtype or len(shape) != 3 or shape[0] < 1 or shape[1:] != tail:
            raise ValueError(f"Invalid PLE {key} storage layout")
        path = _confined_path(root, resource["path"])
        storage_dtype = np.dtype(dtype).newbyteorder("<")
        expected_size = int(np.prod(shape, dtype=object)) * storage_dtype.itemsize
        if path.stat().st_size != expected_size:
            raise ValueError(f"PLE {key} file size differs from declared shape: {path}")
        tables.append(np.memmap(path, dtype=storage_dtype, mode="r", shape=shape))
    if tables[0].shape[0] != tables[1].shape[0]:
        raise ValueError("PLE packed and exponent row counts differ")
    return tuple(tables)


def gather_ple_rows(packed, exponent, indices):
    """Copy only the requested compressed rows, using ONNX negative-id rules."""
    ids = np.asarray(indices)
    if ids.dtype.kind not in "iu":
        raise TypeError("PLE indices must be integers")
    if packed.dtype != np.int8 or packed.ndim != 3 or packed.shape[-1] != 32:
        raise ValueError("PLE packed layout must be int8 [rows,groups,32]")
    if exponent.dtype != np.int16 or exponent.shape != (*packed.shape[:2], 1):
        raise ValueError("PLE exponent layout must be int16 [rows,groups,1]")
    if ids.size and (int(ids.min()) < -packed.shape[0] or int(ids.max()) >= packed.shape[0]):
        raise IndexError("PLE indices out of bounds")
    return packed[ids], exponent[ids]


def dequantize_ple_rows(selected, selected_exp, embedding_dim, output_dtype="float16"):
    """Unpack SDK low-even/high-odd signed nibbles, then apply SEFP exponents.

    BF16 output uses float32 storage containing BF16-rounded values; the caller
    can cast it to its framework's bfloat16 type without losing exponent range.
    """
    selected = np.asarray(selected)
    selected_exp = np.asarray(selected_exp)
    groups = (int(embedding_dim) + 63) // 64
    if embedding_dim < 1 or selected.dtype != np.int8 or selected.shape[-2:] != (groups, 32):
        raise ValueError("Invalid selected PLE packed layout")
    if selected_exp.dtype != np.int16 or selected_exp.shape != (*selected.shape[:-1], 1):
        raise ValueError("Invalid selected PLE exponent layout")
    if output_dtype not in ("float16", "bfloat16"):
        raise ValueError("PLE output_dtype must be float16 or bfloat16")
    raw = selected.view(np.uint8)
    # SDK pack_signed_w4 pairs adjacent K elements, not adjacent groups.
    nibble = np.stack((raw & 15, raw >> 4), axis=-1).reshape(*selected.shape[:-1], 64)
    mantissa = np.where(nibble < 8, nibble.astype(np.int16), nibble.astype(np.int16) - 16)
    dtype = np.float16 if output_dtype == "float16" else np.float32
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scale = np.power(dtype(2), selected_exp.astype(dtype))
        values = mantissa.astype(dtype) * scale
    values = values.reshape(*selected.shape[:-2], groups * 64)[..., :embedding_dim]
    if output_dtype == "bfloat16":
        # Round-to-nearest-even into BF16 while keeping a standard NumPy dtype.
        bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
        bits = (bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)
        values = bits.view(np.float32)
    return values


def compute_ngram_ids(token_ids, history, spec, current_input_length):
    """Native signed-int64 multiply/XOR/modulo hash with EOS segment boundaries.

    Invalid padded tokens are EOS. History stays unchanged: the model graph
    remains responsible for updating its cache from the valid token prefix.
    """
    ids = np.asarray(token_ids)
    previous = np.asarray(history)
    ngram_size = int(spec["ngram_size"])
    heads = int(spec["heads_per_ngram"])
    eos = int(spec["eos_token_id"])
    if ids.ndim != 2 or ids.dtype.kind not in "iu":
        raise ValueError("token_ids must be an integer [batch,sequence] matrix")
    if previous.shape != (ids.shape[0], ngram_size - 1) or previous.dtype.kind not in "iu":
        raise ValueError("history must be integer [batch,ngram_size-1]")
    length_values = np.asarray(current_input_length).reshape(-1)
    if length_values.size != 1:
        raise ValueError("current_input_length must be scalar")
    length = int(length_values[0])
    if not 0 <= length <= ids.shape[1]:
        raise ValueError("current_input_length is outside padded sequence")
    multipliers = [int(item) for item in spec["multipliers"]]
    vocab = [int(item) for item in spec["vocab_sizes"]]
    offsets = [int(item) for item in spec["offsets"]]
    if ngram_size < 2 or heads < 1 or len(multipliers) != ngram_size:
        raise ValueError("Invalid PLE ngram dimensions or multipliers")
    if len(vocab) != (ngram_size - 1) * heads or len(offsets) != len(vocab) or min(vocab) < 1:
        raise ValueError("Invalid PLE head vocab_sizes or offsets")
    current = ids.astype(np.int64, copy=True)
    current[:, length:] = eos
    full = np.concatenate((previous.astype(np.int64), current), axis=1)
    result = np.empty((*ids.shape, len(vocab)), dtype=np.int64)
    mask = (1 << 64) - 1
    for batch, row in enumerate(full):
        segment_start = 0
        for position, token in enumerate(row):
            shifted = [int(token)]
            for shift in range(1, ngram_size):
                shifted.append(int(row[position - shift]) if position - shift >= segment_start else eos)
            if position >= ngram_size - 1:
                mixed = (shifted[0] * multipliers[0]) & mask
                for ngram in range(2, ngram_size + 1):
                    mixed ^= (shifted[ngram - 1] * multipliers[ngram - 1]) & mask
                    signed = mixed - (1 << 64) if mixed >= 1 << 63 else mixed
                    for head in range(heads):
                        index = (ngram - 2) * heads + head
                        result[batch, position - (ngram_size - 1), index] = signed % vocab[index] + offsets[index]
            if int(token) == eos:
                segment_start = position + 1
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parent / "ple/manifest.json")
    parser.add_argument("--indices", type=Path)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected", type=Path)
    args = parser.parse_args(argv)
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("version") != 1:
        raise ValueError("Unsupported PLE manifest version")
    # Every manifest path is relative to its containing ple/ directory.
    root = manifest_path.parent
    samples = manifest.get("samples", [])
    candidates = [item for item in samples if args.layer is None or item.get("layer_idx") == args.layer]
    if args.sample < 0 or (candidates and args.sample >= len(candidates)):
        raise ValueError("Requested PLE sample is absent")
    sample = candidates[args.sample] if candidates else {}
    layer = args.layer if args.layer is not None else sample.get("layer_idx")
    spec = next((item for item in manifest["layers"] if layer is None or item["layer_idx"] == layer), None)
    if spec is None:
        raise ValueError("Requested PLE layer is absent")
    if args.indices is not None:
        indices = np.load(args.indices, allow_pickle=False)
    elif "indices" in sample:
        indices = np.load(_confined_path(root, sample["indices"]), allow_pickle=False)
    elif "token_ids" in sample and "history" in sample:
        indices = compute_ngram_ids(
            np.load(_confined_path(root, sample["token_ids"]), allow_pickle=False),
            np.load(_confined_path(root, sample["history"]), allow_pickle=False),
            spec, sample["current_input_length"],
        )
    else:
        parser.error("Provide --indices or register indices/token_ids+history in manifest.samples")
    packed, exponent = load_tables(root, spec)
    selected, selected_exp = gather_ple_rows(packed, exponent, indices)
    output = dequantize_ple_rows(selected, selected_exp, spec["embedding_dim"], spec["output_dtype"])
    expected_path = args.expected or (
        _confined_path(root, sample["expected"]) if args.indices is None and "expected" in sample else None
    )
    if expected_path is not None:
        np.testing.assert_array_equal(output, np.load(expected_path, allow_pickle=False))
    if args.output is not None:
        np.save(args.output, output, allow_pickle=False)
    print(json.dumps({"layer": spec["layer_idx"], "shape": list(output.shape), "dtype": str(output.dtype),
                      "matched_expected": expected_path is not None}))


if __name__ == "__main__":
    main()
