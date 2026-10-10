"""Spark X2.5 recipe owned by ModelZoo, using GPTQModel's bundled AutoRound."""

from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace


def autoround_root() -> Path:
    spec = importlib.util.find_spec("gptqmodel")
    if spec is None or spec.origin is None:
        raise ImportError("Spark AutoRound requires GPTQModel with bundled third_party/auto-round")
    root = Path(spec.origin).resolve().parents[1] / "third_party" / "auto-round"
    if not (root / "auto_round" / "__init__.py").is_file():
        raise ImportError(f"GPTQModel bundled AutoRound is missing: {root}")
    return root


def require_bundled_autoround():
    root = autoround_root()
    sys.path.insert(0, str(root))
    spec = importlib.util.find_spec("auto_round")
    if spec is None or not spec.origin or not Path(spec.origin).resolve().is_relative_to(root.resolve()):
        raise ImportError("A different auto_round is already imported; run Spark quantization in a fresh worker")
    import auto_round

    return auto_round


def prepare_spark_calibration(tokenizer, dataset, *, seqlen, nsamples, seed, concat=True):
    """Use GPTQModel packing, retaining special tokens and full-length samples."""
    from gptqmodel.utils.calibration import prepare_calibration_dataset

    with Path(dataset).open() as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    texts = [record["text"] for record in records]
    if any(not isinstance(text, str) for text in texts):
        raise ValueError("Spark calibration text must be a string")
    random.Random(seed).shuffle(texts)
    batches = prepare_calibration_dataset(
        SimpleNamespace(tokenizer=tokenizer, support_batch_quantize=True),
        texts,
        calibration_dataset_concat_size=seqlen if concat else None,
        batch_size=1,
        calibration_data_min_length=0,
    )
    samples = []
    for batch in batches:
        if batch["input_ids"].shape[-1] >= seqlen and batch["attention_mask"][..., :seqlen].all():
            samples.append({key: value[..., :seqlen] for key, value in batch.items()})
        if len(samples) == nsamples:
            return samples
    raise ValueError(f"Only {len(samples)} full-length Spark calibration samples; need {nsamples}")


def quantize_spark25_autoround(
    *,
    model_dir,
    output_dir,
    dataset,
    layer_config,
    bits=4,
    group_size=64,
    data_type="int_sym_gptq",
    iters=200,
    nsamples=128,
    seqlen=2048,
    batch_size=8,
    gradient_accumulate_steps=1,
    low_gpu_mem_usage=True,
    device_map="0",
    seed=42,
    trust_remote_code=False,
    format="auto_gptq",
    concat=True,
):
    backend = require_bundled_autoround()
    from transformers import AutoTokenizer

    output = Path(output_dir)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Quantization output is not empty: {output}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    dataset = prepare_spark_calibration(tokenizer, dataset, seqlen=seqlen, nsamples=nsamples, seed=seed, concat=concat)
    print(f"[Spark AutoRound] backend: {backend.__file__}", flush=True)
    quantizer = backend.AutoRound(
        model=str(model_dir),
        tokenizer=tokenizer,
        dataset=dataset,
        scheme=f"W{bits}A16",
        bits=bits,
        group_size=group_size,
        sym=True,
        data_type=data_type,
        layer_config=layer_config,
        iters=iters,
        nsamples=nsamples,
        seqlen=seqlen,
        batch_size=batch_size,
        gradient_accumulate_steps=gradient_accumulate_steps,
        low_gpu_mem_usage=low_gpu_mem_usage,
        device_map=device_map,
        seed=seed,
        trust_remote_code=trust_remote_code,
    )
    quantizer.quantize_and_save(str(output), format=format)
    provenance = {
        "recipe": "xhmodel_merak.xh_llm.models.xh_token_spark_2_5.autoround_recipe.quantize_spark25_autoround",
        "backend": str(Path(backend.__file__).resolve()),
        "method": "autoround",
    }
    (output / "spark25_quant_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return str(output.resolve())
