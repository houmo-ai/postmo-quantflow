# SPDX-License-Identifier: Apache-2.0
"""Spark2_5 dense AutoRound quantization (GPTQModel).

The mixed-precision profile is derived from the checkpoint's ``config.json``:
W4 is used for the bulk of the MLP. Fused attention projections and the
first/last two MLP blocks skip quantization. Embeddings, the tied LM head,
RMSNorm layers, and ``g_proj`` also remain in floating point.
"""

import argparse
import copy
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import torch
from gptqmodel import GPTQModel
from transformers import AutoTokenizer

# Register Spark2_5 in GPTQModel before GPTQModel.load
from xhmodel_merak.xh_llm.models.xh_token_spark_2_5 import gptqmodel_wrap  # noqa: F401


def read_model_config(model_dir: str) -> dict:
    """Read the model configuration used to derive mixed-precision ranges."""
    config_path = Path(model_dir) / "config.json"
    try:
        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read model config: {config_path}") from exc
    if not isinstance(config, dict) or not config.get("num_hidden_layers"):
        raise ValueError(f"config.json must contain num_hidden_layers: {config_path}")
    if config.get("model_type") != "spark2_5":
        raise ValueError(
            f"Expected Spark checkpoint model_type='spark2_5', got {config.get('model_type')!r}: {config_path}"
        )
    return config


def build_autoround_command(args: argparse.Namespace, dataset: str | None = None) -> list[str]:
    if not args.sym:
        raise ValueError("Spark AutoRound only supports symmetric quantization; --no-sym is unsupported")
    if dataset is None:
        dataset = args.calibration_jsonl or args.dataset
        if args.concat_calibration and Path(dataset).is_file():
            dataset += ":concat"
    device_map = args.device_map or args.device
    # AutoRound treats strings containing ':' as regex-to-device mappings
    # (for example "model.layers:0"). For a single visible GPU it expects
    # the logical device index, not the torch spelling "cuda:0".
    match = re.fullmatch(r"cuda:(\d+)", device_map)
    if match:
        device_map = match.group(1)

    command = [
        sys.executable,
        "-m",
        "xhmodel_merak.xh_llm.models.xh_token_spark_2_5.autoround_cli",
        args.model,
        "--output_dir",
        args.out,
        "--scheme",
        f"W{args.bits}A16",
        "--bits",
        str(args.bits),
        "--group_size",
        str(args.group_size),
        "--data_type",
        args.data_type,
        "--iters",
        str(args.iters),
        "--nsamples",
        str(args.nsamples),
        "--seqlen",
        str(args.seqlen),
        "--batch_size",
        str(args.batch_size),
        "--gradient_accumulate_steps",
        str(args.gradient_accumulate_steps),
        "--dataset",
        dataset,
        "--device_map",
        device_map,
        "--format",
        args.format,
        "--seed",
        str(args.seed),
    ]
    # AutoRound's layer_config accepts regex keys.  The global scheme remains
    # W4A16 by default; these entries skip quantization for sensitive Linear layers.
    # RMSNorm is intentionally absent: it is not a supported quantization
    # target in the Spark2_5 module tree and therefore stays BF16.
    command.extend(["--layer_config", build_mixed_layer_config_arg(args)])
    if args.low_gpu_mem_usage:
        command.append("--low_gpu_mem_usage")
    if not args.trust_remote_code:
        command.append("--disable_trust_remote_code")
    return command


def build_mixed_layer_config_arg(args: argparse.Namespace) -> str:
    """Build AutoRound's compact mixed-precision config for Spark2_5 models.

    The number of decoder layers is read from ``config.json`` so the
    first/last two MLP blocks skip quantization for any compatible Spark2_5
    checkpoint.
    RMSNorm is intentionally absent: it is not a supported quantization target
    in the Spark2_5 module tree and therefore stays BF16.
    """
    num_hidden_layers = int(read_model_config(args.model)["num_hidden_layers"])
    edge_count = min(2, num_hidden_layers // 2)
    edge_layers = tuple(sorted(set(range(edge_count)) | set(range(num_hidden_layers - edge_count, num_hidden_layers))))
    edge_pattern = "|".join(str(index) for index in edge_layers)
    entries = [
        r"model.layers.[0-9]+.self_attn.(q_k_v_proj|out_proj):{bits:16}",
    ]
    if edge_pattern:
        entries.append(rf"model.layers.({edge_pattern}).mlp.(gate_proj|up_proj|down_proj):{{bits:16}}")
    return "{" + ",".join(entries) + "}"


def default_calibration_dataset(model_dir: str) -> str:
    """Select a bundled calibration set from the model configuration."""
    config = read_model_config(model_dir)
    # The bundled datasets are architecture-compatible; select by the
    # configured hidden size rather than by a model directory/model_name.
    if int(config.get("hidden_size", 0)) <= 2048:
        return str(Path(__file__).with_name("calib_data") / "spark_tokens_1_7b_generated.jsonl")
    return str(Path(__file__).with_name("calib_data") / "spark_tokens_4b_generated.jsonl")


def pack_calibration_jsonl(args: argparse.Namespace) -> str | None:
    """Pack short JSONL records into samples at least as long as ``seqlen``."""
    if not args.calibration_jsonl or not args.concat_calibration:
        return None

    from xhmodel_merak.xh_llm.models.xh_token_spark_2_5.spark_2_5 import register_spark2_5

    register_spark2_5()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True, trust_remote_code=args.trust_remote_code)

    texts = []
    with open(args.calibration_jsonl, encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            text = record if isinstance(record, str) else record.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Calibration record {line_number} must contain a string 'text' field")
            texts.append(text)

    output_parent = Path(args.out).parent
    output_parent.mkdir(parents=True, exist_ok=True)
    packed_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".jsonl",
        prefix="spark2_5_calibration_",
        dir=output_parent,
        delete=False,
    )
    packed_path = packed_file.name
    packed_count = 0
    buffer = ""
    try:
        with packed_file:
            for text in texts:
                candidate = f"{buffer}\n{text}" if buffer else text
                token_count = len(tokenizer(candidate, add_special_tokens=True)["input_ids"])
                if token_count >= args.seqlen:
                    packed_file.write(json.dumps({"text": candidate}, ensure_ascii=False) + "\n")
                    packed_count += 1
                    buffer = ""
                else:
                    buffer = candidate

        if packed_count == 0:
            raise ValueError(f"Calibration data has too few tokens to form one {args.seqlen}-token sample")
        print(
            f"[calibration] Packed {len(texts)} records into {packed_count} samples with at least {args.seqlen} tokens"
        )
        if packed_count < args.nsamples:
            print(
                f"[calibration] Warning: requested {args.nsamples} samples, "
                f"but only {packed_count} packed samples are available"
            )
        return packed_path
    except Exception:
        Path(packed_path).unlink(missing_ok=True)
        raise


@contextmanager
def compatible_autoround_model(model_dir: str):
    """Stage current Spark code and config without changing shared weights."""
    source = Path(model_dir).absolute()
    config = read_model_config(str(source))
    with tempfile.TemporaryDirectory(prefix="spark_autoround_") as temporary:
        staged = Path(temporary) / source.name
        staged.mkdir()
        replaced = {"config.json", "configuration_spark.py", "modeling_spark.py", "__init__.py"}
        for entry in source.iterdir():
            if entry.name not in replaced:
                (staged / entry.name).symlink_to(entry.resolve(), target_is_directory=entry.is_dir())
        config.update(
            architectures=["Spark2_5ForCausalLM"],
            auto_map={
                "AutoConfig": "configuration_spark.Spark2_5Config",
                "AutoModel": "modeling_spark.Spark2_5Model",
                "AutoModelForCausalLM": "modeling_spark.Spark2_5ForCausalLM",
            },
        )
        (staged / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        sync_spark2_5_remote_code_to_checkpoint(str(staged))
        yield str(staged)


def run_autoround(args: argparse.Namespace) -> None:
    if args.dry_run:
        command = build_autoround_command(args)
        print("[AutoRound]", " ".join(command))
        return

    with compatible_autoround_model(args.model) as model_dir:
        runtime_args = copy.copy(args)
        runtime_args.model = model_dir
        packed_dataset = pack_calibration_jsonl(runtime_args)
        try:
            command = build_autoround_command(runtime_args, dataset=packed_dataset)
            print("[AutoRound]", " ".join(command), flush=True)
            subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[3])
        finally:
            if packed_dataset:
                Path(packed_dataset).unlink(missing_ok=True)


def build_default_output_dir(args):
    model_name = os.path.basename(os.path.normpath(args.model))
    time_suffix = datetime.now().strftime("%m%d%H")
    return os.path.join(
        "work_dirs",
        f"{model_name}-autoround-mixed-w{args.bits}fp16-{args.group_size}g-{args.format}_{time_suffix}",
    )


def resolve_quantized_output_dir(out_dir: str) -> str:
    """Resolve AutoRound's checkpoint directory, including its format subfolder."""
    out = Path(out_dir)
    if (out / "config.json").is_file() and (out / "quantization_config.json").is_file():
        return str(out)

    candidates = [
        path.parent for path in out.glob("*/quantization_config.json") if (path.parent / "config.json").is_file()
    ]
    if len(candidates) == 1:
        resolved = str(candidates[0])
        print(f"[output] AutoRound checkpoint directory: {resolved}")
        return resolved
    if not candidates:
        raise FileNotFoundError(f"No quantized checkpoint found under: {out}")
    raise RuntimeError(f"Multiple quantized checkpoints found under {out}: {candidates}")


def sync_spark2_5_remote_code_to_checkpoint(out_dir: str, fp_model_dir: str | None = None) -> None:
    """Overwrite saved remote-code files with repo Spark2_5 sources (Marlin-safe modeling)."""
    repo_root = Path(__file__).resolve().parents[3]
    ipt_dir = repo_root / "xhmodel_merak" / "xh_llm" / "models" / "xh_token_spark_2_5" / "spark_2_5"
    out = Path(out_dir)
    for name in ("configuration_spark.py", "modeling_spark.py"):
        src = ipt_dir / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing Spark2_5 source: {src}")
        shutil.copy2(src, out / name)
        print(f"[sync] {name} -> {out / name}")

    init_src = ipt_dir / "__init__.py"
    init_dst = out / "__init__.py"
    if init_src.is_file():
        shutil.copy2(init_src, init_dst)
        print(f"[sync] __init__.py -> {init_dst}")
    elif fp_model_dir:
        fp_init = Path(fp_model_dir) / "__init__.py"
        if fp_init.is_file():
            shutil.copy2(fp_init, init_dst)
            print(f"[sync] __init__.py (from fp model) -> {init_dst}")


@torch.inference_mode()
def calculate_avg_ppl(model, tokenizer, device=None):
    from gptqmodel.utils.perplexity import Perplexity

    local_wikitext = Path(__file__).resolve().parents[3] / "data/evaluation/wikitext-2-ppl"
    if not local_wikitext.is_dir():
        raise FileNotFoundError(
            f"Local WikiText2 dataset not found: {local_wikitext}. "
            "Download and save the wikitext-2-raw-v1 test split before using --ppl."
        )
    ppl = Perplexity(
        model=model,
        tokenizer=tokenizer,
        dataset_path=str(local_wikitext),
        split="test",
        text_column="text",
        device=device,
    )
    all_ppl = ppl.calculate(n_ctx=512, n_batch=512)
    if not all_ppl:
        raise ValueError("WikiText2 does not contain a complete 512-token evaluation window")
    # Perplexity.calculate() returns the cumulative PPL after every window;
    # the final element is the metric over the complete evaluation corpus.
    return float(all_ppl[-1])


def _get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        device = getattr(model, "device", "cpu")
        return torch.device(device)


@contextmanager
def temporary_ppl_device(model, target_device: str | None):
    original_device = _get_model_device(model)
    moved = False
    active_device = str(original_device)

    if target_device is None:
        target_device = str(original_device)

    if original_device.type == "cpu" and target_device != "cpu" and torch.cuda.is_available():
        if target_device == "auto":
            target_device = "cuda:0"
        print(f"[ppl] Moving model from cpu to {target_device} for evaluation ...")
        model.to(target_device)
        active_device = target_device
        moved = True

    try:
        yield active_device
    finally:
        if moved:
            print("[ppl] Moving model back to cpu after evaluation ...")
            model.to("cpu")
            torch.cuda.empty_cache()


def build_chat_inputs(tokenizer, prompt: str, device: str):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    return tokenizer(text, return_tensors="pt").to(device)


def parse_args():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    default_model = os.path.join(repo_root, "data/models/hf_spark_4b_ep3")

    parser = argparse.ArgumentParser(description="Spark2_5 mixed-precision AutoRound quantization")
    parser.add_argument("--model", type=str, default=default_model, help="fp model path")
    parser.add_argument("--out", type=str, default=None, help="quantized model output directory")
    parser.add_argument("--bits", type=int, default=4, choices=[4, 5, 8])
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument(
        "--format",
        type=str,
        default="auto_gptq",
        help="AutoRound export format; auto_gptq produces a GPTQModel-compatible checkpoint",
    )
    parser.add_argument("--data-type", type=str, default="int_sym_gptq")
    parser.add_argument("--sym", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--iters", type=int, default=200, help="AutoRound optimization iterations")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulate-steps", type=int, default=1)
    parser.add_argument("--low-gpu-mem-usage", action="store_true")
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Calibration dataset; defaults to the bundled set matching --model",
    )
    parser.add_argument("--calibration-jsonl", type=str, default=None)
    parser.add_argument(
        "--concat-calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pack short local calibration records into exact --seqlen chunks",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="AutoRound compute device")
    parser.add_argument("--infer-device", type=str, default=None)
    parser.add_argument(
        "--device-map",
        type=str,
        default=None,
        choices=["auto", "balanced", "balanced_low_0", "sequential"],
    )
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--prompt", type=str, default="请介绍一下 GPTQ 量化的核心思想。")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--no-trust-remote-code", action="store_false", dest="trust_remote_code")
    parser.add_argument("--skip-quantize", action="store_true", default=False)
    parser.add_argument("--ppl", action="store_false", dest="skip_ppl", help="Evaluate WikiText2 perplexity")
    parser.add_argument("--skip-ppl", action="store_true", dest="skip_ppl", help=argparse.SUPPRESS)
    parser.set_defaults(skip_ppl=True)
    parser.add_argument("--dry-run", action="store_true", help="Print the AutoRound command without executing it")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not args.sym:
        parser.error("Spark AutoRound only supports symmetric quantization; --no-sym is unsupported")
    return args


def main():
    args = parse_args()
    args.model = os.path.abspath(args.model)
    if args.dataset is None and args.calibration_jsonl is None:
        args.dataset = default_calibration_dataset(args.model)
    args.out = args.out or build_default_output_dir(args)
    args.out = os.path.abspath(args.out)
    if args.calibration_jsonl:
        args.calibration_jsonl = os.path.abspath(args.calibration_jsonl)
    if args.dataset and not os.path.isabs(args.dataset) and os.path.exists(args.dataset):
        args.dataset = os.path.abspath(args.dataset)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    tokenizer = None
    tokenizer_source = args.model

    if not args.skip_quantize:
        print("[1/5] Running AutoRound quantization ...")
        run_autoround(args)
        if args.dry_run:
            return
        args.out = resolve_quantized_output_dir(args.out)
        sync_spark2_5_remote_code_to_checkpoint(args.out, fp_model_dir=args.model)
        tokenizer_source = args.out
        print(f"Quantized model saved to: {args.out}")
    else:
        args.out = resolve_quantized_output_dir(args.out)
        print(f"[1/5] Skip quantization, using existing quantized model: {args.out}")
        tokenizer = AutoTokenizer.from_pretrained(args.out, use_fast=True, trust_remote_code=args.trust_remote_code)
        tokenizer_source = args.out

    default_infer = "cuda:0" if torch.cuda.is_available() else "cpu"
    infer_device = args.infer_device or args.device or default_infer
    infer_kwargs = {"trust_remote_code": args.trust_remote_code}
    if args.device_map:
        infer_kwargs["device_map"] = args.device_map
        print(f"[4/5] Loading quantized model with device_map='{args.device_map}' ...")
    else:
        infer_kwargs["device"] = infer_device
        print(f"[4/5] Loading quantized model on {infer_device} ...")

    model = GPTQModel.load(args.out, **infer_kwargs)
    runtime_device = str(_get_model_device(model))

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, use_fast=True, trust_remote_code=args.trust_remote_code
        )

    inputs = build_chat_inputs(tokenizer, args.prompt, runtime_device)
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)

    generated_ids = output_ids[:, inputs["input_ids"].shape[1] :]
    answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    print("=== Prompt ===")
    print(args.prompt)
    print("=== Answer ===")
    print(answer)

    if not args.skip_ppl:
        print("[5/5] Evaluating perplexity on wikitext2 ...")
        with temporary_ppl_device(model, infer_device if not args.device_map else None):
            avg_ppl = calculate_avg_ppl(model, tokenizer)
        print(f"Quantized model avg PPL (wikitext2): {avg_ppl:.4f}")
    else:
        print("[5/5] Skipped perplexity (--skip-ppl)")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    main()
