"""Run the MOSS-Transcribe-Diarize Merak workflow.

Examples::

    PYTHONPATH=$PWD python examples_merak/asr/moss_transcribe_diarize/moss_transcribe_diarize_workflow.py \
        --model-dir <model_dir> --device cuda:0 --overwrite

    PYTHONPATH=$PWD python examples_merak/asr/moss_transcribe_diarize/moss_transcribe_diarize_workflow.py \
        --model-dir <model_dir> --golden-audio <audio_file> --dump-golden
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_CONFIG_PATH = (
    "configs_merak/workflows/xh2a/llm_models/moss_transcribe_diarize/"
    "0_9b/moss_transcribe_diarize.yaml"
)
DEFAULT_OUTPUT_DIR = "work_dirs/moss_transcribe_diarize_09b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MOSS-Transcribe-Diarize Merak workflow.")
    parser.add_argument("--model-dir", required=True, help="Local MOSS model bundle directory.")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--context-length",
        type=int,
        default=None,
        help=(
            "Optional build_model context_length: 2048 default YAML, 16384 plugin FA, "
            "65536 30-minute FA, 131072 long-context FA. Long-context FA graphs also need "
            "--prefill-chunk-length 4096; omitting it keeps the public default 512."
        ),
    )
    parser.add_argument(
        "--prefill-chunk-length",
        type=int,
        default=None,
        help="Optional build_model prefill_chunk_length. 30-minute / 16k / 128k FA requires 4096.",
    )
    parser.add_argument("--golden-audio", default=None, help="Audio path used by --dump-golden.")
    parser.add_argument("--dump-golden", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dump_golden and not args.golden_audio:
        raise ValueError("--dump-golden requires --golden-audio")
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    if args.prefill_chunk_length is not None and args.context_length is None:
        raise ValueError("--prefill-chunk-length requires --context-length")
    if args.context_length is not None:
        from xhmodel_merak.model_build import build_model

        build_kwargs = {
            "model_id": "moss-transcribe-diarize-0.9b",
            "model_dir": args.model_dir,
            "output_dir": str(output_dir),
            "device": args.device,
            "context_length": args.context_length,
        }
        if args.prefill_chunk_length is not None:
            build_kwargs["prefill_chunk_length"] = args.prefill_chunk_length
        result = build_model(**build_kwargs)
        print(f"build_result: {result}")
        if args.dump_golden:
            raise ValueError("--dump-golden is not combined with --context-length; use the default YAML path")
        return

    from xhmodel_merak.workflows import AutoWorkflow

    workflow = AutoWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
        debug=args.debug,
    )
    quant_result = workflow.quant(output_dir=str(output_dir / "quant"), device=args.device)
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=str(output_dir),
        device=args.device,
    )
    print(f"export_result: {export_result}")
    if args.dump_golden:
        golden_file = workflow.dump_golden(
            export_result=export_result,
            device=args.device,
            input_messages={"audio": args.golden_audio},
        )
        print(f"golden_file: {golden_file}")


if __name__ == "__main__":
    main()
