"""Entry point: validate JSON/CLI and customer inputs before loading CUDA."""
import json
import os
import sys

from common import (REPO_ROOT, configure_customer_sources, parse_training_args,
                    prepare_output)


def main(argv=None):
    args = parse_training_args(argv)
    try:
        configure_customer_sources(args.customer_sources)
        if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
            raise FileExistsError(f"Output must be a new or empty directory: {args.output}")
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["USE_FAST_TRITON_MATMUL"] = "0"
    os.environ["USE_TRITON_MATMUL"] = "1"
    sys.path.insert(0, str(REPO_ROOT))
    # Heavy imports follow source/config validation; --help needs only stdlib.
    from trainer import run
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required; choose a Torch/CUDA/Triton environment (see README).")
    prepare_output(args.output)
    effective = {key: str(value) if hasattr(value, "__fspath__") else value
                 for key, value in vars(args).items()}
    (args.output / "effective_config.json").write_text(json.dumps(effective, indent=2))
    run(args)


if __name__ == "__main__":
    main()
