"""Isolated Spark worker for the local recipe and GPTQModel's bundled AutoRound."""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    for name in ("output_dir", "dataset", "layer_config", "scheme", "data_type", "device_map", "format"):
        parser.add_argument(f"--{name}", required=True)
    for name in (
        "bits",
        "group_size",
        "iters",
        "seed",
        "nsamples",
        "seqlen",
        "batch_size",
        "gradient_accumulate_steps",
    ):
        parser.add_argument(f"--{name}", type=int, required=True)
    parser.add_argument("--low_gpu_mem_usage", action="store_true")
    parser.add_argument("--disable_trust_remote_code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from .autoround_recipe import quantize_spark25_autoround, require_bundled_autoround

    require_bundled_autoround()
    from auto_round.utils.common import parse_layer_config_arg

    from .spark_2_5 import register_spark2_5

    register_spark2_5()
    options = vars(args).copy()
    options["model_dir"] = options.pop("model")
    options["trust_remote_code"] = not options.pop("disable_trust_remote_code")
    options.pop("scheme")
    options["layer_config"] = parse_layer_config_arg(options["layer_config"])
    options["concat"] = options["dataset"].endswith(":concat")
    if options["concat"]:
        options["dataset"] = options["dataset"][: -len(":concat")]
    quantize_spark25_autoround(**options)


if __name__ == "__main__":
    main()
