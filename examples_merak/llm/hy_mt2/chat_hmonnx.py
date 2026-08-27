import argparse
import time

from examples_merak.llm.hy_mt2.eval_flores import HMONNXBackend
from xhmodel_merak.xh_other_model.models.hy_mt2.evaluation import build_translation_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Hy-MT2 translation through prefill/decode HMONNX.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--export-path", required=True)
    parser.add_argument("--source-text", default="今天天气真好。")
    parser.add_argument("--source-code", choices=("zho_Hans", "eng_Latn"), default="zho_Hans")
    parser.add_argument("--target-code", choices=("zho_Hans", "eng_Latn"), default="eng_Latn")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execution-device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.6)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--fast", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source_code == args.target_code:
        raise ValueError("source and target languages must differ")

    backend = HMONNXBackend(
        args.model_dir,
        args.export_path,
        args.device,
        args.execution_device,
        args.fast,
        False,
    )
    generation = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "repetition_penalty": args.repetition_penalty,
        "pad_token_id": backend.tokenizer.pad_token_id or backend.tokenizer.eos_token_id,
        "eos_token_id": backend.tokenizer.eos_token_id,
    }
    if args.do_sample:
        generation.update(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)

    prompt = build_translation_prompt(args.source_text, args.source_code, args.target_code)
    started = time.perf_counter()
    response = backend.generate(prompt, generation)
    print(response)
    print(f"latency_seconds={time.perf_counter() - started:.3f}")


if __name__ == "__main__":
    main()
