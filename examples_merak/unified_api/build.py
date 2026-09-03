"""从本地模型权重构建一份可供统一推理接口加载的产物。"""

import argparse
import json

from xhmodel_merak.model_build import build_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True, help="从 catalog 示例中选择的 model_id")
    parser.add_argument("--model-dir", required=True, help="本地原始模型目录")
    parser.add_argument("--output-dir", required=True, help="最终产物目录，必须尚不存在")
    parser.add_argument("--device", help="执行构建的设备，例如 cuda:0；不是目标芯片")
    parser.add_argument("--quant-checkpoint", help="复用匹配的量化 checkpoint，跳过权重量化")
    parser.add_argument("--context-length", type=int, help="覆盖模型的默认上下文长度")
    parser.add_argument("--prefill-chunk-length", type=int, help="覆盖模型的默认 prefill 分块长度")
    args = parser.parse_args(argv)

    result = build_model(
        model_id=args.model_id,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        device=args.device,
        quant_checkpoint=args.quant_checkpoint,
        context_length=args.context_length,
        prefill_chunk_length=args.prefill_chunk_length,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
