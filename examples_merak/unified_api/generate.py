"""加载统一构建接口的产物，进行文本或单张图片问答。"""

import argparse
import json

from xhmodel_merak.model_inference import inference_model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, help="build_model 返回的 output_dir")
    parser.add_argument("--prompt", required=True, help="问题或提示词")
    parser.add_argument("--image", help="本地图片路径；需要模型支持图片输入")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="最多生成多少个 token，默认 64")
    parser.add_argument("--device", help="推理设备，例如 cuda:0")
    parser.add_argument("--golden", nargs="+", help="额外生成指定组件的 golden，例如 visual prefill decode")
    args = parser.parse_args(argv)

    inputs = {"prompt": args.prompt, "max_new_tokens": args.max_new_tokens}
    if args.image is not None:
        inputs["image"] = args.image
    result = inference_model(
        artifact_dir=args.artifact_dir,
        inputs=inputs,
        device=args.device,
        golden=args.golden,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
