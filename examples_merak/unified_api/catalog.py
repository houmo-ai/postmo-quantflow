"""列出模型，或查看一个模型允许修改的构建参数。"""

import argparse
import json

from xhmodel_merak.model_build import describe_model, list_models


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", nargs="?", help="不填则列出所有模型")
    parser.add_argument("--full", action="store_true", help="查看完整描述，包括固定配置和推理输入格式")
    args = parser.parse_args(argv)
    if args.full and args.model_id is None:
        parser.error("--full 需要指定 model_id")

    if args.model_id is None:
        result = list_models()
    else:
        result = describe_model(args.model_id, full=args.full)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
