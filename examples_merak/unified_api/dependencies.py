"""导出已注册模型的依赖声明及安装验证状态。"""

import argparse
import json

from xhmodel_merak.model_build import export_model_dependencies


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--variant", help="不传时使用模型默认变体")
    parser.add_argument("--output-dir", required=True, help="新目录；已有目录不覆盖")
    args = parser.parse_args(argv)
    result = export_model_dependencies(model_id=args.model_id, variant=args.variant, output_dir=args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
