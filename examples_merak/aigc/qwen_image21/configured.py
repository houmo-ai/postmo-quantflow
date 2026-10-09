"""Build a fixed-size Qwen-Image artifact from a user-supplied workflow YAML."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--config', required=True, help='Workflow YAML with export sizes and token capacities')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    from xhmodel_merak.workflows import AutoWorkflow

    root = Path(args.output_dir)
    if root.exists():
        raise ValueError('output-dir must not exist')
    workflow = AutoWorkflow.from_config(model_dir=args.model_dir, config_path=args.config)
    quant = workflow.quant(output_dir=str(root), device=args.device)
    result = workflow.export(quant, output_dir=str(root), device=args.device)
    print(f'Built configured artifact: {result.work_dir}')


if __name__ == '__main__':
    main()
