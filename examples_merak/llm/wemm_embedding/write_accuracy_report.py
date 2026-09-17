"""Write a checked accuracy report from fresh WeMM Flickr30K and MMEB-v3 runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


SIZES = ("2b", "4b", "9b")
TASKS = ("Visual7W", "ChartQA")
EXPECTED_IMAGE_PREPROCESSING = {
    "mode": "dynamic",
    "resized_height": None,
    "resized_width": None,
    "min_pixels": 65536,
    "max_pixels": 1572864,
    "preserve_aspect_ratio": True,
}


def read_json(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def delta_pp(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=Path("work_dirs/wemm_embedding"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.work_root
    output = args.output or root / "ACCURACY_REPORT.md"
    flickr = root / "eval_flickr30k_1000"
    mmeb = root / "mmeb_v3_runs"
    try:
        display_root = root.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        display_root = root
    lines = [
        "# WeMM-Embedding AutoRound 精度复测报告",
        "",
        f"生成时间：{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        "",
        "量化配置：2B、4B、9B 的语言分支 AutoRound W4G64，256 个校准样本、400 次迭代，",
        "`data_type=int_sym_xh`；HMONNX 视觉分支 W8A8、patch token 档位。",
        "",
        "## 动态分辨率审计",
        "",
        "Native BF16 与 HMONNX 使用同一图像预处理协议：保持宽高比，像素数限制为 65,536–1,572,864，",
        "没有设置固定高宽。下表由本次评测数据的全部图像逐张计算。",
        "",
        "| 数据集 | 图像数 | 原始尺寸种类 | 预处理尺寸种类 | token 范围 | m96 | m196 | m384 | m704 | m1536 | 448×448 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    audit = read_json(root / "DYNAMIC_IMAGE_PREPROCESSING_AUDIT.json")
    if audit.get("image_preprocessing") != EXPECTED_IMAGE_PREPROCESSING:
        raise ValueError("dynamic-resolution audit protocol mismatch")
    for dataset in ("Flickr30K-1000", "Visual7W", "ChartQA"):
        row = audit["datasets"][dataset]
        gears = row["gear_counts"]
        lines.append(
            f"| {dataset} | {row['images']} | {row['unique_original_sizes']} | "
            f"{row['unique_resized_sizes']} | {row['visual_token_min']}–{row['visual_token_max']} | "
            f"{gears['m96']} | {gears['m196']} | {gears['m384']} | {gears['m704']} | "
            f"{gears['m1536']} | {row['fixed_448x448_images']} |"
        )
    lines += [
        "",
        "448×448 列只是动态预处理自然产生该尺寸的样本数，不表示强制 resize 到 448×448。",
        "HMONNX 按合并后视觉 token 数选择最小可容纳的已导出档位。",
        "",
        "## Flickr30K-1000",
        "",
        "测试集为 1,000 张图像、5,000 条 caption。Native BF16 与 HMONNX 均保持图片宽高比，",
        "使用相同的动态分辨率范围（65,536–1,572,864 像素）；HMONNX 再按有效视觉 token 数选择最小可容纳档位。",
        "",
        "| 规格 | 后端 | T2I R@1 | I2T R@1 | T2I R@5 | I2T R@5 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    flickr_data = {}
    for size in SIZES:
        provenance = read_json(root / size / "wemm_embedding_quant" / "qwen35_autoround_recipe_provenance.json")
        if provenance.get("data_type") != "int_sym_xh":
            raise ValueError(f"{size}: expected fresh int_sym_xh AutoRound provenance")
        reports = {}
        for backend, suffix in (("native", "native_dynamic"), ("hmonnx", "hmonnx_w4a8_dynamic")):
            report = read_json(flickr / f"{size}_{suffix}_flickr1000.json")
            if report.get("num_images") != 1000 or report.get("num_captions") != 5000:
                raise ValueError(f"{size} {backend}: expected full Flickr30K-1000 test set")
            if report.get("backend") != backend:
                raise ValueError(f"{size} {backend}: backend mismatch")
            if report.get("image_preprocessing") != EXPECTED_IMAGE_PREPROCESSING:
                raise ValueError(f"{size} {backend}: dynamic image preprocessing protocol mismatch")
            reports[backend] = report
            t2i, i2t = report["text_to_image"], report["image_to_text"]
            lines.append(
                f"| {size.upper()} | {backend} | {pct(t2i['recall@1'])} | {pct(i2t['recall@1'])} "
                f"| {pct(t2i['recall@5'])} | {pct(i2t['recall@5'])} |"
            )
        flickr_data[size] = reports
    lines += [
        "",
        "| 规格 | 后端 | T2I R@10 | I2T R@10 | T2I MRR@10 | I2T MRR@10 | T2I nDCG@10 | I2T nDCG@10 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for size in SIZES:
        for backend in ("native", "hmonnx"):
            report = flickr_data[size][backend]
            t2i, i2t = report["text_to_image"], report["image_to_text"]
            lines.append(
                f"| {size.upper()} | {backend} | {pct(t2i['recall@10'])} | {pct(i2t['recall@10'])} "
                f"| {pct(t2i['mrr@10'])} | {pct(i2t['mrr@10'])} "
                f"| {pct(t2i['ndcg@10'])} | {pct(i2t['ndcg@10'])} |"
            )
    lines += [
        "",
        "| 规格 | T2I R@1 差值 | I2T R@1 差值 | 文本 cosine 均值 | 图像 cosine 均值 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for size in SIZES:
        native, hmonnx = flickr_data[size]["native"], flickr_data[size]["hmonnx"]
        cosine = read_json(flickr / f"{size}_full_cosine_dynamic.json")
        if cosine.get("num_images") != 1000 or cosine.get("num_captions") != 5000:
            raise ValueError(f"{size}: incomplete cosine comparison")
        lines.append(
            f"| {size.upper()} | "
            f"{delta_pp(hmonnx['text_to_image']['recall@1'] - native['text_to_image']['recall@1'])} | "
            f"{delta_pp(hmonnx['image_to_text']['recall@1'] - native['image_to_text']['recall@1'])} | "
            f"{cosine['text_cosine']['mean']:.6f} | {cosine['image_cosine']['mean']:.6f} |"
        )
    lines += [
        "",
        "两端使用相同动态图像预处理、相同样本及顺序；上述精度差值和 cosine 用于量化 HMONNX 对照。",
        "",
        "## MMEB-v3 图像任务",
        "",
        "使用官方评测器的 Visual7W 和 ChartQA 完整 test 集；每项 1,000 条查询。",
        "HF 与 HMONNX 均保持图片宽高比并使用相同动态像素范围；比较脚本逐项检查图像协议、",
        "查询、候选及其顺序，协议不一致时拒绝生成对比结果。",
        "此处仅覆盖两项图像任务，不代表 MMEB-v3 全套总分。",
        "",
        "| 规格 | 任务 | 查询数 | 候选数 | HF Hit@1 | HMONNX Hit@1 | 差值 | 查询 cosine 均值 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    mmeb_data = {}
    for size in SIZES:
        for task in TASKS:
            rows = read_json(mmeb / f"{size}_{task}_comparison.json")
            if len(rows) != 1 or rows[0].get("task") != task:
                raise ValueError(f"{size} {task}: invalid comparison")
            row = rows[0]
            if row["queries"] != 1000:
                raise ValueError(f"{size} {task}: expected 1000 queries")
            if row.get("image_preprocessing") != EXPECTED_IMAGE_PREPROCESSING:
                raise ValueError(f"{size} {task}: dynamic image preprocessing protocol mismatch")
            mmeb_data[size, task] = row
            lines.append(
                f"| {size.upper()} | {task} | {row['queries']} | {row['candidates']} | "
                f"{pct(row['hf']['hit@1'])} | {pct(row['hmonnx']['hit@1'])} | "
                f"{delta_pp(row['delta_hmonnx_minus_hf']['hit@1'])} | "
                f"{row['query_cosine']['mean']:.6f} |"
            )
    lines += [
        "",
        "| 规格 | 任务 | 后端 | Hit@5 | Hit@10 | MRR | nDCG-linear@10 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for size in SIZES:
        for task in TASKS:
            row = mmeb_data[size, task]
            for backend in ("hf", "hmonnx"):
                scores = row[backend]
                lines.append(
                    f"| {size.upper()} | {task} | {backend} | {pct(scores['hit@5'])} "
                    f"| {pct(scores['hit@10'])} | {pct(scores['mrr'])} "
                    f"| {pct(scores['ndcg_linear@10'])} |"
                )
    lines += [
        "",
        "## 可复查产物",
        "",
        f"- Flickr30K 逐规格检索结果和向量：`{display_root / 'eval_flickr30k_1000'}`",
        f"- MMEB-v3 官方分数、逐查询预测、HF/HMONNX 对比及协议：`{display_root / 'mmeb_v3_runs'}`",
        f"- 三规格量化与 HMONNX 导出：`{display_root}/<size>/wemm_embedding_quant`、"
        f"`{display_root}/<size>/wemm_embedding_export`",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
