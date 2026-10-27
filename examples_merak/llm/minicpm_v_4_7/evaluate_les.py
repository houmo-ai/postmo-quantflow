# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
"""Evaluate next-layer expert prediction on native MiniCPM weights, without changing routing."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from xhmodel_merak.xh_llm.models.minicpm_v_4_7.les import (
    LESObserver,
    LES_REFERENCE,
    prediction_counts,
    prediction_rates,
)

QUESTIONS = {
    "数学推理": [
        "解方程 3x+7=31，并说明步骤。",
        "一个袋子有3红球2蓝球，无放回抽2球，都是红球的概率是多少？",
        "求函数 f(x)=x^3-3x 的极值点。",
        "等差数列首项3、公差4，求前20项之和。",
        "直角三角形两直角边为6和8，求内切圆半径。",
        "用数学归纳法证明前n个正奇数之和是n平方。",
        "一项工作甲独做6天，乙独做12天，合作需要多少天？",
        "求从1到100中能被3或5整除的整数个数。",
        "抛三次公平硬币，至少出现两次正面的概率是多少？",
        "矩阵[[2,1],[1,2]]的特征值是什么？",
    ],
    "代码": [
        "写一个Python函数，返回数组中两个数之和等于目标值的下标，并说明复杂度。",
        "Python可变对象作为函数默认参数会带来什么问题？给一个修正示例。",
        "用Python实现二分查找，处理空数组。",
        "SQL中如何查找一张员工表中每个部门工资最高的员工？",
        "解释TCP和UDP的区别及各自适用场景。",
        "写一个Python生成器，逐行读取大文件并过滤空行。",
        "比较Python浅拷贝与深拷贝，给出嵌套列表示例。",
        "用Python判断一个字符串的括号是否匹配。",
        "解释数据库事务的ACID，并举出隔离性问题。",
        "解释归并排序的核心步骤以及时间和空间复杂度。",
    ],
    "科学知识": [
        "为什么天空通常是蓝色，而日落时常呈红色？",
        "解释光合作用中光反应和碳反应的关系。",
        "牛顿第三定律为何不会导致物体无法运动？",
        "解释DNA复制时前导链与后随链的差别。",
        "为什么金属通常比木头摸起来更凉？",
        "解释半导体PN结的耗尽层如何形成。",
        "温室效应的物理机制是什么？",
        "解释催化剂如何改变反应速率而不改变平衡常数。",
        "简述板块构造如何形成地震。",
        "比较存算一体与传统计算架构的数据搬运方式。",
    ],
    "中文写作": [
        "写一段80字左右的雨后校园景物描写。",
        "写一封简短邮件，礼貌地说明项目需要延期两天。",
        "用平实的语言给小学生解释什么是勇气。",
        "写一段推介城市公共图书馆的文案，避免夸张。",
        "以旧车票为线索写一段微型故事的开头。",
        "写一份社区垃圾分类活动通知，包含时间地点与参加方式。",
        "把‘他很生气’改写成通过动作和神态表现情绪的段落。",
        "写一段关于团队沟通重要性的议论文开头。",
        "用三个有画面感的句子描写深秋的山路。",
        "写一段感谢同事帮助排查故障的话，语气自然具体。",
    ],
    "图像理解": [
        "图中有哪些颜色的几何形状？请说明相对位置。",
        "图中有几个红色圆形，几个蓝色方形？",
        "比较图中A、B、C三个柱子的高度，哪个最高？",
        "读出图中表格各行的数值，并计算总和。",
        "观察折线图，数值总体上升还是下降？哪里最高？",
        "描述图中箭头的方向以及起点和终点。",
        "图中左边和右边分别是什么形状，颜色如何？",
        "读出图中写出的算式并计算结果。",
        "观察图中三个物体的位置，哪个在中间？",
        "请说明图中四个色块从左到右的颜色顺序。",
    ],
}


def make_image(index: int):
    """Small original diagrams with visible content, not blank-image placeholders."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (448, 336), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    if index in (0, 1, 6, 8):
        draw.ellipse((30, 100, 130, 200), fill="red")
        draw.rectangle((175, 100, 275, 200), fill="blue")
        draw.polygon([(360, 90), (310, 205), (420, 205)], fill="green")
        if index == 1:
            draw.ellipse((30, 220, 100, 290), fill="red")
    elif index == 2:
        for x, h, label in [(60, 70, "A"), (180, 190, "B"), (300, 120, "C")]:
            draw.rectangle((x, 270 - h, x + 65, 270), fill="steelblue")
            draw.text((x + 20, 280), label, fill="black", font=font)
    elif index == 3:
        for row, text in enumerate(("Item     Value", "A          12", "B          18", "C          25")):
            draw.text((55, 35 + row * 65), text, fill="black", font=font)
    elif index == 4:
        points = [(40, 260), (120, 210), (200, 230), (280, 110), (390, 50)]
        draw.line(points, fill="blue", width=5)
        for x, y in points:
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill="red")
    elif index == 5:
        draw.line((70, 170, 360, 170), fill="black", width=8)
        draw.polygon([(360, 170), (325, 145), (325, 195)], fill="black")
        draw.text((40, 220), "START", fill="black", font=font)
        draw.text((330, 220), "END", fill="black", font=font)
    elif index == 7:
        draw.text((75, 140), "17 + 28 = ?", fill="black", font=font)
    else:
        for x, color in enumerate(("red", "yellow", "green", "blue")):
            draw.rectangle((20 + x * 105, 110, 110 + x * 105, 230), fill=color)
    return image


def aggregate_results(questions):
    """Weight every token/layer pair equally, rather than averaging questions."""
    domains = defaultdict(lambda: defaultdict(int))
    layers = defaultdict(lambda: defaultdict(int))
    domain_layers = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    overall = defaultdict(int)
    count_keys = ("tokens", "expert_hits", "actual_experts", "predicted_experts", "complete_hits", "exact_hits")
    for question in questions:
        for layer, counts in question["layers"].items():
            buckets = (domains[question["domain"]], layers[layer], domain_layers[question["domain"]][layer], overall)
            for bucket in buckets:
                for key in count_keys:
                    bucket[key] += counts[key]

    def with_rates(counts):
        return dict(counts, **prediction_rates(counts))

    return dict(
        domains={name: with_rates(counts) for name, counts in domains.items()},
        per_layer={layer: with_rates(counts) for layer, counts in layers.items()},
        per_domain_layer={
            name: {layer: with_rates(counts) for layer, counts in data.items()} for name, data in domain_layers.items()
        },
        overall=with_rates(overall),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prediction-k", type=int, default=8)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    if args.max_new_tokens < 2 or not 1 <= args.limit <= 50:
        parser.error("max-new-tokens >= 2 and limit in [1, 50] are required")
    from transformers import AutoModelForImageTextToText, AutoProcessor

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
    print("Loading native BF16 MiniCPM", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_dir,
        dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="sdpa",
        trust_remote_code=True,
    ).eval()
    examples = []
    torch.manual_seed(47)
    for domain, questions in QUESTIONS.items():
        for index, question in enumerate(questions):
            if len(examples) >= args.limit:
                break
            content = [{"type": "text", "text": question}]
            if domain == "图像理解":
                image = make_image(index)
                image_path = out / f"image_{index + 1:02}.png"
                image.save(image_path)
                content.insert(0, {"type": "image", "image": image})
            inputs = processor.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
                processor_kwargs={"images_kwargs": {"downsample_mode": "16x", "max_slice_nums": 1}},
            ).to(args.device)
            generate = dict(max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True)
            # Compare the complete greedy output for *every* question with and
            # without prediction hooks, not just the prediction statistics.
            with torch.inference_mode():
                baseline = model.generate(**inputs, **generate)
                with LESObserver(model.model.language_model.layers, args.prediction_k) as observer:
                    observed = model.generate(**inputs, **generate)
            if not torch.equal(baseline, observed):
                raise AssertionError(f"LES changed greedy output: {domain}/{index + 1}")
            layer_counts = defaultdict(lambda: defaultdict(int))
            traces = []
            for layer, predicted, actual in observer.records:
                counts = prediction_counts(predicted, actual)
                for key, value in counts.items():
                    layer_counts[layer][key] += value
                traces.append(dict(layer=layer, predicted=predicted.cpu().tolist(), actual=actual.cpu().tolist()))
            entry = dict(
                domain=domain,
                index=index + 1,
                question=question,
                prompt_tokens=inputs.input_ids.shape[-1],
                generated_tokens=observed.shape[-1] - inputs.input_ids.shape[-1],
                output_equal=True,
                baseline_token_ids=baseline[0, inputs.input_ids.shape[-1] :].tolist(),
                observed_token_ids=observed[0, inputs.input_ids.shape[-1] :].tolist(),
                answer=processor.tokenizer.decode(observed[0, inputs.input_ids.shape[-1] :], skip_special_tokens=True),
                layers={str(layer): dict(counts, **prediction_rates(counts)) for layer, counts in layer_counts.items()},
            )
            examples.append(entry)
            (out / f"trace_{len(examples):02}.json").write_text(json.dumps(traces), encoding="utf-8")
            summary = dict(
                model_dir=str(Path(args.model_dir).resolve()),
                precision="native_bfloat16",
                reference=LES_REFERENCE,
                prediction_k=args.prediction_k,
                actual_k=model.config.text_config.num_experts_per_tok,
                phase="single_token_decode_only",
                max_new_tokens=args.max_new_tokens,
                baseline_comparison="every_question_full_greedy_token_ids",
                **aggregate_results(examples),
                questions=examples,
            )
            (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                f"{len(examples)}/50 {domain}/{index + 1}: {entry['generated_tokens']} tokens, output equal", flush=True
            )


if __name__ == "__main__":
    main()
