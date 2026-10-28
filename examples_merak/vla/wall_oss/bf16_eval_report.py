"""Render the expanded paired report and monitor only its selected GPUs."""

import argparse
import json
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .bf16_open_loop import array_sha, read_record
from .evaluation_queue import EvaluationQueue
from .open_loop import numeric_metrics, sha, write_json


def monitor(run, gpus):
    with (run / "resource_samples.jsonl").open("a", buffering=1) as output:
        while True:
            status_path = run / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {"phase": "benchmark"}
            raw = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--id=" + ",".join(map(str, gpus)),
                    "--query-gpu=index,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            )
            devices = [
                dict(zip(("gpu", "memory_mb", "utilization_percent"), map(int, line.split(",")), strict=True))
                for line in raw.strip().splitlines()
            ]
            output.write(
                json.dumps(dict(time=time.time(), phase=status["phase"], counts=status.get("counts"), devices=devices))
                + "\n"
            )
            if status["phase"] == "complete" or (run / "failure.json").exists():
                return
            time.sleep(15)


def legacy_diagnostic(run, manifest):
    import torch

    bf16, fp32, w8 = [], [], []
    for pair in manifest["pairs"][:120]:
        b, _, _ = read_record(run / "bf16" / f"{pair['id']:04d}.npz", pair, manifest["contracts"]["bf16"])
        q, _, _ = read_record(run / "w8" / f"{pair['id']:04d}.npz", pair, manifest["contracts"]["w8"])
        original = torch.load(Path(manifest["previous"]) / "official" / f"{pair['id']:03d}.pt", weights_only=True)
        np.testing.assert_array_equal(original["noise"].numpy(), b[0])
        bf16.append(b[-1])
        w8.append(q[-1])
        fp32.append(original["trajectory"][-1].numpy())
    b, q, f = map(np.stack, (bf16, w8, fp32))
    mask = np.isin(np.arange(26), range(10, 20))
    result = dict(
        samples=120,
        scope="diagnostic on legacy observations only; main baseline remains BF16",
        bf16_vs_native_fp32=numeric_metrics(b, f, mask),
        w8_vs_bf16=numeric_metrics(q, b, mask),
        w8_vs_native_fp32=numeric_metrics(q, f, mask),
    )
    write_json(run / "legacy_dtype_diagnostic.json", result)
    return result


def table_row(name, record):
    result = record["final"]
    return (
        f"| {name} | {record['samples']} | {result['mae']:.6f} | {result['rmse']:.6f} | "
        f"{100 * result['nrmse']:.3f}% | {result['cosine']:.8f} | "
        f"{record['rmse_p95']:.6f} | {record['worst_cosine']:.8f} |"
    )


def render(run):
    report = json.loads((run / "report.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    benchmark = json.loads((run / "benchmark.json").read_text())
    diagnostic = legacy_diagnostic(run, manifest)
    status = json.loads((run / "status.json").read_text())
    gpus = benchmark["gpus"]
    text = [
        "# Wall-OSS-0.5：官方 BF16 与 W8A8 的 LIBERO 开环数值报告",
        "",
        "主基线为下载的官方 checkpoint，使用官方 Wall-X 的 BF16/FP32 混合精度策略；"
        "主干和视觉为 BF16，官方指定的归一化与动作处理模块保留 FP32。没有把 BF16 模型整体升为 FP32。",
        "",
        "## 测试覆盖与复用",
        "",
        "- 40 个任务 × 每任务 8 个 episode × 每 episode 3 个时刻 × 2 个噪声 seed = **1,920 组配对**。",
        "- 共 320 个 episode、960 个不同观测；每观测 2 次噪声重复，不是 1,920 个独立观测。",
        "- 原有 120 组 W8A8 预测经过模型、SDK、处理后输入、图像和噪声校验后复用；新增 W8A8 1,800 组。",
        "- 官方 BF16 的 1,920 组全部为本轮生成。旧 FP32 预测只用于文末诊断，不混入主基线。",
        "- 仍采用现有通用模型协议：双图长边 448、H32、26 维空间、x2_normal、右臂 10 个有效维度、10 步 Euler。",
        "- 新增 episode 在任务内均匀选取，包含原有 anchor；这是数值探针，不是官方训练/测试划分或闭环成绩。",
        "",
        "## 官方 BF16 对 W8A8：最终归一化动作",
        "",
        "| 范围 | 配对数 | MAE | RMSE | 相对 RMSE | cosine | 样本 RMSE P95 | 最差样本 cosine |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        table_row("合并全部", report["overall"]),
        table_row("原有观测/噪声", report["legacy_120"]),
        table_row("新增配对", report["new_1800"]),
        "",
        "以上为描述性结果；没有用户确认的验收门限，也不将自定门限称为官方标准。按固定切点列出分布供排查：",
        "",
        f"- cosine < 0.98：{report['distribution']['cosine_below_098']} / 1920。",
        f"- RMSE > 0.05：{report['distribution']['rmse_above_005']} / 1920。",
        "",
        "去噪轨迹排除相同的初始噪声，所有主要误差排除禁用动作维度。",
        "",
        "## 动作分组",
        "",
        "| 分组 | MAE | RMSE | 相对 RMSE | cosine | 最大绝对误差 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, title in dict(position="位置", rotation="6D 旋转", gripper="夹爪").items():
        v = report["groups"][name]
        text.append(
            f"| {title} | {v['mae']:.6f} | {v['rmse']:.6f} | {100 * v['nrmse']:.3f}% | "
            f"{v['cosine']:.8f} | {v['max_abs']:.6f} |"
        )
    text += [
        "",
        "## 最差样本",
        "",
        "| 配对 ID | 任务 | episode | 帧 | seed | RMSE | cosine | W8A8 来源 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for v in report["worst_cases"][:15]:
        text.append(
            f"| {v['id']} | {v['task']} | {v['episode']} | {v['frame']} | {v['seed']} | "
            f"{v['final']['rmse']:.6f} | {v['final']['cosine']:.8f} | {v['origin']} |"
        )
    text += [
        "",
        "## GPU 与多引擎吞吐实测",
        "",
        f"仅使用物理 GPU {', '.join(map(str, gpus))}。单、双引擎运行相同固定样本，记录实测吞吐；重复预测逐位一致。",
        "",
        "| 后端 | 每卡单引擎：两卡合计组/秒 | 每卡双引擎：两卡合计组/秒 | 实际每卡引擎数 |",
        "| --- | ---: | ---: | --- |",
    ]
    for backend in ("bf16", "w8"):
        one = sum(v["samples_per_second"] for v in benchmark["results"][backend]["1"].values())
        two = sum(v["samples_per_second"] for v in benchmark["results"][backend]["2"].values())
        text.append(f"| {backend} | {one:.3f} | {two:.3f} | {benchmark['layout'][backend]} |")
    text += [
        "",
        "这是 GPU HMONNX/官方代码的测试吞吐，不是目标芯片性能数据。"
        "SQLite 独占领取样本，逐条原子保存，进程退出后释放未完成任务并重试；"
        "合并要求全部配对完整且请求指纹一致。",
    ]
    if status.get("elapsed_seconds"):
        text.append(
            f"本轮调度器从短程吞吐测试至合并的实测耗时：{status['elapsed_seconds'] / 60:.2f} 分钟；"
            "不含前期实现和样本准备。"
        )
    text += [
        "",
        "## 原 120 组：基线精度诊断（不并入 BF16 主成绩）",
        "",
        "| 对比 | 最终 MAE | 最终 RMSE | cosine |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, title in dict(
        bf16_vs_native_fp32="官方 BF16 vs 官方 FP32",
        w8_vs_bf16="W8A8 vs 官方 BF16",
        w8_vs_native_fp32="W8A8 vs 官方 FP32",
    ).items():
        v = diagnostic[key]
        text.append(f"| {title} | {v['mae']:.6f} | {v['rmse']:.6f} | {v['cosine']:.8f} |")
    text += [
        "",
        "## 限制与复现材料",
        "",
        "该报告比较模型预测之间的数值差异，未使用示范动作评分，不代表 LIBERO 成功率。"
        "状态到预训练右臂槽位的转换未验证控制器标定；现有模型没有 LIBERO 专用归一化统计。"
        "噪声包含保留的旧 GPU 采样张量与新增固定 seed 的 CPU Torch 正态采样张量，"
        "两端均使用同一份实际 FP32 噪声。",
        "",
        "- manifest.json：数据、模型、样本、图像、噪声与执行协议身份。",
        "- reuse_audit.json：原 120 组复用校验，含 SDK 与实际 GPU 扩展二进制身份。",
        "- report.json / cases.csv：整体、分任务、原有/新增和逐样本统计。",
        "- predictions.npz：合并后的完整配对轨迹，第 0 维按 manifest.pairs / 配对 ID 顺序排列。",
        "- engines/*_dtype.json：各 BF16 引擎实际参数 dtype 统计。",
        "- benchmark.json / eta.json / resource_samples.jsonl：并发配置、预计与实际资源记录。",
        "- bf16/*.npz / w8/*.npz：各配对预测及来源元数据。",
        "",
        f"数据 revision：`{manifest['dataset_revision']}`。官方代码 revision：`{manifest['official_revision']}`。",
        "",
        f"报告生成时间：{datetime.now(timezone(timedelta(hours=8))).isoformat()}。",
    ]
    (run / "report.md").write_text("\n".join(text) + "\n")


def audit(run):
    import importlib.metadata

    manifest = json.loads((run / "manifest.json").read_text())
    pairs, observations = manifest["pairs"], manifest["observations"]
    assert len(pairs) == 1920 and len(observations) == 960
    assert [p["id"] for p in pairs] == list(range(1920))
    assert len({(o["episode_index"], o["frame_index"]) for o in observations}) == 960
    for task in range(40):
        selected = [o for o in observations if o["task_index"] == task]
        assert len(selected) == 24 and len({o["episode_index"] for o in selected}) == 8
    for observation in observations:
        selected = [p for p in pairs if p["observation_id"] == observation["observation_id"]]
        assert len(selected) == 2 and len({p["seed"] for p in selected}) == 2
        for image, source in zip(observation["images"], observation["sources"], strict=True):
            assert sha(image) == source["png_sha256"]
    origins, cohorts = {}, {}
    for backend in ("bf16", "w8"):
        assert {int(p.stem) for p in (run / backend).glob("*.npz")} == set(range(1920))
        origins[backend], cohorts[backend] = {}, set()
        for pair in pairs:
            noise = np.load(pair["noise"], allow_pickle=False)
            assert array_sha(noise) == pair["noise_sha256"]
            _, _, metadata = read_record(run / backend / f"{pair['id']:04d}.npz", pair, manifest["contracts"][backend])
            origin = metadata["origin"]
            origins[backend][origin] = origins[backend].get(origin, 0) + 1
            if origin == "new":
                assert "_g6_" in metadata["owner"] or "_g7_" in metadata["owner"]
                cohorts[backend].add(metadata["engine_code_sha256"])
            else:
                assert backend == "w8" and pair["id"] < 120
                assert sha(metadata["source_file"]) == metadata["source_sha256"]
    assert origins == {"bf16": {"new": 1920}, "w8": {"reused": 120, "new": 1800}}
    executed_source = sha(Path(__file__).with_name("bf16_open_loop.py"))
    assert cohorts["bf16"] == cohorts["w8"] == {executed_source}
    dtype_files = sorted((run / "engines").glob("*_dtype.json"))
    for path in dtype_files:
        value = json.loads(path.read_text())
        assert value["embedding"] == value["visual"] == "torch.bfloat16"
        assert value["action_projection"] == "torch.float32"
        assert value["parameter_elements"]["torch.bfloat16"] > 0
    assert dtype_files
    for relative, expected in manifest["artifact_files"].items():
        assert sha(Path(manifest["artifact_dir"]) / relative) == expected
    assert sha(Path(manifest["model_dir"]) / "model.safetensors") == manifest["base_weights_sha256"]
    queue = EvaluationQueue(run / "queue.sqlite3")
    assert queue.counts("bf16") == queue.counts("w8") == {"done": 1920}
    record = dict(
        complete=True,
        tasks=40,
        episodes=320,
        observations=960,
        pairs=1920,
        no_missing_or_duplicate_pairs=True,
        origins=origins,
        inference_code_cohorts={k: sorted(v) for k, v in cohorts.items()},
        bf16_dtype_audits=len(dtype_files),
        selected_physical_gpus=[6, 7],
        artifact_files_unchanged=len(manifest["artifact_files"]),
        source_weights_unchanged=True,
        deployment_dependencies={n: importlib.metadata.version(n) for n in ("torch", "transformers", "numpy")},
    )
    write_json(run / "completion_audit.json", record)
    print(json.dumps(record, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["monitor", "render", "audit"])
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="6,7")
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.stage == "monitor":
        monitor(run, list(map(int, args.gpus.split(","))))
    elif args.stage == "render":
        render(run)
    else:
        audit(run)


if __name__ == "__main__":
    main()
