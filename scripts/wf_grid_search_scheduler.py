"""WF网格搜索的并行调度器。

直接用subprocess启动独立Python进程跑每个WF配置，比ProcessPoolExecutor更稳定。
每个worker进程串行处理分配给它的WF配置列表。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

SCRIPT_PATH = Path(__file__).resolve()


def main():
    parser = argparse.ArgumentParser(description="Parallel WF grid search scheduler")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--n-jobs-per-worker", type=int, default=8, help="XGBoost n_jobs per worker")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--smoke", action="store_true", help="Smoke test")
    parser.add_argument("--oos-only", action="store_true", help="OOS only backtest")
    parser.add_argument("--output", type=str, default=None, help="Output dir name")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    # 先获取所有WF配置列表
    sys.path.insert(0, str(SRC_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from wf_grid_search import (
        generate_wf_configs, generate_bt_configs,
        WF_PARAM_GRID, BT_PARAM_GRID,
        SMOKE_WF_PARAM_GRID, SMOKE_BT_PARAM_GRID,
        BASE_OUTPUT,
    )

    if args.smoke:
        wf_grid = SMOKE_WF_PARAM_GRID
        bt_grid = SMOKE_BT_PARAM_GRID
        output_suffix = "smoke"
    else:
        wf_grid = WF_PARAM_GRID
        bt_grid = BT_PARAM_GRID
        output_suffix = "full"

    if args.output:
        output_suffix = args.output

    # 覆盖n_jobs
    wf_grid["model_params"]["n_jobs"] = [args.n_jobs_per_worker]

    wf_configs = generate_wf_configs(wf_grid)
    bt_configs = generate_bt_configs(bt_grid)
    output_base = BASE_OUTPUT / output_suffix
    output_base.mkdir(parents=True, exist_ok=True)
    log_dir = output_base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("WF Grid Search - 并行调度器")
    print("=" * 80)
    print(f"WF配置数: {len(wf_configs)}")
    print(f"回测配置数: {len(bt_configs)}")
    print(f"总组合数: {len(wf_configs) * len(bt_configs)}")
    print(f"Workers: {args.num_workers}")
    print(f"每worker核数: {args.n_jobs_per_worker}")
    print(f"输出目录: {output_base}")
    print()

    # 过滤掉已经完成的WF配置（有预测结果 + 所有回测都完成了）
    pending_wfs = []
    for wf in wf_configs:
        pred_path = output_base / "predictions" / wf.name / "walk_forward_full_pred.parquet"
        if not pred_path.exists():
            pending_wfs.append(wf)
            continue
        # 检查是否所有回测都完成了
        all_bt_done = True
        for bt in bt_configs:
            bt_dir = output_base / "backtest" / wf.name / bt.name
            if not (bt_dir / "analysis.json").exists():
                all_bt_done = False
                break
        if not all_bt_done:
            pending_wfs.append(wf)

    done_count = len(wf_configs) - len(pending_wfs)
    print(f"已完成: {done_count}/{len(wf_configs)}")
    print(f"待处理: {len(pending_wfs)}/{len(wf_configs)}")
    print()

    if not pending_wfs:
        print("所有WF配置都已完成！")
        return

    # 将待处理WF配置分配给workers
    worker_wfs = [[] for _ in range(args.num_workers)]
    for i, wf in enumerate(pending_wfs):
        worker_wfs[i % args.num_workers].append(wf.name)

    # 启动worker进程（错峰启动，避免同时加载数据导致内存峰值）
    print(f"启动 {args.num_workers} 个worker...")
    processes = []
    for worker_id in range(args.num_workers):
        wf_names = worker_wfs[worker_id]
        if not wf_names:
            continue

        log_file = log_dir / f"worker_{worker_id:02d}.log"

        # 构造命令：调用 wf_grid_search_worker.py
        cmd = [
            sys.executable,
            str(SCRIPT_PATH.parent / "wf_grid_search_worker.py"),
            "--worker-id", str(worker_id),
            "--n-jobs", str(args.n_jobs_per_worker),
            "--output", output_suffix,
            "--wf-names", ",".join(wf_names),
        ]
        if args.cpu:
            cmd.append("--cpu")
        if args.smoke:
            cmd.append("--smoke")
        if args.oos_only:
            cmd.append("--oos-only")
        if args.verbose:
            cmd.extend(["--verbose", str(args.verbose)])

        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC_ROOT) + ":" + env.get("PYTHONPATH", "")
        env["PYTHONUNBUFFERED"] = "1"

        log_f = open(log_file, "w", buffering=1)  # line buffering
        proc = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=env,
        )
        processes.append((worker_id, proc, log_file, wf_names))
        print(f"  Worker {worker_id:02d}: {len(wf_names)} 个WF配置, PID={proc.pid}, log={log_file.name}")

        # 错峰启动：每个worker间隔60秒，避免同时加载数据
        if worker_id < args.num_workers - 1:
            time.sleep(60)

    print()
    print("监控中... (按 Ctrl+C 停止)")
    print()

    # 监控进度
    try:
        while True:
            all_done = True
            status_lines = []
            for worker_id, proc, log_file, wf_names in processes:
                ret = proc.poll()
                if ret is None:
                    all_done = False
                    # 读日志最后一行看进度
                    try:
                        with open(log_file, "r") as f:
                            lines = [l.strip() for l in f.readlines() if l.strip()]
                        last_line = lines[-1] if lines else "(starting)"
                        # 截断
                        if len(last_line) > 100:
                            last_line = last_line[:97] + "..."
                        status_lines.append(f"  Worker {worker_id:02d}: running | {last_line}")
                    except Exception:
                        status_lines.append(f"  Worker {worker_id:02d}: running")
                elif ret == 0:
                    status_lines.append(f"  Worker {worker_id:02d}: done (0)")
                else:
                    status_lines.append(f"  Worker {worker_id:02d}: FAILED (exit={ret}) - see {log_file.name}")

            # 打印状态
            print(f"\r[{time.strftime('%H:%M:%S')}] 进度:", end=" ", flush=True)
            done_workers = sum(1 for _, p, _, _ in processes if p.poll() is not None)
            print(f"{done_workers}/{len(processes)} workers done", end="", flush=True)

            if all_done:
                print()
                break

            time.sleep(30)

    except KeyboardInterrupt:
        print("\n\n收到中断信号，停止所有worker...")
        for worker_id, proc, log_file, wf_names in processes:
            if proc.poll() is None:
                proc.terminate()
                print(f"  终止 Worker {worker_id:02d} (PID={proc.pid})")
        print("已终止所有worker")
        return

    # 最终状态
    print(f"\n{'='*80}")
    print("所有worker完成!")
    print(f"{'='*80}")
    for worker_id, proc, log_file, wf_names in processes:
        ret = proc.returncode
        if ret == 0:
            print(f"  Worker {worker_id:02d}: 成功 ({len(wf_names)} WF配置)")
        else:
            print(f"  Worker {worker_id:02d}: 失败 (exit={ret}), 日志: {log_file}")

    # 汇总结果
    print(f"\n汇总结果CSV: {output_base / 'grid_search_results.csv'}")
    print(f"详细日志目录: {log_dir}")


if __name__ == "__main__":
    main()
