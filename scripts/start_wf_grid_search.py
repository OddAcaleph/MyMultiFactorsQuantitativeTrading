#!/usr/bin/env python3
"""启动WF网格搜索的并行worker。

用法:
    python scripts/start_wf_grid_search.py --num-workers 6 --n-jobs-per-worker 8
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from wf_grid_search import (  # noqa: E402
    generate_wf_configs, WF_PARAM_GRID, BT_PARAM_GRID, BASE_OUTPUT,
)

WORKER_SCRIPT = Path(__file__).resolve().parent / "wf_grid_search_worker.py"


def main():
    parser = argparse.ArgumentParser(description="Start parallel WF grid search workers")
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--n-jobs-per-worker", type=int, default=8)
    parser.add_argument("--output", type=str, default="full")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--oos-only", action="store_true")
    args = parser.parse_args()

    # 生成WF配置列表
    if args.smoke:
        from wf_grid_search import SMOKE_WF_PARAM_GRID
        wf_grid = SMOKE_WF_PARAM_GRID
    else:
        wf_grid = WF_PARAM_GRID
    wf_grid["model_params"]["n_jobs"] = [args.n_jobs_per_worker]
    wf_configs = generate_wf_configs(wf_grid)

    output_base = BASE_OUTPUT / args.output
    log_dir = output_base / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 分配WF配置给workers
    worker_wfs = [[] for _ in range(args.num_workers)]
    for i, wf in enumerate(wf_configs):
        worker_wfs[i % args.num_workers].append(wf.name)

    print(f"WF配置总数: {len(wf_configs)}")
    print(f"Workers: {args.num_workers}")
    print(f"每worker核数: {args.n_jobs_per_worker}")
    print(f"输出目录: {output_base}")
    print()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT) + ":" + env.get("PYTHONPATH", "")
    env["PYTHONUNBUFFERED"] = "1"

    processes = []
    for wid in range(args.num_workers):
        wf_names = worker_wfs[wid]
        if not wf_names:
            continue

        log_file = log_dir / f"worker_{wid:02d}.log"
        cmd = [
            sys.executable, str(WORKER_SCRIPT),
            "--worker-id", str(wid),
            "--wf-names", ",".join(wf_names),
            "--n-jobs", str(args.n_jobs_per_worker),
            "--output", args.output,
        ]
        if args.cpu:
            cmd.append("--cpu")
        if args.smoke:
            cmd.append("--smoke")
        if args.oos_only:
            cmd.append("--oos-only")

        log_f = open(log_file, "a", buffering=1)
        proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT),
            stdout=log_f, stderr=subprocess.STDOUT,
            env=env,
        )
        processes.append((wid, proc, log_file, len(wf_names)))
        print(f"  Worker {wid:02d}: {len(wf_names)} WF配置, PID={proc.pid}, log={log_file.name}")

    print(f"\n所有worker已启动。监控日志: tail -f {log_dir}/worker_00.log")
    print(f"查看所有worker状态: {sys.executable} {__file__} --status --output {args.output}")


if __name__ == "__main__":
    main()
