"""WF网格搜索的worker进程。

每个worker串行处理分配给它的WF配置列表（训练+预测+回测）。
由 wf_grid_search_scheduler.py 调度。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from wf_grid_search import (  # noqa: E402
    generate_wf_configs, generate_bt_configs,
    WF_PARAM_GRID, BT_PARAM_GRID,
    SMOKE_WF_PARAM_GRID, SMOKE_BT_PARAM_GRID,
    BASE_OUTPUT,
    train_wf_config, run_backtest_for_config,
    extract_metrics, save_results_csv, get_scenario,
    DEFAULT_TRAIN_START, DEFAULT_TEST_START, DEFAULT_TEST_END,
    DEFAULT_BACKTEST_START, DEFAULT_OOS_START,
    SMOKE_TRAIN_START, SMOKE_TEST_START, SMOKE_TEST_END, SMOKE_BACKTEST_START,
)


def main():
    # 强制行缓冲输出，确保日志实时写入
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
    sys.stderr = sys.stdout

    parser = argparse.ArgumentParser(description="WF grid search worker")
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--wf-names", type=str, required=True,
                        help="Comma-separated WF config names to process")
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--output", type=str, default="full")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--oos-only", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args()

    worker_id = args.worker_id
    wf_names = [n.strip() for n in args.wf_names.split(",") if n.strip()]

    # 选择网格
    if args.smoke:
        wf_grid = SMOKE_WF_PARAM_GRID
        bt_grid = SMOKE_BT_PARAM_GRID
        train_start = SMOKE_TRAIN_START
        test_start = SMOKE_TEST_START
        test_end = SMOKE_TEST_END
        bt_start = SMOKE_BACKTEST_START
        bt_end = SMOKE_TEST_END
    else:
        wf_grid = WF_PARAM_GRID
        bt_grid = BT_PARAM_GRID
        train_start = DEFAULT_TRAIN_START
        test_start = DEFAULT_TEST_START
        test_end = DEFAULT_TEST_END
        if args.oos_only:
            bt_start = DEFAULT_OOS_START
        else:
            bt_start = DEFAULT_BACKTEST_START
        bt_end = DEFAULT_TEST_END

    # 覆盖n_jobs
    wf_grid["model_params"]["n_jobs"] = [args.n_jobs]

    output_base = BASE_OUTPUT / args.output
    output_base.mkdir(parents=True, exist_ok=True)

    # 生成所有配置，按name过滤
    all_wf_configs = generate_wf_configs(wf_grid)
    wf_configs = [w for w in all_wf_configs if w.name in wf_names]
    bt_configs = generate_bt_configs(bt_grid)

    print(f"[Worker {worker_id:02d}] 启动，处理 {len(wf_configs)}/{len(all_wf_configs)} 个WF配置")
    print(f"[Worker {worker_id:02d}] 回测配置数: {len(bt_configs)}")
    print(f"[Worker {worker_id:02d}] n_jobs per training: {args.n_jobs}")
    print(f"[Worker {worker_id:02d}] 输出目录: {output_base}")
    print()
    sys.stdout.flush()

    all_results = []
    t_total = time.time()

    for wf_idx, wf_cfg in enumerate(wf_configs):
        print(f"[Worker {worker_id:02d}] {'='*60}")
        print(f"[Worker {worker_id:02d}] WF [{wf_idx+1}/{len(wf_configs)}]: {wf_cfg.name}")
        print(f"[Worker {worker_id:02d}] {'='*60}")
        sys.stdout.flush()

        t0 = time.time()
        try:
            pred_path = train_wf_config(
                wf_cfg, output_base,
                prefer_gpu=not args.cpu,
                verbose=args.verbose,
                train_start=train_start,
                test_start=test_start,
                test_end=test_end,
            )
            print(f"[Worker {worker_id:02d}] 训练+预测完成，用时: {time.time()-t0:.0f}s")
        except Exception as e:
            import traceback
            print(f"[Worker {worker_id:02d}] 训练失败: {e}")
            print(f"[Worker {worker_id:02d}] {traceback.format_exc()}")
            sys.stdout.flush()
            continue
        sys.stdout.flush()

        # 回测
        print(f"[Worker {worker_id:02d}] 回测阶段 ({len(bt_configs)} 个配置):")
        sys.stdout.flush()

        for bt_idx, bt_cfg in enumerate(bt_configs):
            print(f"[Worker {worker_id:02d}]   [{bt_idx+1}/{len(bt_configs)}] {bt_cfg.name}...",
                  end=" ", flush=True)
            try:
                analysis = run_backtest_for_config(
                    pred_path, wf_cfg, bt_cfg, output_base,
                    start_time=bt_start, end_time=bt_end,
                )
                metrics = extract_metrics(analysis)

                row = {
                    "wf_name": wf_cfg.name,
                    "train_window_years": wf_cfg.train_window_years,
                    "step_years": wf_cfg.step_years,
                    "mode": wf_cfg.mode,
                    "label_name": wf_cfg.label_name,
                    "n_estimators": wf_cfg.n_estimators,
                    "max_depth": wf_cfg.max_depth,
                    "learning_rate": wf_cfg.learning_rate,
                    "scenario": get_scenario(bt_cfg.scenario_key)["short"],
                    "topk": bt_cfg.topk,
                    "n_drop": bt_cfg.n_drop,
                    "freq": bt_cfg.freq,
                    **metrics,
                }
                all_results.append(row)

                sharpe = metrics.get("sharpe", 0) or 0
                ret = metrics.get("total_return", 0) or 0
                mdd = metrics.get("max_drawdown", 0) or 0
                print(f"ret={ret*100:.2f}% sharpe={sharpe:.3f} mdd={mdd*100:.2f}%")
            except Exception as e:
                import traceback
                print(f"失败: {e}")
                print(f"[Worker {worker_id:02d}]   {traceback.format_exc()}")
                continue
            sys.stdout.flush()

        # 保存本worker的中间结果
        worker_csv = output_base / f"grid_search_results_worker{worker_id:02d}.csv"
        save_results_csv(all_results, worker_csv)

    print(f"\n[Worker {worker_id:02d}] 全部完成!")
    print(f"[Worker {worker_id:02d}] 总用时: {time.time()-t_total:.0f}s")
    print(f"[Worker {worker_id:02d}] 有效结果: {len(all_results)}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
