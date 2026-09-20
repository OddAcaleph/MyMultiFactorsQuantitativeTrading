"""Walk-Forward 训练参数 + 回测参数 联合网格搜索。

对WF训练参数（窗口长度、步进、标签、模型超参）和回测参数（场景、topk、n_drop、调仓频率）
进行联合网格搜索，找出全局最优配置。

支持：
- 训练参数网格 + 回测参数网格 的笛卡尔积
- 每个WF配置训练一次，多种回测配置复用预测结果
- 断点续跑（模型/预测/回测结果都缓存）
- smoke模式：小参数集快速验证
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from utils import load_walk_forward_config, resolve_path  # noqa: E402

# ============ 网格定义 ============

# WF训练参数网格
WF_PARAM_GRID = {
    "train_window_years": [5],
    "step_years": [1],
    "mode": ["rolling"],
    "label_name": ["label_rank_5d", "label_rank_10d", "label_rank_20d"],
    "model_params": {
        "n_estimators": [100, 200, 300],
        "max_depth": [6, 8, 10],
        "learning_rate": [0.01, 0.03, 0.05],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "min_child_weight": [50],
        "gamma": [0.1],
        "objective": ["reg:squarederror"],
        "tree_method": ["hist"],
        "random_state": [42],
        "n_jobs": [8],
    },
}

# Smoke test用的小网格
SMOKE_WF_PARAM_GRID = {
    "train_window_years": [5],
    "step_years": [1],
    "mode": ["rolling"],
    "label_name": ["label_rank_5d"],
    "model_params": {
        "n_estimators": [50],
        "max_depth": [6],
        "learning_rate": [0.05],
        "subsample": [0.8],
        "colsample_bytree": [0.8],
        "min_child_weight": [100],
        "gamma": [0.1],
        "objective": ["reg:squarederror"],
        "tree_method": ["hist"],
        "random_state": [42],
        "n_jobs": [4],
    },
}

# 回测场景
SCENARIOS = [
    {
        "key": "A_score_nofilter",
        "short": "A",
        "label": "score sell, no filter",
        "drop_criteria": "score",
        "filter_st": False,
        "filter_suspend": False,
        "min_avg_amount_20d": 0,
    },
    {
        "key": "B_return_nofilter",
        "short": "B",
        "label": "return sell, no filter",
        "drop_criteria": "return",
        "filter_st": False,
        "filter_suspend": False,
        "min_avg_amount_20d": 0,
    },
    {
        "key": "C_return_st_suspend",
        "short": "C",
        "label": "return sell + ST + suspend",
        "drop_criteria": "return",
        "filter_st": True,
        "filter_suspend": True,
        "min_avg_amount_20d": 0,
    },
    {
        "key": "D_return_st_suspend_liq50m",
        "short": "D",
        "label": "return sell + ST + suspend + liq50M",
        "drop_criteria": "return",
        "filter_st": True,
        "filter_suspend": True,
        "min_avg_amount_20d": 50000,
    },
]

# 回测参数网格
BT_PARAM_GRID = {
    "scenario_keys": ["D_return_st_suspend_liq50m"],  # 默认只跑D场景
    "topk": [10, 20, 30],
    "n_drop": [2, 3, 6],
    "freq": ["5d", "10d", "20d"],
}

SMOKE_BT_PARAM_GRID = {
    "scenario_keys": ["D_return_st_suspend_liq50m"],
    "topk": [10],
    "n_drop": [2],
    "freq": ["20d"],
}

# 时间范围
DEFAULT_TRAIN_START = "2000-01-01"
DEFAULT_TEST_START = "2005-01-01"
DEFAULT_TEST_END = "2026-07-14"
DEFAULT_BACKTEST_START = "2023-01-01"
DEFAULT_OOS_START = "2026-01-01"

# Smoke test用的时间范围（只跑最后2个窗口，快速验证）
SMOKE_TRAIN_START = "2019-01-01"
SMOKE_TEST_START = "2024-01-01"
SMOKE_TEST_END = "2026-07-14"
SMOKE_BACKTEST_START = "2025-01-01"

PRICE_PATH = str(PROJECT_ROOT / "data" / "cross_sectional_processd_data" / "wide_table_daily_bars")
BENCHMARK_PATH = "/opt/tiger/qyd/qlib_data_cn/features"
LOADER_CONFIG_PATH = str(PROJECT_ROOT / "conf" / "parquet_loader_config.json")
BASE_OUTPUT = PROJECT_ROOT / "output" / "wf_grid_search"


# ============ 数据类 ============

@dataclass
class WFConfig:
    """单个WF训练配置。"""
    train_window_years: int
    step_years: int
    mode: str
    label_name: str
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float
    min_child_weight: int
    gamma: float
    n_jobs: int = 8

    @property
    def name(self) -> str:
        lr_str = f"lr{self.learning_rate}".replace(".", "")
        return (
            f"wf_{self.mode}_tw{self.train_window_years}y_step{self.step_years}y_"
            f"{self.label_name}_est{self.n_estimators}_d{self.max_depth}_{lr_str}"
        )

    @property
    def model_params(self) -> dict:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "gamma": self.gamma,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "random_state": 42,
            "n_jobs": self.n_jobs,
        }


@dataclass
class BTConfig:
    """单个回测配置。"""
    scenario_key: str
    topk: int
    n_drop: int
    freq: str

    @property
    def name(self) -> str:
        return f"{self.scenario_key}_top{self.topk}_ndrop{self.n_drop}_{self.freq}"


# ============ 网格生成 ============

def generate_wf_configs(grid: dict) -> list[WFConfig]:
    """从WF参数网格生成所有WFConfig。"""
    mp = grid["model_params"]
    combos = list(product(
        grid["train_window_years"],
        grid["step_years"],
        grid["mode"],
        grid["label_name"],
        mp["n_estimators"],
        mp["max_depth"],
        mp["learning_rate"],
        mp["subsample"],
        mp["colsample_bytree"],
        mp["min_child_weight"],
        mp["gamma"],
        mp.get("n_jobs", [8]),
    ))
    return [
        WFConfig(
            train_window_years=tw,
            step_years=sy,
            mode=mode,
            label_name=label,
            n_estimators=ne,
            max_depth=md,
            learning_rate=lr,
            subsample=ss,
            colsample_bytree=cs,
            min_child_weight=mcw,
            gamma=g,
            n_jobs=nj,
        )
        for tw, sy, mode, label, ne, md, lr, ss, cs, mcw, g, nj in combos
    ]


def generate_bt_configs(grid: dict) -> list[BTConfig]:
    """从回测参数网格生成所有BTConfig。"""
    combos = list(product(
        grid["scenario_keys"],
        grid["topk"],
        grid["n_drop"],
        grid["freq"],
    ))
    configs = []
    for sk, tk, nd, fq in combos:
        if nd >= tk:
            continue
        configs.append(BTConfig(scenario_key=sk, topk=tk, n_drop=nd, freq=fq))
    return configs


def get_scenario(key: str) -> dict:
    for s in SCENARIOS:
        if s["key"] == key:
            return s
    raise ValueError(f"Unknown scenario: {key}")


# ============ WF训练 ============

def generate_windows(
    train_start: str,
    test_start: str,
    test_end: str,
    train_window_years: int,
    step_years: int,
    valid_ratio: float,
    mode: str = "rolling",
) -> list:
    """生成滚动窗口列表（复用run_walk_forward.py的逻辑）。"""
    from dataclasses import dataclass as _dc

    @_dc
    class _Win:
        idx: int
        train_start: str
        train_end: str
        valid_start: str
        valid_end: str
        test_start: str
        test_end: str

        @property
        def name(self):
            return f"wf{self.idx:02d}_{self.train_start[:4]}-{self.train_end[:4]}_test{self.test_start[:4]}"

    ts = pd.Timestamp(train_start)
    test_s = pd.Timestamp(test_start)
    test_e = pd.Timestamp(test_end)

    windows = []
    idx = 0

    if mode == "rolling":
        current_train_start = ts
        while True:
            current_train_end = current_train_start + pd.DateOffset(years=train_window_years) - pd.Timedelta(days=1)
            current_test_start = current_train_end + pd.Timedelta(days=1)
            current_test_end = current_test_start + pd.DateOffset(years=step_years) - pd.Timedelta(days=1)

            if current_test_start > test_e:
                break
            if current_test_end < test_s:
                current_train_start += pd.DateOffset(years=step_years)
                continue

            actual_test_start = max(current_test_start, test_s)
            actual_test_end = min(current_test_end, test_e)

            train_days = (current_train_end - current_train_start).days
            valid_days = int(train_days * valid_ratio)
            valid_start = current_train_end - pd.Timedelta(days=valid_days) + pd.Timedelta(days=1)
            train_end_for_split = valid_start - pd.Timedelta(days=1)

            windows.append(_Win(
                idx=idx,
                train_start=current_train_start.strftime("%Y-%m-%d"),
                train_end=train_end_for_split.strftime("%Y-%m-%d"),
                valid_start=valid_start.strftime("%Y-%m-%d"),
                valid_end=current_train_end.strftime("%Y-%m-%d"),
                test_start=actual_test_start.strftime("%Y-%m-%d"),
                test_end=actual_test_end.strftime("%Y-%m-%d"),
            ))
            idx += 1
            current_train_start += pd.DateOffset(years=step_years)

    elif mode == "expanding":
        current_train_end = ts + pd.DateOffset(years=train_window_years) - pd.Timedelta(days=1)
        while current_train_end < test_e:
            current_test_start = current_train_end + pd.Timedelta(days=1)
            current_test_end = current_test_start + pd.DateOffset(years=step_years) - pd.Timedelta(days=1)

            if current_test_start > test_e:
                break
            if current_test_end < test_s:
                current_train_end += pd.DateOffset(years=step_years)
                continue

            actual_test_start = max(current_test_start, test_s)
            actual_test_end = min(current_test_end, test_e)

            train_days = (current_train_end - ts).days
            valid_days = int(train_days * valid_ratio)
            valid_start = current_train_end - pd.Timedelta(days=valid_days) + pd.Timedelta(days=1)
            train_end_for_split = valid_start - pd.Timedelta(days=1)

            windows.append(_Win(
                idx=idx,
                train_start=ts.strftime("%Y-%m-%d"),
                train_end=train_end_for_split.strftime("%Y-%m-%d"),
                valid_start=valid_start.strftime("%Y-%m-%d"),
                valid_end=current_train_end.strftime("%Y-%m-%d"),
                test_start=actual_test_start.strftime("%Y-%m-%d"),
                test_end=actual_test_end.strftime("%Y-%m-%d"),
            ))
            idx += 1
            current_train_end += pd.DateOffset(years=step_years)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return windows


def train_wf_config(wf_cfg: WFConfig, output_base: Path, prefer_gpu: bool = False,
                    verbose: int = 0, instruments: str = "all",
                    train_start: str = DEFAULT_TRAIN_START,
                    test_start: str = DEFAULT_TEST_START,
                    test_end: str = DEFAULT_TEST_END) -> Path:
    """训练一个WF配置的所有窗口，返回完整预测文件路径。"""
    model_dir = output_base / "models" / wf_cfg.name
    pred_dir = output_base / "predictions" / wf_cfg.name
    model_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    full_pred_path = pred_dir / "walk_forward_full_pred.parquet"
    if full_pred_path.exists():
        print(f"  预测已存在: {full_pred_path}")
        return full_pred_path

    windows = generate_windows(
        train_start=train_start,
        test_start=test_start,
        test_end=test_end,
        train_window_years=wf_cfg.train_window_years,
        step_years=wf_cfg.step_years,
        valid_ratio=0.15,
        mode=wf_cfg.mode,
    )

    print(f"  共 {len(windows)} 个窗口")

    # 训练所有窗口（子进程模式避免内存泄漏）
    max_retries = 5
    for i, w in enumerate(windows):
        model_path = model_dir / f"{w.name}.json"
        if model_path.exists():
            continue
        print(f"  [{i+1}/{len(windows)}] 训练 {w.name}...", end=" ", flush=True)

        # 写临时配置文件
        tmp_config = {
            "loader_config_path": LOADER_CONFIG_PATH,
            "instruments": instruments,
            "prefer_gpu": prefer_gpu,
            "label_name": wf_cfg.label_name,
            "walk_forward": {
                "train_start": train_start,
                "train_end": "2025-12-31",
                "test_start": test_start,
                "test_end": test_end,
                "mode": wf_cfg.mode,
                "train_window_years": wf_cfg.train_window_years,
                "step_years": wf_cfg.step_years,
                "valid_ratio": 0.15,
                "model_params": wf_cfg.model_params,
            },
            "backtest": {"topk": 50, "n_drop": 5, "signal_delay": 1,
                         "open_cost": 0.0005, "close_cost": 0.0015,
                         "slippage": 0.001, "min_cost": 5.0, "account": 1000000},
            "output": {
                "model_dir": str(model_dir),
                "prediction_dir": str(pred_dir),
                "backtest_dir": str(output_base / "backtest" / wf_cfg.name),
            },
        }
        tmp_config_path = pred_dir / f"_tmp_config_{w.name}.json"
        with open(tmp_config_path, "w") as f:
            json.dump(tmp_config, f)

        env = os.environ.copy()
        env["PYTHONPATH"] = str(SRC_ROOT) + ":" + env.get("PYTHONPATH", "")
        env["MALLOC_TRIM_THRESHOLD_"] = "0"
        cmd = [
            sys.executable, str(PROJECT_ROOT / "scripts" / "run_walk_forward.py"),
            "--config", str(tmp_config_path),
            "--train-window", str(i),
            "--verbose", str(verbose),
        ]
        if not prefer_gpu:
            cmd.append("--cpu")

        success = False
        last_error = ""
        last_returncode = 0
        for attempt in range(max_retries):
            if attempt > 0:
                gc.collect()
                wait_sec = 10 * attempt
                print(f"重试 {attempt}/{max_retries-1} (等待{wait_sec}s)...", end=" ", flush=True)
                time.sleep(wait_sec)
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, env=env)
            if result.returncode == 0:
                success = True
                break
            last_error = result.stderr
            last_returncode = result.returncode

        tmp_config_path.unlink(missing_ok=True)

        if not success:
            print(f"失败! (已重试{max_retries}次, exit_code={last_returncode})")
            # 过滤掉Gym警告，显示真正的错误
            err_lines = [l for l in last_error.split("\n") if "gym" not in l.lower() and "gymnasium" not in l.lower()]
            filtered_stderr = "\n".join(err_lines).strip()
            if filtered_stderr:
                print(f"  stderr:\n{filtered_stderr[-3000:] if len(filtered_stderr) > 3000 else filtered_stderr}")
            else:
                print(f"  stderr为空 (可能是OOM被kill, exit_code={last_returncode})")
            raise RuntimeError(f"Training failed for {w.name} (exit_code={last_returncode})")
        print("done")

        # 训练完一个窗口后清理内存
        gc.collect()

    # 预测所有窗口
    all_preds = []
    for i, w in enumerate(windows):
        model_path = model_dir / f"{w.name}.json"
        if not model_path.exists():
            continue
        pred_path = pred_dir / f"{w.name}_pred.parquet"
        if pred_path.exists():
            all_preds.append(pd.read_parquet(pred_path))
            continue

        print(f"  预测 {w.name}...", end=" ", flush=True)
        from trainer.xgboost_inferencer import XGBoostInferencer
        inf = XGBoostInferencer(
            model_path=str(model_path),
            label_name=wf_cfg.label_name,
            instruments=instruments,
            start_time=w.test_start,
            end_time=w.test_end,
            loader_config_path=LOADER_CONFIG_PATH,
            prefer_gpu=False,
        )
        inf.build_dataset()
        pred_df = inf.predict(save=False)
        pred_df.to_parquet(pred_path)
        all_preds.append(pred_df)
        del inf
        gc.collect()
        print("done")

    if not all_preds:
        raise RuntimeError("No predictions available")

    all_pred_df = pd.concat(all_preds).sort_index()
    all_pred_df.to_parquet(full_pred_path)
    print(f"  完整预测已保存: {full_pred_path} ({all_pred_df.shape})")
    return full_pred_path


# ============ 回测 ============

def run_backtest_for_config(
    pred_path: Path,
    wf_cfg: WFConfig,
    bt_cfg: BTConfig,
    output_base: Path,
    start_time: str = DEFAULT_BACKTEST_START,
    end_time: str = DEFAULT_TEST_END,
) -> dict:
    """对单个WF配置 + 单个回测配置运行回测。"""
    scenario = get_scenario(bt_cfg.scenario_key)
    bt_dir = output_base / "backtest" / wf_cfg.name / bt_cfg.name
    bt_dir.mkdir(parents=True, exist_ok=True)

    analysis_path = bt_dir / "analysis.json"
    if analysis_path.exists():
        with open(analysis_path) as f:
            return json.load(f)

    from backtester.simple_backtester import SimpleBacktester

    strategy = {
        "topk": bt_cfg.topk,
        "n_drop": bt_cfg.n_drop,
        "drop_criteria": scenario["drop_criteria"],
        "filter_st": scenario["filter_st"],
        "filter_suspend": scenario["filter_suspend"],
        "min_avg_amount_20d": scenario["min_avg_amount_20d"],
    }

    bt = SimpleBacktester(
        prediction_path=str(pred_path),
        price_path=PRICE_PATH,
        benchmark_path=BENCHMARK_PATH,
        output_dir=str(bt_dir),
        start_time=start_time,
        end_time=end_time,
        account=1_000_000,
        benchmark="000300.SH",
        freq=bt_cfg.freq,
        signal_delay=1,
        price_adjustment="qfq",
        strategy=strategy,
        exchange_kwargs={
            "freq": "day",
            "deal_price": "open",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5.0,
            "slippage": 0.001,
            "limit_threshold": 0.095,
            "lot_size": 100,
            "forbid_buy_limit_up": True,
            "forbid_buy_limit_down": True,
            "forbid_sell_limit_down": True,
        },
        columns={"score": "pred"},
        save_positions=False,
    )

    result = bt.run(save=True)
    return result["analysis"]


# ============ 结果汇总 ============

def extract_metrics(analysis: dict) -> dict:
    """从回测结果中提取关键指标。"""
    s = analysis.get("summary", {})
    r = analysis.get("return", {})
    risk = analysis.get("risk", {})
    return {
        "total_return": s.get("total_return"),
        "benchmark_return": s.get("benchmark_total_return"),
        "annualized_return": r.get("annualized_return"),
        "annualized_vol": r.get("annualized_volatility"),
        "sharpe": r.get("sharpe"),
        "sortino": r.get("sortino"),
        "max_drawdown": r.get("max_drawdown"),
        "calmar": r.get("calmar"),
        "win_rate": r.get("win_rate"),
        "excess_return": risk.get("excess_total_return"),
        "excess_annual": risk.get("excess_annualized_return"),
        "information_ratio": risk.get("information_ratio"),
        "alpha": risk.get("alpha"),
        "beta": risk.get("beta"),
        "mean_turnover": s.get("mean_turnover"),
        "total_cost": s.get("total_cost"),
        "trade_count": s.get("trade_count"),
        "trading_days": s.get("trading_days"),
    }


def save_results_csv(results: list[dict], output_path: Path):
    """保存结果到CSV。"""
    if not results:
        return
    keys = list(results[0].keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


def train_wf_config_worker(args_tuple):
    """多进程worker：训练一个WF配置，返回(pred_path, wf_name, error_msg)。

    为了支持pickle，用tuple传参。
    """
    wf_cfg_dict, output_base_str, prefer_gpu, verbose, train_start, test_start, test_end = args_tuple

    # 重建WFConfig
    wf_cfg = WFConfig(**wf_cfg_dict)
    output_base = Path(output_base_str)

    try:
        pred_path = train_wf_config(
            wf_cfg, output_base,
            prefer_gpu=prefer_gpu,
            verbose=verbose,
            train_start=train_start,
            test_start=test_start,
            test_end=test_end,
        )
        return (str(pred_path), wf_cfg.name, None)
    except Exception as e:
        import traceback
        return (None, wf_cfg.name, f"{e}\n{traceback.format_exc()}")


def backtest_one_wf_worker(args_tuple):
    """多进程worker：对一个WF配置跑所有回测配置，返回结果列表。"""
    wf_cfg_dict, bt_cfg_dicts, pred_path_str, output_base_str, bt_start, bt_end = args_tuple

    wf_cfg = WFConfig(**wf_cfg_dict)
    bt_configs = [BTConfig(**d) for d in bt_cfg_dicts]
    pred_path = Path(pred_path_str)
    output_base = Path(output_base_str)

    results = []
    for bt_cfg in bt_configs:
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
            results.append(row)
        except Exception as e:
            import traceback
            print(f"  [ERROR] {wf_cfg.name} / {bt_cfg.name}: {e}", flush=True)
            print(f"  {traceback.format_exc()}", flush=True)
    return results


# ============ 主流程 ============

def main():
    parser = argparse.ArgumentParser(description="WF training + backtest grid search")
    parser.add_argument("--smoke", action="store_true", help="Smoke test with tiny grid")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, use existing models")
    parser.add_argument("--skip-backtest", action="store_true", help="Skip backtest phase")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--verbose", type=int, default=0, help="XGBoost verbose level")
    parser.add_argument("--oos-only", action="store_true",
                        help="Only evaluate on OOS period (2026) instead of full backtest period")
    parser.add_argument("--output", type=str, default=None, help="Custom output directory name")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Number of parallel WF configs to train (default: 1)")
    parser.add_argument("--n-jobs-per-worker", type=int, default=8,
                        help="XGBoost n_jobs per worker (default: 8)")
    args = parser.parse_args()

    # 选择网格
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

    output_base = BASE_OUTPUT / output_suffix
    output_base.mkdir(parents=True, exist_ok=True)

    wf_configs = generate_wf_configs(wf_grid)
    bt_configs = generate_bt_configs(bt_grid)

    # 覆盖每个WF配置的n_jobs
    for wf in wf_configs:
        wf.n_jobs = args.n_jobs_per_worker
    # 同步更新网格定义中的n_jobs
    wf_grid["model_params"]["n_jobs"] = [args.n_jobs_per_worker]

    num_workers = args.num_workers

    print("=" * 80)
    print("Walk-Forward 训练 + 回测 联合网格搜索")
    print("=" * 80)
    print(f"WF训练配置数: {len(wf_configs)}")
    print(f"回测配置数: {len(bt_configs)}")
    print(f"总组合数: {len(wf_configs) * len(bt_configs)}")
    print(f"并行WF workers: {num_workers}")
    print(f"每个worker CPU核数: {args.n_jobs_per_worker}")
    print(f"总使用核数: ~{num_workers * args.n_jobs_per_worker}")
    print(f"输出目录: {output_base}")
    print()

    # 打印WF配置
    print("WF训练配置:")
    for i, wf in enumerate(wf_configs):
        print(f"  [{i+1}] {wf.name}")
    print()
    print("回测配置:")
    for i, bt in enumerate(bt_configs):
        print(f"  [{i+1}] {bt.name}")
    print()

    # 确定回测时间范围和训练时间范围
    if args.smoke:
        train_start = SMOKE_TRAIN_START
        test_start = SMOKE_TEST_START
        test_end = SMOKE_TEST_END
        bt_start = SMOKE_BACKTEST_START
        bt_end = SMOKE_TEST_END
        print(f"训练时间范围: {train_start} ~ {test_end}")
        print(f"回测时间范围: {bt_start} ~ {bt_end}")
    elif args.oos_only:
        train_start = DEFAULT_TRAIN_START
        test_start = DEFAULT_TEST_START
        test_end = DEFAULT_TEST_END
        bt_start = DEFAULT_OOS_START
        bt_end = DEFAULT_TEST_END
        print(f"回测时间范围 (OOS): {bt_start} ~ {bt_end}")
    else:
        train_start = DEFAULT_TRAIN_START
        test_start = DEFAULT_TEST_START
        test_end = DEFAULT_TEST_END
        bt_start = DEFAULT_BACKTEST_START
        bt_end = DEFAULT_TEST_END
        print(f"回测时间范围: {bt_start} ~ {bt_end}")
    print()

    all_results = []
    t_total = time.time()

    # ========== 阶段1: 并行训练所有WF配置 ==========
    if not args.skip_train:
        print(f"\n{'='*70}")
        print(f"阶段1: 并行训练WF配置 ({num_workers} workers)")
        print(f"{'='*70}")

        # 过滤掉已经有预测的配置（断点续跑）
        wf_to_train = []
        for wf_cfg in wf_configs:
            pred_path = output_base / "predictions" / wf_cfg.name / "walk_forward_full_pred.parquet"
            if pred_path.exists():
                print(f"  [跳过] {wf_cfg.name} (预测已存在)")
            else:
                wf_to_train.append(wf_cfg)

        print(f"  需训练: {len(wf_to_train)}/{len(wf_configs)} 个WF配置")

        if wf_to_train:
            # 准备worker参数
            worker_args = [
                (asdict(wf), str(output_base), not args.cpu, args.verbose,
                 train_start, test_start, test_end)
                for wf in wf_to_train
            ]

            completed = 0
            failed = 0
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = {executor.submit(train_wf_config_worker, arg): arg[0]["label_name"] + "_" + str(arg[0]["n_estimators"]) for arg in worker_args}
                for future in as_completed(futures):
                    completed += 1
                    pred_path, wf_name, error = future.result()
                    if error:
                        failed += 1
                        print(f"  [{completed}/{len(wf_to_train)}] 失败: {wf_name}")
                        print(f"    {error.split(chr(10))[0]}")
                    else:
                        print(f"  [{completed}/{len(wf_to_train)}] 完成: {wf_name}")

            print(f"\n训练阶段完成: 成功{len(wf_to_train)-failed}/{len(wf_to_train)}, 失败{failed}")

    # ========== 阶段2: 回测 ==========
    if args.skip_backtest:
        pass
    else:
        print(f"\n{'='*70}")
        print(f"阶段2: 回测 (所有WF配置 × 所有回测配置)")
        print(f"{'='*70}")

        # 收集所有有预测结果的WF配置
        wf_with_preds = []
        for wf_cfg in wf_configs:
            pred_path = output_base / "predictions" / wf_cfg.name / "walk_forward_full_pred.parquet"
            if pred_path.exists():
                wf_with_preds.append((wf_cfg, pred_path))
            else:
                print(f"  [跳过] {wf_cfg.name} (无预测结果)")

        print(f"  可回测WF配置: {len(wf_with_preds)}/{len(wf_configs)}")
        print(f"  回测配置数: {len(bt_configs)}")
        print(f"  总回测组合: {len(wf_with_preds) * len(bt_configs)}")

        # 串行回测（回测本身很快，串行足够）
        for wf_idx, (wf_cfg, pred_path) in enumerate(wf_with_preds):
            print(f"\n  [{wf_idx+1}/{len(wf_with_preds)}] {wf_cfg.name}")
            print(f"    预测文件: {pred_path}")

            bt_idx = 0
            for bt_cfg in bt_configs:
                bt_idx += 1
                print(f"    [{bt_idx}/{len(bt_configs)}] {bt_cfg.name}...", end=" ", flush=True)
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
                    print(f"失败: {e}")
                    continue

            # 保存中间结果
            save_results_csv(all_results, output_base / "grid_search_results.csv")

    # 最终结果
    print(f"\n{'='*80}")
    print("搜索完成!")
    print(f"总用时: {time.time()-t_total:.0f}s")
    print(f"有效结果数: {len(all_results)}")
    print(f"结果已保存: {output_base / 'grid_search_results.csv'}")

    if all_results:
        # 按夏普排序
        sorted_by_sharpe = sorted(all_results, key=lambda x: x.get("sharpe") or 0, reverse=True)
        print(f"\nTop 10 by Sharpe:")
        print(f"  {'#':<3} {'WF配置':<50} {'场景':<4} {'topk':<5} {'ndrop':<5} {'freq':<5} "
              f"{'收益':>8} {'夏普':>7} {'最大回撤':>9} {'Calmar':>7}")
        print(f"  {'-'*100}")
        for i, r in enumerate(sorted_by_sharpe[:10]):
            print(f"  {i+1:<3} {r['wf_name']:<50} {r['scenario']:<4} "
                  f"{r['topk']:<5} {r['n_drop']:<5} {r['freq']:<5} "
                  f"{(r.get('total_return') or 0)*100:>7.2f}% "
                  f"{r.get('sharpe') or 0:>7.3f} "
                  f"{(r.get('max_drawdown') or 0)*100:>8.2f}% "
                  f"{r.get('calmar') or 0:>7.3f}")

        # 按收益排序
        sorted_by_return = sorted(all_results, key=lambda x: x.get("total_return") or 0, reverse=True)
        print(f"\nTop 10 by Total Return:")
        print(f"  {'#':<3} {'WF配置':<50} {'场景':<4} {'topk':<5} {'ndrop':<5} {'freq':<5} "
              f"{'收益':>8} {'夏普':>7} {'最大回撤':>9} {'Calmar':>7}")
        print(f"  {'-'*100}")
        for i, r in enumerate(sorted_by_return[:10]):
            print(f"  {i+1:<3} {r['wf_name']:<50} {r['scenario']:<4} "
                  f"{r['topk']:<5} {r['n_drop']:<5} {r['freq']:<5} "
                  f"{(r.get('total_return') or 0)*100:>7.2f}% "
                  f"{r.get('sharpe') or 0:>7.3f} "
                  f"{(r.get('max_drawdown') or 0)*100:>8.2f}% "
                  f"{r.get('calmar') or 0:>7.3f}")

    # 保存配置信息
    config_info = {
        "wf_grid": wf_grid,
        "bt_grid": bt_grid,
        "num_wf_configs": len(wf_configs),
        "num_bt_configs": len(bt_configs),
        "total_combos": len(wf_configs) * len(bt_configs),
        "train_start": train_start,
        "test_start": test_start,
        "test_end": test_end,
        "backtest_start": bt_start,
        "backtest_end": bt_end,
        "smoke": args.smoke,
    }
    with open(output_base / "search_config.json", "w") as f:
        json.dump(config_info, f, indent=2, default=str)


if __name__ == "__main__":
    main()
