"""Walk-forward 训练与回测脚本。

使用滚动窗口训练模型，每个窗口用过去N年数据训练，预测下一个step窗口。
支持：
- 可配置训练窗口长度、步进长度
- 自动划分train/valid（valid_ratio）
- 每个窗口保存模型和预测结果
- 拼接所有窗口的预测结果进行统一回测
- 分年度输出回测指标
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from utils import load_walk_forward_config, resolve_path  # noqa: E402
from trainer.xgboost_trainer import XGBoostTrainer  # noqa: E402
from backtester.simple_backtester import SimpleBacktester  # noqa: E402


@dataclass
class WindowSpec:
    idx: int
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str

    @property
    def name(self) -> str:
        return f"wf{self.idx:02d}_{self.train_start[:4]}-{self.train_end[:4]}_test{self.test_start[:4]}"


def generate_windows(
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
    train_window_years: int,
    step_years: int,
    valid_ratio: float,
    mode: str = "rolling",
) -> list[WindowSpec]:
    """生成滚动训练窗口列表。

    rolling模式：从train_start开始，每次滚动step_years年。
    - 第一个窗口：train=[train_start, train_start+train_window_years-1day], test紧随其后
    - 一直滚动到test_end

    expanding模式：训练起点固定为train_start，终点逐年后移。
    """
    ts = pd.Timestamp(train_start)
    te = pd.Timestamp(train_end)
    test_s = pd.Timestamp(test_start)
    test_e = pd.Timestamp(test_end)

    windows = []
    idx = 0

    if mode == "rolling":
        # 从train_start开始滚动，找到第一个test >= test_start的窗口
        current_train_start = ts
        while True:
            current_train_end = current_train_start + pd.DateOffset(years=train_window_years) - pd.Timedelta(days=1)
            current_test_start = current_train_end + pd.Timedelta(days=1)
            current_test_end = current_test_start + pd.DateOffset(years=step_years) - pd.Timedelta(days=1)

            if current_test_start > test_e:
                break
            if current_test_end < test_s:
                # 这个窗口的test在test_start之前，跳过（不训练）
                current_train_start += pd.DateOffset(years=step_years)
                continue

            # 裁剪到test范围
            actual_test_start = max(current_test_start, test_s)
            actual_test_end = min(current_test_end, test_e)

            # valid：train窗口末尾的valid_ratio
            train_days = (current_train_end - current_train_start).days
            valid_days = int(train_days * valid_ratio)
            valid_start = current_train_end - pd.Timedelta(days=valid_days) + pd.Timedelta(days=1)
            train_end_for_split = valid_start - pd.Timedelta(days=1)

            windows.append(WindowSpec(
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

            windows.append(WindowSpec(
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


def train_one_window(
    window: WindowSpec,
    label_name: str,
    model_params: dict,
    loader_config_path: str,
    instruments: str,
    prefer_gpu: bool,
    model_dir: Path,
    verbose: int,
) -> Path:
    """训练一个窗口的模型，返回模型文件路径。"""
    model_path = model_dir / f"{window.name}.json"
    if model_path.exists():
        print(f"  模型已存在，跳过: {model_path.name}", flush=True)
        return model_path

    print(f"  训练窗口 {window.name}: "
          f"train={window.train_start}~{window.train_end}, "
          f"valid={window.valid_start}~{window.valid_end}", flush=True)

    trainer = XGBoostTrainer(
        label_name=label_name,
        instruments=instruments,
        prefer_gpu=prefer_gpu,
        loader_config_path=loader_config_path,
        segments={
            "train": (window.train_start, window.train_end),
            "valid": (window.valid_start, window.valid_end),
        },
        model_params=model_params,
        model_path=str(model_path),
    )
    trainer.fit(verbose=verbose)
    trainer.save_model()

    del trainer
    gc.collect()

    print(f"  模型已保存: {model_path}", flush=True)
    return model_path


def predict_one_window(
    window: WindowSpec,
    model_path: Path,
    label_name: str,
    loader_config_path: str,
    instruments: str,
    pred_dir: Path,
) -> pd.DataFrame:
    """对一个窗口的test区间进行预测，返回pred_df。"""
    pred_path = pred_dir / f"{window.name}_pred.parquet"
    if pred_path.exists():
        print(f"  预测已存在，加载: {pred_path.name}", flush=True)
        return pd.read_parquet(pred_path)

    print(f"  预测窗口 {window.name}: test={window.test_start}~{window.test_end}", flush=True)

    from trainer.xgboost_inferencer import XGBoostInferencer

    inf = XGBoostInferencer(
        model_path=str(model_path),
        label_name=label_name,
        instruments=instruments,
        start_time=window.test_start,
        end_time=window.test_end,
        loader_config_path=loader_config_path,
        prefer_gpu=False,
    )
    inf.build_dataset()
    pred_df = inf.predict(save=False)

    pred_df.to_parquet(pred_path)
    print(f"  预测已保存: {pred_path}", flush=True)
    return pred_df


def run_backtest(
    all_preds: pd.DataFrame,
    bt_config: dict,
    price_path: str,
    output_dir: Path,
) -> dict:
    """对拼接后的预测结果进行回测。"""
    bt = SimpleBacktester(
        prediction_path="dummy",
        start_time=all_preds.index.get_level_values("datetime").min().strftime("%Y-%m-%d"),
        end_time=all_preds.index.get_level_values("datetime").max().strftime("%Y-%m-%d"),
        price_path=price_path,
        strategy={
            "topk": bt_config["topk"],
            "n_drop": bt_config["n_drop"],
            "signal_delay": bt_config["signal_delay"],
        },
        exchange_kwargs={
            "open_cost": bt_config["open_cost"],
            "close_cost": bt_config["close_cost"],
            "slippage": bt_config["slippage"],
            "min_cost": bt_config["min_cost"],
        },
        account=bt_config["account"],
        benchmark="market_mean",
        save_positions=False,
        output_dir=str(output_dir),
    )
    result = bt.run(save=False, pred_df=all_preds)
    return result


def calc_yearly_metrics(report_df: pd.DataFrame) -> pd.DataFrame:
    """计算分年度收益指标。"""
    report_df = report_df.copy()
    report_df["year"] = report_df.index.year

    yearly = []
    for year, group in report_df.groupby("year"):
        if len(group) < 5:
            continue
        start_val = group["account_value"].iloc[0]
        end_val = group["account_value"].iloc[-1]
        total_ret = end_val / start_val - 1.0

        daily_rets = group["account_value"].pct_change().dropna()
        sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0

        roll_max = group["account_value"].cummax()
        drawdown = (group["account_value"] - roll_max) / roll_max
        max_dd = drawdown.min()

        yearly.append({
            "year": year,
            "total_return": total_ret,
            "annual_return": total_ret * 252 / len(group),
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "trading_days": len(group),
        })

    return pd.DataFrame(yearly)


def main():
    parser = argparse.ArgumentParser(description="Walk-forward training and backtesting")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "conf" / "walk_forward_config.json"),
                        help="Walk-forward config path")
    parser.add_argument("--verbose", default="0", help="XGBoost verbose level")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, use existing models")
    parser.add_argument("--skip-predict", action="store_true", help="Skip prediction, use existing predictions")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")
    parser.add_argument("--train-window", type=int, default=None,
                        help="Train only the specified window index (for subprocess mode)")
    args = parser.parse_args()

    config = load_walk_forward_config(args.config)
    wf = config["walk_forward"]
    bt_cfg = config["backtest"]
    out_cfg = config["output"]

    label_name = config["label_name"]
    loader_config_path = resolve_path(config["loader_config_path"])
    instruments = config["instruments"]
    prefer_gpu = not args.cpu and config.get("prefer_gpu", False)
    model_params = wf["model_params"]
    verbose = int(args.verbose) if args.verbose.isdigit() else 0

    model_dir = resolve_path(out_cfg["model_dir"])
    pred_dir = resolve_path(out_cfg["prediction_dir"])
    bt_dir = resolve_path(out_cfg["backtest_dir"])
    for d in (model_dir, pred_dir, bt_dir):
        Path(d).mkdir(parents=True, exist_ok=True)

    price_path = str(PROJECT_ROOT / "data" / "processd_data" / "wide_table_daily_bars")

    # 生成窗口
    windows = generate_windows(
        train_start=wf["train_start"],
        train_end=wf["train_end"],
        test_start=wf["test_start"],
        test_end=wf["test_end"],
        train_window_years=int(wf["train_window_years"]),
        step_years=int(wf["step_years"]),
        valid_ratio=float(wf["valid_ratio"]),
        mode=wf.get("mode", "rolling"),
    )

    # 单窗口训练模式（子进程调用）
    if args.train_window is not None:
        w = windows[args.train_window]
        train_one_window(w, label_name, model_params, str(loader_config_path), instruments, prefer_gpu, model_dir, verbose)
        return

    print("=" * 80, flush=True)
    print("Walk-Forward 训练与回测", flush=True)
    print(f"标签: {label_name}", flush=True)
    print(f"模式: {wf.get('mode', 'rolling')}, 训练窗口: {wf['train_window_years']}年, 步进: {wf['step_years']}年, valid比例: {wf['valid_ratio']}", flush=True)
    print("=" * 80, flush=True)

    print(f"\n共生成 {len(windows)} 个滚动窗口:", flush=True)
    for w in windows:
        print(f"  {w.name}: train={w.train_start}~{w.valid_end}, test={w.test_start}~{w.test_end}", flush=True)

    # 训练
    if not args.skip_train:
        print(f"\n{'='*60}", flush=True)
        print("阶段1: 训练模型 (子进程模式，避免内存泄漏)", flush=True)
        print(f"{'='*60}", flush=True)
        t0 = time.time()
        for i, w in enumerate(windows):
            print(f"\n[{i+1}/{len(windows)}]", flush=True)
            model_path = model_dir / f"{w.name}.json"
            if model_path.exists():
                print(f"  模型已存在，跳过: {model_path.name}", flush=True)
                continue
            import subprocess
            import os
            env = os.environ.copy()
            env["PYTHONPATH"] = str(SRC_ROOT) + ":" + env.get("PYTHONPATH", "")
            cmd = [
                sys.executable, str(Path(__file__)),
                "--config", str(args.config),
                "--train-window", str(i),
                "--verbose", str(verbose),
            ]
            if args.cpu:
                cmd.append("--cpu")
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, env=env)
            if result.returncode != 0:
                print(f"  训练失败: {w.name}", flush=True)
                print(f"  stderr: {result.stderr[-1000:]}", flush=True)
                raise RuntimeError(f"Training failed for {w.name}")
            for line in result.stdout.strip().split('\n'):
                if line.strip() and not line.startswith('Gym') and 'gymnasium' not in line:
                    print(f"  {line.strip()}", flush=True)
        print(f"\n训练完成，用时: {time.time()-t0:.0f}s", flush=True)
    else:
        print("\n跳过训练阶段", flush=True)

    # 预测
    print(f"\n{'='*60}", flush=True)
    print("阶段2: 滚动预测", flush=True)
    print(f"{'='*60}", flush=True)

    all_preds = []
    t0 = time.time()
    for i, w in enumerate(windows):
        print(f"\n[{i+1}/{len(windows)}]", flush=True)
        model_path = model_dir / f"{w.name}.json"
        if not model_path.exists():
            print(f"  模型不存在，跳过: {model_path.name}", flush=True)
            continue
        pred_df = predict_one_window(w, model_path, label_name, str(loader_config_path), instruments, pred_dir)
        all_preds.append(pred_df)

    if not all_preds:
        print("没有可用的预测结果，退出", flush=True)
        return

    all_pred_df = pd.concat(all_preds).sort_index()
    print(f"\n预测拼接完成: {all_pred_df.shape}, 用时: {time.time()-t0:.0f}s", flush=True)
    print(f"时间范围: {all_pred_df.index.get_level_values('datetime').min()} ~ {all_pred_df.index.get_level_values('datetime').max()}", flush=True)

    # 保存完整预测
    full_pred_path = pred_dir / "walk_forward_full_pred.parquet"
    all_pred_df.to_parquet(full_pred_path)
    print(f"完整预测已保存: {full_pred_path}", flush=True)

    # 回测
    print(f"\n{'='*60}", flush=True)
    print("阶段3: 回测", flush=True)
    print(f"{'='*60}", flush=True)

    result = run_backtest(all_pred_df, bt_cfg, price_path, bt_dir)
    summary = result["analysis"]["summary"]
    report_df = result["report"]

    print(f"\n整体结果:", flush=True)
    print(f"  总收益: {summary['total_return']*100:.2f}%", flush=True)
    print(f"  基准收益: {summary['benchmark_total_return']*100:.2f}%", flush=True)
    print(f"  日均换手: {summary['mean_turnover']*100:.2f}%", flush=True)
    print(f"  总成本: {summary['total_cost']*100:.2f}%", flush=True)
    print(f"  总交易次数: {summary['trade_count']}", flush=True)

    # 计算整体年化、夏普、最大回撤
    daily_rets = report_df["account_value"].pct_change().dropna()
    trading_days = summary["trading_days"]
    annual_return = (1 + summary["total_return"]) ** (252 / trading_days) - 1 if trading_days > 0 else 0
    sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0
    roll_max = report_df["account_value"].cummax()
    max_drawdown = ((report_df["account_value"] - roll_max) / roll_max).min()
    print(f"  年化收益: {annual_return*100:.2f}%", flush=True)
    print(f"  夏普比率: {sharpe:.2f}", flush=True)
    print(f"  最大回撤: {max_drawdown*100:.2f}%", flush=True)

    # 分年度
    yearly = calc_yearly_metrics(report_df)
    print(f"\n分年度结果:", flush=True)
    print(f"  {'年份':<6} {'收益':>10} {'年化':>10} {'夏普':>8} {'最大回撤':>10} {'交易日':>8}", flush=True)
    print(f"  {'-'*60}", flush=True)
    for _, row in yearly.iterrows():
        print(f"  {int(row['year']):<6} {row['total_return']*100:>+9.2f}% {row['annual_return']*100:>+9.2f}% "
              f"{row['sharpe']:>7.2f} {row['max_drawdown']*100:>+9.2f}% {int(row['trading_days']):>8}", flush=True)

    # 保存结果
    summary_path = bt_dir / "walk_forward_summary.json"
    summary_out = {
        "label_name": label_name,
        "train_window_years": wf["train_window_years"],
        "step_years": wf["step_years"],
        "model_params": model_params,
        "windows": len(windows),
        "overall": {
            "total_return": float(summary["total_return"]),
            "benchmark_total_return": float(summary["benchmark_total_return"]),
            "annual_return": float(annual_return),
            "sharpe": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "mean_turnover": float(summary["mean_turnover"]),
            "total_cost": float(summary["total_cost"]),
            "trade_count": int(summary["trade_count"]),
            "trading_days": int(trading_days),
        },
        "yearly": yearly.to_dict(orient="records"),
    }
    with open(summary_path, "w") as f:
        json.dump(summary_out, f, indent=2, default=str)
    print(f"\n结果已保存: {summary_path}", flush=True)

    # 保存report
    report_path = bt_dir / "walk_forward_report.csv"
    report_df.to_csv(report_path)
    print(f"日报表已保存: {report_path}", flush=True)


if __name__ == "__main__":
    main()
