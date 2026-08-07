"""Walk-Forward 完整回测：拼接所有窗口预测结果，进行统一回测。

输出：
- 全时段回测净值曲线
- 分年度收益/夏普/最大回撤
- 每个窗口的独立回测指标

支持完整的 A 股回测约束：
1. 股票池过滤：ST/*ST、停牌、新股60日、涨跌停
2. 流动性约束：20日日均成交额门槛、单次买入上限
3. 行业约束：申万一级行业数量/市值上限
4. 个股约束：单只权重上限
5. 换手约束：违反约束顺延下一只
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from trainer.walk_forward_trainer import WalkForwardTrainer  # noqa: E402
from backtester.simple_backtester import SimpleBacktester  # noqa: E402


def build_industry_map_from_wide_table(price_path: str | Path) -> dict[str, str]:
    """从 wide_table_daily_bars 的最新数据构建 instrument -> L1行业 映射。"""
    price_path = Path(price_path)
    # 找最新的一个 parquet 文件
    latest_file = None
    latest_date = None
    for f in sorted(price_path.rglob("*.parquet")):
        try:
            d = pd.to_datetime(f.stem, format="%Y%m%d")
            if latest_date is None or d > latest_date:
                latest_date = d
                latest_file = f
        except ValueError:
            continue

    if latest_file is None:
        print("Warning: no parquet files found for industry map", flush=True)
        return {}

    df = pd.read_parquet(latest_file)
    l1_cols = [c for c in df.columns if c.startswith("L1_")]
    if not l1_cols:
        print("Warning: no L1_ industry columns found", flush=True)
        return {}

    industry_map: dict[str, str] = {}
    for _, row in df.iterrows():
        ts_code = row.get("ts_code", "")
        if not ts_code:
            continue
        for col in l1_cols:
            if row[col] == 1:
                industry_map[ts_code] = col.replace("L1_", "")
                break

    print(f"Built industry map with {len(industry_map)} stocks, "
          f"{len(set(industry_map.values()))} industries", flush=True)
    return industry_map


def run_full_backtest(
    config_path: str = "conf/walk_forward_config.json",
    price_path: str | None = None,
    save: bool = True,
    enable_all_constraints: bool = True,
) -> dict:
    trainer = WalkForwardTrainer(config_path=config_path)
    trainer.generate_windows()

    all_preds = trainer.collect_all_predictions()
    print(f"Total prediction rows: {len(all_preds):,}", flush=True)
    print(f"Date range: {all_preds.index.get_level_values('datetime').min().date()} "
          f"~ {all_preds.index.get_level_values('datetime').max().date()}", flush=True)

    wf_cfg = trainer.config.get("walk_forward", {})
    bt_cfg = trainer.config.get("backtest", {})
    output_cfg = trainer.config.get("output", {})

    if price_path is None:
        loader_cfg = trainer.loader_config
        price_path = str(loader_cfg.get("daily_bars_dir", ""))

    # 构建行业映射
    industry_map = build_industry_map_from_wide_table(price_path) if enable_all_constraints else {}

    backtest_dir = Path(output_cfg.get("backtest_dir", "output/walk_forward/backtest"))
    if enable_all_constraints:
        backtest_dir = backtest_dir.parent / (backtest_dir.name + "_full_constraints")
    if save:
        backtest_dir.mkdir(parents=True, exist_ok=True)

    start_date = all_preds.index.get_level_values("datetime").min().strftime("%Y-%m-%d")
    end_date = all_preds.index.get_level_values("datetime").max().strftime("%Y-%m-%d")

    # 约束参数
    strategy_params = {
        "topk": bt_cfg["topk"],
        "n_drop": bt_cfg["n_drop"],
        "signal_delay": bt_cfg["signal_delay"],
    }

    if enable_all_constraints:
        strategy_params.update({
            # 1. 股票池过滤
            "filter_st": True,
            "filter_suspend": True,
            "filter_new_stock_days": 60,
            # 涨跌停过滤已在 exchange_kwargs 中默认开启

            # 2. 流动性约束（amount 单位：万元）
            "min_avg_amount_20d": bt_cfg.get("min_avg_amount_20d", 5000),  # 5000万 = 5000万元
            "max_buy_pct_of_avg_amount": 0.05,  # 单次买入不超过日均成交额5%

            # 3. 行业约束
            "max_industry_count": 10,  # 单行业持仓不超过10只
            "max_industry_weight": 0.20,  # 单行业市值占比不超过20%

            # 4. 个股约束
            "max_single_weight": 0.03,  # 单只权重不超过3%
        })

    bt = SimpleBacktester(
        prediction_path="dummy",
        start_time=start_date,
        end_time=end_date,
        price_path=price_path,
        strategy=strategy_params,
        exchange_kwargs={
            "open_cost": bt_cfg["open_cost"],
            "close_cost": bt_cfg["close_cost"],
            "slippage": bt_cfg["slippage"],
            "min_cost": bt_cfg["min_cost"],
            "forbid_buy_limit_up": True,
            "forbid_sell_limit_down": True,
            "board_specific_limits": True,
        },
        account=bt_cfg["account"],
        benchmark="market_mean",
        save_positions=False,
        output_dir=str(backtest_dir),
    )

    # 注入行业映射（绕过文件加载）
    if industry_map:
        bt._industry_map = industry_map

    result = bt.run(save=save, pred_df=all_preds)

    report_df = result.get("report", pd.DataFrame())
    if not report_df.empty:
        yearly = calc_yearly_metrics(report_df)
        print("\n" + "=" * 80)
        label = "Full Constraints" if enable_all_constraints else "No Constraints"
        print(f"Walk-Forward Full Backtest ({label}) — Yearly Results")
        print("=" * 80)
        print(yearly.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        print("=" * 80)

        total_ret = report_df["account_value"].iloc[-1] / report_df["account_value"].iloc[0] - 1.0
        daily_rets = report_df["account_value"].pct_change().dropna()
        sharpe = daily_rets.mean() / daily_rets.std() * np.sqrt(252) if daily_rets.std() > 0 else 0
        roll_max = report_df["account_value"].cummax()
        max_dd = ((report_df["account_value"] - roll_max) / roll_max).min()
        print(f"\nOverall: total_return={total_ret:.4f}, sharpe={sharpe:.4f}, max_drawdown={max_dd:.4f}")

        if save:
            yearly.to_csv(backtest_dir / "yearly_metrics.csv", index=False)
            with (backtest_dir / "overall_metrics.json").open("w", encoding="utf-8") as f:
                json.dump({
                    "total_return": float(total_ret),
                    "sharpe": float(sharpe),
                    "max_drawdown": float(max_dd),
                    "start_date": start_date,
                    "end_date": end_date,
                    "trading_days": len(report_df),
                    "constraints_enabled": enable_all_constraints,
                }, f, ensure_ascii=False, indent=2)

    return result


def calc_yearly_metrics(report_df: pd.DataFrame) -> pd.DataFrame:
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
    parser = argparse.ArgumentParser(description="Walk-Forward Full Backtest")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to walk-forward config")
    parser.add_argument("--price-path", type=str, default=None,
                        help="Override price data path")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-constraints", action="store_true",
                        help="Disable all constraints (baseline comparison)")
    args = parser.parse_args()

    run_full_backtest(
        config_path=args.config or "conf/walk_forward_config.json",
        price_path=args.price_path,
        save=not args.no_save,
        enable_all_constraints=not args.no_constraints,
    )


if __name__ == "__main__":
    main()
