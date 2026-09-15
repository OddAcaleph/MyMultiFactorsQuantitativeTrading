"""Run risk-parity backtest on v3 pruned predictions and compare with baseline.

Usage:
    PYTHONPATH=src python scripts/run_v3_rp_comparison.py \\
        --v3-pred output/walk_forward_v3_pruned/predictions/all_predictions_2005_2025.parquet \\
        --baseline-pred output/walk_forward_expanded/predictions/all_predictions_2005_2025.parquet \\
        --risk-dir outputs/risk_model/v1 \\
        --bars-dir data/cleaned_data/daily_bars \\
        --output-dir output/optimizer_backtest/v3_comparison
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import pandas as pd

from optimizer.risk_interface import RiskInterface
from optimizer.run_optimizer_backtest import run_backtest, compute_metrics, load_list_dates


def run_rp_backtest(
    pred_path: Path,
    risk_dir: Path,
    bars_dir: Path,
    output_dir: Path,
    pool_size: int = 25,
    alpha_tilt: float = 0.7,
    trend_ma_days: int = 50,
    trend_ma_bear: float = 0.3,
    rebalance_freq: int = 5,
    slippage: float = 0.003,
    label: str = "",
) -> dict:
    """Run a single RP backtest configuration."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading predictions from %s", pred_path)
    pred_full = pd.read_parquet(pred_path)
    pred_full.index = pred_full.index.set_names(["datetime", "instrument"])

    risk_if = RiskInterface({"output_dir": str(risk_dir)})

    config = {
        "strategy": "risk_parity",
        "alpha": {
            "method": "zscore",
            "winsorize": True,
            "winsorize_quantile": 0.01,
            "industry_neutral": False,
            "rank": False,
            "smooth_span": 0,
        },
        "candidate_pool": {
            "pool_size": pool_size,
            "min_avg_amount_20d": 5000,
            "filter_st": True,
            "filter_suspend": True,
            "filter_new_stock_days": 60,
        },
        "objective": {
            "risk_aversion": 0.1,
            "turnover_penalty": 0.01,
        },
        "constraints": {
            "long_only": True,
            "fully_invested": True,
            "max_weight": 0.03,
            "industry": {"max_weight": 0.20},
        },
        "risk_parity": {
            "alpha_tilt": alpha_tilt,
            "industry_neutral_selection": True,
            "max_weight": 0.10,
            "industry_max_weight": 0.20,
        },
        "target_vol": 0.0,
        "solver": {"solver": "OSQP", "verbose": False, "max_iter": 4000},
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    list_dates = load_list_dates(str(bars_dir))

    t0 = time.time()
    results = run_backtest(
        pred_full, risk_if, str(bars_dir), config,
        rebalance_freq=rebalance_freq,
        list_dates=list_dates,
        enable_limit_constraints=True,
        buy_cost=0.0005,
        sell_cost=0.0015,
        slippage=slippage,
        board_lot=100,
        target_vol=0.0,
        signal_delay=1,
        trend_ma_days=trend_ma_days,
        trend_ma_bear_position=trend_ma_bear,
        trend_ma_smooth=False,
        trend_ma_smooth_band=0.05,
        trend_ma2_days=0,
        trend_ma2_bear_position=0.15,
    )
    elapsed = time.time() - t0

    metrics = compute_metrics(results["nav"], results["diagnostics"])
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["n_optimized"] = results["n_optimized"]
    metrics["n_skipped"] = results["n_skipped"]
    metrics["total_fee"] = round(results.get("total_fee", 0.0), 6)
    metrics["total_slippage"] = round(results.get("total_slippage", 0.0), 6)
    metrics["total_transaction_cost"] = round(
        results.get("total_fee", 0.0) + results.get("total_slippage", 0.0), 6
    )

    results["nav"].to_parquet(output_dir / "nav.parquet")
    if not results["diagnostics"].empty:
        results["diagnostics"].to_parquet(output_dir / "diagnostics.parquet")
    if not results["holdings"].empty:
        results["holdings"].to_parquet(output_dir / "holdings.parquet")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  RP Backtest: {label}")
    print(f"{'='*60}")
    print(f"  年化收益    : {metrics.get('annual_return', 0):>8.2%}")
    print(f"  夏普比率    : {metrics.get('sharpe_ratio', 0):>8.3f}")
    print(f"  最大回撤    : {metrics.get('max_drawdown', 0):>8.2%}")
    print(f"  卡玛比率    : {metrics.get('calmar_ratio', 0):>8.3f}")
    print(f"  索提诺比率  : {metrics.get('sortino_ratio', 0):>8.3f}")
    print(f"  平均持仓    : {metrics.get('avg_holdings', 0):>8.1f}")
    print(f"  平均换手率  : {metrics.get('avg_turnover', 0):>8.2%}")
    print(f"  交易天数    : {metrics.get('n_trading_days', 0):>8.0f}")
    print(f"  耗时        : {elapsed:>6.1f}s")

    return metrics


def compare_results(v3_metrics: dict, base_metrics: dict) -> None:
    """Print side-by-side comparison."""
    print(f"\n{'='*70}")
    print(f"  对比：v3 pruned (112f) vs baseline (110f)")
    print(f"{'='*70}")
    print(f"  {'指标':<15} {'v3 pruned':>12} {'baseline':>12} {'差值':>12}")
    print(f"  {'-'*65}")

    pairs = [
        ("年化收益", "annual_return", "%"),
        ("夏普比率", "sharpe_ratio", "f"),
        ("最大回撤", "max_drawdown", "%"),
        ("卡玛比率", "calmar_ratio", "f"),
        ("索提诺比率", "sortino_ratio", "f"),
        ("胜率", "win_rate", "%"),
        ("平均持仓", "avg_holdings", ".1f"),
        ("平均换手率", "avg_turnover", "%"),
    ]

    for label, key, fmt in pairs:
        v = v3_metrics.get(key, 0)
        b = base_metrics.get(key, 0)
        d = v - b
        if fmt == "%":
            print(f"  {label:<15} {v:>11.2%} {b:>11.2%} {d:>+11.2%}")
        elif fmt == ".1f":
            print(f"  {label:<15} {v:>12.1f} {b:>12.1f} {d:>+12.1f}")
        else:
            print(f"  {label:<15} {v:>12.3f} {b:>12.3f} {d:>+12.3f}")

    print(f"  {'='*65}")


def main() -> None:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="V3 pruned vs baseline RP comparison")
    parser.add_argument("--v3-pred", required=True, help="V3 pruned predictions path")
    parser.add_argument("--baseline-pred", required=True, help="Baseline predictions path")
    parser.add_argument("--risk-dir", required=True, help="Risk model directory")
    parser.add_argument("--bars-dir", required=True, help="Daily bars directory")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--pool-size", type=int, default=25)
    parser.add_argument("--alpha-tilt", type=float, default=0.7)
    parser.add_argument("--trend-ma-days", type=int, default=50)
    parser.add_argument("--slippage", type=float, default=0.003)
    args = parser.parse_args()

    v3_pred = Path(args.v3_pred)
    base_pred = Path(args.baseline_pred)
    risk_dir = Path(args.risk_dir)
    bars_dir = Path(args.bars_dir)
    out_dir = Path(args.output_dir)

    print("Running baseline RP backtest...")
    base_metrics = run_rp_backtest(
        base_pred, risk_dir, bars_dir,
        out_dir / "baseline_rp_pool25_tilt07",
        pool_size=args.pool_size,
        alpha_tilt=args.alpha_tilt,
        trend_ma_days=args.trend_ma_days,
        slippage=args.slippage,
        label="Baseline (110f) RP pool25 tilt0.7",
    )

    print("\nRunning v3 pruned RP backtest...")
    v3_metrics = run_rp_backtest(
        v3_pred, risk_dir, bars_dir,
        out_dir / "v3_pruned_rp_pool25_tilt07",
        pool_size=args.pool_size,
        alpha_tilt=args.alpha_tilt,
        trend_ma_days=args.trend_ma_days,
        slippage=args.slippage,
        label="V3 Pruned (112f) RP pool25 tilt0.7",
    )

    compare_results(v3_metrics, base_metrics)

    comparison = {
        "v3_pruned": v3_metrics,
        "baseline": base_metrics,
        "diff": {k: v3_metrics.get(k, 0) - base_metrics.get(k, 0)
                 for k in set(v3_metrics) | set(base_metrics)},
    }
    with open(out_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)

    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
