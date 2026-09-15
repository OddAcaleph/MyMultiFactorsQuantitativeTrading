"""Phase 2 optimization backtests: industry-neutral, style-neutral, multi-strategy."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backtester.simple_backtester import SimpleBacktester

PRED_PATH = PROJECT_ROOT / "output" / "walk_forward_optimized" / "predictions" / "all_predictions_2005_2025.parquet"
PRICE_PATH = PROJECT_ROOT / "data" / "processd_data" / "wide_table_daily_bars"
INDUSTRY_PATH = PRICE_PATH
OUTPUT_DIR = PROJECT_ROOT / "output" / "phase2_optimization"
START = "2005-01-01"
END = "2025-12-31"

BASE_STRATEGY = {
    "topk": 100,
    "n_drop": 10,
    "dynamic_position_enabled": True,
    "dyn_pos_method": "trend_ma",
    "dyn_pos_trend_ma_days": 50,
    "dyn_pos_trend_bear_position": 0.3,
}

BASE_EXCHANGE = {
    "deal_price": "open",
    "open_cost": 0.0005,
    "close_cost": 0.0015,
    "slippage": 0.003,
    "min_cost": 5.0,
    "limit_threshold": 0.095,
    "lot_size": 100,
    "forbid_buy_limit_up": True,
    "forbid_buy_limit_down": True,
    "forbid_sell_limit_down": False,
}


def run_backtest(name: str, strategy_overrides: dict | None = None, slippage: float = 0.003) -> dict:
    strategy = {**BASE_STRATEGY, **(strategy_overrides or {})}
    exchange = {**BASE_EXCHANGE, "slippage": slippage}
    out_dir = OUTPUT_DIR / name

    bt = SimpleBacktester(
        config_path=str(PROJECT_ROOT / "conf" / "simple_backtester_config.json"),
        prediction_path=str(PRED_PATH),
        price_path=str(PRICE_PATH),
        industry_data_path=str(INDUSTRY_PATH),
        start_time=START,
        end_time=END,
        output_dir=str(out_dir),
        account=1_000_000,
        freq="5d",
        strategy=strategy,
        exchange_kwargs=exchange,
        signal_delay=1,
        save_positions=False,
    )
    t0 = time.time()
    result = bt.run(save=True)
    elapsed = time.time() - t0
    a = result["analysis"]
    r = a["return"]
    summary = {
        "name": name,
        "annualized_return": r["annualized_return"],
        "annualized_volatility": r["annualized_volatility"],
        "sharpe": r["sharpe"],
        "max_drawdown": r["max_drawdown"],
        "sortino": r["sortino"],
        "calmar": r["calmar"],
        "total_return": r["total_return"],
        "mean_turnover": a["summary"]["mean_turnover"],
        "elapsed_sec": round(elapsed, 1),
    }
    print(f"  {name}: Ret={summary['annualized_return']:.4f} Vol={summary['annualized_volatility']:.4f} "
          f"Sharpe={summary['sharpe']:.4f} MaxDD={summary['max_drawdown']:.4f} ({elapsed:.0f}s)")
    return summary


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    print("=== Phase 2 Optimization Backtests ===")
    print(f"Period: {START} ~ {END}")
    print()

    # 1. Baseline (re-run for consistency)
    print("[1/12] Baseline: Top100 + MA50 bear=30% + 0.3% slip")
    results.append(run_backtest("baseline_top100_ma50_bear30_slip03"))

    # 2. Industry neutral (z-score within industry)
    print("[2/12] Industry neutral + Top100 + MA50")
    results.append(run_backtest("industry_neutral_top100_ma50_slip03", {
        "industry_neutral": True,
    }))

    # 3. Industry neutral + industry cap (max 10 per industry)
    print("[3/12] Industry neutral + max 10 per industry")
    results.append(run_backtest("ind_neutral_cap10_top100_ma50_slip03", {
        "industry_neutral": True,
        "max_industry_count": 10,
    }))

    # 4. Industry neutral + industry cap (max 8 per industry)
    print("[4/12] Industry neutral + max 8 per industry")
    results.append(run_backtest("ind_neutral_cap8_top100_ma50_slip03", {
        "industry_neutral": True,
        "max_industry_count": 8,
    }))

    # 5. Industry neutral + Top150 (more diversification)
    print("[5/12] Industry neutral + Top150 + MA50")
    results.append(run_backtest("ind_neutral_top150_ma50_slip03", {
        "industry_neutral": True,
        "topk": 150,
        "n_drop": 15,
    }))

    # 6. Industry neutral + Top200
    print("[6/12] Industry neutral + Top200 + MA50")
    results.append(run_backtest("ind_neutral_top200_ma50_slip03", {
        "industry_neutral": True,
        "topk": 200,
        "n_drop": 20,
    }))

    # 7. Industry neutral + MA200 bear=30% (more conservative trend)
    print("[7/12] Industry neutral + MA200 bear=30%")
    results.append(run_backtest("ind_neutral_top100_ma200_bear30_slip03", {
        "industry_neutral": True,
        "dyn_pos_trend_ma_days": 200,
    }))

    # 8. Industry neutral + bear=40% (more aggressive)
    print("[8/12] Industry neutral + MA50 bear=40%")
    results.append(run_backtest("ind_neutral_top100_ma50_bear40_slip03", {
        "industry_neutral": True,
        "dyn_pos_trend_bear_position": 0.4,
    }))

    # 9. Industry neutral + 0.5% slip (retail assumption)
    print("[9/12] Industry neutral + 0.5% slip")
    results.append(run_backtest("ind_neutral_top100_ma50_slip05", {
        "industry_neutral": True,
    }, slippage=0.005))

    # 10. Industry neutral + vol-weighted
    print("[10/12] Industry neutral + vol-weighted")
    results.append(run_backtest("ind_neutral_volweight_top100_ma50_slip03", {
        "industry_neutral": True,
        "vol_weight_enabled": True,
        "vol_weight_power": 0.5,
    }))

    # 11. Industry neutral + max industry weight 10%
    print("[11/12] Industry neutral + max 10% industry weight")
    results.append(run_backtest("ind_neutral_maxind10pct_top100_ma50_slip03", {
        "industry_neutral": True,
        "max_industry_weight": 0.10,
    }))

    # 12. Industry neutral + Top50 (more concentrated)
    print("[12/12] Industry neutral + Top50 + MA50")
    results.append(run_backtest("ind_neutral_top50_ma50_slip03", {
        "industry_neutral": True,
        "topk": 50,
        "n_drop": 5,
    }))

    # Save summary
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to {summary_path}")

    # Print comparison table
    print("\n=== Results Summary ===")
    print(f"{'Strategy':<50} {'AnnRet':>8} {'AnnVol':>8} {'Sharpe':>8} {'MaxDD':>10} {'Turnover':>9}")
    print("-" * 95)
    for r in sorted(results, key=lambda x: x["sharpe"], reverse=True):
        print(f"{r['name']:<50} {r['annualized_return']:>7.2%} {r['annualized_volatility']:>7.2%} "
              f"{r['sharpe']:>8.3f} {r['max_drawdown']:>9.2%} {r['mean_turnover']:>8.2%}")


if __name__ == "__main__":
    main()
