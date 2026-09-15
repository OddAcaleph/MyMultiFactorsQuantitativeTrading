"""Grid search for dynamic position + vol-weight optimization.

Searches over:
- trend MA window (20, 50, 100, 150, 200)
- bear market position (0.0, 0.2, 0.3, 0.4, 0.5)
- target vol (0.15, 0.20, 0.25, 0.30)
- vol lookback (60, 80, 100, 120)
- vol weight power (0 = equal weight, 0.5, 1.0)
- topk (20, 30, 50)
- rebalance freq (5d, 10d, 20d)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backtester.simple_backtester import SimpleBacktester

PRED_PATH = "output/walk_forward_optimized/predictions/all_predictions_2005_2025.parquet"
PRICE_PATH = "data/processd_data/wide_table_daily_bars"
BENCH_PATH = "/opt/tiger/qyd/qlib_data_cn/features"
OUTPUT_DIR = "output/optimization_grid_search"
START = "2005-01-01"
END = "2025-12-31"


def run_one(params: dict) -> dict:
    strategy = {
        "class": "TopkDropoutStrategy",
        "topk": params.get("topk", 30),
        "n_drop": params.get("n_drop", 5),
        "filter_st": False,
        "filter_suspend": False,
        "filter_new_stock_days": 0,
        "min_avg_amount_20d": 0,
        "dynamic_position_enabled": params.get("dyn_pos_enabled", True),
        "dyn_pos_method": params.get("dyn_pos_method", "trend_plus_vol"),
        "dyn_pos_min_position": 0.0,
        "dyn_pos_max_position": 1.0,
        "dyn_pos_trend_ma_days": params.get("trend_ma", 50),
        "dyn_pos_trend_bear_position": params.get("bear_pos", 0.3),
        "dyn_pos_target_vol": params.get("target_vol", 0.25),
        "dyn_pos_vol_lookback": params.get("vol_lookback", 100),
        "dyn_pos_vol_smooth": 5,
        "vol_weight_enabled": params.get("vol_weight", False),
        "vol_lookback_days": params.get("vol_weight_lb", 60),
        "vol_weight_power": params.get("vol_weight_power", 1.0),
    }

    config = {
        "prediction_path": PRED_PATH,
        "price_path": PRICE_PATH,
        "benchmark_path": BENCH_PATH,
        "start_time": START,
        "end_time": END,
        "output_dir": OUTPUT_DIR + "/tmp",
        "strategy": strategy,
        "exchange_kwargs": {
            "deal_price": "open",
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "min_cost": 5.0,
            "slippage": 0.005,
            "limit_threshold": 0.095,
            "lot_size": 100,
            "forbid_buy_limit_up": True,
            "forbid_buy_limit_down": True,
            "forbid_sell_limit_down": True,
        },
        "freq": params.get("freq", "5d"),
        "signal_delay": 1,
        "save_positions": False,
    }

    try:
        bt = SimpleBacktester(config_path="conf/simple_backtester_config.json", **config)
        result = bt.run(save=False)
        r = result["analysis"].get("return", {})
        s = result["analysis"].get("summary", {})
        return {
            "params": params,
            "annual_return": float(r.get("annualized_return", 0)),
            "annual_vol": float(r.get("annualized_volatility", 0)),
            "sharpe": float(r.get("sharpe", 0)),
            "max_drawdown": float(r.get("max_drawdown", 0)),
            "sortino": float(r.get("sortino", 0)),
            "calmar": float(r.get("calmar", 0)),
            "total_return": float(r.get("total_return", 0)),
            "mean_turnover": float(s.get("mean_turnover", 0)),
            "total_cost": float(s.get("total_cost", 0)),
            "success": True,
        }
    except Exception as e:
        return {"params": params, "success": False, "error": str(e)}


def main():
    # Phase 1: coarse grid over key parameters
    grid = []

    # Baseline (no dynamic position)
    grid.append({"topk": 30, "freq": "5d", "dyn_pos_enabled": False, "vol_weight": False, "name": "baseline_top30_5d"})

    # Trend MA only
    for ma in [20, 50, 100, 150, 200]:
        for bear in [0.0, 0.2, 0.3, 0.4, 0.5]:
            grid.append({
                "topk": 30, "freq": "5d",
                "dyn_pos_method": "trend_ma",
                "trend_ma": ma, "bear_pos": bear,
                "vol_weight": False,
                "name": f"trend_ma{ma}_bear{int(bear*100)}",
            })

    # Target vol only
    for tv in [0.15, 0.20, 0.25, 0.30, 0.35]:
        for lb in [60, 80, 100, 120]:
            grid.append({
                "topk": 30, "freq": "5d",
                "dyn_pos_method": "target_vol",
                "target_vol": tv, "vol_lookback": lb,
                "vol_weight": False,
                "name": f"tv{int(tv*100)}_lb{lb}",
            })

    # Trend + vol combined
    for ma in [20, 50, 100, 150]:
        for bear in [0.2, 0.3, 0.4]:
            for tv in [0.20, 0.25, 0.30]:
                for lb in [80, 100, 120]:
                    grid.append({
                        "topk": 30, "freq": "5d",
                        "dyn_pos_method": "trend_plus_vol",
                        "trend_ma": ma, "bear_pos": bear,
                        "target_vol": tv, "vol_lookback": lb,
                        "vol_weight": False,
                        "name": f"tpv_ma{ma}_bear{int(bear*100)}_tv{int(tv*100)}_lb{lb}",
                    })

    print(f"Total grid points: {len(grid)}")
    print(f"Estimated time: ~{len(grid) * 3 / 60:.0f} minutes (at 3 min each)")
    print()

    results = []
    start_time = time.time()

    for i, params in enumerate(grid):
        name = params.get("name", f"run_{i}")
        t0 = time.time()
        result = run_one(params)
        elapsed = time.time() - t0

        if result["success"]:
            results.append(result)
            print(f"[{i+1}/{len(grid)}] {name}: "
                  f"ret={result['annual_return']:.2%} "
                  f"vol={result['annual_vol']:.2%} "
                  f"sharpe={result['sharpe']:.3f} "
                  f"maxdd={result['max_drawdown']:.2%} "
                  f"({elapsed:.0f}s)")
        else:
            print(f"[{i+1}/{len(grid)}] {name}: FAILED - {result.get('error', 'unknown')}")

    # Sort by Sharpe
    results.sort(key=lambda x: x["sharpe"], reverse=True)

    print()
    print("=" * 80)
    print(f"TOP 20 by Sharpe (total time: {(time.time()-start_time)/60:.1f} min)")
    print("=" * 80)
    for i, r in enumerate(results[:20]):
        p = r["params"]
        print(f"{i+1:2d}. {p.get('name',''):40s} "
              f"sharpe={r['sharpe']:.3f} "
              f"ret={r['annual_return']:.2%} "
              f"vol={r['annual_vol']:.2%} "
              f"maxdd={r['max_drawdown']:.2%} "
              f"calmar={r['calmar']:.3f}")

    # Save results
    out_path = Path(OUTPUT_DIR) / "grid_search_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Top by return (>=20%)
    print()
    print("=" * 80)
    print("TOP 20 by Sharpe (return >= 20%)")
    print("=" * 80)
    filtered = [r for r in results if r["annual_return"] >= 0.20]
    filtered.sort(key=lambda x: x["sharpe"], reverse=True)
    for i, r in enumerate(filtered[:20]):
        p = r["params"]
        print(f"{i+1:2d}. {p.get('name',''):40s} "
              f"sharpe={r['sharpe']:.3f} "
              f"ret={r['annual_return']:.2%} "
              f"vol={r['annual_vol']:.2%} "
              f"maxdd={r['max_drawdown']:.2%}")


if __name__ == "__main__":
    main()
