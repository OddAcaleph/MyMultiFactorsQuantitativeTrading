"""Grid-search runner for SimpleBacktester parameters.

The search phase always runs in-memory without saving artifacts.  After all
parameter combinations finish, the best combination by ``summary.total_return``
is re-run once with saving and plotting enabled.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import itertools
import json
import multiprocessing as mp
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import (
    DEFAULT_SIMPLE_BACKTEST_GRID_SEARCH_CONFIG_PATH,
    DEFAULT_SIMPLE_BACKTESTER_CONFIG_PATH,
    PROJECT_ROOT,
    BacktestPlotter,
    deep_merge,
    load_simple_backtest_grid_search_config,
    load_simple_backtester_config,
)

try:
    from .simple_backtester import SimpleBacktester
except ImportError:  # pragma: no cover - supports running this file directly.
    from backtester.simple_backtester import SimpleBacktester


def expand_project_path(value: str) -> str:
    """Expand project-root placeholders used by grid-search config files."""

    return value.replace("${PROJECT_ROOT}", str(PROJECT_ROOT))


def load_grid_search_defaults(config_path: str | Path | None = None) -> dict[str, Any]:
    defaults = load_simple_backtest_grid_search_config(config_path)
    if "output_root" in defaults and defaults["output_root"] is not None:
        defaults["output_root"] = expand_project_path(str(defaults["output_root"]))
    return defaults


_DEFAULT_GRID_SEARCH_CONFIG = load_grid_search_defaults()
DEFAULT_OUTPUT_ROOT = Path(_DEFAULT_GRID_SEARCH_CONFIG["output_root"])
DEFAULT_FREQ_GRID = str(_DEFAULT_GRID_SEARCH_CONFIG["freq_grid"])
DEFAULT_TOPK_GRID = str(_DEFAULT_GRID_SEARCH_CONFIG["topk_grid"])
DEFAULT_N_DROP_GRID = str(_DEFAULT_GRID_SEARCH_CONFIG["n_drop_grid"])
DEFAULT_ACCOUNT_GRID = str(_DEFAULT_GRID_SEARCH_CONFIG["account_grid"])
DEFAULT_JOBS = int(_DEFAULT_GRID_SEARCH_CONFIG.get("jobs", 4))

PreloadedData = tuple[pd.DataFrame, pd.DataFrame, pd.Series]

_WORKER_CONFIG_PATH: str | None = None
_WORKER_BASE_OVERRIDES: dict[str, Any] | None = None
_WORKER_PRELOADED_DATA: PreloadedData | None = None


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--grid-search-config",
        default=str(DEFAULT_SIMPLE_BACKTEST_GRID_SEARCH_CONFIG_PATH),
        help="Path to SimpleBacktester grid-search defaults JSON.",
    )
    pre_args, _ = pre_parser.parse_known_args()
    grid_defaults = load_grid_search_defaults(pre_args.grid_search_config)

    parser = argparse.ArgumentParser(
        description="Grid search SimpleBacktester freq/topk/n_drop/account, then save and plot the best run.",
        parents=[pre_parser],
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_SIMPLE_BACKTESTER_CONFIG_PATH),
        help="Path to SimpleBacktester config JSON. Defaults to conf/simple_backtester_config.json.",
    )
    parser.add_argument(
        "--freq-grid",
        default=str(grid_defaults["freq_grid"]),
        help=f"Comma-separated freq grid. Defaults to {grid_defaults['freq_grid']}.",
    )
    parser.add_argument(
        "--topk-grid",
        default=str(grid_defaults["topk_grid"]),
        help=f"Comma-separated topk grid. Defaults to {grid_defaults['topk_grid']}.",
    )
    parser.add_argument(
        "--n-drop-grid",
        default=str(grid_defaults["n_drop_grid"]),
        help=f"Comma-separated n_drop grid. Defaults to {grid_defaults['n_drop_grid']}.",
    )
    parser.add_argument(
        "--account-grid",
        default=str(grid_defaults["account_grid"]),
        help=f"Comma-separated account grid. Defaults to {grid_defaults['account_grid']}.",
    )
    parser.add_argument(
        "--output-root",
        default=str(grid_defaults["output_root"]),
        help="Root directory for the final saved best run. A timestamped child directory will be created.",
    )

    # Common optional overrides inherited from run_simple_backtest.py.
    parser.add_argument("--trainer-config", default=None, help="Optional trainer config JSON.")
    parser.add_argument("--prediction-path", default=None, help="Prediction parquet path.")
    parser.add_argument("--price-path", default=None, help="Daily wide-table bar parquet root.")
    parser.add_argument("--benchmark-path", default=None, help="Qlib features directory for benchmark data.")
    parser.add_argument("--start-time", default=None, help="Backtest start date, e.g. 2023-01-01.")
    parser.add_argument("--end-time", default=None, help="Backtest end date, e.g. 2025-12-31.")
    parser.add_argument("--score-col", default=None, help="Prediction score column. Defaults to columns.score.")
    parser.add_argument("--benchmark", default=None, help="Benchmark index, e.g. 000300.SH.")
    parser.add_argument("--deal-price", choices=["open", "close"], default=None, help="Execute orders at open or close.")
    parser.add_argument("--open-cost", type=float, default=None, help="Buy-side commission rate.")
    parser.add_argument("--close-cost", type=float, default=None, help="Sell-side commission/stamp-duty rate.")
    parser.add_argument("--min-cost", type=float, default=None, help="Minimum fee per trade.")
    parser.add_argument("--slippage", type=float, default=None, help="One-way slippage rate.")
    parser.add_argument("--limit-threshold", type=float, default=None, help="Limit-up/down threshold, e.g. 0.095.")
    parser.add_argument("--lot-size", type=int, default=None, help="Board lot size. A-share default is 100.")
    parser.add_argument("--allow-buy-limit-up", action="store_true", help="Allow buying limit-up stocks.")
    parser.add_argument("--allow-buy-limit-down", action="store_true", help="Allow buying limit-down stocks.")
    parser.add_argument("--forbid-sell-limit-down", action="store_true", help="Forbid selling limit-down stocks.")
    parser.add_argument("--no-positions", action="store_true", help="Do not save positions parquet in the final best run.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=int(grid_defaults.get("jobs", DEFAULT_JOBS)),
        help=f"Parallel grid-search workers. Defaults to {grid_defaults.get('jobs', DEFAULT_JOBS)}. Use 0 to auto-select CPU count.",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately if any grid combination fails.")
    return parser.parse_args()


def parse_str_grid(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"Grid is empty: {value!r}")
    return items


def parse_int_grid(value: str, name: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} grid is empty")
    if any(item <= 0 for item in items):
        raise ValueError(f"{name} grid values must be positive: {items}")
    return items


def parse_float_grid(value: str, name: str) -> list[float]:
    items = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} grid is empty")
    if any(item <= 0 for item in items):
        raise ValueError(f"{name} grid values must be positive: {items}")
    return items


def build_base_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Build non-grid overrides while preserving nested config values."""

    config = load_simple_backtester_config(args.config)
    overrides: dict[str, Any] = {
        "trainer_config_path": args.trainer_config,
        "prediction_path": args.prediction_path,
        "price_path": args.price_path,
        "benchmark_path": args.benchmark_path,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "benchmark": args.benchmark,
        "strategy": config.get("strategy", {}),
    }

    column_overrides = {"score": args.score_col}
    column_overrides = {key: value for key, value in column_overrides.items() if value is not None}
    if column_overrides:
        overrides["columns"] = deep_merge(config.get("columns", {}), column_overrides)

    exchange_overrides: dict[str, Any] = {
        "deal_price": args.deal_price,
        "open_cost": args.open_cost,
        "close_cost": args.close_cost,
        "min_cost": args.min_cost,
        "slippage": args.slippage,
        "limit_threshold": args.limit_threshold,
        "lot_size": args.lot_size,
    }
    if args.allow_buy_limit_up:
        exchange_overrides["forbid_buy_limit_up"] = False
    if args.allow_buy_limit_down:
        exchange_overrides["forbid_buy_limit_down"] = False
    if args.forbid_sell_limit_down:
        exchange_overrides["forbid_sell_limit_down"] = True
    exchange_overrides = {key: value for key, value in exchange_overrides.items() if value is not None}
    if exchange_overrides:
        overrides["exchange_kwargs"] = deep_merge(config.get("exchange_kwargs", {}), exchange_overrides)
    if args.no_positions:
        overrides["save_positions"] = False

    return {key: value for key, value in overrides.items() if value is not None}


def build_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    freqs = parse_str_grid(args.freq_grid)
    topks = parse_int_grid(args.topk_grid, "topk")
    n_drops = parse_int_grid(args.n_drop_grid, "n_drop")
    accounts = parse_float_grid(args.account_grid, "account")

    grid = []
    for freq, topk, n_drop, account in itertools.product(freqs, topks, n_drops, accounts):
        grid.append({"freq": freq, "topk": topk, "n_drop": n_drop, "account": account})
    return grid


def preload_data(config_path: str, base_overrides: dict[str, Any]) -> PreloadedData:
    """Load immutable input data once so grid points do not repeatedly hit parquet/bin files."""

    backtester = SimpleBacktester(config_path=config_path, **base_overrides)
    return backtester.load_predictions(), backtester.load_prices(), backtester.load_benchmark_returns()


def run_one(
    config_path: str,
    base_overrides: dict[str, Any],
    params: dict[str, Any],
    save: bool,
    output_dir: Path | None = None,
    preloaded_data: PreloadedData | None = None,
) -> dict[str, Any]:
    strategy_base = base_overrides.get("strategy", {})
    overrides = dict(base_overrides)
    overrides["freq"] = params["freq"]
    overrides["account"] = params["account"]
    overrides["strategy"] = deep_merge(strategy_base, {"topk": params["topk"], "n_drop": params["n_drop"]})
    if output_dir is not None:
        overrides["output_dir"] = output_dir
    if not save:
        # Searching does not need positions and must not write files.
        overrides["save_positions"] = False

    backtester = SimpleBacktester(config_path=config_path, **overrides)
    if preloaded_data is None:
        result = backtester.run(save=save)
    else:
        pred_df, price_df, benchmark_returns = preloaded_data
        result = backtester.run(save=save, pred_df=pred_df, price_df=price_df, benchmark_returns=benchmark_returns)
    return {"backtester": backtester, "result": result}


def evaluate_one(config_path: str, base_overrides: dict[str, Any], params: dict[str, Any], preloaded_data: PreloadedData) -> dict[str, Any]:
    """Run one search point and return a flattened grid-search record."""

    try:
        run = run_one(config_path, base_overrides, params, save=False, preloaded_data=preloaded_data)
        record = flatten_record(params, run["result"])
        return {"params": params, "record": record, "error": None}
    except Exception as exc:  # noqa: BLE001 - caller decides whether to fail fast.
        record = flatten_record(params, {}, status="failed", error=str(exc))
        return {"params": params, "record": record, "error": str(exc)}


def init_worker(config_path: str, base_overrides: dict[str, Any]) -> None:
    """Initialize process-pool globals; forked workers reuse parent-preloaded data."""

    global _WORKER_CONFIG_PATH, _WORKER_BASE_OVERRIDES, _WORKER_PRELOADED_DATA
    _WORKER_CONFIG_PATH = config_path
    _WORKER_BASE_OVERRIDES = base_overrides
    if _WORKER_PRELOADED_DATA is None:
        _WORKER_PRELOADED_DATA = preload_data(config_path, base_overrides)


def evaluate_one_worker(params: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_CONFIG_PATH is None or _WORKER_BASE_OVERRIDES is None or _WORKER_PRELOADED_DATA is None:
        raise RuntimeError("Grid-search worker is not initialized")
    return evaluate_one(_WORKER_CONFIG_PATH, _WORKER_BASE_OVERRIDES, params, _WORKER_PRELOADED_DATA)


def resolve_jobs(requested_jobs: int, grid_size: int) -> int:
    if requested_jobs < 0:
        raise ValueError("--jobs must be >= 0")
    if requested_jobs == 0:
        requested_jobs = os.cpu_count() or 1
    return max(1, min(requested_jobs, grid_size))


def flatten_record(params: dict[str, Any], result: dict[str, Any], status: str = "success", error: str | None = None) -> dict[str, Any]:
    analysis = result.get("analysis", {}) if result else {}
    summary = analysis.get("summary", {})
    ret = analysis.get("return", {})
    risk = analysis.get("risk", {})
    return {
        **params,
        "status": status,
        "error": error,
        "total_return": summary.get("total_return"),
        "annualized_return": ret.get("annualized_return"),
        "sharpe": ret.get("sharpe"),
        "risk_sharpe": risk.get("sharpe"),
        "max_drawdown": ret.get("max_drawdown"),
        "trade_count": summary.get("trade_count"),
        "rebalance_days": summary.get("rebalance_days"),
        "mean_turnover": summary.get("mean_turnover"),
        "final_account_value": summary.get("final_account_value"),
    }


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(key): json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(value) for value in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return obj


def main() -> dict[str, Any]:
    args = parse_args()
    base_overrides = build_base_overrides(args)
    grid = build_grid(args)
    jobs = resolve_jobs(args.jobs, len(grid))
    records: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None

    print(f"开始网格搜索，共 {len(grid)} 组参数；搜索阶段全部使用 no-save；并行 workers={jobs}。")
    print("预加载预测、行情和 benchmark 数据，避免每个网格点重复读取文件。")

    global _WORKER_CONFIG_PATH, _WORKER_BASE_OVERRIDES, _WORKER_PRELOADED_DATA
    _WORKER_CONFIG_PATH = args.config
    _WORKER_BASE_OVERRIDES = base_overrides
    _WORKER_PRELOADED_DATA = preload_data(args.config, base_overrides)

    def consume_evaluation(idx: int, evaluation: dict[str, Any]) -> None:
        nonlocal best
        params = evaluation["params"]
        record = evaluation["record"]
        records.append(record)
        if evaluation["error"] is None:
            print(f"[{idx}/{len(grid)}] 完成参数: {params}")
            print(f"    total_return={record['total_return']}, sharpe={record['sharpe']}")
            if record["total_return"] is not None and (best is None or record["total_return"] > best["record"]["total_return"]):
                best = {"params": dict(params), "record": record}
        else:
            print(f"[{idx}/{len(grid)}] 参数失败: {params}")
            print(f"    失败: {evaluation['error']}")
            if args.fail_fast:
                raise RuntimeError(evaluation["error"])

    if jobs == 1:
        for idx, params in enumerate(grid, start=1):
            print(f"[{idx}/{len(grid)}] 搜索参数: {params}")
            evaluation = evaluate_one(args.config, base_overrides, params, _WORKER_PRELOADED_DATA)
            consume_evaluation(idx, evaluation)
    else:
        mp_context = mp.get_context("fork") if hasattr(mp, "get_context") else None
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=jobs,
            mp_context=mp_context,
            initializer=init_worker,
            initargs=(args.config, base_overrides),
        ) as executor:
            futures = {executor.submit(evaluate_one_worker, params): params for params in grid}
            for idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                evaluation = future.result()
                consume_evaluation(idx, evaluation)

    if best is None:
        raise RuntimeError("All grid-search combinations failed or produced no total_return")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / f"grid_search_best_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    search_df = pd.DataFrame(records)
    search_results_path = output_dir / "grid_search_results.csv"
    search_df.to_csv(search_results_path, index=False)

    print(f"搜索完成，最优参数: {best['params']}，total_return={best['record']['total_return']}")
    print(f"使用最优参数执行最终保存和绘图，输出目录: {output_dir}")
    final_run = run_one(args.config, base_overrides, best["params"], save=True, output_dir=output_dir, preloaded_data=_WORKER_PRELOADED_DATA)
    backtester: SimpleBacktester = final_run["backtester"]
    final_result = final_run["result"]

    plot_path = output_dir / "backtest_plots.png"
    plot_info = BacktestPlotter(report_path=output_dir / "backtest_report.parquet", output_path=plot_path).plot(
        title="SimpleBacktester Best Grid Search Performance"
    )

    summary = {
        "best_params": best["params"],
        "best_search_record": best["record"],
        "grid_size": len(grid),
        "successful_runs": int((search_df["status"] == "success").sum()) if not search_df.empty else 0,
        "failed_runs": int((search_df["status"] == "failed").sum()) if not search_df.empty else 0,
        "final_output_dir": str(output_dir),
        "grid_search_results": str(search_results_path),
        "plot": plot_info,
        "final_analysis_summary": final_result["analysis"].get("summary", {}),
        "final_return_metrics": final_result["analysis"].get("return", {}),
        "final_risk_metrics": final_result["analysis"].get("risk", {}),
    }
    with (output_dir / "grid_search_summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, indent=2, allow_nan=True)

    print("最终保存完成")
    print(f"每日报表: {Path(backtester.output_dir) / 'backtest_report.parquet'}")
    if backtester.save_positions:
        print(f"持仓文件: {Path(backtester.output_dir) / 'positions.parquet'}")
    print(f"成交文件: {Path(backtester.output_dir) / 'trades.parquet'}")
    print(f"分析文件: {Path(backtester.output_dir) / 'analysis.json'}")
    print(f"搜索结果: {search_results_path}")
    print(f"搜索摘要: {output_dir / 'grid_search_summary.json'}")
    print(f"回测图表: {plot_info['output_path']}")
    print("最终风险指标:")
    print(json.dumps(json_safe(final_result["analysis"].get("risk", {})), ensure_ascii=False, indent=2, allow_nan=True))
    return summary


if __name__ == "__main__":
    main()
