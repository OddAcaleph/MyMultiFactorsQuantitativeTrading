"""Command-line entry point for running SimpleBacktester."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from utils import DEFAULT_SIMPLE_BACKTESTER_CONFIG_PATH, BacktestPlotter, deep_merge, load_simple_backtester_config

try:
    from .simple_backtester import SimpleBacktester
except ImportError:  # pragma: no cover - supports running this file directly.
    from backtester.simple_backtester import SimpleBacktester


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run simple A-share backtest with executable price, lot and T+1 rules.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_SIMPLE_BACKTESTER_CONFIG_PATH),
        help="Path to SimpleBacktester config JSON. Defaults to conf/simple_backtester_config.json.",
    )
    parser.add_argument(
        "--trainer-config",
        default=None,
        help="Optional trainer config JSON. Used to infer test segment when start/end are missing.",
    )
    parser.add_argument("--prediction-path", default=None, help="Prediction parquet path.")
    parser.add_argument("--price-path", default=None, help="Daily wide-table bar parquet root, e.g. data/processd_data/wide_table_daily_bars.")
    parser.add_argument(
        "--benchmark-path",
        default=None,
        help="Qlib features directory for benchmark data, e.g. /opt/tiger/qyd/qlib_data_cn/features.",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory for backtest artifacts.")
    parser.add_argument("--start-time", default=None, help="Backtest start date, e.g. 2023-01-01.")
    parser.add_argument("--end-time", default=None, help="Backtest end date, e.g. 2025-12-31.")
    parser.add_argument("--score-col", default=None, help="Prediction score column. Defaults to columns.score.")
    parser.add_argument("--topk", type=int, default=None, help="Number of target holdings. Overrides strategy.topk.")
    parser.add_argument("--n-drop", type=int, default=None, help="Number of weakest sell candidates. Overrides strategy.n_drop.")
    parser.add_argument("--account", type=float, default=None, help="Initial cash/account value.")
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Benchmark index, e.g. 000300.SH, 000001.SH, SH000300, 沪深300, 上证综指.",
    )
    parser.add_argument(
        "--freq",
        default=None,
        help="Rebalance frequency, e.g. day/week/month/10d/20d.",
    )
    parser.add_argument(
        "--signal-delay",
        type=int,
        default=None,
        help="Trading-signal delay in prediction dates. Default 1 uses date-t scores from t+1 onward to avoid look-ahead.",
    )
    parser.add_argument(
        "--price-adjustment",
        choices=["qfq", "none", "raw"],
        default=None,
        help="Price adjustment for accounting. Default qfq uses adj_factor-normalized OHLC when available.",
    )
    parser.add_argument(
        "--deal-price",
        choices=["open", "close"],
        default=None,
        help="Execute buy/sell orders at open or close. Overrides exchange_kwargs.deal_price.",
    )
    parser.add_argument("--open-cost", type=float, default=None, help="Buy-side commission rate.")
    parser.add_argument("--close-cost", type=float, default=None, help="Sell-side commission/stamp-duty rate.")
    parser.add_argument("--min-cost", type=float, default=None, help="Minimum fee per trade.")
    parser.add_argument("--slippage", type=float, default=None, help="One-way slippage rate. Buy +slippage, sell -slippage.")
    parser.add_argument("--limit-threshold", type=float, default=None, help="Limit-up/down threshold, e.g. 0.095.")
    parser.add_argument("--lot-size", type=int, default=None, help="Board lot size. A-share default is 100.")
    parser.add_argument(
        "--allow-buy-limit-up",
        action="store_true",
        help="Allow buying limit-up stocks. Default is forbidden.",
    )
    parser.add_argument(
        "--allow-buy-limit-down",
        action="store_true",
        help="Allow buying limit-down stocks. Default is forbidden.",
    )
    parser.add_argument(
        "--forbid-sell-limit-down",
        action="store_true",
        help="Forbid selling limit-down stocks. By default only zero-volume/invalid-price sells are blocked unless strategy forbids all limit trades.",
    )
    parser.add_argument("--no-save", action="store_true", help="Run backtest without saving artifacts.")
    parser.add_argument("--no-positions", action="store_true", help="Do not save positions parquet.")
    parser.add_argument("--plot", action="store_true", help="Generate backtest plot PNG after saving report.")
    parser.add_argument("--plot-output", default=None, help="Plot PNG path. Defaults to <output-dir>/backtest_plots.png.")
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Build SimpleBacktester keyword overrides while preserving nested config values."""

    config = load_simple_backtester_config(args.config)
    overrides: dict[str, Any] = {
        "trainer_config_path": args.trainer_config,
        "prediction_path": args.prediction_path,
        "price_path": args.price_path,
        "benchmark_path": args.benchmark_path,
        "output_dir": args.output_dir,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "account": args.account,
        "benchmark": args.benchmark,
        "freq": args.freq,
        "signal_delay": args.signal_delay,
        "price_adjustment": args.price_adjustment,
    }

    column_overrides = {"score": args.score_col}
    column_overrides = {key: value for key, value in column_overrides.items() if value is not None}
    if column_overrides:
        overrides["columns"] = deep_merge(config.get("columns", {}), column_overrides)

    strategy_overrides = {"topk": args.topk, "n_drop": args.n_drop}
    strategy_overrides = {key: value for key, value in strategy_overrides.items() if value is not None}
    if strategy_overrides:
        overrides["strategy"] = deep_merge(config.get("strategy", {}), strategy_overrides)

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


def json_safe(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=True)


def maybe_plot(backtester: SimpleBacktester, enabled: bool, plot_output: str | None) -> dict[str, Any] | None:
    if not enabled:
        return None

    report_path = Path(backtester.output_dir) / "backtest_report.parquet"
    output_path = Path(plot_output) if plot_output else Path(backtester.output_dir) / "backtest_plots.png"
    plotter = BacktestPlotter(report_path=report_path, output_path=output_path)
    return plotter.plot(title="SimpleBacktester Performance")


def main() -> dict[str, Any]:
    args = parse_args()
    overrides = build_overrides(args)

    backtester = SimpleBacktester(config_path=args.config, **overrides)
    result = backtester.run(save=not args.no_save)
    plot_info = maybe_plot(backtester, enabled=args.plot and not args.no_save, plot_output=args.plot_output)

    print("回测完成")
    print(f"配置文件: {args.config}")
    print(f"预测文件: {backtester.prediction_path}")
    print(f"行情目录: {backtester.price_path}")
    print(f"基准: {backtester.benchmark}")
    print(f"基准 qlib features 目录: {backtester.benchmark_path}")
    print(f"输出目录: {backtester.output_dir}")
    print(f"回测区间: {backtester.start_time.date()} ~ {backtester.end_time.date()}")
    print(f"策略参数: topk={backtester.strategy_config.get('topk')}, n_drop={backtester.strategy_config.get('n_drop')}")
    print(f"信号延迟: signal_delay={backtester.signal_delay}")
    print(f"价格复权: price_adjustment={backtester.price_adjustment}")
    print(f"成交价: {backtester.deal_price_col}")
    print(f"成本参数: open_cost={backtester.open_cost}, close_cost={backtester.close_cost}, slippage={backtester.slippage}, min_cost={backtester.min_cost}")
    print(f"交易限制: lot_size={backtester.lot_size}, limit_threshold={backtester.limit_threshold}")
    if not args.no_save:
        print(f"每日报表: {Path(backtester.output_dir) / 'backtest_report.parquet'}")
        if backtester.save_positions:
            print(f"持仓文件: {Path(backtester.output_dir) / 'positions.parquet'}")
        print(f"成交文件: {Path(backtester.output_dir) / 'trades.parquet'}")
        print(f"分析文件: {Path(backtester.output_dir) / 'analysis.json'}")
    if plot_info is not None:
        print(f"回测图表: {plot_info['output_path']}")
    elif args.plot and args.no_save:
        print("已跳过绘图：--plot 需要保存 backtest_report.parquet，请不要同时使用 --no-save。")

    print("绩效摘要:")
    print(json_safe(result["analysis"].get("summary", {})))
    print("风险指标:")
    print(json_safe(result["analysis"].get("risk", {})))
    return result


if __name__ == "__main__":
    main()
