"""Command-line entry point for running Preparatory backtest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from utils import DEFAULT_BACKTESTER_CONFIG_PATH, BacktestPlotter, deep_merge, load_backtester_config

try:
    from .preparatory_backtester import PreparatoryBacktester
except ImportError:  # pragma: no cover - supports running this file directly.
    from backtester.preparatory_backtester import PreparatoryBacktester


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Preparatory TopkDropout backtest with project config.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_BACKTESTER_CONFIG_PATH),
        help="Path to preparatory backtester config JSON.",
    )
    parser.add_argument(
        "--trainer-config",
        default=None,
        help="Optional path to trainer config JSON. Used to infer test segment when start/end are missing.",
    )
    parser.add_argument(
        "--prediction-path",
        default=None,
        help="Optional prediction parquet path. Overrides preparatory backtester config prediction_path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for backtest artifacts.",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="Optional backtest start date, e.g. 2023-01-01.",
    )
    parser.add_argument(
        "--end-time",
        default=None,
        help="Optional backtest end date, e.g. 2025-12-31.",
    )
    parser.add_argument(
        "--score-col",
        default=None,
        help="Prediction score column name. Overrides columns.score.",
    )
    parser.add_argument(
        "--label-col",
        default=None,
        help="Realized return/label column name. Overrides columns.label.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="Number of instruments to hold. Overrides strategy.topk.",
    )
    parser.add_argument(
        "--n-drop",
        type=int,
        default=None,
        help="Number of weakest held instruments to drop on each rebalance day. Overrides strategy.n_drop.",
    )
    parser.add_argument(
        "--account",
        type=float,
        default=None,
        help="Initial account value. Overrides account.",
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Benchmark name. Use market_mean for cross-sectional market average, or an instrument code in labels.",
    )
    parser.add_argument(
        "--freq",
        default=None,
        help="Frequency passed to qlib risk_analysis, e.g. day.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run backtest without saving report, positions and analysis JSON.",
    )
    parser.add_argument(
        "--no-positions",
        action="store_true",
        help="Do not save positions parquet.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate backtest plot PNG after saving backtest report.",
    )
    parser.add_argument(
        "--plot-output",
        default=None,
        help="Optional plot PNG path. Defaults to <output-dir>/backtest_plots.png.",
    )
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Build PreparatoryBacktester keyword overrides while preserving nested config values."""

    config = load_backtester_config(args.config)
    overrides: dict[str, Any] = {
        "trainer_config_path": args.trainer_config,
        "prediction_path": args.prediction_path,
        "output_dir": args.output_dir,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "account": args.account,
        "benchmark": args.benchmark,
        "freq": args.freq,
    }

    column_overrides = {"score": args.score_col, "label": args.label_col}
    column_overrides = {key: value for key, value in column_overrides.items() if value is not None}
    if column_overrides:
        overrides["columns"] = deep_merge(config.get("columns", {}), column_overrides)

    strategy_overrides = {"topk": args.topk, "n_drop": args.n_drop}
    strategy_overrides = {key: value for key, value in strategy_overrides.items() if value is not None}
    if strategy_overrides:
        overrides["strategy"] = deep_merge(config.get("strategy", {}), strategy_overrides)

    if args.no_positions:
        overrides["save_positions"] = False

    return {key: value for key, value in overrides.items() if value is not None}


def json_safe(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=True)


def maybe_plot(backtester: PreparatoryBacktester, enabled: bool, plot_output: str | None) -> dict[str, Any] | None:
    if not enabled:
        return None

    report_path = Path(backtester.output_dir) / "backtest_report.parquet"
    output_path = Path(plot_output) if plot_output else Path(backtester.output_dir) / "backtest_plots.png"
    plotter = BacktestPlotter(report_path=report_path, output_path=output_path)
    return plotter.plot(title="PreparatoryBacktester Performance")


def main() -> dict[str, Any]:
    args = parse_args()
    overrides = build_overrides(args)

    backtester = PreparatoryBacktester(config_path=args.config, **overrides)
    result = backtester.run(save=not args.no_save)
    plot_info = maybe_plot(backtester, enabled=args.plot and not args.no_save, plot_output=args.plot_output)

    print("回测完成")
    print(f"配置文件: {args.config}")
    print(f"预测文件: {backtester.prediction_path}")
    print(f"输出目录: {backtester.output_dir}")
    print(f"回测区间: {backtester.start_time.date()} ~ {backtester.end_time.date()}")
    print(f"策略参数: topk={backtester.strategy_config.get('topk')}, n_drop={backtester.strategy_config.get('n_drop')}")
    print(f"分数字段: {backtester.score_col}")
    print(f"标签字段: {backtester.label_col}")
    if not args.no_save:
        print(f"每日报表: {Path(backtester.output_dir) / 'backtest_report.parquet'}")
        if backtester.save_positions:
            print(f"持仓文件: {Path(backtester.output_dir) / 'positions.parquet'}")
        print(f"分析文件: {Path(backtester.output_dir) / 'analysis.json'}")
    if plot_info is not None:
        print(f"回测图表: {plot_info['output_path']}")
    elif args.plot and args.no_save:
        print("已跳过绘图：--plot 需要保存 backtest_report.parquet，请不要同时使用 --no-save。")

    print("绩效摘要:")
    print(json_safe(result["analysis"].get("summary", {})))
    return result


if __name__ == "__main__":
    main()

