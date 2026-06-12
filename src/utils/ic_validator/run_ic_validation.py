"""Command line entrypoint for running all IC validators on one feature.

Example
-------
python /opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/utils/ic_validator/run_ic_validation.py \
  --feature-path /opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cross_sectional_processd_data/industry_factors \
  --feature-col industry_ret_1_cc_processed \
  --label-path /opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/generated_label/daily_labels \
  --label-col label_rank_1d \
  --output-dir /opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/ic_validation/industry_ret_1 \
  --max-files 0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.ic_validator import ICValidator  # noqa: E402
from utils.ic_validator.base import BaseICValidationStep, load_local_test_pred_label  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/ic_validation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all qlib-style IC validators for a feature/label pair.")
    parser.add_argument("--feature-path", required=True, help="特征 parquet 文件或分区目录路径")
    parser.add_argument("--feature-col", required=True, help="要验证的特征列名")
    parser.add_argument("--label-path", required=True, help="标签 parquet 文件或分区目录路径")
    parser.add_argument("--label-col", required=True, help="标签列名")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"输出目录，默认 {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=0,
        help="最多读取多少个最新的匹配日度 parquet 分片；<=0 表示读取全部匹配分片。默认 0（读取全部）",
    )
    parser.add_argument("--groups", type=int, default=5, help="分层收益/turnover 分组数量，默认 5")
    parser.add_argument("--topk", type=int, default=50, help="Long-Short topk，默认 50")
    parser.add_argument("--turnover-lag", type=int, default=1, help="Turnover lag，默认 1")
    parser.add_argument("--freq", default="day", help="传给 qlib risk_analysis 的频率，默认 day")
    parser.add_argument("--reverse", action="store_true", help="分层收益中反转 score 排序方向")
    parser.add_argument(
        "--use-qlib-exchange",
        action="store_true",
        help="Long-Short 使用 qlib.contrib.evaluate.long_short_backtest；需要已初始化 qlib quote provider",
    )
    parser.add_argument("--async-run", action="store_true", help="异步按 feature 执行验证；单特征场景通常不需要")
    return parser.parse_args()


def load_pred_label(args: argparse.Namespace) -> pd.DataFrame:
    max_files = args.max_files if args.max_files > 0 else 10**12
    return load_local_test_pred_label(
        feature_dir=args.feature_path,
        label_dir=args.label_path,
        feature_col=args.feature_col,
        label_col=args.label_col,
        max_files=max_files,
    )


def build_validator(args: argparse.Namespace) -> ICValidator:
    return ICValidator(
        label_col=args.label_col,
        feature_cols=[args.feature_col],
        groups=args.groups,
        topk=args.topk,
        turnover_lag=args.turnover_lag,
        freq=args.freq,
        reverse=args.reverse,
        use_qlib_exchange=args.use_qlib_exchange,
    )


def save_summary_table(results: dict[str, Any], output_dir: Path) -> Path:
    summary_rows = results.get("summary", {}).get("table", [])
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    return summary_path


def print_brief_report(results: dict[str, Any], feature_col: str, output_dir: Path) -> None:
    print("\n========== IC Validation Finished ==========")
    print(f"Feature: {feature_col}")
    print(f"Output dir: {output_dir}")
    print("\nSummary:")
    print(pd.DataFrame(results.get("summary", {}).get("table", [])).to_string(index=False))

    feature_result = results.get("features", {}).get(feature_col, {})
    compact_metrics = {step: payload.get("metrics", {}) for step, payload in feature_result.items()}
    print("\nAll step metrics:")
    print(json.dumps(BaseICValidationStep.json_safe(compact_metrics), ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) / args.feature_col
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_label = load_pred_label(args)
    print(f"Loaded pred_label shape: {pred_label.shape}")
    print(f"Date range: {pred_label.index.get_level_values('datetime').min()} -> {pred_label.index.get_level_values('datetime').max()}")
    print(f"Instrument count: {pred_label.index.get_level_values('instrument').nunique()}")

    validator = build_validator(args)
    if args.async_run:
        results = asyncio.run(validator.async_validate(pred_label, save_dir=output_dir))
    else:
        results = validator.validate(pred_label, save_dir=output_dir)

    summary_path = save_summary_table(results, output_dir)
    print_brief_report(results, args.feature_col, output_dir)
    print(f"\nSaved metrics.json and CSV artifacts under: {output_dir}")
    print(f"Saved summary table: {summary_path}")


if __name__ == "__main__":
    # python /opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/utils/ic_validator/run_ic_validation.py \
    #   --feature-path /opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cross_sectional_processd_data/industry_factors \
    #   --feature-col industry_ret_1_cc_processed \
    #   --label-path /opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/generated_label/daily_labels \
    #   --label-col label_rank_1d \
    #   --output-dir /opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/ic_validation_csv_test \
    #   --topk 50 \
    #   --groups 5
    main()
