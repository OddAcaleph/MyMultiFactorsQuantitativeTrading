"""Command-line entry point for running the full evaluation report.

Usage
-----
python src/evaluation/run_evaluation.py \
  --pred-path outputs/.../pred_test.parquet \
  --label-dir data/generated_label/daily_labels \
  --primary-label label_5d \
  --output-dir outputs/evaluation_report \
  --run-backtest
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation import EvaluationConfig, run_evaluation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full model evaluation report.")
    parser.add_argument(
        "--pred-path",
        required=True,
        help="Path to prediction parquet file (indexed by datetime/instrument, with 'pred' column).",
    )
    parser.add_argument(
        "--label-dir",
        default=None,
        help="Path to daily label parquet directory (year=*/month=*/*.parquet). "
             "If omitted, label columns must already be present in pred-path.",
    )
    parser.add_argument(
        "--primary-label",
        default="label_5d",
        help="Primary label column for IC/hit-rate calculations (default: label_5d).",
    )
    parser.add_argument(
        "--return-labels",
        nargs="+",
        default=["label_5d", "label_10d", "label_20d"],
        help="Return label columns for top-quantile analysis (default: label_5d label_10d label_20d).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save evaluation report artifacts.",
    )
    parser.add_argument(
        "--n-groups",
        type=int,
        default=10,
        help="Number of quantile groups for top-quantile return (default: 10).",
    )
    parser.add_argument(
        "--top-ks",
        type=int,
        nargs="+",
        default=[50, 100, 200],
        help="Top-K sizes for hit-rate analysis (default: 50 100 200).",
    )
    parser.add_argument(
        "--no-backtest",
        action="store_true",
        help="Skip the SimpleBacktester long-backtest metrics.",
    )
    parser.add_argument(
        "--backtest-config",
        default=None,
        help="Path to simple_backtester config JSON.",
    )
    parser.add_argument(
        "--score-col",
        default="pred",
        help="Score column name in prediction file (default: pred).",
    )
    return parser.parse_args()


def _load_labels(label_dir: str | Path, return_labels: list[str]) -> pd.DataFrame:
    """Load label columns from daily partitioned parquet files."""
    label_path = Path(label_dir)
    dfs: list[pd.DataFrame] = []
    for parquet_file in sorted(label_path.rglob("*.parquet")):
        df = pd.read_parquet(parquet_file, columns=["trade_date", "ts_code", *return_labels])
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No parquet files found under {label_path}")
    label_df = pd.concat(dfs, ignore_index=True)
    label_df["datetime"] = pd.to_datetime(label_df["trade_date"].astype(str))
    label_df = label_df.rename(columns={"ts_code": "instrument"})
    label_df = label_df.set_index(["datetime", "instrument"]).drop(columns=["trade_date"])
    return label_df


def main() -> dict[str, Any]:
    args = parse_args()

    # Load predictions
    pred_df = pd.read_parquet(args.pred_path)
    if not isinstance(pred_df.index, pd.MultiIndex):
        raise ValueError("Prediction file must have MultiIndex (datetime, instrument)")

    # Load and merge labels if provided
    if args.label_dir:
        label_df = _load_labels(args.label_dir, args.return_labels)
        pred_label = pred_df.join(label_df, how="inner")
    else:
        pred_label = pred_df.copy()

    # Build return_cols mapping
    return_cols = {}
    for col in args.return_labels:
        # Extract horizon number from label name like "label_5d" -> "5d"
        if col.startswith("label_") and col.endswith("d"):
            horizon = col[len("label_"):]
            return_cols[horizon] = col
        else:
            return_cols[col] = col

    config = EvaluationConfig(
        score_col=args.score_col,
        primary_label_col=args.primary_label,
        return_cols=return_cols,
        n_groups=args.n_groups,
        top_ks=tuple(args.top_ks),
        run_backtest=not args.no_backtest,
        backtest_config_path=args.backtest_config,
    )

    result = run_evaluation(pred_label, config=config, output_dir=args.output_dir)

    print("=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    import json
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False, default=str))
    print("=" * 60)
    print(f"Report saved to: {args.output_dir}")

    return result


if __name__ == "__main__":
    main()
