"""Compute IC and group returns for all 21 enhanced alpha factors.

Usage:
    python scripts/analyze_enhanced_alpha_ic.py \
        --factor-dir data/cross_sectional_processd_data/enhanced_alpha_factors \
        --label-dir data/generated_label/daily_labels \
        --label-col label_rank_20d \
        --save-dir output/enhanced_alpha_ic_analysis \
        --start-date 20200101 \
        --end-date 20251231
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.ic_validator.ic_analysis import DailyICAnalyzer
from utils.ic_validator.group_return import GroupReturnAnalyzer

logger = logging.getLogger("enhanced_alpha_ic")

FACTOR_LIST = [
    "momentum_12m_skip1m",
    "momentum_6m_skip2w",
    "momentum_52w_high_dist",
    "momentum_52w_high_break",
    "rsi_14",
    "williams_r_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "consecutive_up_days",
    "consecutive_down_days",
    "obv",
    "obv_ma5_ratio",
    "volume_price_trend",
    "volume_momentum",
    "main_net_momentum_5d",
    "main_net_trend_20d",
    "price_mf_divergence",
    "roe_qoq_change",
    "revenue_yoy_acceleration",
    "gross_margin_change",
]


def load_factor_label_data(
    factor_dir: Path,
    label_dir: Path,
    factor_col: str,
    label_col: str,
    start_date: str = "",
    end_date: str = "",
) -> pd.DataFrame:
    """Load factor + label data, merge, return MultiIndex df with score/label cols."""

    factor_cc_col = f"{factor_col}_cc_processed"

    factor_files = sorted(factor_dir.rglob("*.parquet"))
    label_files = {p.name: p for p in label_dir.rglob("*.parquet")}

    frames = []
    for fpath in factor_files:
        fname = fpath.name
        date_str = fname.replace(".parquet", "")
        if start_date and date_str < start_date:
            continue
        if end_date and date_str > end_date:
            continue
        if fname not in label_files:
            continue

        try:
            fdf = pd.read_parquet(fpath, columns=["trade_date", "ts_code", factor_cc_col])
            ldf = pd.read_parquet(label_files[fname], columns=["trade_date", "ts_code", label_col])
        except Exception as e:
            logger.warning(f"skip {fname}: {e}")
            continue

        merged = fdf.merge(ldf, on=["trade_date", "ts_code"], how="inner")
        merged = merged.dropna(subset=[factor_cc_col, label_col])
        if merged.empty:
            continue
        frames.append(merged)

    if not frames:
        raise ValueError(f"No data loaded for factor={factor_col}")

    df = pd.concat(frames, ignore_index=True)
    df["datetime"] = pd.to_datetime(df["trade_date"].astype(str))
    df["instrument"] = df["ts_code"].astype(str)
    df = df.set_index(["datetime", "instrument"])[[factor_cc_col, label_col]]
    df = df.rename(columns={factor_cc_col: "score", label_col: "label"})
    return df.sort_index()


def analyze_factor(
    factor_col: str,
    factor_dir: Path,
    label_dir: Path,
    label_col: str,
    start_date: str,
    end_date: str,
    groups: int = 10,
) -> dict:
    """Analyze single factor: IC + group returns."""

    logger.info(f"Analyzing {factor_col}...")
    try:
        df = load_factor_label_data(factor_dir, label_dir, factor_col, label_col, start_date, end_date)
    except Exception as e:
        logger.error(f"Failed to load {factor_col}: {e}")
        return {"factor": factor_col, "error": str(e)}

    n_days = df.index.get_level_values("datetime").nunique()
    n_stocks = df.index.get_level_values("instrument").nunique()
    n_rows = len(df)

    # IC analysis
    ic_analyzer = DailyICAnalyzer(score_col="score", label_col="label")
    ic_result = ic_analyzer.validate(df)
    ic_metrics = ic_result.metrics["daily"]
    rank_ic = ic_metrics.get("Rank IC", {})

    # Group return analysis
    gr_analyzer = GroupReturnAnalyzer(groups=groups, score_col="score", label_col="label")
    gr_result = gr_analyzer.validate(df)
    gr_metrics = gr_result.metrics

    mean_by_group = gr_metrics.get("mean_return_by_group", {})
    long_short_mean = mean_by_group.get("long-short", 0)
    top_group = mean_by_group.get("Group1", 0)
    bottom_group = mean_by_group.get(f"Group{groups}", 0)

    # Top/bottom IC
    daily_ic = ic_result.data["daily_ic"]["Rank IC"]
    group_df = gr_result.data["group_returns"]

    rank_ic_mean = rank_ic.get("ic_mean")
    rank_icir = rank_ic.get("icir")
    rank_ic_pos = rank_ic.get("positive_rate")
    pearson_ic_mean = ic_metrics.get("IC", {}).get("ic_mean")

    result = {
        "factor": factor_col,
        "n_days": n_days,
        "n_stocks": n_stocks,
        "n_rows": n_rows,
        "rank_ic_mean": rank_ic_mean,
        "rank_ic_std": rank_ic.get("ic_std"),
        "rank_icir": rank_icir,
        "rank_ic_positive_rate": rank_ic_pos,
        "pearson_ic_mean": pearson_ic_mean,
        "long_short_mean": long_short_mean,
        "top_group_mean": top_group,
        "bottom_group_mean": bottom_group,
        "group_monotonic_decreasing": gr_metrics.get("monotonicity", {}).get("is_decreasing_from_group1"),
        "mean_by_group": mean_by_group,
    }
    def _fmt(v):
        return f"{v:.4f}" if v is not None else "N/A"
    logger.info(
        f"  {factor_col}: RankIC={_fmt(rank_ic_mean)}, "
        f"ICIR={_fmt(rank_icir)}, "
        f"long_short={_fmt(long_short_mean)}, "
        f"top={_fmt(top_group)}, bottom={_fmt(bottom_group)}"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Enhanced alpha factor IC analysis")
    parser.add_argument("--factor-dir", required=True, help="Cross-sectional processed factor dir")
    parser.add_argument("--label-dir", required=True, help="Label data dir")
    parser.add_argument("--label-col", default="label_rank_20d", help="Label column name")
    parser.add_argument("--save-dir", required=True, help="Output directory")
    parser.add_argument("--start-date", default="", help="Start date YYYYMMDD")
    parser.add_argument("--end-date", default="", help="End date YYYYMMDD")
    parser.add_argument("--groups", type=int, default=10, help="Number of groups")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    factor_dir = Path(args.factor_dir)
    label_dir = Path(args.label_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for factor in FACTOR_LIST:
        r = analyze_factor(
            factor_col=factor,
            factor_dir=factor_dir,
            label_dir=label_dir,
            label_col=args.label_col,
            start_date=args.start_date,
            end_date=args.end_date,
            groups=args.groups,
        )
        results.append(r)

    # Summary table
    summary_rows = []
    for r in results:
        if "error" in r:
            continue
        summary_rows.append({
            "factor": r["factor"],
            "Rank IC": r["rank_ic_mean"],
            "ICIR": r["rank_icir"],
            "IC>0%": r["rank_ic_positive_rate"],
            "Long-Short": r["long_short_mean"],
            "Top Group": r["top_group_mean"],
            "Bottom Group": r["bottom_group_mean"],
            "Monotonic": r["group_monotonic_decreasing"],
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("Rank IC", ascending=False, key=lambda x: x.abs())

    print("\n" + "=" * 100)
    print(f"Enhanced Alpha Factor IC Summary (label={args.label_col}, {args.start_date or 'ALL'}-{args.end_date or 'ALL'})")
    print("=" * 100)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if isinstance(x, float) else str(x)))
    print("=" * 100 + "\n")

    # Save results
    summary_df.to_csv(save_dir / "ic_summary.csv", index=False)
    with open(save_dir / "ic_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {save_dir}")


if __name__ == "__main__":
    main()
