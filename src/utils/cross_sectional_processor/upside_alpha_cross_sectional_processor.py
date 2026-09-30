"""Upside alpha factors cross-sectional processor.

Applies MAD winsorization -> z-score standardization -> industry neutralization
to each upside alpha factor column, producing ``{factor}_cc_processed`` columns.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

# Factor list (raw names, without _cc_processed suffix)
FACTOR_COLS = [
    # 1. Relative strength
    "excess_mkt_ret_5d", "excess_mkt_ret_10d", "excess_mkt_ret_20d",
    "excess_ind_ret_5d", "excess_ind_ret_10d", "excess_ind_ret_20d",
    "ind_amount_rank_20d", "ind_turnover_rank_20d",
    # 2. Momentum quality
    "ret_slope_stability_20d", "up_day_ratio_10d", "up_day_ratio_20d",
    "consecutive_up_gain_ratio", "new_high_dist_60d",
    "ma_bullish_alignment", "breakout_pullback_ratio_20d",
    # 3. Volume-price confirmation
    "amount_expansion_ratio", "amount_trend_5d", "volume_price_corr_20d",
    "up_down_volume_ratio_adv", "turnover_expansion_ratio",
    "obv_slope_20d", "ind_amount_share_change_20d",
    # 4. Industry/theme strength
    "industry_ret_rank_5d", "industry_ret_rank_20d",
    "industry_up_ratio_5d", "industry_up_ratio_20d",
    "industry_new_high_ratio_20d", "industry_amount_expansion_20d",
    "industry_leader_strength_20d", "industry_breadth_20d",
    # 5. Reversal/overheating
    "short_term_overheat_3d", "short_term_overheat_5d",
    "ma20_deviation", "high_volume_stagnation",
    "upper_shadow_ratio", "pullback_from_high_5d",
    # 6. Fundamental changes
    "ocf_yoy_change", "debt_ratio_change",
    "inventory_turnover_change", "receivable_turnover_change",
    # 7. Event-driven
    "forecast_type_score", "forecast_surprise_magnitude", "forecast_recency_30d",
    "top_list_net_amount_5d", "top_list_count_20d", "top_list_institution_ratio_20d",
    "repurchase_amount_ratio_30d", "holder_increase_ratio_30d",
    "block_trade_discount_30d", "block_trade_amount_ratio_30d",
]

KEY_COLS = ["trade_date", "ts_code"]


def mad_winsorize(s: pd.Series, n: float = 5.0) -> pd.Series:
    """MAD winsorization."""
    s = s.copy()
    valid = s.notna()
    if valid.sum() < 3:
        return s
    v = s[valid]
    med = v.median()
    mad = (v - med).abs().median()
    if mad < 1e-12:
        return s
    upper = med + n * 1.4826 * mad
    lower = med - n * 1.4826 * mad
    s[valid & (s > upper)] = upper
    s[valid & (s < lower)] = lower
    return s


def zscore(s: pd.Series) -> pd.Series:
    """Z-score standardization."""
    valid = s.notna()
    if valid.sum() < 2:
        return s
    mu = s[valid].mean()
    std = s[valid].std()
    if std < 1e-12:
        return s
    return (s - mu) / std


def industry_neutralize(df_day: pd.DataFrame, factor_col: str, ind_col: str = "industry_l1") -> pd.Series:
    """Industry neutralization via group demean."""
    result = df_day[factor_col].copy()
    for _, group in df_day.groupby(ind_col):
        if len(group) < 2:
            continue
        valid = group[factor_col].notna()
        if valid.sum() < 2:
            continue
        mean_val = group.loc[valid, factor_col].mean()
        result.loc[group.index[valid]] = group.loc[valid, factor_col] - mean_val
    return result


def process_file(
    file_path: Path,
    output_path: Path,
    industry_df: pd.DataFrame,
    logger: logging.Logger,
) -> tuple[int, int]:
    """Process a single parquet file. Returns (rows_read, rows_written)."""
    df = pd.read_parquet(file_path)
    rows_read = len(df)

    if df.empty:
        return rows_read, 0

    # Merge industry (if available)
    if industry_df is not None and "industry_l1" not in df.columns:
        df = df.merge(industry_df, on=["trade_date", "ts_code"], how="left")

    # Process each factor
    for factor in FACTOR_COLS:
        if factor not in df.columns:
            continue
        out_col = f"{factor}_cc_processed"
        # MAD winsorize
        df[out_col] = mad_winsorize(df[factor])
        # Z-score
        df[out_col] = zscore(df[out_col])
        # Industry neutralize (group by industry_l1, then z-score within group)
        if "industry_l1" in df.columns and df["industry_l1"].notna().any():
            # Demean by industry, then re-zscore
            ind_means = df.groupby("industry_l1")[out_col].transform("mean")
            df[out_col] = df[out_col] - ind_means
            # Re-zscore after neutralization
            df[out_col] = zscore(df[out_col])

    # Keep only key + processed columns
    keep_cols = KEY_COLS + [f"{f}_cc_processed" for f in FACTOR_COLS if f"{f}_cc_processed" in df.columns]
    out_df = df[keep_cols].copy()

    # Write
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)

    return rows_read, len(out_df)


def main():
    parser = argparse.ArgumentParser(description="Upside alpha factors cross-sectional processor")
    parser.add_argument("--input-dir", required=True, help="Input factor directory (partitioned parquet)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--industry-file", required=True, help="Industry data directory or file")
    parser.add_argument("--log-file", help="Log file path")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        filename=args.log_file,
    )
    logger = logging.getLogger("upside_cross_section")
    # Also log to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(console)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    ind_dir = Path(args.industry_file)

    logger.info(f"Input: {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Factors: {len(FACTOR_COLS)}")

    # Find all input files - sort by year/month/date
    input_ds = ds.dataset(input_dir, format="parquet", partitioning="hive")
    files = sorted([f for f in input_ds.files if f.endswith(".parquet")])
    # Sort by extracting year/month from path to ensure chronological order
    def sort_key(path):
        p = Path(path)
        year = 0
        month = 0
        fname = p.stem
        for part in p.parts:
            if part.startswith("year="):
                year = int(part.split("=")[1])
            elif part.startswith("month="):
                month = int(part.split("=")[1])
        return (year, month, fname)
    files.sort(key=sort_key)
    logger.info(f"Files found: {len(files)}")

    # Process by year to limit memory usage
    total_read = 0
    total_written = 0
    failed = 0
    current_year = None
    industry_df = None

    # Get L1 columns from industry dataset
    ind_ds = ds.dataset(ind_dir, format="parquet", partitioning="hive")
    l1_cols = [f.name for f in ind_ds.schema if f.name.startswith("L1_")]
    logger.info(f"Industry L1 categories: {len(l1_cols)}")
    del ind_ds

    def load_industry_year(year: int) -> pd.DataFrame:
        """Load industry data for a single year and convert one-hot to label."""
        year_ds = ds.dataset(
            ind_dir / f"year={year}",
            format="parquet",
        )
        cols = ["trade_date", "ts_code"] + l1_cols
        df = year_ds.to_table(columns=cols).to_pandas()
        # Convert one-hot to label
        df["industry_l1"] = df[l1_cols].idxmax(axis=1).str.replace("L1_", "")
        all_zero = df[l1_cols].sum(axis=1) == 0
        df.loc[all_zero, "industry_l1"] = np.nan
        df = df[["trade_date", "ts_code", "industry_l1"]]
        df["trade_date"] = pd.to_numeric(df["trade_date"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["trade_date"])
        df["trade_date"] = df["trade_date"].astype(int)
        return df

    for i, file_path in enumerate(files):
        try:
            # Determine year from path (year=YYYY)
            file_p = Path(file_path)
            year = None
            for part in file_p.parts:
                if part.startswith("year="):
                    year = int(part.split("=")[1])
                    break

            if year is None:
                raise ValueError(f"Cannot determine year from path: {file_path}")

            # Load industry data for the year if not already loaded
            if year != current_year:
                logger.info(f"Loading industry data for year {year}...")
                try:
                    industry_df = load_industry_year(year)
                    current_year = year
                    logger.info(f"  Industry rows: {len(industry_df)}")
                except FileNotFoundError:
                    logger.warning(f"  No industry data for year {year}, skipping industry neutralization")
                    industry_df = None
                    current_year = year

            # Compute output path preserving partition structure
            rel_path = file_p.relative_to(input_dir)
            out_path = output_dir / rel_path

            rows_read, rows_written = process_file(
                file_p, out_path, industry_df, logger
            )
            total_read += rows_read
            total_written += rows_written

            if (i + 1) % 200 == 0:
                logger.info(f"Progress: {i+1}/{len(files)} files, {total_written:,} rows written")
        except Exception as e:
            failed += 1
            logger.error(f"Failed: {file_path}: {e}")
            import traceback
            logger.error(traceback.format_exc())

    logger.info(f"Done: {len(files)-failed}/{len(files)} files, {total_read:,} read, {total_written:,} written")
    if failed:
        logger.error(f"Failed files: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
