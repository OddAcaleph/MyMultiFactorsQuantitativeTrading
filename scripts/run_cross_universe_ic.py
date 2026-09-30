"""Cross-universe IC analysis: full market, CSI300, CSI500, CSI800, and market-cap tiers.

Reads prediction parquet (MultiIndex [datetime, instrument]), index constituent
data from the raw_data directory, and computes size proxy from daily amount.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


INDEX_CODES = {
    "csi300": "000300.SH",
    "csi500": "000905.SH",
    "csi800": "000906.SH",
}

SIZE_PCT_TIERS = [10, 20, 30, 50]  # top N% by size
SIZE_ABS_THRESHOLDS = [1, 2, 5, 10, 20, 50]  # 亿元, daily amount threshold (min daily amount)


def load_index_constituents(index_path: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """Load daily index constituents between start_date and end_date (YYYYMMDD)."""
    dfs = []
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    for year_dir in sorted(index_path.glob("year=*")):
        year = int(year_dir.name.split("=")[1])
        if year < start_dt.year or year > end_dt.year:
            continue
        for month_dir in sorted(year_dir.glob("month=*")):
            month = int(month_dir.name.split("=")[1])
            for f in sorted(month_dir.glob("*.parquet")):
                date_str = f.stem
                try:
                    d = pd.Timestamp(date_str)
                except ValueError:
                    continue
                if d < start_dt or d > end_dt:
                    continue
                df = pd.read_parquet(f)
                dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["trade_date", "ts_code", "weight"])

    result = pd.concat(dfs, ignore_index=True)
    result["trade_date"] = pd.to_datetime(result["trade_date"].astype(str))
    return result


def compute_size_proxy(
    wide_table_path: Path,
    start_date: str,
    end_date: str,
    window: int = 20,
) -> pd.DataFrame:
    """Compute log of N-day median daily amount as size proxy per stock per day.

    Returns DataFrame with index [trade_date, ts_code] and column 'size_proxy'
    (log median amount, in yuan).
    """
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    # Load extra days before start_date for rolling window warmup
    warmup_start = start_dt - pd.Timedelta(days=window * 2 + 30)

    dfs = []
    for year_dir in sorted(wide_table_path.glob("year=*")):
        year = int(year_dir.name.split("=")[1])
        if year < warmup_start.year or year > end_dt.year:
            continue
        for month_dir in sorted(year_dir.glob("month=*")):
            month = int(month_dir.name.split("=")[1])
            for f in sorted(month_dir.glob("*.parquet")):
                date_str = f.stem
                try:
                    d = pd.Timestamp(date_str)
                except ValueError:
                    continue
                if d < warmup_start or d > end_dt:
                    continue
                df = pd.read_parquet(f, columns=["ts_code", "trade_date", "amount"])
                dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["trade_date", "ts_code", "size_proxy"])

    panel = pd.concat(dfs, ignore_index=True)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"].astype(str))
    panel = panel.sort_values(["ts_code", "trade_date"]).set_index(["ts_code", "trade_date"])

    # Rolling median amount per stock
    med_amount = (
        panel["amount"]
        .groupby("ts_code")
        .rolling(window=window, min_periods=max(1, window // 4))
        .median()
        .reset_index(level=0, drop=True)
    )
    # amount is in 千元 (tushare convention) — convert to 亿元 for readability
    # Actually tushare amount unit is 千元, so 亿元 = amount / 10000
    size_proxy = np.log(med_amount.replace(0, np.nan) / 10000.0)  # log of 亿元

    result = size_proxy.reset_index()
    result.columns = ["ts_code", "trade_date", "size_proxy"]
    result = result[result["trade_date"] >= start_dt]
    return result


def compute_ic_by_universe(
    pred: pd.DataFrame,
    constituents: dict[str, pd.DataFrame] | None,
    size_df: pd.DataFrame | None,
    size_pct_tiers: list[float],
    size_abs_thresholds: list[float],
    label_col: str = "label_rank_10d",
    pred_col: str = "pred",
) -> pd.DataFrame:
    """Compute IC and Rank IC for each universe.

    Returns a DataFrame with one row per universe.
    """
    pred = pred.copy()
    pred["datetime"] = pred.index.get_level_values(0)
    pred["ts_code"] = pred.index.get_level_values(1)
    pred = pred.reset_index(drop=True)

    results = []

    # Full market
    full_ic = _compute_ic_stats(pred, pred_col, label_col, "full_market")
    results.append(full_ic)

    # Index universes
    if constituents:
        for name, const_df in constituents.items():
            merged = pred.merge(
                const_df[["trade_date", "ts_code"]],
                left_on=["datetime", "ts_code"],
                right_on=["trade_date", "ts_code"],
                how="inner",
            )
            ic = _compute_ic_stats(merged, pred_col, label_col, name)
            results.append(ic)

    # Size tiers (top N% by size proxy each day)
    if size_df is not None:
        pred_with_size = pred.merge(
            size_df, left_on=["datetime", "ts_code"], right_on=["trade_date", "ts_code"], how="left"
        )
        # Daily size rank (1 = largest)
        pred_with_size["size_rank_pct"] = (
            pred_with_size.groupby("datetime")["size_proxy"]
            .rank(ascending=False, pct=True)
        )

        for thresh in size_pct_tiers:
            mask = pred_with_size["size_rank_pct"] <= (thresh / 100.0)
            subset = pred_with_size[mask]
            ic = _compute_ic_stats(
                subset, pred_col, label_col, f"top_{thresh}pct_by_size"
            )
            results.append(ic)

        # Bottom tiers
        for thresh in [30, 50]:
            mask = pred_with_size["size_rank_pct"] > (1 - thresh / 100.0)
            subset = pred_with_size[mask]
            ic = _compute_ic_stats(
                subset, pred_col, label_col, f"bottom_{thresh}pct_by_size"
            )
            results.append(ic)

        # Absolute size thresholds (filter out stocks below size proxy)
        # size_proxy is log(亿元 of 20d-median amount)
        for thresh_yi in size_abs_thresholds:
            log_thresh = np.log(thresh_yi)
            mask = pred_with_size["size_proxy"] >= log_thresh
            subset = pred_with_size[mask]
            ic = _compute_ic_stats(
                subset, pred_col, label_col, f"size_ge_{thresh_yi}yi"
            )
            results.append(ic)

    return pd.DataFrame(results)


def _compute_ic_stats(
    df: pd.DataFrame,
    pred_col: str,
    label_col: str,
    name: str,
) -> dict:
    """Compute IC, Rank IC, ICIR, win rate, etc."""
    if df.empty:
        return {
            "universe": name,
            "n_days": 0,
            "avg_stocks_per_day": 0,
            "total_rows": 0,
            "ic_mean": np.nan,
            "rank_ic_mean": np.nan,
            "ic_std": np.nan,
            "rank_ic_std": np.nan,
            "icir": np.nan,
            "rank_icir": np.nan,
            "ic_win_rate": np.nan,
            "rank_ic_win_rate": np.nan,
        }

    daily_ic = df.groupby("datetime").apply(
        lambda g: g[pred_col].corr(g[label_col], method="pearson")
    )
    daily_rank_ic = df.groupby("datetime").apply(
        lambda g: g[pred_col].corr(g[label_col], method="spearman")
    )

    daily_ic = daily_ic.dropna()
    daily_rank_ic = daily_rank_ic.dropna()

    n_days = len(daily_ic)
    avg_stocks = df.groupby("datetime").size().mean()

    return {
        "universe": name,
        "n_days": n_days,
        "avg_stocks_per_day": round(avg_stocks, 1),
        "total_rows": len(df),
        "ic_mean": round(daily_ic.mean(), 6),
        "rank_ic_mean": round(daily_rank_ic.mean(), 6),
        "ic_std": round(daily_ic.std(), 6),
        "rank_ic_std": round(daily_rank_ic.std(), 6),
        "icir": round(daily_ic.mean() / daily_ic.std() * np.sqrt(252), 4) if daily_ic.std() > 0 else np.nan,
        "rank_icir": round(daily_rank_ic.mean() / daily_rank_ic.std() * np.sqrt(252), 4) if daily_rank_ic.std() > 0 else np.nan,
        "ic_win_rate": round((daily_ic > 0).mean(), 4),
        "rank_ic_win_rate": round((daily_rank_ic > 0).mean(), 4),
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-universe IC analysis")
    parser.add_argument("--pred-path", required=True, help="Path to prediction parquet")
    parser.add_argument("--save-dir", required=True, help="Output directory")
    parser.add_argument(
        "--index-constituents-dir",
        default="/opt/tiger/qyd/MyMultiFactorsQuantitativeTrading/data/raw_data/index_constituents",
        help="Path to index constituents data",
    )
    parser.add_argument(
        "--wide-table-dir",
        default="data/cross_sectional_processd_data/wide_table_daily_bars",
        help="Path to wide table daily bars (for size proxy)",
    )
    parser.add_argument("--label-col", default="label_rank_10d")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading predictions...")
    pred = pd.read_parquet(args.pred_path)
    print(f"  Shape: {pred.shape}")
    start_date = pred.index.get_level_values(0).min().strftime("%Y%m%d")
    end_date = pred.index.get_level_values(0).max().strftime("%Y%m%d")
    print(f"  Date range: {start_date} - {end_date}")

    # Load index constituents
    print("\nLoading index constituents...")
    constituents = {}
    idx_dir = Path(args.index_constituents_dir)
    for name, code in INDEX_CODES.items():
        const_path = idx_dir / code
        if const_path.exists():
            const_df = load_index_constituents(const_path, start_date, end_date)
            constituents[name] = const_df
            print(f"  {name} ({code}): {len(const_df)} rows, {const_df['trade_date'].nunique()} days")
        else:
            print(f"  {name} ({code}): NOT FOUND")

    # Compute size proxy
    print("\nComputing size proxy (20-day median amount)...")
    size_df = compute_size_proxy(Path(args.wide_table_dir), start_date, end_date)
    print(f"  Size proxy rows: {len(size_df)}")
    print(f"  Size proxy date range: {size_df['trade_date'].min()} - {size_df['trade_date'].max()}")

    # Compute IC across universes
    print("\nComputing IC across universes...")
    result = compute_ic_by_universe(
        pred,
        constituents,
        size_df,
        size_pct_tiers=SIZE_PCT_TIERS,
        size_abs_thresholds=SIZE_ABS_THRESHOLDS,
        label_col=args.label_col,
    )

    # Save results
    out_path = save_dir / "ic_by_universe.csv"
    result.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}")
    print("\n" + "=" * 100)
    print(result.to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()
