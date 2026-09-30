"""Decile analysis: compute IC within each return decile to check if model
predicts tail events (big drops / high volatility stocks) better.

For each trading day:
  1. Sort stocks by forward 10-day return into 10 deciles (decile 1 = worst, decile 10 = best)
  2. Compute Rank IC within each decile
  3. Also compute IC by absolute return magnitude (big gain / big loss / middle)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_labels(label_dir: Path, start_date: str, end_date: str) -> pd.DataFrame:
    """Load daily labels (raw forward returns)."""
    dfs = []
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    for year_dir in sorted(label_dir.glob("year=*")):
        year = int(year_dir.name.split("=")[1])
        if year < start_dt.year or year > end_dt.year:
            continue
        for month_dir in sorted(year_dir.glob("month=*")):
            for f in sorted(month_dir.glob("*.parquet")):
                date_str = f.stem
                try:
                    d = pd.Timestamp(date_str)
                except ValueError:
                    continue
                if d < start_dt or d > end_dt:
                    continue
                df = pd.read_parquet(f, columns=["trade_date", "ts_code", "label_10d"])
                dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    result = pd.concat(dfs, ignore_index=True)
    result["trade_date"] = pd.to_datetime(result["trade_date"].astype(str))
    return result


def decile_ic_analysis(
    pred: pd.DataFrame,
    ret: pd.DataFrame,
    n_deciles: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute IC within each return decile.

    Returns:
        decile_stats: DataFrame with IC stats per decile
        daily_decile_ic: daily IC per decile (for time series analysis)
    """
    pred = pred.copy()
    pred["datetime"] = pred.index.get_level_values(0)
    pred["ts_code"] = pred.index.get_level_values(1)
    pred = pred.reset_index(drop=True)

    merged = pred.merge(
        ret, left_on=["datetime", "ts_code"], right_on=["trade_date", "ts_code"], how="inner"
    )
    merged = merged.dropna(subset=["label_10d", "pred"])

    # Assign decile per day (1 = lowest return, 10 = highest return)
    merged["ret_decile"] = (
        merged.groupby("datetime")["label_10d"]
        .rank(pct=True, method="first")
        .apply(lambda x: min(int(x * n_deciles) + 1, n_deciles))
    )

    # Daily IC per decile
    daily_ic_list = []
    for (dt, dec), group in merged.groupby(["datetime", "ret_decile"]):
        if len(group) < 20:
            continue
        rank_ic = group["pred"].corr(group["label_10d"], method="spearman")
        pearson_ic = group["pred"].corr(group["label_10d"], method="pearson")
        daily_ic_list.append({
            "datetime": dt,
            "decile": dec,
            "n_stocks": len(group),
            "rank_ic": rank_ic,
            "pearson_ic": pearson_ic,
            "mean_ret": group["label_10d"].mean(),
            "std_ret": group["label_10d"].std(),
            "mean_pred": group["pred"].mean(),
        })

    daily_decile_ic = pd.DataFrame(daily_ic_list)

    # Aggregate stats per decile
    decile_stats = daily_decile_ic.groupby("decile").agg(
        n_days=("rank_ic", "count"),
        avg_stocks=("n_stocks", "mean"),
        avg_ret=("mean_ret", "mean"),
        avg_ret_std=("std_ret", "mean"),
        rank_ic_mean=("rank_ic", "mean"),
        rank_ic_std=("rank_ic", "std"),
        pearson_ic_mean=("pearson_ic", "mean"),
        pearson_ic_std=("pearson_ic", "std"),
        rank_ic_win_rate=("rank_ic", lambda x: (x > 0).mean()),
    ).reset_index()

    decile_stats["rank_icir"] = (
        decile_stats["rank_ic_mean"] / decile_stats["rank_ic_std"] * np.sqrt(252)
    )
    decile_stats["pearson_icir"] = (
        decile_stats["pearson_ic_mean"] / decile_stats["pearson_ic_std"] * np.sqrt(252)
    )

    return decile_stats, daily_decile_ic


def tail_vs_middle_analysis(
    pred: pd.DataFrame,
    ret: pd.DataFrame,
) -> pd.DataFrame:
    """Compare IC in tail vs middle of return distribution.

    Buckets:
      - big_loss: bottom 10% by return
      - loss: bottom 10-30%
      - middle: middle 40% (30-70%)
      - gain: top 30-10%
      - big_gain: top 10%
    """
    pred = pred.copy()
    pred["datetime"] = pred.index.get_level_values(0)
    pred["ts_code"] = pred.index.get_level_values(1)
    pred = pred.reset_index(drop=True)

    merged = pred.merge(
        ret, left_on=["datetime", "ts_code"], right_on=["trade_date", "ts_code"], how="inner"
    )
    merged = merged.dropna(subset=["label_10d", "pred"])

    # Assign bucket per day
    merged["ret_pct"] = merged.groupby("datetime")["label_10d"].rank(pct=True)

    def assign_bucket(pct):
        if pct <= 0.1:
            return "big_loss (bottom 10%)"
        elif pct <= 0.3:
            return "loss (10-30%)"
        elif pct <= 0.7:
            return "middle (30-70%)"
        elif pct <= 0.9:
            return "gain (70-90%)"
        else:
            return "big_gain (top 10%)"

    merged["bucket"] = merged["ret_pct"].apply(assign_bucket)

    results = []
    for bucket in ["big_loss (bottom 10%)", "loss (10-30%)", "middle (30-70%)",
                   "gain (70-90%)", "big_gain (top 10%)"]:
        subset = merged[merged["bucket"] == bucket]
        daily_ic = subset.groupby("datetime").apply(
            lambda g: g["pred"].corr(g["label_10d"], method="spearman"),
            include_groups=False,
        ).dropna()

        daily_ret_std = subset.groupby("datetime")["label_10d"].std().mean()

        results.append({
            "bucket": bucket,
            "n_days": len(daily_ic),
            "avg_stocks_per_day": len(subset) / len(daily_ic) if len(daily_ic) > 0 else 0,
            "avg_daily_ret_std": daily_ret_std,
            "rank_ic_mean": daily_ic.mean(),
            "rank_ic_std": daily_ic.std(),
            "rank_icir": daily_ic.mean() / daily_ic.std() * np.sqrt(252) if daily_ic.std() > 0 else np.nan,
            "rank_ic_win_rate": (daily_ic > 0).mean(),
        })

    return pd.DataFrame(results)


def volatility_decile_analysis(
    pred: pd.DataFrame,
    ret: pd.DataFrame,
    wide_table_dir: Path,
    n_deciles: int = 5,
) -> pd.DataFrame:
    """Compute IC by volatility decile (20-day rolling vol of returns)."""
    # Load daily close prices to compute volatility
    dfs = []
    start_dt = pred.index.get_level_values(0).min()
    end_dt = pred.index.get_level_values(0).max()
    warmup_start = start_dt - pd.Timedelta(days=60)

    for year_dir in sorted(wide_table_dir.glob("year=*")):
        year = int(year_dir.name.split("=")[1])
        if year < warmup_start.year or year > end_dt.year:
            continue
        for month_dir in sorted(year_dir.glob("month=*")):
            for f in sorted(month_dir.glob("*.parquet")):
                date_str = f.stem
                try:
                    d = pd.Timestamp(date_str)
                except ValueError:
                    continue
                if d < warmup_start or d > end_dt:
                    continue
                df = pd.read_parquet(f, columns=["ts_code", "trade_date", "close", "adj_factor"])
                dfs.append(df)

    panel = pd.concat(dfs, ignore_index=True)
    panel["trade_date"] = pd.to_datetime(panel["trade_date"].astype(str))
    panel = panel.sort_values(["ts_code", "trade_date"])

    # Compute adj close and daily returns
    panel["adj_close"] = panel["close"] * panel["adj_factor"]
    panel["daily_ret"] = panel.groupby("ts_code")["adj_close"].pct_change()

    # 20-day rolling volatility
    panel["vol_20d"] = (
        panel.groupby("ts_code")["daily_ret"]
        .rolling(window=20, min_periods=10)
        .std()
        .reset_index(level=0, drop=True)
    )

    panel = panel[panel["trade_date"] >= start_dt]
    vol_df = panel[["trade_date", "ts_code", "vol_20d"]].copy()

    # Merge with pred
    pred_df = pred.copy()
    pred_df["datetime"] = pred_df.index.get_level_values(0)
    pred_df["ts_code"] = pred_df.index.get_level_values(1)
    pred_df = pred_df.reset_index(drop=True)

    merged = pred_df.merge(
        ret, left_on=["datetime", "ts_code"], right_on=["trade_date", "ts_code"], how="inner"
    )
    merged = merged.merge(vol_df, left_on=["datetime", "ts_code"], right_on=["trade_date", "ts_code"], how="left")
    merged = merged.dropna(subset=["label_10d", "pred", "vol_20d"])

    # Assign vol decile per day (1 = lowest vol, 10 = highest vol)
    merged["vol_decile"] = (
        merged.groupby("datetime")["vol_20d"]
        .rank(pct=True, method="first")
        .apply(lambda x: min(int(x * n_deciles) + 1, n_deciles))
    )

    results = []
    for dec in range(1, n_deciles + 1):
        subset = merged[merged["vol_decile"] == dec]
        daily_ic = subset.groupby("datetime").apply(
            lambda g: g["pred"].corr(g["label_10d"], method="spearman"),
            include_groups=False,
        ).dropna()

        avg_vol = subset["vol_20d"].mean()

        results.append({
            "vol_decile": dec,
            "n_days": len(daily_ic),
            "avg_stocks_per_day": len(subset) / len(daily_ic) if len(daily_ic) > 0 else 0,
            "avg_20d_vol": avg_vol,
            "rank_ic_mean": daily_ic.mean(),
            "rank_ic_std": daily_ic.std(),
            "rank_icir": daily_ic.mean() / daily_ic.std() * np.sqrt(252) if daily_ic.std() > 0 else np.nan,
            "rank_ic_win_rate": (daily_ic > 0).mean(),
        })

    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Decile IC analysis by return / volatility")
    parser.add_argument("--pred-path", required=True)
    parser.add_argument("--save-dir", required=True)
    parser.add_argument(
        "--label-dir",
        default="data/generated_label/daily_labels",
    )
    parser.add_argument(
        "--wide-table-dir",
        default="data/cross_sectional_processd_data/wide_table_daily_bars",
    )
    parser.add_argument("--n-deciles", type=int, default=10)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Loading predictions...")
    pred = pd.read_parquet(args.pred_path)
    start_date = pred.index.get_level_values(0).min().strftime("%Y%m%d")
    end_date = pred.index.get_level_values(0).max().strftime("%Y%m%d")
    print(f"  Shape: {pred.shape}, {start_date} ~ {end_date}")

    print("\nLoading labels (10-day forward returns)...")
    ret = load_labels(Path(args.label_dir), start_date, end_date)
    print(f"  Rows: {len(ret)}")

    # 1. Return decile IC
    print("\n=== Return Decile IC Analysis ===")
    decile_stats, daily_decile_ic = decile_ic_analysis(pred, ret, n_deciles=args.n_deciles)
    decile_stats.to_csv(save_dir / "ic_by_return_decile.csv", index=False)
    print(decile_stats.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 2. Tail vs middle
    print("\n=== Tail vs Middle IC Analysis ===")
    tail_stats = tail_vs_middle_analysis(pred, ret)
    tail_stats.to_csv(save_dir / "ic_tail_vs_middle.csv", index=False)
    print(tail_stats.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 3. Volatility decile IC
    print("\n=== Volatility Decile IC Analysis ===")
    vol_stats = volatility_decile_analysis(pred, ret, Path(args.wide_table_dir), n_deciles=5)
    vol_stats.to_csv(save_dir / "ic_by_volatility_decile.csv", index=False)
    print(vol_stats.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nAll results saved to {save_dir}/")


if __name__ == "__main__":
    main()
