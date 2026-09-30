"""Core evaluation metrics for model prediction assessment.

All metric functions accept a ``pred_label`` DataFrame indexed by
``(datetime, instrument)`` with at minimum a ``pred`` column and the
relevant label/return columns.  Each function returns a JSON-safe
dictionary of summary statistics and, where useful, intermediate
time-series data.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _clean_df(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    """Drop rows where any of *cols* is NaN or inf."""
    out = df.copy()
    for col in cols:
        out = out[out[col].replace([np.inf, -np.inf], np.nan).notna()]
    return out


def _ic_stats(daily_ic: pd.Series) -> dict[str, float | int | None]:
    """Summarize a daily IC series into mean / std / ICIR / positive rate."""
    clean = daily_ic.dropna()
    if clean.empty:
        return {"ic_mean": None, "ic_std": None, "icir": None, "positive_rate": None, "count": 0}
    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    return {
        "ic_mean": mean,
        "ic_std": std,
        "icir": float(mean / std) if std != 0 and pd.notna(std) else None,
        "positive_rate": float((clean > 0).mean()),
        "count": int(len(clean)),
    }


def calc_overall_ic(
    pred_label: pd.DataFrame,
    score_col: str = "pred",
    label_col: str = "label",
) -> dict[str, Any]:
    """Overall IC, Rank IC and ICIR computed cross-sectionally per day."""
    df = _clean_df(pred_label, [score_col, label_col])
    if df.empty:
        return {
            "ic": _ic_stats(pd.Series(dtype=float)),
            "rank_ic": _ic_stats(pd.Series(dtype=float)),
            "daily_ic": pd.Series(dtype=float, name="IC"),
            "daily_rank_ic": pd.Series(dtype=float, name="Rank IC"),
        }

    daily_ic = df.groupby(level="datetime", group_keys=False).apply(
        lambda x: x[score_col].corr(x[label_col], method="pearson")
    )
    daily_rank_ic = df.groupby(level="datetime", group_keys=False).apply(
        lambda x: x[score_col].corr(x[label_col], method="spearman")
    )
    daily_ic.name = "IC"
    daily_rank_ic.name = "Rank IC"

    return {
        "ic": _ic_stats(daily_ic),
        "rank_ic": _ic_stats(daily_rank_ic),
        "daily_ic": daily_ic,
        "daily_rank_ic": daily_rank_ic,
    }


def calc_long_side_ic(
    pred_label: pd.DataFrame,
    score_col: str = "pred",
    label_col: str = "label",
) -> dict[str, Any]:
    """IC computed only on stocks whose future realized return is positive."""
    df = _clean_df(pred_label, [score_col, label_col])
    if df.empty:
        return {
            "ic": _ic_stats(pd.Series(dtype=float)),
            "rank_ic": _ic_stats(pd.Series(dtype=float)),
            "daily_long_side_ic": pd.Series(dtype=float, name="long_side_ic"),
            "daily_long_side_rank_ic": pd.Series(dtype=float, name="long_side_rank_ic"),
        }

    long_df = df[df[label_col] > 0]
    if long_df.empty:
        return {
            "ic": _ic_stats(pd.Series(dtype=float)),
            "rank_ic": _ic_stats(pd.Series(dtype=float)),
            "daily_long_side_ic": pd.Series(dtype=float, name="long_side_ic"),
            "daily_long_side_rank_ic": pd.Series(dtype=float, name="long_side_rank_ic"),
        }

    daily_ic = long_df.groupby(level="datetime", group_keys=False).apply(
        lambda x: x[score_col].corr(x[label_col], method="pearson") if len(x) >= 3 else np.nan
    )
    daily_rank_ic = long_df.groupby(level="datetime", group_keys=False).apply(
        lambda x: x[score_col].corr(x[label_col], method="spearman") if len(x) >= 3 else np.nan
    )
    daily_ic.name = "long_side_ic"
    daily_rank_ic.name = "long_side_rank_ic"

    return {
        "ic": _ic_stats(daily_ic),
        "rank_ic": _ic_stats(daily_rank_ic),
        "daily_long_side_ic": daily_ic,
        "daily_long_side_rank_ic": daily_rank_ic,
    }


def calc_top_quantile_return(
    pred_label: pd.DataFrame,
    score_col: str = "pred",
    return_cols: Mapping[str, str] | None = None,
    n_groups: int = 10,
) -> dict[str, Any]:
    """Average forward return of the top quantile (decile by default)."""
    if return_cols is None:
        return_cols = {"5d": "label_5d", "10d": "label_10d", "20d": "label_20d"}

    result: dict[str, Any] = {"horizons": {}, "n_groups": n_groups}
    available = {name: col for name, col in return_cols.items() if col in pred_label.columns}
    if not available:
        return result

    df = pred_label.copy()
    df = df[df[score_col].replace([np.inf, -np.inf], np.nan).notna()]

    def _assign_group(series: pd.Series) -> pd.Series:
        n = len(series)
        ranks = series.rank(method="first", ascending=False)
        groups = np.ceil(ranks / n * n_groups).astype(int)
        groups = groups.clip(upper=n_groups)
        return pd.Series(groups, index=series.index)

    df["_group"] = df.groupby(level="datetime", group_keys=False)[score_col].apply(_assign_group)

    for horizon_name, ret_col in available.items():
        clean = df[df[ret_col].replace([np.inf, -np.inf], np.nan).notna()]
        if clean.empty:
            result["horizons"][horizon_name] = {
                "top_group_mean_return": None,
                "all_groups_mean_return": {},
            }
            continue

        top_group = clean[clean["_group"] == 1]
        daily_top = top_group.groupby(level="datetime", group_keys=False)[ret_col].mean()
        daily_top.name = f"top_{n_groups}_group_return"

        all_groups_mean = (
            clean.groupby("_group")[ret_col].mean().rename(index=lambda g: f"Group{g}").to_dict()
        )

        result["horizons"][horizon_name] = {
            "top_group_mean_return": float(daily_top.mean()) if not daily_top.empty else None,
            "top_group_daily_return": daily_top,
            "all_groups_mean_return": {k: float(v) for k, v in all_groups_mean.items()},
        }

    return result


def calc_top_k_hit_rate(
    pred_label: pd.DataFrame,
    score_col: str = "pred",
    return_col: str = "label",
    ks: Sequence[int] = (50, 100, 200),
    benchmark_returns: pd.Series | None = None,
) -> dict[str, Any]:
    """Hit-rate statistics for daily top-K predicted stocks."""
    df = _clean_df(pred_label, [score_col, return_col])
    if df.empty:
        return {str(k): {} for k in ks}

    result: dict[str, Any] = {}

    for k in ks:
        daily_median = df.groupby(level="datetime", group_keys=False)[return_col].median()

        def _top_k(group: pd.DataFrame, k: int) -> pd.DataFrame:
            return group.nlargest(k, score_col)

        topk_df = df.groupby(level="datetime", group_keys=False).apply(lambda g: _top_k(g, k))

        records: list[dict[str, float]] = []
        date_index = []
        for date, group in topk_df.groupby(level="datetime"):
            rets = group[return_col]
            med = daily_median.loc[date] if date in daily_median.index else np.nan
            entry: dict[str, float] = {
                "beat_median": float((rets > med).mean()) if pd.notna(med) else np.nan,
                "positive": float((rets > 0).mean()),
                "mean_return": float(rets.mean()),
            }
            if benchmark_returns is not None and date in benchmark_returns.index:
                bench = benchmark_returns.loc[date]
                entry["beat_benchmark"] = float((rets > bench).mean())
            records.append(entry)
            date_index.append(date)

        stats_df = pd.DataFrame(records, index=pd.DatetimeIndex(date_index))

        k_result: dict[str, Any] = {
            "beat_median_mean": float(stats_df["beat_median"].mean()) if "beat_median" in stats_df else None,
            "positive_mean": float(stats_df["positive"].mean()),
            "mean_return": float(stats_df["mean_return"].mean()),
            "daily": stats_df,
        }
        if benchmark_returns is not None and "beat_benchmark" in stats_df.columns:
            k_result["beat_benchmark_mean"] = float(stats_df["beat_benchmark"].mean())

        result[str(k)] = k_result

    return result


def calc_upside_capture(
    pred_label: pd.DataFrame,
    score_col: str = "pred",
    return_col: str = "label",
    realized_top_frac: float = 0.10,
    predicted_top_fracs: Sequence[float] = (0.10, 0.20),
) -> dict[str, Any]:
    """Upside capture: fraction of top-realized-return stocks caught by model."""
    df = _clean_df(pred_label, [score_col, return_col])
    if df.empty:
        return {str(int(frac * 100)) + "%": None for frac in predicted_top_fracs}

    result: dict[str, Any] = {
        "realized_top_frac": realized_top_frac,
        "predicted_top_fracs": list(predicted_top_fracs),
    }

    daily_capture: dict[str, pd.Series] = {}

    for pred_frac in predicted_top_fracs:
        key = f"{int(pred_frac * 100)}%"
        captures: list[float] = []
        dates: list[pd.Timestamp] = []

        for date, group in df.groupby(level="datetime"):
            n = len(group)
            if n < 10:
                continue
            n_realized_top = max(1, int(n * realized_top_frac))
            n_predicted_top = max(1, int(n * pred_frac))

            realized_top_set = set(group.nlargest(n_realized_top, return_col).index.get_level_values("instrument"))
            predicted_top_set = set(group.nlargest(n_predicted_top, score_col).index.get_level_values("instrument"))

            if not realized_top_set:
                continue
            capture = len(realized_top_set & predicted_top_set) / len(realized_top_set)
            captures.append(capture)
            dates.append(date)

        series = pd.Series(captures, index=pd.DatetimeIndex(dates), name=f"upside_capture_{key}")
        daily_capture[key] = series
        result[key] = float(series.mean()) if not series.empty else None

    result["daily"] = daily_capture
    return result


def calc_downside_filter_score(
    pred_label: pd.DataFrame,
    score_col: str = "pred",
    return_col: str = "label",
    bottom_pred_frac: float = 0.10,
    crash_threshold: float = -0.095,
) -> dict[str, Any]:
    """Downside risk identification: bottom-score stocks crash hit rate."""
    df = _clean_df(pred_label, [score_col, return_col])
    if df.empty:
        return {
            "crash_hit_rate": None,
            "crash_precision": None,
            "crash_recall": None,
            "bottom_frac": bottom_pred_frac,
            "crash_threshold": crash_threshold,
        }

    daily_hit_rates: list[float] = []
    daily_precisions: list[float] = []
    daily_recalls: list[float] = []
    dates: list[pd.Timestamp] = []

    for date, group in df.groupby(level="datetime"):
        n = len(group)
        if n < 10:
            continue

        n_bottom_count = max(1, int(n * bottom_pred_frac))
        bottom_pred = group.nsmallest(n_bottom_count, score_col)
        crashed = group[group[return_col] <= crash_threshold]

        hit_rate = (
            len(bottom_pred.index.intersection(crashed.index)) / len(bottom_pred)
            if len(bottom_pred) > 0
            else np.nan
        )
        precision = hit_rate
        recall = (
            len(bottom_pred.index.intersection(crashed.index)) / len(crashed)
            if len(crashed) > 0
            else np.nan
        )

        daily_hit_rates.append(hit_rate)
        daily_precisions.append(precision)
        daily_recalls.append(recall)
        dates.append(date)

    idx = pd.DatetimeIndex(dates)
    hit_series = pd.Series(daily_hit_rates, index=idx, name="downside_hit_rate")
    prec_series = pd.Series(daily_precisions, index=idx, name="downside_precision")
    recall_series = pd.Series(daily_recalls, index=idx, name="downside_recall")

    return {
        "crash_hit_rate": float(hit_series.mean()) if not hit_series.empty else None,
        "crash_precision": float(prec_series.mean()) if not prec_series.empty else None,
        "crash_recall": (
            float(recall_series.dropna().mean())
            if not recall_series.empty and recall_series.notna().any()
            else None
        ),
        "bottom_frac": bottom_pred_frac,
        "crash_threshold": crash_threshold,
        "daily_hit_rate": hit_series,
        "daily_precision": prec_series,
        "daily_recall": recall_series,
    }


def calc_return_decile_ic(
    pred_label: pd.DataFrame,
    score_col: str = "pred",
    return_col: str = "label",
    n_groups: int = 10,
) -> dict[str, Any]:
    """Rank IC within each decile grouped by future realized return.

    For each day, stocks are sorted by *return_col* into ``n_groups``
    equal-sized buckets (deciles by default).  Within each bucket, the
    daily Rank IC between *score_col* and *return_col* is computed.

    This metric reveals whether the model's ranking ability is uniform
    across the return distribution or concentrated in certain regions
    (e.g., only works well in the top/bottom deciles).

    Parameters
    ----------
    pred_label
        DataFrame indexed by (datetime, instrument).
    score_col
        Prediction score column name.
    return_col
        Future return column used for both grouping and IC calculation.
    n_groups
        Number of groups to split by realized return (default 10 = deciles).

    Returns
    -------
    dict
        - ``per_group_ic``: dict of group_name -> {rank_ic_mean, rank_ic_std, icir, positive_rate, count}
        - ``daily_per_group_ic``: DataFrame (date x group) of daily Rank IC
        - ``n_groups``: number of groups
    """
    df = _clean_df(pred_label, [score_col, return_col])
    if df.empty:
        return {
            "per_group_ic": {},
            "daily_per_group_ic": pd.DataFrame(),
            "n_groups": n_groups,
        }

    # For each day, assign return decile and compute within-decile rank IC
    daily_group_ic: dict[str, dict[pd.Timestamp, float]] = {}  # group_name -> {date: ic}

    for date, group in df.groupby(level="datetime"):
        n = len(group)
        if n < n_groups * 5:  # need at least ~5 stocks per group
            continue

        # Assign decile by realized return (1 = lowest return, n_groups = highest)
        ranked = group[return_col].rank(method="first")
        decile = pd.cut(ranked, bins=n_groups, labels=[f"Decile{i+1}" for i in range(n_groups)])

        for dec_name in decile.cat.categories:
            mask = decile == dec_name
            sub = group[mask]
            if len(sub) < 3:
                continue
            # Rank IC within the decile
            rank_ic = float(sub[score_col].corr(sub[return_col], method="spearman"))
            if dec_name not in daily_group_ic:
                daily_group_ic[dec_name] = {}
            daily_group_ic[dec_name][date] = rank_ic

    # Build daily DataFrame
    if daily_group_ic:
        daily_df = pd.DataFrame(daily_group_ic)
        daily_df.index.name = "datetime"
        daily_df = daily_df.sort_index()
    else:
        daily_df = pd.DataFrame()

    # Per-group summary stats
    per_group: dict[str, dict[str, float | int | None]] = {}
    for dec_name in [f"Decile{i+1}" for i in range(n_groups)]:
        if dec_name not in daily_df.columns:
            per_group[dec_name] = {
                "rank_ic_mean": None,
                "rank_ic_std": None,
                "icir": None,
                "positive_rate": None,
                "count": 0,
            }
            continue
        series = daily_df[dec_name].dropna()
        if series.empty:
            per_group[dec_name] = {
                "rank_ic_mean": None,
                "rank_ic_std": None,
                "icir": None,
                "positive_rate": None,
                "count": 0,
            }
            continue
        mean = float(series.mean())
        std = float(series.std(ddof=1))
        per_group[dec_name] = {
            "rank_ic_mean": mean,
            "rank_ic_std": std,
            "icir": float(mean / std) if std != 0 and pd.notna(std) else None,
            "positive_rate": float((series > 0).mean()),
            "count": int(len(series)),
        }

    return {
        "per_group_ic": per_group,
        "daily_per_group_ic": daily_df,
        "n_groups": n_groups,
    }
