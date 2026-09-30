"""grid0188 模型深度验证：行业中性 IC、市场环境分层、多头诊断。

验证一：行业中性化 IC
  - 原始 IC（基准）
  - 行业中性预测 IC（预测值减去行业均值后算 IC）
  - 行业内 IC（每天每个行业内单独算 IC，按行业市值/数量加权平均）

验证二：市场环境分层
  - 按中证全指（000985）20 日收益划分：上涨市 / 下跌市 / 震荡市
  - 按 20 日波动率划分：高波动 / 低波动
  - 每个环境下的 IC、多头收益、空头收益

验证三：多头诊断
  - long_side_ic（仅未来收益为正的股票中计算 IC）
  - short_side_ic（仅未来收益为负的股票中计算 IC）
  - top_k_return（Top 50/100/200 平均未来收益）
  - top_decile_return（预测分最高 10% 平均未来收益）
  - bottom_decile_return（预测分最低 10% 平均未来收益）
  - upside_capture（真实涨幅前 10% 被预测进前 10%/20% 的比例）
  - top_group_excess（Top 组相对全市场、行业中位数超额）
  - top_k_hit_rate（Top-K 中跑赢全市场中位数的比例）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# 数据加载
# ──────────────────────────────────────────────

def load_predictions(pred_path: str) -> pd.DataFrame:
    df = pd.read_parquet(pred_path)
    if df.index.names != ['datetime', 'instrument']:
        df = df.set_index(['datetime', 'instrument'])
    df.index = df.index.set_levels([pd.to_datetime(df.index.levels[0]), df.index.levels[1]])
    return df


def load_industry_data(start_date: pd.Timestamp, end_date: pd.Timestamp,
                       industry_dir: str) -> pd.DataFrame:
    """加载行业分类数据，返回 MultiIndex [datetime, instrument] -> l1_name"""
    industry_path = Path(industry_dir)
    records = []
    for year_dir in sorted(industry_path.glob("year=*")):
        year = int(year_dir.name.split("=")[1])
        if year < start_date.year - 1 or year > end_date.year + 1:
            continue
        for month_dir in sorted(year_dir.glob("month=*")):
            for f in sorted(month_dir.glob("*.parquet")):
                df = pd.read_parquet(f)
                if "trade_date" in df.columns and "ts_code" in df.columns:
                    df["datetime"] = pd.to_datetime(df["trade_date"].astype(str))
                    df = df.rename(columns={"ts_code": "instrument"})
                    records.append(df[["datetime", "instrument", "l1_name"]])
    full = pd.concat(records, ignore_index=True)
    full = full.set_index(["datetime", "instrument"])
    full = full.sort_index()
    # 过滤日期范围
    dates = full.index.get_level_values(0)
    full = full[(dates >= start_date) & (dates <= end_date)]
    return full


def load_index_daily(index_code: str, index_dir: str) -> pd.DataFrame:
    """加载指数日线数据，返回 datetime 索引的 DataFrame"""
    idx_path = Path(index_dir) / index_code
    records = []
    for year_dir in sorted(idx_path.glob("year=*")):
        for month_dir in sorted(year_dir.glob("month=*")):
            for f in sorted(month_dir.glob("*.parquet")):
                df = pd.read_parquet(f)
                records.append(df)
    if not records:
        return pd.DataFrame()
    full = pd.concat(records, ignore_index=True)
    full["datetime"] = pd.to_datetime(full["trade_date"].astype(str))
    full = full.set_index("datetime").sort_index()
    return full


# ──────────────────────────────────────────────
# 验证一：行业中性化 IC
# ──────────────────────────────────────────────

def rank_ic(x: pd.Series, y: pd.Series) -> float:
    if len(x) < 10:
        return np.nan
    return float(x.rank().corr(y.rank()))


def raw_ic_daily(pred_df: pd.DataFrame, label_col: str) -> pd.Series:
    """原始 Rank IC（每日）"""
    ic_series = pred_df.groupby(level="datetime").apply(
        lambda g: rank_ic(g["pred"], g[label_col])
    )
    ic_series.name = "raw_ic"
    return ic_series


def industry_neutral_ic_daily(pred_df: pd.DataFrame, industry_df: pd.DataFrame,
                              label_col: str) -> pd.Series:
    """行业中性预测 IC：预测值减去行业均值后再算 Rank IC"""
    merged = pred_df.join(industry_df["l1_name"], how="inner")
    # 减去行业均值
    merged["pred_ind_mean"] = merged.groupby(["datetime", "l1_name"])["pred"].transform("mean")
    merged["pred_neutral"] = merged["pred"] - merged["pred_ind_mean"]
    merged["label_ind_mean"] = merged.groupby(["datetime", "l1_name"])[label_col].transform("mean")
    merged["label_neutral"] = merged[label_col] - merged["label_ind_mean"]

    ic_series = merged.groupby(level="datetime").apply(
        lambda g: rank_ic(g["pred_neutral"], g["label_neutral"])
    )
    ic_series.name = "ind_neutral_ic"
    return ic_series


def within_industry_ic_daily(pred_df: pd.DataFrame, industry_df: pd.DataFrame,
                             label_col: str) -> pd.DataFrame:
    """行业内 IC：每天每个行业单独算 IC，然后按行业股票数加权平均"""
    merged = pred_df.join(industry_df["l1_name"], how="inner")

    def _daily_ind_ic(g):
        results = {}
        for ind, sub in g.groupby("l1_name"):
            if len(sub) >= 10:
                results[ind] = (rank_ic(sub["pred"], sub[label_col]), len(sub))
        if not results:
            return pd.Series({"within_ind_ic": np.nan, "n_industries": 0})
        ics = [v[0] for v in results.values() if not np.isnan(v[0])]
        weights = [v[1] for v in results.values() if not np.isnan(v[0])]
        if not ics:
            return pd.Series({"within_ind_ic": np.nan, "n_industries": len(results)})
        weighted = np.average(ics, weights=weights)
        return pd.Series({"within_ind_ic": weighted, "n_industries": len(results)})

    result = merged.groupby(level="datetime").apply(_daily_ind_ic)
    return result


def industry_exposure_daily(pred_df: pd.DataFrame, industry_df: pd.DataFrame) -> pd.DataFrame:
    """行业暴露分析：预测值排名前 10% 的股票行业分布 vs 全市场分布"""
    merged = pred_df.join(industry_df["l1_name"], how="inner")

    def _daily_exposure(g):
        n = len(g)
        top_n = max(1, int(n * 0.1))
        top_stocks = g.nlargest(top_n, "pred")
        top_dist = top_stocks["l1_name"].value_counts(normalize=True)
        all_dist = g["l1_name"].value_counts(normalize=True)
        diff = (top_dist - all_dist).sort_values(ascending=False)
        # 只返回 top3 超配和 bottom3 低配
        top3 = diff.head(3)
        bot3 = diff.tail(3)
        result = {}
        for i, (ind, val) in enumerate(top3.items()):
            result[f"top_overweight_{i+1}_ind"] = ind
            result[f"top_overweight_{i+1}_pct"] = val
        for i, (ind, val) in enumerate(bot3.items()):
            result[f"top_underweight_{i+1}_ind"] = ind
            result[f"top_underweight_{i+1}_pct"] = val
        return pd.Series(result)

    return merged.groupby(level="datetime").apply(_daily_exposure)


# ──────────────────────────────────────────────
# 验证二：市场环境分层
# ──────────────────────────────────────────────

def classify_market_env(index_df: pd.DataFrame) -> pd.DataFrame:
    """根据指数走势划分市场环境

    返回列：
      - ret_20d: 20 日收益率
      - vol_20d: 20 日波动率（年化）
      - trend_regime: up / down / range
      - vol_regime: high / low
      - combined: 组合标签（如 up_high, down_low 等）
    """
    df = index_df.copy()
    df["ret_1d"] = df["close"].pct_change()
    df["ret_20d"] = df["close"].pct_change(20)
    df["vol_20d"] = df["ret_1d"].rolling(20).std() * np.sqrt(252)

    # 趋势划分
    df["trend_regime"] = "range"
    df.loc[df["ret_20d"] > 0.03, "trend_regime"] = "up"
    df.loc[df["ret_20d"] < -0.03, "trend_regime"] = "down"

    # 波动率划分（按历史分位数）
    vol_median = df["vol_20d"].median()
    df["vol_regime"] = np.where(df["vol_20d"] > vol_median, "high", "low")

    df["combined"] = df["trend_regime"] + "_" + df["vol_regime"]
    return df[["ret_20d", "vol_20d", "trend_regime", "vol_regime", "combined"]]


def ic_by_env(ic_series: pd.Series, env_df: pd.DataFrame) -> pd.DataFrame:
    """按市场环境分组统计 IC"""
    combined = ic_series.to_frame("ic").join(env_df, how="inner")

    def _stats(g):
        ic_vals = g["ic"].dropna()
        return pd.Series({
            "n_days": len(ic_vals),
            "ic_mean": ic_vals.mean(),
            "ic_std": ic_vals.std(),
            "ic_ir": ic_vals.mean() / ic_vals.std() * np.sqrt(252) if ic_vals.std() > 0 else np.nan,
            "ic_win_rate": (ic_vals > 0).mean(),
        })

    result = combined.groupby("trend_regime").apply(_stats)
    result.index.name = "regime"
    return result


def long_short_return_by_env(pred_df: pd.DataFrame, label_col: str,
                             env_df: pd.DataFrame, top_pct: float = 0.1) -> pd.DataFrame:
    """按市场环境分组统计多头/空头收益"""
    # 计算每日 Top/Bottom 组收益
    def _daily_ret(g):
        n = len(g)
        top_n = max(1, int(n * top_pct))
        bot_n = max(1, int(n * top_pct))
        top_ret = g.nlargest(top_n, "pred")[label_col].mean()
        bot_ret = g.nsmallest(bot_n, "pred")[label_col].mean()
        med_ret = g[label_col].median()
        return pd.Series({
            "top_ret": top_ret,
            "bottom_ret": bot_ret,
            "long_short": top_ret - bot_ret,
            "top_excess": top_ret - med_ret,
            "market_median": med_ret,
        })

    daily_ret = pred_df.groupby(level="datetime").apply(_daily_ret)
    combined = daily_ret.join(env_df, how="inner")

    def _stats(g):
        return pd.Series({
            "n_days": len(g),
            "top_ret_mean": g["top_ret"].mean(),
            "bottom_ret_mean": g["bottom_ret"].mean(),
            "long_short_mean": g["long_short"].mean(),
            "top_excess_mean": g["top_excess"].mean(),
            "market_median_mean": g["market_median"].mean(),
        })

    result = combined.groupby("trend_regime").apply(_stats)
    result.index.name = "regime"
    return result


# ──────────────────────────────────────────────
# 验证三：多头诊断
# ──────────────────────────────────────────────

def long_side_ic_daily(pred_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """多空两侧 IC 分解"""
    def _daily(g):
        # label 是 rank 值，0.5 以上视为未来跑赢中位数
        long_mask = g[label_col] > 0.5
        short_mask = g[label_col] < 0.5
        long_ic = rank_ic(g.loc[long_mask, "pred"], g.loc[long_mask, label_col]) if long_mask.sum() >= 10 else np.nan
        short_ic = rank_ic(g.loc[short_mask, "pred"], g.loc[short_mask, label_col]) if short_mask.sum() >= 10 else np.nan
        return pd.Series({
            "long_side_ic": long_ic,
            "short_side_ic": short_ic,
            "n_long": long_mask.sum(),
            "n_short": short_mask.sum(),
        })

    return pred_df.groupby(level="datetime").apply(_daily)


def top_k_analysis_daily(pred_df: pd.DataFrame, label_col: str,
                         ks: list[int] = [50, 100, 200]) -> pd.DataFrame:
    """Top-K 收益分析"""
    def _daily(g):
        g_sorted = g.sort_values("pred", ascending=False)
        n = len(g)
        result = {}
        for k in ks:
            k_actual = min(k, n)
            top_k_ret = g_sorted.iloc[:k_actual][label_col].mean()
            result[f"top_{k}_ret"] = top_k_ret
            result[f"top_{k}_hit_rate"] = (g_sorted.iloc[:k_actual][label_col] > g[label_col].median()).mean()
        # 十分位
        decile_size = max(1, n // 10)
        result["top_decile_ret"] = g_sorted.iloc[:decile_size][label_col].mean()
        result["bottom_decile_ret"] = g_sorted.iloc[-decile_size:][label_col].mean()
        result["decile_spread"] = result["top_decile_ret"] - result["bottom_decile_ret"]
        result["market_median"] = g[label_col].median()
        result["market_mean"] = g[label_col].mean()
        return pd.Series(result)

    return pred_df.groupby(level="datetime").apply(_daily)


def upside_capture_daily(pred_df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """上行捕获率：真实涨幅前 N% 的股票中有多少被预测进前 M%"""
    def _daily(g):
        n = len(g)
        if n < 50:
            return pd.Series({"upside_capture_top10_in10": np.nan,
                              "upside_capture_top10_in20": np.nan,
                              "downside_capture_bot10_in10": np.nan})
        top10_n = max(1, int(n * 0.1))
        top20_n = max(1, int(n * 0.2))

        true_top10 = set(g.nlargest(top10_n, label_col).index.get_level_values("instrument"))
        pred_top10 = set(g.nlargest(top10_n, "pred").index.get_level_values("instrument"))
        pred_top20 = set(g.nlargest(top20_n, "pred").index.get_level_values("instrument"))

        true_bot10 = set(g.nsmallest(top10_n, label_col).index.get_level_values("instrument"))
        pred_bot10 = set(g.nsmallest(top10_n, "pred").index.get_level_values("instrument"))

        return pd.Series({
            "upside_capture_top10_in10": len(true_top10 & pred_top10) / len(true_top10),
            "upside_capture_top10_in20": len(true_top10 & pred_top20) / len(true_top10),
            "downside_capture_bot10_in10": len(true_bot10 & pred_bot10) / len(true_bot10),
        })

    return pred_df.groupby(level="datetime").apply(_daily)


def top_group_industry_excess_daily(pred_df: pd.DataFrame, industry_df: pd.DataFrame,
                                    label_col: str, top_pct: float = 0.1) -> pd.Series:
    """Top 组相对行业中位数的超额收益"""
    merged = pred_df.join(industry_df["l1_name"], how="inner")

    def _daily(g):
        n = len(g)
        top_n = max(1, int(n * top_pct))
        top_stocks = g.nlargest(top_n, "pred")
        # 每只股票相对其所在行业中位数的超额
        ind_medians = g.groupby("l1_name")[label_col].median()
        top_stocks["ind_median"] = top_stocks["l1_name"].map(ind_medians)
        top_stocks["ind_excess"] = top_stocks[label_col] - top_stocks["ind_median"]
        market_median = g[label_col].median()
        return pd.Series({
            "top_ind_excess_mean": top_stocks["ind_excess"].mean(),
            "top_market_excess_mean": top_stocks[label_col].mean() - market_median,
            "top_ind_excess_positive_rate": (top_stocks["ind_excess"] > 0).mean(),
        })

    return merged.groupby(level="datetime").apply(_daily)


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="grid0188 模型深度验证")
    parser.add_argument("--pred-path", type=str, required=True, help="预测 parquet 路径")
    parser.add_argument("--label-col", type=str, default="label_rank_10d", help="标签列名")
    parser.add_argument("--industry-dir", type=str,
                        default="data/cross_sectional_processd_data/industry_factors",
                        help="行业因子数据目录")
    parser.add_argument("--index-dir", type=str, default="data/raw_data/index_daily",
                        help="指数日线数据目录")
    parser.add_argument("--index-code", type=str, default="000985.SH",
                        help="市场环境划分用指数（默认中证全指）")
    parser.add_argument("--save-dir", type=str, default="output/model_diagnostic_grid0188",
                        help="结果保存目录")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("grid0188 模型深度验证")
    print("=" * 60)

    # 加载数据
    print("\n[1/5] 加载数据...")
    pred_df = load_predictions(args.pred_path)
    dates = pred_df.index.get_level_values(0)
    print(f"  预测数据: {pred_df.shape}, 日期范围: {dates.min().date()} ~ {dates.max().date()}")

    start_date = pred_df.index.get_level_values(0).min()
    end_date = pred_df.index.get_level_values(0).max()

    industry_df = load_industry_data(start_date, end_date, args.industry_dir)
    print(f"  行业数据: {industry_df.shape}")

    index_df = load_index_daily(args.index_code, args.index_dir)
    print(f"  指数数据({args.index_code}): {index_df.shape}")

    # ── 验证一：行业中性化 IC ──
    print("\n[2/5] 验证一：行业中性化 IC...")

    raw_ic = raw_ic_daily(pred_df, args.label_col)
    print(f"  原始 IC: 均值={raw_ic.mean():.4f}, IR={raw_ic.mean()/raw_ic.std()*np.sqrt(252):.2f}")

    ind_neutral_ic = industry_neutral_ic_daily(pred_df, industry_df, args.label_col)
    print(f"  行业中性 IC: 均值={ind_neutral_ic.mean():.4f}, IR={ind_neutral_ic.mean()/ind_neutral_ic.std()*np.sqrt(252):.2f}")
    print(f"  IC 衰减比例: {(1 - ind_neutral_ic.mean()/raw_ic.mean())*100:.1f}%")

    within_ind = within_industry_ic_daily(pred_df, industry_df, args.label_col)
    print(f"  行业内 IC: 均值={within_ind['within_ind_ic'].mean():.4f}, "
          f"平均行业数={within_ind['n_industries'].mean():.0f}")

    # 保存
    ic_summary = pd.DataFrame({
        "raw_ic": raw_ic,
        "ind_neutral_ic": ind_neutral_ic,
        "within_ind_ic": within_ind["within_ind_ic"],
        "n_industries": within_ind["n_industries"],
    })
    ic_summary.to_csv(save_dir / "industry_neutral_ic_daily.csv")
    print(f"  已保存: industry_neutral_ic_daily.csv")

    # 行业暴露
    print("  计算行业暴露...")
    exposure = industry_exposure_daily(pred_df, industry_df)
    exposure.to_csv(save_dir / "top10_industry_exposure_daily.csv")
    # 时间序列平均
    avg_overweight = {}
    for i in range(1, 4):
        col = f"top_overweight_{i}_ind"
        if col in exposure.columns:
            counts = exposure[col].value_counts().head(5)
            avg_overweight[f"top{i}"] = counts.to_dict()
    with open(save_dir / "industry_exposure_summary.json", "w") as f:
        json.dump(avg_overweight, f, ensure_ascii=False, indent=2)
    print(f"  已保存: industry_exposure_summary.json")

    # ── 验证二：市场环境分层 ──
    print("\n[3/5] 验证二：市场环境分层...")

    if not index_df.empty:
        env_df = classify_market_env(index_df)
        env_df.to_csv(save_dir / "market_regime_daily.csv")

        ic_by_regime = ic_by_env(raw_ic, env_df)
        print("  IC 按趋势划分:")
        print(ic_by_regime.to_string(float_format="%.4f"))

        ls_by_regime = long_short_return_by_env(pred_df, args.label_col, env_df)
        print("\n  多空收益按趋势划分:")
        print(ls_by_regime.to_string(float_format="%.4f"))

        vol_env = env_df[["ret_20d", "vol_20d"]].copy()
        vol_env["trend_regime"] = env_df["vol_regime"]
        ic_by_vol = ic_by_env(raw_ic, vol_env)
        print("\n  IC 按波动率划分:")
        print(ic_by_vol.to_string(float_format="%.4f"))

        ic_by_regime.to_csv(save_dir / "ic_by_trend_regime.csv")
        ls_by_regime.to_csv(save_dir / "longshort_by_trend_regime.csv")
        ic_by_vol.to_csv(save_dir / "ic_by_vol_regime.csv")
        print(f"  已保存市场环境分析结果")
    else:
        print("  警告: 无指数数据，跳过市场环境分层")

    # ── 验证三：多头诊断 ──
    print("\n[4/5] 验证三：多头诊断...")

    ls_ic = long_side_ic_daily(pred_df, args.label_col)
    print(f"  多头侧 IC: 均值={ls_ic['long_side_ic'].mean():.4f}, "
          f"IR={ls_ic['long_side_ic'].mean()/ls_ic['long_side_ic'].std()*np.sqrt(252):.2f}")
    print(f"  空头侧 IC: 均值={ls_ic['short_side_ic'].mean():.4f}, "
          f"IR={ls_ic['short_side_ic'].mean()/ls_ic['short_side_ic'].std()*np.sqrt(252):.2f}")

    top_k = top_k_analysis_daily(pred_df, args.label_col, ks=[50, 100, 200])
    print(f"  Top 50 平均收益: {top_k['top_50_ret'].mean():.4f}")
    print(f"  Top 100 平均收益: {top_k['top_100_ret'].mean():.4f}")
    print(f"  Top 200 平均收益: {top_k['top_200_ret'].mean():.4f}")
    print(f"  Top 10% 平均收益: {top_k['top_decile_ret'].mean():.4f}")
    print(f"  Bottom 10% 平均收益: {top_k['bottom_decile_ret'].mean():.4f}")
    print(f"  多空价差(10%): {top_k['decile_spread'].mean():.4f}")
    print(f"  Top 50 命中率(跑赢中位数): {top_k['top_50_hit_rate'].mean():.2%}")
    print(f"  Top 100 命中率(跑赢中位数): {top_k['top_100_hit_rate'].mean():.2%}")
    print(f"  全市场中位数: {top_k['market_median'].mean():.4f}")
    print(f"  全市场均值: {top_k['market_mean'].mean():.4f}")

    upside = upside_capture_daily(pred_df, args.label_col)
    print(f"  上行捕获率(真实Top10在预测Top10中): {upside['upside_capture_top10_in10'].mean():.2%}")
    print(f"  上行捕获率(真实Top10在预测Top20中): {upside['upside_capture_top10_in20'].mean():.2%}")
    print(f"  下行捕获率(真实Bot10在预测Bot10中): {upside['downside_capture_bot10_in10'].mean():.2%}")

    top_excess = top_group_industry_excess_daily(pred_df, industry_df, args.label_col)
    print(f"  Top10% 相对行业中位数超额: {top_excess['top_ind_excess_mean'].mean():.4f}")
    print(f"  Top10% 相对全市场中位数超额: {top_excess['top_market_excess_mean'].mean():.4f}")
    print(f"  Top10% 行业内正超额比例: {top_excess['top_ind_excess_positive_rate'].mean():.2%}")

    # 保存
    ls_ic.to_csv(save_dir / "long_short_side_ic_daily.csv")
    top_k.to_csv(save_dir / "top_k_analysis_daily.csv")
    upside.to_csv(save_dir / "upside_capture_daily.csv")
    top_excess.to_csv(save_dir / "top_group_excess_daily.csv")
    print(f"  已保存多头诊断结果")

    # ── 汇总报告 ──
    print("\n[5/5] 生成汇总报告...")
    summary = {
        "pred_path": args.pred_path,
        "label_col": args.label_col,
        "date_range": f"{start_date.date()} ~ {end_date.date()}",
        "n_days": len(raw_ic),
        "industry_neutral_ic": {
            "raw_ic_mean": float(raw_ic.mean()),
            "raw_ic_ir": float(raw_ic.mean() / raw_ic.std() * np.sqrt(252)),
            "ind_neutral_ic_mean": float(ind_neutral_ic.mean()),
            "ind_neutral_ic_ir": float(ind_neutral_ic.mean() / ind_neutral_ic.std() * np.sqrt(252)),
            "ic_decay_pct": float((1 - ind_neutral_ic.mean() / raw_ic.mean()) * 100),
            "within_ind_ic_mean": float(within_ind["within_ind_ic"].mean()),
        },
        "long_short_side_ic": {
            "long_side_ic_mean": float(ls_ic["long_side_ic"].mean()),
            "long_side_ic_ir": float(ls_ic["long_side_ic"].mean() / ls_ic["long_side_ic"].std() * np.sqrt(252)),
            "short_side_ic_mean": float(ls_ic["short_side_ic"].mean()),
            "short_side_ic_ir": float(ls_ic["short_side_ic"].mean() / ls_ic["short_side_ic"].std() * np.sqrt(252)),
        },
        "top_k_analysis": {
            "top_50_ret_mean": float(top_k["top_50_ret"].mean()),
            "top_100_ret_mean": float(top_k["top_100_ret"].mean()),
            "top_200_ret_mean": float(top_k["top_200_ret"].mean()),
            "top_decile_ret_mean": float(top_k["top_decile_ret"].mean()),
            "bottom_decile_ret_mean": float(top_k["bottom_decile_ret"].mean()),
            "decile_spread_mean": float(top_k["decile_spread"].mean()),
            "top_50_hit_rate": float(top_k["top_50_hit_rate"].mean()),
            "top_100_hit_rate": float(top_k["top_100_hit_rate"].mean()),
            "market_median_mean": float(top_k["market_median"].mean()),
            "market_mean_mean": float(top_k["market_mean"].mean()),
        },
        "upside_capture": {
            "top10_in10": float(upside["upside_capture_top10_in10"].mean()),
            "top10_in20": float(upside["upside_capture_top10_in20"].mean()),
            "bot10_in10": float(upside["downside_capture_bot10_in10"].mean()),
        },
        "top_group_excess": {
            "industry_excess_mean": float(top_excess["top_ind_excess_mean"].mean()),
            "market_excess_mean": float(top_excess["top_market_excess_mean"].mean()),
            "industry_positive_rate": float(top_excess["top_ind_excess_positive_rate"].mean()),
        },
    }

    if not index_df.empty:
        summary["market_regime_ic"] = ic_by_regime.to_dict(orient="index")
        summary["market_regime_longshort"] = ls_by_regime.to_dict(orient="index")

    with open(save_dir / "diagnostic_summary.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n所有结果已保存到: {save_dir}/")
    print("=" * 60)
    print("验证完成！")


if __name__ == "__main__":
    main()
