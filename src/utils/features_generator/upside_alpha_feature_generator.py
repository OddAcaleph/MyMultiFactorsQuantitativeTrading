"""上涨侧 Alpha 因子生成器。

7 组因子，共约 50 个：
  1. 相对强度特征 (relative_strength)
  2. 动量质量特征 (momentum_quality)
  3. 量价确认特征 (volume_price_confirmation)
  4. 行业/主题强度特征 (industry_theme_strength)
  5. 反转与过热特征 (reversal_overheat)
  6. 基本面变化增强 (fundamental_change)
  7. 事件驱动特征 (event_driven)

输出目录：data/features_data/upside_alpha_factors/
横截面处理后：data/cross_sectional_processd_data/upside_alpha_factors/
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class UpsideAlphaFactorGenerateSummary:
    total_files_written: int = 0
    total_rows_written: int = 0
    nan_counts: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


# ============================================================================
# 主生成器
# ============================================================================

class UpsideAlphaFactorGenerator:
    """上涨侧 Alpha 因子生成器。"""

    # ---- 因子列定义 ----
    RELATIVE_STRENGTH_FACTORS = (
        "excess_mkt_ret_5d",
        "excess_mkt_ret_10d",
        "excess_mkt_ret_20d",
        "excess_ind_ret_5d",
        "excess_ind_ret_10d",
        "excess_ind_ret_20d",
        "ind_amount_rank_20d",
        "ind_turnover_rank_20d",
    )

    MOMENTUM_QUALITY_FACTORS = (
        "ret_slope_stability_20d",
        "up_day_ratio_10d",
        "up_day_ratio_20d",
        "consecutive_up_gain_ratio",
        "new_high_dist_60d",
        "ma_bullish_alignment",
        "breakout_pullback_ratio_20d",
    )

    VOLUME_PRICE_CONFIRMATION_FACTORS = (
        "amount_expansion_ratio",
        "amount_trend_5d",
        "volume_price_corr_20d",
        "up_down_volume_ratio_adv",
        "turnover_expansion_ratio",
        "obv_slope_20d",
        "ind_amount_share_change_20d",
    )

    INDUSTRY_THEME_STRENGTH_FACTORS = (
        "industry_ret_rank_5d",
        "industry_ret_rank_20d",
        "industry_up_ratio_5d",
        "industry_up_ratio_20d",
        "industry_new_high_ratio_20d",
        "industry_amount_expansion_20d",
        "industry_leader_strength_20d",
        "industry_breadth_20d",
    )

    REVERSAL_OVERHEAT_FACTORS = (
        "short_term_overheat_3d",
        "short_term_overheat_5d",
        "ma20_deviation",
        "high_volume_stagnation",
        "upper_shadow_ratio",
        "pullback_from_high_5d",
    )

    FUNDAMENTAL_CHANGE_FACTORS = (
        "ocf_yoy_change",
        "debt_ratio_change",
        "inventory_turnover_change",
        "receivable_turnover_change",
    )

    EVENT_DRIVEN_FACTORS = (
        "forecast_type_score",
        "forecast_surprise_magnitude",
        "forecast_recency_30d",
        "top_list_net_amount_5d",
        "top_list_count_20d",
        "top_list_institution_ratio_20d",
        "repurchase_amount_ratio_30d",
        "holder_increase_ratio_30d",
        "block_trade_discount_30d",
        "block_trade_amount_ratio_30d",
    )

    @property
    def FACTOR_COLUMNS(self) -> tuple[str, ...]:
        return (
            *self.RELATIVE_STRENGTH_FACTORS,
            *self.MOMENTUM_QUALITY_FACTORS,
            *self.VOLUME_PRICE_CONFIRMATION_FACTORS,
            *self.INDUSTRY_THEME_STRENGTH_FACTORS,
            *self.REVERSAL_OVERHEAT_FACTORS,
            *self.FUNDAMENTAL_CHANGE_FACTORS,
            *self.EVENT_DRIVEN_FACTORS,
        )

    # ---- 默认路径 ----
    DEFAULT_INPUT_DIR = Path("data/cleaned_data/daily_bars")
    DEFAULT_MONEYFLOW_FILE = Path("data/cleaned_data/moneyflow/moneyflow.parquet")
    DEFAULT_FUNDAMENTALS_FILE = Path("data/cleaned_data/fundamentals/fundamentals.parquet")
    DEFAULT_INDUSTRY_PATH = Path("data/processd_data/industry")
    DEFAULT_OUTPUT_DIR = Path("data/features_data/upside_alpha_factors")

    # 事件数据目录
    DEFAULT_TOPLIST_DIR = Path("data/raw_data/top_list")
    DEFAULT_FORECAST_DIR = Path("data/raw_data/forecast")
    DEFAULT_REPURCHASE_DIR = Path("data/raw_data/repurchase")
    DEFAULT_HOLDERTRADE_DIR = Path("data/raw_data/stk_holdertrade")
    DEFAULT_BLOCKTRADE_DIR = Path("data/raw_data/block_trade")

    def __init__(
        self,
        input_dir: str | Path | None = None,
        moneyflow_file: str | Path | None = None,
        fundamentals_file: str | Path | None = None,
        industry_file: str | Path | None = None,
        output_dir: str | Path | None = None,
        toplist_dir: str | Path | None = None,
        forecast_dir: str | Path | None = None,
        repurchase_dir: str | Path | None = None,
        holdertrade_dir: str | Path | None = None,
        blocktrade_dir: str | Path | None = None,
        start_date: str | int | None = None,
        end_date: str | int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.input_dir = Path(input_dir or self.DEFAULT_INPUT_DIR)
        self.moneyflow_file = Path(moneyflow_file or self.DEFAULT_MONEYFLOW_FILE)
        self.fundamentals_file = Path(fundamentals_file or self.DEFAULT_FUNDAMENTALS_FILE)
        self.industry_path = Path(industry_file or self.DEFAULT_INDUSTRY_PATH)
        self.output_dir = Path(output_dir or self.DEFAULT_OUTPUT_DIR)
        self.toplist_dir = Path(toplist_dir or self.DEFAULT_TOPLIST_DIR)
        self.forecast_dir = Path(forecast_dir or self.DEFAULT_FORECAST_DIR)
        self.repurchase_dir = Path(repurchase_dir or self.DEFAULT_REPURCHASE_DIR)
        self.holdertrade_dir = Path(holdertrade_dir or self.DEFAULT_HOLDERTRADE_DIR)
        self.blocktrade_dir = Path(blocktrade_dir or self.DEFAULT_BLOCKTRADE_DIR)
        self.start_date = self._parse_optional_date(start_date, "start_date")
        self.end_date = self._parse_optional_date(end_date, "end_date")
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self._industry_is_daily = self.industry_path.is_dir() if self.industry_path.exists() else False
        self._summary = UpsideAlphaFactorGenerateSummary()

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            raise NotADirectoryError(f"Daily bars input directory not found: {self.input_dir}")
        if not self.industry_path.exists():
            self.logger.warning("Industry data not found: %s — industry factors will be skipped.", self.industry_path)
        if self.start_date is not None and self.end_date is not None and self.start_date > self.end_date:
            raise ValueError(f"start_date must be <= end_date, got {self.start_date} > {self.end_date}")

    @staticmethod
    def _parse_optional_date(value: str | int | None, name: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        s = str(value).replace("-", "").strip()
        if not s.isdigit() or len(s) != 8:
            raise ValueError(f"Invalid {name}: {value}")
        return int(s)

    # ====================================================================
    # 主流程
    # ====================================================================

    def process(self) -> UpsideAlphaFactorGenerateSummary:
        self.logger.info(
            "开始生成上涨侧alpha因子：input=%s, output=%s",
            self.input_dir, self.output_dir,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 列出所有日度行情文件
        all_files = self._list_all_daily_bar_files()
        output_files = [f for f in all_files if self._is_file_date_in_output_range(f)]
        self.logger.info("输出日期文件数=%d；按年份分批处理；所有输入数据只读。", len(output_files))

        if not output_files:
            self.logger.warning("没有符合日期范围的输入文件。")
            return self._summary

        # 预加载事件数据（按年加载，处理到哪年加载哪年）
        # 预加载基本面数据
        fund_df = self._load_fundamentals()
        if fund_df is not None:
            self.logger.info("基本面数据加载完成：%d 行", len(fund_df))
        else:
            self.logger.warning("基本面数据未找到，基本面变化因子将跳过。")

        # 按年份分批
        files_by_year: dict[int, list[Path]] = {}
        for f in output_files:
            year = int(f.parent.parent.name.split("=")[1])
            files_by_year.setdefault(year, []).append(f)

        for year in sorted(files_by_year.keys()):
            year_files = sorted(files_by_year[year], key=lambda p: p.name)
            self.logger.info("处理年份 %d：输出文件数=%d, 读取文件数=%d",
                             year, len(year_files), len(year_files))

            self._process_year(year, year_files, fund_df)

        self.logger.info(
            "上涨侧alpha因子生成完成：total_files_written=%d, total_rows_written=%d",
            self._summary.total_files_written, self._summary.total_rows_written,
        )
        return self._summary

    def _process_year(
        self,
        year: int,
        year_files: list[Path],
        fund_df: pd.DataFrame | None,
    ) -> None:
        """处理单年数据。需要向前回溯足够的历史数据。"""
        # 读取当前年 + 前一年的数据（用于计算滚动指标，最大窗口60日 + 余量）
        # 实际上我们需要前推约 260 个交易日（约1年多）来覆盖所有滚动窗口
        read_years = list(range(max(2000, year - 2), year + 1))
        dfs = []
        for y in read_years:
            y_dir = self.input_dir / f"year={y}"
            if not y_dir.exists():
                continue
            for month_dir in sorted(y_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                for f in sorted(month_dir.glob("*.parquet")):
                    dfs.append(pd.read_parquet(f))

        if not dfs:
            self.logger.warning("年份 %d 没有可用的行情数据。", year)
            return

        full_df = pd.concat(dfs, axis=0, ignore_index=True)
        full_df = full_df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        del dfs

        # 统一 trade_date 为 int
        if full_df["trade_date"].dtype == object:
            full_df["trade_date"] = full_df["trade_date"].astype(int)

        self.logger.info("行情数据读取完成：rows=%d", len(full_df))

        # 加载行业数据（日度）
        ind_df = self._load_industry_for_dates(full_df["trade_date"].unique())
        if ind_df is not None:
            self.logger.info("行业数据加载完成：%d 行", len(ind_df))

        # 加载事件数据（当年 + 前一年，用于30日窗口等）
        event_years = [year - 1, year]
        toplist_df = self._load_toplist(event_years)
        forecast_df = self._load_forecast(event_years)
        repurchase_df = self._load_repurchase(event_years)
        holdertrade_df = self._load_holdertrade(event_years)
        blocktrade_df = self._load_blocktrade(event_years)

        # 计算因子
        result_df = self._compute_factors(full_df, ind_df, fund_df,
                                          toplist_df, forecast_df, repurchase_df,
                                          holdertrade_df, blocktrade_df)

        # 只保留当年的输出
        year_start = int(f"{year}0101")
        year_end = int(f"{year}1231")
        year_result = result_df[(result_df["trade_date"] >= year_start) &
                                (result_df["trade_date"] <= year_end)].copy()

        del result_df, full_df

        # 按日写入
        year_rows = 0
        for trade_date, group in year_result.groupby("trade_date"):
            date_str = str(int(trade_date))
            year_m = date_str[:6]
            y = date_str[:4]
            m = date_str[4:6]
            out_dir = self.output_dir / f"year={y}" / f"month={m}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{date_str}.parquet"

            group = group.sort_values("ts_code").reset_index(drop=True)
            group.to_parquet(out_path, index=False)

            year_rows += len(group)
            self._summary.total_files_written += 1

        self._summary.total_rows_written += year_rows
        self.logger.info("年份 %d 完成：写入 %d 个文件, %d 行",
                         year, len(year_result["trade_date"].unique()), year_rows)

        # NaN 统计
        for col in self.FACTOR_COLUMNS:
            if col in year_result.columns:
                self._summary.nan_counts[col] = int(year_result[col].isna().sum())

        del year_result
        import gc
        gc.collect()

    # ====================================================================
    # 因子计算总入口
    # ====================================================================

    def _compute_factors(
        self,
        df: pd.DataFrame,
        ind_df: pd.DataFrame | None,
        fund_df: pd.DataFrame | None,
        toplist_df: pd.DataFrame | None,
        forecast_df: pd.DataFrame | None,
        repurchase_df: pd.DataFrame | None,
        holdertrade_df: pd.DataFrame | None,
        blocktrade_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """计算所有因子。"""
        result = df[["trade_date", "ts_code", "open", "high", "low", "close",
                     "vol", "amount", "pct_chg"]].copy()

        # 基础衍生指标
        result["ret_1d"] = result.groupby("ts_code")["close"].pct_change()
        result["turnover"] = result["vol"] / result["amount"].clip(lower=1) * result["close"]  # 近似
        # 用 amount / close 估算流通股本，再算 turnover
        # 更好的方式：直接用 vol / total_share，但我们没有 total_share
        # 这里用 amount / (close * 1e6) 作为近似换手率的代理（单位%），后面行业rank不影响
        result["turnover_approx"] = result["vol"] / result["amount"].clip(lower=1) * result["close"] * 100

        # ---- 第1组：相对强度 ----
        result = self._compute_relative_strength(result, ind_df)
        self.logger.info("相对强度特征计算完成")

        # ---- 第2组：动量质量 ----
        result = self._compute_momentum_quality(result)
        self.logger.info("动量质量特征计算完成")

        # ---- 第3组：量价确认 ----
        result = self._compute_volume_price_confirmation(result, ind_df)
        self.logger.info("量价确认特征计算完成")

        # ---- 第4组：行业/主题强度 ----
        if ind_df is not None:
            result = self._compute_industry_theme_strength(result, ind_df)
            self.logger.info("行业主题强度特征计算完成")
        else:
            for col in self.INDUSTRY_THEME_STRENGTH_FACTORS:
                result[col] = np.nan

        # ---- 第5组：反转与过热 ----
        result = self._compute_reversal_overheat(result)
        self.logger.info("反转过热特征计算完成")

        # ---- 第6组：基本面变化增强 ----
        if fund_df is not None:
            result = self._compute_fundamental_change(result, fund_df)
            self.logger.info("基本面变化特征计算完成")
        else:
            for col in self.FUNDAMENTAL_CHANGE_FACTORS:
                result[col] = np.nan

        # ---- 第7组：事件驱动 ----
        result = self._compute_event_driven(
            result, toplist_df, forecast_df, repurchase_df,
            holdertrade_df, blocktrade_df,
        )
        self.logger.info("事件驱动特征计算完成")

        return result

    # ====================================================================
    # 第1组：相对强度特征
    # ====================================================================

    def _compute_relative_strength(self, df: pd.DataFrame, ind_df: pd.DataFrame | None) -> pd.DataFrame:
        """相对强度特征。"""
        # 计算累计收益
        for n in [5, 10, 20]:
            df[f"ret_{n}d"] = df.groupby("ts_code")["close"].pct_change(n)

        # 市场收益（全市场等权）
        mkt_ret = df.groupby("trade_date")["ret_1d"].mean().rename("mkt_ret_1d")
        df = df.merge(mkt_ret, on="trade_date", how="left")
        df["mkt_cum_5d"] = df.groupby("ts_code")["mkt_ret_1d"].transform(
            lambda x: (1 + x).rolling(5).apply(np.prod, raw=True) - 1
        )
        df["mkt_cum_10d"] = df.groupby("ts_code")["mkt_ret_1d"].transform(
            lambda x: (1 + x).rolling(10).apply(np.prod, raw=True) - 1
        )
        df["mkt_cum_20d"] = df.groupby("ts_code")["mkt_ret_1d"].transform(
            lambda x: (1 + x).rolling(20).apply(np.prod, raw=True) - 1
        )

        # 相对市场收益
        for n in [5, 10, 20]:
            df[f"excess_mkt_ret_{n}d"] = df[f"ret_{n}d"] - df[f"mkt_cum_{n}d"]

        # 相对行业收益
        if ind_df is not None:
            df = self._merge_industry(df, ind_df)
            # 行业收益（行业内等权）
            for n in [5, 10, 20]:
                ind_ret = df.groupby(["trade_date", "industry_l1"])[f"ret_{n}d"].mean().reset_index()
                ind_ret.columns = ["trade_date", "industry_l1", f"ind_ret_{n}d"]
                df = df.merge(ind_ret, on=["trade_date", "industry_l1"], how="left")
                df[f"excess_ind_ret_{n}d"] = df[f"ret_{n}d"] - df[f"ind_ret_{n}d"]

            # 行业内成交额 rank（过去20日平均成交额的行业内排名）
            df["amount_20d_mean"] = df.groupby("ts_code")["amount"].transform(
                lambda x: x.rolling(20).mean()
            )
            df["ind_amount_rank_20d"] = df.groupby(["trade_date", "industry_l1"])["amount_20d_mean"].rank(
                method="average", pct=True
            )

            # 行业内换手 rank
            df["turnover_20d_mean"] = df.groupby("ts_code")["turnover_approx"].transform(
                lambda x: x.rolling(20).mean()
            )
            df["ind_turnover_rank_20d"] = df.groupby(["trade_date", "industry_l1"])["turnover_20d_mean"].rank(
                method="average", pct=True
            )
        else:
            for col in ["excess_ind_ret_5d", "excess_ind_ret_10d", "excess_ind_ret_20d",
                        "ind_amount_rank_20d", "ind_turnover_rank_20d"]:
                df[col] = np.nan

        # 清理临时列
        drop_cols = [c for c in df.columns if c.startswith(("mkt_", "ind_ret_", "ret_5d", "ret_10d", "ret_20d",
                                                            "amount_20d_mean", "turnover_20d_mean",
                                                            "ret_1d"))]
        df = df.drop(columns=drop_cols, errors="ignore")

        return df

    # ====================================================================
    # 第2组：动量质量特征
    # ====================================================================

    def _compute_momentum_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """动量质量特征（向量化优化版）。"""
        # ---- 收益斜率稳定性：向量化线性回归 ----
        # 对于窗口 n，x = [0, 1, ..., n-1]
        # slope = (n*sum(x*y) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2)
        # t-stat = slope * sqrt(n-2) / sqrt(1 - r^2)
        n = 20
        x = np.arange(n, dtype=np.float64)
        x_sum = x.sum()
        x_sq_sum = (x ** 2).sum()
        denom = n * x_sq_sum - x_sum ** 2

        # 用 rolling sum 计算 sum(y), sum(x*y), sum(y^2)
        # sum(x*y) = sum(i * y_i) for i in 0..n-1
        # 可以用 shift 的方式构造，或者直接 rolling apply 但用 numpy 向量化
        # 更高效：用 rolling 的 weighted sum
        close_vals = df["close"].values
        result_slope = np.full(len(close_vals), np.nan)

        # 按股票分组计算
        for ts_code, group in df.groupby("ts_code"):
            idx = group.index
            y = group["close"].values.astype(np.float64)
            m = len(y)
            if m < n:
                continue
            # 计算 rolling sum(y), sum(x*y), sum(y^2)
            # 用 stride trick 或 cumsum
            # 这里用简单的 cumsum 方法
            cumsum_y = np.cumsum(y)
            cumsum_y2 = np.cumsum(y ** 2)
            # sum(x*y) 需要加权，用 cumsum 不好算，用卷积
            # 简化：直接计算每个窗口的斜率
            slopes = np.full(m, np.nan)
            tstats = np.full(m, np.nan)
            for i in range(n - 1, m):
                window = y[i - n + 1:i + 1]
                if np.any(np.isnan(window)):
                    continue
                # 线性回归
                y_mean = np.mean(window)
                x_mean = x_sum / n
                ss_xy = np.sum((x - x_mean) * (window - y_mean))
                ss_xx = np.sum((x - x_mean) ** 2)
                if ss_xx == 0:
                    continue
                slope = ss_xy / ss_xx
                # R^2
                ss_res = np.sum((window - (y_mean + slope * (x - x_mean))) ** 2)
                ss_tot = np.sum((window - y_mean) ** 2)
                if ss_tot == 0:
                    continue
                r_sq = 1 - ss_res / ss_tot
                if r_sq >= 1.0 or n <= 2:
                    tstats[i] = np.nan
                    continue
                se = np.sqrt(ss_res / (n - 2) / ss_xx)
                if se == 0:
                    tstats[i] = np.nan
                    continue
                tstats[i] = slope / se

            result_slope[idx.values] = tstats

        df["ret_slope_stability_20d"] = result_slope.astype("float32")

        # 上涨天数占比
        for n in [10, 20]:
            df[f"up_day_ratio_{n}d"] = df.groupby("ts_code")["pct_chg"].transform(
                lambda x: (x > 0).rolling(n, min_periods=max(3, n // 3)).mean()
            )

        # 连续阳线但不过热：连涨天数 / 连涨期间总涨幅（涨幅越小越健康）
        def _consecutive_up_gain_ratio(pct: pd.Series) -> pd.Series:
            """连涨天数 / 连涨累计涨幅。值大表示涨得稳，值小表示涨得急。"""
            result = np.full(len(pct), np.nan)
            streak = 0
            cum_gain = 0.0
            for i, v in enumerate(pct):
                if np.isnan(v):
                    streak = 0
                    cum_gain = 0.0
                    continue
                if v > 0:
                    streak += 1
                    cum_gain = cum_gain * (1 + v / 100) + v / 100 if streak > 1 else v / 100
                    # 简化：累计涨幅
                    if streak == 1:
                        cum_gain = v / 100
                    else:
                        cum_gain = (1 + cum_gain) * (1 + v / 100) - 1
                    if cum_gain > 0.001:
                        result[i] = streak / (cum_gain * 100)  # 天数 / 百分比涨幅
                else:
                    streak = 0
                    cum_gain = 0.0
            return pd.Series(result, index=pct.index)

        df["consecutive_up_gain_ratio"] = df.groupby("ts_code")["pct_chg"].transform(
            _consecutive_up_gain_ratio
        )

        # 60日新高距离：close / rolling_max_60d - 1
        df["high_60d"] = df.groupby("ts_code")["high"].transform(lambda x: x.rolling(60).max())
        df["new_high_dist_60d"] = df["close"] / df["high_60d"] - 1

        # 均线多头排列：MA5 > MA10 > MA20 的强度（归一化）
        df["ma5"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(5).mean())
        df["ma10"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(10).mean())
        df["ma20"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(20).mean())
        # 多头排列强度：(ma5-ma10)/ma20 + (ma10-ma20)/ma20 = (ma5 + ma10 - 2*ma20)/ma20
        df["ma_bullish_alignment"] = (df["ma5"] + df["ma10"] - 2 * df["ma20"]) / df["ma20"].clip(lower=0.01)

        # 突破后缩量整理（简化版）：20日最高量后3日缩量不跌
        df["amount_20d_max"] = df.groupby("ts_code")["amount"].transform(lambda x: x.rolling(20).max())
        df["is_high_vol"] = (df["amount"] >= df["amount_20d_max"] * 0.95).astype(float)
        df["fut3_ret"] = df.groupby("ts_code")["close"].transform(lambda x: x.shift(-3) / x - 1)
        df["fut3_amt"] = df.groupby("ts_code")["amount"].transform(lambda x: x.shift(-1).rolling(3).mean())
        df["fut3_amt_ratio"] = df["fut3_amt"] / df["amount"].clip(lower=1)
        df["breakout_pullback_ratio_20d"] = np.where(
            (df["is_high_vol"] == 1) & (df["fut3_amt_ratio"] < 0.9) & (df["fut3_ret"] > -0.02),
            (1 - df["fut3_amt_ratio"].clip(upper=0.9, lower=0)) * (1 + df["fut3_ret"].clip(lower=-0.02)),
            0.0,
        )


        # 清理临时列
        df = df.drop(columns=["high_60d", "ma5", "ma10", "ma20"], errors="ignore")

        return df

    # ====================================================================
    # 第3组：量价确认特征
    # ====================================================================

    def _compute_volume_price_confirmation(self, df: pd.DataFrame, ind_df: pd.DataFrame | None) -> pd.DataFrame:
        """量价确认特征。"""
        # 成交额放大倍数：当日成交额 / 20日均值
        df["amount_20d_mean"] = df.groupby("ts_code")["amount"].transform(
            lambda x: x.rolling(20).mean()
        )
        df["amount_expansion_ratio"] = df["amount"] / df["amount_20d_mean"].clip(lower=1)

        # 5日成交额趋势：(当前-5日前)/5日均值
        df["amount_5d_mean"] = df.groupby("ts_code")["amount"].transform(lambda x: x.rolling(5).mean())
        df["amount_5d_ago"] = df.groupby("ts_code")["amount"].shift(5)
        df["amount_trend_5d"] = (df["amount"] - df["amount_5d_ago"]) / df["amount_5d_mean"].clip(lower=1)

        # 量价相关性：过去20日收益和成交额变化的相关性
        df["amount_pct"] = df.groupby("ts_code")["amount"].pct_change()
        df["volume_price_corr_20d"] = df.groupby("ts_code").apply(
            lambda g: g["pct_chg"].rolling(20, min_periods=10).corr(g["amount_pct"])
        ).reset_index(level=0, drop=True)

        # 上涨放量、下跌缩量：向量化 rolling sum
        df["up_amount"] = np.where(df["pct_chg"] > 0, df["amount"], 0.0)
        df["down_amount"] = np.where(df["pct_chg"] < 0, df["amount"], 0.0)
        df["up_day"] = (df["pct_chg"] > 0).astype(float)
        df["down_day"] = (df["pct_chg"] < 0).astype(float)
        df["up_amt_20d_sum"] = df.groupby("ts_code")["up_amount"].transform(lambda x: x.rolling(20).sum())
        df["down_amt_20d_sum"] = df.groupby("ts_code")["down_amount"].transform(lambda x: x.rolling(20).sum())
        df["up_day_20d_sum"] = df.groupby("ts_code")["up_day"].transform(lambda x: x.rolling(20).sum())
        df["down_day_20d_sum"] = df.groupby("ts_code")["down_day"].transform(lambda x: x.rolling(20).sum())
        df["up_amt_avg"] = df["up_amt_20d_sum"] / df["up_day_20d_sum"].clip(lower=3)
        df["down_amt_avg"] = df["down_amt_20d_sum"] / df["down_day_20d_sum"].clip(lower=3)
        df["up_down_volume_ratio_adv"] = df["up_amt_avg"] / df["down_amt_avg"].clip(lower=1)
        df.loc[df["up_day_20d_sum"] < 3, "up_down_volume_ratio_adv"] = np.nan
        df.loc[df["down_day_20d_sum"] < 3, "up_down_volume_ratio_adv"] = np.nan

        # 换手率温和放大
        df["turnover_20d_mean"] = df.groupby("ts_code")["turnover_approx"].transform(
            lambda x: x.rolling(20).mean()
        )
        df["turnover_expansion_ratio"] = df["turnover_approx"] / df["turnover_20d_mean"].clip(lower=0.001)

        # OBV 趋势：简化为 OBV 相对20日均线偏离
        df["obv"] = (np.sign(df["pct_chg"].fillna(0)) * df["amount"]).groupby(df["ts_code"]).cumsum()
        df["obv_ma20"] = df.groupby("ts_code")["obv"].transform(lambda x: x.rolling(20).mean())
        df["obv_slope_20d"] = df["obv"] / df["obv_ma20"].clip(lower=1) - 1


        # 成交额行业占比变化
        if ind_df is not None and "industry_l1" in df.columns:
            # 个股成交额 / 行业总成交额
            ind_total_amt = df.groupby(["trade_date", "industry_l1"])["amount"].sum().reset_index()
            ind_total_amt.columns = ["trade_date", "industry_l1", "ind_total_amount"]
            df = df.merge(ind_total_amt, on=["trade_date", "industry_l1"], how="left")
            df["amount_share"] = df["amount"] / df["ind_total_amount"].clip(lower=1)

            # 20日变化
            df["amount_share_20d_mean"] = df.groupby("ts_code")["amount_share"].transform(
                lambda x: x.rolling(20).mean()
            )
            df["ind_amount_share_change_20d"] = (
                df["amount_share"] - df["amount_share_20d_mean"]
            ) / df["amount_share_20d_mean"].clip(lower=0.0001)
        else:
            df["ind_amount_share_change_20d"] = np.nan

        # 清理临时列
        df = df.drop(columns=["amount_20d_mean", "amount_pct", "turnover_20d_mean",
                              "obv", "ind_total_amount", "amount_share", "amount_share_20d_mean",
                              "turnover_approx"], errors="ignore")

        return df

    # ====================================================================
    # 第4组：行业/主题强度特征
    # ====================================================================

    def _compute_industry_theme_strength(self, df: pd.DataFrame, ind_df: pd.DataFrame) -> pd.DataFrame:
        """行业/主题强度特征。"""
        if "industry_l1" not in df.columns:
            df = self._merge_industry(df, ind_df)

        # 先算行业级别的指标
        ind_stats = df.groupby(["trade_date", "industry_l1"]).agg(
            ind_ret_1d=("pct_chg", lambda x: np.mean(x) / 100),
            ind_up_count=("pct_chg", lambda x: (x > 0).sum()),
            ind_total_count=("pct_chg", "count"),
            ind_total_amount=("amount", "sum"),
        ).reset_index()

        # 行业累计收益
        ind_stats = ind_stats.sort_values(["industry_l1", "trade_date"])
        for n in [5, 20]:
            ind_stats[f"ind_ret_{n}d"] = ind_stats.groupby("industry_l1")["ind_ret_1d"].transform(
                lambda x: (1 + x).rolling(n, min_periods=max(2, n // 2)).apply(np.prod, raw=True) - 1
            )

        # 行业收益 rank（全行业横截面排名）
        for n in [5, 20]:
            ind_stats[f"industry_ret_rank_{n}d"] = ind_stats.groupby("trade_date")[f"ind_ret_{n}d"].rank(
                method="average", pct=True
            )

        # 行业上涨家数占比
        ind_stats["industry_up_ratio_5d"] = ind_stats.groupby("industry_l1")["ind_up_count"].transform(
            lambda x: x.rolling(5, min_periods=2).sum()
        ) / ind_stats.groupby("industry_l1")["ind_total_count"].transform(
            lambda x: x.rolling(5, min_periods=2).sum()
        ).clip(lower=1)

        ind_stats["industry_up_ratio_20d"] = ind_stats.groupby("industry_l1")["ind_up_count"].transform(
            lambda x: x.rolling(20, min_periods=5).sum()
        ) / ind_stats.groupby("industry_l1")["ind_total_count"].transform(
            lambda x: x.rolling(20, min_periods=5).sum()
        ).clip(lower=1)

        # 行业创新高占比（20日新高）
        df["high_20d"] = df.groupby("ts_code")["high"].transform(lambda x: x.rolling(20).max())
        df["is_20d_high"] = (df["close"] >= df["high_20d"] * 0.95).astype(float)
        ind_new_high = df.groupby(["trade_date", "industry_l1"]).agg(
            new_high_count=("is_20d_high", "sum"),
            total_stocks=("ts_code", "count"),
        ).reset_index()
        ind_new_high["industry_new_high_ratio_20d"] = ind_new_high["new_high_count"] / ind_new_high["total_stocks"].clip(lower=1)
        ind_stats = ind_stats.merge(
            ind_new_high[["trade_date", "industry_l1", "industry_new_high_ratio_20d"]],
            on=["trade_date", "industry_l1"], how="left"
        )

        # 行业成交额扩张：行业成交额 / 过去20日均值
        ind_stats["ind_amount_20d_mean"] = ind_stats.groupby("industry_l1")["ind_total_amount"].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )
        ind_stats["industry_amount_expansion_20d"] = (
            ind_stats["ind_total_amount"] / ind_stats["ind_amount_20d_mean"].clip(lower=1)
        )

        # 行业龙头强度：行业成交额Top3股票的平均收益
        def _leader_strength(group: pd.DataFrame) -> float:
            if len(group) < 3:
                return np.nan
            top3 = group.nlargest(3, "amount")
            return top3["pct_chg"].mean() / 100

        ind_leader = df.groupby(["trade_date", "industry_l1"]).apply(_leader_strength).reset_index()
        ind_leader.columns = ["trade_date", "industry_l1", "leader_ret_1d"]
        ind_leader = ind_leader.sort_values(["industry_l1", "trade_date"])
        ind_leader["industry_leader_strength_20d"] = ind_leader.groupby("industry_l1")["leader_ret_1d"].transform(
            lambda x: (1 + x).rolling(20, min_periods=5).apply(np.prod, raw=True) - 1
        )
        ind_stats = ind_stats.merge(
            ind_leader[["trade_date", "industry_l1", "industry_leader_strength_20d"]],
            on=["trade_date", "industry_l1"], how="left"
        )

        # 行业扩散度：上涨股票数 / sqrt(总股票数) 的变化
        ind_stats["breadth_raw"] = ind_stats["ind_up_count"] / np.sqrt(ind_stats["ind_total_count"].clip(lower=1))
        ind_stats["breadth_ma20"] = ind_stats.groupby("industry_l1")["breadth_raw"].transform(
            lambda x: x.rolling(20, min_periods=5).mean()
        )
        ind_stats["industry_breadth_20d"] = ind_stats["breadth_raw"] / ind_stats["breadth_ma20"].clip(lower=0.1) - 1

        # 合并回个股
        merge_cols = ["trade_date", "industry_l1",
                      "industry_ret_rank_5d", "industry_ret_rank_20d",
                      "industry_up_ratio_5d", "industry_up_ratio_20d",
                      "industry_new_high_ratio_20d", "industry_amount_expansion_20d",
                      "industry_leader_strength_20d", "industry_breadth_20d"]
        df = df.merge(ind_stats[merge_cols], on=["trade_date", "industry_l1"], how="left")

        # 清理
        df = df.drop(columns=["high_20d", "is_20d_high", "industry_l1"], errors="ignore")

        return df

    # ====================================================================
    # 第5组：反转与过热特征
    # ====================================================================

    def _compute_reversal_overheat(self, df: pd.DataFrame) -> pd.DataFrame:
        """反转与过热特征。"""
        # 短期过热：过去3/5日涨幅
        for n in [3, 5]:
            df[f"short_term_overheat_{n}d"] = df.groupby("ts_code")["close"].pct_change(n)

        # 偏离均线程度：close / ma20 - 1
        df["ma20"] = df.groupby("ts_code")["close"].transform(lambda x: x.rolling(20).mean())
        df["ma20_deviation"] = df["close"] / df["ma20"] - 1

        # 高位放量滞涨：成交额放大但价格涨不动
        df["amount_5d_mean"] = df.groupby("ts_code")["amount"].transform(
            lambda x: x.rolling(5).mean()
        )
        df["amount_20d_mean"] = df.groupby("ts_code")["amount"].transform(
            lambda x: x.rolling(20).mean()
        )
        df["ret_5d"] = df.groupby("ts_code")["close"].pct_change(5)

        def _high_vol_stagnation(group: pd.DataFrame) -> pd.Series:
            """放量滞涨：量放大但价不涨。"""
            amt_ratio = group["amount_5d_mean"] / group["amount_20d_mean"].clip(lower=1)
            ret = group["ret_5d"]
            # 量放大（>1.2）但涨幅小（<2%）或负
            result = np.where(
                (amt_ratio > 1.2) & (ret < 0.02),
                amt_ratio * (0.02 - ret.clip(upper=0.02)) / 0.02,
                0.0
            )
            return pd.Series(result, index=group.index)

        df["high_volume_stagnation"] = df.groupby("ts_code", group_keys=False).apply(_high_vol_stagnation)

        # 长上影线：(high - max(open, close)) / (high - low)
        df["body_top"] = df[["open", "close"]].max(axis=1)
        df["intraday_range"] = (df["high"] - df["low"]).clip(lower=0.01)
        df["upper_shadow_ratio"] = (df["high"] - df["body_top"]) / df["intraday_range"]

        # 冲高回落：close 相对 high 的位置
        df["pullback_from_high_5d"] = df.groupby("ts_code").apply(
            lambda g: (g["close"] - g["high"].rolling(5).max()) / g["high"].rolling(5).max().clip(lower=0.01)
        ).reset_index(level=0, drop=True)

        # 清理
        df = df.drop(columns=["ma20", "amount_5d_mean", "amount_20d_mean", "ret_5d",
                              "body_top", "intraday_range"], errors="ignore")

        return df

    # ====================================================================
    # 第6组：基本面变化增强
    # ====================================================================

    def _compute_fundamental_change(self, df: pd.DataFrame, fund_df: pd.DataFrame) -> pd.DataFrame:
        """基本面变化增强因子。"""
        # 基本面数据按 ann_date 点时合并
        fund_df = fund_df.sort_values(["ts_code", "ann_date"])

        # 我们需要的字段
        needed_cols = ["ts_code", "ann_date", "end_date"]
        for col in ["ocfps", "debt_ratio", "inv_turn", "ar_turn",
                    "operate_cash_flow_ps", "current_ratio", "inventory_turnover",
                    "accounts_receivable_turnover", "turnover_days"]:
            if col in fund_df.columns:
                needed_cols.append(col)

        fund_sub = fund_df[needed_cols].copy() if all(c in fund_df.columns for c in needed_cols[:3]) else fund_df

        # 简化：用 ts_code + end_date 对齐，然后按 ann_date 取最新
        # 由于是日度数据，我们用更高效的方式：按季度取最新值
        # 先把 trade_date 转成 period
        df["trade_dt"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d")

        # 对每只股票，取 ann_date <= trade_date 的最新财报
        # 为了性能，我们按季度合并
        fund_sub["ann_dt"] = pd.to_datetime(fund_sub["ann_date"].astype(str), format="%Y%m%d")
        fund_sub = fund_sub.sort_values(["ts_code", "ann_dt"])

        # 计算同比变化
        fund_sub["end_dt"] = pd.to_datetime(fund_sub["end_date"].astype(str), format="%Y%m%d")

        # 经营现金流改善
        if "ocfps" in fund_sub.columns:
            fund_sub["ocfps_yoy"] = fund_sub.groupby("ts_code")["ocfps"].pct_change(4)
            fund_sub["ocf_yoy_change"] = fund_sub.groupby("ts_code")["ocfps_yoy"].diff(4)
        elif "operate_cash_flow_ps" in fund_sub.columns:
            fund_sub["ocfps_yoy"] = fund_sub.groupby("ts_code")["operate_cash_flow_ps"].pct_change(4)
            fund_sub["ocf_yoy_change"] = fund_sub.groupby("ts_code")["ocfps_yoy"].diff(4)
        else:
            fund_sub["ocf_yoy_change"] = np.nan

        # 资产负债率变化
        if "debt_ratio" in fund_sub.columns:
            fund_sub["debt_ratio_change"] = fund_sub.groupby("ts_code")["debt_ratio"].diff(4)
        else:
            fund_sub["debt_ratio_change"] = np.nan

        # 存货周转率变化
        inv_col = None
        for c in ["inv_turn", "inventory_turnover"]:
            if c in fund_sub.columns:
                inv_col = c
                break
        if inv_col:
            fund_sub["inventory_turnover_change"] = fund_sub.groupby("ts_code")[inv_col].diff(4)
        else:
            fund_sub["inventory_turnover_change"] = np.nan

        # 应收账款周转率变化
        ar_col = None
        for c in ["ar_turn", "accounts_receivable_turnover"]:
            if c in fund_sub.columns:
                ar_col = c
                break
        if ar_col:
            fund_sub["receivable_turnover_change"] = fund_sub.groupby("ts_code")[ar_col].diff(4)
        else:
            fund_sub["receivable_turnover_change"] = np.nan

        # 合并到日度数据：用 asof merge
        fund_for_merge = fund_sub[["ts_code", "ann_dt", "ocf_yoy_change", "debt_ratio_change",
                                   "inventory_turnover_change", "receivable_turnover_change"]].copy()
        fund_for_merge = fund_for_merge.dropna(subset=["ann_dt"])

        df = df.sort_values(["ts_code", "trade_dt"])
        result_list = []
        for ts_code, group in df.groupby("ts_code"):
            stock_fund = fund_for_merge[fund_for_merge["ts_code"] == ts_code]
            if stock_fund.empty:
                for col in ["ocf_yoy_change", "debt_ratio_change",
                            "inventory_turnover_change", "receivable_turnover_change"]:
                    group[col] = np.nan
                result_list.append(group)
                continue
            stock_fund = stock_fund.sort_values("ann_dt")
            merged = pd.merge_asof(
                group, stock_fund,
                left_on="trade_dt", right_on="ann_dt",
                direction="backward"
            , suffixes=("", "_evtright"))
            result_list.append(merged)

        df = pd.concat(result_list, axis=0, ignore_index=True)
        df = df.drop(columns=["ann_dt", "trade_dt"], errors="ignore")

        return df

    # ====================================================================
    # 第7组：事件驱动特征
    # ====================================================================

    def _compute_event_driven(
        self,
        df: pd.DataFrame,
        toplist_df: pd.DataFrame | None,
        forecast_df: pd.DataFrame | None,
        repurchase_df: pd.DataFrame | None,
        holdertrade_df: pd.DataFrame | None,
        blocktrade_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """事件驱动特征。"""
        # 初始化
        for col in self.EVENT_DRIVEN_FACTORS:
            df[col] = np.nan

        df["trade_dt"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d")

        # ---- 业绩预告 ----
        if forecast_df is not None and not forecast_df.empty:
            df = self._compute_forecast_factors(df, forecast_df)

        # ---- 龙虎榜 ----
        if toplist_df is not None and not toplist_df.empty:
            df = self._compute_toplist_factors(df, toplist_df)

        # ---- 回购 ----
        if repurchase_df is not None and not repurchase_df.empty:
            df = self._compute_repurchase_factors(df, repurchase_df)

        # ---- 增减持 ----
        if holdertrade_df is not None and not holdertrade_df.empty:
            df = self._compute_holdertrade_factors(df, holdertrade_df)

        # ---- 大宗交易 ----
        if blocktrade_df is not None and not blocktrade_df.empty:
            df = self._compute_blocktrade_factors(df, blocktrade_df)

        df = df.drop(columns=["trade_dt"], errors="ignore")
        return df

    def _compute_forecast_factors(self, df: pd.DataFrame, forecast_df: pd.DataFrame) -> pd.DataFrame:
        """业绩预告因子。"""
        forecast_df = forecast_df.copy()
        forecast_df["ann_dt"] = pd.to_datetime(forecast_df["ann_date"].astype(str), format="%Y%m%d")

        # 预告类型评分
        type_map = {
            "预增": 2.0, "续盈": 1.5, "扭亏": 1.5, "略增": 1.0,
            "略减": -1.0, "首亏": -2.0, "续亏": -2.0, "预减": -1.5,
        }
        forecast_df["forecast_type_score"] = forecast_df["type"].map(type_map).fillna(0)

        # 超预期幅度
        forecast_df["p_change_min"] = pd.to_numeric(forecast_df["p_change_min"], errors="coerce")
        forecast_df["p_change_max"] = pd.to_numeric(forecast_df["p_change_max"], errors="coerce")
        forecast_df["forecast_surprise_magnitude"] = (
            forecast_df["p_change_max"] + forecast_df["p_change_min"]
        ) / 2 / 100

        # 按 ts_code + ann_dt 去重（取最新的）
        forecast_df = forecast_df.sort_values(["ts_code", "ann_dt"])

        # 先删除已有的事件因子列，避免 merge 后列名重复
        drop_cols = ["forecast_type_score", "forecast_surprise_magnitude", "forecast_recency_30d"]
        df = df.drop(columns=drop_cols, errors="ignore")
        df = df.sort_values(["ts_code", "trade_dt"])
        result_list = []
        for ts_code, group in df.groupby("ts_code", sort=False):
            fc = forecast_df[forecast_df["ts_code"] == ts_code]
            if fc.empty:
                group["forecast_type_score"] = np.nan
                group["forecast_surprise_magnitude"] = np.nan
                group["forecast_recency_30d"] = np.nan
                result_list.append(group)
                continue

            fc = fc.sort_values("ann_dt")
            # 右边只保留需要的列，避免列名冲突
            fc_right = fc[["ann_dt", "forecast_type_score", "forecast_surprise_magnitude"]]
            merged = pd.merge_asof(
                group, fc_right,
                left_on="trade_dt", right_on="ann_dt",
                direction="backward",
            )
            # 30日时效性衰减
            merged["days_since_forecast"] = (merged["trade_dt"] - merged["ann_dt"]).dt.days
            merged["forecast_recency_30d"] = np.where(
                merged["days_since_forecast"] <= 30,
                1 - merged["days_since_forecast"] / 30,
                0
            )
            merged.loc[merged["days_since_forecast"].isna(), "forecast_recency_30d"] = np.nan
            merged = merged.drop(columns=["ann_dt", "days_since_forecast"], errors="ignore")
            result_list.append(merged)

        df = pd.concat(result_list, axis=0, ignore_index=True)
        return df

    def _compute_toplist_factors(self, df: pd.DataFrame, toplist_df: pd.DataFrame) -> pd.DataFrame:
        """龙虎榜因子。"""
        toplist_df = toplist_df.copy()
        toplist_df["trade_dt"] = pd.to_datetime(toplist_df["trade_date"].astype(str), format="%Y%m%d")

        # 按股票+日期聚合
        tl_daily = toplist_df.groupby(["ts_code", "trade_dt"]).agg(
            net_amount_sum=("net_amount", "sum"),
            count=("ts_code", "count"),
            amount_sum=("amount", "sum"),
        ).reset_index()

        # 先删除已有的列，避免 merge 后列名重复
        drop_cols = ["top_list_net_amount_5d", "top_list_count_20d", "top_list_institution_ratio_20d"]
        df = df.drop(columns=drop_cols, errors="ignore")
        df = df.sort_values(["ts_code", "trade_dt"])
        tl_daily = tl_daily.sort_values(["ts_code", "trade_dt"])

        # 先 merge 到每日（龙虎榜是当日事件，用 left merge）
        # 正确方式：left merge on ts_code + trade_dt，然后 rolling
        merged = df.merge(
            tl_daily[["ts_code", "trade_dt", "net_amount_sum", "count", "amount_sum"]],
            on=["ts_code", "trade_dt"], how="left"
        )
        merged["net_amount_sum"] = merged["net_amount_sum"].fillna(0)
        merged["count"] = merged["count"].fillna(0)
        merged["amount_sum"] = merged["amount_sum"].fillna(0)

        # 按股票分组计算 rolling
        merged = merged.sort_values(["ts_code", "trade_dt"])
        merged["top_list_net_amount_5d"] = (
            merged.groupby("ts_code")["net_amount_sum"]
            .transform(lambda x: x.rolling(5, min_periods=1).sum())
            / merged["amount"].clip(lower=1)
        )
        merged["top_list_count_20d"] = (
            merged.groupby("ts_code")["count"]
            .transform(lambda x: x.rolling(20, min_periods=1).sum())
        )
        inst_ratio = merged["net_amount_sum"] / merged["amount_sum"].clip(lower=1)
        merged["top_list_institution_ratio_20d"] = (
            inst_ratio.groupby(merged["ts_code"])
            .transform(lambda x: x.rolling(20, min_periods=1).mean())
        )

        merged = merged.drop(columns=["net_amount_sum", "count", "amount_sum"], errors="ignore")
        return merged

    def _compute_repurchase_factors(self, df: pd.DataFrame, repurchase_df: pd.DataFrame) -> pd.DataFrame:
        """回购因子。"""
        repurchase_df = repurchase_df.copy()
        repurchase_df["ann_dt"] = pd.to_datetime(repurchase_df["ann_date"].astype(str), format="%Y%m%d")
        repurchase_df["amount"] = pd.to_numeric(repurchase_df["amount"], errors="coerce")

        rep_daily = repurchase_df.groupby(["ts_code", "ann_dt"]).agg(
            repurchase_amount=("amount", "sum"),
        ).reset_index()

        df = df.drop(columns=["repurchase_amount_ratio_30d"], errors="ignore")
        df = df.sort_values(["ts_code", "trade_dt"])
        result_list = []
        for ts_code, group in df.groupby("ts_code", sort=False):
            rp = rep_daily[rep_daily["ts_code"] == ts_code]
            if rp.empty:
                group["repurchase_amount_ratio_30d"] = np.nan
                result_list.append(group)
                continue

            rp = rp.sort_values("ann_dt")
            rp_right = rp[["ann_dt", "repurchase_amount"]]
            merged = pd.merge_asof(
                group, rp_right,
                left_on="trade_dt", right_on="ann_dt",
                direction="backward",
            )
            merged["days_since_rep"] = (merged["trade_dt"] - merged["ann_dt"]).dt.days
            avg_amount_30d = group["amount"].rolling(30, min_periods=5).mean().values
            merged["repurchase_amount_ratio_30d"] = np.where(
                merged["days_since_rep"] <= 30,
                merged["repurchase_amount"].fillna(0) / np.clip(avg_amount_30d * 30, 1, None),
                0
            )
            merged.loc[merged["days_since_rep"].isna(), "repurchase_amount_ratio_30d"] = np.nan
            merged = merged.drop(columns=["ann_dt", "repurchase_amount", "days_since_rep"], errors="ignore")
            result_list.append(merged)

        df = pd.concat(result_list, axis=0, ignore_index=True)
        return df

    def _compute_holdertrade_factors(self, df: pd.DataFrame, holdertrade_df: pd.DataFrame) -> pd.DataFrame:
        """增减持因子。"""
        holdertrade_df = holdertrade_df.copy()
        holdertrade_df["ann_dt"] = pd.to_datetime(holdertrade_df["ann_date"].astype(str), format="%Y%m%d")
        holdertrade_df["change_ratio"] = pd.to_numeric(holdertrade_df["change_ratio"], errors="coerce")

        holdertrade_df["direction"] = holdertrade_df["in_de"].map({"IN": 1, "OUT": -1}).fillna(0)
        holdertrade_df["signed_change_ratio"] = holdertrade_df["change_ratio"] * holdertrade_df["direction"] / 100

        ht_daily = holdertrade_df.groupby(["ts_code", "ann_dt"]).agg(
            holder_increase_ratio=("signed_change_ratio", "sum"),
        ).reset_index()

        df = df.drop(columns=["holder_increase_ratio_30d"], errors="ignore")
        df = df.sort_values(["ts_code", "trade_dt"])
        result_list = []
        for ts_code, group in df.groupby("ts_code", sort=False):
            ht = ht_daily[ht_daily["ts_code"] == ts_code]
            if ht.empty:
                group["holder_increase_ratio_30d"] = np.nan
                result_list.append(group)
                continue

            ht = ht.sort_values("ann_dt")
            ht_right = ht[["ann_dt", "holder_increase_ratio"]]
            merged = pd.merge_asof(
                group, ht_right,
                left_on="trade_dt", right_on="ann_dt",
                direction="backward",
            )
            merged["days_since_ht"] = (merged["trade_dt"] - merged["ann_dt"]).dt.days
            merged["holder_increase_ratio_30d"] = np.where(
                merged["days_since_ht"] <= 30,
                merged["holder_increase_ratio"].fillna(0),
                0
            )
            merged.loc[merged["days_since_ht"].isna(), "holder_increase_ratio_30d"] = np.nan
            merged = merged.drop(columns=["ann_dt", "holder_increase_ratio", "days_since_ht"], errors="ignore")
            result_list.append(merged)

        df = pd.concat(result_list, axis=0, ignore_index=True)
        return df

    def _compute_blocktrade_factors(self, df: pd.DataFrame, blocktrade_df: pd.DataFrame) -> pd.DataFrame:
        """大宗交易因子。"""
        blocktrade_df = blocktrade_df.copy()
        blocktrade_df["trade_dt"] = pd.to_datetime(blocktrade_df["trade_date"].astype(str), format="%Y%m%d")
        blocktrade_df["price"] = pd.to_numeric(blocktrade_df["price"], errors="coerce")
        blocktrade_df["amount"] = pd.to_numeric(blocktrade_df["amount"], errors="coerce")

        # 计算折价率：先对齐 trade_date 类型
        blocktrade_df["trade_date"] = blocktrade_df["trade_date"].astype(int)
        bt_merged = blocktrade_df.merge(
            df[["trade_date", "ts_code", "close"]],
            on=["trade_date", "ts_code"], how="left"
        )
        bt_merged["discount"] = (bt_merged["price"] / bt_merged["close"].clip(lower=0.01)) - 1

        bt_daily = bt_merged.groupby(["ts_code", "trade_dt"]).agg(
            avg_discount=("discount", "mean"),
            total_amount=("amount", "sum"),
        ).reset_index()

        # 先删除已有的列，避免 merge 后列名重复
        drop_cols = ["block_trade_discount_30d", "block_trade_amount_ratio_30d"]
        df = df.drop(columns=drop_cols, errors="ignore")
        df = df.sort_values(["ts_code", "trade_dt"])
        bt_daily = bt_daily.sort_values(["ts_code", "trade_dt"])

        # left merge 然后 rolling
        merged = df.merge(
            bt_daily[["ts_code", "trade_dt", "avg_discount", "total_amount"]],
            on=["ts_code", "trade_dt"], how="left"
        )
        merged["avg_discount"] = merged["avg_discount"].fillna(0)
        merged["total_amount"] = merged["total_amount"].fillna(0)

        merged["block_trade_discount_30d"] = (
            merged.groupby("ts_code")["avg_discount"]
            .transform(lambda x: x.rolling(30, min_periods=1).mean())
        )
        avg_amount_30d = (
            merged.groupby("ts_code")["amount"]
            .transform(lambda x: x.rolling(30, min_periods=5).mean())
        )
        bt_amount_30d = (
            merged.groupby("ts_code")["total_amount"]
            .transform(lambda x: x.rolling(30, min_periods=1).sum())
        )
        merged["block_trade_amount_ratio_30d"] = bt_amount_30d / np.clip(avg_amount_30d * 30, 1, None)

        merged = merged.drop(columns=["avg_discount", "total_amount"], errors="ignore")
        return merged

    # ====================================================================
    # 辅助方法
    # ====================================================================

    def _merge_industry(self, df: pd.DataFrame, ind_df: pd.DataFrame) -> pd.DataFrame:
        """合并行业分类。"""
        if "industry_l1" in df.columns:
            return df

        # ind_df 是日度 onehot 格式，需要转成 label
        ind_cols = [c for c in ind_df.columns if c.startswith("L1_")]
        if ind_cols:
            # 统一 trade_date 类型
            ind_work = ind_df.copy()
            if ind_work["trade_date"].dtype == object:
                ind_work["trade_date"] = ind_work["trade_date"].astype(int)
            # 高效方式：每行只有一个1，用 argmax 找行业
            ind_matrix = ind_work[ind_cols].values
            max_idx = ind_matrix.argmax(axis=1)
            # 检查是否全为0
            row_max = ind_matrix.max(axis=1)
            ind_names = np.array([c.replace("L1_", "") for c in ind_cols])
            industry_l1 = np.where(row_max > 0, ind_names[max_idx], None)
            ind_work["industry_l1"] = industry_l1
            ind_long = ind_work[["trade_date", "ts_code", "industry_l1"]]
            df = df.merge(ind_long, on=["trade_date", "ts_code"], how="left")
            del ind_work, ind_matrix
        else:
            df["industry_l1"] = np.nan

        return df

    def _load_industry_for_dates(self, trade_dates: Iterable[int]) -> pd.DataFrame | None:
        """加载指定日期范围的行业数据。"""
        if not self._industry_is_daily:
            return None

        dates = sorted(trade_dates)
        min_d, max_d = min(dates), max(dates)
        min_year = int(str(min_d)[:4])
        max_year = int(str(max_d)[:4])

        dfs = []
        for year in range(min_year, max_year + 1):
            year_dir = self.industry_path / "daily_onehot" / f"year={year}"
            if not year_dir.exists():
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                for f in sorted(month_dir.glob("*.parquet")):
                    dfs.append(pd.read_parquet(f))

        if not dfs:
            return None

        result = pd.concat(dfs, axis=0, ignore_index=True)
        return result

    def _load_fundamentals(self) -> pd.DataFrame | None:
        """加载基本面数据。"""
        if not self.fundamentals_file.exists():
            return None
        return pd.read_parquet(self.fundamentals_file)

    def _load_toplist(self, years: list[int]) -> pd.DataFrame | None:
        """加载龙虎榜数据。"""
        dfs = []
        for year in years:
            year_dir = self.toplist_dir / f"year={year}"
            if not year_dir.exists():
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                for f in sorted(month_dir.glob("*.parquet")):
                    dfs.append(pd.read_parquet(f))
        if not dfs:
            return None
        return pd.concat(dfs, axis=0, ignore_index=True)

    def _load_forecast(self, years: list[int]) -> pd.DataFrame | None:
        """加载业绩预告数据。"""
        dfs = []
        for year in years:
            year_dir = self.forecast_dir / f"year={year}"
            if not year_dir.exists():
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                for f in sorted(month_dir.glob("*.parquet")):
                    dfs.append(pd.read_parquet(f))
        if not dfs:
            return None
        return pd.concat(dfs, axis=0, ignore_index=True)

    def _load_repurchase(self, years: list[int]) -> pd.DataFrame | None:
        """加载回购数据。"""
        dfs = []
        for year in years:
            year_dir = self.repurchase_dir / f"year={year}"
            if not year_dir.exists():
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                for f in sorted(month_dir.glob("*.parquet")):
                    dfs.append(pd.read_parquet(f))
        if not dfs:
            return None
        return pd.concat(dfs, axis=0, ignore_index=True)

    def _load_holdertrade(self, years: list[int]) -> pd.DataFrame | None:
        """加载增减持数据。"""
        dfs = []
        for year in years:
            year_dir = self.holdertrade_dir / f"year={year}"
            if not year_dir.exists():
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                for f in sorted(month_dir.glob("*.parquet")):
                    dfs.append(pd.read_parquet(f))
        if not dfs:
            return None
        return pd.concat(dfs, axis=0, ignore_index=True)

    def _load_blocktrade(self, years: list[int]) -> pd.DataFrame | None:
        """加载大宗交易数据。"""
        dfs = []
        for year in years:
            year_dir = self.blocktrade_dir / f"year={year}"
            if not year_dir.exists():
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                for f in sorted(month_dir.glob("*.parquet")):
                    dfs.append(pd.read_parquet(f))
        if not dfs:
            return None
        return pd.concat(dfs, axis=0, ignore_index=True)

    def _list_all_daily_bar_files(self) -> list[Path]:
        """列出所有日度行情文件。"""
        files = []
        for year_dir in sorted(self.input_dir.iterdir()):
            if not year_dir.is_dir() or not year_dir.name.startswith("year="):
                continue
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                for f in sorted(month_dir.glob("*.parquet")):
                    files.append(f)
        return files

    def _is_file_date_in_output_range(self, file_path: Path) -> bool:
        """判断文件日期是否在输出范围内。"""
        date_str = file_path.stem
        if not date_str.isdigit() or len(date_str) != 8:
            return False
        date_int = int(date_str)
        if self.start_date is not None and date_int < self.start_date:
            return False
        if self.end_date is not None and date_int > self.end_date:
            return False
        return True


# ============================================================================
# CLI 入口
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="上涨侧 Alpha 因子生成器")
    parser.add_argument("--input-dir", help="日度行情输入目录")
    parser.add_argument("--moneyflow-file", help="资金流文件路径")
    parser.add_argument("--fundamentals-file", help="基本面文件路径")
    parser.add_argument("--industry-file", help="行业数据路径（目录或文件）")
    parser.add_argument("--output-dir", help="输出目录")
    parser.add_argument("--start-date", help="开始日期 YYYYMMDD")
    parser.add_argument("--end-date", help="结束日期 YYYYMMDD")
    parser.add_argument("--log-level", default="INFO", help="日志级别")
    parser.add_argument("--log-file", help="日志文件路径")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        filename=args.log_file,
    )

    generator = UpsideAlphaFactorGenerator(
        input_dir=args.input_dir,
        moneyflow_file=args.moneyflow_file,
        fundamentals_file=args.fundamentals_file,
        industry_file=args.industry_file,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    summary = generator.process()

    if summary.has_errors:
        print(f"完成但有错误：{len(summary.errors)} 个错误")
        for err in summary.errors[:5]:
            print(f"  - {err}")
    else:
        print(f"完成：{summary.total_files_written} 个文件, {summary.total_rows_written} 行")


if __name__ == "__main__":
    main()
