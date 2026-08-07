"""Generate enhanced alpha factors from cleaned daily bars and moneyflow data.

This module adds offensive (alpha-seeking) factors that complement the existing
defensive factors.  Source data is never modified; all outputs are written to a
separate directory under ``features_data/enhanced_alpha_factors``.

Factors currently generated:

Momentum (offensive):
  - ``momentum_12m_skip1m``: 12-month return skipping the most recent month (JT momentum)
  - ``momentum_6m_skip2w``: 6-month return skipping the most recent 2 weeks
  - ``momentum_52w_high_dist``: distance to 52-week high (0 = at high, 1 = far below)
  - ``momentum_52w_high_break``: 1 if close is at/near 52w high, else 0 (breakout signal)
  - ``rsi_14``: Relative Strength Index (14-day)
  - ``williams_r_14``: Williams %R (14-day)
  - ``macd``: MACD line (12-26 EMA difference)
  - ``macd_signal``: MACD signal line (9-day EMA of MACD)
  - ``macd_hist``: MACD histogram (MACD - signal)
  - ``consecutive_up_days``: number of consecutive up days
  - ``consecutive_down_days``: number of consecutive down days

Volume-Price (offensive):
  - ``obv``: On-Balance Volume (energy tide)
  - ``obv_ma5_ratio``: OBV / 5-day MA of OBV (momentum of money flow)
  - ``volume_price_trend``: Volume Price Trend (accumulation/distribution)
  - ``turnover_5d``: 5-day average turnover ratio (vol / shares) — approximated by vol ratio
  - ``volume_momentum``: 5d volume change / 20d volume average

Moneyflow (offensive):
  - ``main_net_momentum_5d``: 5-day change in main net inflow ratio
  - ``main_net_trend_20d``: slope of main net inflow over 20 days
  - ``price_mf_divergence``: price up but main net inflow down (bearish divergence) or vice versa

Fundamental momentum (offensive):
  - ``roe_qoq_change``: ROE quarter-over-quarter change
  - ``revenue_yoy_acceleration``: revenue YoY growth acceleration (current - prior)
  - ``gross_margin_change``: gross margin change vs 4 quarters ago
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EnhancedAlphaFactorGenerateSummary:
    """Aggregate statistics for enhanced alpha factor generation."""

    input_dir: Path
    moneyflow_file: Path
    fundamentals_file: Path
    output_dir: Path
    files_read: int
    files_written: int
    rows_read: int
    rows_written: int
    start_date: int | None
    end_date: int | None
    factor_columns: tuple[str, ...]


class EnhancedAlphaFactorGenerator:
    """Generate enhanced offensive alpha factors.

    Reads cleaned daily bars (OHLCV), moneyflow data, and fundamentals, then
    produces a set of alpha-oriented factors.  All inputs are read-only.
    """

    DEFAULT_INPUT_DIR = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/daily_bars"
    )
    DEFAULT_MONEYFLOW_FILE = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/moneyflow/moneyflow.parquet"
    )
    DEFAULT_FUNDAMENTALS_FILE = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/fundamentals/fundamentals.parquet"
    )
    DEFAULT_INDUSTRY_FILE = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/industry/industry.parquet"
    )
    DEFAULT_OUTPUT_DIR = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/features_data/enhanced_alpha_factors"
    )
    DEFAULT_LOG_FILE = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/features_generate/enhanced_alpha_factors.log"
    )

    KEY_COLUMNS: tuple[str, str] = ("trade_date", "ts_code")
    DAILY_BAR_SOURCE_COLUMNS: tuple[str, ...] = (
        "open", "high", "low", "close", "pct_chg", "vol", "amount",
    )
    MAX_LOOKBACK_DAYS = 300  # need ~250 trading days for 52-week high

    MOMENTUM_FACTORS: tuple[str, ...] = (
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
    )

    VOLUME_PRICE_FACTORS: tuple[str, ...] = (
        "obv",
        "obv_ma5_ratio",
        "volume_price_trend",
        "volume_momentum",
    )

    MONEYFLOW_FACTORS: tuple[str, ...] = (
        "main_net_momentum_5d",
        "main_net_trend_20d",
        "price_mf_divergence",
    )

    FUNDAMENTAL_MOMENTUM_FACTORS: tuple[str, ...] = (
        "roe_qoq_change",
        "revenue_yoy_acceleration",
        "gross_margin_change",
    )

    LIQUIDITY_FACTORS: tuple[str, ...] = (
        "amihud_20",
        "turnover_vol_20",
    )

    VOLATILITY_FACTORS: tuple[str, ...] = (
        "downside_vol_20",
        "max_drawdown_60",
    )

    MONEYFLOW_ADVANCED_FACTORS: tuple[str, ...] = (
        "main_net_ema_20",
        "main_net_ema_slope_20",
        "main_net_consecutive",
        "main_net_amount_ratio_20",
    )

    INDUSTRY_RELATIVE_FACTORS: tuple[str, ...] = (
        "industry_rank_ret_20",
        "industry_vol_20",
    )

    INTERACTION_FACTORS: tuple[str, ...] = (
        "momentum_volume_interaction",
        "roe_growth_interaction",
    )

    @property
    def FACTOR_COLUMNS(self) -> tuple[str, ...]:
        return (
            *self.MOMENTUM_FACTORS,
            *self.VOLUME_PRICE_FACTORS,
            *self.MONEYFLOW_FACTORS,
            *self.FUNDAMENTAL_MOMENTUM_FACTORS,
            *self.LIQUIDITY_FACTORS,
            *self.VOLATILITY_FACTORS,
            *self.MONEYFLOW_ADVANCED_FACTORS,
            *self.INDUSTRY_RELATIVE_FACTORS,
            *self.INTERACTION_FACTORS,
        )

    def __init__(
        self,
        input_dir: str | Path | None = None,
        moneyflow_file: str | Path | None = None,
        fundamentals_file: str | Path | None = None,
        industry_file: str | Path | None = None,
        output_dir: str | Path | None = None,
        start_date: str | int | None = None,
        end_date: str | int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.input_dir = Path(input_dir or self.DEFAULT_INPUT_DIR)
        self.moneyflow_file = Path(moneyflow_file or self.DEFAULT_MONEYFLOW_FILE)
        self.fundamentals_file = Path(fundamentals_file or self.DEFAULT_FUNDAMENTALS_FILE)
        self.industry_file = Path(industry_file or self.DEFAULT_INDUSTRY_FILE)
        self.output_dir = Path(output_dir or self.DEFAULT_OUTPUT_DIR)
        self.start_date = self._parse_optional_date(start_date, "start_date")
        self.end_date = self._parse_optional_date(end_date, "end_date")
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            raise NotADirectoryError(f"Daily bars input directory not found: {self.input_dir}")
        if not self.moneyflow_file.exists():
            self.logger.warning("Moneyflow file not found: %s — moneyflow factors will be skipped.", self.moneyflow_file)
        if not self.fundamentals_file.exists():
            self.logger.warning("Fundamentals file not found: %s — fundamental momentum factors will be skipped.", self.fundamentals_file)
        if self.start_date is not None and self.end_date is not None and self.start_date > self.end_date:
            raise ValueError(f"start_date must be <= end_date, got {self.start_date} > {self.end_date}")

    def process(self) -> EnhancedAlphaFactorGenerateSummary:
        """Generate all enhanced alpha factors and write partitioned output.

        Processes data in yearly batches to control memory usage. Each batch
        includes the output year plus MAX_LOOKBACK_DAYS of history for rolling
        calculations. Results are written incrementally.
        """

        all_files = self._list_all_daily_bar_files()
        output_files = [f for f in all_files if self._is_file_date_in_output_range(f)]
        if not output_files:
            raise FileNotFoundError(
                f"No parquet files found under {self.input_dir}/year=*/month=*/*.parquet "
                f"for date range [{self.start_date}, {self.end_date}]"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            "开始生成增强alpha因子：input=%s, output=%s",
            self.input_dir, self.output_dir,
        )
        self.logger.info(
            "输出日期文件数=%d；按年份分批处理；所有输入数据只读。",
            len(output_files),
        )

        # Pre-load shared data (moneyflow, fundamentals, industry) - these are
        # single parquet files that we filter per batch
        mf_df_full = None
        fund_df_full = None
        ind_df_full = None

        if self.moneyflow_file.exists():
            self.logger.info("预加载 moneyflow 数据...")
            mf_df_full = self._load_moneyflow()
            self.logger.info("moneyflow 加载完成：rows=%d", len(mf_df_full))

        if self.fundamentals_file.exists():
            self.logger.info("预加载 fundamentals 数据...")
            fund_df_full = self._load_fundamentals()
            self.logger.info("fundamentals 加载完成：rows=%d", len(fund_df_full))

        if self.industry_file.exists():
            self.logger.info("预加载 industry 数据...")
            ind_df_full = self._load_industry()
            self.logger.info("industry 加载完成：rows=%d", len(ind_df_full))

        # Split output files by year for batch processing
        year_batches: dict[int, list[Path]] = {}
        for f in output_files:
            year = int(f.parent.parent.name.replace("year=", ""))
            year_batches.setdefault(year, []).append(f)

        total_files_read = 0
        total_rows_read = 0
        total_files_written = 0
        total_rows_written = 0

        for year in sorted(year_batches.keys()):
            batch_output_files = year_batches[year]
            batch_read_files = self._select_files_with_history(all_files, batch_output_files)
            batch_output_dates = {self._date_from_file_name(f) for f in batch_output_files}

            self.logger.info(
                "处理年份 %d：输出文件数=%d, 读取文件数=%d",
                year, len(batch_output_files), len(batch_read_files),
            )

            # Load batch bars data
            bars_df = self._load_daily_bars(batch_read_files)
            batch_rows_read = len(bars_df)
            total_rows_read += batch_rows_read
            total_files_read += len(batch_read_files)

            # Compute factors that only need bars data
            factor_df = self._generate_momentum_factors(bars_df)
            factor_df = self._generate_volume_price_factors(factor_df)
            factor_df = self._generate_liquidity_factors(factor_df)
            factor_df = self._generate_volatility_factors(factor_df)

            # Moneyflow factors
            if mf_df_full is not None:
                batch_mf = mf_df_full[mf_df_full["trade_date"].isin(factor_df["trade_date"].unique())].copy()
                bars_amount = bars_df[["trade_date", "ts_code", "amount"]].copy()
                batch_mf = batch_mf.merge(bars_amount, on=["trade_date", "ts_code"], how="left")
                batch_mf["amount_wan"] = batch_mf["amount"] / 10.0
                factor_df = self._generate_moneyflow_factors(factor_df, batch_mf)
                factor_df = self._generate_moneyflow_advanced_factors(factor_df)
                del batch_mf, bars_amount
            else:
                for col in [*self.MONEYFLOW_FACTORS, *self.MONEYFLOW_ADVANCED_FACTORS]:
                    factor_df[col] = np.nan

            # Fundamental momentum factors
            if fund_df_full is not None:
                factor_df = self._generate_fundamental_momentum_factors(factor_df, fund_df_full, bars_df)
            else:
                for col in self.FUNDAMENTAL_MOMENTUM_FACTORS:
                    factor_df[col] = np.nan

            # Industry-relative factors
            if ind_df_full is not None:
                batch_ind = ind_df_full[ind_df_full["trade_date"].isin(factor_df["trade_date"].unique())].copy()
                factor_df = self._generate_industry_relative_factors(factor_df, batch_ind)
                del batch_ind
            else:
                for col in self.INDUSTRY_RELATIVE_FACTORS:
                    factor_df[col] = np.nan

            # Interaction factors
            factor_df = self._generate_interaction_factors(factor_df)

            # Slice to output dates and write
            output_df = factor_df.loc[
                factor_df["trade_date"].isin(batch_output_dates),
                [*self.KEY_COLUMNS, *self.FACTOR_COLUMNS],
            ].copy()
            output_df = output_df.sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)

            files_written = self._write_partitioned_by_date(output_df)
            total_files_written += files_written
            total_rows_written += len(output_df)

            self.logger.info(
                "年份 %d 完成：写入 %d 个文件, %d 行",
                year, files_written, len(output_df),
            )

            # Free memory
            del factor_df, output_df, bars_df
            import gc
            gc.collect()

        summary = EnhancedAlphaFactorGenerateSummary(
            input_dir=self.input_dir,
            moneyflow_file=self.moneyflow_file,
            fundamentals_file=self.fundamentals_file,
            output_dir=self.output_dir,
            files_read=total_files_read,
            files_written=total_files_written,
            rows_read=total_rows_read,
            rows_written=total_rows_written,
            start_date=self.start_date,
            end_date=self.end_date,
            factor_columns=self.FACTOR_COLUMNS,
        )
        self.logger.info(
            "增强alpha因子生成完成：total_files_written=%d, total_rows_written=%d",
            total_files_written, total_rows_written,
        )
        return summary

    # ------------------------------------------------------------------
    # Momentum factors
    # ------------------------------------------------------------------
    def _generate_momentum_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
        stock_group = df.groupby("ts_code", sort=False, group_keys=False)

        close = df["close"].astype("float64")
        pct_chg = df["pct_chg"].astype("float64") / 100.0  # convert pct_chg to decimal

        # 12-month (252 trading days) momentum, skip 1 month (22 days)
        df["momentum_12m_skip1m"] = (
            close / stock_group["close"].shift(252).replace(0, np.nan) - 1
        )
        # Remove the most recent month's return from the 12m return
        ret_1m = close / stock_group["close"].shift(22).replace(0, np.nan) - 1
        df["momentum_12m_skip1m"] = (
            (1 + df["momentum_12m_skip1m"]) / (1 + ret_1m) - 1
        )

        # 6-month (126 trading days) momentum, skip 2 weeks (10 days)
        ret_6m = close / stock_group["close"].shift(126).replace(0, np.nan) - 1
        ret_2w = close / stock_group["close"].shift(10).replace(0, np.nan) - 1
        df["momentum_6m_skip2w"] = (1 + ret_6m) / (1 + ret_2w) - 1

        # 52-week high distance
        high_252 = stock_group["high"].transform(
            lambda s: s.rolling(window=252, min_periods=126).max()
        )
        df["momentum_52w_high_dist"] = 1 - close / high_252.replace(0, np.nan)

        # 52-week high breakout (close within 2% of 52w high)
        df["momentum_52w_high_break"] = (
            (close >= high_252 * 0.98).astype("float64")
        )
        df.loc[high_252.isna(), "momentum_52w_high_break"] = np.nan

        # RSI (14-day)
        gain = pct_chg.clip(lower=0)
        loss = (-pct_chg).clip(lower=0)
        avg_gain = stock_group[gain.name if False else "pct_chg"].transform(
            lambda s: s.clip(lower=0).rolling(window=14, min_periods=10).mean()
        )
        # Need to compute properly
        avg_gain = pd.Series(index=df.index, dtype="float64")
        avg_loss = pd.Series(index=df.index, dtype="float64")
        for _, group in df.groupby("ts_code", sort=False):
            g = group["pct_chg"].astype("float64") / 100.0
            gains = g.clip(lower=0)
            losses = (-g).clip(lower=0)
            ag = gains.rolling(window=14, min_periods=10).mean()
            al = losses.rolling(window=14, min_periods=10).mean()
            avg_gain.loc[group.index] = ag.values
            avg_loss.loc[group.index] = al.values

        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi_14"] = 100 - 100 / (1 + rs)
        df.loc[avg_loss == 0, "rsi_14"] = 100.0

        # Williams %R (14-day)
        high_14 = stock_group["high"].transform(
            lambda s: s.rolling(window=14, min_periods=10).max()
        )
        low_14 = stock_group["low"].transform(
            lambda s: s.rolling(window=14, min_periods=10).min()
        )
        df["williams_r_14"] = -100 * (high_14 - close) / (high_14 - low_14).replace(0, np.nan)

        # MACD (12, 26, 9)
        ema12 = pd.Series(index=df.index, dtype="float64")
        ema26 = pd.Series(index=df.index, dtype="float64")
        for _, group in df.groupby("ts_code", sort=False):
            c = group["close"].astype("float64")
            e12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
            e26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
            ema12.loc[group.index] = e12.values
            ema26.loc[group.index] = e26.values

        df["macd"] = ema12 - ema26
        # Signal line: 9-day EMA of MACD
        macd_signal = pd.Series(index=df.index, dtype="float64")
        for ts, group_idx in df.groupby("ts_code", sort=False).groups.items():
            macd_vals = df.loc[group_idx, "macd"].astype("float64")
            sig = macd_vals.ewm(span=9, adjust=False, min_periods=9).mean()
            macd_signal.loc[group_idx] = sig.values
        df["macd_signal"] = macd_signal
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Consecutive up/down days
        is_up = (pct_chg > 0).astype("int")
        is_down = (pct_chg < 0).astype("int")

        consec_up = pd.Series(index=df.index, dtype="float64")
        consec_down = pd.Series(index=df.index, dtype="float64")
        for _, group in df.groupby("ts_code", sort=False):
            g = group["pct_chg"].astype("float64") / 100.0
            up = (g > 0).astype("int")
            down = (g < 0).astype("int")
            # Consecutive up days counter
            cu = up.groupby((up != up.shift()).cumsum()).cumcount() + 1
            cu = cu * up  # reset to 0 on down days
            cd = down.groupby((down != down.shift()).cumsum()).cumcount() + 1
            cd = cd * down
            consec_up.loc[group.index] = cu.values.astype("float64")
            consec_down.loc[group.index] = cd.values.astype("float64")

        df["consecutive_up_days"] = consec_up
        df["consecutive_down_days"] = consec_down

        nan_counts = df.loc[:, list(self.MOMENTUM_FACTORS)].isna().sum().to_dict()
        self.logger.info("动量类因子计算完成：momentum_factor_nan_counts=%s", nan_counts)
        return df

    # ------------------------------------------------------------------
    # Volume-price factors
    # ------------------------------------------------------------------
    def _generate_volume_price_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        stock_group = df.groupby("ts_code", sort=False, group_keys=False)
        close = df["close"].astype("float64")
        vol = df["vol"].astype("float64")
        pct_chg = df["pct_chg"].astype("float64") / 100.0

        # OBV: On-Balance Volume
        obv = pd.Series(index=df.index, dtype="float64")
        for _, group in df.groupby("ts_code", sort=False):
            v = group["vol"].astype("float64")
            c = group["close"].astype("float64")
            direction = np.sign(c.diff().fillna(0))
            obv_vals = (direction * v).cumsum()
            obv.loc[group.index] = obv_vals.values
        df["obv"] = obv

        # OBV / 5-day MA of OBV
        obv_ma5 = stock_group["obv"].transform(
            lambda s: s.rolling(window=5, min_periods=3).mean()
        )
        df["obv_ma5_ratio"] = df["obv"] / obv_ma5.replace(0, np.nan)

        # Volume Price Trend (VPT): cumulative of (volume * pct_change)
        vpt = pd.Series(index=df.index, dtype="float64")
        for _, group in df.groupby("ts_code", sort=False):
            v = group["vol"].astype("float64")
            pc = group["pct_chg"].astype("float64") / 100.0
            vpt_vals = (v * pc).cumsum()
            vpt.loc[group.index] = vpt_vals.values
        df["volume_price_trend"] = vpt

        # Volume momentum: 5d avg vol / 20d avg vol
        vol_ma5 = stock_group["vol"].transform(
            lambda s: s.rolling(window=5, min_periods=3).mean()
        )
        vol_ma20 = stock_group["vol"].transform(
            lambda s: s.rolling(window=20, min_periods=10).mean()
        )
        df["volume_momentum"] = vol_ma5 / vol_ma20.replace(0, np.nan)

        nan_counts = df.loc[:, list(self.VOLUME_PRICE_FACTORS)].isna().sum().to_dict()
        self.logger.info("量价类因子计算完成：volume_price_factor_nan_counts=%s", nan_counts)
        return df

    # ------------------------------------------------------------------
    # Moneyflow factors
    # ------------------------------------------------------------------
    def _generate_moneyflow_factors(self, df: pd.DataFrame, mf_df: pd.DataFrame) -> pd.DataFrame:
        """Generate moneyflow-based offensive factors."""
        # Compute main_net_inflow and main_net_ratio
        mf_slim = mf_df[["trade_date", "ts_code"]].copy()

        buy_lg = pd.to_numeric(mf_df.get("buy_lg_amount", 0), errors="coerce")
        sell_lg = pd.to_numeric(mf_df.get("sell_lg_amount", 0), errors="coerce")
        buy_elg = pd.to_numeric(mf_df.get("buy_elg_amount", 0), errors="coerce")
        sell_elg = pd.to_numeric(mf_df.get("sell_elg_amount", 0), errors="coerce")
        main_net = (buy_lg + buy_elg) - (sell_lg + sell_elg)

        amount_wan = pd.to_numeric(mf_df.get("amount_wan", np.nan), errors="coerce")
        mf_slim["main_net_inflow"] = main_net
        mf_slim["main_net_ratio"] = main_net / amount_wan.replace(0, np.nan)

        merged = df.merge(mf_slim, on=["trade_date", "ts_code"], how="left")

        stock_group = merged.groupby("ts_code", sort=False, group_keys=False)

        # Main net inflow momentum: 5d change in main_net_ratio
        main_net_5d_avg = stock_group["main_net_ratio"].transform(
            lambda s: s.rolling(window=5, min_periods=3).mean()
        )
        main_net_20d_avg = stock_group["main_net_ratio"].transform(
            lambda s: s.rolling(window=20, min_periods=10).mean()
        )
        merged["main_net_momentum_5d"] = main_net_5d_avg - main_net_20d_avg

        # Main net trend 20d: slope (simplified as last 5d avg vs prior 15d avg)
        main_net_prior15 = stock_group["main_net_ratio"].transform(
            lambda s: s.shift(5).rolling(window=15, min_periods=8).mean()
        )
        merged["main_net_trend_20d"] = (main_net_5d_avg - main_net_prior15) / main_net_prior15.abs().replace(0, np.nan)

        # Price-moneyflow divergence:
        # Price up (5d) but main net inflow down = bearish divergence (negative)
        # Price down but main net inflow up = bullish divergence (positive)
        ret_5d = stock_group["close"].transform(
            lambda s: s.pct_change(periods=5)
        )
        mf_5d_change = main_net_5d_avg - stock_group["main_net_ratio"].shift(5)
        # Divergence: positive when price and moneyflow disagree in bullish way
        merged["price_mf_divergence"] = np.where(
            (ret_5d < 0) & (mf_5d_change > 0), 1.0,  # bullish divergence
            np.where(
                (ret_5d > 0) & (mf_5d_change < 0), -1.0,  # bearish divergence
                0.0  # no divergence
            )
        )
        merged.loc[ret_5d.isna() | mf_5d_change.isna(), "price_mf_divergence"] = np.nan

        nan_counts = merged.loc[:, list(self.MONEYFLOW_FACTORS)].isna().sum().to_dict()
        self.logger.info("资金流类因子计算完成：moneyflow_factor_nan_counts=%s", nan_counts)
        return merged

    # ------------------------------------------------------------------
    # Fundamental momentum factors
    # ------------------------------------------------------------------
    def _generate_fundamental_momentum_factors(
        self, df: pd.DataFrame, fund_df: pd.DataFrame, bars_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Generate fundamental momentum factors (PIT).

        For each stock we compute quarter-over-quarter changes in ROE, revenue
        YoY growth acceleration, and gross margin change.  Values are then
        merged point-in-time: only reports with ``ann_date <= trade_date`` are
        visible on each trade date.
        """
        if fund_df.empty:
            for col in self.FUNDAMENTAL_MOMENTUM_FACTORS:
                df[col] = np.nan
            self.logger.warning("基本面动量因子跳过：fund_df为空")
            return df

        fund_df = fund_df.sort_values(["ts_code", "end_date", "ann_date"]).reset_index(drop=True)

        # Compute quarter-over-quarter changes (per stock, ordered by report period)
        fund_df["roe_prev"] = fund_df.groupby("ts_code")["roe"].shift(1)
        fund_df["or_yoy_prev"] = fund_df.groupby("ts_code")["or_yoy"].shift(1)
        fund_df["gross_margin_prev"] = fund_df.groupby("ts_code")["grossprofit_margin"].shift(1)
        fund_df["roe_qoq_change"] = fund_df["roe"] - fund_df["roe_prev"]
        fund_df["revenue_yoy_acceleration"] = fund_df["or_yoy"] - fund_df["or_yoy_prev"]
        fund_df["gross_margin_change"] = fund_df["grossprofit_margin"] - fund_df["gross_margin_prev"]

        # Keep only needed columns and sort by ann_date for PIT iteration
        fund_cols = ["ts_code", "ann_date", "end_date"] + list(self.FUNDAMENTAL_MOMENTUM_FACTORS)
        fund_sorted = fund_df[fund_cols].sort_values("ann_date").reset_index(drop=True)

        trade_dates = sorted(df["trade_date"].unique())
        result = df.copy()
        for col in self.FUNDAMENTAL_MOMENTUM_FACTORS:
            result[col] = np.nan

        # PIT merge using merge_asof-style approach: for each trade date,
        # take the latest report per stock with ann_date <= trade_date.
        # Implemented as a rolling dict for O(n) performance.
        latest_fund: dict[str, pd.Series] = {}
        fund_idx = 0
        fund_count = len(fund_sorted)

        for trade_date in trade_dates:
            while fund_idx < fund_count and int(fund_sorted.iat[fund_idx, 1]) <= trade_date:
                row = fund_sorted.iloc[fund_idx]
                latest_fund[row["ts_code"]] = row
                fund_idx += 1

            if not latest_fund:
                continue

            mask = result["trade_date"] == trade_date
            day_stocks = result.loc[mask, "ts_code"]
            for col in self.FUNDAMENTAL_MOMENTUM_FACTORS:
                col_idx = fund_sorted.columns.get_loc(col)
                vals = day_stocks.map(
                    lambda ts, col_idx=col_idx:
                    latest_fund[ts].iat[col_idx] if ts in latest_fund else np.nan
                )
                result.loc[mask, col] = pd.to_numeric(vals.values, errors="coerce")

        nan_counts = result.loc[:, list(self.FUNDAMENTAL_MOMENTUM_FACTORS)].isna().sum().to_dict()
        self.logger.info("基本面动量因子计算完成：fund_momentum_factor_nan_counts=%s", nan_counts)
        return result

    # ------------------------------------------------------------------
    # Liquidity factors
    # ------------------------------------------------------------------
    def _generate_liquidity_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Amihud illiquidity and turnover volatility."""
        stock_group = df.groupby("ts_code", sort=False, group_keys=False)
        close = df["close"].astype("float64")
        amount = df["amount"].astype("float64")
        vol = df["vol"].astype("float64")
        pct_chg = df["pct_chg"].astype("float64") / 100.0

        # Amihud_20: |ret| / amount, 20-day rolling mean
        amihud_daily = pct_chg.abs() / amount.replace(0, np.nan)
        df["amihud_20"] = stock_group["pct_chg"].transform(
            lambda s: (s.abs() / df.loc[s.index, "amount"].replace(0, np.nan)).rolling(20, min_periods=10).mean()
        )
        # Fix: use proper per-stock rolling
        amihud_20 = pd.Series(index=df.index, dtype="float64")
        for _, group in df.groupby("ts_code", sort=False):
            ret = group["pct_chg"].astype("float64") / 100.0
            amt = group["amount"].astype("float64").replace(0, np.nan)
            daily = ret.abs() / amt
            amihud_20.loc[group.index] = daily.rolling(20, min_periods=10).mean().values
        df["amihud_20"] = amihud_20

        # TurnoverVol_20: 20-day rolling std of volume
        df["turnover_vol_20"] = stock_group["vol"].transform(
            lambda s: s.rolling(20, min_periods=10).std()
        )

        nan_counts = df.loc[:, list(self.LIQUIDITY_FACTORS)].isna().sum().to_dict()
        self.logger.info("流动性因子计算完成：liquidity_factor_nan_counts=%s", nan_counts)
        return df

    # ------------------------------------------------------------------
    # Volatility factors
    # ------------------------------------------------------------------
    def _generate_volatility_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Downside volatility and max drawdown."""
        stock_group = df.groupby("ts_code", sort=False, group_keys=False)
        close = df["close"].astype("float64")
        pct_chg = df["pct_chg"].astype("float64") / 100.0

        # DownsideVol_20: std of negative returns only
        downside_vol = pd.Series(index=df.index, dtype="float64")
        for _, group in df.groupby("ts_code", sort=False):
            ret = group["pct_chg"].astype("float64") / 100.0
            neg_ret = ret.where(ret < 0, 0.0)
            downside_vol.loc[group.index] = neg_ret.rolling(20, min_periods=10).std().values
        df["downside_vol_20"] = downside_vol

        # MaxDrawdown_60: close / rolling_max(60) - 1
        df["max_drawdown_60"] = stock_group["close"].transform(
            lambda s: s / s.rolling(60, min_periods=30).max() - 1
        )

        nan_counts = df.loc[:, list(self.VOLATILITY_FACTORS)].isna().sum().to_dict()
        self.logger.info("波动率因子计算完成：volatility_factor_nan_counts=%s", nan_counts)
        return df

    # ------------------------------------------------------------------
    # Advanced moneyflow factors
    # ------------------------------------------------------------------
    def _generate_moneyflow_advanced_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """EMA, slope, consecutive days, amount ratio of main net inflow."""
        stock_group = df.groupby("ts_code", sort=False, group_keys=False)

        # MainNetEMA20: 20-day EMA of main_net_inflow
        main_net_ema = pd.Series(index=df.index, dtype="float64")
        main_net_ema_slope = pd.Series(index=df.index, dtype="float64")
        main_net_consec = pd.Series(index=df.index, dtype="float64")

        for _, group in df.groupby("ts_code", sort=False):
            mn = group["main_net_inflow"].astype("float64")
            # EMA20
            ema20 = mn.ewm(span=20, adjust=False, min_periods=10).mean()
            main_net_ema.loc[group.index] = ema20.values
            # Slope of EMA20 over 20 days (linear regression slope approximated)
            slope = ema20.diff(19) / 19.0
            main_net_ema_slope.loc[group.index] = slope.values
            # Consecutive positive days
            is_pos = (mn > 0).astype("int")
            consec = is_pos.groupby((is_pos != is_pos.shift()).cumsum()).cumcount() + 1
            consec = consec * is_pos
            main_net_consec.loc[group.index] = consec.values.astype("float64")

        df["main_net_ema_20"] = main_net_ema
        df["main_net_ema_slope_20"] = main_net_ema_slope
        df["main_net_consecutive"] = main_net_consec

        # MainNetAmountRatio_20: rolling mean of (main_net / amount)
        # main_net_ratio already exists from moneyflow factors
        if "main_net_ratio" in df.columns:
            df["main_net_amount_ratio_20"] = stock_group["main_net_ratio"].transform(
                lambda s: s.rolling(20, min_periods=10).mean()
            )
        else:
            df["main_net_amount_ratio_20"] = np.nan

        nan_counts = df.loc[:, list(self.MONEYFLOW_ADVANCED_FACTORS)].isna().sum().to_dict()
        self.logger.info("高级资金流因子计算完成：mf_advanced_factor_nan_counts=%s", nan_counts)
        return df

    # ------------------------------------------------------------------
    # Industry-relative factors
    # ------------------------------------------------------------------
    def _generate_industry_relative_factors(self, df: pd.DataFrame, ind_df: pd.DataFrame) -> pd.DataFrame:
        """Industry rank of returns and industry volatility."""
        ind_slim = ind_df[["trade_date", "ts_code", "l1_name"]].copy()
        merged = df.merge(ind_slim, on=["trade_date", "ts_code"], how="left")

        # IndustryRank_RET20: percentile rank of ret_20 within industry
        if "ret_20" in merged.columns:
            merged["industry_rank_ret_20"] = merged.groupby(["trade_date", "l1_name"])["ret_20"].rank(
                method="average", pct=True
            )
        else:
            # Compute ret_20 from close
            stock_group = merged.groupby("ts_code", sort=False, group_keys=False)
            close = merged["close"].astype("float64")
            merged["_ret_20_temp"] = close / stock_group["close"].shift(20).replace(0, np.nan) - 1
            merged["industry_rank_ret_20"] = merged.groupby(["trade_date", "l1_name"])["_ret_20_temp"].rank(
                method="average", pct=True
            )
            merged = merged.drop(columns=["_ret_20_temp"])

        # IndustryVol_20: 20-day rolling std of industry daily return
        # Step 1: compute industry daily return (mean of pct_chg)
        ind_daily_ret = merged.groupby(["trade_date", "l1_name"])["pct_chg"].mean().reset_index()
        ind_daily_ret.columns = ["trade_date", "l1_name", "ind_daily_ret"]
        ind_daily_ret = ind_daily_ret.sort_values(["l1_name", "trade_date"]).reset_index(drop=True)
        # Step 2: rolling 20d std per industry
        ind_daily_ret["industry_vol_20"] = ind_daily_ret.groupby("l1_name")["ind_daily_ret"].transform(
            lambda s: s.rolling(20, min_periods=10).std()
        )
        # Step 3: merge back
        merged = merged.merge(
            ind_daily_ret[["trade_date", "l1_name", "industry_vol_20"]],
            on=["trade_date", "l1_name"],
            how="left",
        )

        merged = merged.drop(columns=["l1_name"])
        nan_counts = merged.loc[:, list(self.INDUSTRY_RELATIVE_FACTORS)].isna().sum().to_dict()
        self.logger.info("行业相对因子计算完成：industry_relative_factor_nan_counts=%s", nan_counts)
        return merged

    # ------------------------------------------------------------------
    # Interaction factors
    # ------------------------------------------------------------------
    def _generate_interaction_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum×Volume and ROE×Growth interactions."""
        stock_group = df.groupby("ts_code", sort=False, group_keys=False)
        close = df["close"].astype("float64")
        vol = df["vol"].astype("float64")

        # Compute ret_20 locally if not present
        if "ret_20" not in df.columns:
            df["_ret_20_temp"] = close / stock_group["close"].shift(20).replace(0, np.nan) - 1
            ret_20_col = "_ret_20_temp"
        else:
            ret_20_col = "ret_20"

        # Compute vol_ratio_20 locally if not present
        if "vol_ratio_20" not in df.columns:
            vol_ma20 = stock_group["vol"].transform(
                lambda s: s.rolling(window=20, min_periods=10).mean()
            )
            df["_vol_ratio_20_temp"] = vol / vol_ma20.replace(0, np.nan)
            vol_ratio_col = "_vol_ratio_20_temp"
        else:
            vol_ratio_col = "vol_ratio_20"

        # Momentum×Volume interaction
        df["momentum_volume_interaction"] = (
            pd.to_numeric(df[ret_20_col], errors="coerce") *
            pd.to_numeric(df[vol_ratio_col], errors="coerce")
        )

        # Clean up temp columns
        for col in ["_ret_20_temp", "_vol_ratio_20_temp"]:
            if col in df.columns:
                df = df.drop(columns=[col])

        # ROE×Growth interaction: use fundamental momentum factors as proxies
        # roe_qoq_change and revenue_yoy_acceleration are available in enhanced alpha
        roe_col = None
        yoy_col = None
        for col in ["roe_qoq_change", "roe"]:
            if col in df.columns and df[col].notna().any():
                roe_col = col
                break
        for col in ["revenue_yoy_acceleration", "revenue_yoy"]:
            if col in df.columns and df[col].notna().any():
                yoy_col = col
                break

        if roe_col and yoy_col:
            df["roe_growth_interaction"] = (
                pd.to_numeric(df[roe_col], errors="coerce") *
                pd.to_numeric(df[yoy_col], errors="coerce")
            )
        else:
            df["roe_growth_interaction"] = np.nan

        nan_counts = df.loc[:, list(self.INTERACTION_FACTORS)].isna().sum().to_dict()
        self.logger.info("交互因子计算完成：interaction_factor_nan_counts=%s", nan_counts)
        return df

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------
    def _load_moneyflow(self) -> pd.DataFrame:
        df = pd.read_parquet(self.moneyflow_file)
        df["trade_date"] = self._normalize_yyyymmdd(df["trade_date"], "trade_date", self.moneyflow_file)
        df["ts_code"] = df["ts_code"].astype("string").str.strip()
        return df

    def _load_fundamentals(self) -> pd.DataFrame:
        df = pd.read_parquet(self.fundamentals_file)
        df["ts_code"] = df["ts_code"].astype("string").str.strip()
        df["ann_date"] = self._normalize_yyyymmdd(df["ann_date"], "ann_date", self.fundamentals_file)
        df["end_date"] = self._normalize_yyyymmdd(df["end_date"], "end_date", self.fundamentals_file)
        for col in ["roe", "roa", "or_yoy", "gross_margin", "grossprofit_margin", "debt_to_assets", "eps", "bps"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def _load_industry(self) -> pd.DataFrame:
        df = pd.read_parquet(self.industry_file, columns=["trade_date", "ts_code", "l1_name"])
        df["trade_date"] = self._normalize_yyyymmdd(df["trade_date"], "trade_date", self.industry_file)
        df["ts_code"] = df["ts_code"].astype("string").str.strip()
        df["l1_name"] = df["l1_name"].astype("string").str.strip()
        return df

    def _list_all_daily_bar_files(self) -> list[Path]:
        files = sorted(self.input_dir.glob("year=*/month=*/*.parquet"))
        valid = []
        for fp in files:
            if self._date_from_file_name(fp) is not None:
                valid.append(fp)
            else:
                self.logger.warning("跳过无法解析日期的 daily_bars 文件：%s", fp)
        if not valid:
            raise FileNotFoundError(f"No parquet files found under {self.input_dir}/year=*/month=*/*.parquet")
        return valid

    def _is_file_date_in_output_range(self, file_path: Path) -> bool:
        fd = self._date_from_file_name(file_path)
        if fd is None:
            return False
        if self.start_date is not None and fd < self.start_date:
            return False
        if self.end_date is not None and fd > self.end_date:
            return False
        return True

    def _select_files_with_history(
        self, all_files: Sequence[Path], output_files: Sequence[Path]
    ) -> list[Path]:
        output_set = set(output_files)
        output_indices = [i for i, f in enumerate(all_files) if f in output_set]
        if not output_indices:
            return []
        first_idx = max(0, min(output_indices) - self.MAX_LOOKBACK_DAYS)
        last_idx = max(output_indices)
        return list(all_files[first_idx : last_idx + 1])

    def _load_daily_bars(self, parquet_files: Sequence[Path]) -> pd.DataFrame:
        frames = []
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            prepared = self._prepare_daily_bars(df, pf)
            frames.append(prepared)
        if not frames:
            return pd.DataFrame(columns=[*self.KEY_COLUMNS, *self.DAILY_BAR_SOURCE_COLUMNS])
        combined = pd.concat(frames, ignore_index=True)
        self.logger.info("daily_bars 读取完成：files=%d, rows=%d", len(parquet_files), len(combined))
        return combined

    def _prepare_daily_bars(self, daily_df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
        prepared = daily_df.copy()
        prepared["trade_date"] = self._normalize_yyyymmdd(
            prepared["trade_date"], "trade_date", file_path
        )
        prepared["ts_code"] = prepared["ts_code"].astype("string").str.strip()
        for col in self.DAILY_BAR_SOURCE_COLUMNS:
            prepared[col] = pd.to_numeric(prepared[col], errors="coerce")
        return prepared

    def _write_partitioned_by_date(self, output_df: pd.DataFrame) -> int:
        files_written = 0
        for trade_date, date_df in output_df.groupby("trade_date", sort=True):
            td_int = int(trade_date)
            td_text = f"{td_int:08d}"
            out_file = (
                self.output_dir
                / f"year={td_text[:4]}"
                / f"month={td_text[4:6]}"
                / f"{td_text}.parquet"
            )
            out_file.parent.mkdir(parents=True, exist_ok=True)
            date_df.to_parquet(out_file, index=False)
            files_written += 1
        self.logger.info("增强alpha因子写入完成：files=%d", files_written)
        return files_written

    def _log_summary(
        self, summary: EnhancedAlphaFactorGenerateSummary, output_df: pd.DataFrame
    ) -> None:
        self.logger.info(
            "增强alpha因子生成汇总：files_read=%d, files_written=%d, rows_read=%d, rows_written=%d, "
            "start_date=%s, end_date=%s, factors=%d, output=%s",
            summary.files_read, summary.files_written, summary.rows_read, summary.rows_written,
            summary.start_date, summary.end_date, len(summary.factor_columns), summary.output_dir,
        )
        if not output_df.empty:
            nan_counts = output_df.loc[:, list(summary.factor_columns)].isna().sum().to_dict()
            self.logger.info("输出因子缺失值统计：%s", nan_counts)

    @staticmethod
    def _validate_columns(columns: Iterable[str], required: Sequence[str], file_path: Path) -> None:
        col_set = set(columns)
        missing = [c for c in required if c not in col_set]
        if missing:
            raise ValueError(f"{file_path} missing required columns: {missing}")

    @classmethod
    def _normalize_yyyymmdd(cls, series: pd.Series, column_name: str, file_path: Path) -> pd.Series:
        normalized = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
        parsed = pd.to_datetime(normalized, format="%Y%m%d", errors="coerce")
        invalid = int(parsed.isna().sum())
        if invalid:
            raise ValueError(f"{file_path} has {invalid} invalid {column_name} rows.")
        return parsed.dt.strftime("%Y%m%d").astype("int32")

    @staticmethod
    def _parse_optional_date(value: str | int | None, name: str) -> int | None:
        if value is None or value == "":
            return None
        text = str(value).strip()
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"{name} must be in YYYYMMDD format, got {value!r}")
        return int(parsed.strftime("%Y%m%d"))

    @staticmethod
    def _date_from_file_name(file_path: Path) -> int | None:
        try:
            return int(file_path.stem)
        except ValueError:
            return None


def configure_logging(log_level: str = "INFO", log_file: str | Path | None = None) -> None:
    handlers = [logging.StreamHandler()]
    target = Path(log_file) if log_file else EnhancedAlphaFactorGenerator.DEFAULT_LOG_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(target, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate enhanced offensive alpha factors.")
    parser.add_argument("--input-dir", default=str(EnhancedAlphaFactorGenerator.DEFAULT_INPUT_DIR),
                        help="Cleaned daily_bars directory.")
    parser.add_argument("--moneyflow-file", default=str(EnhancedAlphaFactorGenerator.DEFAULT_MONEYFLOW_FILE),
                        help="Cleaned moneyflow parquet file.")
    parser.add_argument("--fundamentals-file", default=str(EnhancedAlphaFactorGenerator.DEFAULT_FUNDAMENTALS_FILE),
                        help="Cleaned fundamentals parquet file.")
    parser.add_argument("--industry-file", default=str(EnhancedAlphaFactorGenerator.DEFAULT_INDUSTRY_FILE),
                        help="Cleaned industry parquet file.")
    parser.add_argument("--output-dir", default=str(EnhancedAlphaFactorGenerator.DEFAULT_OUTPUT_DIR),
                        help="Output directory for enhanced alpha factors.")
    parser.add_argument("--start-date", default=None, help="Start date (YYYYMMDD).")
    parser.add_argument("--end-date", default=None, help="End date (YYYYMMDD).")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    parser.add_argument("--log-file", default=str(EnhancedAlphaFactorGenerator.DEFAULT_LOG_FILE),
                        help="Log file path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> EnhancedAlphaFactorGenerateSummary:
    args = parse_args(argv)
    configure_logging(log_level=args.log_level, log_file=args.log_file)
    generator = EnhancedAlphaFactorGenerator(
        input_dir=args.input_dir,
        moneyflow_file=args.moneyflow_file,
        fundamentals_file=args.fundamentals_file,
        industry_file=args.industry_file,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    return generator.process()


if __name__ == "__main__":
    main()


__all__ = [
    "EnhancedAlphaFactorGenerateSummary",
    "EnhancedAlphaFactorGenerator",
    "configure_logging",
    "main",
    "parse_args",
]
