"""Risk exposure matrix builder.

Constructs daily risk factor exposures (X_t) from market and fundamental
data.  V1 implements 8 style factors plus industry one-hot factors:

Style factors (cross-sectionally winsorized + z-scored):
    SIZE, BETA, MOMENTUM, VALUE, GROWTH, VOLATILITY, LIQUIDITY, LEVERAGE

Industry factors:
    One-hot encoding of Shenwan L1 industry classification (N-1 to avoid
    perfect collinearity with the intercept in cross-sectional regression).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

STYLE_FACTOR_NAMES: tuple[str, ...] = (
    "SIZE",
    "BETA",
    "MOMENTUM",
    "VALUE",
    "GROWTH",
    "VOLATILITY",
    "LIQUIDITY",
    "LEVERAGE",
)


@dataclass(frozen=True)
class ExposureBuildResult:
    """Result of risk exposure building."""

    exposures: pd.DataFrame
    style_factors: tuple[str, ...]
    industry_factors: tuple[str, ...]
    start_date: int
    end_date: int
    n_stocks: int
    n_dates: int


class RiskExposureBuilder:
    """Build daily risk exposure matrices.

    Parameters
    ----------
    config
        Risk model config dictionary (loaded from risk_model_v1.json).
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.style_factors: tuple[str, ...] = tuple(
            name.upper() for name in config.get("style_factors", [])
        )
        self.industry_cfg = config.get("industry_factor", {})
        self.cs_cfg = config.get("cross_sectional", {})
        self.data_cfg = config.get("data", {})
        self.beta_cfg = config.get("beta", {})
        self.size_cfg = config.get("size", {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        daily_bars: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
        industry_onehot: pd.DataFrame | None = None,
    ) -> ExposureBuildResult:
        """Build the full exposure matrix from input data.

        Parameters
        ----------
        daily_bars
            DataFrame with columns ``trade_date``, ``ts_code``, and OHLCV
            columns including ``close``, ``pct_chg``, ``amount``, ``vol``.
            May also contain fundamental and industry columns (wide-table
            format), in which case *fundamentals* and *industry_onehot*
            can be omitted.
        fundamentals
            Point-in-time fundamental data with ``trade_date``, ``ts_code``
            and value columns (roe, revenue_yoy, debt_ratio, bps, eps,
            gross_margin, etc.).  If ``None``, attempts to extract
            fundamental columns from *daily_bars*.
        industry_onehot
            Industry one-hot table with validity intervals (``ts_code``,
            ``in_date``, ``out_date``, ``L1_*`` columns).  If ``None``,
            attempts to use industry columns already present in
            *daily_bars* (``industry_*`` or ``L1_*``).
        """

        df = daily_bars.copy()
        df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)

        # If fundamentals not provided but columns exist in df, extract them
        if fundamentals is None:
            fundamentals = self._extract_fundamentals_from_df(df)

        # If industry_onehot not provided but industry columns exist in df,
        # use them directly (already point-in-time aligned)
        industry_in_df = self._has_industry_columns(df)
        if industry_onehot is None and industry_in_df:
            industry_exposures = self._extract_industry_from_df(df)
        else:
            industry_exposures = self._build_industry_exposures(df, industry_onehot)

        style_exposures = self._build_style_exposures(df, fundamentals)

        exposures = style_exposures.merge(
            industry_exposures, on=["trade_date", "ts_code"], how="left"
        )

        industry_cols = tuple(
            c for c in industry_exposures.columns if c not in ("trade_date", "ts_code")
        )

        exposures = exposures.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

        return ExposureBuildResult(
            exposures=exposures,
            style_factors=self.style_factors,
            industry_factors=industry_cols,
            start_date=int(exposures["trade_date"].min()),
            end_date=int(exposures["trade_date"].max()),
            n_stocks=exposures["ts_code"].nunique(),
            n_dates=exposures["trade_date"].nunique(),
        )

    # ------------------------------------------------------------------
    # Wide-table helpers
    # ------------------------------------------------------------------

    _FUNDAMENTAL_ALIASES: dict[str, tuple[str, ...]] = {
        "revenue_yoy": ("revenue_yoy", "or_yoy"),
        "debt_ratio": ("debt_ratio", "debt_to_assets"),
        "roe": ("roe",),
        "roa": ("roa",),
        "eps": ("eps",),
        "bps": ("bps",),
        "gross_margin": ("gross_margin",),
        "gross_margin_change": ("gross_margin_change",),
        "revenue_yoy_acceleration": ("revenue_yoy_acceleration",),
    }

    def _extract_fundamentals_from_df(self, df: pd.DataFrame) -> pd.DataFrame | None:
        """Extract fundamental columns from a wide-table DataFrame.

        Handles column name aliasing (e.g. ``or_yoy`` → ``revenue_yoy``).
        Returns ``None`` if no fundamental columns are found.
        """

        fund_cols = ["trade_date", "ts_code"]
        rename_map = {}

        for canonical, aliases in self._FUNDAMENTAL_ALIASES.items():
            for alias in aliases:
                if alias in df.columns:
                    fund_cols.append(alias)
                    if alias != canonical:
                        rename_map[alias] = canonical
                    break

        if len(fund_cols) <= 2:
            return None

        result = df[fund_cols].copy()
        if rename_map:
            result = result.rename(columns=rename_map)
        return result

    def _has_industry_columns(self, df: pd.DataFrame) -> bool:
        """Check if df contains industry one-hot columns."""

        for col in df.columns:
            if col.startswith("industry_") or col.startswith("L1_"):
                return True
        return False

    def _extract_industry_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract industry one-hot columns directly from wide-table df."""

        industry_cols = [
            c for c in df.columns
            if c.startswith("industry_") or c.startswith("L1_")
        ]

        # Normalize: rename industry_* to L1_* for consistency
        rename_map = {}
        for col in industry_cols:
            if col.startswith("industry_"):
                new_name = "L1_" + col[len("industry_"):]
                rename_map[col] = new_name

        result = df[["trade_date", "ts_code", *industry_cols]].copy()
        if rename_map:
            result = result.rename(columns=rename_map)

        # Apply benchmark industry drop
        if self.industry_cfg.get("drop_benchmark_industry", True):
            benchmark = self.industry_cfg.get("benchmark_industry", "银行")
            bench_col = f"L1_{benchmark}"
            if bench_col in result.columns:
                result = result.drop(columns=[bench_col])

        return result

    # ------------------------------------------------------------------
    # Style factors
    # ------------------------------------------------------------------

    def _build_style_exposures(
        self,
        df: pd.DataFrame,
        fundamentals: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Compute all 8 style factors and return long-format DataFrame."""

        result = df[["trade_date", "ts_code"]].copy()

        if "SIZE" in self.style_factors:
            result["SIZE"] = self._compute_size(df)

        if "BETA" in self.style_factors:
            result["BETA"] = self._compute_beta(df)

        if "MOMENTUM" in self.style_factors:
            result["MOMENTUM"] = self._compute_momentum(df)

        if "VALUE" in self.style_factors:
            result["VALUE"] = self._compute_value(df, fundamentals)

        if "GROWTH" in self.style_factors:
            result["GROWTH"] = self._compute_growth(df, fundamentals)

        if "VOLATILITY" in self.style_factors:
            result["VOLATILITY"] = self._compute_volatility(df)

        if "LIQUIDITY" in self.style_factors:
            result["LIQUIDITY"] = self._compute_liquidity(df)

        if "LEVERAGE" in self.style_factors:
            result["LEVERAGE"] = self._compute_leverage(df, fundamentals)

        style_cols = [f for f in self.style_factors if f in result.columns]
        result = self._cross_sectional_normalize(result, style_cols)

        return result

    def _compute_size(self, df: pd.DataFrame) -> pd.Series:
        """Size factor — log of 20-day median amount (proxy for market cap).

        Since the current pipeline does not include total_mv, we use
        20-day median daily amount as a size proxy.  This is highly
        correlated with market cap and sufficient for V1.
        """

        amount_col = "amount"
        if amount_col not in df.columns:
            return pd.Series(np.nan, index=df.index)

        amount_20 = (
            df.groupby("ts_code")[amount_col]
            .rolling(window=20, min_periods=5)
            .median()
            .reset_index(level=0, drop=True)
        )

        log_size = np.log(amount_20.replace(0, np.nan))
        return log_size

    def _compute_beta(self, df: pd.DataFrame) -> pd.Series:
        """Beta factor — 252-day rolling regression of stock return on market return.

        Market return is proxied by the equal-weighted cross-sectional mean
        return (no external benchmark data required for V1).
        """

        if "pct_chg" not in df.columns:
            return pd.Series(np.nan, index=df.index)

        lookback = int(self.beta_cfg.get("lookback", 252))
        min_obs = int(self.beta_cfg.get("min_observations", 120))

        market_ret = (
            df.groupby("trade_date")["pct_chg"].mean().rename("market_ret")
        )

        df_with_mkt = df.merge(market_ret, on="trade_date", how="left")
        df_with_mkt = df_with_mkt.sort_values(["ts_code", "trade_date"])

        def _rolling_beta(group: pd.DataFrame) -> pd.Series:
            x = group["market_ret"].values
            y = group["pct_chg"].values
            n = len(y)
            betas = np.full(n, np.nan)

            for i in range(min_obs - 1, n):
                start = max(0, i - lookback + 1)
                xs = x[start : i + 1]
                ys = y[start : i + 1]
                mask = ~(np.isnan(xs) | np.isnan(ys))
                if mask.sum() < min_obs:
                    continue
                xs_c = xs[mask]
                ys_c = ys[mask]
                var_x = np.var(xs_c, ddof=1)
                if var_x == 0 or np.isnan(var_x):
                    continue
                cov = np.cov(xs_c, ys_c, ddof=1)[0, 1]
                betas[i] = cov / var_x

            return pd.Series(betas, index=group.index)

        beta_series = (
            df_with_mkt.groupby("ts_code", group_keys=False)
            .apply(_rolling_beta, include_groups=False)
            .sort_index()
        )

        return beta_series

    def _compute_momentum(self, df: pd.DataFrame) -> pd.Series:
        """Momentum factor — 12-month return skipping the most recent month.

        Approximated with 252-day lookback minus 21-day skip using trading
        days (not calendar days).
        """

        if "close" not in df.columns:
            return pd.Series(np.nan, index=df.index)

        lookback = 252
        skip = 21

        close = df.groupby("ts_code")["close"]
        mom = (close.shift(skip) / close.shift(lookback)) - 1.0
        return mom.reset_index(level=0, drop=True)

    def _compute_value(
        self,
        df: pd.DataFrame,
        fundamentals: pd.DataFrame | None,
    ) -> pd.Series:
        """Value factor — composite of B/P, E/P (negative sign for value).

        V1 uses bps/close and eps/close (both positively signed for value
        — higher = cheaper = higher value exposure).
        """

        if fundamentals is None or "close" not in df.columns:
            return pd.Series(np.nan, index=df.index)

        fund_subset = fundamentals[["trade_date", "ts_code"]].copy()
        has_bps = "bps" in fundamentals.columns
        has_eps = "eps" in fundamentals.columns

        if not has_bps and not has_eps:
            return pd.Series(np.nan, index=df.index)

        if has_bps:
            fund_subset["bps"] = fundamentals["bps"]
        if has_eps:
            fund_subset["eps"] = fundamentals["eps"]

        merged = df[["trade_date", "ts_code", "close"]].merge(
            fund_subset, on=["trade_date", "ts_code"], how="left"
        )

        components = []
        if has_bps:
            bp = merged["bps"] / merged["close"].replace(0, np.nan)
            components.append(("bp", bp))
        if has_eps:
            ep = merged["eps"] / merged["close"].replace(0, np.nan)
            components.append(("ep", ep))

        temp = pd.DataFrame({"trade_date": merged["trade_date"]})
        for name, vals in components:
            temp[name] = vals.values

        for name, _ in components:
            temp[name] = temp.groupby("trade_date")[name].transform(_winsorize_series)
            temp[name] = temp.groupby("trade_date")[name].transform(_zscore_series)

        value_raw = temp[[n for n, _ in components]].mean(axis=1)
        return value_raw

    def _compute_growth(
        self,
        df: pd.DataFrame,
        fundamentals: pd.DataFrame | None,
    ) -> pd.Series:
        """Growth factor — composite of revenue_yoy, roe, gross_margin_change."""

        if fundamentals is None:
            return pd.Series(np.nan, index=df.index)

        avail = [
            c for c in ("revenue_yoy", "roe", "gross_margin_change")
            if c in fundamentals.columns
        ]
        if not avail:
            return pd.Series(np.nan, index=df.index)

        fund_subset = fundamentals[["trade_date", "ts_code", *avail]].copy()
        merged = df[["trade_date", "ts_code"]].merge(
            fund_subset, on=["trade_date", "ts_code"], how="left"
        )

        temp = pd.DataFrame({"trade_date": merged["trade_date"]})
        for col in avail:
            temp[col] = merged[col].values

        for col in avail:
            temp[col] = temp.groupby("trade_date")[col].transform(_winsorize_series)
            temp[col] = temp.groupby("trade_date")[col].transform(_zscore_series)

        growth_raw = temp[avail].mean(axis=1)
        return growth_raw

    def _compute_volatility(self, df: pd.DataFrame) -> pd.Series:
        """Volatility factor — composite of 20d and 60d return volatility."""

        if "pct_chg" not in df.columns:
            return pd.Series(np.nan, index=df.index)

        vol_20 = (
            df.groupby("ts_code")["pct_chg"]
            .rolling(window=20, min_periods=10)
            .std()
            .reset_index(level=0, drop=True)
        )
        vol_60 = (
            df.groupby("ts_code")["pct_chg"]
            .rolling(window=60, min_periods=30)
            .std()
            .reset_index(level=0, drop=True)
        )

        temp = pd.DataFrame({"trade_date": df["trade_date"], "vol_20": vol_20.values, "vol_60": vol_60.values})
        for col in ("vol_20", "vol_60"):
            temp[col] = temp.groupby("trade_date")[col].transform(_winsorize_series)
            temp[col] = temp.groupby("trade_date")[col].transform(_zscore_series)

        vol_raw = temp[["vol_20", "vol_60"]].mean(axis=1)
        return vol_raw

    def _compute_liquidity(self, df: pd.DataFrame) -> pd.Series:
        """Liquidity factor — Amihud illiquidity (20-day mean of |r|/amount).

        Higher Amihud = less liquid = higher LIQUIDITY exposure.
        """

        if "pct_chg" not in df.columns or "amount" not in df.columns:
            return pd.Series(np.nan, index=df.index)

        abs_ret = df["pct_chg"].abs()
        amount = df["amount"].replace(0, np.nan)
        daily_amihud = abs_ret / amount

        amihud_20 = (
            daily_amihud.groupby(df["ts_code"])
            .rolling(window=20, min_periods=5)
            .mean()
            .reset_index(level=0, drop=True)
        )

        log_amihud = np.log1p(amihud_20 * 1e6)
        return log_amihud

    def _compute_leverage(
        self,
        df: pd.DataFrame,
        fundamentals: pd.DataFrame | None,
    ) -> pd.Series:
        """Leverage factor — debt_to_assets ratio."""

        if fundamentals is None:
            return pd.Series(np.nan, index=df.index)

        lev_col = None
        for candidate in ("debt_ratio", "debt_to_assets"):
            if candidate in fundamentals.columns:
                lev_col = candidate
                break

        if lev_col is None:
            return pd.Series(np.nan, index=df.index)

        fund_subset = fundamentals[["trade_date", "ts_code", lev_col]].copy()
        merged = df[["trade_date", "ts_code"]].merge(
            fund_subset, on=["trade_date", "ts_code"], how="left"
        )

        return merged[lev_col]

    # ------------------------------------------------------------------
    # Industry factors
    # ------------------------------------------------------------------

    def _build_industry_exposures(
        self,
        df: pd.DataFrame,
        industry_onehot: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Build industry one-hot exposure columns.

        Uses N-1 industries (drops benchmark industry) to avoid perfect
        collinearity with the intercept in cross-sectional regression.
        """

        base = df[["trade_date", "ts_code"]].copy()

        if industry_onehot is None or not self.industry_cfg.get("enabled", True):
            return base

        l1_cols = [c for c in industry_onehot.columns if c.startswith("L1_")]
        if not l1_cols:
            return base

        io = industry_onehot.copy()
        if "in_date" in io.columns and "out_date" in io.columns:
            merged = _join_point_in_time_onehot(base, io, l1_cols)
        else:
            merged = base.merge(io, on="ts_code", how="left")
            for col in l1_cols:
                if col not in merged.columns:
                    merged[col] = 0
            merged = merged[["trade_date", "ts_code", *l1_cols]]

        drop_benchmark = self.industry_cfg.get("drop_benchmark_industry", True)
        if drop_benchmark:
            benchmark = self.industry_cfg.get("benchmark_industry", "银行")
            bench_col = f"L1_{benchmark}"
            if bench_col in merged.columns:
                merged = merged.drop(columns=[bench_col])

        return merged

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cross_sectional_normalize(
        self, df: pd.DataFrame, factor_cols: Sequence[str]
    ) -> pd.DataFrame:
        """Apply winsorization and z-score to each factor by trade_date."""

        result = df.copy()
        do_winsorize = self.cs_cfg.get("winsorize", True)
        do_zscore = self.cs_cfg.get("zscore", True)
        q = float(self.cs_cfg.get("winsorize_quantile", 0.01))

        for col in factor_cols:
            if col not in result.columns:
                continue
            if do_winsorize:
                result[col] = result.groupby("trade_date")[col].transform(
                    lambda x: _winsorize_series(x, q)
                )
            if do_zscore:
                result[col] = result.groupby("trade_date")[col].transform(_zscore_series)

        return result


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def _winsorize_series(x: pd.Series, quantile: float = 0.01) -> pd.Series:
    """Cross-sectional winsorization at symmetric quantiles."""

    if x.notna().sum() < 3:
        return x
    lower = x.quantile(quantile)
    upper = x.quantile(1 - quantile)
    return x.clip(lower, upper)


def _zscore_series(x: pd.Series) -> pd.Series:
    """Cross-sectional z-score standardization."""

    std = x.std()
    if pd.isna(std) or std == 0:
        return x - x.mean()
    return (x - x.mean()) / std


def _join_point_in_time_onehot(
    base: pd.DataFrame,
    onehot: pd.DataFrame,
    value_cols: list[str],
) -> pd.DataFrame:
    """Join point-in-time one-hot industry data onto daily observations.

    For each (ts_code, trade_date), finds the one-hot row where
    in_date <= trade_date <= out_date.
    """

    base = base.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    onehot = onehot.sort_values(["ts_code", "in_date"]).reset_index(drop=True)

    result_rows = []
    for code, group in base.groupby("ts_code"):
        code_oh = onehot[onehot["ts_code"] == code]
        if code_oh.empty:
            continue
        for _, row in group.iterrows():
            td = row["trade_date"]
            match = code_oh[
                (code_oh["in_date"] <= td) & (code_oh["out_date"] >= td)
            ]
            if not match.empty:
                r = {"trade_date": td, "ts_code": code}
                for col in value_cols:
                    r[col] = match.iloc[0][col]
                result_rows.append(r)

    if not result_rows:
        result = base[["trade_date", "ts_code"]].copy()
        for col in value_cols:
            result[col] = np.nan
        return result

    result = pd.DataFrame(result_rows)
    return result
