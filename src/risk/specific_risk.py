"""Specific (idiosyncratic) risk estimation.

Computes stock-level specific volatility from the residuals of the daily
factor return cross-sectional regression.  Uses a rolling window of
residual returns, with shrinkage toward industry-size group medians for
stocks with insufficient history.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpecificRiskResult:
    """Result of specific risk estimation."""

    specific_risk: pd.DataFrame
    start_date: int
    end_date: int


class SpecificRiskEstimator:
    """Estimate stock-specific risk from residual returns.

    Parameters
    ----------
    config
        Risk model config dictionary.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        sr_cfg = config.get("specific_risk", {})
        self.lookback = int(sr_cfg.get("lookback", 252))
        self.min_obs = int(sr_cfg.get("min_observations", 120))
        self.method = sr_cfg.get("method", "rolling_std")
        self.shrinkage_target = sr_cfg.get("shrinkage_target", "industry_size_median")

    def estimate(
        self,
        residual_returns: pd.DataFrame,
        exposures: pd.DataFrame | None = None,
    ) -> SpecificRiskResult:
        """Estimate specific risk from residual returns.

        Parameters
        ----------
        residual_returns
            DataFrame with columns ``trade_date``, ``ts_code``, ``residual``.
        exposures
            Optional exposure DataFrame with industry columns, used for
            small-sample shrinkage grouping.
        """

        df = residual_returns.sort_values(["ts_code", "trade_date"]).reset_index(drop=True).copy()

        df["residual_sq"] = df["residual"] ** 2

        spec_var = (
            df.groupby("ts_code")["residual_sq"]
            .rolling(window=self.lookback, min_periods=self.min_obs)
            .mean()
            .reset_index(level=0, drop=True)
        )

        sample_count = (
            df.groupby("ts_code")["residual"]
            .rolling(window=self.lookback, min_periods=1)
            .count()
            .reset_index(level=0, drop=True)
        )

        result = df[["trade_date", "ts_code"]].copy()
        result["specific_variance"] = spec_var.values
        result["specific_vol"] = np.sqrt(np.maximum(result["specific_variance"], 0))
        result["sample_count"] = sample_count.values.astype(int)

        result = self._apply_shrinkage(result, exposures)

        valid = result["specific_vol"].notna()
        if valid.any():
            start_date = int(result.loc[valid, "trade_date"].min())
            end_date = int(result.loc[valid, "trade_date"].max())
        else:
            start_date = 0
            end_date = 0

        return SpecificRiskResult(
            specific_risk=result,
            start_date=start_date,
            end_date=end_date,
        )

    def _apply_shrinkage(
        self,
        result: pd.DataFrame,
        exposures: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """Apply small-sample shrinkage toward group median.

        Stocks with fewer than ``min_obs`` residual observations get their
        specific volatility shrunk toward their peer group median.
        """

        if exposures is None or self.shrinkage_target is None:
            return result

        low_sample = result["sample_count"] < self.min_obs
        if not low_sample.any():
            return result

        group_col = self._get_group_column(exposures)

        if group_col is None:
            return result

        exp_subset = exposures[["trade_date", "ts_code", group_col]].copy()
        merged = result.merge(exp_subset, on=["trade_date", "ts_code"], how="left")

        daily_group_median = (
            merged[~low_sample]
            .groupby(["trade_date", group_col])["specific_vol"]
            .median()
            .reset_index()
            .rename(columns={"specific_vol": "group_median_vol"})
        )

        merged = merged.merge(daily_group_median, on=["trade_date", group_col], how="left")

        shrinkage_strength = 1.0 - (merged["sample_count"] / self.min_obs).clip(0, 1)
        shrunk_vol = (
            (1 - shrinkage_strength) * merged["specific_vol"]
            + shrinkage_strength * merged["group_median_vol"]
        )

        shrunk_var = shrunk_vol ** 2

        result = result.copy()
        need_shrink = low_sample & merged["group_median_vol"].notna()
        result.loc[need_shrink, "specific_vol"] = shrunk_vol[need_shrink].values
        result.loc[need_shrink, "specific_variance"] = shrunk_var[need_shrink].values
        result["shrunk"] = need_shrink.values

        return result

    def _get_group_column(self, exposures: pd.DataFrame) -> str | None:
        """Find a suitable grouping column for shrinkage."""

        industry_cols = [c for c in exposures.columns if c.startswith("L1_") or c.startswith("industry_")]
        if not industry_cols:
            return None

        if "SIZE" in exposures.columns:
            size_col = "SIZE"
        else:
            size_col = None

        return industry_cols[0]
