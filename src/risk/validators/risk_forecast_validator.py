"""Risk forecast validator.

Compares predicted portfolio risk with realized (future) risk to assess
the forecasting power of the risk model.

Key metrics:
    - Predicted vs realized volatility
    - Volatility bias
    - RMSE
    - Pearson correlation
    - Spearman rank correlation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class RiskForecastValidationResult:
    """Result of risk forecast validation."""

    n_observations: int = 0
    mean_predicted_vol: float = 0.0
    mean_realized_vol: float = 0.0
    vol_bias: float = 0.0
    vol_bias_pct: float = 0.0
    rmse: float = 0.0
    pearson_correlation: float = 0.0
    spearman_correlation: float = 0.0
    passed: bool = True
    issues: list[str] = field(default_factory=list)
    details: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_observations": self.n_observations,
            "mean_predicted_vol": self.mean_predicted_vol,
            "mean_realized_vol": self.mean_realized_vol,
            "vol_bias": self.vol_bias,
            "vol_bias_pct": self.vol_bias_pct,
            "rmse": self.rmse,
            "pearson_correlation": self.pearson_correlation,
            "spearman_correlation": self.spearman_correlation,
            "passed": self.passed,
            "issues": self.issues,
        }


class RiskForecastValidator:
    """Validate risk model forecasts against realized volatility.

    Parameters
    ----------
    config
        Risk model config dictionary.
    horizon
        Forecast horizon in trading days (default: 20).
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        horizon: int = 20,
    ) -> None:
        self.config = dict(config or {})
        self.horizon = int(horizon)

    def validate_portfolio(
        self,
        predicted_vol: pd.Series,
        realized_vol: pd.Series,
    ) -> RiskForecastValidationResult:
        """Validate portfolio-level risk forecasts.

        Parameters
        ----------
        predicted_vol
            Series of predicted volatilities indexed by trade_date.
        realized_vol
            Series of realized (forward-looking) volatilities indexed by
            trade_date.
        """

        result = RiskForecastValidationResult()

        aligned = pd.DataFrame({
            "predicted": predicted_vol,
            "realized": realized_vol,
        }).dropna()

        result.n_observations = len(aligned)
        result.details = aligned

        if len(aligned) < 10:
            result.passed = False
            result.issues.append(f"Insufficient observations: {len(aligned)} < 10")
            return result

        pred = aligned["predicted"].values
        real = aligned["realized"].values

        result.mean_predicted_vol = float(np.mean(pred))
        result.mean_realized_vol = float(np.mean(real))
        result.vol_bias = float(np.mean(pred - real))
        result.vol_bias_pct = float(result.vol_bias / np.mean(real)) if np.mean(real) != 0 else float("nan")
        result.rmse = float(np.sqrt(np.mean((pred - real) ** 2)))

        if np.std(pred) > 0 and np.std(real) > 0:
            result.pearson_correlation = float(stats.pearsonr(pred, real)[0])
            result.spearman_correlation = float(stats.spearmanr(pred, real)[0])
        else:
            result.pearson_correlation = float("nan")
            result.spearman_correlation = float("nan")

        if result.pearson_correlation <= 0:
            result.passed = False
            result.issues.append("Predicted vol has non-positive correlation with realized vol")

        if abs(result.vol_bias_pct) > 0.5:
            result.issues.append(f"Large volatility bias: {result.vol_bias_pct:.1%}")

        return result

    def validate_stock_level(
        self,
        predicted_vol: pd.DataFrame,
        realized_vol: pd.DataFrame,
    ) -> RiskForecastValidationResult:
        """Validate stock-level risk forecasts (cross-sectional ranking).

        Parameters
        ----------
        predicted_vol
            DataFrame with ``trade_date``, ``ts_code``, ``predicted_vol``.
        realized_vol
            DataFrame with ``trade_date``, ``ts_code``, ``realized_vol``.
        """

        merged = predicted_vol.merge(realized_vol, on=["trade_date", "ts_code"], how="inner")
        merged = merged.dropna()

        result = RiskForecastValidationResult()
        result.n_observations = len(merged)

        if len(merged) < 100:
            result.passed = False
            result.issues.append(f"Insufficient observations: {len(merged)} < 100")
            return result

        pred = merged["predicted_vol"].values
        real = merged["realized_vol"].values

        result.mean_predicted_vol = float(np.mean(pred))
        result.mean_realized_vol = float(np.mean(real))
        result.rmse = float(np.sqrt(np.mean((pred - real) ** 2)))

        if np.std(pred) > 0 and np.std(real) > 0:
            result.pearson_correlation = float(stats.pearsonr(pred, real)[0])
            result.spearman_correlation = float(stats.spearmanr(pred, real)[0])

        if result.spearman_correlation <= 0:
            result.passed = False
            result.issues.append("Stock-level rank correlation is non-positive")

        return result

    def compute_realized_vol(
        self,
        returns: pd.DataFrame,
        horizon: int | None = None,
    ) -> pd.DataFrame:
        """Compute forward-looking realized volatility from returns.

        Parameters
        ----------
        returns
            DataFrame with ``trade_date``, ``ts_code``, ``ret`` columns.
        horizon
            Forward-looking window in trading days.  Defaults to
            ``self.horizon``.

        Returns
        -------
        pd.DataFrame
            Columns: ``trade_date``, ``ts_code``, ``realized_vol``.
        """

        h = horizon or self.horizon
        df = returns.sort_values(["ts_code", "trade_date"]).copy()

        df["ret_sq"] = df["ret"] ** 2

        realized_var = (
            df.groupby("ts_code")["ret_sq"]
            .rolling(window=h, min_periods=max(h // 2, 5))
            .mean()
            .reset_index(level=0, drop=True)
        )

        result = df[["trade_date", "ts_code"]].copy()
        result["realized_vol"] = np.sqrt(np.maximum(realized_var.values, 0))
        result["realized_vol"] = result.groupby("ts_code")["realized_vol"].shift(-h + 1)

        return result
