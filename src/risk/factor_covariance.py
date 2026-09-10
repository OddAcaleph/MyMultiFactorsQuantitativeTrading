"""Factor covariance matrix estimation.

V1 implements Ledoit-Wolf shrinkage covariance estimation on a rolling
window of factor returns.  Produces a PSD factor covariance matrix ``F_t``
for each trading day.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FactorCovarianceResult:
    """Result of factor covariance estimation."""

    dates: list[int]
    factor_names: list[str]
    covariance_matrices: dict[int, np.ndarray]
    shrinkage_intensities: dict[int, float]
    start_date: int
    end_date: int


class FactorCovarianceEstimator:
    """Estimate rolling factor covariance matrices.

    Parameters
    ----------
    config
        Risk model config dictionary.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        fc_cfg = config.get("factor_covariance", {})
        self.method = fc_cfg.get("method", "ledoit_wolf")
        self.lookback = int(fc_cfg.get("lookback", 252))
        self.min_obs = int(fc_cfg.get("min_observations", 60))

    def estimate(self, factor_returns: pd.DataFrame) -> FactorCovarianceResult:
        """Estimate rolling factor covariance matrices.

        Parameters
        ----------
        factor_returns
            DataFrame with ``trade_date`` column and one column per factor
            (including INTERCEPT).  The INTERCEPT column is excluded from
            covariance computation.
        """

        fr = factor_returns.sort_values("trade_date").reset_index(drop=True).copy()
        has_intercept = "INTERCEPT" in fr.columns
        all_factor_cols = [c for c in fr.columns if c != "trade_date"]
        dates = fr["trade_date"].tolist()

        # Keep only factor columns with sufficient non-NaN data overall
        valid_counts = fr[all_factor_cols].notna().sum()
        factor_cols = [
            c for c in all_factor_cols
            if valid_counts[c] >= self.min_obs
        ]

        # Always keep INTERCEPT if present (market factor)
        if has_intercept and "INTERCEPT" not in factor_cols and "INTERCEPT" in fr.columns:
            if fr["INTERCEPT"].notna().sum() >= self.min_obs:
                factor_cols = ["INTERCEPT"] + factor_cols

        if not factor_cols:
            return FactorCovarianceResult(
                dates=[],
                factor_names=all_factor_cols,
                covariance_matrices={},
                shrinkage_intensities={},
                start_date=0,
                end_date=0,
            )

        fr_values = fr[factor_cols].values

        n_dates = len(dates)
        cov_matrices: dict[int, np.ndarray] = {}
        shrinkage: dict[int, float] = {}

        for i in range(n_dates):
            if i < self.min_obs - 1:
                continue

            start = max(0, i - self.lookback + 1)
            window = fr_values[start : i + 1, :]

            valid_mask = ~np.isnan(window).any(axis=1)
            window_clean = window[valid_mask]

            if window_clean.shape[0] < self.min_obs:
                continue

            if self.method == "ledoit_wolf":
                cov, delta = _ledoit_wolf_shrinkage(window_clean)
            else:
                cov = np.cov(window_clean, rowvar=False, ddof=1)
                delta = 0.0

            cov_matrices[int(dates[i])] = cov
            shrinkage[int(dates[i])] = float(delta)

        valid_dates = sorted(cov_matrices.keys())

        return FactorCovarianceResult(
            dates=valid_dates,
            factor_names=factor_cols,
            covariance_matrices=cov_matrices,
            shrinkage_intensities=shrinkage,
            start_date=valid_dates[0] if valid_dates else 0,
            end_date=valid_dates[-1] if valid_dates else 0,
        )


def _ledoit_wolf_shrinkage(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage covariance estimator.

    Shrinks the sample covariance toward a constant-correlation (or
    scaled-identity) target.  V1 uses the simpler scaled-identity target
    (shrink toward mu * I) which is numerically stable and sufficient.

    Parameters
    ----------
    X
        ``T x N`` matrix of observations (T samples, N variables).

    Returns
    -------
    tuple
        (shrunk_covariance, shrinkage_intensity)
    """

    T, N = X.shape
    if T < 2 or N == 0:
        raise ValueError("Insufficient data for covariance estimation")

    sample_cov = np.cov(X, rowvar=False, ddof=1)

    mean_var = np.mean(np.diag(sample_cov))
    target = mean_var * np.eye(N)

    X_centered = X - X.mean(axis=0, keepdims=True)

    # Estimate pi (sum of asymptotic variances of sample cov entries)
    # Using the Ledoit-Wolf (2004) formula for scaled-identity target
    X_sq = X_centered ** 2
    pi_mat = (X_sq.T @ X_sq) / T - sample_cov ** 2
    pi_hat = np.sum(pi_mat)

    # Estimate rho (sum of covariances between sample and target)
    rho_hat = 0.0
    for i in range(N):
        rho_hat += np.mean(X_sq[:, i] * mean_var) - mean_var * sample_cov[i, i]

    # Compute shrinkage intensity
    gamma_hat = np.sum((sample_cov - target) ** 2)
    if gamma_hat == 0:
        delta = 0.0
    else:
        delta = max(0.0, min(1.0, (pi_hat - rho_hat) / (T * gamma_hat)))

    shrunk = (1 - delta) * sample_cov + delta * target

    return shrunk, delta
