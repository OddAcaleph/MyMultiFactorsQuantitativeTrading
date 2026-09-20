"""Risk attribution computation.

Decomposes portfolio risk into:
  - Factor risk (from factor covariance)
  - Specific (idiosyncratic) risk
  - Per-factor risk contribution percentages
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RiskAttributionResult:
    """Result of risk attribution.

    Attributes
    ----------
    total_variance : float
        Total portfolio variance (w^T Σ w).
    factor_variance : float
        Variance from factor risk (z^T F z).
    specific_variance : float
        Variance from specific/idiosyncratic risk (Σ w_i^2 D_i).
    factor_contribution_pct : pd.Series
        Percentage of total risk contributed by each factor.
        Index = factor names.
    specific_contribution_pct : float
        Percentage of total risk from specific risk.
    """

    total_variance: float
    factor_variance: float
    specific_variance: float
    factor_contribution_pct: pd.Series
    specific_contribution_pct: float


class RiskAttribution:
    """Compute risk attribution for a portfolio.

    Uses the Barra risk model decomposition:
        σ_p² = z^T F z + Σ(w_i² D_i)
    where z = X^T w is the factor exposure vector.

    Factor risk contribution for factor k:
        RC_k = z_k * (F z)_k / σ_p²
    """

    def compute(
        self,
        weights: np.ndarray,
        exposures: np.ndarray,
        factor_cov: np.ndarray,
        specific_var: np.ndarray,
        factor_names: list[str],
    ) -> RiskAttributionResult:
        """Compute full risk attribution.

        Parameters
        ----------
        weights : np.ndarray (N,)
            Portfolio weights.
        exposures : np.ndarray (N, K)
            Factor exposure matrix.
        factor_cov : np.ndarray (K, K)
            Factor covariance matrix.
        specific_var : np.ndarray (N,)
            Specific (idiosyncratic) variance per stock.
        factor_names : list[str]
            Names of the K factors.

        Returns
        -------
        RiskAttributionResult
        """
        N = len(weights)
        K = len(factor_names)

        # Factor exposure: z = X^T w  (K,)
        z = exposures.T @ weights  # (K,)

        # Factor variance: z^T F z
        Fz = factor_cov @ z  # (K,)
        factor_var = float(z @ Fz)

        # Specific variance: sum(w_i^2 * D_i)
        specific_var_total = float(np.sum(weights ** 2 * specific_var))

        # Total variance
        total_var = factor_var + specific_var_total

        if total_var <= 0:
            zero_contrib = pd.Series(np.zeros(K), index=factor_names)
            return RiskAttributionResult(
                total_variance=0.0,
                factor_variance=0.0,
                specific_variance=0.0,
                factor_contribution_pct=zero_contrib,
                specific_contribution_pct=0.0,
            )

        # Per-factor risk contribution: z_k * (Fz)_k / total_var
        factor_rc = z * Fz / total_var
        factor_rc_series = pd.Series(factor_rc, index=factor_names)

        # Specific risk contribution
        specific_rc = specific_var_total / total_var

        return RiskAttributionResult(
            total_variance=float(total_var),
            factor_variance=float(factor_var),
            specific_variance=float(specific_var_total),
            factor_contribution_pct=factor_rc_series,
            specific_contribution_pct=float(specific_rc),
        )

    @staticmethod
    def check_attribution_sum(
        result: RiskAttributionResult,
        tolerance: float = 0.001,
    ) -> bool:
        """Verify that factor + specific contributions sum to ~100%.

        Parameters
        ----------
        result : RiskAttributionResult
        tolerance : float
            Maximum allowed absolute deviation from 1.0.

        Returns
        -------
        bool
            True if sum is within tolerance.
        """
        total_pct = result.factor_contribution_pct.sum() + result.specific_contribution_pct
        return abs(total_pct - 1.0) < tolerance
