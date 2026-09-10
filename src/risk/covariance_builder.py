"""Stock-level covariance matrix builder.

Assembles the full stock covariance matrix Sigma from factor exposures,
factor covariance, and specific risk:

    Sigma = X @ F @ X.T + D

where D is diagonal (specific variances).  Includes PSD enforcement and
numerical stability checks.

For memory-efficient portfolio volatility computation without materializing
the full NxN matrix, use :meth:`compute_portfolio_vol`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CovarianceBuildResult:
    """Result of stock covariance matrix building."""

    dates: list[int]
    stock_codes: dict[int, list[str]]
    covariance_matrices: dict[int, np.ndarray]
    min_eigenvalues: dict[int, float]
    condition_numbers: dict[int, float]
    start_date: int
    end_date: int


class CovarianceBuilder:
    """Build stock-level covariance matrices.

    Parameters
    ----------
    config
        Risk model config dictionary.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    def build(
        self,
        exposures: pd.DataFrame,
        factor_covariance: np.ndarray,
        specific_variances: pd.Series | pd.DataFrame,
    ) -> tuple[np.ndarray, list[str]]:
        """Build a single stock covariance matrix for one date.

        Parameters
        ----------
        exposures
            DataFrame with one row per stock and factor exposure columns.
            Must include a ``ts_code`` column.
        factor_covariance
            ``K x K`` factor covariance matrix (matching factor column order
            in *exposures* excluding ``ts_code``).
        specific_variances
            Diagonal elements of specific risk covariance matrix.  Either a
            Series indexed by ``ts_code`` or a DataFrame with ``ts_code``
            and ``specific_variance`` columns.

        Returns
        -------
        tuple
            (N x N covariance matrix, list of stock codes in row/col order)
        """

        exp = exposures.copy()
        factor_cols = [c for c in exp.columns if c not in ("ts_code", "trade_date")]

        if isinstance(specific_variances, pd.DataFrame):
            sv = specific_variances.set_index("ts_code")["specific_variance"]
        else:
            sv = specific_variances

        common_stocks = exp["ts_code"].isin(sv.index)
        exp = exp[common_stocks].reset_index(drop=True)

        X = exp[factor_cols].values.astype(float)
        X = np.nan_to_num(X, nan=0.0)

        F = np.asarray(factor_covariance, dtype=float)

        stock_codes = exp["ts_code"].tolist()
        D_diag = sv.loc[stock_codes].values.astype(float)
        D_diag = np.nan_to_num(D_diag, nan=0.0)
        D_diag = np.maximum(D_diag, 0.0)

        systematic = X @ F @ X.T
        D = np.diag(D_diag)
        Sigma = systematic + D

        Sigma = self._ensure_psd(Sigma)

        return Sigma, stock_codes

    @staticmethod
    def compute_portfolio_vol(
        weights: np.ndarray,
        X: np.ndarray,
        F: np.ndarray,
        D_diag: np.ndarray,
    ) -> float:
        """Compute portfolio volatility without materializing NxN Sigma.

        Uses the identity:

            sigma_p^2 = w^T X F X^T w + w^T D w
                     = (X^T w)^T F (X^T w) + sum(w_i^2 * D_ii)

        This is O(K^2 + N) instead of O(N^2), which is critical for large
        stock universes (N ~ 5000).

        Parameters
        ----------
        weights
            ``(N,)`` portfolio weight vector.
        X
            ``(N, K)`` factor exposure matrix.
        F
            ``(K, K)`` factor covariance matrix.
        D_diag
            ``(N,)`` specific variance vector (diagonal of D).

        Returns
        -------
        float
            Portfolio volatility (sqrt of portfolio variance).
        """

        Xw = X.T @ weights  # (K,)
        systematic_var = Xw @ F @ Xw  # scalar
        specific_var = np.sum(weights ** 2 * D_diag)  # scalar
        port_var = systematic_var + specific_var
        return float(np.sqrt(max(port_var, 0.0)))

    def build_many(
        self,
        exposures: pd.DataFrame,
        factor_covariances: dict[int, np.ndarray],
        specific_risk: pd.DataFrame,
        factor_names: list[str] | None = None,
        save_dir: str | Path | None = None,
        keep_in_memory: bool = True,
    ) -> CovarianceBuildResult:
        """Build covariance matrices for multiple dates.

        Parameters
        ----------
        exposures
            Long-format exposure DataFrame with ``trade_date``, ``ts_code``,
            and factor columns.
        factor_covariances
            Mapping from trade_date (int) to factor covariance matrix.
        specific_risk
            DataFrame with ``trade_date``, ``ts_code``, ``specific_variance``.
        factor_names
            Names of factors in the covariance matrix (column order).  If
            ``None``, assumes all non-key exposure columns match.
        save_dir
            If provided, save each daily covariance matrix to this directory
            as ``YYYYMMDD.npy`` plus a ``stock_codes_YYYYMMDD.npy`` file.
        keep_in_memory
            If ``False`` and *save_dir* is set, don't keep matrices in the
            result dict (saves memory for large runs).  Default ``True``.
        """

        all_factor_cols = [
            c for c in exposures.columns
            if c not in ("trade_date", "ts_code")
        ]

        if factor_names is None:
            factor_names = all_factor_cols

        # Select only the exposure columns that match factor_names
        avail_factors = [f for f in factor_names if f in all_factor_cols]
        # Always include INTERCEPT from factor_names (market factor, exposure = 1)
        if factor_names and "INTERCEPT" in factor_names and "INTERCEPT" not in avail_factors:
            avail_factors = ["INTERCEPT"] + avail_factors
        if not avail_factors:
            return CovarianceBuildResult(
                dates=[], stock_codes={}, covariance_matrices={},
                min_eigenvalues={}, condition_numbers={},
                start_date=0, end_date=0,
            )

        dates = sorted(set(factor_covariances.keys()) & set(specific_risk["trade_date"].unique()))
        dates = [int(d) for d in dates]

        save_path = Path(save_dir) if save_dir else None
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)

        cov_matrices: dict[int, np.ndarray] = {}
        stock_codes_map: dict[int, list[str]] = {}
        min_eigs: dict[int, float] = {}
        cond_nums: dict[int, float] = {}

        for i, td in enumerate(dates):
            exp_day = exposures[exposures["trade_date"] == td].copy()
            sr_day = specific_risk[specific_risk["trade_date"] == td].copy()

            if exp_day.empty or sr_day.empty:
                continue

            common = exp_day["ts_code"].isin(sr_day["ts_code"])
            exp_day = exp_day[common].reset_index(drop=True)

            if len(exp_day) < 2:
                continue

            F = factor_covariances[td]
            if F.shape[0] != len(avail_factors):
                logger.warning(
                    "Factor count mismatch on %s: exp has %d active, F has %d",
                    td, len(avail_factors), F.shape[0],
                )
                continue

            # Subset exposures to only factors present in F
            exp_subset = exp_day[["ts_code"] + [f for f in avail_factors if f != "INTERCEPT"]].copy()
            if "INTERCEPT" in avail_factors:
                exp_subset["INTERCEPT"] = 1.0
            Sigma, codes = self.build(exp_subset, F, sr_day)

            if save_path:
                np.save(save_path / f"{td}.npy", Sigma)
                np.save(save_path / f"codes_{td}.npy", np.array(codes, dtype=object))

            if keep_in_memory or not save_path:
                cov_matrices[td] = Sigma
                stock_codes_map[td] = codes
            else:
                # Still track stock codes for lookup
                stock_codes_map[td] = codes

            # Compute diagnostics (cheap relative to building Sigma)
            eigvals = np.linalg.eigvalsh(Sigma)
            min_eigs[td] = float(eigvals.min())
            cond_nums[td] = float(eigvals.max() / max(eigvals.min(), 1e-12))

            if (i + 1) % 20 == 0:
                logger.info(
                    "Built covariance for %d / %d dates (current: %s, N=%d)",
                    i + 1, len(dates), td, len(codes),
                )

        valid_dates = sorted(stock_codes_map.keys())

        return CovarianceBuildResult(
            dates=valid_dates,
            stock_codes=stock_codes_map,
            covariance_matrices=cov_matrices,
            min_eigenvalues=min_eigs,
            condition_numbers=cond_nums,
            start_date=valid_dates[0] if valid_dates else 0,
            end_date=valid_dates[-1] if valid_dates else 0,
        )

    def _ensure_psd(self, Sigma: np.ndarray, tol: float = 1e-8) -> np.ndarray:
        """Ensure matrix is symmetric positive semi-definite.

        1. Symmetrize: (Sigma + Sigma.T) / 2
        2. Eigenvalue clip: set negative eigenvalues to *tol*
        """

        Sigma = (Sigma + Sigma.T) / 2.0

        eigvals, eigvecs = np.linalg.eigh(Sigma)
        if eigvals.min() >= -tol:
            return Sigma

        eigvals_clipped = np.maximum(eigvals, tol)
        Sigma_psd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
        Sigma_psd = (Sigma_psd + Sigma_psd.T) / 2.0

        return Sigma_psd
