"""QP constraint builder.

Builds all constraints for portfolio optimization:
  - sum(w) = 1  (fully invested)
  - 0 <= w_i <= w_max  (long-only + single-stock cap)
  - t_i >= w_i - w_prev_i  (turnover auxiliary)
  - t_i >= w_prev_i - w_i  (turnover auxiliary)
  - t_i >= 0  (turnover non-negative)
  - industry weight caps  (Phase 2)
  - style factor exposure bounds  (Phase 2)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass
class QPConstraints:
    """QP constraints in standard form.

    G x <= h  (linear inequality)
    A x == b  (linear equality)
    lb <= x <= ub  (box)
    soc_constraints: list of (P, q, r) for ||P x + q||_2 <= r  (SOCP)
    """

    G: np.ndarray
    h: np.ndarray
    A: np.ndarray
    b: np.ndarray
    lb: np.ndarray
    ub: np.ndarray
    soc_constraints: list | None = None


class ConstraintBuilder:
    """Build QP constraints for portfolio optimization.

    Supports:
    - Basic: sum=1, long-only, weight cap, turnover auxiliary
    - Industry: max weight per industry
    - Style: upper/lower bounds on style factor exposures
    - Volatility: max portfolio volatility (quadratic constraint)
    """

    def build(
        self,
        n_stocks: int,
        prev_weights: np.ndarray,
        exposures: np.ndarray | None = None,
        style_factor_idx: list[int] | None = None,
        industry_factor_idx: list[int] | None = None,
        factor_names: list[str] | None = None,
        factor_cov: np.ndarray | None = None,
        specific_var: np.ndarray | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> QPConstraints:
        """Build constraints for extended variable x = [w; t].

        Parameters
        ----------
        n_stocks : int
            Number of stocks (N).
        prev_weights : np.ndarray (N,)
            Previous period weights.
        exposures : np.ndarray (N, K), optional
            Factor exposure matrix. Required for industry/style/volatility constraints.
        style_factor_idx : list[int], optional
            Indices of style factors in exposure columns.
        industry_factor_idx : list[int], optional
            Indices of industry factors in exposure columns.
        factor_names : list[str], optional
            Names of all K factors in exposure columns, used to map
            style constraint keys (e.g. "SIZE") to column indices.
        factor_cov : np.ndarray (K, K), optional
            Factor covariance matrix. Required for volatility constraint.
        specific_var : np.ndarray (N,), optional
            Specific (idiosyncratic) variance. Required for volatility constraint.
        config : dict, optional
            Constraint configuration. Keys:
            - long_only : bool (default True)
            - fully_invested : bool (default True)
            - max_weight : float (default 0.03)
            - industry : dict with "max_weight" per industry
            - style : dict with factor_name -> [lower, upper]
            - max_vol : float, annualized volatility target (0 = no constraint)

        Returns
        -------
        QPConstraints
        """
        config = config or {}
        long_only = bool(config.get("long_only", True))
        fully_invested = bool(config.get("fully_invested", True))
        max_weight = float(config.get("max_weight", 0.03))
        industry_cfg = config.get("industry", {})
        style_cfg = config.get("style", {})

        N = n_stocks
        n_vars = 2 * N

        # ---- Box constraints: lb <= x <= ub ----
        lb = np.full(n_vars, -np.inf, dtype=np.float64)
        ub = np.full(n_vars, np.inf, dtype=np.float64)

        # w bounds
        if long_only:
            lb[:N] = 0.0
        ub[:N] = max_weight

        # t bounds: t >= 0
        lb[N:] = 0.0

        # ---- Equality constraints: A x = b ----
        eq_rows = []
        eq_rhs = []

        if fully_invested:
            row = np.zeros(n_vars, dtype=np.float64)
            row[:N] = 1.0
            eq_rows.append(row)
            eq_rhs.append(1.0)

        if eq_rows:
            A = np.vstack(eq_rows)
            b = np.array(eq_rhs, dtype=np.float64)
        else:
            A = np.zeros((0, n_vars), dtype=np.float64)
            b = np.zeros(0, dtype=np.float64)

        # ---- Inequality constraints: G x <= h ----
        ineq_rows = []
        ineq_rhs = []

        # Turnover auxiliary: t_i >= w_i - w_prev_i
        for i in range(N):
            row = np.zeros(n_vars, dtype=np.float64)
            row[i] = 1.0
            row[N + i] = -1.0
            ineq_rows.append(row)
            ineq_rhs.append(prev_weights[i])

        # Turnover auxiliary: t_i >= w_prev_i - w_i
        for i in range(N):
            row = np.zeros(n_vars, dtype=np.float64)
            row[i] = -1.0
            row[N + i] = -1.0
            ineq_rows.append(row)
            ineq_rhs.append(-prev_weights[i])

        # ---- Industry constraints (Phase 2) ----
        if industry_cfg and exposures is not None and industry_factor_idx:
            max_ind_weight = float(industry_cfg.get("max_weight", 0.20))
            X_industry = exposures[:, industry_factor_idx]  # (N, n_industries)
            for j in range(X_industry.shape[1]):
                row = np.zeros(n_vars, dtype=np.float64)
                row[:N] = X_industry[:, j]
                ineq_rows.append(row)
                ineq_rhs.append(max_ind_weight)

        # ---- Style constraints (Phase 2) ----
        if style_cfg and exposures is not None and style_factor_idx:
            # Build name -> column index map
            name_to_idx: dict[str, int] = {}
            if factor_names:
                for i, name in enumerate(factor_names):
                    name_to_idx[name] = i

            for factor_name, bounds in style_cfg.items():
                # Try to find column index by name first
                col_idx = None
                if factor_name in name_to_idx:
                    col_idx = name_to_idx[factor_name]
                elif isinstance(factor_name, int):
                    col_idx = factor_name
                elif isinstance(factor_name, str) and factor_name.isdigit():
                    col_idx = int(factor_name)

                if col_idx is None:
                    continue

                lower, upper = bounds
                exposure_col = exposures[:, col_idx]  # (N,)

                # Upper bound: exposure^T w <= upper
                row = np.zeros(n_vars, dtype=np.float64)
                row[:N] = exposure_col
                ineq_rows.append(row)
                ineq_rhs.append(upper)

                # Lower bound: exposure^T w >= lower  →  -exposure^T w <= -lower
                row = np.zeros(n_vars, dtype=np.float64)
                row[:N] = -exposure_col
                ineq_rows.append(row)
                ineq_rhs.append(-lower)

        if ineq_rows:
            G = np.vstack(ineq_rows)
            h = np.array(ineq_rhs, dtype=np.float64)
        else:
            G = np.zeros((0, n_vars), dtype=np.float64)
            h = np.zeros(0, dtype=np.float64)

        # ---- Volatility constraint (quadratic) ----
        # w^T Σ w <= σ_target^2  (daily variance)
        soc_constraints = None
        max_vol_annual = float(config.get("max_vol", 0.0))
        if max_vol_annual > 0 and exposures is not None and factor_cov is not None and specific_var is not None:
            # Convert annualized vol target to daily variance
            # σ_daily = σ_annual / sqrt(252)
            # var_daily = σ_annual² / 252
            target_var_daily = (max_vol_annual ** 2) / 252.0

            # Build sigma matrix for the N stock weights
            # Σ = X F X^T + diag(D)
            sigma = exposures @ factor_cov @ exposures.T
            np.fill_diagonal(sigma, sigma.diagonal() + specific_var)

            # For SOCP form: ||L^T w|| <= sqrt(target_var)
            # where L L^T = sigma (Cholesky)
            # But we can just pass sigma and target_var to the solver
            # and let it use quad_form.
            soc_constraints = [{
                "type": "quad_form_le",
                "P": sigma,  # N×N, only for weight variables
                "rhs": target_var_daily,
                "n_weights": N,
            }]

        return QPConstraints(G=G, h=h, A=A, b=b, lb=lb, ub=ub, soc_constraints=soc_constraints)
