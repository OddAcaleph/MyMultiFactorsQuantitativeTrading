"""Risk-parity portfolio strategy using factor risk model.

Computes risk-parity (equal risk contribution) weights using the full
covariance matrix from the risk model: Σ = X F X^T + D.

Uses Newton's method on the risk-budgeting objective to find weights
where each stock contributes equal portfolio risk.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd

from optimizer.alpha_processor import AlphaProcessor
from optimizer.candidate_pool import CandidatePoolBuilder
from optimizer.optimization_result import OptimizationResult
from optimizer.risk_interface import RiskInterface

logger = logging.getLogger(__name__)


class RiskParityStrategy:
    """Risk-parity portfolio construction with risk model.

    Pipeline:
      1. Alpha standardization (with optional industry neutralization)
      2. Candidate pool selection (top-K by alpha, with filters)
      3. Industry-stratified selection (optional) to ensure industry diversity
      4. Risk-parity weighting using factor model covariance
      5. Alpha-tilted risk parity (optional): blend risk-parity with alpha scores

    Parameters
    ----------
    config : dict
        Strategy configuration with keys:
        - alpha: AlphaProcessor config
        - candidate_pool: CandidatePoolBuilder config
        - risk_parity:
            - alpha_tilt: float, blend factor between risk-parity and alpha-weighted
              (0 = pure risk-parity, 1 = pure alpha-weighted, default 0.3)
            - industry_neutral_selection: bool, select top-N per industry (default True)
            - max_weight: float, single-stock weight cap (default 0.10)
            - min_weight: float, minimum weight for holdings (default 0.005)
            - industry_max_weight: float, max industry weight (default 0.20)
            - n_iter: int, Newton iterations for risk-parity (default 100)
            - tol: float, convergence tolerance (default 1e-8)
    risk_interface : RiskInterface
        Risk model data provider.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        risk_interface: RiskInterface,
    ) -> None:
        self.config = dict(config)
        self.risk_interface = risk_interface

        rp_cfg = self.config.get("risk_parity", {})
        self.alpha_tilt: float = float(rp_cfg.get("alpha_tilt", 0.3))
        self.industry_neutral_selection: bool = bool(rp_cfg.get("industry_neutral_selection", True))
        self.max_weight: float = float(rp_cfg.get("max_weight", 0.10))
        self.min_weight: float = float(rp_cfg.get("min_weight", 0.005))
        self.industry_max_weight: float = float(rp_cfg.get("industry_max_weight", 0.20))
        self.n_iter: int = int(rp_cfg.get("n_iter", 100))
        self.tol: float = float(rp_cfg.get("tol", 1e-8))
        # Alpha scaling: convert z-score alpha to expected return units
        # alpha_scaling = IC * sigma_idio, where IC is the rank IC of the alpha
        # Default 0.1 means alpha z-score * 0.1 * sigma_idio = expected return
        self.alpha_scaling: float = float(rp_cfg.get("alpha_scaling", 0.1))
        # Max weight for alpha tilt (prevents extreme concentration)
        self.alpha_tilt_max_weight: float = float(rp_cfg.get("alpha_tilt_max_weight", 0.20))

        self.alpha_processor = AlphaProcessor(self.config.get("alpha", {}))
        self.candidate_builder = CandidatePoolBuilder(self.config.get("candidate_pool", {}))

    def optimize(
        self,
        trade_date: int,
        alpha_raw: pd.Series,
        current_weights: pd.Series | None = None,
        market_data: pd.DataFrame | None = None,
    ) -> OptimizationResult:
        """Run risk-parity portfolio construction.

        Parameters
        ----------
        trade_date : int
            Trading date (YYYYMMDD).
        alpha_raw : pd.Series
            Raw alpha predictions, index=ts_code.
        current_weights : pd.Series, optional
            Current position weights.
        market_data : pd.DataFrame, optional
            Per-stock market data for filtering.

        Returns
        -------
        OptimizationResult
        """
        if current_weights is None:
            current_weights = pd.Series(dtype=float)

        # Step 1: Alpha standardization with industry neutralization
        alpha_cfg = self.config.get("alpha", {})
        industry_map = None

        if alpha_cfg.get("industry_neutral", False) or self.industry_neutral_selection:
            try:
                risk_data_all = self.risk_interface.get_day_risk_data(
                    trade_date, alpha_raw.index.tolist(),
                )
                industry_map = self._build_industry_map(risk_data_all)
            except Exception:
                pass

        alpha_std = self.alpha_processor.process(
            alpha_raw, industry_map=industry_map,
        )
        if len(alpha_std) == 0:
            return self._empty_result(trade_date, alpha_raw, current_weights)

        # Step 2: Build candidate pool (apply filters, get top by alpha)
        candidates = self.candidate_builder.build(
            alpha_std, market_data=market_data, current_holdings=current_weights,
        )
        if len(candidates) == 0:
            return self._empty_result(trade_date, alpha_raw, current_weights)

        alpha = alpha_std.reindex(candidates).fillna(0.0)

        # Step 3: Industry-stratified selection (if enabled)
        if self.industry_neutral_selection and industry_map is not None:
            candidates = self._industry_stratified_select(
                alpha, industry_map, self.candidate_builder.pool_size,
            )
            alpha = alpha_std.reindex(candidates).fillna(0.0)

        if len(candidates) < 2:
            return self._empty_result(trade_date, alpha_raw, current_weights)

        # Step 4: Get risk data
        try:
            risk_data = self.risk_interface.get_day_risk_data(
                trade_date, candidates.tolist(),
            )
        except Exception:
            return self._empty_result(trade_date, alpha_raw, current_weights)

        # Align
        valid_idx = pd.Index(risk_data.stock_codes)
        if len(valid_idx) < len(candidates):
            alpha = alpha.reindex(valid_idx).fillna(0.0)
            candidates = valid_idx

        if len(candidates) < 2:
            return self._empty_result(trade_date, alpha_raw, current_weights)

        X = risk_data.exposures
        F = risk_data.factor_cov
        D = risk_data.specific_variance

        # Step 5: Compute risk-parity weights
        w_rp = self._risk_parity_weights(X, F, D)

        # Step 6: Alpha tilt — blend with alpha-based weights
        if self.alpha_tilt > 0:
            # Alpha weights: positive alpha normalized
            alpha_pos = alpha.clip(lower=0.0)
            if alpha_pos.sum() > 0:
                w_alpha = alpha_pos / alpha_pos.sum()
            else:
                w_alpha = pd.Series(1.0 / len(alpha), index=alpha.index)

            # Cap alpha weights to avoid extreme concentration
            w_alpha = w_alpha.clip(upper=self.alpha_tilt_max_weight)
            w_alpha = w_alpha / w_alpha.sum()

            w_mixed = (1.0 - self.alpha_tilt) * w_rp + self.alpha_tilt * w_alpha.values
            w_mixed = w_mixed / w_mixed.sum()
        else:
            w_mixed = w_rp.copy()

        # Step 7: Apply weight constraints (max_weight, industry_max_weight)
        w_final = self._apply_weight_constraints(
            w_mixed, risk_data,
        )

        # Step 8: Remove tiny positions and renormalize
        w_final = np.where(w_final < self.min_weight, 0.0, w_final)
        w_sum = w_final.sum()
        if w_sum > 0:
            w_final = w_final / w_sum

        weights = pd.Series(w_final, index=candidates, name="weight")
        w_prev = current_weights.reindex(candidates, fill_value=0.0).values.astype(np.float64)

        # Compute metrics
        port_var = RiskInterface.portfolio_variance(w_final, X, F, D)
        port_vol = float(np.sqrt(max(port_var, 0.0)))
        expected_alpha = float(alpha.values @ w_final)
        turnover = float(0.5 * np.sum(np.abs(w_final - w_prev)))

        factor_exp = RiskInterface.factor_exposure_portfolio(
            w_final, X, risk_data.factor_names,
        )
        style_exp = factor_exp.iloc[risk_data.style_factor_idx] if risk_data.style_factor_idx else pd.Series(dtype=float)
        industry_exp = factor_exp.iloc[risk_data.industry_factor_idx] if risk_data.industry_factor_idx else pd.Series(dtype=float)

        trade_list = pd.DataFrame({
            "current_weight": w_prev,
            "target_weight": w_final,
            "trade_weight": w_final - w_prev,
        }, index=candidates)

        top_n = min(self.candidate_builder.pool_size, len(alpha_std))
        raw_top_alpha = alpha_std.nlargest(top_n).mean() if top_n > 0 else 0.0
        alpha_retention = expected_alpha / raw_top_alpha if raw_top_alpha != 0 else 0.0

        n_holdings = int(np.sum(w_final > 1e-6))

        return OptimizationResult(
            weights=weights,
            expected_alpha=expected_alpha,
            portfolio_variance=float(port_var),
            portfolio_volatility=port_vol,
            turnover=turnover,
            trade_list=trade_list,
            factor_exposure=factor_exp,
            style_exposure=style_exp,
            industry_exposure=industry_exp,
            solver_status="risk_parity",
            objective_value=0.0,
            solve_time_ms=0.0,
            alpha_retention=alpha_retention,
            raw_alpha_portfolio=float(raw_top_alpha),
            n_candidates=len(candidates),
            n_holdings=n_holdings,
        )

    # ------------------------------------------------------------------
    # Core algorithms
    # ------------------------------------------------------------------

    def _risk_parity_weights(
        self,
        X: np.ndarray,
        F: np.ndarray,
        D: np.ndarray,
    ) -> np.ndarray:
        """Compute risk-parity weights using Newton's method.

        Solves: find w such that RC_i = w_i * (Σw)_i / sqrt(w^T Σ w)
        is equal for all i, with Σ = X F X^T + diag(D).

        Uses the cyclical coordinate descent approach which is robust and
        efficient for moderate N.
        """
        N = len(D)
        w = np.ones(N) / N  # initial: equal weight

        # Precompute factor loadings for efficiency
        # Σ w = X F X^T w + D * w
        # We compute iteratively using coordinate descent

        for iteration in range(self.n_iter):
            w_old = w.copy()

            for i in range(N):
                # Compute marginal contribution to risk for stock i
                # MCR_i = (Σ w)_i / sqrt(w^T Σ w)
                # RC_i = w_i * MCR_i

                # Σ w (full vector): N x K @ K x 1 -> N
                Xw = X.T @ w  # (K,) = factor-weighted exposures
                FXw = F @ Xw  # (K,) = factor risk contributions
                sigma_w = X @ FXw + D * w  # (N,) = marginal risk

                port_var = float(w @ sigma_w)
                port_vol = np.sqrt(max(port_var, 1e-20))

                # Target risk contribution per stock
                target_rc = port_vol / N

                # Current risk contribution of stock i
                rc_i = w[i] * sigma_w[i] / port_vol

                # Newton update for w_i
                # We want: w_i * sigma_w_i / vol = target_rc
                # Approximate: w_i_new = target_rc * vol / sigma_w_i
                # But sigma_w_i depends on w_i, so iterate
                if sigma_w[i] > 1e-20:
                    w[i] = target_rc * port_vol / sigma_w[i]
                else:
                    w[i] = 1.0 / N

            # Normalize
            w = w / w.sum()

            # Check convergence
            change = np.max(np.abs(w - w_old))
            if change < self.tol:
                break

        # Ensure non-negative
        w = np.maximum(w, 0.0)
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        else:
            w = np.ones(N) / N

        return w

    def _apply_weight_constraints(
        self,
        w: np.ndarray,
        risk_data,
    ) -> np.ndarray:
        """Apply max_weight and industry_max_weight constraints.

        Uses a simple iterative clipping approach:
        1. Clip individual weights to max_weight
        2. Clip industry weights to industry_max_weight
        3. Renormalize and repeat until stable
        """
        w = w.copy()
        N = len(w)

        for _ in range(20):
            w_old = w.copy()

            # 1. Individual weight cap
            w = np.minimum(w, self.max_weight)

            # 2. Industry weight cap
            if self.industry_max_weight > 0 and risk_data.industry_factor_idx:
                ind_idx = risk_data.industry_factor_idx
                X_ind = risk_data.exposures[:, ind_idx]

                for j in range(X_ind.shape[1]):
                    mask = X_ind[:, j] > 0.5  # stocks in this industry
                    if not np.any(mask):
                        continue
                    ind_weight = w[mask].sum()
                    if ind_weight > self.industry_max_weight:
                        scale = self.industry_max_weight / ind_weight
                        w[mask] *= scale

            # 3. Renormalize
            w_sum = w.sum()
            if w_sum > 0:
                w = w / w_sum

            if np.max(np.abs(w - w_old)) < 1e-8:
                break

        return w

    def _industry_stratified_select(
        self,
        alpha: pd.Series,
        industry_map: pd.Series,
        target_count: int,
    ) -> pd.Index:
        """Select top stocks per industry for diversification.

        Allocates target_count slots across industries proportionally to
        the number of stocks in each industry (with a minimum of 1 per industry),
        then picks the top alpha stock(s) within each industry.
        """
        common = alpha.index.intersection(industry_map.index)
        if len(common) == 0:
            return alpha.head(target_count).index

        alpha_common = alpha.loc[common]
        industries = industry_map.loc[common]

        # Get valid industries (those with at least 1 stock)
        ind_counts = industries.value_counts()
        valid_industries = ind_counts[ind_counts > 0].index.tolist()

        if len(valid_industries) == 0:
            return alpha.head(target_count).index

        # Allocate slots: at least 1 per industry, rest by size
        n_industries = len(valid_industries)
        base_per_industry = max(1, target_count // n_industries)
        remaining = target_count - base_per_industry * n_industries

        # Sort industries by number of stocks (descending) for extra slots
        sorted_industries = ind_counts.loc[valid_industries].sort_values(ascending=False).index.tolist()

        selected = []
        for i, ind in enumerate(sorted_industries):
            n_select = base_per_industry + (1 if i < remaining else 0)
            ind_stocks = alpha_common[industries == ind].sort_values(ascending=False)
            selected.extend(ind_stocks.head(n_select).index.tolist())

        return pd.Index(selected[:target_count])

    def _build_industry_map(self, risk_data) -> pd.Series:
        """Build industry map from risk data industry exposures.

        Uses idxmax to assign each stock to its dominant industry.
        """
        ind_idx = risk_data.industry_factor_idx
        if len(ind_idx) == 0:
            return pd.Series(dtype=str)

        X_ind = risk_data.exposures[:, ind_idx]
        factor_names = risk_data.factor_names
        ind_names = [factor_names[i] for i in ind_idx]

        # Assign each stock to the industry with highest exposure
        ind_codes = pd.Series(index=risk_data.stock_codes, dtype=str)
        for i, code in enumerate(risk_data.stock_codes):
            row = X_ind[i]
            max_j = int(np.argmax(row))
            if row[max_j] > 0.01:  # at least some exposure
                ind_codes.iloc[i] = ind_names[max_j]

        return ind_codes

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_result(
        self,
        trade_date: int,
        alpha_raw: pd.Series,
        current_weights: pd.Series,
    ) -> OptimizationResult:
        """Return an empty result."""
        empty_s = pd.Series(dtype=float)
        return OptimizationResult(
            weights=empty_s,
            expected_alpha=0.0,
            portfolio_variance=0.0,
            portfolio_volatility=0.0,
            turnover=0.0,
            trade_list=pd.DataFrame(),
            factor_exposure=empty_s,
            style_exposure=empty_s,
            industry_exposure=empty_s,
            solver_status="no_candidates",
            objective_value=float("inf"),
            solve_time_ms=0.0,
            alpha_retention=0.0,
            raw_alpha_portfolio=0.0,
            n_candidates=0,
            n_holdings=0,
        )
