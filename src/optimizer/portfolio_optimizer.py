"""Portfolio Optimizer facade — end-to-end optimization entry point."""

from __future__ import annotations

import logging
from typing import Any, Mapping

import numpy as np
import pandas as pd

from optimizer.alpha_processor import AlphaProcessor
from optimizer.candidate_pool import CandidatePoolBuilder
from optimizer.constraint_builder import ConstraintBuilder
from optimizer.diagnostics import OptimizationDiagnostics
from optimizer.objective_builder import ObjectiveBuilder
from optimizer.optimization_result import OptimizationResult
from optimizer.qp_solver import QPSolver
from optimizer.risk_attribution import RiskAttribution
from optimizer.risk_interface import RiskInterface

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """Mean-variance portfolio optimizer with risk model.

    Orchestrates the full pipeline: alpha processing → candidate pool →
    risk data loading → QP building → solving → result assembly.
    """

    def __init__(
        self,
        config: Mapping[str, Any],
        risk_interface: RiskInterface,
    ) -> None:
        self.config = dict(config)
        self.risk_interface = risk_interface

        self.alpha_processor = AlphaProcessor(self.config.get("alpha", {}))
        self.candidate_builder = CandidatePoolBuilder(self.config.get("candidate_pool", {}))
        self.objective_builder = ObjectiveBuilder()
        self.constraint_builder = ConstraintBuilder()
        self.solver = QPSolver(self.config.get("solver", {}))
        self.risk_attribution = RiskAttribution()

    def optimize(
        self,
        trade_date: int,
        alpha_raw: pd.Series,
        current_weights: pd.Series | None = None,
        market_data: pd.DataFrame | None = None,
    ) -> OptimizationResult:
        """Run a single optimization step.

        Parameters
        ----------
        trade_date : int
            Trading date (YYYYMMDD).
        alpha_raw : pd.Series
            Raw alpha predictions, index=ts_code.
        current_weights : pd.Series, optional
            Current position weights. If None, starts from 0.
        market_data : pd.DataFrame, optional
            Per-stock market data for candidate pool filtering.

        Returns
        -------
        OptimizationResult
        """
        if current_weights is None:
            current_weights = pd.Series(dtype=float)

        # Step 1: Alpha standardization
        alpha_std = self.alpha_processor.process(alpha_raw)
        if len(alpha_std) == 0:
            return self._empty_result(trade_date, alpha_raw, current_weights)

        # Step 2: Build candidate pool
        candidates = self.candidate_builder.build(
            alpha_std, market_data=market_data, current_holdings=current_weights,
        )
        if len(candidates) == 0:
            return self._empty_result(trade_date, alpha_raw, current_weights)

        alpha = alpha_std.reindex(candidates).fillna(0.0)
        w_prev = current_weights.reindex(candidates, fill_value=0.0).values.astype(np.float64)

        # Step 3: Get risk data
        risk_data = self.risk_interface.get_day_risk_data(
            trade_date, candidates.tolist(),
        )

        # Align: only keep stocks that have risk data
        if len(risk_data.stock_codes) < len(candidates):
            valid_idx = pd.Index(risk_data.stock_codes)
            alpha = alpha.reindex(valid_idx).fillna(0.0)
            w_prev = pd.Series(w_prev, index=candidates).reindex(valid_idx, fill_value=0.0).values
            candidates = valid_idx

        X = risk_data.exposures
        F = risk_data.factor_cov
        D = risk_data.specific_variance

        # Step 4: Build QP
        obj = self.objective_builder.build(
            alpha=alpha.values,
            exposures=X,
            factor_cov=F,
            specific_var=D,
            risk_aversion=float(self.config.get("objective", {}).get("risk_aversion", 0.1)),
            turnover_penalty=float(self.config.get("objective", {}).get("turnover_penalty", 0.0)),
            prev_weights=w_prev,
        )
        cons = self.constraint_builder.build(
            n_stocks=len(candidates),
            prev_weights=w_prev,
            exposures=X,
            style_factor_idx=risk_data.style_factor_idx,
            industry_factor_idx=risk_data.industry_factor_idx,
            factor_names=risk_data.factor_names,
            factor_cov=F,
            specific_var=D,
            config=self.config.get("constraints", {}),
        )

        # Step 5: Solve
        sol = self.solver.solve(obj, cons)
        N = len(candidates)
        w_opt = sol.x[:N]

        # Numerical cleanup: clip tiny negatives to 0
        w_opt = np.where(w_opt < 1e-8, 0.0, w_opt)

        # Renormalize if sum is off due to solver tolerance
        w_sum = w_opt.sum()
        if w_sum > 0 and abs(w_sum - 1.0) > 1e-4:
            logger.debug(
                "Renormalizing weights from sum=%.6f to 1.0 (date=%d)",
                w_sum, trade_date,
            )
            w_opt = w_opt / w_sum

        weights = pd.Series(w_opt, index=candidates, name="weight")

        # Step 6: Compute risk metrics
        port_var = RiskInterface.portfolio_variance(w_opt, X, F, D)
        port_vol = float(np.sqrt(max(port_var, 0.0)))
        expected_alpha = float(alpha.values @ w_opt)
        turnover = float(0.5 * np.sum(np.abs(w_opt - w_prev)))

        # Factor exposure
        factor_exp = RiskInterface.factor_exposure_portfolio(
            w_opt, X, risk_data.factor_names,
        )
        style_exp = factor_exp.iloc[risk_data.style_factor_idx] if risk_data.style_factor_idx else pd.Series(dtype=float)
        industry_exp = factor_exp.iloc[risk_data.industry_factor_idx] if risk_data.industry_factor_idx else pd.Series(dtype=float)

        # Trade list
        trade_list = pd.DataFrame({
            "current_weight": w_prev,
            "target_weight": w_opt,
            "trade_weight": w_opt - w_prev,
        }, index=candidates)

        # Alpha retention: optimized alpha vs raw top-N equal-weight alpha
        top_n = min(self.candidate_builder.pool_size, len(alpha_std))
        raw_top_alpha = alpha_std.nlargest(top_n).mean() if top_n > 0 else 0.0
        alpha_retention = expected_alpha / raw_top_alpha if raw_top_alpha != 0 else 0.0

        n_holdings = int(np.sum(w_opt > 1e-6))

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
            solver_status=sol.status,
            objective_value=sol.objective_value,
            solve_time_ms=sol.solve_time_ms,
            alpha_retention=alpha_retention,
            raw_alpha_portfolio=float(raw_top_alpha),
            n_candidates=len(candidates),
            n_holdings=n_holdings,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_result(
        self,
        trade_date: int,
        alpha_raw: pd.Series,
        current_weights: pd.Series,
    ) -> OptimizationResult:
        """Return an empty result (no valid candidates)."""
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
