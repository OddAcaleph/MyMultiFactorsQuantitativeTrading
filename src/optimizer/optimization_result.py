"""Optimization result data classes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd


@dataclass
class OptimizationResult:
    """Result of a single portfolio optimization run.

    Attributes
    ----------
    weights : pd.Series
        Target weights, index=ts_code.
    expected_alpha : float
        Portfolio expected alpha: alpha^T w.
    portfolio_variance : float
        Portfolio variance: w^T Σ w (daily).
    portfolio_volatility : float
        Portfolio volatility = sqrt(variance) (daily).
    turnover : float
        One-sided turnover: 0.5 * sum(|w - w_prev|).
    trade_list : pd.DataFrame
        Trading list with columns: current_weight, target_weight, trade_weight.
    factor_exposure : pd.Series
        Portfolio factor exposure (all factors), index=factor name.
    style_exposure : pd.Series
        Style factor exposure only.
    industry_exposure : pd.Series
        Industry factor exposure only.
    solver_status : str
        Solver status string (e.g. "optimal", "infeasible").
    objective_value : float
        Final objective function value.
    solve_time_ms : float
        Solver wall time in milliseconds.
    alpha_retention : float
        Ratio of optimized alpha to raw top-N alpha.
    raw_alpha_portfolio : float
        Alpha of the raw top-N equal-weight portfolio.
    n_candidates : int
        Number of candidate stocks.
    n_holdings : int
        Number of holdings with weight > 0.
    """

    weights: pd.Series
    expected_alpha: float
    portfolio_variance: float
    portfolio_volatility: float
    turnover: float
    trade_list: pd.DataFrame
    factor_exposure: pd.Series
    style_exposure: pd.Series
    industry_exposure: pd.Series
    solver_status: str
    objective_value: float
    solve_time_ms: float
    alpha_retention: float
    raw_alpha_portfolio: float
    n_candidates: int
    n_holdings: int


@dataclass
class DayRiskData:
    """Risk data for a single trading day.

    Attributes
    ----------
    trade_date : int
        Trading date (YYYYMMDD).
    stock_codes : list[str]
        Stock codes in order matching rows of exposures.
    exposures : np.ndarray
        Factor exposure matrix (N, K).
    factor_cov : np.ndarray
        Factor covariance matrix (K, K).
    specific_variance : np.ndarray
        Specific (idiosyncratic) variance vector (N,).
    factor_names : list[str]
        Names of the K factors.
    style_factor_idx : list[int]
        Indices of style factors in factor_names.
    industry_factor_idx : list[int]
        Indices of industry factors in factor_names.
    """

    trade_date: int
    stock_codes: List[str]
    exposures: np.ndarray
    factor_cov: np.ndarray
    specific_variance: np.ndarray
    factor_names: List[str]
    style_factor_idx: List[int] = field(default_factory=list)
    industry_factor_idx: List[int] = field(default_factory=list)

    @property
    def n_stocks(self) -> int:
        return len(self.stock_codes)

    @property
    def n_factors(self) -> int:
        return len(self.factor_names)
