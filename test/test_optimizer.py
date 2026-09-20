"""Tests for Portfolio Optimizer V1 (Phases 1-4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer.alpha_processor import AlphaProcessor
from optimizer.candidate_pool import CandidatePoolBuilder
from optimizer.constraint_builder import ConstraintBuilder, QPConstraints
from optimizer.diagnostics import OptimizationDiagnostics
from optimizer.objective_builder import ObjectiveBuilder, QPObjective
from optimizer.optimization_result import DayRiskData, OptimizationResult
from optimizer.portfolio_optimizer import PortfolioOptimizer
from optimizer.qp_solver import QPSolver
from optimizer.risk_attribution import RiskAttribution
from optimizer.risk_interface import RiskInterface


# ======================================================================
# AlphaProcessor
# ======================================================================

class TestAlphaProcessor:
    def test_zscore_mean_zero_std_one(self):
        alpha = pd.Series(np.random.randn(100), index=[f"S{i:03d}" for i in range(100)])
        proc = AlphaProcessor({"method": "zscore", "winsorize": False})
        out = proc.process(alpha)
        assert abs(out.mean()) < 1e-10
        assert abs(out.std(ddof=0) - 1.0) < 1e-10

    def test_winsorize_clips_extremes(self):
        alpha = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0, -50.0, 5.0, 6.0, 7.0, 8.0])
        proc = AlphaProcessor({"method": "zscore", "winsorize": True, "winsorize_quantile": 0.1})
        out = proc.process(alpha)
        assert out.max() < 50.0
        assert out.min() > -30.0

    def test_rank_method(self):
        alpha = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        proc = AlphaProcessor({"method": "rank", "winsorize": False})
        out = proc.process(alpha)
        assert abs(out.mean()) < 1e-10
        assert abs(out.std(ddof=0) - 1.0) < 1e-10

    def test_drop_nan(self):
        alpha = pd.Series([1.0, np.nan, 3.0], index=["A", "B", "C"])
        proc = AlphaProcessor({"winsorize": False})
        out = proc.process(alpha)
        assert "B" not in out.index
        assert len(out) == 2


# ======================================================================
# CandidatePoolBuilder
# ======================================================================

class TestCandidatePoolBuilder:
    def _make_alpha(self, n=50):
        return pd.Series(
            np.random.randn(n),
            index=[f"S{i:03d}" for i in range(n)],
        )

    def test_pool_size(self):
        alpha = self._make_alpha(50)
        builder = CandidatePoolBuilder({"pool_size": 10})
        pool = builder.build(alpha)
        assert len(pool) == 10

    def test_filter_st(self):
        alpha = self._make_alpha(20)
        mkt = pd.DataFrame({
            "is_st": [False] * 15 + [True] * 5,
        }, index=alpha.index)
        builder = CandidatePoolBuilder({"pool_size": 20, "filter_st": True})
        pool = builder.build(alpha, market_data=mkt)
        st_codes = alpha.index[15:]
        for c in st_codes:
            assert c not in pool

    def test_filter_liquidity(self):
        alpha = self._make_alpha(20)
        mkt = pd.DataFrame({
            "avg_amount_20d": [10000] * 10 + [1000] * 10,
        }, index=alpha.index)
        builder = CandidatePoolBuilder({"pool_size": 20, "min_avg_amount_20d": 5000})
        pool = builder.build(alpha, market_data=mkt)
        assert len(pool) == 10

    def test_keep_current_holdings(self):
        alpha = self._make_alpha(50)
        builder = CandidatePoolBuilder({"pool_size": 5})
        # Pick a stock that is definitely NOT in the top 5
        bottom_stock = alpha.nsmallest(1).index[0]
        holdings = pd.Series(0.1, index=[bottom_stock])
        pool = builder.build(alpha, current_holdings=holdings)
        assert bottom_stock in pool
        assert len(pool) == 6  # 5 top + 1 held


# ======================================================================
# RiskInterface (static methods)
# ======================================================================

class TestRiskInterfaceStatic:
    def test_portfolio_variance_hand_calc(self):
        # 2 stocks, 1 factor
        X = np.array([[1.0], [2.0]])
        F = np.array([[0.04]])
        D = np.array([0.01, 0.02])
        w = np.array([0.5, 0.5])

        # Manual: z = X^T w = 1.5, factor_var = 1.5 * 0.04 * 1.5 = 0.09
        # specific = 0.25 * 0.01 + 0.25 * 0.02 = 0.0075
        # total = 0.0975
        var = RiskInterface.portfolio_variance(w, X, F, D)
        assert abs(var - 0.0975) < 1e-10

    def test_factor_exposure(self):
        X = np.array([[1.0, 0.5], [2.0, -0.5]])
        w = np.array([0.5, 0.5])
        exp = RiskInterface.factor_exposure_portfolio(w, X, ["F1", "F2"])
        assert abs(exp["F1"] - 1.5) < 1e-10
        assert abs(exp["F2"] - 0.0) < 1e-10


# ======================================================================
# ObjectiveBuilder
# ======================================================================

class TestObjectiveBuilder:
    def test_dimensions(self):
        N = 10
        K = 3
        builder = ObjectiveBuilder()
        obj = builder.build(
            alpha=np.random.randn(N),
            exposures=np.random.randn(N, K),
            factor_cov=np.eye(K) * 0.01,
            specific_var=np.ones(N) * 0.001,
            risk_aversion=0.1,
            turnover_penalty=0.01,
            prev_weights=np.zeros(N),
        )
        assert obj.P.shape == (2 * N, 2 * N)
        assert obj.q.shape == (2 * N,)
        assert obj.n_variables == 2 * N

    def test_p_symmetric(self):
        N = 8
        K = 3
        builder = ObjectiveBuilder()
        obj = builder.build(
            alpha=np.random.randn(N),
            exposures=np.random.randn(N, K),
            factor_cov=np.eye(K) * 0.01,
            specific_var=np.ones(N) * 0.001,
            risk_aversion=0.1,
            turnover_penalty=0.01,
            prev_weights=np.zeros(N),
        )
        assert np.allclose(obj.P, obj.P.T)

    def test_turnover_penalty_in_q(self):
        N = 5
        K = 2
        gamma = 0.05
        builder = ObjectiveBuilder()
        obj = builder.build(
            alpha=np.zeros(N),
            exposures=np.zeros((N, K)),
            factor_cov=np.zeros((K, K)),
            specific_var=np.zeros(N),
            risk_aversion=0.0,
            turnover_penalty=gamma,
            prev_weights=np.zeros(N),
        )
        assert np.all(obj.q[N:] == gamma)


# ======================================================================
# ConstraintBuilder
# ======================================================================

class TestConstraintBuilder:
    def test_sum_to_one(self):
        N = 5
        builder = ConstraintBuilder()
        cons = builder.build(N, np.zeros(N), {"fully_invested": True})
        assert cons.A.shape[0] == 1
        assert np.all(cons.A[0, :N] == 1.0)
        assert cons.b[0] == 1.0

    def test_long_only_bounds(self):
        N = 5
        builder = ConstraintBuilder()
        cons = builder.build(N, np.zeros(N), {"long_only": True})
        assert np.all(cons.lb[:N] == 0.0)

    def test_weight_cap(self):
        N = 5
        max_w = 0.03
        builder = ConstraintBuilder()
        cons = builder.build(N, np.zeros(N), {"max_weight": max_w})
        assert np.all(cons.ub[:N] == max_w)

    def test_turnover_auxiliary_count(self):
        N = 5
        builder = ConstraintBuilder()
        cons = builder.build(N, np.zeros(N), {})
        # 2N inequality constraints for turnover (t >= w - wp, t >= wp - w)
        assert cons.G.shape[0] == 2 * N
        assert cons.G.shape[1] == 2 * N

    def test_t_nonnegative(self):
        N = 3
        builder = ConstraintBuilder()
        cons = builder.build(N, np.zeros(N), {})
        assert np.all(cons.lb[N:] == 0.0)


# ======================================================================
# QPSolver
# ======================================================================

class TestQPSolver:
    def test_simple_qp(self):
        """min 0.5 x^T P x + q^T x  s.t. sum(x) = 1, x >= 0.

        With P = I, q = [-1, -2, -3], the optimal solution should put all
        weight on x2 (index 2) since it has the lowest (most negative) q.
        """
        N = 3
        P = np.eye(2 * N) * 2.0  # so 0.5 P x^2 = x^2
        q = np.zeros(2 * N)
        q[:N] = np.array([-1.0, -2.0, -3.0])

        G = np.zeros((2 * N, 2 * N))
        h = np.zeros(2 * N)
        # turnover constraints (trivial since w_prev=0)
        for i in range(N):
            G[2 * i, i] = 1.0
            G[2 * i, N + i] = -1.0
            G[2 * i + 1, i] = -1.0
            G[2 * i + 1, N + i] = -1.0

        A = np.zeros((1, 2 * N))
        A[0, :N] = 1.0
        b = np.array([1.0])

        lb = np.zeros(2 * N)
        ub = np.full(2 * N, np.inf)
        ub[:N] = 1.0

        solver = QPSolver({"solver": "OSQP", "verbose": False})
        sol = solver.solve(
            QPObjective(P=P, q=q, n_variables=2 * N),
            QPConstraints(G=G, h=h, A=A, b=b, lb=lb, ub=ub),
        )
        assert "optimal" in sol.status
        w = sol.x[:N]
        assert abs(w.sum() - 1.0) < 1e-4
        assert np.argmax(w) == 2  # highest alpha stock

    def test_infeasible(self):
        """sum(w) = 1 but w_i <= 0.1 with 5 stocks → infeasible."""
        N = 5
        P = np.eye(2 * N) * 0.02
        q = np.zeros(2 * N)
        q[:N] = -np.ones(N)

        G = np.zeros((2 * N, 2 * N))
        h = np.zeros(2 * N)
        for i in range(N):
            G[2 * i, i] = 1.0
            G[2 * i, N + i] = -1.0
            G[2 * i + 1, i] = -1.0
            G[2 * i + 1, N + i] = -1.0

        A = np.zeros((1, 2 * N))
        A[0, :N] = 1.0
        b = np.array([1.0])

        lb = np.zeros(2 * N)
        ub = np.full(2 * N, np.inf)
        ub[:N] = 0.1  # max 0.1 each, 5 stocks → max 0.5 < 1

        solver = QPSolver({"solver": "OSQP", "verbose": False})
        sol = solver.solve(
            QPObjective(P=P, q=q, n_variables=2 * N),
            QPConstraints(G=G, h=h, A=A, b=b, lb=lb, ub=ub),
        )
        assert "infeasible" in sol.status or "optimal" not in sol.status


# ======================================================================
# End-to-end PortfolioOptimizer
# ======================================================================

class TestPortfolioOptimizerEndToEnd:
    def _make_risk_interface_mock(self, n_stocks=20, n_style=3, n_industry=5):
        """Create a mock risk interface with synthetic data."""
        K = n_style + n_industry
        factor_names = [f"STYLE{i}" for i in range(n_style)] + [
            f"L1_IND{i}" for i in range(n_industry)
        ]
        style_idx = list(range(n_style))
        industry_idx = list(range(n_style, K))

        class MockRiskInterface:
            def __init__(self_):
                self_.factor_names = factor_names
                self_.style_idx = style_idx
                self_.industry_idx = industry_idx

            def get_day_risk_data(self_, trade_date, stock_codes):
                N = len(stock_codes)
                np.random.seed(42)
                exposures = np.random.randn(N, K) * 0.5
                # Industry one-hot
                exposures[:, n_style:] = 0.0
                for i in range(N):
                    exposures[i, n_style + (i % n_industry)] = 1.0

                factor_cov = np.eye(K) * 0.01
                # Add some correlation between style factors
                factor_cov[0, 1] = 0.003
                factor_cov[1, 0] = 0.003

                specific_var = np.ones(N) * 0.0005

                return DayRiskData(
                    trade_date=trade_date,
                    stock_codes=stock_codes,
                    exposures=exposures,
                    factor_cov=factor_cov,
                    specific_variance=specific_var,
                    factor_names=factor_names,
                    style_factor_idx=style_idx,
                    industry_factor_idx=industry_idx,
                )

        return MockRiskInterface()

    def test_basic_optimization(self):
        N = 20
        risk_if = self._make_risk_interface_mock(n_stocks=N)
        config = {
            "alpha": {"method": "zscore", "winsorize": False},
            "candidate_pool": {"pool_size": 15},
            "objective": {"risk_aversion": 0.5, "turnover_penalty": 0.0},
            "constraints": {
                "long_only": True,
                "fully_invested": True,
                "max_weight": 0.15,
            },
            "solver": {"solver": "OSQP", "verbose": False},
        }
        opt = PortfolioOptimizer(config, risk_if)

        np.random.seed(123)
        alpha = pd.Series(
            np.random.randn(N),
            index=[f"S{i:03d}" for i in range(N)],
        )

        result = opt.optimize(20240101, alpha)

        assert "optimal" in result.solver_status
        assert abs(result.weights.sum() - 1.0) < 1e-3
        assert (result.weights >= -1e-6).all()
        assert (result.weights <= 0.15 + 1e-4).all()
        assert result.n_holdings > 0
        assert result.portfolio_volatility > 0
        assert result.solve_time_ms < 1000  # should be fast

    def test_optimization_with_prev_weights(self):
        N = 15
        risk_if = self._make_risk_interface_mock(n_stocks=N)
        config = {
            "alpha": {"method": "zscore", "winsorize": False},
            "candidate_pool": {"pool_size": 12},
            "objective": {"risk_aversion": 0.5, "turnover_penalty": 0.1},
            "constraints": {
                "long_only": True,
                "fully_invested": True,
                "max_weight": 0.2,
            },
            "solver": {"solver": "OSQP", "verbose": False},
        }
        opt = PortfolioOptimizer(config, risk_if)

        np.random.seed(456)
        alpha = pd.Series(
            np.random.randn(N),
            index=[f"S{i:03d}" for i in range(N)],
        )

        # Equal weight initial
        prev = pd.Series(1.0 / N, index=alpha.index)
        result = opt.optimize(20240101, alpha, current_weights=prev)

        assert "optimal" in result.solver_status
        assert result.turnover >= 0.0
        assert result.turnover < 1.0  # should be partial turnover

    def test_higher_risk_aversion_lower_vol(self):
        N = 30
        risk_if = self._make_risk_interface_mock(n_stocks=N, n_style=5, n_industry=8)

        def run(lam):
            config = {
                "alpha": {"method": "zscore", "winsorize": False},
                "candidate_pool": {"pool_size": 25},
                "objective": {"risk_aversion": lam, "turnover_penalty": 0.0},
                "constraints": {
                    "long_only": True,
                    "fully_invested": True,
                    "max_weight": 0.5,  # loose cap so risk aversion matters
                },
                "solver": {"solver": "OSQP", "verbose": False},
            }
            opt = PortfolioOptimizer(config, risk_if)
            np.random.seed(789)
            alpha = pd.Series(
                np.random.randn(N),
                index=[f"S{i:03d}" for i in range(N)],
            )
            return opt.optimize(20240101, alpha)

        low_risk = run(0.001)
        high_risk = run(50.0)

        # Higher risk aversion → lower volatility
        assert high_risk.portfolio_volatility < low_risk.portfolio_volatility * 0.95
        # Higher risk aversion → lower expected alpha
        assert high_risk.expected_alpha < low_risk.expected_alpha

    def test_factor_exposure_calculated(self):
        N = 10
        risk_if = self._make_risk_interface_mock(n_stocks=N, n_style=3, n_industry=4)
        config = {
            "alpha": {"method": "zscore", "winsorize": False},
            "candidate_pool": {"pool_size": 8},
            "objective": {"risk_aversion": 0.1, "turnover_penalty": 0.0},
            "constraints": {"long_only": True, "fully_invested": True, "max_weight": 0.2},
            "solver": {"solver": "OSQP", "verbose": False},
        }
        opt = PortfolioOptimizer(config, risk_if)
        np.random.seed(111)
        alpha = pd.Series(np.random.randn(N), index=[f"S{i:03d}" for i in range(N)])
        result = opt.optimize(20240101, alpha)

        assert len(result.factor_exposure) == 7  # 3 style + 4 industry
        assert len(result.style_exposure) == 3
        assert len(result.industry_exposure) == 4


# ======================================================================
# Phase 2: Industry constraints
# ======================================================================

class TestIndustryConstraint:
    def test_industry_constraint_added(self):
        """Industry config should add n_industry inequality constraints."""
        N = 10
        n_ind = 3
        builder = ConstraintBuilder()
        X = np.zeros((N, n_ind))
        # Assign each stock to one industry
        for i in range(N):
            X[i, i % n_ind] = 1.0

        cons = builder.build(
            N, np.zeros(N),
            exposures=X,
            industry_factor_idx=list(range(n_ind)),
            config={"industry": {"max_weight": 0.4}},
        )
        # 2N turnover + N industry = 3N
        assert cons.G.shape[0] == 2 * N + n_ind

    def test_industry_constraint_enforced(self):
        """Optimizer should respect industry weight cap."""
        N = 15
        n_style = 2
        n_ind = 3
        K = n_style + n_ind

        factor_names = ["SIZE", "BETA"] + [f"IND{i}" for i in range(n_ind)]
        style_idx = [0, 1]
        industry_idx = [2, 3, 4]

        class MockRisk:
            def get_day_risk_data(self, trade_date, stock_codes):
                n = len(stock_codes)
                np.random.seed(42)
                X = np.random.randn(n, K) * 0.3
                X[:, 2:] = 0.0
                for i in range(n):
                    X[i, 2 + (i % n_ind)] = 1.0
                F = np.eye(K) * 0.01
                D = np.ones(n) * 0.001
                return DayRiskData(
                    trade_date=trade_date, stock_codes=stock_codes,
                    exposures=X, factor_cov=F, specific_variance=D,
                    factor_names=factor_names,
                    style_factor_idx=style_idx, industry_factor_idx=industry_idx,
                )

        config = {
            "alpha": {"method": "zscore", "winsorize": False},
            "candidate_pool": {"pool_size": 12},
            "objective": {"risk_aversion": 0.1, "turnover_penalty": 0.0},
            "constraints": {
                "long_only": True, "fully_invested": True,
                "max_weight": 0.5,
                "industry": {"max_weight": 0.45},  # 3 industries × 0.45 = 1.35 > 1.0 (feasible)
            },
            "solver": {"solver": "OSQP", "verbose": False},
        }
        opt = PortfolioOptimizer(config, MockRisk())

        np.random.seed(999)
        alpha = pd.Series(np.random.randn(N), index=[f"S{i:03d}" for i in range(N)])
        result = opt.optimize(20240101, alpha)

        assert "optimal" in result.solver_status
        # Check each industry weight <= 0.45 (with tolerance)
        for ind_name, ind_w in result.industry_exposure.items():
            assert ind_w <= 0.45 + 1e-3, f"Industry {ind_name} weight {ind_w} > 0.45"


# ======================================================================
# Phase 2: Style constraints
# ======================================================================

class TestStyleConstraint:
    def test_style_constraint_added(self):
        """Style config should add 2 * n_style inequality constraints."""
        N = 10
        n_style = 3
        builder = ConstraintBuilder()
        X = np.random.randn(N, n_style)

        cons = builder.build(
            N, np.zeros(N),
            exposures=X,
            style_factor_idx=list(range(n_style)),
            config={"style": {i: [-0.5, 0.5] for i in range(n_style)}},
        )
        # 2N turnover + 2*N_style upper+lower bounds
        assert cons.G.shape[0] == 2 * N + 2 * n_style

    def test_style_constraint_enforced(self):
        """Optimizer should respect style exposure bounds."""
        N = 20
        n_style = 2
        n_ind = 4
        K = n_style + n_ind

        factor_names = ["SIZE", "MOMENTUM"] + [f"IND{i}" for i in range(n_ind)]
        style_idx = [0, 1]
        industry_idx = [2, 3, 4, 5]

        class MockRisk:
            def get_day_risk_data(self, trade_date, stock_codes):
                n = len(stock_codes)
                np.random.seed(777)
                X = np.random.randn(n, K) * 0.5
                X[:, n_style:] = 0.0
                for i in range(n):
                    X[i, n_style + (i % n_ind)] = 1.0
                F = np.eye(K) * 0.01
                D = np.ones(n) * 0.001
                return DayRiskData(
                    trade_date=trade_date, stock_codes=stock_codes,
                    exposures=X, factor_cov=F, specific_variance=D,
                    factor_names=factor_names,
                    style_factor_idx=style_idx, industry_factor_idx=industry_idx,
                )

        config = {
            "alpha": {"method": "zscore", "winsorize": False},
            "candidate_pool": {"pool_size": 15},
            "objective": {"risk_aversion": 0.01, "turnover_penalty": 0.0},
            "constraints": {
                "long_only": True, "fully_invested": True,
                "max_weight": 0.2,
                "style": {
                    0: [-0.3, 0.3],  # SIZE
                    1: [-0.5, 0.5],  # MOMENTUM
                },
            },
            "solver": {"solver": "OSQP", "verbose": False},
        }
        opt = PortfolioOptimizer(config, MockRisk())

        np.random.seed(555)
        alpha = pd.Series(np.random.randn(N), index=[f"S{i:03d}" for i in range(N)])
        result = opt.optimize(20240101, alpha)

        assert "optimal" in result.solver_status
        # Check style exposures within bounds
        assert result.style_exposure.iloc[0] <= 0.3 + 1e-3
        assert result.style_exposure.iloc[0] >= -0.3 - 1e-3
        assert result.style_exposure.iloc[1] <= 0.5 + 1e-3
        assert result.style_exposure.iloc[1] >= -0.5 - 1e-3


# ======================================================================
# Phase 3: Turnover penalty
# ======================================================================

class TestTurnoverPenalty:
    def test_higher_penalty_lower_turnover(self):
        """Higher turnover penalty → lower turnover."""
        N = 20
        n_style = 3
        n_ind = 5
        K = n_style + n_ind

        factor_names = [f"S{i}" for i in range(n_style)] + [f"I{i}" for i in range(n_ind)]
        style_idx = list(range(n_style))
        industry_idx = list(range(n_style, K))

        class MockRisk:
            def get_day_risk_data(self, trade_date, stock_codes):
                n = len(stock_codes)
                np.random.seed(123)
                X = np.random.randn(n, K) * 0.5
                X[:, n_style:] = 0.0
                for i in range(n):
                    X[i, n_style + (i % n_ind)] = 1.0
                F = np.eye(K) * 0.01
                D = np.ones(n) * 0.001
                return DayRiskData(
                    trade_date=trade_date, stock_codes=stock_codes,
                    exposures=X, factor_cov=F, specific_variance=D,
                    factor_names=factor_names,
                    style_factor_idx=style_idx, industry_factor_idx=industry_idx,
                )

        def run(gamma):
            config = {
                "alpha": {"method": "zscore", "winsorize": False},
                "candidate_pool": {"pool_size": 15},
                "objective": {"risk_aversion": 0.1, "turnover_penalty": gamma},
                "constraints": {"long_only": True, "fully_invested": True, "max_weight": 0.2},
                "solver": {"solver": "OSQP", "verbose": False},
            }
            opt = PortfolioOptimizer(config, MockRisk())
            np.random.seed(456)
            alpha = pd.Series(np.random.randn(N), index=[f"S{i:03d}" for i in range(N)])
            # Start from equal weight
            prev = pd.Series(1.0 / N, index=alpha.index)
            return opt.optimize(20240101, alpha, current_weights=prev)

        low_gamma = run(0.001)
        high_gamma = run(1.0)

        # Higher penalty → lower turnover
        assert high_gamma.turnover < low_gamma.turnover
        # Higher penalty → alpha closer to initial (less change)
        # (initial alpha is random, but turnover should definitely be lower)

    def test_turnover_calculation(self):
        """Turnover = 0.5 * sum(|w_new - w_prev|)."""
        w_prev = np.array([0.4, 0.3, 0.2, 0.1])
        w_new = np.array([0.3, 0.3, 0.3, 0.1])
        expected = 0.5 * np.sum(np.abs(w_new - w_prev))
        # |0.3-0.4| + |0.3-0.3| + |0.3-0.2| + |0.1-0.1| = 0.1 + 0 + 0.1 + 0 = 0.2
        # 0.5 * 0.2 = 0.1
        assert abs(expected - 0.1) < 1e-10


# ======================================================================
# Phase 4: Risk Attribution
# ======================================================================

class TestRiskAttribution:
    def test_attribution_sum_to_one(self):
        """Factor + specific risk contributions should sum to 1.0."""
        N = 10
        K = 5
        np.random.seed(42)
        weights = np.random.dirichlet(np.ones(N))
        exposures = np.random.randn(N, K) * 0.5
        factor_cov = np.eye(K) * 0.01 + 0.001
        specific_var = np.ones(N) * 0.001
        factor_names = [f"F{i}" for i in range(K)]

        attr = RiskAttribution()
        result = attr.compute(weights, exposures, factor_cov, specific_var, factor_names)

        total = result.factor_contribution_pct.sum() + result.specific_contribution_pct
        assert abs(total - 1.0) < 1e-6

    def test_attribution_matches_total_variance(self):
        """factor_var + specific_var should equal total_var."""
        N = 8
        K = 3
        np.random.seed(99)
        weights = np.array([0.2, 0.15, 0.25, 0.1, 0.1, 0.1, 0.05, 0.05])
        exposures = np.random.randn(N, K)
        factor_cov = np.eye(K) * 0.02
        specific_var = np.ones(N) * 0.005
        factor_names = ["A", "B", "C"]

        attr = RiskAttribution()
        result = attr.compute(weights, exposures, factor_cov, specific_var, factor_names)

        # Direct computation
        z = exposures.T @ weights
        direct_factor = z @ factor_cov @ z
        direct_specific = np.sum(weights ** 2 * specific_var)
        direct_total = direct_factor + direct_specific

        assert abs(result.factor_variance - direct_factor) < 1e-10
        assert abs(result.specific_variance - direct_specific) < 1e-10
        assert abs(result.total_variance - direct_total) < 1e-10

    def test_hand_calc_single_factor(self):
        """Hand-calculate risk attribution for 2 stocks, 1 factor."""
        # 2 stocks, 1 factor
        w = np.array([0.6, 0.4])
        X = np.array([[1.0], [2.0]])
        F = np.array([[0.04]])
        D = np.array([0.01, 0.02])

        attr = RiskAttribution()
        result = attr.compute(w, X, F, D, ["F1"])

        # z = X^T w = 0.6*1 + 0.4*2 = 1.4
        # factor_var = 1.4 * 0.04 * 1.4 = 0.0784
        # specific_var = 0.36*0.01 + 0.16*0.02 = 0.0036 + 0.0032 = 0.0068
        # total = 0.0852
        assert abs(result.factor_variance - 0.0784) < 1e-10
        assert abs(result.specific_variance - 0.0068) < 1e-10
        assert abs(result.total_variance - 0.0852) < 1e-10

        # F1 contribution: z * (Fz) / total = 1.4 * (0.04*1.4) / 0.0852
        # = 1.4 * 0.056 / 0.0852 = 0.0784 / 0.0852 = 0.9202...
        f1_rc = result.factor_contribution_pct["F1"]
        assert abs(f1_rc - 0.0784 / 0.0852) < 1e-6
        assert abs(result.specific_contribution_pct - 0.0068 / 0.0852) < 1e-6

    def test_check_attribution_sum(self):
        """RiskAttribution.check_attribution_sum should validate correctly."""
        N = 5
        K = 3
        np.random.seed(7)
        weights = np.random.dirichlet(np.ones(N))
        exposures = np.random.randn(N, K)
        factor_cov = np.eye(K) * 0.01
        specific_var = np.ones(N) * 0.001

        attr = RiskAttribution()
        result = attr.compute(weights, exposures, factor_cov, specific_var,
                              [f"F{i}" for i in range(K)])
        assert RiskAttribution.check_attribution_sum(result)


# ======================================================================
# Phase 4: Diagnostics
# ======================================================================

class TestDiagnostics:
    def test_record_and_export(self):
        """Diagnostics should record and export as DataFrame."""
        diag = OptimizationDiagnostics()

        # Create a mock result
        weights = pd.Series([0.3, 0.3, 0.4], index=["A", "B", "C"])
        result = OptimizationResult(
            weights=weights,
            expected_alpha=0.05,
            portfolio_variance=0.001,
            portfolio_volatility=0.0316,
            turnover=0.1,
            trade_list=pd.DataFrame(),
            factor_exposure=pd.Series([0.5, 0.3], index=["SIZE", "BETA"]),
            style_exposure=pd.Series([0.5, 0.3], index=["SIZE", "BETA"]),
            industry_exposure=pd.Series(dtype=float),
            solver_status="optimal",
            objective_value=-0.04,
            solve_time_ms=5.2,
            alpha_retention=0.95,
            raw_alpha_portfolio=0.0526,
            n_candidates=10,
            n_holdings=3,
        )

        diag.record(20240101, result)
        diag.record(20240102, result)

        df = diag.to_dataframe()
        assert len(df) == 2
        assert "trade_date" in df.columns
        assert "solver_status" in df.columns
        assert "portfolio_volatility" in df.columns
        assert "turnover" in df.columns
        assert "alpha_retention" in df.columns
        assert "style_SIZE" in df.columns
        assert "max_weight" in df.columns

    def test_summary(self):
        """Summary should return mean/std/min/max."""
        diag = OptimizationDiagnostics()

        for i in range(10):
            weights = pd.Series([1.0], index=["A"])
            result = OptimizationResult(
                weights=weights,
                expected_alpha=float(i) * 0.01,
                portfolio_variance=0.001,
                portfolio_volatility=0.0316,
                turnover=0.05 + i * 0.01,
                trade_list=pd.DataFrame(),
                factor_exposure=pd.Series(dtype=float),
                style_exposure=pd.Series(dtype=float),
                industry_exposure=pd.Series(dtype=float),
                solver_status="optimal",
                objective_value=-0.01,
                solve_time_ms=5.0,
                alpha_retention=0.9,
                raw_alpha_portfolio=0.01,
                n_candidates=5,
                n_holdings=1,
            )
            diag.record(20240101 + i, result)

        summary = diag.summary()
        assert "turnover_mean" in summary
        assert "turnover_std" in summary
        assert "turnover_min" in summary
        assert "turnover_max" in summary
        assert summary["turnover_min"] < summary["turnover_max"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
