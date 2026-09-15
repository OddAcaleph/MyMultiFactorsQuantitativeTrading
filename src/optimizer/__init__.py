"""Portfolio Optimizer V1 — mean-variance optimization with risk model.

Phases 1-4: Alpha + Risk + basic constraints + industry/style + turnover + attribution.
"""

from optimizer.optimization_result import OptimizationResult, DayRiskData
from optimizer.alpha_processor import AlphaProcessor
from optimizer.factor_timing import FactorTiming
from optimizer.candidate_pool import CandidatePoolBuilder
from optimizer.risk_interface import RiskInterface
from optimizer.objective_builder import ObjectiveBuilder, QPObjective
from optimizer.constraint_builder import ConstraintBuilder, QPConstraints
from optimizer.qp_solver import QPSolver, QPSolution
from optimizer.portfolio_optimizer import PortfolioOptimizer
from optimizer.risk_parity_strategy import RiskParityStrategy
from optimizer.risk_attribution import RiskAttribution, RiskAttributionResult
from optimizer.diagnostics import OptimizationDiagnostics

from optimizer.risk_parity_strategy import RiskParityStrategy

__all__ = [
    "OptimizationResult",
    "DayRiskData",
    "AlphaProcessor",
    "FactorTiming",
    "CandidatePoolBuilder",
    "RiskInterface",
    "ObjectiveBuilder",
    "QPObjective",
    "ConstraintBuilder",
    "QPConstraints",
    "QPSolver",
    "QPSolution",
    "PortfolioOptimizer",
    "RiskParityStrategy",
    "RiskAttribution",
    "RiskAttributionResult",
    "OptimizationDiagnostics",
]
