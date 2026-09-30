"""Model evaluation framework — unified report after training."""

from .evaluation_report import EvaluationConfig, EvaluationReport, run_evaluation
from .metrics import (
    calc_downside_filter_score,
    calc_long_side_ic,
    calc_overall_ic,
    calc_return_decile_ic,
    calc_top_k_hit_rate,
    calc_top_quantile_return,
    calc_upside_capture,
)

__all__ = [
    "EvaluationConfig",
    "EvaluationReport",
    "calc_downside_filter_score",
    "calc_long_side_ic",
    "calc_overall_ic",
    "calc_return_decile_ic",
    "calc_top_k_hit_rate",
    "calc_top_quantile_return",
    "calc_upside_capture",
    "run_evaluation",
]
