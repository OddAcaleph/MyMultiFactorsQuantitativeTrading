"""Preparatory grouped return / monotonicity validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from qlib.contrib.evaluate import risk_analysis

try:
    from .base import DEFAULT_LOCAL_FEATURE_COL, DEFAULT_LOCAL_LABEL_COL, BaseICValidationStep, ValidationResult, load_local_test_pred_label
except ImportError:  # pragma: no cover - enables `python group_return.py`
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from base import DEFAULT_LOCAL_FEATURE_COL, DEFAULT_LOCAL_LABEL_COL, BaseICValidationStep, ValidationResult, load_local_test_pred_label


class GroupReturnAnalyzer(BaseICValidationStep):
    """Analyze stratified returns by score quantile.

    The grouping logic follows Qlib's ``_group_return``: sort each day by score
    descending, split into N equal groups, and compute mean realized label for
    each group plus long-short and long-average spreads.

    Qlib 0.9.7 exposes this exact N-group routine as the private report helper
    ``qlib.contrib.report.analysis_model.analysis_model_performance._group_return``.
    That helper is plot-oriented: it imports plotly and returns figures instead
    of the raw ``Group1..GroupN`` table.  This class therefore keeps the same
    Qlib formula locally so downstream code can consume raw metrics/dataframes.
    Public Qlib ``risk_analysis`` is still used for spread statistics.
    """

    def __init__(self, groups: int = 5, reverse: bool = False, freq: str = "day", **kwargs) -> None:
        super().__init__(**kwargs)
        if groups < 2:
            raise ValueError("groups must be >= 2")
        self.groups = groups
        self.reverse = reverse
        self.freq = freq

    def validate(self, pred_label: pd.DataFrame) -> ValidationResult:
        df = self.prepare_pred_label(pred_label)
        group_returns = self.calc_group_returns(df)
        metrics = self.calc_metrics(group_returns)
        return ValidationResult(metrics=metrics, data={"group_returns": group_returns, "cumulative": group_returns.cumsum()})

    def calc_group_returns(self, pred_label: pd.DataFrame) -> pd.DataFrame:
        df = pred_label.copy()
        if self.reverse:
            df[self.canonical_score_col] *= -1
        df = df.sort_values(self.canonical_score_col, ascending=False)
        df_drop = df.dropna(subset=[self.canonical_score_col])

        grouped = pd.DataFrame(
            {
                f"Group{i + 1}": df_drop.groupby(level="datetime", group_keys=False)[self.canonical_label_col].apply(
                    lambda x, i=i: x[len(x) // self.groups * i : len(x) // self.groups * (i + 1)].mean()
                )
                for i in range(self.groups)
            }
        )
        grouped.index = pd.to_datetime(grouped.index)
        grouped["long-short"] = grouped["Group1"] - grouped[f"Group{self.groups}"]
        grouped["long-average"] = grouped["Group1"] - df.groupby(level="datetime", group_keys=False)[
            self.canonical_label_col
        ].mean()
        return grouped.dropna(how="all")

    def calc_metrics(self, group_returns: pd.DataFrame) -> dict[str, object]:
        metrics: dict[str, object] = {
            "mean_return_by_group": self.json_safe(group_returns.mean().to_dict()),
            "cumulative_return_by_group": self.json_safe(group_returns.sum().to_dict()),
        }
        group_cols = [f"Group{i + 1}" for i in range(self.groups)]
        mean_by_group = group_returns[group_cols].mean()
        metrics["monotonicity"] = {
            "is_decreasing_from_group1": bool(mean_by_group.is_monotonic_decreasing),
            "top_minus_bottom_mean": self.json_safe(mean_by_group.iloc[0] - mean_by_group.iloc[-1]),
        }
        for col in ["long-short", "long-average"]:
            if col in group_returns:
                metrics[f"{col}_risk"] = self.risk_to_dict(risk_analysis(group_returns[col].dropna(), freq=self.freq))
        return metrics


__all__ = ["GroupReturnAnalyzer"]


if __name__ == "__main__":
    pred_label = load_local_test_pred_label()
    analyzer = GroupReturnAnalyzer(score_col=DEFAULT_LOCAL_FEATURE_COL, label_col=DEFAULT_LOCAL_LABEL_COL, groups=5)
    result = analyzer.validate(pred_label)
    print(f"Loaded local test data: {pred_label.shape}")
    print("Group return metrics:")
    print(result.metrics)
    print("Group returns head:")
    print(result.data["group_returns"].head())
