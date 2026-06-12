"""Score distribution diagnostics for feature validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .base import DEFAULT_LOCAL_FEATURE_COL, DEFAULT_LOCAL_LABEL_COL, BaseICValidationStep, ValidationResult, load_local_test_pred_label
except ImportError:  # pragma: no cover - enables `python score_distribution.py`
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from base import DEFAULT_LOCAL_FEATURE_COL, DEFAULT_LOCAL_LABEL_COL, BaseICValidationStep, ValidationResult, load_local_test_pred_label


class ScoreDistributionAnalyzer(BaseICValidationStep):
    """Summarize overall and daily score distribution."""

    def __init__(self, bins: int = 20, **kwargs) -> None:
        super().__init__(**kwargs)
        if bins <= 0:
            raise ValueError("bins must be positive")
        self.bins = bins

    def validate(self, pred_label: pd.DataFrame) -> ValidationResult:
        df = self.prepare_pred_label(pred_label)
        score = df[self.canonical_score_col].dropna()
        overall = score.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        daily = df.groupby(level="datetime")[self.canonical_score_col].describe()
        histogram = self._histogram(score)
        metrics = {
            "overall": self.json_safe(overall.to_dict()),
            "daily_mean_summary": self.json_safe(daily["mean"].describe().to_dict()) if not daily.empty else {},
            "daily_std_summary": self.json_safe(daily["std"].describe().to_dict()) if not daily.empty else {},
        }
        return ValidationResult(metrics=metrics, data={"daily_distribution": daily, "histogram": histogram})

    def _histogram(self, score: pd.Series) -> pd.DataFrame:
        if score.empty:
            return pd.DataFrame(columns=["left", "right", "count"])
        counts = pd.cut(score, bins=self.bins).value_counts(sort=False)
        return pd.DataFrame(
            {
                "left": [interval.left for interval in counts.index],
                "right": [interval.right for interval in counts.index],
                "count": counts.to_numpy(),
            }
        )


__all__ = ["ScoreDistributionAnalyzer"]


if __name__ == "__main__":
    pred_label = load_local_test_pred_label()
    analyzer = ScoreDistributionAnalyzer(score_col=DEFAULT_LOCAL_FEATURE_COL, label_col=DEFAULT_LOCAL_LABEL_COL, bins=20)
    result = analyzer.validate(pred_label)
    print(f"Loaded local test data: {pred_label.shape}")
    print("Score distribution metrics:")
    print(result.metrics)
    print("Histogram head:")
    print(result.data["histogram"].head())
