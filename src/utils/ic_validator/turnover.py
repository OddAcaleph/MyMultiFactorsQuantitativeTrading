"""Top/bottom score turnover validation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from .base import DEFAULT_LOCAL_FEATURE_COL, DEFAULT_LOCAL_LABEL_COL, BaseICValidationStep, ValidationResult, load_local_test_pred_label
except ImportError:  # pragma: no cover - enables `python turnover.py`
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from base import DEFAULT_LOCAL_FEATURE_COL, DEFAULT_LOCAL_LABEL_COL, BaseICValidationStep, ValidationResult, load_local_test_pred_label


class TurnoverAnalyzer(BaseICValidationStep):
    """Compute top/bottom score turnover using Qlib's model report formula."""

    def __init__(self, groups: int = 5, lag: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        if groups < 2:
            raise ValueError("groups must be >= 2")
        if lag <= 0:
            raise ValueError("lag must be positive")
        self.groups = groups
        self.lag = lag

    def validate(self, pred_label: pd.DataFrame) -> ValidationResult:
        df = self.prepare_pred_label(pred_label)
        turnover = self.calc_turnover(df)
        metrics = {
            "mean_turnover": self.json_safe(turnover.mean().to_dict()),
            "std_turnover": self.json_safe(turnover.std(ddof=1).to_dict()),
            "latest_turnover": self.json_safe(turnover.iloc[-1].to_dict()) if not turnover.empty else {},
            "count": int(len(turnover)),
        }
        return ValidationResult(metrics=metrics, data={"turnover": turnover})

    def calc_turnover(self, pred_label: pd.DataFrame) -> pd.DataFrame:
        pred = pred_label.copy()
        pred["score_last"] = pred.groupby(level="instrument", group_keys=False)[self.canonical_score_col].shift(self.lag)

        def _turnover(x: pd.DataFrame, largest: bool) -> float:
            bucket_size = len(x) // self.groups
            if bucket_size <= 0 or x["score_last"].dropna().empty:
                return float("nan")
            current = x.nlargest(bucket_size, columns=self.canonical_score_col) if largest else x.nsmallest(
                bucket_size, columns=self.canonical_score_col
            )
            previous = x.nlargest(bucket_size, columns="score_last") if largest else x.nsmallest(bucket_size, columns="score_last")
            return 1.0 - current.index.isin(previous.index).sum() / bucket_size

        top = pred.groupby(level="datetime", group_keys=False).apply(lambda x: _turnover(x, largest=True))
        bottom = pred.groupby(level="datetime", group_keys=False).apply(lambda x: _turnover(x, largest=False))
        return pd.DataFrame({"Top": top, "Bottom": bottom}).dropna(how="all")


__all__ = ["TurnoverAnalyzer"]


if __name__ == "__main__":
    pred_label = load_local_test_pred_label()
    analyzer = TurnoverAnalyzer(score_col=DEFAULT_LOCAL_FEATURE_COL, label_col=DEFAULT_LOCAL_LABEL_COL, groups=5, lag=1)
    result = analyzer.validate(pred_label)
    print(f"Loaded local test data: {pred_label.shape}")
    print("Turnover metrics:")
    print(result.metrics)
    print("Turnover head:")
    print(result.data["turnover"].head())
