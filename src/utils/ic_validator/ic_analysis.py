"""Daily IC and ICIR validation through Qlib alpha evaluation APIs."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd
from qlib.contrib.evaluate import risk_analysis
from qlib.contrib.eva.alpha import calc_ic

try:
    from .base import DEFAULT_LOCAL_FEATURE_COL, DEFAULT_LOCAL_LABEL_COL, BaseICValidationStep, ValidationResult, load_local_test_pred_label
except ImportError:  # pragma: no cover - enables `python ic_analysis.py`
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from base import DEFAULT_LOCAL_FEATURE_COL, DEFAULT_LOCAL_LABEL_COL, BaseICValidationStep, ValidationResult, load_local_test_pred_label


class DailyICAnalyzer(BaseICValidationStep):
    """Compute daily IC / Rank IC and summary metrics.

    Daily IC and Rank IC are calculated by Qlib's public
    ``qlib.contrib.eva.alpha.calc_ic``.  The returned series are then summarized
    and passed to Qlib ``risk_analysis`` for consistency with other reports.
    """

    METHOD_MAPPING = {"IC": "pearson", "Rank IC": "spearman"}

    def __init__(self, methods: Sequence[str] = ("IC", "Rank IC"), freq: str = "day", **kwargs) -> None:
        super().__init__(**kwargs)
        unknown = [m for m in methods if m not in self.METHOD_MAPPING]
        if unknown:
            raise ValueError(f"Unsupported IC methods: {unknown}; supported={list(self.METHOD_MAPPING)}")
        self.methods = tuple(methods)
        self.freq = freq

    def validate(self, pred_label: pd.DataFrame) -> ValidationResult:
        df = self.prepare_pred_label(pred_label)
        ic_df = self.calc_daily_ic(df)
        monthly_ic = self.calc_monthly_ic(ic_df)
        metrics = {
            "daily": {col: self._series_stats(ic_df[col]) for col in ic_df.columns},
            "risk_analysis": {col: self.risk_to_dict(risk_analysis(ic_df[col].dropna(), freq=self.freq)) for col in ic_df.columns},
        }
        return ValidationResult(metrics=metrics, data={"daily_ic": ic_df, "monthly_ic": monthly_ic})

    def calc_daily_ic(self, pred_label: pd.DataFrame) -> pd.DataFrame:
        ic, rank_ic = calc_ic(
            pred_label[self.canonical_score_col],
            pred_label[self.canonical_label_col],
            date_col="datetime",
            dropna=False,
        )
        series_map = {"IC": ic.rename("IC"), "Rank IC": rank_ic.rename("Rank IC")}
        return pd.concat([series_map[method] for method in self.methods], axis=1)

    @staticmethod
    def calc_monthly_ic(ic_df: pd.DataFrame) -> pd.DataFrame:
        if ic_df.empty:
            return pd.DataFrame(columns=ic_df.columns)
        month_index = pd.to_datetime(ic_df.index).strftime("%Y%m")
        monthly = ic_df.groupby(month_index, group_keys=False).mean()
        monthly.index = pd.MultiIndex.from_arrays(
            [monthly.index.str.slice(0, 4), monthly.index.str.slice(4, 6)], names=["year", "month"]
        )
        return monthly

    @staticmethod
    def _series_stats(series: pd.Series) -> dict[str, float | int | None]:
        clean = series.dropna()
        if clean.empty:
            return {"ic_mean": None, "ic_std": None, "icir": None, "positive_rate": None, "count": 0}
        mean = clean.mean()
        std = clean.std(ddof=1)
        return {
            "ic_mean": float(mean),
            "ic_std": float(std) if pd.notna(std) else None,
            "icir": float(mean / std) if pd.notna(std) and std != 0 else None,
            "positive_rate": float((clean > 0).mean()),
            "count": int(clean.count()),
        }


__all__ = ["DailyICAnalyzer"]


if __name__ == "__main__":
    pred_label = load_local_test_pred_label()
    analyzer = DailyICAnalyzer(score_col=DEFAULT_LOCAL_FEATURE_COL, label_col=DEFAULT_LOCAL_LABEL_COL)
    result = analyzer.validate(pred_label)
    print(f"Loaded local test data: {pred_label.shape}")
    print("Daily IC metrics:")
    print(result.metrics)
    print("Daily IC head:")
    print(result.data["daily_ic"].head())
