"""Long-short validation through Qlib's evaluation interfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qlib.contrib.evaluate import long_short_backtest, risk_analysis
from qlib.contrib.eva.alpha import calc_long_short_return

try:
    from .base import DEFAULT_LOCAL_FEATURE_COL, DEFAULT_LOCAL_LABEL_COL, BaseICValidationStep, ValidationResult, load_local_test_pred_label
except ImportError:  # pragma: no cover - enables `python long_short_backtest.py`
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from base import DEFAULT_LOCAL_FEATURE_COL, DEFAULT_LOCAL_LABEL_COL, BaseICValidationStep, ValidationResult, load_local_test_pred_label


class LongShortBacktestAnalyzer(BaseICValidationStep):
    """Run long-short analysis.

    When ``use_qlib_exchange`` is true this class delegates to Qlib's
    ``qlib.contrib.evaluate.long_short_backtest``.  For parquet-only research
    data that has no initialized Qlib quote provider, it falls back to the same
    score-sorted long/short label-spread calculation and still uses Qlib
    ``risk_analysis`` for metrics.
    """

    def __init__(
        self,
        topk: int = 50,
        freq: str = "day",
        use_qlib_exchange: bool = False,
        qlib_backtest_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if topk <= 0:
            raise ValueError("topk must be positive")
        self.topk = topk
        self.freq = freq
        self.use_qlib_exchange = use_qlib_exchange
        self.qlib_backtest_kwargs = dict(qlib_backtest_kwargs or {})

    def validate(self, pred_label: pd.DataFrame) -> ValidationResult:
        df = self.prepare_pred_label(pred_label)
        returns, backend, warning = self.calc_long_short_returns(df)
        metrics = {
            "backend": backend,
            "warning": warning,
            "risk_analysis": {
                col: self.risk_to_dict(risk_analysis(returns[col].dropna(), freq=self.freq)) for col in returns.columns
            },
            "mean_return": self.json_safe(returns.mean().to_dict()),
            "cumulative_return": self.json_safe(returns.sum().to_dict()),
        }
        return ValidationResult(metrics=metrics, data={"long_short_returns": returns, "cumulative": returns.cumsum()})

    def calc_long_short_returns(self, pred_label: pd.DataFrame) -> tuple[pd.DataFrame, str, str | None]:
        if self.use_qlib_exchange:
            try:
                qlib_input = pred_label[[self.canonical_score_col]].rename(columns={self.canonical_score_col: "score"})
                result = long_short_backtest(qlib_input, topk=self.topk, **self.qlib_backtest_kwargs)
                return pd.DataFrame(result).sort_index(), "qlib.contrib.evaluate.long_short_backtest", None
            except Exception as exc:  # pragma: no cover - depends on external qlib provider state
                warning = f"Qlib long_short_backtest failed, fallback to label-based calculation: {exc}"
                return self._label_based_long_short(pred_label), "label_based_fallback", warning
        return self._label_based_long_short(pred_label), "label_based_fallback", None

    def _label_based_long_short(self, pred_label: pd.DataFrame) -> pd.DataFrame:
        quantile = min(self.topk / day_count for day_count in pred_label.groupby(level="datetime").size() if day_count > 0)
        if 0 < quantile < 0.5:
            half_long_short, market_average = calc_long_short_return(
                pred_label[self.canonical_score_col],
                pred_label[self.canonical_label_col],
                date_col="datetime",
                quantile=quantile,
                dropna=True,
            )
            # Qlib's public alpha helper returns daily `(long - short) / 2` and
            # market average.  Multiply by 2 to keep the conventional
            # long-short spread in this validator's output.
            return pd.DataFrame(
                {
                    "long_short": half_long_short * 2,
                    "market_average": market_average,
                }
            ).sort_index()

        records: list[dict[str, float | pd.Timestamp]] = []
        for trade_date, day_df in pred_label.groupby(level="datetime", sort=True):
            day_df = day_df.droplevel("datetime").sort_values(self.canonical_score_col, ascending=False)
            if day_df.empty:
                continue
            k = min(self.topk, len(day_df) // 2) if len(day_df) > 1 else 1
            if k <= 0:
                continue
            long_ret = day_df.head(k)[self.canonical_label_col].mean()
            short_leg = -day_df.tail(k)[self.canonical_label_col].mean()
            market = day_df[self.canonical_label_col].mean()
            records.append(
                {
                    "datetime": pd.Timestamp(trade_date),
                    "long": float(long_ret - market) if np.isfinite(long_ret - market) else np.nan,
                    "short": float(short_leg + market) if np.isfinite(short_leg + market) else np.nan,
                    "long_short": float(long_ret + short_leg) if np.isfinite(long_ret + short_leg) else np.nan,
                }
            )
        if not records:
            return pd.DataFrame(columns=["long", "short", "long_short"])
        return pd.DataFrame(records).set_index("datetime").sort_index()


__all__ = ["LongShortBacktestAnalyzer"]


if __name__ == "__main__":
    pred_label = load_local_test_pred_label()
    analyzer = LongShortBacktestAnalyzer(
        score_col=DEFAULT_LOCAL_FEATURE_COL,
        label_col=DEFAULT_LOCAL_LABEL_COL,
        topk=50,
        use_qlib_exchange=False,
    )
    result = analyzer.validate(pred_label)
    print(f"Loaded local test data: {pred_label.shape}")
    print("Long-short metrics:")
    print(result.metrics)
    print("Long-short returns head:")
    print(result.data["long_short_returns"].head())
