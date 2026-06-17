"""XGBoost prediction IC metrics calculator.

This module is intentionally lightweight and only depends on pandas/numpy so it
can evaluate saved ``pred_test.parquet`` files without initializing qlib.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from .base import BaseICValidationStep, DATETIME_LEVEL
except ImportError:  # pragma: no cover - enables `python xgb_ic_metrics.py`
    sys.path.append(str(Path(__file__).resolve().parent))
    from base import BaseICValidationStep, DATETIME_LEVEL


@dataclass(frozen=True)
class XGBICMetricsResult:
    """Result returned by :class:`XGBModelICCalculator`."""

    metrics: dict[str, Any]
    daily_ic: pd.DataFrame


class XGBModelICCalculator:
    """Calculate IC / ICIR / RankIC from an XGBoost prediction parquet.

    The expected input is the parquet saved by ``XGBoostTrainer`` / inferencer,
    containing a ``pred`` column and one model label column such as
    ``label_1d`` or ``label_5d``.  The label column can be passed explicitly; if
    omitted, it is inferred from the parquet schema.
    """

    def __init__(
        self,
        pred_col: str = "pred",
        label_col: str | None = None,
        date_col: str = DATETIME_LEVEL,
        min_periods: int = 2,
    ) -> None:
        self.pred_col = pred_col
        self.label_col = label_col
        self.date_col = date_col
        self.min_periods = min_periods

    def calculate_from_parquet(self, parquet_path: str | Path, output_dir: str | Path | None = None) -> XGBICMetricsResult:
        """Load a prediction parquet, calculate metrics and optionally save outputs."""

        parquet_path = Path(parquet_path)
        df = self.load_pred_label(parquet_path)
        result = self.calculate(df)
        if output_dir is not None:
            self.save_result(result, output_dir)
        return result

    def load_pred_label(self, parquet_path: str | Path) -> pd.DataFrame:
        """Read parquet and return cleaned data with columns ``pred`` and label."""

        parquet_path = Path(parquet_path)
        if not parquet_path.exists():
            raise FileNotFoundError(f"prediction parquet does not exist: {parquet_path}")

        df = pd.read_parquet(parquet_path)
        if self.pred_col not in df.columns:
            raise KeyError(f"prediction parquet missing required pred column: {self.pred_col}")

        label_col = self._infer_label_col(df)
        clean = self._normalize_datetime(df.copy())[[self.pred_col, label_col]]
        clean = clean.replace([np.inf, -np.inf], np.nan).dropna(subset=[self.pred_col, label_col])
        if clean.empty:
            raise ValueError(f"No valid non-null rows for pred={self.pred_col}, label={label_col}")
        self.label_col = label_col
        return clean.sort_index()

    def calculate(self, pred_label: pd.DataFrame) -> XGBICMetricsResult:
        """Calculate daily IC/RankIC and overall non-annualized ICIR metrics."""

        if self.pred_col not in pred_label.columns:
            raise KeyError(f"pred_label missing required pred column: {self.pred_col}")

        label_col = self.label_col or self._infer_label_col(pred_label)
        if label_col not in pred_label.columns:
            raise KeyError(f"pred_label missing label column: {label_col}")

        df = self._normalize_datetime(pred_label.copy())[[self.pred_col, label_col]]
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=[self.pred_col, label_col])
        if df.empty:
            raise ValueError(f"No valid non-null rows for pred={self.pred_col}, label={label_col}")

        daily_ic = self._calc_daily_ic(df, label_col)
        ic_stats = self._series_stats(daily_ic["IC"])
        rank_ic_stats = self._series_stats(daily_ic["RankIC"])
        date_index = df.index.get_level_values(self.date_col)
        metrics = {
            "pred_col": self.pred_col,
            "label_col": label_col,
            "row_count": int(len(df)),
            "date_count": int(date_index.nunique()),
            "start_date": BaseICValidationStep.json_safe(date_index.min()),
            "end_date": BaseICValidationStep.json_safe(date_index.max()),
            "ic": ic_stats["mean"],
            "ic_std": ic_stats["std"],
            "icir": ic_stats["icir"],
            "rank_ic": rank_ic_stats["mean"],
            "rank_ic_std": rank_ic_stats["std"],
            "rank_icir": rank_ic_stats["icir"],
            "valid_ic_days": ic_stats["count"],
            "valid_rank_ic_days": rank_ic_stats["count"],
        }
        return XGBICMetricsResult(metrics=metrics, daily_ic=daily_ic)

    def save_result(self, result: XGBICMetricsResult, output_dir: str | Path) -> None:
        """Save ``xgb_ic_metrics.json`` and ``xgb_daily_ic.csv`` under output_dir."""

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / "xgb_ic_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(BaseICValidationStep.json_safe(result.metrics), f, ensure_ascii=False, indent=2)
        result.daily_ic.to_csv(output_path / "xgb_daily_ic.csv")

    def _infer_label_col(self, df: pd.DataFrame) -> str:
        if self.label_col is not None:
            if self.label_col not in df.columns:
                raise KeyError(f"specified label column not found: {self.label_col}")
            return self.label_col

        candidates = [col for col in df.columns if col != self.pred_col]
        label_like = [col for col in candidates if str(col).startswith("label")]
        numeric_candidates = [col for col in candidates if pd.api.types.is_numeric_dtype(df[col])]

        if len(label_like) == 1:
            return str(label_like[0])
        if len(numeric_candidates) == 1:
            return str(numeric_candidates[0])
        if len(label_like) > 1:
            raise ValueError(f"Multiple label-like columns found, please specify label_col: {label_like}")
        raise ValueError(f"Unable to infer label column from columns={list(df.columns)}; please specify label_col")

    def _normalize_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.index, pd.MultiIndex):
            names = list(df.index.names)
            if self.date_col in names:
                if names[0] != self.date_col:
                    other_levels = [name for name in names if name != self.date_col]
                    df = df.reorder_levels([self.date_col, *other_levels])
                df.index = df.index.set_levels(pd.to_datetime(df.index.levels[0]), level=0)
                return df.sort_index()

            for i in range(df.index.nlevels):
                values = df.index.get_level_values(i)
                if pd.api.types.is_datetime64_any_dtype(values):
                    names[i] = self.date_col
                    df.index = df.index.set_names(names)
                    if i != 0:
                        df = df.reorder_levels([self.date_col, *[name for name in names if name != self.date_col]])
                    return df.sort_index()

        if self.date_col in df.columns:
            df[self.date_col] = pd.to_datetime(df[self.date_col])
            return df.set_index(self.date_col, append=True).reorder_levels([self.date_col, *df.index.names]).sort_index()
        if "trade_date" in df.columns:
            df[self.date_col] = pd.to_datetime(df["trade_date"].astype(str))
            return df.set_index(self.date_col, append=True).reorder_levels([self.date_col, *df.index.names]).sort_index()

        if pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)
            df.index.name = self.date_col
            return df.sort_index()

        raise ValueError(f"Unable to locate datetime level/column '{self.date_col}' in prediction data")

    def _calc_daily_ic(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for dt, group in df.groupby(level=self.date_col, sort=True):
            if len(group) < self.min_periods:
                ic = np.nan
                rank_ic = np.nan
            elif group[self.pred_col].nunique(dropna=True) < 2 or group[label_col].nunique(dropna=True) < 2:
                ic = np.nan
                rank_ic = np.nan
            else:
                ic = group[self.pred_col].corr(group[label_col], method="pearson")
                rank_ic = group[self.pred_col].corr(group[label_col], method="spearman")
            rows.append({self.date_col: dt, "IC": ic, "RankIC": rank_ic, "count": int(len(group))})
        return pd.DataFrame(rows).set_index(self.date_col)

    @staticmethod
    def _series_stats(series: pd.Series) -> dict[str, float | int | None]:
        clean = series.dropna()
        if clean.empty:
            return {"mean": None, "std": None, "icir": None, "count": 0}
        mean = clean.mean()
        std = clean.std(ddof=1)
        return {
            "mean": float(mean),
            "std": float(std) if pd.notna(std) else None,
            "icir": float(mean / std) if pd.notna(std) and std != 0 else None,
            "count": int(clean.count()),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate IC / ICIR / RankIC for XGBoost pred_test.parquet.")
    parser.add_argument("pred_path", help="XGBoost prediction parquet path, e.g. .../train_outputs/pred_test.parquet")
    parser.add_argument("--pred-col", default="pred", help="预测列名，默认 pred")
    parser.add_argument("--label-col", default=None, help="标签列名；不传则自动从 parquet 列中推断")
    parser.add_argument("--date-col", default=DATETIME_LEVEL, help="日期索引层/列名，默认 datetime")
    parser.add_argument("--output-dir", default=None, help="可选输出目录；设置后保存 json/csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calculator = XGBModelICCalculator(pred_col=args.pred_col, label_col=args.label_col, date_col=args.date_col)
    result = calculator.calculate_from_parquet(args.pred_path, output_dir=args.output_dir)
    print(json.dumps(BaseICValidationStep.json_safe(result.metrics), ensure_ascii=False, indent=2))


__all__ = ["XGBICMetricsResult", "XGBModelICCalculator"]


if __name__ == "__main__":
    main()
