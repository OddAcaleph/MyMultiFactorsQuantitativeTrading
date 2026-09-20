"""Exposure validator.

Checks risk factor exposures for missing values, distributional properties,
and cross-sectional standardization quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass
class ExposureValidationResult:
    """Result of exposure validation."""

    style_factor_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    industry_factor_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    missing_rates: dict[str, float] = field(default_factory=dict)
    passed: bool = True
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "issues": self.issues,
            "style_factor_stats": self.style_factor_stats,
            "industry_factor_stats": self.industry_factor_stats,
            "missing_rates": self.missing_rates,
        }


class ExposureValidator:
    """Validate risk factor exposures.

    Parameters
    ----------
    config
        Risk model config dictionary.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    def validate(self, exposures: pd.DataFrame) -> ExposureValidationResult:
        """Run all exposure validation checks.

        Parameters
        ----------
        exposures
            Long-format exposure DataFrame with ``trade_date``, ``ts_code``,
            and factor columns.
        """

        result = ExposureValidationResult()

        style_cols = self._identify_style_factors(exposures)
        industry_cols = [c for c in exposures.columns if c.startswith("L1_") or c.startswith("industry_")]

        for col in style_cols:
            stats = self._style_factor_stats(exposures, col)
            result.style_factor_stats[col] = stats
            result.missing_rates[col] = float(exposures[col].isna().mean())

            if abs(stats["mean_overall"]) > 0.5:
                result.issues.append(f"Style factor {col}: overall mean {stats['mean_overall']:.4f} deviates from 0")
                result.passed = False
            if abs(stats["std_overall"] - 1.0) > 0.5:
                result.issues.append(f"Style factor {col}: overall std {stats['std_overall']:.4f} deviates from 1")
                result.passed = False
            if result.missing_rates[col] > 0.5:
                result.issues.append(f"Style factor {col}: missing rate {result.missing_rates[col]:.2%} > 50%")
                result.passed = False

        for col in industry_cols:
            stats = self._industry_factor_stats(exposures, col)
            result.industry_factor_stats[col] = stats
            result.missing_rates[col] = float(exposures[col].isna().mean())

        return result

    def _identify_style_factors(self, exposures: pd.DataFrame) -> list[str]:
        """Identify style factor columns (non-industry, non-key)."""

        key_cols = {"trade_date", "ts_code"}
        industry_cols = {
            c for c in exposures.columns
            if c.startswith("L1_") or c.startswith("industry_")
        }
        return [c for c in exposures.columns if c not in key_cols and c not in industry_cols]

    def _style_factor_stats(self, exposures: pd.DataFrame, col: str) -> dict[str, float]:
        """Compute distributional statistics for a style factor."""

        vals = exposures[col].dropna()
        if vals.empty:
            return {
                "mean_overall": np.nan,
                "std_overall": np.nan,
                "min_overall": np.nan,
                "max_overall": np.nan,
                "mean_daily_mean": np.nan,
                "mean_daily_std": np.nan,
            }

        daily_mean = exposures.groupby("trade_date")[col].mean()
        daily_std = exposures.groupby("trade_date")[col].std()

        return {
            "mean_overall": float(vals.mean()),
            "std_overall": float(vals.std()),
            "min_overall": float(vals.min()),
            "max_overall": float(vals.max()),
            "mean_daily_mean": float(daily_mean.mean()),
            "mean_daily_std": float(daily_std.mean()),
        }

    def _industry_factor_stats(self, exposures: pd.DataFrame, col: str) -> dict[str, float]:
        """Compute coverage statistics for an industry factor."""

        vals = exposures[col].dropna()
        if vals.empty:
            return {"mean": np.nan, "coverage": 0.0}

        return {
            "mean": float(vals.mean()),
            "coverage": float((vals > 0).mean()),
        }
