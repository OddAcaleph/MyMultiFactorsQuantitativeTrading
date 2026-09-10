"""Diagnostics collection for portfolio optimization.

Records per-optimization diagnostic data for later analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from optimizer.optimization_result import OptimizationResult


@dataclass
class OptimizationDiagnostics:
    """Collect and export optimization diagnostics.

    Stores one record per trade date with key metrics for analysis.
    """

    records: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        trade_date: int,
        result: OptimizationResult,
        risk_attribution: Any | None = None,
    ) -> None:
        """Record diagnostics for a single optimization run.

        Parameters
        ----------
        trade_date : int
            Trading date (YYYYMMDD).
        result : OptimizationResult
            Optimization result.
        risk_attribution : RiskAttributionResult, optional
            Risk attribution result if available.
        """
        record: dict[str, Any] = {
            "trade_date": trade_date,
            "solver_status": result.solver_status,
            "objective_value": result.objective_value,
            "solve_time_ms": result.solve_time_ms,
            "expected_alpha": result.expected_alpha,
            "portfolio_volatility": result.portfolio_volatility,
            "portfolio_variance": result.portfolio_variance,
            "turnover": result.turnover,
            "alpha_retention": result.alpha_retention,
            "raw_alpha_portfolio": result.raw_alpha_portfolio,
            "n_candidates": result.n_candidates,
            "n_holdings": result.n_holdings,
        }

        # Style exposures
        if len(result.style_exposure) > 0:
            for name, val in result.style_exposure.items():
                record[f"style_{name}"] = float(val)

        # Industry exposure stats
        if len(result.industry_exposure) > 0:
            record["max_industry_weight"] = float(result.industry_exposure.max())
            record["n_industries"] = int((result.industry_exposure > 1e-6).sum())

        # Max single weight
        if len(result.weights) > 0:
            record["max_weight"] = float(result.weights.max())

        # Risk attribution
        if risk_attribution is not None:
            record["factor_risk_pct"] = float(
                risk_attribution.factor_variance / risk_attribution.total_variance
                if risk_attribution.total_variance > 0 else 0.0
            )
            record["specific_risk_pct"] = float(
                risk_attribution.specific_variance / risk_attribution.total_variance
                if risk_attribution.total_variance > 0 else 0.0
            )
            for name, val in risk_attribution.factor_contribution_pct.items():
                record[f"rc_{name}"] = float(val)
            record["rc_specific"] = float(risk_attribution.specific_contribution_pct)

        self.records.append(record)

    def to_dataframe(self) -> pd.DataFrame:
        """Export all records as a DataFrame.

        Returns
        -------
        pd.DataFrame
            One row per trade date, with trade_date as a column.
        """
        if not self.records:
            return pd.DataFrame()
        df = pd.DataFrame(self.records)
        return df

    def summary(self) -> dict[str, float]:
        """Compute summary statistics across all recorded dates.

        Returns
        -------
        dict
            Summary metrics (mean, std, min, max for key fields).
        """
        df = self.to_dataframe()
        if df.empty:
            return {}

        numeric_cols = [
            "expected_alpha", "portfolio_volatility", "turnover",
            "alpha_retention", "n_holdings", "solve_time_ms",
        ]
        summary: dict[str, float] = {}
        for col in numeric_cols:
            if col in df.columns:
                summary[f"{col}_mean"] = float(df[col].mean())
                summary[f"{col}_std"] = float(df[col].std())
                summary[f"{col}_min"] = float(df[col].min())
                summary[f"{col}_max"] = float(df[col].max())

        return summary
