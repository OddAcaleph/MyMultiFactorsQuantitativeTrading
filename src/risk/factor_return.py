"""Daily factor return estimation via weighted least squares (WLS).

For each trading day, estimates factor returns ``f_t`` from the
cross-sectional regression:

    r_{i,t} = alpha_t + X_{i,t} f_t + epsilon_{i,t}

where weights are proportional to sqrt(20-day average amount) to downweight
noisy small-cap stocks without letting mega-caps dominate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FactorReturnResult:
    """Result of daily factor return estimation."""

    factor_returns: pd.DataFrame
    residual_returns: pd.DataFrame
    r_squared: pd.Series
    n_stocks: pd.Series
    start_date: int
    end_date: int


class FactorReturnEstimator:
    """Estimate daily factor returns via WLS cross-sectional regression.

    Parameters
    ----------
    config
        Risk model config dictionary.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        fr_cfg = config.get("factor_return", {})
        self.method = fr_cfg.get("method", "wls")
        self.weight_type = fr_cfg.get("weight", "sqrt_amount_20")
        self.min_stocks = int(fr_cfg.get("min_stocks", 100))
        self.amount_cap_multiple = float(fr_cfg.get("amount_cap_multiple", 10.0))

    def estimate(
        self,
        exposures: pd.DataFrame,
        returns: pd.Series | pd.DataFrame,
        weights: pd.Series | None = None,
    ) -> FactorReturnResult:
        """Estimate daily factor returns.

        Parameters
        ----------
        exposures
            DataFrame with columns ``trade_date``, ``ts_code``, and factor
            exposure columns (style + industry).
        returns
            Daily stock returns.  Either a Series aligned with *exposures*
            or a DataFrame with ``trade_date``, ``ts_code``, ``ret``.
        weights
            Optional regression weights.  If ``None``, weights are computed
            from 20-day average amount according to config.
        """

        df = exposures.copy()

        if isinstance(returns, pd.DataFrame):
            ret_df = returns[["trade_date", "ts_code", "ret"]].copy()
            df = df.merge(ret_df, on=["trade_date", "ts_code"], how="inner")
        elif isinstance(returns, pd.Series):
            if isinstance(returns.index, pd.MultiIndex):
                ret_df = returns.rename("ret").reset_index()
                df = df.merge(ret_df, on=["trade_date", "ts_code"], how="inner")
            else:
                # Assume series is aligned with df index
                df["ret"] = returns.reindex(df.index).values
        else:
            raise TypeError("returns must be a DataFrame or Series")

        if weights is None and self.weight_type == "equal":
            w = pd.Series(1.0, index=df.index)
        elif weights is None:
            w = self._compute_weights(df)
        else:
            w = weights

        df["_weight"] = w.reindex(df.index).fillna(0.0)

        factor_cols = [
            c for c in df.columns
            if c not in ("trade_date", "ts_code", "ret", "_weight")
        ]

        all_factor_returns = []
        all_residuals = []
        all_r2 = []
        all_n = []

        for td, group in df.groupby("trade_date"):
            # Drop factor columns that are all-NaN on this date
            active_factors = [
                col for col in factor_cols
                if group[col].notna().any()
            ]

            valid = group["ret"].notna() & (group["_weight"] > 0)
            for col in active_factors:
                valid &= group[col].notna()

            g = group[valid].copy()
            n = len(g)
            all_n.append((td, n))

            if n < self.min_stocks or n <= len(active_factors) + 1:
                fr_row = {"trade_date": td, "INTERCEPT": np.nan}
                for col in factor_cols:
                    fr_row[col] = np.nan
                all_factor_returns.append(fr_row)
                resid = group[["trade_date", "ts_code"]].copy()
                resid["residual"] = np.nan
                all_residuals.append(resid)
                all_r2.append((td, np.nan))
                continue

            X = np.column_stack([np.ones(n), g[active_factors].values])
            y = g["ret"].values
            w_vec = g["_weight"].values

            sqrt_w = np.sqrt(np.clip(w_vec, 0, None))
            X_w = X * sqrt_w[:, None]
            y_w = y * sqrt_w

            try:
                beta, residuals_w, rank, sv = np.linalg.lstsq(X_w, y_w, rcond=None)
            except np.linalg.LinAlgError:
                fr_row = {"trade_date": td, "INTERCEPT": np.nan}
                for col in factor_cols:
                    fr_row[col] = np.nan
                all_factor_returns.append(fr_row)
                resid = group[["trade_date", "ts_code"]].copy()
                resid["residual"] = np.nan
                all_residuals.append(resid)
                all_r2.append((td, np.nan))
                continue

            y_pred = X @ beta
            resid_vals = y - y_pred
            ss_res = np.sum(resid_vals ** 2 * w_vec)
            ss_tot = np.sum((y - np.average(y, weights=w_vec)) ** 2 * w_vec)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

            fr_row = {"trade_date": td, "INTERCEPT": float(beta[0])}
            for col in factor_cols:
                fr_row[col] = np.nan
            for i, col in enumerate(active_factors):
                fr_row[col] = float(beta[i + 1])
            all_factor_returns.append(fr_row)

            resid = pd.DataFrame({
                "trade_date": td,
                "ts_code": g["ts_code"].values,
                "residual": resid_vals,
            })
            all_residuals.append(resid)
            all_r2.append((td, r2))

        factor_returns_df = pd.DataFrame(all_factor_returns)
        factor_returns_df = factor_returns_df.sort_values("trade_date").reset_index(drop=True)

        residuals_df = pd.concat(all_residuals, ignore_index=True)
        residuals_df = residuals_df.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)

        r2_series = pd.Series(dict(all_r2)).sort_index()
        n_series = pd.Series(dict(all_n)).sort_index()

        return FactorReturnResult(
            factor_returns=factor_returns_df,
            residual_returns=residuals_df,
            r_squared=r2_series,
            n_stocks=n_series,
            start_date=int(factor_returns_df["trade_date"].min()) if len(factor_returns_df) else 0,
            end_date=int(factor_returns_df["trade_date"].max()) if len(factor_returns_df) else 0,
        )

    def _compute_weights(self, df: pd.DataFrame) -> pd.Series:
        """Compute WLS weights from 20-day average amount."""

        if "amount" in df.columns:
            amount = df["amount"]
        else:
            return pd.Series(1.0, index=df.index)

        amount_sorted = df[["ts_code", "trade_date", "amount"]].sort_values(["ts_code", "trade_date"])
        avg_amount = (
            amount_sorted.groupby("ts_code")["amount"]
            .rolling(window=20, min_periods=5)
            .mean()
            .reset_index(level=0, drop=True)
        )
        avg_amount = avg_amount.reindex(df.index)

        if self.weight_type == "sqrt_amount_20":
            capped = avg_amount.clip(upper=avg_amount.quantile(0.99) * self.amount_cap_multiple / 10.0)
            w = np.sqrt(capped.replace(0, np.nan)).fillna(0.0)
        elif self.weight_type == "amount_20":
            w = avg_amount.clip(lower=0).fillna(0.0)
        else:
            w = pd.Series(1.0, index=df.index)

        return w
