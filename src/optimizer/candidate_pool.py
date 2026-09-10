"""Candidate stock pool builder for optimization."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd


class CandidatePoolBuilder:
    """Build the candidate stock universe for portfolio optimization.

    Applies filters (ST, suspend, new stock, liquidity) and keeps current
    holdings in the pool so the optimizer can properly model risk and turnover.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = config or {}
        self.pool_size: int = int(config.get("pool_size", 100))
        self.min_avg_amount_20d: float = float(config.get("min_avg_amount_20d", 5000))
        self.filter_st: bool = bool(config.get("filter_st", True))
        self.filter_suspend: bool = bool(config.get("filter_suspend", True))
        self.filter_new_stock_days: int = int(config.get("filter_new_stock_days", 60))

    def build(
        self,
        alpha: pd.Series,
        market_data: pd.DataFrame | None = None,
        current_holdings: pd.Series | None = None,
    ) -> pd.Index:
        """Build candidate pool.

        Parameters
        ----------
        alpha : pd.Series
            Standardized alpha scores, index=ts_code.
        market_data : pd.DataFrame, optional
            Per-stock market status for the day. Expected columns:
            ``is_st``, ``is_suspended``, ``list_days``, ``avg_amount_20d``.
            Index must be ts_code. If None, no market filters are applied.
        current_holdings : pd.Series, optional
            Current position weights, index=ts_code.  Holdings with weight > 0
            are always retained in the candidate pool.

        Returns
        -------
        pd.Index
            Candidate stock codes.
        """
        candidates = alpha.dropna().index

        if market_data is not None and len(market_data) > 0:
            mask = pd.Series(True, index=candidates)
            common = candidates.intersection(market_data.index)

            if self.filter_st and "is_st" in market_data.columns:
                mask.loc[common] &= ~market_data.loc[common, "is_st"].astype(bool)

            if self.filter_suspend and "is_suspended" in market_data.columns:
                mask.loc[common] &= ~market_data.loc[common, "is_suspended"].astype(bool)

            if self.filter_new_stock_days > 0 and "list_days" in market_data.columns:
                mask.loc[common] &= (
                    market_data.loc[common, "list_days"] >= self.filter_new_stock_days
                )

            if self.min_avg_amount_20d > 0 and "avg_amount_20d" in market_data.columns:
                mask.loc[common] &= (
                    market_data.loc[common, "avg_amount_20d"] >= self.min_avg_amount_20d
                )

            candidates = mask[mask].index

        ranked = alpha.loc[candidates].sort_values(ascending=False)
        top_n = ranked.head(self.pool_size).index

        if current_holdings is not None and len(current_holdings) > 0:
            held = current_holdings[current_holdings > 0].index
            top_n = top_n.union(held)

        return top_n
