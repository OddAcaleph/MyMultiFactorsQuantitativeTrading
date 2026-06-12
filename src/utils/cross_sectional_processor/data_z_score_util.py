"""Cross-sectional z-score standardization utilities."""

from __future__ import annotations

import pandas as pd


def zscore(x: pd.Series) -> pd.Series:
    """Standardize a cross-sectional series with z-score."""

    std = x.std()
    if pd.isna(std) or std == 0:
        return x - x.mean()
    return (x - x.mean()) / std


def zscore_by_trade_date(df: pd.DataFrame, factor: str, date_column: str = "trade_date") -> pd.Series:
    """Apply z-score standardization to one factor by ``trade_date`` cross-section."""

    return df.groupby(date_column)[factor].transform(zscore)
