"""Cross-sectional winsorization utilities."""

from __future__ import annotations

import pandas as pd


def winsorize(x: pd.Series) -> pd.Series:
    """Clip a cross-sectional series to its 1% and 99% quantiles."""

    lower = x.quantile(0.01)
    upper = x.quantile(0.99)
    return x.clip(lower, upper)


def winsorize_by_trade_date(df: pd.DataFrame, factor: str, date_column: str = "trade_date") -> pd.Series:
    """Winsorize one factor by ``trade_date`` cross-section."""

    return df.groupby(date_column)[factor].transform(winsorize)
