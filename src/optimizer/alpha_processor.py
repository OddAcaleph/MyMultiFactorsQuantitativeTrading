"""Alpha signal processing — cross-sectional winsorization and standardization."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


class AlphaProcessor:
    """Cross-sectional alpha standardization.

    Pipeline: drop NaN → winsorize → z-score / rank standardize.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = config or {}
        self.winsorize_quantile: float = float(config.get("winsorize_quantile", 0.01))
        self.method: str = config.get("method", "zscore")  # zscore | rank
        self.do_winsorize: bool = bool(config.get("winsorize", True))

    def process(self, alpha: pd.Series) -> pd.Series:
        """Standardize a cross-section of alpha predictions.

        Parameters
        ----------
        alpha : pd.Series
            Raw alpha predictions, index=ts_code.

        Returns
        -------
        pd.Series
            Standardized alpha scores with NaN rows dropped.
        """
        s = alpha.dropna()
        if len(s) == 0:
            return s

        if self.do_winsorize and self.winsorize_quantile > 0:
            lo = s.quantile(self.winsorize_quantile)
            hi = s.quantile(1.0 - self.winsorize_quantile)
            s = s.clip(lower=lo, upper=hi)

        if self.method == "rank":
            s = s.rank(pct=True)
            s = (s - s.mean()) / s.std(ddof=0)
        else:  # zscore
            s = (s - s.mean()) / s.std(ddof=0)

        return s
