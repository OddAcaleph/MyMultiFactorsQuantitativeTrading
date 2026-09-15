"""Alpha signal processing — cross-sectional winsorization, standardization, and neutralization."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


class AlphaProcessor:
    """Cross-sectional alpha standardization and neutralization.

    Pipeline: drop NaN → winsorize → industry neutral → style neutral → z-score / rank.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = config or {}
        self.winsorize_quantile: float = float(config.get("winsorize_quantile", 0.01))
        self.method: str = config.get("method", "zscore")  # zscore | rank
        self.do_winsorize: bool = bool(config.get("winsorize", True))
        self.industry_neutral: bool = bool(config.get("industry_neutral", False))
        self.style_neutral_factors: list[str] = list(config.get("style_neutral_factors", []))
        # Time-series smoothing (EMA) of alpha scores per stock
        # Applied BEFORE cross-sectional standardization.
        # Reduces turnover by dampening day-to-day alpha noise.
        self.smooth_span: int = int(config.get("smooth_span", 0))  # 0 = disabled
        self._alpha_history: dict[str, float] = {}  # ts_code -> ema_value

    def process(
        self,
        alpha: pd.Series,
        industry_map: pd.Series | None = None,
        style_exposures: pd.DataFrame | None = None,
    ) -> pd.Series:
        """Standardize a cross-section of alpha predictions.

        Parameters
        ----------
        alpha : pd.Series
            Raw alpha predictions, index=ts_code.
        industry_map : pd.Series, optional
            Industry name for each stock (index=ts_code, value=industry).
            Required when industry_neutral=True.
        style_exposures : pd.DataFrame, optional
            Style factor exposures (index=ts_code, columns=factor names).
            Required when style_neutral_factors is non-empty.

        Returns
        -------
        pd.Series
            Standardized alpha scores with NaN rows dropped.
        """
        s = alpha.dropna()
        if len(s) == 0:
            return s

        # Time-series EMA smoothing (per-stock)
        if self.smooth_span > 0 and self._alpha_history:
            alpha_series = s.copy()
            common = alpha_series.index.intersection(list(self._alpha_history.keys()))
            if len(common) > 0:
                prev_vals = pd.Series(
                    [self._alpha_history[c] for c in common],
                    index=common,
                )
                # EMA: alpha_t = (1 - 2/(span+1)) * alpha_{t-1} + 2/(span+1) * x_t
                decay = 2.0 / (self.smooth_span + 1)
                alpha_series.loc[common] = (
                    (1.0 - decay) * prev_vals + decay * alpha_series.loc[common]
                )
            s = alpha_series

        # Update history with current (possibly smoothed) values
        for code, val in s.items():
            self._alpha_history[code] = val

        if self.do_winsorize and self.winsorize_quantile > 0:
            lo = s.quantile(self.winsorize_quantile)
            hi = s.quantile(1.0 - self.winsorize_quantile)
            s = s.clip(lower=lo, upper=hi)

        # Industry neutralization: subtract industry mean alpha
        if self.industry_neutral and industry_map is not None:
            common = s.index.intersection(industry_map.index)
            if len(common) > 0:
                s_common = s.loc[common]
                industries = industry_map.loc[common]
                # Only use stocks with a valid industry assignment
                valid_mask = industries.notna() & (industries != "")
                if valid_mask.any():
                    s_valid = s_common[valid_mask]
                    ind_valid = industries[valid_mask]
                    industry_means = s_valid.groupby(ind_valid).transform("mean")
                    s.loc[s_valid.index] = s_valid - industry_means

        # Style factor neutralization: residualize alpha against style factors
        if self.style_neutral_factors and style_exposures is not None:
            avail_factors = [f for f in self.style_neutral_factors if f in style_exposures.columns]
            if avail_factors:
                common = s.index.intersection(style_exposures.index)
                if len(common) > len(avail_factors) + 1:
                    y = s.loc[common].values
                    X = style_exposures.loc[common, avail_factors].values
                    # Add intercept
                    X = np.column_stack([np.ones(len(common)), X])
                    # OLS: beta = (X'X)^-1 X'y
                    try:
                        beta = np.linalg.lstsq(X, y, rcond=None)[0]
                        residual = y - X @ beta
                        s.loc[common] = residual
                    except np.linalg.LinAlgError:
                        pass

        if self.method == "rank":
            s = s.rank(pct=True)
            s = (s - s.mean()) / s.std(ddof=0)
        else:  # zscore
            s = (s - s.mean()) / s.std(ddof=0)

        return s
