"""Factor timing — regime-based alpha scaling to address alpha decay.

Implements:
1. Volatility regime scaling: reduce alpha exposure when market vol is high (alpha tends to be weaker)
2. Alpha momentum scaling: scale alpha by recent IC / performance
"""
from __future__ import annotations

from typing import Any
from collections import deque

import numpy as np
import pandas as pd


class FactorTiming:
    """Regime-based alpha timing module.

    Scales alpha scores up/down based on market regime signals.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        config = config or {}
        self.enabled: bool = bool(config.get("enabled", False))
        self.vol_lookback: int = int(config.get("vol_lookback", 20))
        self.vol_low_threshold: float = float(config.get("vol_low_threshold", 0.015))
        self.vol_high_threshold: float = float(config.get("vol_high_threshold", 0.025))
        self.vol_low_scale: float = float(config.get("vol_low_scale", 1.2))
        self.vol_high_scale: float = float(config.get("vol_high_scale", 0.5))

        self.alpha_momentum_enabled: bool = bool(config.get("alpha_momentum_enabled", False))
        self.alpha_mom_lookback: int = int(config.get("alpha_mom_lookback", 20))
        self.alpha_mom_min_scale: float = float(config.get("alpha_mom_min_scale", 0.3))
        self.alpha_mom_max_scale: float = float(config.get("alpha_mom_max_scale", 1.5))
        self.alpha_mom_ic_threshold: float = float(config.get("alpha_mom_ic_threshold", 0.0))
        self.alpha_mom_ic_scale: float = float(config.get("alpha_mom_ic_scale", 0.05))

        self._market_returns: deque[float] = deque(maxlen=self.vol_lookback + 60)
        self._alpha_history: deque[tuple[pd.Series, pd.Series]] = deque(
            maxlen=self.alpha_mom_lookback + 5)
        self._prev_alpha: pd.Series | None = None
        self._scale_history: list[tuple[pd.Timestamp, float]] = []

    def update_market(self, market_return: float) -> None:
        """Update with latest market return for volatility regime detection."""
        self._market_returns.append(market_return)

    def update_alpha_performance(self, alpha: pd.Series, forward_returns: pd.Series) -> None:
        """Update alpha performance tracking.

        Parameters
        ----------
        alpha : pd.Series
            Alpha scores (index=ts_code).
        forward_returns : pd.Series
            Realized forward returns (index=ts_code).
        """
        if not self.alpha_momentum_enabled:
            return
        common = alpha.index.intersection(forward_returns.index)
        if len(common) < 10:
            return
        self._alpha_history.append((alpha.loc[common], forward_returns.loc[common]))

    def get_scale(self, dt: pd.Timestamp | None = None) -> float:
        """Get current alpha scale factor based on regime.

        Returns
        -------
        float
            Scale factor to multiply alpha by. 1.0 = neutral.
        """
        if not self.enabled:
            return 1.0

        scale = 1.0

        # Volatility regime scaling
        if len(self._market_returns) >= self.vol_lookback:
            recent = np.array(list(self._market_returns)[-self.vol_lookback:])
            recent_vol = np.std(recent, ddof=0)
            if recent_vol < self.vol_low_threshold:
                scale *= self.vol_low_scale
            elif recent_vol > self.vol_high_threshold:
                scale *= self.vol_high_scale

        # Alpha momentum scaling
        if self.alpha_momentum_enabled and len(self._alpha_history) >= self.alpha_mom_lookback:
            ic_values = []
            for alpha, ret in list(self._alpha_history)[-self.alpha_mom_lookback:]:
                common = alpha.index.intersection(ret.index)
                if len(common) >= 10:
                    ic = alpha.loc[common].corr(ret.loc[common])
                    if not np.isnan(ic):
                        ic_values.append(ic)
            if ic_values:
                avg_ic = np.mean(ic_values)
                norm_ic = np.clip(
                    (avg_ic - self.alpha_mom_ic_threshold) / self.alpha_mom_ic_scale,
                    0.0, 1.0,
                )
                mom_scale = (
                    self.alpha_mom_min_scale
                    + norm_ic * (self.alpha_mom_max_scale - self.alpha_mom_min_scale)
                )
                scale *= mom_scale

        if dt is not None:
            self._scale_history.append((dt, scale))

        return scale

    def apply(self, alpha: pd.Series, dt: pd.Timestamp | None = None) -> pd.Series:
        """Apply factor timing to alpha scores.

        Parameters
        ----------
        alpha : pd.Series
            Raw alpha scores.
        dt : pd.Timestamp, optional
            Current date for logging.

        Returns
        -------
        pd.Series
            Scaled alpha scores.
        """
        scale = self.get_scale(dt)
        if abs(scale - 1.0) < 1e-6:
            return alpha
        return alpha * scale

    @property
    def scale_history(self) -> pd.Series:
        """Return history of scale factors as a Series."""
        if not self._scale_history:
            return pd.Series(dtype=float)
        dates, scales = zip(*self._scale_history)
        return pd.Series(scales, index=dates, name="scale")
