"""Risk Model V1 facade.

High-level interface that orchestrates the full risk model pipeline:
exposures -> factor returns -> factor covariance -> specific risk ->
stock covariance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from risk.covariance_builder import CovarianceBuilder, CovarianceBuildResult
from risk.factor_covariance import FactorCovarianceEstimator, FactorCovarianceResult
from risk.factor_return import FactorReturnEstimator, FactorReturnResult
from risk.risk_exposure import ExposureBuildResult, RiskExposureBuilder
from risk.specific_risk import SpecificRiskEstimator, SpecificRiskResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskModelResult:
    """Full risk model output."""

    exposure: ExposureBuildResult
    factor_return: FactorReturnResult
    factor_covariance: FactorCovarianceResult
    specific_risk: SpecificRiskResult
    covariance: CovarianceBuildResult


class RiskModel:
    """Risk Model V1 — end-to-end facade.

    Parameters
    ----------
    config
        Risk model config dictionary.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.exposure_builder = RiskExposureBuilder(config)
        self.factor_return_estimator = FactorReturnEstimator(config)
        self.factor_cov_estimator = FactorCovarianceEstimator(config)
        self.specific_risk_estimator = SpecificRiskEstimator(config)
        self.covariance_builder = CovarianceBuilder(config)

    def run(
        self,
        daily_bars: pd.DataFrame,
        fundamentals: pd.DataFrame | None = None,
        industry_onehot: pd.DataFrame | None = None,
        save_covariance_dir: str | Path | None = None,
        keep_covariance_in_memory: bool = True,
        build_covariance: bool = True,
    ) -> RiskModelResult:
        """Run the full risk model pipeline.

        Parameters
        ----------
        daily_bars
            Daily OHLCV data with ``trade_date``, ``ts_code``, ``close``,
            ``pct_chg``, ``amount``, ``vol`` columns.
        fundamentals
            Point-in-time fundamental data.
        industry_onehot
            Industry one-hot classification with validity intervals.
        save_covariance_dir
            If provided, save daily stock covariance matrices to this
            directory.  Useful for large runs to avoid OOM.
        keep_covariance_in_memory
            If ``False`` and *save_covariance_dir* is set, don't keep
            full matrices in memory (saves RAM).
        build_covariance
            If ``False``, skip stock-level covariance matrix building.
            Saves significant time when only X/F/D are needed.
        """

        logger.info("Building risk exposures...")
        exposure_result = self.exposure_builder.build(
            daily_bars, fundamentals, industry_onehot
        )
        logger.info(
            "Exposures built: %d dates, %d stocks, %d style + %d industry factors",
            exposure_result.n_dates,
            exposure_result.n_stocks,
            len(exposure_result.style_factors),
            len(exposure_result.industry_factors),
        )

        logger.info("Estimating daily factor returns...")
        fr_result = self.factor_return_estimator.estimate(
            exposure_result.exposures,
            returns=self._extract_returns(daily_bars),
        )
        logger.info(
            "Factor returns estimated: %d dates, avg R²=%.4f",
            len(fr_result.factor_returns),
            fr_result.r_squared.mean(),
        )

        logger.info("Estimating factor covariance...")
        fc_result = self.factor_cov_estimator.estimate(fr_result.factor_returns)
        logger.info(
            "Factor covariance estimated: %d dates, %d factors",
            len(fc_result.dates),
            len(fc_result.factor_names),
        )

        logger.info("Estimating specific risk...")
        sr_result = self.specific_risk_estimator.estimate(
            fr_result.residual_returns,
            exposures=exposure_result.exposures,
        )
        logger.info("Specific risk estimated")

        if build_covariance:
            logger.info("Building stock covariance matrices...")
            cov_result = self.covariance_builder.build_many(
                exposure_result.exposures,
                fc_result.covariance_matrices,
                sr_result.specific_risk,
                factor_names=fc_result.factor_names,
                save_dir=save_covariance_dir,
                keep_in_memory=keep_covariance_in_memory,
            )
            logger.info(
                "Stock covariance built: %d dates",
                len(cov_result.dates),
            )
        else:
            logger.info("Skipping stock covariance building (build_covariance=False)")
            from risk.covariance_builder import CovarianceBuildResult
            cov_result = CovarianceBuildResult(
                dates=[], stock_codes={}, covariance_matrices={},
                min_eigenvalues={}, condition_numbers={},
                start_date=0, end_date=0,
            )

        return RiskModelResult(
            exposure=exposure_result,
            factor_return=fr_result,
            factor_covariance=fc_result,
            specific_risk=sr_result,
            covariance=cov_result,
        )

    def get_covariance(self, trade_date: int, result: RiskModelResult) -> tuple[np.ndarray, list[str]]:
        """Get the stock covariance matrix for a specific date.

        Parameters
        ----------
        trade_date
            Date in ``YYYYMMDD`` integer format.
        result
            Risk model result from :meth:`run`.

        Returns
        -------
        tuple
            (N x N covariance matrix, list of stock codes)
        """

        td = int(trade_date)
        if td not in result.covariance.covariance_matrices:
            raise KeyError(f"No covariance matrix for date {td}")

        return (
            result.covariance.covariance_matrices[td],
            result.covariance.stock_codes[td],
        )

    def portfolio_vol(
        self,
        trade_date: int,
        weights: pd.Series,
        result: RiskModelResult,
    ) -> float:
        """Compute predicted portfolio volatility.

        Parameters
        ----------
        trade_date
            Date in ``YYYYMMDD`` integer format.
        weights
            Series of portfolio weights indexed by ``ts_code``.
        result
            Risk model result.

        Returns
        -------
        float
            Predicted portfolio volatility (daily, in same units as returns).
        """

        Sigma, codes = self.get_covariance(trade_date, result)
        w = np.array([weights.get(code, 0.0) for code in codes])
        port_var = w @ Sigma @ w
        return float(np.sqrt(max(port_var, 0.0)))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_returns(daily_bars: pd.DataFrame) -> pd.DataFrame:
        """Extract daily returns from bars DataFrame."""

        if "pct_chg" in daily_bars.columns:
            ret = daily_bars[["trade_date", "ts_code", "pct_chg"]].copy()
            ret = ret.rename(columns={"pct_chg": "ret"})
            ret["ret"] = ret["ret"] / 100.0  # pct_chg is in percent
            return ret

        if "close" in daily_bars.columns:
            df = daily_bars[["trade_date", "ts_code", "close"]].copy()
            df = df.sort_values(["ts_code", "trade_date"])
            df["ret"] = df.groupby("ts_code")["close"].pct_change()
            return df[["trade_date", "ts_code", "ret"]]

        raise ValueError("Cannot compute returns: need pct_chg or close column")
