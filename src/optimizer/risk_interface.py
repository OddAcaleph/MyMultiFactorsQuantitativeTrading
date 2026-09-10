"""Risk Model interface — loads exposures, factor covariance, specific risk."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping

import numpy as np
import pandas as pd

from optimizer.optimization_result import DayRiskData

logger = logging.getLogger(__name__)


class RiskInterface:
    """Load risk model outputs and provide portfolio risk utilities.

    Data is loaded from the risk model output directory structure::

        outputs/risk_model/v1/year=YYYY/
            exposures/exposures.parquet
            specific_risk/specific_risk.parquet
            factor_covariance/YYYYMMDD.npy
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.output_dir = Path(config["output_dir"]).expanduser().resolve()
        self._exposure_cache: Dict[int, pd.DataFrame] = {}
        self._specific_cache: Dict[int, pd.DataFrame] = {}
        self._factor_cov_cache: Dict[int, np.ndarray] = {}
        self._factor_names_cache: Dict[int, List[str]] = {}
        self._style_idx_cache: Dict[int, List[int]] = {}
        self._industry_idx_cache: Dict[int, List[int]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_day_risk_data(
        self,
        trade_date: int,
        stock_codes: List[str],
    ) -> DayRiskData:
        """Get risk data for a specific date and stock list.

        Parameters
        ----------
        trade_date : int
            Trading date in YYYYMMDD format.
        stock_codes : list[str]
            List of ts_code strings.

        Returns
        -------
        DayRiskData
        """
        year = trade_date // 10000

        exp_df = self._load_exposures(year)
        sr_df = self._load_specific_risk(year)
        factor_cov = self._load_factor_covariance(trade_date)
        factor_names = self._get_factor_names(year)
        style_idx, industry_idx = self._get_factor_indices(year)

        day_exp = exp_df[exp_df["trade_date"] == trade_date].set_index("ts_code")
        day_sr = sr_df[sr_df["trade_date"] == trade_date].set_index("ts_code")

        codes = [c for c in stock_codes if c in day_exp.index]
        if len(codes) < len(stock_codes):
            missing = set(stock_codes) - set(codes)
            logger.warning(
                "RiskInterface: %d stocks missing exposure data on %d: %s",
                len(missing), trade_date, sorted(missing)[:5],
            )

        # Build exposure matrix aligned with factor_names order
        # INTERCEPT is always first and has value 1.0
        exp_cols = [f for f in factor_names if f != "INTERCEPT"]
        available_cols = [c for c in exp_cols if c in day_exp.columns]
        missing_cols = set(exp_cols) - set(available_cols)
        if missing_cols:
            logger.debug(
                "RiskInterface: %d factors missing from exposures on %d: %s",
                len(missing_cols), trade_date, sorted(missing_cols),
            )

        exposures = np.zeros((len(codes), len(factor_names)), dtype=np.float64)
        for i, fname in enumerate(factor_names):
            if fname == "INTERCEPT":
                exposures[:, i] = 1.0
            elif fname in day_exp.columns:
                vals = day_exp.loc[codes, fname].values.astype(np.float64)
                exposures[:, i] = np.nan_to_num(vals, nan=0.0)
            # else: stays 0.0

        specific_var = np.zeros(len(codes), dtype=np.float64)
        sr_codes = [c for c in codes if c in day_sr.index]
        if sr_codes:
            idx_map = {c: i for i, c in enumerate(codes)}
            for c in sr_codes:
                val = day_sr.loc[c, "specific_variance"]
                specific_var[idx_map[c]] = val if not np.isnan(val) else 0.0

        median_sr = np.median(specific_var[specific_var > 0]) if np.any(specific_var > 0) else 0.0003
        specific_var[specific_var == 0.0] = median_sr

        return DayRiskData(
            trade_date=trade_date,
            stock_codes=codes,
            exposures=exposures,
            factor_cov=factor_cov.astype(np.float64),
            specific_variance=specific_var,
            factor_names=list(factor_names),
            style_factor_idx=list(style_idx),
            industry_factor_idx=list(industry_idx),
        )

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def portfolio_variance(
        weights: np.ndarray,
        exposures: np.ndarray,
        factor_cov: np.ndarray,
        specific_variance: np.ndarray,
    ) -> float:
        """Compute portfolio variance efficiently.

        Uses factor model structure: z = X^T w, then z^T F z + sum(w_i^2 D_i).
        O(NK + K^2) — no need to build N×N covariance matrix.
        """
        z = exposures.T @ weights
        factor_var = float(z @ factor_cov @ z)
        specific_var = float(np.sum(weights ** 2 * specific_variance))
        return factor_var + specific_var

    @staticmethod
    def factor_exposure_portfolio(
        weights: np.ndarray,
        exposures: np.ndarray,
        factor_names: List[str],
    ) -> pd.Series:
        """Compute portfolio factor exposure: X^T w."""
        exp = exposures.T @ weights
        return pd.Series(exp, index=factor_names)

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load_exposures(self, year: int) -> pd.DataFrame:
        if year not in self._exposure_cache:
            path = self.output_dir / f"year={year}" / "exposures" / "exposures.parquet"
            logger.info("Loading exposures for year %d from %s", year, path)
            self._exposure_cache[year] = pd.read_parquet(path)
        return self._exposure_cache[year]

    def _load_specific_risk(self, year: int) -> pd.DataFrame:
        if year not in self._specific_cache:
            path = self.output_dir / f"year={year}" / "specific_risk" / "specific_risk.parquet"
            logger.info("Loading specific risk for year %d from %s", year, path)
            self._specific_cache[year] = pd.read_parquet(path)
        return self._specific_cache[year]

    def _load_factor_covariance(self, trade_date: int) -> np.ndarray:
        if trade_date not in self._factor_cov_cache:
            year = trade_date // 10000
            path = self.output_dir / f"year={year}" / "factor_covariance" / f"{trade_date}.npy"
            if not path.exists():
                raise FileNotFoundError(f"Factor covariance not found: {path}")
            self._factor_cov_cache[trade_date] = np.load(path)
        return self._factor_cov_cache[trade_date]

    def _get_factor_names(self, year: int) -> List[str]:
        """Get factor names for a given year (from factor_names.npy)."""
        if year not in self._factor_names_cache:
            path = self.output_dir / f"year={year}" / "factor_covariance" / "factor_names.npy"
            if not path.exists():
                raise FileNotFoundError(f"factor_names.npy not found for year {year}: {path}")
            self._factor_names_cache[year] = np.load(path).tolist()
        return self._factor_names_cache[year]

    def _get_factor_indices(self, year: int) -> tuple[List[int], List[int]]:
        """Get style and industry factor indices for a given year."""
        if year not in self._style_idx_cache:
            names = self._get_factor_names(year)
            style, industry = self._classify_factors(names)
            self._style_idx_cache[year] = style
            self._industry_idx_cache[year] = industry
        return self._style_idx_cache[year], self._industry_idx_cache[year]

    @staticmethod
    def _classify_factors(factor_names: List[str]) -> tuple[List[int], List[int]]:
        """Classify factors into style vs industry based on naming convention."""
        style_idx = []
        industry_idx = []
        for i, name in enumerate(factor_names):
            if name.startswith("L1_") or name.startswith("industry_"):
                industry_idx.append(i)
            else:
                style_idx.append(i)
        return style_idx, industry_idx
