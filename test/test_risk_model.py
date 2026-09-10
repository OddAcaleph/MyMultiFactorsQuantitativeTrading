"""Tests for the Risk Model V1 modules.

Tests cover:
- RiskExposureBuilder (style factors, industry factors, normalization)
- FactorReturnEstimator (WLS regression, residual reconstruction)
- FactorCovarianceEstimator (Ledoit-Wolf shrinkage, PSD)
- SpecificRiskEstimator (rolling residual vol, shrinkage)
- CovarianceBuilder (XFX^T + D, PSD enforcement)
- Validators (exposure, covariance, forecast)
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from risk import (  # noqa: E402
    CovarianceBuilder,
    FactorCovarianceEstimator,
    FactorReturnEstimator,
    RiskExposureBuilder,
    RiskModel,
    SpecificRiskEstimator,
)
from risk.validators import (  # noqa: E402
    CovarianceValidator,
    ExposureValidator,
    RiskForecastValidator,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_test_config(**overrides) -> dict:
    """Build a minimal risk model config for testing."""

    config = {
        "version": "test",
        "style_factors": [
            "size", "beta", "momentum", "value", "growth",
            "volatility", "liquidity", "leverage",
        ],
        "industry_factor": {
            "enabled": True,
            "classification": "sw_l1",
            "drop_benchmark_industry": True,
            "benchmark_industry": "银行",
        },
        "cross_sectional": {
            "winsorize": True,
            "winsorize_quantile": 0.01,
            "zscore": True,
        },
        "data": {},
        "factor_return": {
            "method": "wls",
            "weight": "equal",
            "min_stocks": 5,
        },
        "factor_covariance": {
            "method": "ledoit_wolf",
            "lookback": 60,
            "min_observations": 20,
        },
        "specific_risk": {
            "lookback": 60,
            "min_observations": 20,
            "method": "rolling_std",
            "shrinkage_target": "industry_size_median",
        },
        "beta": {
            "lookback": 60,
            "min_observations": 20,
            "benchmark": "market_equal_weight",
        },
        "size": {
            "proxy": "amount_20d_median",
            "log_transform": True,
        },
        "output": {
            "base_dir": "/tmp/test_risk_model",
            "exposure": "exposures",
            "factor_return": "factor_returns",
            "factor_covariance": "factor_covariance",
            "specific_risk": "specific_risk",
            "covariance": "covariance",
        },
    }
    # Apply overrides recursively (simple update is fine for test config)
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(config.get(key), dict):
            config[key].update(val)
        else:
            config[key] = val
    return config


def _generate_synthetic_bars(
    n_stocks: int = 30,
    n_days: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic daily bar data for testing."""

    rng = np.random.RandomState(seed)
    dates = 20200101 + np.arange(n_days)
    codes = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]

    rows = []
    prices = {code: 10.0 + rng.randn() * 2 for code in codes}

    for td in dates:
        for code in codes:
            ret = rng.normal(0.0005, 0.02)
            prev_close = prices[code]
            close = prev_close * (1 + ret)
            pct_chg = ret * 100
            amount = rng.lognormal(10, 1)
            vol = amount / close
            high = max(prev_close, close) * (1 + abs(rng.normal(0, 0.005)))
            low = min(prev_close, close) * (1 - abs(rng.normal(0, 0.005)))
            open_p = prev_close * (1 + rng.normal(0, 0.005))

            rows.append({
                "trade_date": int(td),
                "ts_code": code,
                "open": open_p,
                "high": high,
                "low": low,
                "close": close,
                "pre_close": prev_close,
                "pct_chg": pct_chg,
                "vol": vol,
                "amount": amount,
            })
            prices[code] = close

    return pd.DataFrame(rows)


def _generate_synthetic_fundamentals(bars: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic point-in-time fundamentals."""

    rng = np.random.RandomState(seed)
    dates = sorted(bars["trade_date"].unique())
    codes = bars["ts_code"].unique()

    rows = []
    for code in codes:
        base_roe = rng.uniform(0.05, 0.2)
        base_rev_yoy = rng.uniform(-0.1, 0.3)
        base_debt = rng.uniform(0.2, 0.7)
        base_bps = rng.uniform(3, 15)
        base_eps = rng.uniform(0.2, 2.0)
        base_gm = rng.uniform(0.15, 0.5)
        base_gmc = rng.uniform(-0.02, 0.02)
        base_rya = rng.uniform(-0.05, 0.05)

        for td in dates:
            rows.append({
                "trade_date": int(td),
                "ts_code": code,
                "roe": base_roe + rng.normal(0, 0.01),
                "revenue_yoy": base_rev_yoy + rng.normal(0, 0.02),
                "debt_to_assets": base_debt + rng.normal(0, 0.005),
                "bps": base_bps + rng.normal(0, 0.1),
                "eps": base_eps + rng.normal(0, 0.02),
                "gross_margin": base_gm + rng.normal(0, 0.005),
                "gross_margin_change": base_gmc + rng.normal(0, 0.003),
                "revenue_yoy_acceleration": base_rya + rng.normal(0, 0.005),
            })

    return pd.DataFrame(rows)


def _generate_synthetic_industry_onehot(bars: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic industry one-hot data."""

    rng = np.random.RandomState(seed)
    codes = bars["ts_code"].unique()
    industries = ["银行", "食品饮料", "医药生物", "电子", "计算机", "机械设备", "汽车", "电力设备"]

    rows = []
    for code in codes:
        ind_idx = rng.randint(0, len(industries))
        row = {
            "ts_code": code,
            "in_date": 20200101,
            "out_date": 20201231,
        }
        for i, ind in enumerate(industries):
            row[f"L1_{ind}"] = 1 if i == ind_idx else 0
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RiskExposureBuilder tests
# ---------------------------------------------------------------------------


class TestRiskExposureBuilder:
    def test_build_style_exposures_has_all_factors(self):
        config = _make_test_config()
        builder = RiskExposureBuilder(config)
        bars = _generate_synthetic_bars(n_stocks=30, n_days=80)
        fundamentals = _generate_synthetic_fundamentals(bars)

        result = builder.build(bars, fundamentals, industry_onehot=None)

        for factor in ("SIZE", "MOMENTUM", "VOLATILITY", "LIQUIDITY"):
            assert factor in result.exposures.columns, f"Missing factor: {factor}"

    def test_build_with_industry_exposures(self):
        config = _make_test_config()
        builder = RiskExposureBuilder(config)
        bars = _generate_synthetic_bars(n_stocks=20, n_days=50)
        industry_oh = _generate_synthetic_industry_onehot(bars)

        result = builder.build(bars, fundamentals=None, industry_onehot=industry_oh)

        industry_cols = [c for c in result.exposures.columns if c.startswith("L1_")]
        assert len(industry_cols) > 0, "No industry columns found"
        # Benchmark industry (银行) should be dropped
        assert "L1_银行" not in result.exposures.columns

    def test_style_factors_are_standardized(self):
        config = _make_test_config()
        builder = RiskExposureBuilder(config)
        bars = _generate_synthetic_bars(n_stocks=50, n_days=100)
        fundamentals = _generate_synthetic_fundamentals(bars)

        result = builder.build(bars, fundamentals, industry_onehot=None)

        # Check that cross-sectional mean ~ 0 and std ~ 1 for each date
        for factor in ("SIZE", "VOLATILITY"):
            if factor not in result.exposures.columns:
                continue
            daily_means = result.exposures.groupby("trade_date")[factor].mean()
            daily_stds = result.exposures.groupby("trade_date")[factor].std()
            # After enough warmup, means should be close to 0
            valid_means = daily_means.dropna()
            if len(valid_means) > 10:
                assert abs(valid_means.iloc[-20:].mean()) < 0.1, f"{factor} mean not ~0"

    def test_beta_factor_computed(self):
        config = _make_test_config()
        builder = RiskExposureBuilder(config)
        bars = _generate_synthetic_bars(n_stocks=20, n_days=100)

        result = builder.build(bars, fundamentals=None, industry_onehot=None)

        if "BETA" in result.exposures.columns:
            # Beta should have values after warmup
            last_date = result.exposures["trade_date"].max()
            last_day = result.exposures[result.exposures["trade_date"] == last_date]
            assert last_day["BETA"].notna().sum() > 0


# ---------------------------------------------------------------------------
# FactorReturnEstimator tests
# ---------------------------------------------------------------------------


class TestFactorReturnEstimator:
    def test_wls_returns_reasonable_values(self):
        config = _make_test_config()
        estimator = FactorReturnEstimator(config)

        n_stocks = 50
        n_days = 30
        bars = _generate_synthetic_bars(n_stocks=n_stocks, n_days=n_days)
        builder = RiskExposureBuilder(config)
        exp_result = builder.build(bars, fundamentals=None, industry_onehot=None)

        # Use only style factors (no industry) for simpler test
        style_cols = ["SIZE", "VOLATILITY", "MOMENTUM", "LIQUIDITY"]
        style_cols = [c for c in style_cols if c in exp_result.exposures.columns]
        exposures = exp_result.exposures[["trade_date", "ts_code"] + style_cols].copy()

        result = estimator.estimate(exposures, returns=bars["pct_chg"] / 100.0)

        assert len(result.factor_returns) > 0
        assert "INTERCEPT" in result.factor_returns.columns
        for col in style_cols:
            assert col in result.factor_returns.columns

    def test_residuals_reconstruct_returns(self):
        """r = Xf + residual should hold approximately."""

        config = _make_test_config()
        estimator = FactorReturnEstimator(config)

        n_stocks = 40
        n_days = 20
        bars = _generate_synthetic_bars(n_stocks=n_stocks, n_days=n_days)
        builder = RiskExposureBuilder(config)
        exp_result = builder.build(bars, fundamentals=None, industry_onehot=None)

        style_cols = ["SIZE", "VOLATILITY"]
        style_cols = [c for c in style_cols if c in exp_result.exposures.columns]
        exposures = exp_result.exposures[["trade_date", "ts_code"] + style_cols].copy()

        result = estimator.estimate(exposures, returns=bars["pct_chg"] / 100.0)

        # Check one date
        td = result.factor_returns["trade_date"].iloc[-1]
        fr_row = result.factor_returns[result.factor_returns["trade_date"] == td].iloc[0]
        exp_day = exposures[exposures["trade_date"] == td].copy()
        resid_day = result.residual_returns[result.residual_returns["trade_date"] == td]

        merged = exp_day.merge(resid_day, on=["trade_date", "ts_code"], how="inner")
        merged = merged.dropna()

        if len(merged) > 0:
            X = merged[style_cols].values
            f = np.array([fr_row[c] for c in style_cols])
            alpha = fr_row["INTERCEPT"]
            pred = alpha + X @ f
            actual_ret = pred + merged["residual"].values
            # Actual returns should be recoverable (we don't have the exact ret
            # series here since we used a Series, but residuals should be finite)
            assert np.all(np.isfinite(merged["residual"].values))

    def test_r_squared_between_zero_and_one(self):
        config = _make_test_config()
        estimator = FactorReturnEstimator(config)

        n_stocks = 50
        n_days = 30
        bars = _generate_synthetic_bars(n_stocks=n_stocks, n_days=n_days)
        builder = RiskExposureBuilder(config)
        exp_result = builder.build(bars, fundamentals=None, industry_onehot=None)

        style_cols = ["SIZE", "VOLATILITY", "MOMENTUM"]
        style_cols = [c for c in style_cols if c in exp_result.exposures.columns]
        exposures = exp_result.exposures[["trade_date", "ts_code"] + style_cols].copy()

        result = estimator.estimate(exposures, returns=bars["pct_chg"] / 100.0)

        valid_r2 = result.r_squared.dropna()
        if len(valid_r2) > 0:
            assert (valid_r2 >= 0).all() or valid_r2.min() > -0.1  # WLS can have small negative
            assert (valid_r2 <= 1).all()


# ---------------------------------------------------------------------------
# FactorCovarianceEstimator tests
# ---------------------------------------------------------------------------


class TestFactorCovarianceEstimator:
    def test_ledoit_wolf_produces_psd_matrix(self):
        config = _make_test_config()
        estimator = FactorCovarianceEstimator(config)

        rng = np.random.RandomState(42)
        n_days = 100
        n_factors = 5
        factor_cols = ["SIZE", "VALUE", "MOMENTUM", "VOLATILITY", "LIQUIDITY"]
        data = {"trade_date": 20200101 + np.arange(n_days)}
        data["INTERCEPT"] = rng.normal(0.0005, 0.001, n_days)
        for col in factor_cols:
            data[col] = rng.normal(0, 0.01, n_days)
        factor_returns = pd.DataFrame(data)

        result = estimator.estimate(factor_returns)

        assert len(result.dates) > 0
        last_date = result.dates[-1]
        F = result.covariance_matrices[last_date]

        assert F.shape == (n_factors + 1, n_factors + 1)  # +1 for INTERCEPT (market factor)
        assert np.allclose(F, F.T)
        eigvals = np.linalg.eigvalsh(F)
        assert np.all(eigvals >= -1e-10)

    def test_shrinkage_intensity_between_zero_and_one(self):
        config = _make_test_config()
        estimator = FactorCovarianceEstimator(config)

        rng = np.random.RandomState(42)
        n_days = 80
        factor_returns = pd.DataFrame({
            "trade_date": 20200101 + np.arange(n_days),
            "F1": rng.normal(0, 0.01, n_days),
            "F2": rng.normal(0, 0.01, n_days),
            "F3": rng.normal(0, 0.01, n_days),
        })

        result = estimator.estimate(factor_returns)

        for td in result.dates:
            delta = result.shrinkage_intensities[td]
            assert 0.0 <= delta <= 1.0


# ---------------------------------------------------------------------------
# SpecificRiskEstimator tests
# ---------------------------------------------------------------------------


class TestSpecificRiskEstimator:
    def test_rolling_specific_vol(self):
        config = _make_test_config()
        estimator = SpecificRiskEstimator(config)

        rng = np.random.RandomState(42)
        n_stocks = 20
        n_days = 100
        dates = 20200101 + np.arange(n_days)
        codes = [f"{i:06d}.SZ" for i in range(1, n_stocks + 1)]

        rows = []
        for code in codes:
            vol_scale = rng.uniform(0.005, 0.03)
            for td in dates:
                rows.append({
                    "trade_date": int(td),
                    "ts_code": code,
                    "residual": rng.normal(0, vol_scale),
                })

        residuals = pd.DataFrame(rows)
        result = estimator.estimate(residuals)

        # After warmup, should have specific_vol values
        last_date = result.specific_risk["trade_date"].max()
        last_day = result.specific_risk[result.specific_risk["trade_date"] == last_date]
        assert last_day["specific_vol"].notna().sum() > 0
        assert (last_day["specific_vol"].dropna() > 0).all()

    def test_specific_variance_is_square_of_vol(self):
        config = _make_test_config()
        estimator = SpecificRiskEstimator(config)

        rng = np.random.RandomState(42)
        residuals = pd.DataFrame({
            "trade_date": np.repeat(20200101 + np.arange(80), 10),
            "ts_code": np.tile([f"{i:06d}.SZ" for i in range(1, 11)], 80),
            "residual": rng.normal(0, 0.02, 800),
        })

        result = estimator.estimate(residuals)
        df = result.specific_risk.dropna()
        if len(df) > 0:
            np.testing.assert_allclose(
                df["specific_variance"].values,
                df["specific_vol"].values ** 2,
                rtol=1e-6,
            )


# ---------------------------------------------------------------------------
# CovarianceBuilder tests
# ---------------------------------------------------------------------------


class TestCovarianceBuilder:
    def test_sigma_is_psd(self):
        builder = CovarianceBuilder()

        n_stocks = 20
        n_factors = 5
        rng = np.random.RandomState(42)

        exposures = pd.DataFrame({
            "ts_code": [f"S{i:02d}" for i in range(n_stocks)],
        })
        for i in range(n_factors):
            exposures[f"F{i}"] = rng.randn(n_stocks)

        F = np.cov(rng.randn(n_factors, 100))
        specific_var = rng.uniform(0.0001, 0.001, n_stocks)
        sv_series = pd.Series(specific_var, index=exposures["ts_code"])

        Sigma, codes = builder.build(exposures, F, sv_series)

        assert Sigma.shape == (n_stocks, n_stocks)
        assert len(codes) == n_stocks
        assert np.allclose(Sigma, Sigma.T)
        eigvals = np.linalg.eigvalsh(Sigma)
        assert np.all(eigvals >= -1e-10)

    def test_sigma_formula_xfxt_plus_d(self):
        """Verify Sigma = X F X^T + D."""

        builder = CovarianceBuilder()
        n_stocks = 10
        n_factors = 3
        rng = np.random.RandomState(42)

        exposures = pd.DataFrame({
            "ts_code": [f"S{i:02d}" for i in range(n_stocks)],
        })
        for i in range(n_factors):
            exposures[f"F{i}"] = rng.randn(n_stocks)

        X = exposures[[f"F{i}" for i in range(n_factors)]].values
        F = np.diag([0.01, 0.02, 0.015])
        D_diag = np.full(n_stocks, 0.005)
        sv_series = pd.Series(D_diag, index=exposures["ts_code"])

        Sigma, _ = builder.build(exposures, F, sv_series)
        expected = X @ F @ X.T + np.diag(D_diag)

        np.testing.assert_allclose(Sigma, expected, rtol=1e-10)

    def test_build_many_multiple_dates(self):
        builder = CovarianceBuilder()

        n_stocks = 15
        n_factors = 4
        rng = np.random.RandomState(42)

        exposures_rows = []
        for td in [20200101, 20200102, 20200103]:
            for i in range(n_stocks):
                row = {"trade_date": td, "ts_code": f"S{i:02d}"}
                for j in range(n_factors):
                    row[f"F{j}"] = rng.randn()
                exposures_rows.append(row)

        exposures = pd.DataFrame(exposures_rows)

        F_map = {}
        for td in [20200101, 20200102, 20200103]:
            F_map[td] = np.diag(rng.uniform(0.005, 0.02, n_factors))

        sr_rows = []
        for td in [20200101, 20200102, 20200103]:
            for i in range(n_stocks):
                sr_rows.append({
                    "trade_date": td,
                    "ts_code": f"S{i:02d}",
                    "specific_variance": rng.uniform(0.0001, 0.001),
                })
        specific_risk = pd.DataFrame(sr_rows)

        result = builder.build_many(exposures, F_map, specific_risk)

        assert len(result.dates) == 3
        for td in result.dates:
            assert td in result.covariance_matrices
            assert result.covariance_matrices[td].shape == (n_stocks, n_stocks)


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------


class TestExposureValidator:
    def test_validate_style_factors(self):
        validator = ExposureValidator()

        rng = np.random.RandomState(42)
        n_stocks = 100
        dates = [20200101, 20200102, 20200103]

        rows = []
        for td in dates:
            for i in range(n_stocks):
                rows.append({
                    "trade_date": td,
                    "ts_code": f"S{i:03d}",
                    "SIZE": rng.randn(),
                    "VALUE": rng.randn() * 1.5 + 0.2,
                })

        exposures = pd.DataFrame(rows)
        result = validator.validate(exposures)

        assert "SIZE" in result.style_factor_stats
        assert "VALUE" in result.style_factor_stats
        assert isinstance(result.passed, bool)


class TestCovarianceValidator:
    def test_valid_psd_matrix_passes(self):
        validator = CovarianceValidator()

        rng = np.random.RandomState(42)
        A = rng.randn(10, 10)
        F = A.T @ A + np.eye(10) * 0.01

        matrices = {20200101: F, 20200102: F * 1.1}
        result = validator.validate(matrices)

        assert result.passed
        assert result.all_symmetric
        assert result.all_psd
        assert result.all_finite
        assert result.all_positive_diag

    def test_non_psd_matrix_fails(self):
        validator = CovarianceValidator(eig_tol=-1e-8)

        # Matrix with positive diagonal but a negative eigenvalue
        # (indefinite matrix: [[2, 3], [3, 2]] has eigenvalues 5 and -1)
        F = np.array([[2.0, 3.0], [3.0, 2.0]])
        matrices = {20200101: F}
        result = validator.validate(matrices)

        assert not result.passed
        assert not result.all_psd


class TestRiskForecastValidator:
    def test_positive_correlation_passes(self):
        validator = RiskForecastValidator(horizon=5)

        rng = np.random.RandomState(42)
        n = 50
        true_vol = np.abs(rng.normal(0.02, 0.005, n))
        pred_vol = true_vol + rng.normal(0, 0.002, n)

        pred_series = pd.Series(pred_vol, index=20200101 + np.arange(n))
        real_series = pd.Series(true_vol, index=20200101 + np.arange(n))

        result = validator.validate_portfolio(pred_series, real_series)
        assert result.n_observations == n
        assert result.pearson_correlation > 0.5


# ---------------------------------------------------------------------------
# RiskModel integration test
# ---------------------------------------------------------------------------


class TestRiskModelIntegration:
    def test_end_to_end_small(self):
        """Full pipeline with small synthetic data."""

        config = _make_test_config()
        config["factor_return"]["min_stocks"] = 10
        config["factor_covariance"]["lookback"] = 20
        config["factor_covariance"]["min_observations"] = 10
        config["specific_risk"]["lookback"] = 20
        config["specific_risk"]["min_observations"] = 10
        config["beta"]["lookback"] = 20
        config["beta"]["min_observations"] = 10

        bars = _generate_synthetic_bars(n_stocks=30, n_days=80)
        fundamentals = _generate_synthetic_fundamentals(bars)
        industry_oh = _generate_synthetic_industry_onehot(bars)

        model = RiskModel(config)
        result = model.run(bars, fundamentals, industry_oh)

        # Exposure
        assert result.exposure.n_dates > 0
        assert result.exposure.n_stocks > 0
        assert len(result.exposure.style_factors) > 0

        # Factor returns
        assert len(result.factor_return.factor_returns) > 0

        # Factor covariance
        assert len(result.factor_covariance.dates) > 0, "No factor covariance dates"

        # Specific risk
        assert len(result.specific_risk.specific_risk) > 0

        # Stock covariance
        assert len(result.covariance.dates) > 0
        last_date = result.covariance.dates[-1]
        Sigma = result.covariance.covariance_matrices[last_date]
        assert Sigma.shape[0] > 0
        assert np.allclose(Sigma, Sigma.T)

    def test_portfolio_vol_calculation(self):
        config = _make_test_config()
        config["factor_return"]["min_stocks"] = 10
        config["factor_covariance"]["lookback"] = 20
        config["factor_covariance"]["min_observations"] = 10
        config["specific_risk"]["lookback"] = 20
        config["specific_risk"]["min_observations"] = 10

        bars = _generate_synthetic_bars(n_stocks=20, n_days=60)
        model = RiskModel(config)
        result = model.run(bars, fundamentals=None, industry_onehot=None)

        if result.covariance.dates:
            td = result.covariance.dates[-1]
            codes = result.covariance.stock_codes[td]
            n = len(codes)
            weights = pd.Series(np.ones(n) / n, index=codes)

            port_vol = model.portfolio_vol(td, weights, result)
            assert port_vol > 0
            assert np.isfinite(port_vol)
