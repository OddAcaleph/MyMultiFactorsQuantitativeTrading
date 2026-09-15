"""Run the Risk Model V1 pipeline.

Usage:
    python src/risk/run_risk_model.py --config conf/risk_model/risk_model_v1.json \\
        --start-date 20200101 --end-date 20231231
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import load_risk_model_config, resolve_path  # noqa: E402

from risk import RiskModel  # noqa: E402
from risk.validators import (  # noqa: E402
    CovarianceValidator,
    ExposureValidator,
    RiskForecastValidator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_risk_model")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Risk Model V1 pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to risk model config JSON",
    )
    parser.add_argument(
        "--start-date",
        type=int,
        default=None,
        help="Start date of output range (YYYYMMDD)",
    )
    parser.add_argument(
        "--end-date",
        type=int,
        default=None,
        help="End date of output range (YYYYMMDD)",
    )
    parser.add_argument(
        "--warmup-days",
        type=int,
        default=300,
        help="Number of extra trading days to load before start-date for rolling windows",
    )
    parser.add_argument(
        "--no-covariance",
        action="store_true",
        help="Skip stock-level covariance matrix building (much faster)",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only run validation on existing outputs",
    )
    return parser.parse_args()


def load_wide_table(data_dir: Path, start_date: int | None, end_date: int | None) -> pd.DataFrame:
    """Load wide-table daily bars from Hive-partitioned parquet files.

    The wide table already contains OHLCV + PIT fundamentals + industry
    one-hot + ST/suspend flags, so we don't need to load and merge
    separate data sources.
    """

    import pyarrow as pa
    import pyarrow.dataset as ds

    dataset = ds.dataset(str(data_dir), partitioning="hive")
    filter_expr = None
    if start_date is not None:
        filter_expr = ds.field("trade_date") >= start_date
    if end_date is not None:
        end_expr = ds.field("trade_date") <= end_date
        filter_expr = end_expr if filter_expr is None else (filter_expr & end_expr)

    # Only load columns we need
    needed_prefixes = (
        "ts_code", "trade_date",
        "open", "high", "low", "close", "pre_close", "pct_chg", "vol", "amount",
        "roe", "roa", "or_yoy", "debt_to_assets", "gross_margin", "eps", "bps",
        "gross_margin_change", "revenue_yoy_acceleration",
        "L1_",
    )
    all_cols = [f.name for f in dataset.schema if not pa.types.is_null(f.type)]
    use_cols = []
    for col in all_cols:
        for prefix in needed_prefixes:
            if col == prefix or col.startswith(prefix):
                use_cols.append(col)
                break

    table = dataset.to_table(columns=use_cols, filter=filter_expr)
    df = table.to_pandas()
    logger.info("Loaded %d rows of wide-table data (%d columns)", len(df), len(use_cols))
    return df


def _trim_result_to_date_range(
    result: "RiskModelResult",
    start_date: int | None,
    end_date: int | None,
) -> "RiskModelResult":
    """Trim all result components to the output date range."""
    from risk.risk_model import RiskModelResult
    from risk.risk_exposure import ExposureBuildResult
    from risk.factor_return import FactorReturnResult
    from risk.factor_covariance import FactorCovarianceResult
    from risk.specific_risk import SpecificRiskResult
    from risk.covariance_builder import CovarianceBuildResult

    sd = int(start_date) if start_date is not None else None
    ed = int(end_date) if end_date is not None else None

    def _in_range(d):
        if sd is not None and d < sd:
            return False
        if ed is not None and d > ed:
            return False
        return True

    # Exposures
    exp_df = result.exposure.exposures.copy()
    if sd is not None:
        exp_df = exp_df[exp_df["trade_date"] >= sd]
    if ed is not None:
        exp_df = exp_df[exp_df["trade_date"] <= ed]
    exp_df = exp_df.reset_index(drop=True)
    exp_result = ExposureBuildResult(
        exposures=exp_df,
        style_factors=result.exposure.style_factors,
        industry_factors=result.exposure.industry_factors,
        start_date=int(exp_df["trade_date"].min()) if len(exp_df) else 0,
        end_date=int(exp_df["trade_date"].max()) if len(exp_df) else 0,
        n_stocks=exp_df["ts_code"].nunique() if len(exp_df) else 0,
        n_dates=exp_df["trade_date"].nunique() if len(exp_df) else 0,
    )

    # Factor returns
    fr_df = result.factor_return.factor_returns.copy()
    res_df = result.factor_return.residual_returns.copy()
    r2_series = result.factor_return.r_squared.copy()
    n_stocks_series = result.factor_return.n_stocks.copy()
    if sd is not None:
        fr_df = fr_df[fr_df["trade_date"] >= sd]
        res_df = res_df[res_df["trade_date"] >= sd]
        r2_series = r2_series[r2_series.index >= sd]
        n_stocks_series = n_stocks_series[n_stocks_series.index >= sd]
    if ed is not None:
        fr_df = fr_df[fr_df["trade_date"] <= ed]
        res_df = res_df[res_df["trade_date"] <= ed]
        r2_series = r2_series[r2_series.index <= ed]
        n_stocks_series = n_stocks_series[n_stocks_series.index <= ed]
    fr_dates = sorted(fr_df["trade_date"].unique()) if len(fr_df) else []
    fr_result = FactorReturnResult(
        factor_returns=fr_df.reset_index(drop=True),
        residual_returns=res_df.reset_index(drop=True),
        r_squared=r2_series,
        n_stocks=n_stocks_series,
        start_date=int(fr_dates[0]) if fr_dates else 0,
        end_date=int(fr_dates[-1]) if fr_dates else 0,
    )

    # Factor covariance
    fc_mats = {d: m for d, m in result.factor_covariance.covariance_matrices.items() if _in_range(d)}
    fc_shrink = {d: v for d, v in result.factor_covariance.shrinkage_intensities.items() if _in_range(d)}
    fc_dates = sorted(fc_mats.keys())
    fc_result = FactorCovarianceResult(
        dates=fc_dates,
        factor_names=result.factor_covariance.factor_names,
        covariance_matrices=fc_mats,
        shrinkage_intensities=fc_shrink,
        start_date=fc_dates[0] if fc_dates else 0,
        end_date=fc_dates[-1] if fc_dates else 0,
    )

    # Specific risk
    sr_df = result.specific_risk.specific_risk.copy()
    if sd is not None:
        sr_df = sr_df[sr_df["trade_date"] >= sd]
    if ed is not None:
        sr_df = sr_df[sr_df["trade_date"] <= ed]
    sr_df = sr_df.reset_index(drop=True)
    sr_valid = sr_df["specific_vol"].notna()
    sr_result = SpecificRiskResult(
        specific_risk=sr_df,
        start_date=int(sr_df.loc[sr_valid, "trade_date"].min()) if sr_valid.any() else 0,
        end_date=int(sr_df.loc[sr_valid, "trade_date"].max()) if sr_valid.any() else 0,
    )

    # Covariance
    cov_dates = [d for d in result.covariance.dates if _in_range(d)]
    cov_stock_codes = {d: v for d, v in result.covariance.stock_codes.items() if _in_range(d)}
    cov_mats = {d: v for d, v in result.covariance.covariance_matrices.items() if _in_range(d)}
    cov_min_eig = {d: v for d, v in result.covariance.min_eigenvalues.items() if _in_range(d)}
    cov_cond = {d: v for d, v in result.covariance.condition_numbers.items() if _in_range(d)}
    cov_result = CovarianceBuildResult(
        dates=cov_dates,
        stock_codes=cov_stock_codes,
        covariance_matrices=cov_mats,
        min_eigenvalues=cov_min_eig,
        condition_numbers=cov_cond,
        start_date=cov_dates[0] if cov_dates else 0,
        end_date=cov_dates[-1] if cov_dates else 0,
    )

    return RiskModelResult(
        exposure=exp_result,
        factor_return=fr_result,
        factor_covariance=fc_result,
        specific_risk=sr_result,
        covariance=cov_result,
    )


def _clean_warmup_cov_files(cov_dir: Path, start_date: int | None, end_date: int | None) -> None:
    """Remove covariance .npy files outside the output date range."""
    if start_date is None and end_date is None:
        return

    sd = int(start_date) if start_date is not None else None
    ed = int(end_date) if end_date is not None else None

    for f in cov_dir.glob("*.npy"):
        stem = f.stem
        if stem.startswith("codes_"):
            date_str = stem[len("codes_"):]
        else:
            date_str = stem
        try:
            td = int(date_str)
        except ValueError:
            continue
        if sd is not None and td < sd:
            f.unlink()
        elif ed is not None and td > ed:
            f.unlink()


def main() -> None:
    args = parse_args()
    config = load_risk_model_config(args.config)

    data_cfg = config["data"]
    output_cfg = config["output"]
    base_dir = resolve_path(output_cfg["base_dir"])
    out_dir = base_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    wide_table_dir = resolve_path(data_cfg["wide_table_dir"])

    # Determine load date range: start_date minus warmup for rolling windows
    load_start_date = args.start_date
    output_start_date = args.start_date
    if args.start_date is not None and args.warmup_days > 0:
        # Approximate warmup calendar days (trading days * 1.5 + buffer)
        warmup_calendar = int(args.warmup_days * 1.6)
        import datetime
        sd = datetime.datetime.strptime(str(args.start_date), "%Y%m%d")
        warmup_start = sd - datetime.timedelta(days=warmup_calendar)
        load_start_date = int(warmup_start.strftime("%Y%m%d"))

    logger.info("Loading data...")
    wide_df = load_wide_table(wide_table_dir, load_start_date, args.end_date)

    logger.info("Running Risk Model V1...")
    risk_model = RiskModel(config)

    # Save stock covariances to disk but don't keep all in memory (avoid OOM)
    cov_dir = out_dir / output_cfg["covariance"]
    build_cov = not args.no_covariance
    if build_cov:
        cov_dir.mkdir(parents=True, exist_ok=True)

    result = risk_model.run(
        wide_df,
        fundamentals=None,
        industry_onehot=None,
        save_covariance_dir=cov_dir if build_cov else None,
        keep_covariance_in_memory=False,
        build_covariance=build_cov,
    )

    # Trim results to output date range (exclude warmup period)
    if output_start_date is not None:
        result = _trim_result_to_date_range(result, output_start_date, args.end_date)
        # Also remove covariance files from warmup period
        _clean_warmup_cov_files(cov_dir, output_start_date, args.end_date)

    # Save outputs

    # Exposures
    exp_dir = out_dir / output_cfg["exposure"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    result.exposure.exposures.to_parquet(exp_dir / "exposures.parquet", index=False)
    logger.info("Saved exposures to %s", exp_dir)

    # Factor returns
    fr_dir = out_dir / output_cfg["factor_return"]
    fr_dir.mkdir(parents=True, exist_ok=True)
    result.factor_return.factor_returns.to_parquet(fr_dir / "factor_returns.parquet", index=False)
    result.factor_return.residual_returns.to_parquet(fr_dir / "residual_returns.parquet", index=False)
    result.factor_return.r_squared.to_frame(name="r_squared").to_parquet(fr_dir / "r_squared.parquet")
    logger.info("Saved factor returns to %s", fr_dir)

    # Factor covariance
    fc_dir = out_dir / output_cfg["factor_covariance"]
    fc_dir.mkdir(parents=True, exist_ok=True)
    np.save(fc_dir / "factor_names.npy", np.array(result.factor_covariance.factor_names))
    for td, mat in result.factor_covariance.covariance_matrices.items():
        np.save(fc_dir / f"{td}.npy", mat)
    logger.info("Saved factor covariance to %s", fc_dir)

    # Specific risk
    sr_dir = out_dir / output_cfg["specific_risk"]
    sr_dir.mkdir(parents=True, exist_ok=True)
    result.specific_risk.specific_risk.to_parquet(sr_dir / "specific_risk.parquet", index=False)
    logger.info("Saved specific risk to %s", sr_dir)

    # Stock covariance metadata
    if build_cov and result.covariance.dates:
        meta = {
            "dates": result.covariance.dates,
            "start_date": result.covariance.start_date,
            "end_date": result.covariance.end_date,
            "n_dates": len(result.covariance.dates),
        }
        with open(cov_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
        logger.info("Saved stock covariance to %s (%d dates)", cov_dir, len(result.covariance.dates))
    else:
        logger.info("Stock covariance building skipped")

    # Validation
    logger.info("Running validation...")
    diag_dir = out_dir / output_cfg.get("diagnostics", "diagnostics")
    diag_dir.mkdir(parents=True, exist_ok=True)

    exp_validator = ExposureValidator(config)
    exp_result = exp_validator.validate(result.exposure.exposures)
    with open(diag_dir / "exposure_validation.json", "w") as f:
        json.dump(exp_result.to_dict(), f, indent=2, default=str)
    logger.info("Exposure validation: passed=%s, issues=%d", exp_result.passed, len(exp_result.issues))

    if build_cov and result.covariance.dates:
        cov_validator = CovarianceValidator(config)
        # Load matrices one by one for validation (don't keep all in memory)
        cov_sample = {}
        sample_dates = result.covariance.dates[:5] + result.covariance.dates[-5:] if len(result.covariance.dates) >= 10 else result.covariance.dates
        for td in sample_dates:
            mat_path = cov_dir / f"{td}.npy"
            if mat_path.exists():
                cov_sample[td] = np.load(mat_path)
        cov_val_result = cov_validator.validate(cov_sample)
        with open(diag_dir / "covariance_validation.json", "w") as f:
            json.dump(cov_val_result.to_dict(), f, indent=2, default=str)
        logger.info("Covariance validation (sample of %d): passed=%s, issues=%d",
                    len(cov_sample), cov_val_result.passed, len(cov_val_result.issues))
    else:
        logger.info("Covariance validation skipped (no covariance matrices)")

    # Risk forecast validation (equal-weighted market portfolio)
    logger.info("Running risk forecast validation...")
    rf_validator = RiskForecastValidator(config, horizon=20)

    # Compute equal-weight portfolio returns
    daily_ret = wide_df[["trade_date", "ts_code", "pct_chg"]].copy()
    daily_ret["ret"] = daily_ret["pct_chg"] / 100.0
    port_ret = daily_ret.groupby("trade_date")["ret"].mean()
    realized_20d = port_ret.rolling(20, min_periods=10).std().shift(-19)

    # Predicted portfolio vol — compute efficiently using factor model formula
    # sigma_p^2 = w^T X F X^T w + w^T D w = (X^T w)^T F (X^T w) + sum(w_i^2 D_ii)
    from risk.covariance_builder import CovarianceBuilder

    exposures_df = result.exposure.exposures
    fc_mats = result.factor_covariance.covariance_matrices
    sr_df = result.specific_risk.specific_risk
    factor_names = result.factor_covariance.factor_names

    # Get active factor columns from exposures
    all_exp_cols = [c for c in exposures_df.columns if c not in ("trade_date", "ts_code")]
    active_factors = [f for f in factor_names if f in all_exp_cols]
    # Include INTERCEPT if present in factor covariance (market factor)
    has_intercept = "INTERCEPT" in factor_names
    if has_intercept and "INTERCEPT" not in active_factors:
        active_factors = ["INTERCEPT"] + active_factors

    pred_vols = {}
    exp_by_date = {td: g for td, g in exposures_df.groupby("trade_date")}
    sr_by_date = {td: g.set_index("ts_code")["specific_variance"] for td, g in sr_df.groupby("trade_date")}

    # Use factor covariance dates as the prediction dates
    forecast_dates = sorted(fc_mats.keys())

    for td in forecast_dates:
        if td not in exp_by_date or td not in fc_mats or td not in sr_by_date:
            continue
        exp_day = exp_by_date[td]
        sr_day = sr_by_date[td]

        # Align stocks
        codes = exp_day["ts_code"].values
        valid = exp_day["ts_code"].isin(sr_day.index)
        exp_day = exp_day[valid]
        codes = exp_day["ts_code"].values
        n = len(codes)
        if n < 2:
            continue

        # Build exposure matrix X, adding INTERCEPT column of ones if needed
        exp_factor_cols = [f for f in active_factors if f in exp_day.columns and f != "INTERCEPT"]
        X_base = exp_day[exp_factor_cols].values.astype(float) if exp_factor_cols else np.zeros((len(exp_day), 0))
        if "INTERCEPT" in active_factors:
            intercept_col = np.ones((len(exp_day), 1))
            # INTERCEPT is first in factor_names order
            X = np.hstack([intercept_col, X_base])
        else:
            X = X_base
        X = np.nan_to_num(X, nan=0.0)
        F = fc_mats[td]
        D_diag = sr_day.loc[codes].values.astype(float)
        D_diag = np.nan_to_num(D_diag, nan=0.0)
        D_diag = np.maximum(D_diag, 0.0)

        w = np.ones(n) / n
        pred_vols[td] = CovarianceBuilder.compute_portfolio_vol(w, X, F, D_diag)

    pred_series = pd.Series(pred_vols).sort_index()
    forecast_result = rf_validator.validate_portfolio(pred_series, realized_20d)
    with open(diag_dir / "risk_forecast_validation.json", "w") as f:
        json.dump(forecast_result.to_dict(), f, indent=2, default=str)
    logger.info(
        "Risk forecast: Pearson=%.4f, Spearman=%.4f, bias=%.2f%%",
        forecast_result.pearson_correlation,
        forecast_result.spearman_correlation,
        forecast_result.vol_bias_pct * 100,
    )

    logger.info("Risk Model V1 pipeline complete.")
    logger.info("Output directory: %s", out_dir)


if __name__ == "__main__":
    main()
