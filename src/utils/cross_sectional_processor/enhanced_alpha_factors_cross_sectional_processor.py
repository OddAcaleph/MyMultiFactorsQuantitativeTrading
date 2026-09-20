"""Cross-sectional processing for enhanced_alpha_factors.

Applies winsorization then z-score standardization to each enhanced alpha
factor column, producing ``{factor}_cc_processed`` output columns.  Source
files are never modified.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

try:
    from src.utils.cross_sectional_processor.data_winsorize_util import winsorize_by_trade_date
    from src.utils.cross_sectional_processor.data_z_score_util import zscore_by_trade_date
except ModuleNotFoundError:
    try:
        from utils.cross_sectional_processor.data_winsorize_util import winsorize_by_trade_date
        from utils.cross_sectional_processor.data_z_score_util import zscore_by_trade_date
    except ModuleNotFoundError:
        from data_winsorize_util import winsorize_by_trade_date
        from data_z_score_util import zscore_by_trade_date


class MissingRequiredColumnsError(ValueError):
    """Raised when an input file is missing required factor columns."""

    def __init__(self, missing_columns: Sequence[str]) -> None:
        self.missing_columns = tuple(missing_columns)
        super().__init__(f"Missing required columns: {self.missing_columns}")


@dataclass(frozen=True)
class EnhancedAlphaFactorsCrossSectionalSummary:
    """Processing statistics for enhanced_alpha_factors cross-sectional features."""

    input_dir: Path
    output_dir: Path
    files_found: int
    files_processed: int
    rows_read: int
    rows_written: int
    factor_columns: tuple[str, ...]
    missing_required_files: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    failed_files: Mapping[str, str] = field(default_factory=dict)

    @property
    def processed_columns(self) -> tuple[str, ...]:
        return tuple(f"{c}_cc_processed" for c in self.factor_columns)

    @property
    def has_errors(self) -> bool:
        return bool(self.missing_required_files or self.failed_files)


class EnhancedAlphaFactorsCrossSectionalProcessor:
    """Apply winsorize then z-score to enhanced alpha factors."""

    DEFAULT_INPUT_DIR = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/features_data/enhanced_alpha_factors"
    )
    DEFAULT_OUTPUT_DIR = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cross_sectional_processd_data/enhanced_alpha_factors"
    )
    DEFAULT_LOG_FILE = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/cross_sectional_process/enhanced_alpha_factors_cross_sectional_processor.log"
    )

    DATE_COLUMN = "trade_date"
    KEY_COLUMNS: tuple[str, str] = ("trade_date", "ts_code")
    FACTOR_COLUMNS: tuple[str, ...] = (
        # Momentum
        "momentum_12m_skip1m",
        "momentum_6m_skip2w",
        "momentum_52w_high_dist",
        "momentum_52w_high_break",
        "rsi_14",
        "williams_r_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "consecutive_up_days",
        "consecutive_down_days",
        # Volume-price
        "obv",
        "obv_ma5_ratio",
        "volume_price_trend",
        "volume_momentum",
        # Moneyflow
        "main_net_momentum_5d",
        "main_net_trend_20d",
        "price_mf_divergence",
        # Fundamental momentum
        "roe_qoq_change",
        "revenue_yoy_acceleration",
        "gross_margin_change",
        # Liquidity
        "amihud_20",
        "turnover_vol_20",
        # Volatility
        "downside_vol_20",
        "max_drawdown_60",
        "bollinger_width_20",
        "bollinger_position_20",
        "atr_14",
        "price_position_20",
        "price_position_60",
        "ret_skew_20",
        "ret_kurt_60",
        # Technical oscillators
        "kdj_k",
        "kdj_d",
        "kdj_j",
        "cci_20",
        # Volume advanced
        "up_down_volume_ratio_20",
        "volume_price_divergence_20",
        # Advanced moneyflow
        "main_net_ema_20",
        "main_net_ema_slope_20",
        "main_net_consecutive",
        "main_net_amount_ratio_20",
        # Industry-relative
        "industry_rank_ret_20",
        "industry_vol_20",
        # Interaction
        "momentum_volume_interaction",
        "roe_growth_interaction",
        # Quality
        "accruals_ratio",
        "ocf_to_profit",
        "gross_profitability",
        "asset_turnover",
        "interest_coverage",
        "net_operating_assets",
        "roe_stability_8q",
        "earnings_growth_stability",
        "profit_margin_change",
        "roic_change",
        # New quality factors
        "current_ratio",
        "quick_ratio",
        "debt_to_equity",
        "ocf_to_debt",
        "ebitda_to_debt",
        "roic_level",
        "roe_yearly",
        "roa_yearly",
        "cash_to_liqdebt",
        "tangible_asset_ratio",
        "operating_leverage",
        "financial_leverage",
        # Value
        "ep_ratio",
        "bp_ratio",
        "sp_ratio",
        "cfp_ratio",
        "dividend_yield_approx",
        # Piotroski F-score
        "piotroski_f_score",
        "f_profitability",
        "f_leverage_liquidity",
        "f_efficiency",
        # Growth factors
        "op_yoy",
        "netprofit_yoy",
        "roe_yoy",
        "bps_yoy",
        "assets_yoy",
        "equity_yoy",
        "basic_eps_yoy",
        "cfps_yoy",
        "revenue_acceleration_2q",
        "profit_acceleration_2q",
        "roe_momentum_4q",
        "earnings_surprise_qoq",
        # Moneyflow depth factors
        "lg_net_ratio",
        "elg_net_ratio",
        "retail_net_ratio",
        "sm_net_ratio",
        "md_net_ratio",
        "main_retail_ratio_20",
        "lg_elg_ratio_20",
        "moneyflow_strength_5",
        "moneyflow_dispersion_20",
        "net_mf_amount_ratio",
        "net_mf_vol_ratio",
        "buy_pressure_5",
        "sell_pressure_5",
        # Industry depth factors
        "industry_rank_roe",
        "industry_rank_gross_margin",
        "industry_rank_roic",
        "industry_rank_netprofit_yoy",
        "industry_rank_turn_days",
        "industry_momentum_5",
        "industry_momentum_20",
        "relative_momentum_20",
        "industry_concentration",
        # Sentiment proxy factors
        "rd_intensity",
        "rd_growth",
        "earnings_quality_composite",
        "profit_consistency",
        "dividend_payout_approx",
    )

    INDUSTRY_NEUTRAL_FACTORS: tuple[str, ...] = (
        "amihud_20",
        "turnover_vol_20",
        "downside_vol_20",
        "max_drawdown_60",
        "bollinger_width_20",
        "bollinger_position_20",
        "atr_14",
        "price_position_20",
        "price_position_60",
        "ret_skew_20",
        "ret_kurt_60",
        "kdj_k",
        "kdj_d",
        "kdj_j",
        "cci_20",
        "up_down_volume_ratio_20",
        "volume_price_divergence_20",
        "main_net_ema_20",
        "main_net_ema_slope_20",
        "main_net_amount_ratio_20",
        "momentum_volume_interaction",
        "roe_growth_interaction",
        "accruals_ratio",
        "ocf_to_profit",
        "gross_profitability",
        "asset_turnover",
        "interest_coverage",
        "net_operating_assets",
        "roe_stability_8q",
        "earnings_growth_stability",
        "profit_margin_change",
        "roic_change",
        # New quality factors
        "current_ratio",
        "quick_ratio",
        "debt_to_equity",
        "ocf_to_debt",
        "ebitda_to_debt",
        "roic_level",
        "roe_yearly",
        "roa_yearly",
        "cash_to_liqdebt",
        "tangible_asset_ratio",
        "operating_leverage",
        "financial_leverage",
        # Value
        "ep_ratio",
        "bp_ratio",
        "sp_ratio",
        "cfp_ratio",
        "dividend_yield_approx",
        # Piotroski F-score
        "piotroski_f_score",
        "f_profitability",
        "f_leverage_liquidity",
        "f_efficiency",
        # Growth factors
        "op_yoy",
        "netprofit_yoy",
        "roe_yoy",
        "bps_yoy",
        "assets_yoy",
        "equity_yoy",
        "basic_eps_yoy",
        "cfps_yoy",
        "revenue_acceleration_2q",
        "profit_acceleration_2q",
        "roe_momentum_4q",
        "earnings_surprise_qoq",
        # Moneyflow depth factors
        "lg_net_ratio",
        "elg_net_ratio",
        "retail_net_ratio",
        "sm_net_ratio",
        "md_net_ratio",
        "main_retail_ratio_20",
        "lg_elg_ratio_20",
        "moneyflow_strength_5",
        "moneyflow_dispersion_20",
        "net_mf_amount_ratio",
        "net_mf_vol_ratio",
        "buy_pressure_5",
        "sell_pressure_5",
        # Sentiment proxy factors
        "rd_intensity",
        "rd_growth",
        "earnings_quality_composite",
        "profit_consistency",
        "dividend_payout_approx",
    )

    DEFAULT_INDUSTRY_FILE = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/industry/industry.parquet"
    )

    def __init__(
        self,
        input_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        factor_columns: Sequence[str] | None = None,
        industry_file: str | Path | None = None,
        industry_neutral: bool = True,
        start_date: str | int | None = None,
        end_date: str | int | None = None,
        logger: logging.Logger | None = None,
        fail_fast: bool = False,
    ) -> None:
        self.input_dir = Path(input_dir or self.DEFAULT_INPUT_DIR)
        self.output_dir = Path(output_dir or self.DEFAULT_OUTPUT_DIR)
        self.factor_columns = tuple(factor_columns) if factor_columns else self.FACTOR_COLUMNS
        self.industry_file = Path(industry_file or self.DEFAULT_INDUSTRY_FILE)
        self.industry_neutral = industry_neutral
        self.start_date = self._parse_optional_date(start_date, "start_date")
        self.end_date = self._parse_optional_date(end_date, "end_date")
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.fail_fast = fail_fast
        self._industry_df: pd.DataFrame | None = None
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            raise NotADirectoryError(f"Input directory not found: {self.input_dir}")
        if self.industry_neutral and not self.industry_file.exists():
            self.logger.warning("Industry file not found: %s — industry neutralization will be skipped.", self.industry_file)
            self.industry_neutral = False
        if self.start_date is not None and self.end_date is not None and self.start_date > self.end_date:
            raise ValueError("start_date must be <= end_date")

    def process(self) -> EnhancedAlphaFactorsCrossSectionalSummary:
        input_files = self._list_input_files()
        in_range_files = [f for f in input_files if self._is_file_date_in_range(f)]
        if not in_range_files:
            raise FileNotFoundError(f"No input files found in date range under {self.input_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            "增强alpha因子横截面处理：input=%s, output=%s, files=%d, factors=%d",
            self.input_dir, self.output_dir, len(in_range_files), len(self.factor_columns),
        )

        rows_read = 0
        rows_written = 0
        files_processed = 0
        missing_required: dict[str, tuple[str, ...]] = {}
        failed: dict[str, str] = {}

        for idx, input_file in enumerate(in_range_files, start=1):
            try:
                output_file = self._output_file_for(input_file)
                processed_df = self._process_one_file(input_file)
                rows_read += len(processed_df)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                processed_df.to_parquet(output_file, index=False)
                rows_written += len(processed_df)
                files_processed += 1

                if idx == 1 or idx % 100 == 0 or idx == len(in_range_files):
                    self.logger.info(
                        "处理进度：%d/%d files, latest=%s, rows=%d",
                        idx, len(in_range_files), input_file.name, len(processed_df),
                    )
            except MissingRequiredColumnsError as exc:
                missing_required[str(input_file)] = tuple(exc.missing_columns)
                self.logger.error("文件缺少必需列，已跳过：file=%s, missing=%s", input_file, exc.missing_columns)
                if self.fail_fast:
                    raise
            except Exception as exc:
                failed[str(input_file)] = repr(exc)
                self.logger.exception("文件处理失败，已跳过：file=%s", input_file)
                if self.fail_fast:
                    raise

        summary = EnhancedAlphaFactorsCrossSectionalSummary(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            files_found=len(in_range_files),
            files_processed=files_processed,
            rows_read=rows_read,
            rows_written=rows_written,
            factor_columns=self.factor_columns,
            missing_required_files=missing_required,
            failed_files=failed,
        )
        self._log_summary(summary)
        return summary

    def _process_one_file(self, input_file: Path) -> pd.DataFrame:
        df = pd.read_parquet(input_file)
        self._validate_columns(df.columns)

        processed = df.copy()

        # Load industry data once for neutralization
        ind_map = None
        if self.industry_neutral:
            if self._industry_df is None:
                self._load_industry_data()
            if self._industry_df is not None:
                # Get industry for each stock in this file
                file_dates = processed[self.DATE_COLUMN].unique()
                ind_subset = self._industry_df[self._industry_df[self.DATE_COLUMN].isin(file_dates)]
                if not ind_subset.empty:
                    ind_map = ind_subset[["trade_date", "ts_code", "l1_name"]]

        for factor in self.factor_columns:
            if factor not in processed.columns:
                continue
            temp_col = f"__{factor}_winsorized"
            processed[temp_col] = pd.to_numeric(processed[factor], errors="coerce")

            # Step 1: MAD winsorization (5x MAD)
            processed[temp_col] = winsorize_by_trade_date(
                processed, temp_col, date_column=self.DATE_COLUMN
            )

            # Step 2: Industry neutralization (subtract industry mean)
            if self.industry_neutral and factor in self.INDUSTRY_NEUTRAL_FACTORS and ind_map is not None:
                merged = processed[[self.DATE_COLUMN, "ts_code", temp_col]].merge(
                    ind_map, on=[self.DATE_COLUMN, "ts_code"], how="left"
                )
                ind_mean = merged.groupby([self.DATE_COLUMN, "l1_name"])[temp_col].transform("mean")
                processed[temp_col] = merged[temp_col].values - ind_mean.values

            # Step 3: Z-score standardization
            processed[f"{factor}_cc_processed"] = zscore_by_trade_date(
                processed, temp_col, date_column=self.DATE_COLUMN
            )
            processed = processed.drop(columns=[temp_col])

        return processed

    def _load_industry_data(self) -> None:
        """Load industry mapping data for neutralization."""
        try:
            ind_df = pd.read_parquet(self.industry_file, columns=["trade_date", "ts_code", "l1_name"])
            ind_df["trade_date"] = self._normalize_yyyymmdd(ind_df["trade_date"])
            ind_df["ts_code"] = ind_df["ts_code"].astype("string").str.strip()
            ind_df["l1_name"] = ind_df["l1_name"].astype("string").str.strip()
            self._industry_df = ind_df
            self.logger.info("行业数据加载完成：rows=%d", len(ind_df))
        except Exception as exc:
            self.logger.warning("行业数据加载失败，将跳过高业中性化：%s", exc)
            self.industry_neutral = False
            self._industry_df = None

    @staticmethod
    def _normalize_yyyymmdd(series: pd.Series) -> pd.Series:
        normalized = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
        parsed = pd.to_datetime(normalized, format="%Y%m%d", errors="coerce")
        return parsed.dt.strftime("%Y%m%d").astype("int32")

    def _validate_columns(self, columns: Iterable[str]) -> None:
        col_set = set(columns)
        required = (*self.KEY_COLUMNS,)
        missing = tuple(c for c in required if c not in col_set)
        if missing:
            raise MissingRequiredColumnsError(missing)

    def _list_input_files(self) -> list[Path]:
        return sorted(self.input_dir.glob("year=*/month=*/*.parquet"))

    def _is_file_date_in_range(self, file_path: Path) -> bool:
        try:
            fd = int(file_path.stem)
        except ValueError:
            return False
        if self.start_date is not None and fd < self.start_date:
            return False
        if self.end_date is not None and fd > self.end_date:
            return False
        return True

    def _output_file_for(self, input_file: Path) -> Path:
        return self.output_dir / input_file.relative_to(self.input_dir)

    def _log_summary(self, summary: EnhancedAlphaFactorsCrossSectionalSummary) -> None:
        self.logger.info(
            "增强alpha因子横截面处理汇总：files_found=%d, files_processed=%d, "
            "rows_read=%d, rows_written=%d, factors=%d",
            summary.files_found, summary.files_processed,
            summary.rows_read, summary.rows_written, len(summary.factor_columns),
        )
        if summary.missing_required_files:
            self.logger.warning("缺少必需列的文件数=%d", len(summary.missing_required_files))
        if summary.failed_files:
            self.logger.warning("处理失败的文件数=%d", len(summary.failed_files))

    @staticmethod
    def _parse_optional_date(value: str | int | None, name: str) -> int | None:
        if value is None or value == "":
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be YYYYMMDD, got {value!r}") from exc
        if len(str(parsed)) != 8:
            raise ValueError(f"{name} must be YYYYMMDD, got {value!r}")
        return parsed


def configure_logging(log_level: str = "INFO", log_file: str | Path | None = None) -> None:
    handlers = [logging.StreamHandler()]
    target = Path(log_file) if log_file else EnhancedAlphaFactorsCrossSectionalProcessor.DEFAULT_LOG_FILE
    target.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(target, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-sectional processing for enhanced alpha factors.")
    parser.add_argument("--input-dir", default=str(EnhancedAlphaFactorsCrossSectionalProcessor.DEFAULT_INPUT_DIR))
    parser.add_argument("--output-dir", default=str(EnhancedAlphaFactorsCrossSectionalProcessor.DEFAULT_OUTPUT_DIR))
    parser.add_argument("--industry-file", default=str(EnhancedAlphaFactorsCrossSectionalProcessor.DEFAULT_INDUSTRY_FILE),
                        help="Industry parquet file for industry neutralization.")
    parser.add_argument("--no-industry-neutral", action="store_true",
                        help="Skip industry neutralization step.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default=str(EnhancedAlphaFactorsCrossSectionalProcessor.DEFAULT_LOG_FILE))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> EnhancedAlphaFactorsCrossSectionalSummary:
    args = parse_args(argv)
    configure_logging(log_level=args.log_level, log_file=args.log_file)
    processor = EnhancedAlphaFactorsCrossSectionalProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        industry_file=args.industry_file,
        industry_neutral=not args.no_industry_neutral,
        start_date=args.start_date,
        end_date=args.end_date,
        fail_fast=args.fail_fast,
    )
    return processor.process()


if __name__ == "__main__":
    main()


__all__ = [
    "EnhancedAlphaFactorsCrossSectionalSummary",
    "EnhancedAlphaFactorsCrossSectionalProcessor",
    "MissingRequiredColumnsError",
    "configure_logging",
    "main",
    "parse_args",
]
