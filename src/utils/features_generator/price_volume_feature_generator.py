"""Generate price-volume technical factors from cleaned daily bars.

The generator treats the input daily-bars parquet files as read-only and writes
factor files to a separate partitioned output directory.  The output preserves
the source ``year=YYYY/month=MM/YYYYMMDD.parquet`` layout so downstream jobs can
consume factors by trade date without touching raw/cleaned inputs.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PriceVolumeFeatureGenerateSummary:
    """Aggregate statistics for price-volume factor generation."""

    input_dir: Path
    output_dir: Path
    files_read: int
    files_written: int
    rows_read: int
    rows_written: int
    start_date: int | None
    end_date: int | None
    factor_columns: tuple[str, ...]
    zero_close_rows: int
    zero_volume_mean_rows: int
    duplicate_key_rows: int


class PriceVolumeFeatureGenerator:
    """Generate momentum, bias, volatility, amplitude and volume-price factors.

    Factors currently generated from ``open/high/low/close/pct_chg/vol/amount``:

    - Momentum: ``ret_5``, ``ret_10``, ``ret_20``, ``ret_60``
    - Moving-average bias: ``ma5_bias``, ``ma10_bias``, ``ma20_bias``, ``ma60_bias``
    - Volatility: ``volatility_5``, ``volatility_20``, ``volatility_60``
    - Amplitude: ``amplitude``, ``amplitude_5``, ``amplitude_20``
    - Volume ratios: ``vol_ratio_5``, ``vol_ratio_20``
    - Price-volume rolling correlations: ``corr_price_vol_5``, ``corr_price_vol_20``

    Rolling/shift calculations are performed independently for each ``ts_code``
    after sorting by ``trade_date``.  Source parquet files are never modified.
    """

    DEFAULT_INPUT_DIR = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/daily_bars")
    DEFAULT_OUTPUT_DIR = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/features_data/price_volume_factors")
    DEFAULT_LOG_FILE = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/features_generate/price_volume_features.log")

    KEY_COLUMNS: tuple[str, str] = ("trade_date", "ts_code")
    SOURCE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "pct_chg", "vol", "amount")
    REQUIRED_COLUMNS: tuple[str, ...] = (*KEY_COLUMNS, *SOURCE_COLUMNS)
    FACTOR_COLUMNS: tuple[str, ...] = (
        "ret_5",
        "ret_10",
        "ret_20",
        "ret_60",
        "ma5_bias",
        "ma10_bias",
        "ma20_bias",
        "ma60_bias",
        "volatility_5",
        "volatility_20",
        "volatility_60",
        "amplitude",
        "amplitude_5",
        "amplitude_20",
        "vol_ratio_5",
        "vol_ratio_20",
        "corr_price_vol_5",
        "corr_price_vol_20",
    )
    MAX_LOOKBACK_DAYS = 60

    def __init__(
        self,
        input_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        start_date: str | int | None = None,
        end_date: str | int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.input_dir = Path(input_dir or self.DEFAULT_INPUT_DIR)
        self.output_dir = Path(output_dir or self.DEFAULT_OUTPUT_DIR)
        self.start_date = self._parse_optional_date(start_date, "start_date")
        self.end_date = self._parse_optional_date(end_date, "end_date")
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self._validate_inputs()

    def process(self) -> PriceVolumeFeatureGenerateSummary:
        """Generate factor parquet files and return aggregate processing stats."""

        all_files = self._list_all_daily_bar_files()
        output_files = [file_path for file_path in all_files if self._is_file_date_in_output_range(file_path)]
        if not output_files:
            raise FileNotFoundError(
                f"No parquet files found under {self.input_dir}/year=*/month=*/*.parquet "
                f"for output date range [{self.start_date}, {self.end_date}]"
            )

        read_files = self._select_files_with_history(all_files, output_files)
        output_dates = {self._date_from_file_name(file_path) for file_path in output_files}

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("开始生成价格量特征因子：input=%s, output=%s", self.input_dir, self.output_dir)
        self.logger.info(
            "待输出交易日文件数=%d；为保证 rolling/shift 历史窗口，实际读取文件数=%d；所有输入数据只读。",
            len(output_files),
            len(read_files),
        )

        bars_df = self._load_daily_bars(read_files)
        rows_read = len(bars_df)
        duplicate_key_rows = int(bars_df.duplicated(subset=list(self.KEY_COLUMNS), keep=False).sum())
        if duplicate_key_rows:
            self.logger.warning("输入 daily_bars 存在 %d 行重复 trade_date+ts_code，将全部保留并按原顺序计算。", duplicate_key_rows)

        factor_df = self._generate_features(bars_df)
        output_df = factor_df.loc[factor_df["trade_date"].isin(output_dates), self._output_columns()].copy()
        output_df = output_df.sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)

        files_written = self._write_partitioned_by_date(output_df)
        summary = PriceVolumeFeatureGenerateSummary(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            files_read=len(read_files),
            files_written=files_written,
            rows_read=rows_read,
            rows_written=len(output_df),
            start_date=self.start_date,
            end_date=self.end_date,
            factor_columns=self.FACTOR_COLUMNS,
            zero_close_rows=int((bars_df["close"] == 0).sum()),
            zero_volume_mean_rows=int((output_df[["vol_ratio_5", "vol_ratio_20"]].isna().all(axis=1)).sum()),
            duplicate_key_rows=duplicate_key_rows,
        )
        self._log_summary(summary, output_df)
        return summary

    def _validate_inputs(self) -> None:
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            raise NotADirectoryError(f"Daily bars input directory does not exist or is not a directory: {self.input_dir}")
        if self.start_date is not None and self.end_date is not None and self.start_date > self.end_date:
            raise ValueError(f"start_date must be <= end_date, got {self.start_date} > {self.end_date}")

    def _list_all_daily_bar_files(self) -> list[Path]:
        files = sorted(self.input_dir.glob("year=*/month=*/*.parquet"))
        valid_files: list[Path] = []
        for file_path in files:
            if self._date_from_file_name(file_path) is None:
                self.logger.warning("跳过无法从文件名解析交易日的 daily_bars 文件：%s", file_path)
                continue
            valid_files.append(file_path)
        if not valid_files:
            raise FileNotFoundError(f"No parquet files found under {self.input_dir}/year=*/month=*/*.parquet")
        return valid_files

    def _is_file_date_in_output_range(self, file_path: Path) -> bool:
        file_date = self._date_from_file_name(file_path)
        if file_date is None:
            return False
        if self.start_date is not None and file_date < self.start_date:
            return False
        if self.end_date is not None and file_date > self.end_date:
            return False
        return True

    def _select_files_with_history(self, all_files: Sequence[Path], output_files: Sequence[Path]) -> list[Path]:
        """Read up to 60 prior trade-date files before the first output date."""

        output_file_set = set(output_files)
        output_indices = [idx for idx, file_path in enumerate(all_files) if file_path in output_file_set]
        if not output_indices:
            return []
        first_idx = max(0, min(output_indices) - self.MAX_LOOKBACK_DAYS)
        last_idx = max(output_indices)
        return list(all_files[first_idx : last_idx + 1])

    def _load_daily_bars(self, parquet_files: Sequence[Path]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)
            self._validate_columns(df.columns, self.REQUIRED_COLUMNS, parquet_file)
            prepared = self._prepare_daily_bars(df.loc[:, list(self.REQUIRED_COLUMNS)].copy(), parquet_file)
            frames.append(prepared)
            self.logger.debug("读取 daily_bars 文件完成：%s rows=%d", parquet_file, len(prepared))
        if not frames:
            return pd.DataFrame(columns=list(self.REQUIRED_COLUMNS))
        combined = pd.concat(frames, ignore_index=True)
        self.logger.info("daily_bars 读取完成：files=%d, rows=%d", len(parquet_files), len(combined))
        return combined

    def _prepare_daily_bars(self, daily_df: pd.DataFrame, parquet_file: Path) -> pd.DataFrame:
        prepared = daily_df.copy()
        prepared["trade_date"] = self._normalize_yyyymmdd(prepared["trade_date"], "trade_date", parquet_file)
        prepared["ts_code"] = prepared["ts_code"].astype("string").str.strip()

        invalid_key_mask = prepared["trade_date"].isna() | self._is_null_or_empty(prepared["ts_code"])
        invalid_key_rows = int(invalid_key_mask.sum())
        if invalid_key_rows:
            raise ValueError(f"{parquet_file} has {invalid_key_rows} invalid trade_date/ts_code rows.")

        for column in self.SOURCE_COLUMNS:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        missing_source_rows = int(prepared.loc[:, list(self.SOURCE_COLUMNS)].isna().any(axis=1).sum())
        if missing_source_rows:
            self.logger.warning("%s 存在 %d 行价格量源字段缺失/非数值，相关因子将产生 NaN。", parquet_file, missing_source_rows)
        return prepared

    def _generate_features(self, bars_df: pd.DataFrame) -> pd.DataFrame:
        df = bars_df.copy()
        df = df.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)
        stock_group = df.groupby("ts_code", sort=False, group_keys=False)

        close = df["close"].astype("float64")
        high = df["high"].astype("float64")
        low = df["low"].astype("float64")
        vol = df["vol"].astype("float64")

        for window in (5, 10, 20, 60):
            shifted_close = stock_group["close"].shift(window).replace(0, np.nan)
            df[f"ret_{window}"] = close.div(shifted_close) - 1

            ma = stock_group["close"].transform(lambda series: series.rolling(window=window, min_periods=window).mean())
            df[f"ma{window}_bias"] = close.div(ma.replace(0, np.nan)) - 1

        for window in (5, 20, 60):
            df[f"volatility_{window}"] = stock_group["pct_chg"].transform(
                lambda series: series.rolling(window=window, min_periods=window).std()
            )

        df["amplitude"] = (high - low).div(close.replace(0, np.nan))
        for window in (5, 20):
            df[f"amplitude_{window}"] = stock_group["amplitude"].transform(
                lambda series: series.rolling(window=window, min_periods=window).mean()
            )
            volume_mean = stock_group["vol"].transform(lambda series: series.rolling(window=window, min_periods=window).mean())
            df[f"vol_ratio_{window}"] = vol.div(volume_mean.replace(0, np.nan))
            df[f"corr_price_vol_{window}"] = self._rolling_corr(df, window)

        nan_counts = df.loc[:, list(self.FACTOR_COLUMNS)].isna().sum().to_dict()
        self.logger.info("价格量因子计算完成：rows=%d, factor_nan_counts=%s", len(df), nan_counts)
        return df

    def _rolling_corr(self, df: pd.DataFrame, window: int) -> pd.Series:
        corr = pd.Series(index=df.index, dtype="float64")
        for _, group in df.groupby("ts_code", sort=False):
            corr.loc[group.index] = group["close"].rolling(window=window, min_periods=window).corr(group["vol"])
        return corr

    def _write_partitioned_by_date(self, output_df: pd.DataFrame) -> int:
        files_written = 0
        for trade_date, date_df in output_df.groupby("trade_date", sort=True):
            trade_date_int = int(trade_date)
            trade_date_text = f"{trade_date_int:08d}"
            output_file = self.output_dir / f"year={trade_date_text[:4]}" / f"month={trade_date_text[4:6]}" / f"{trade_date_text}.parquet"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            date_df.to_parquet(output_file, index=False)
            files_written += 1
            self.logger.info("价格量因子文件生成完成：trade_date=%s, rows=%d -> %s", trade_date_text, len(date_df), output_file)
        return files_written

    def _output_columns(self) -> list[str]:
        return [*self.KEY_COLUMNS, *self.SOURCE_COLUMNS, *self.FACTOR_COLUMNS]

    def _log_summary(self, summary: PriceVolumeFeatureGenerateSummary, output_df: pd.DataFrame) -> None:
        self.logger.info(
            "价格量因子生成汇总：files_read=%d, files_written=%d, rows_read=%d, rows_written=%d, "
            "start_date=%s, end_date=%s, factors=%d, output=%s",
            summary.files_read,
            summary.files_written,
            summary.rows_read,
            summary.rows_written,
            summary.start_date,
            summary.end_date,
            len(summary.factor_columns),
            summary.output_dir,
        )
        if summary.duplicate_key_rows:
            self.logger.warning("输入数据存在 %d 行重复 trade_date+ts_code，请关注上游 daily_bars 清洗结果。", summary.duplicate_key_rows)
        if summary.zero_close_rows:
            self.logger.warning("输入数据存在 %d 行 close=0，收益率/偏离/振幅相关因子已置为 NaN。", summary.zero_close_rows)
        if not output_df.empty:
            factor_nan_counts = output_df.loc[:, list(self.FACTOR_COLUMNS)].isna().sum().to_dict()
            self.logger.info("输出因子缺失值统计：%s", factor_nan_counts)

    @staticmethod
    def _validate_columns(columns: Iterable[str], required_columns: Sequence[str], file_path: Path) -> None:
        column_set = set(columns)
        missing_columns = [column for column in required_columns if column not in column_set]
        if missing_columns:
            raise ValueError(f"{file_path} is missing required columns: {missing_columns}")

    @classmethod
    def _normalize_yyyymmdd(cls, series: pd.Series, column_name: str, file_path: Path) -> pd.Series:
        normalized = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
        parsed = pd.to_datetime(normalized, format="%Y%m%d", errors="coerce")
        invalid_rows = int(parsed.isna().sum())
        if invalid_rows:
            raise ValueError(f"{file_path} has {invalid_rows} invalid {column_name} rows that cannot be parsed as YYYYMMDD.")
        return parsed.dt.strftime("%Y%m%d").astype("int32")

    @staticmethod
    def _parse_optional_date(value: str | int | None, name: str) -> int | None:
        if value is None or value == "":
            return None
        text = str(value).strip()
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"{name} must be in YYYYMMDD format, got {value!r}")
        return int(parsed.strftime("%Y%m%d"))

    @staticmethod
    def _is_null_or_empty(series: pd.Series) -> pd.Series:
        return series.isna() | series.astype("string").str.strip().eq("").fillna(False)

    @staticmethod
    def _date_from_file_name(file_path: Path) -> int | None:
        try:
            return int(file_path.stem)
        except ValueError:
            return None


def configure_logging(log_level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure console and file logging for CLI usage."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    target_log_file = Path(log_file) if log_file is not None else PriceVolumeFeatureGenerator.DEFAULT_LOG_FILE
    target_log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(target_log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate price-volume factors from cleaned A-share daily bars.")
    parser.add_argument("--input-dir", default=str(PriceVolumeFeatureGenerator.DEFAULT_INPUT_DIR), help="Cleaned daily_bars directory.")
    parser.add_argument(
        "--output-dir",
        default=str(PriceVolumeFeatureGenerator.DEFAULT_OUTPUT_DIR),
        help="Output directory for partitioned price-volume factors.",
    )
    parser.add_argument("--start-date", default=None, help="Optional inclusive output start date in YYYYMMDD format.")
    parser.add_argument("--end-date", default=None, help="Optional inclusive output end date in YYYYMMDD format.")
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    parser.add_argument("--log-file", default=str(PriceVolumeFeatureGenerator.DEFAULT_LOG_FILE), help="Log file path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> PriceVolumeFeatureGenerateSummary:
    args = parse_args(argv)
    configure_logging(log_level=args.log_level, log_file=args.log_file)
    generator = PriceVolumeFeatureGenerator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    return generator.process()


if __name__ == "__main__":
    # 使用方法：
    # PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
    # python -m utils.features_generator.price_volume_feature_generator \
    #   --output-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/features_data/price_volume_factors" \
    #   --log-file "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/features_generate/price_volume_features.log"
    main()


__all__ = [
    "PriceVolumeFeatureGenerateSummary",
    "PriceVolumeFeatureGenerator",
    "configure_logging",
    "main",
    "parse_args",
]
