"""Generate daily fundamental factors from cleaned A-share fundamentals data.

The generator treats cleaned source datasets as read-only.  It converts
announcement-date based fundamental records into daily features by applying the
latest disclosed record available on each trade date, then writes partitioned
factor files under ``data/features_data`` using the same
``year=YYYY/month=MM/YYYYMMDD.parquet`` layout as ``cleaned_data/daily_bars``.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class FundamentalFeatureGenerateSummary:
    """Aggregate statistics for fundamental factor generation."""

    fundamentals_input_file: Path
    daily_bars_input_dir: Path
    output_dir: Path
    daily_bar_files_read: int
    files_written: int
    fundamentals_rows_read: int
    daily_bar_rows_read: int
    rows_written: int
    start_date: int | None
    end_date: int | None
    factor_columns: tuple[str, ...]
    duplicate_daily_key_rows: int
    rows_without_available_fundamentals: int


class FundamentalFeatureGenerator:
    """Generate daily fundamental factors and cross-sectional ranks.

    Source factors are preserved directly from cleaned ``fundamentals``:
    ``roe``, ``roa``, ``or_yoy``, ``gross_margin``, ``debt_to_assets``,
    ``eps`` and ``bps``.

    Derived factors:

    - Profit quality: ``roe_roa_gap = roe - roa``
    - Daily cross-sectional percentile ranks: ``roe_rank``, ``roa_rank`` and
      ``revenue_yoy_rank``.  Higher source values receive higher percentile
      ranks; missing source values keep missing ranks.

    For every daily-bars trade date, each stock uses the latest fundamental
    record whose ``ann_date`` is not later than the trade date.  If multiple
    records share the same announcement date, the latest ``end_date`` is used.
    Source parquet files are never modified.
    """

    DEFAULT_FUNDAMENTALS_FILE = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/fundamentals/fundamentals.parquet"
    )
    DEFAULT_DAILY_BARS_DIR = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/daily_bars")
    DEFAULT_OUTPUT_DIR = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/features_data/fundamental_factors")
    DEFAULT_LOG_FILE = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/features_generate/fundamental_features.log")

    KEY_COLUMNS: tuple[str, str] = ("trade_date", "ts_code")
    FUNDAMENTAL_KEY_COLUMNS: tuple[str, str, str] = ("ts_code", "ann_date", "end_date")
    SOURCE_COLUMNS: tuple[str, ...] = (
        "roe",
        "roa",
        "or_yoy",
        "gross_margin",
        "debt_to_assets",
        "eps",
        "bps",
    )
    DERIVED_COLUMNS: tuple[str, ...] = ("roe_roa_gap", "roe_rank", "roa_rank", "revenue_yoy_rank")
    FACTOR_COLUMNS: tuple[str, ...] = (*SOURCE_COLUMNS, *DERIVED_COLUMNS)
    FUNDAMENTALS_REQUIRED_COLUMNS: tuple[str, ...] = (*FUNDAMENTAL_KEY_COLUMNS, *SOURCE_COLUMNS)
    DAILY_BARS_REQUIRED_COLUMNS: tuple[str, ...] = KEY_COLUMNS

    def __init__(
        self,
        fundamentals_file: str | Path | None = None,
        daily_bars_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        start_date: str | int | None = None,
        end_date: str | int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.fundamentals_file = Path(fundamentals_file or self.DEFAULT_FUNDAMENTALS_FILE)
        self.daily_bars_dir = Path(daily_bars_dir or self.DEFAULT_DAILY_BARS_DIR)
        self.output_dir = Path(output_dir or self.DEFAULT_OUTPUT_DIR)
        self.start_date = self._parse_optional_date(start_date, "start_date")
        self.end_date = self._parse_optional_date(end_date, "end_date")
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self._validate_inputs()

    def process(self) -> FundamentalFeatureGenerateSummary:
        """Generate daily fundamental factor parquet files and return stats."""

        daily_bar_files = self._list_output_daily_bar_files()
        if not daily_bar_files:
            raise FileNotFoundError(
                f"No daily_bars parquet files found under {self.daily_bars_dir}/year=*/month=*/*.parquet "
                f"for output date range [{self.start_date}, {self.end_date}]"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            "开始生成财务特征因子：fundamentals=%s, daily_bars=%s, output=%s",
            self.fundamentals_file,
            self.daily_bars_dir,
            self.output_dir,
        )
        self.logger.info("输入 fundamentals 与 daily_bars 均按只读处理，生成结果仅写入 features_data 对应目录。")

        fundamentals_df = self._load_fundamentals()
        files_written = 0
        daily_bar_rows_read = 0
        rows_written = 0
        duplicate_daily_key_rows = 0
        rows_without_available_fundamentals = 0

        current_fundamentals = pd.DataFrame(columns=[*self.FUNDAMENTAL_KEY_COLUMNS, *self.SOURCE_COLUMNS])
        next_fundamental_idx = 0
        fundamental_count = len(fundamentals_df)

        for daily_bar_file in daily_bar_files:
            trade_date = self._date_from_file_name(daily_bar_file)
            if trade_date is None:
                continue

            next_fundamental_idx, current_fundamentals = self._advance_current_fundamentals(
                fundamentals_df,
                next_fundamental_idx,
                trade_date,
                current_fundamentals,
            )
            if next_fundamental_idx == fundamental_count and fundamental_count:
                self.logger.debug("截至 trade_date=%s 已加载全部可用 fundamentals 记录。", trade_date)

            daily_universe = self._load_daily_universe(daily_bar_file)
            daily_bar_rows_read += len(daily_universe)
            duplicate_daily_key_rows += int(daily_universe.duplicated(subset=list(self.KEY_COLUMNS), keep=False).sum())

            output_df = self._generate_daily_features(daily_universe, current_fundamentals)
            rows_without_available_fundamentals += int(output_df["ann_date"].isna().sum())
            rows_written += len(output_df)
            self._write_daily_file(output_df, trade_date)
            files_written += 1

        summary = FundamentalFeatureGenerateSummary(
            fundamentals_input_file=self.fundamentals_file,
            daily_bars_input_dir=self.daily_bars_dir,
            output_dir=self.output_dir,
            daily_bar_files_read=len(daily_bar_files),
            files_written=files_written,
            fundamentals_rows_read=len(fundamentals_df),
            daily_bar_rows_read=daily_bar_rows_read,
            rows_written=rows_written,
            start_date=self.start_date,
            end_date=self.end_date,
            factor_columns=self.FACTOR_COLUMNS,
            duplicate_daily_key_rows=duplicate_daily_key_rows,
            rows_without_available_fundamentals=rows_without_available_fundamentals,
        )
        self._log_summary(summary)
        return summary

    def _validate_inputs(self) -> None:
        if not self.fundamentals_file.exists() or not self.fundamentals_file.is_file():
            raise FileNotFoundError(f"Fundamentals input file does not exist: {self.fundamentals_file}")
        if not self.daily_bars_dir.exists() or not self.daily_bars_dir.is_dir():
            raise NotADirectoryError(f"Daily bars input directory does not exist or is not a directory: {self.daily_bars_dir}")
        if self.start_date is not None and self.end_date is not None and self.start_date > self.end_date:
            raise ValueError(f"start_date must be <= end_date, got {self.start_date} > {self.end_date}")

    def _load_fundamentals(self) -> pd.DataFrame:
        df = pd.read_parquet(self.fundamentals_file, columns=list(self.FUNDAMENTALS_REQUIRED_COLUMNS))
        self._validate_columns(df.columns, self.FUNDAMENTALS_REQUIRED_COLUMNS, self.fundamentals_file)
        prepared = self._prepare_fundamentals(df)
        self.logger.info("fundamentals 读取完成：rows=%d, columns=%d", len(prepared), len(prepared.columns))
        return prepared

    def _prepare_fundamentals(self, fundamentals_df: pd.DataFrame) -> pd.DataFrame:
        prepared = fundamentals_df.copy()
        prepared["ts_code"] = prepared["ts_code"].astype("string").str.strip()
        prepared["ann_date"] = self._normalize_yyyymmdd(prepared["ann_date"], "ann_date", self.fundamentals_file)
        prepared["end_date"] = self._normalize_yyyymmdd(prepared["end_date"], "end_date", self.fundamentals_file)

        invalid_key_mask = self._is_null_or_empty(prepared["ts_code"])
        invalid_key_rows = int(invalid_key_mask.sum())
        if invalid_key_rows:
            raise ValueError(f"{self.fundamentals_file} has {invalid_key_rows} invalid ts_code rows.")

        for column in self.SOURCE_COLUMNS:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        missing_source_rows = int(prepared.loc[:, list(self.SOURCE_COLUMNS)].isna().any(axis=1).sum())
        if missing_source_rows:
            self.logger.warning("fundamentals 存在 %d 行财务源字段缺失/非数值，相关因子和排名将产生 NaN。", missing_source_rows)

        duplicate_key_rows = int(prepared.duplicated(subset=list(self.FUNDAMENTAL_KEY_COLUMNS), keep=False).sum())
        if duplicate_key_rows:
            self.logger.warning("输入 fundamentals 存在 %d 行重复 ts_code+ann_date+end_date，将按排序后最后一条更新日频状态。", duplicate_key_rows)

        return prepared.sort_values(["ann_date", "end_date", "ts_code"], kind="mergesort").reset_index(drop=True)

    def _list_output_daily_bar_files(self) -> list[Path]:
        files = sorted(self.daily_bars_dir.glob("year=*/month=*/*.parquet"))
        output_files: list[Path] = []
        for file_path in files:
            file_date = self._date_from_file_name(file_path)
            if file_date is None:
                self.logger.warning("跳过无法从文件名解析交易日的 daily_bars 文件：%s", file_path)
                continue
            if self.start_date is not None and file_date < self.start_date:
                continue
            if self.end_date is not None and file_date > self.end_date:
                continue
            output_files.append(file_path)
        self.logger.info("财务因子待输出交易日文件数=%d", len(output_files))
        return output_files

    def _advance_current_fundamentals(
        self,
        fundamentals_df: pd.DataFrame,
        start_idx: int,
        trade_date: int,
        current_fundamentals: pd.DataFrame,
    ) -> tuple[int, pd.DataFrame]:
        end_idx = int(fundamentals_df["ann_date"].searchsorted(trade_date, side="right"))
        if end_idx == start_idx:
            return start_idx, current_fundamentals

        new_records = fundamentals_df.iloc[start_idx:end_idx]
        if current_fundamentals.empty:
            combined = new_records.copy()
        else:
            combined = pd.concat([current_fundamentals, new_records], ignore_index=True)
        combined = combined.sort_values(["ts_code", "ann_date", "end_date"], kind="mergesort")
        latest = combined.drop_duplicates(subset=["ts_code"], keep="last").reset_index(drop=True)
        self.logger.debug(
            "更新财务日频状态：trade_date=%s, new_records=%d, active_stocks=%d",
            trade_date,
            len(new_records),
            len(latest),
        )
        return end_idx, latest

    def _load_daily_universe(self, parquet_file: Path) -> pd.DataFrame:
        df = pd.read_parquet(parquet_file, columns=list(self.DAILY_BARS_REQUIRED_COLUMNS))
        self._validate_columns(df.columns, self.DAILY_BARS_REQUIRED_COLUMNS, parquet_file)
        prepared = df.copy()
        prepared["trade_date"] = self._normalize_yyyymmdd(prepared["trade_date"], "trade_date", parquet_file)
        prepared["ts_code"] = prepared["ts_code"].astype("string").str.strip()

        invalid_key_mask = prepared["trade_date"].isna() | self._is_null_or_empty(prepared["ts_code"])
        invalid_key_rows = int(invalid_key_mask.sum())
        if invalid_key_rows:
            raise ValueError(f"{parquet_file} has {invalid_key_rows} invalid trade_date/ts_code rows.")
        return prepared

    def _generate_daily_features(self, daily_universe: pd.DataFrame, current_fundamentals: pd.DataFrame) -> pd.DataFrame:
        df = daily_universe.merge(current_fundamentals, on="ts_code", how="left", validate="many_to_one")
        df["ann_date"] = df["ann_date"].astype("Int32")
        df["end_date"] = df["end_date"].astype("Int32")
        df["roe_roa_gap"] = df["roe"] - df["roa"]
        df["roe_rank"] = df["roe"].rank(method="average", ascending=True, pct=True)
        df["roa_rank"] = df["roa"].rank(method="average", ascending=True, pct=True)
        df["revenue_yoy_rank"] = df["or_yoy"].rank(method="average", ascending=True, pct=True)
        return df.loc[:, self._output_columns()].sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)

    def _write_daily_file(self, output_df: pd.DataFrame, trade_date: int) -> Path:
        trade_date_text = f"{int(trade_date):08d}"
        output_file = self.output_dir / f"year={trade_date_text[:4]}" / f"month={trade_date_text[4:6]}" / f"{trade_date_text}.parquet"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_parquet(output_file, index=False)
        factor_nan_counts = output_df.loc[:, list(self.FACTOR_COLUMNS)].isna().sum().to_dict()
        self.logger.info(
            "财务因子文件生成完成：trade_date=%s, rows=%d, factor_nan_counts=%s -> %s",
            trade_date_text,
            len(output_df),
            factor_nan_counts,
            output_file,
        )
        return output_file

    def _output_columns(self) -> list[str]:
        return [*self.KEY_COLUMNS, "ann_date", "end_date", *self.FACTOR_COLUMNS]

    def _log_summary(self, summary: FundamentalFeatureGenerateSummary) -> None:
        self.logger.info(
            "财务因子生成汇总：fundamentals_rows_read=%d, daily_bar_files_read=%d, daily_bar_rows_read=%d, "
            "files_written=%d, rows_written=%d, start_date=%s, end_date=%s, factors=%d, output=%s",
            summary.fundamentals_rows_read,
            summary.daily_bar_files_read,
            summary.daily_bar_rows_read,
            summary.files_written,
            summary.rows_written,
            summary.start_date,
            summary.end_date,
            len(summary.factor_columns),
            summary.output_dir,
        )
        if summary.duplicate_daily_key_rows:
            self.logger.warning("输入 daily_bars 存在 %d 行重复 trade_date+ts_code，请关注上游清洗结果。", summary.duplicate_daily_key_rows)
        if summary.rows_without_available_fundamentals:
            self.logger.warning(
                "输出中有 %d 行在对应 trade_date 尚无可用 ann_date<=trade_date 的财务数据，财务因子已置为 NaN。",
                summary.rows_without_available_fundamentals,
            )

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
    target_log_file = Path(log_file) if log_file is not None else FundamentalFeatureGenerator.DEFAULT_LOG_FILE
    target_log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(target_log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily fundamental factors from cleaned A-share fundamentals data.")
    parser.add_argument(
        "--fundamentals-file",
        default=str(FundamentalFeatureGenerator.DEFAULT_FUNDAMENTALS_FILE),
        help="Cleaned fundamentals parquet file.",
    )
    parser.add_argument(
        "--daily-bars-dir",
        default=str(FundamentalFeatureGenerator.DEFAULT_DAILY_BARS_DIR),
        help="Cleaned daily_bars directory used as trading calendar and stock universe.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(FundamentalFeatureGenerator.DEFAULT_OUTPUT_DIR),
        help="Output directory for partitioned fundamental factors.",
    )
    parser.add_argument("--start-date", default=None, help="Optional inclusive output start date in YYYYMMDD format.")
    parser.add_argument("--end-date", default=None, help="Optional inclusive output end date in YYYYMMDD format.")
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    parser.add_argument("--log-file", default=str(FundamentalFeatureGenerator.DEFAULT_LOG_FILE), help="Log file path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> FundamentalFeatureGenerateSummary:
    args = parse_args(argv)
    configure_logging(log_level=args.log_level, log_file=args.log_file)
    generator = FundamentalFeatureGenerator(
        fundamentals_file=args.fundamentals_file,
        daily_bars_dir=args.daily_bars_dir,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    return generator.process()


if __name__ == "__main__":
    # 使用方法：
    # PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
    # python -m utils.features_generator.fundamental_feature_generator \
    #   --output-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/features_data/fundamental_factors" \
    #   --log-file "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/features_generate/fundamental_features.log"
    main()


__all__ = [
    "FundamentalFeatureGenerateSummary",
    "FundamentalFeatureGenerator",
    "configure_logging",
    "main",
    "parse_args",
]
