"""Daily bars data quality checker and non-destructive cleaner.

This module checks the raw partitioned daily-bars parquet dataset without
modifying it. Cleaned files are written to a separate output directory while
preserving the original ``year=YYYY/month=MM/*.parquet`` layout.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class DailyBarsProcessSummary:
    """Aggregate quality-check and cleaning statistics for DailyBars."""

    files_processed: int
    rows_read: int
    rows_written: int
    missing_rows_removed: int
    duplicate_rows_removed: int
    files_with_missing: int
    files_with_duplicates: int

    @property
    def has_missing_issue(self) -> bool:
        """Whether any rows containing missing values were found."""

        return self.missing_rows_removed > 0

    @property
    def has_duplicate_issue(self) -> bool:
        """Whether any duplicate ``ts_code`` + ``trade_date`` rows were found."""

        return self.duplicate_rows_removed > 0


class DailyBarsCleaner:
    """Check and clean partitioned DailyBars parquet files.

    The cleaner performs two mandatory data-quality checks:

    1. ``ts_code`` + ``trade_date`` uniqueness. Duplicate rows are removed from
       the cleaned output, keeping the first occurrence encountered in the
       sorted parquet-file order.
    2. Missing values in ``open``/``high``/``low``/``close``. Rows containing
       missing OHLC values are removed from the cleaned output. ``pre_close``,
       ``change`` and ``pct_chg`` are allowed to be missing.

    Raw input parquet files are only read; they are never modified in place.
    """

    DEFAULT_INPUT_DIR = Path("/opt/tiger/qyd/quant_llm/A_stocks_all_data/daily_bars")
    DEFAULT_OUTPUT_DIR = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/daily_bars")

    KEY_COLUMNS: tuple[str, str] = ("ts_code", "trade_date")
    VALUE_COLUMNS: tuple[str, ...] = (
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
    )
    REQUIRED_COLUMNS: tuple[str, ...] = (*KEY_COLUMNS, *VALUE_COLUMNS)
    MISSING_CHECK_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")

    def __init__(
        self,
        input_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.input_dir = Path(input_dir or self.DEFAULT_INPUT_DIR)
        self.output_dir = Path(output_dir or self.DEFAULT_OUTPUT_DIR)
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        if not self.input_dir.exists():
            raise FileNotFoundError(f"DailyBars input directory does not exist: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"DailyBars input path is not a directory: {self.input_dir}")

    def process(self) -> DailyBarsProcessSummary:
        """Run quality checks and write cleaned parquet files.

        Returns
        -------
        DailyBarsProcessSummary
            Aggregate statistics describing data-quality issues and output row
            counts.
        """

        parquet_files = self._list_parquet_files()
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found under {self.input_dir}/year=*/month=*/*.parquet")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("开始处理 DailyBars 数据：input=%s, output=%s", self.input_dir, self.output_dir)
        self.logger.info("发现 %d 个 parquet 文件，原始数据只读，清洗结果将单独落盘。", len(parquet_files))

        rows_read = 0
        rows_written = 0
        missing_rows_removed = 0
        duplicate_rows_removed = 0
        files_with_missing = 0
        files_with_duplicates = 0
        seen_keys: set[tuple[str, str]] = set()

        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)
            self._validate_columns(df.columns, parquet_file)

            original_count = len(df)
            rows_read += original_count

            cleaned, file_missing_count, file_duplicate_count = self._clean_dataframe(df, seen_keys)

            if file_missing_count > 0:
                files_with_missing += 1
            if file_duplicate_count > 0:
                files_with_duplicates += 1

            missing_rows_removed += file_missing_count
            duplicate_rows_removed += file_duplicate_count
            rows_written += len(cleaned)

            output_file = self._output_path_for(parquet_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            cleaned.to_parquet(output_file, index=False)

            self.logger.info(
                "文件处理完成：%s | 原始=%d, 缺失删除=%d, 重复删除=%d, 输出=%d -> %s",
                parquet_file,
                original_count,
                file_missing_count,
                file_duplicate_count,
                len(cleaned),
                output_file,
            )

        summary = DailyBarsProcessSummary(
            files_processed=len(parquet_files),
            rows_read=rows_read,
            rows_written=rows_written,
            missing_rows_removed=missing_rows_removed,
            duplicate_rows_removed=duplicate_rows_removed,
            files_with_missing=files_with_missing,
            files_with_duplicates=files_with_duplicates,
        )
        self._log_summary(summary)
        return summary

    def _list_parquet_files(self) -> list[Path]:
        return sorted(self.input_dir.glob("year=*/month=*/*.parquet"))

    def _validate_columns(self, columns: Iterable[str], parquet_file: Path) -> None:
        column_set = set(columns)
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in column_set]
        if missing_columns:
            raise ValueError(f"{parquet_file} is missing required columns: {missing_columns}")

    def _clean_dataframe(self, df: pd.DataFrame, seen_keys: set[tuple[str, str]]) -> tuple[pd.DataFrame, int, int]:
        cleaned = df.copy()

        missing_mask = cleaned.loc[:, self.MISSING_CHECK_COLUMNS].isna().any(axis=1)
        missing_count = int(missing_mask.sum())
        if missing_count:
            cleaned = cleaned.loc[~missing_mask].copy()

        duplicate_mask = cleaned.duplicated(subset=list(self.KEY_COLUMNS), keep="first")

        current_keys = self._make_key_tuples(cleaned.loc[:, self.KEY_COLUMNS])
        global_duplicate_mask = pd.Series([key in seen_keys for key in current_keys], index=cleaned.index)

        combined_duplicate_mask = duplicate_mask | global_duplicate_mask
        duplicate_count = int(combined_duplicate_mask.sum())
        if duplicate_count:
            cleaned = cleaned.loc[~combined_duplicate_mask].copy()
            current_keys = self._make_key_tuples(cleaned.loc[:, self.KEY_COLUMNS])

        seen_keys.update(current_keys)
        return cleaned, missing_count, duplicate_count

    @staticmethod
    def _make_key_tuples(keys_df: pd.DataFrame) -> list[tuple[str, str]]:
        normalized = keys_df.astype({"ts_code": "string", "trade_date": "string"})
        return list(normalized.itertuples(index=False, name=None))

    def _output_path_for(self, input_file: Path) -> Path:
        return self.output_dir / input_file.relative_to(self.input_dir)

    def _log_summary(self, summary: DailyBarsProcessSummary) -> None:
        self.logger.info(
            "DailyBars 清洗汇总：files=%d, rows_read=%d, rows_written=%d, "
            "missing_rows_removed=%d, duplicate_rows_removed=%d",
            summary.files_processed,
            summary.rows_read,
            summary.rows_written,
            summary.missing_rows_removed,
            summary.duplicate_rows_removed,
        )
        if summary.has_duplicate_issue:
            self.logger.warning(
                "唯一性检查发现问题：共删除 %d 行重复 ts_code+trade_date，涉及 %d 个文件。",
                summary.duplicate_rows_removed,
                summary.files_with_duplicates,
            )
        else:
            self.logger.info("唯一性检查通过：未发现重复 ts_code+trade_date。")

        if summary.has_missing_issue:
            self.logger.warning(
                "缺失值检查发现问题：共删除 %d 行缺失数据，涉及 %d 个文件。",
                summary.missing_rows_removed,
                summary.files_with_missing,
            )
        else:
            self.logger.info("缺失值检查通过：所有必需字段均无缺失。")


def configure_logging(log_level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure console logging and optional file logging for CLI usage."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check and clean DailyBars parquet data without modifying raw files.")
    parser.add_argument("--input-dir", default=str(DailyBarsCleaner.DEFAULT_INPUT_DIR), help="Raw daily_bars directory.")
    parser.add_argument(
        "--output-dir",
        default=str(DailyBarsCleaner.DEFAULT_OUTPUT_DIR),
        help="Directory for cleaned daily_bars parquet files.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    parser.add_argument("--log-file", default=None, help="Optional log file path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> DailyBarsProcessSummary:
    args = parse_args(argv)
    configure_logging(log_level=args.log_level, log_file=args.log_file)
    cleaner = DailyBarsCleaner(input_dir=args.input_dir, output_dir=args.output_dir)
    return cleaner.process()


if __name__ == "__main__":
    # 使用方法
    # PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src"  python -m utils.dataset_cleaner.daily_bars_cleaner  --input-dir "/opt/tiger/qyd/quant_llm/A_stocks_all_data/daily_bars"  --output-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/daily_bars"  --log-file "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/daily_bars_cleaning.log"
    main()


__all__ = ["DailyBarsProcessSummary", "DailyBarsCleaner", "configure_logging", "main", "parse_args"]
