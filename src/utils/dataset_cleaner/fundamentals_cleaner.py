"""Fundamentals data quality checker and non-destructive cleaner.

This module checks the aggregated ``fundamentals.parquet`` file without
modifying the raw data. The cleaned parquet is written to the project-level
``data/cleaned_data/fundamentals`` directory.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class FundamentalsProcessSummary:
    """Quality-check and cleaning statistics for fundamentals data."""

    input_file: Path
    output_file: Path
    rows_read: int
    rows_written: int
    missing_rows_found: int
    invalid_date_rows_removed: int
    duplicate_rows_removed: int

    @property
    def has_missing_issue(self) -> bool:
        """Whether any rows with missing required values were found."""

        return self.missing_rows_found > 0

    @property
    def has_date_issue(self) -> bool:
        """Whether invalid ``ann_date`` / ``end_date`` rows were found."""

        return self.invalid_date_rows_removed > 0

    @property
    def has_duplicate_issue(self) -> bool:
        """Whether duplicate ``ts_code`` + ``ann_date`` + ``end_date`` keys were found."""

        return self.duplicate_rows_removed > 0


class FundamentalsCleaner:
    """Check and clean the aggregated fundamentals parquet file.

    Cleaning rules are intentionally narrow and non-destructive:

    1. Keep records with missing factor values unchanged; missing-value removal
       is intentionally disabled for this dataset.
    2. Drop records whose dates cannot be parsed or whose ``ann_date`` is not
       greater than ``end_date``.
    3. Treat ``ts_code`` + ``ann_date`` + ``end_date`` as the primary key. If
       duplicated, keep the last row in the original file order.

    The original parquet file is only read; it is never overwritten.
    """

    DEFAULT_INPUT_FILE = Path("/opt/tiger/qyd/quant_llm/A_stocks_all_data/fundamentals/fundamentals.parquet")
    DEFAULT_OUTPUT_FILE = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/fundamentals/fundamentals.parquet"
    )
    DEFAULT_LOG_FILE = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/fundamentals_cleaning.log")

    KEY_COLUMNS: tuple[str, str, str] = ("ts_code", "ann_date", "end_date")
    FACTOR_COLUMNS: tuple[str, ...] = (
        "roe",
        "roa",
        "revenue_yoy",
        "debt_ratio",
        "gross_margin",
        "eps",
        "bps",
    )
    REQUIRED_COLUMNS: tuple[str, ...] = (*KEY_COLUMNS, *FACTOR_COLUMNS)

    def __init__(
        self,
        input_file: str | Path | None = None,
        output_file: str | Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.input_file = Path(input_file or self.DEFAULT_INPUT_FILE)
        self.output_file = Path(output_file or self.DEFAULT_OUTPUT_FILE)
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        if not self.input_file.exists():
            raise FileNotFoundError(f"Fundamentals input file does not exist: {self.input_file}")
        if not self.input_file.is_file():
            raise ValueError(f"Fundamentals input path is not a file: {self.input_file}")

    def process(self) -> FundamentalsProcessSummary:
        """Run quality checks and write the cleaned parquet file."""

        self.logger.info("开始处理 fundamentals 数据：input=%s, output=%s", self.input_file, self.output_file)
        self.logger.info("原始 fundamentals parquet 只读，清洗结果将单独落盘。")

        df = pd.read_parquet(self.input_file)
        self._validate_columns(df.columns)

        rows_read = len(df)
        missing_rows_found = self._count_missing_rows(df)
        cleaned, invalid_date_rows_removed, duplicate_rows_removed = self._clean_dataframe(df)

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_parquet(self.output_file, index=False)

        summary = FundamentalsProcessSummary(
            input_file=self.input_file,
            output_file=self.output_file,
            rows_read=rows_read,
            rows_written=len(cleaned),
            missing_rows_found=missing_rows_found,
            invalid_date_rows_removed=invalid_date_rows_removed,
            duplicate_rows_removed=duplicate_rows_removed,
        )
        self._log_summary(summary)
        return summary

    def _validate_columns(self, columns: Iterable[str]) -> None:
        column_set = set(columns)
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in column_set]
        if missing_columns:
            raise ValueError(f"{self.input_file} is missing required columns: {missing_columns}")

    def _clean_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
        cleaned = df.copy()

        invalid_date_mask = self._invalid_date_mask(cleaned)
        invalid_date_count = int(invalid_date_mask.sum())
        if invalid_date_count:
            cleaned = cleaned.loc[~invalid_date_mask].copy()

        duplicate_mask = cleaned.duplicated(subset=list(self.KEY_COLUMNS), keep="last")
        duplicate_count = int(duplicate_mask.sum())
        if duplicate_count:
            cleaned = cleaned.loc[~duplicate_mask].copy()

        return cleaned, invalid_date_count, duplicate_count

    def _count_missing_rows(self, df: pd.DataFrame) -> int:
        return int(self._required_missing_mask(df).sum())

    def _required_missing_mask(self, df: pd.DataFrame) -> pd.Series:
        missing_mask = df.loc[:, self.FACTOR_COLUMNS].isna().any(axis=1)
        for column in self.KEY_COLUMNS:
            missing_mask = missing_mask | self._is_null_or_empty(df[column])
        return missing_mask

    @staticmethod
    def _invalid_date_mask(df: pd.DataFrame) -> pd.Series:
        ann_date = pd.to_datetime(df["ann_date"].astype("string"), format="%Y%m%d", errors="coerce")
        end_date = pd.to_datetime(df["end_date"].astype("string"), format="%Y%m%d", errors="coerce")
        return ann_date.isna() | end_date.isna() | (ann_date <= end_date)

    @staticmethod
    def _is_null_or_empty(series: pd.Series) -> pd.Series:
        return series.isna() | series.astype("string").str.strip().eq("").fillna(False)

    def _log_summary(self, summary: FundamentalsProcessSummary) -> None:
        self.logger.info(
            "fundamentals 清洗汇总：rows_read=%d, rows_written=%d, "
            "missing_rows_found=%d, missing_rows_removed=0, invalid_date_rows_removed=%d, "
            "duplicate_rows_removed=%d, output=%s",
            summary.rows_read,
            summary.rows_written,
            summary.missing_rows_found,
            summary.invalid_date_rows_removed,
            summary.duplicate_rows_removed,
            summary.output_file,
        )

        if summary.has_duplicate_issue:
            self.logger.warning(
                "唯一性检查发现问题：共删除 %d 行重复 ts_code+ann_date+end_date，重复主键仅保留最后一行。",
                summary.duplicate_rows_removed,
            )
        else:
            self.logger.info("唯一性检查通过：未发现重复 ts_code+ann_date+end_date。")

        if summary.has_missing_issue:
            self.logger.info(
                "缺失值处理已关闭：发现 %d 行必需字段缺失的记录，但未因缺失值删除任何记录。",
                summary.missing_rows_found,
            )
        else:
            self.logger.info("缺失值处理已关闭：未发现必需字段缺失记录。")

        if summary.has_date_issue:
            self.logger.warning(
                "日期检查发现问题：共删除 %d 行 ann_date/end_date 无法解析或 ann_date <= end_date 的记录。",
                summary.invalid_date_rows_removed,
            )
        else:
            self.logger.info("日期检查通过：所有记录均满足 ann_date > end_date。")


def configure_logging(log_level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure console and file logging for CLI usage."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    target_log_file = Path(log_file) if log_file is not None else FundamentalsCleaner.DEFAULT_LOG_FILE
    target_log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(target_log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check and clean fundamentals parquet data without modifying raw files.")
    parser.add_argument(
        "--input-file",
        default=str(FundamentalsCleaner.DEFAULT_INPUT_FILE),
        help="Raw fundamentals parquet file.",
    )
    parser.add_argument(
        "--output-file",
        default=str(FundamentalsCleaner.DEFAULT_OUTPUT_FILE),
        help="Cleaned fundamentals parquet file.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    parser.add_argument("--log-file", default=str(FundamentalsCleaner.DEFAULT_LOG_FILE), help="Log file path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> FundamentalsProcessSummary:
    args = parse_args(argv)
    configure_logging(log_level=args.log_level, log_file=args.log_file)
    cleaner = FundamentalsCleaner(input_file=args.input_file, output_file=args.output_file)
    return cleaner.process()


if __name__ == "__main__":
    # PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
    # python -m utils.dataset_cleaner.fundamentals_cleaner \
    # --input-file "/opt/tiger/qyd/quant_llm/A_stocks_all_data/fundamentals/fundamentals.parquet" \
    # --output-file "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/fundamentals/fundamentals.parquet" \
    # --log-file "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/fundamentals_cleaning.log"
    main()


__all__ = ["FundamentalsProcessSummary", "FundamentalsCleaner", "configure_logging", "main", "parse_args"]
