"""DailyBars 增量清洗工具。

只清洗指定日期范围内的文件，避免全量重跑。
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class IncrementalCleanSummary:
    files_processed: int
    rows_read: int
    rows_written: int
    missing_rows_removed: int
    duplicate_rows_removed: int
    start_date: str | None
    end_date: str | None


class DailyBarsIncrementalCleaner:
    """按日期范围增量清洗 daily_bars 数据。

    只处理 start_date ~ end_date 范围内的文件，其他日期的文件保持不变。
    """

    KEY_COLUMNS: tuple[str, str] = ("ts_code", "trade_date")
    VALUE_COLUMNS: tuple[str, ...] = (
        "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount",
    )
    REQUIRED_COLUMNS: tuple[str, ...] = (*KEY_COLUMNS, *VALUE_COLUMNS)
    MISSING_CHECK_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        start_date: str | None = None,
        end_date: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.start_date = start_date
        self.end_date = end_date
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")

    def process(self) -> IncrementalCleanSummary:
        """运行增量清洗。"""
        parquet_files = self._list_target_files()
        if not parquet_files:
            self.logger.info("没有需要处理的文件，跳过。")
            return IncrementalCleanSummary(
                files_processed=0, rows_read=0, rows_written=0,
                missing_rows_removed=0, duplicate_rows_removed=0,
                start_date=self.start_date, end_date=self.end_date,
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            "增量清洗 DailyBars：%s ~ %s，共 %d 个文件",
            self.start_date or "最早", self.end_date or "最晚", len(parquet_files),
        )

        rows_read = 0
        rows_written = 0
        missing_rows_removed = 0
        duplicate_rows_removed = 0

        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file)
            self._validate_columns(df.columns, parquet_file)

            original_count = len(df)
            rows_read += original_count

            cleaned, file_missing_count, file_duplicate_count = self._clean_dataframe(df)

            missing_rows_removed += file_missing_count
            duplicate_rows_removed += file_duplicate_count
            rows_written += len(cleaned)

            output_file = self._output_path_for(parquet_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            cleaned.to_parquet(output_file, index=False)

            self.logger.info(
                "文件处理完成：%s | 原始=%d, 缺失删除=%d, 重复删除=%d, 输出=%d",
                parquet_file.name, original_count, file_missing_count, file_duplicate_count, len(cleaned),
            )

        summary = IncrementalCleanSummary(
            files_processed=len(parquet_files),
            rows_read=rows_read,
            rows_written=rows_written,
            missing_rows_removed=missing_rows_removed,
            duplicate_rows_removed=duplicate_rows_removed,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        self._log_summary(summary)
        return summary

    def _list_target_files(self) -> list[Path]:
        all_files = sorted(self.input_dir.glob("year=*/month=*/*.parquet"))
        target_files = []

        for f in all_files:
            date_str = f.stem
            if len(date_str) != 8 or not date_str.isdigit():
                continue
            if self.start_date and date_str < self.start_date:
                continue
            if self.end_date and date_str > self.end_date:
                continue
            target_files.append(f)

        return target_files

    def _validate_columns(self, columns, parquet_file: Path) -> None:
        column_set = set(columns)
        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in column_set]
        if missing_columns:
            raise ValueError(f"{parquet_file} is missing required columns: {missing_columns}")

    def _clean_dataframe(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
        cleaned = df.copy()

        missing_mask = cleaned.loc[:, self.MISSING_CHECK_COLUMNS].isna().any(axis=1)
        missing_count = int(missing_mask.sum())
        if missing_count:
            cleaned = cleaned.loc[~missing_mask].copy()

        duplicate_mask = cleaned.duplicated(subset=list(self.KEY_COLUMNS), keep="first")
        duplicate_count = int(duplicate_mask.sum())
        if duplicate_count:
            cleaned = cleaned.loc[~duplicate_mask].copy()

        return cleaned, missing_count, duplicate_count

    def _output_path_for(self, input_file: Path) -> Path:
        return self.output_dir / input_file.relative_to(self.input_dir)

    def _log_summary(self, summary: IncrementalCleanSummary) -> None:
        self.logger.info("=" * 60)
        self.logger.info("增量清洗完成：")
        self.logger.info("  日期范围：%s ~ %s", summary.start_date or "最早", summary.end_date or "最晚")
        self.logger.info("  处理文件数：%d", summary.files_processed)
        self.logger.info("  读取行数：%d", summary.rows_read)
        self.logger.info("  写入行数：%d", summary.rows_written)
        self.logger.info("  缺失删除：%d", summary.missing_rows_removed)
        self.logger.info("  重复删除：%d", summary.duplicate_rows_removed)
        self.logger.info("=" * 60)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="增量清洗 DailyBars parquet 数据（按日期范围）")
    parser.add_argument("--input-dir", required=True, help="原始 daily_bars 目录")
    parser.add_argument("--output-dir", required=True, help="清洗后输出目录")
    parser.add_argument("--start-date", default=None, help="起始日期 YYYYMMDD（含）")
    parser.add_argument("--end-date", default=None, help="结束日期 YYYYMMDD（含）")
    parser.add_argument("--log-level", default="INFO", help="日志级别")
    parser.add_argument("--log-file", default=None, help="日志文件路径")
    return parser.parse_args(argv)


def configure_logging(log_level: str = "INFO", log_file: str | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def main(argv: Sequence[str] | None = None) -> IncrementalCleanSummary:
    args = parse_args(argv)
    configure_logging(log_level=args.log_level, log_file=args.log_file)
    cleaner = DailyBarsIncrementalCleaner(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    return cleaner.process()


if __name__ == "__main__":
    main()
