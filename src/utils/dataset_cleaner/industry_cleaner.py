"""Industry data duplicate-row cleaner.

This module reads the aggregated ``industry.parquet`` file without modifying
the raw data, removes fully duplicated rows, and writes the cleaned parquet to
the project-level ``data/cleaned_data/industry`` directory.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd


@dataclass(frozen=True)
class IndustryProcessSummary:
    """Cleaning statistics for industry data."""

    input_file: Path
    output_file: Path
    rows_read: int
    rows_written: int
    duplicate_rows_removed: int

    @property
    def has_duplicate_issue(self) -> bool:
        """Whether fully duplicated rows were found."""

        return self.duplicate_rows_removed > 0


class IndustryCleaner:
    """Clean the aggregated industry parquet file by removing duplicate rows.

    Only full-row duplicate removal is performed. No missing-value, date, or
    schema validation is applied, per the requested processing rule.
    """

    DEFAULT_INPUT_FILE = Path("/opt/tiger/qyd/quant_llm/A_stocks_all_data/industry/industry.parquet")
    DEFAULT_OUTPUT_FILE = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/industry/industry.parquet"
    )
    DEFAULT_LOG_FILE = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/industry_cleaning.log")

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
            raise FileNotFoundError(f"Industry input file does not exist: {self.input_file}")
        if not self.input_file.is_file():
            raise ValueError(f"Industry input path is not a file: {self.input_file}")

    def process(self) -> IndustryProcessSummary:
        """Remove fully duplicated rows and write the cleaned parquet file."""

        self.logger.info("开始处理 industry 数据：input=%s, output=%s", self.input_file, self.output_file)
        self.logger.info("原始 industry parquet 只读；本流程仅删除完全重复行，不做其他检测。")

        df = pd.read_parquet(self.input_file)
        rows_read = len(df)

        duplicate_mask = df.duplicated(keep="last")
        duplicate_rows_removed = int(duplicate_mask.sum())
        cleaned = df.loc[~duplicate_mask].copy() if duplicate_rows_removed else df.copy()

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_parquet(self.output_file, index=False)

        summary = IndustryProcessSummary(
            input_file=self.input_file,
            output_file=self.output_file,
            rows_read=rows_read,
            rows_written=len(cleaned),
            duplicate_rows_removed=duplicate_rows_removed,
        )
        self._log_summary(summary)
        return summary

    def _log_summary(self, summary: IndustryProcessSummary) -> None:
        self.logger.info(
            "industry 清洗汇总：rows_read=%d, rows_written=%d, duplicate_rows_removed=%d, output=%s",
            summary.rows_read,
            summary.rows_written,
            summary.duplicate_rows_removed,
            summary.output_file,
        )
        if summary.has_duplicate_issue:
            self.logger.warning(
                "重复行检查发现问题：共删除 %d 行完全重复记录，重复记录仅保留最后一行。",
                summary.duplicate_rows_removed,
            )
        else:
            self.logger.info("重复行检查通过：未发现完全重复行。")


def configure_logging(log_level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure console and file logging for CLI usage."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    target_log_file = Path(log_file) if log_file is not None else IndustryCleaner.DEFAULT_LOG_FILE
    target_log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(target_log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean industry parquet data by removing duplicate rows only.")
    parser.add_argument("--input-file", default=str(IndustryCleaner.DEFAULT_INPUT_FILE), help="Raw industry parquet file.")
    parser.add_argument(
        "--output-file",
        default=str(IndustryCleaner.DEFAULT_OUTPUT_FILE),
        help="Cleaned industry parquet file.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    parser.add_argument("--log-file", default=str(IndustryCleaner.DEFAULT_LOG_FILE), help="Log file path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> IndustryProcessSummary:
    args = parse_args(argv)
    configure_logging(log_level=args.log_level, log_file=args.log_file)
    cleaner = IndustryCleaner(input_file=args.input_file, output_file=args.output_file)
    return cleaner.process()


if __name__ == "__main__":
    # PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \                                                                                                                                 
    # python -m utils.dataset_cleaner.industry_cleaner \                                                                                                                                               
    #   --input-file "/opt/tiger/qyd/quant_llm/A_stocks_all_data/industry/industry.parquet" \                                                                                                          
    #   --output-file "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/industry/industry.parquet" \                                                                                    
    #   --log-file "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/industry_cleaning.log" 
    main()


__all__ = ["IndustryProcessSummary", "IndustryCleaner", "configure_logging", "main", "parse_args"]
