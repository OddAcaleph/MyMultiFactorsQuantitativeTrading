"""Moneyflow data quality checker and non-destructive cleaner.

This module checks the aggregated ``moneyflow.parquet`` file without modifying
the raw data. The cleaned parquet is written to the project-level
``data/cleaned_data/moneyflow`` directory.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class MoneyflowProcessSummary:
    """Quality-check and cleaning statistics for moneyflow data."""

    input_file: Path
    output_file: Path
    rows_read: int
    rows_written: int
    missing_rows_removed: int
    duplicate_rows_removed: int

    @property
    def has_missing_issue(self) -> bool:
        """Whether any rows with empty ``ts_code`` or ``trade_date`` were found."""

        return self.missing_rows_removed > 0

    @property
    def has_duplicate_issue(self) -> bool:
        """Whether duplicate ``ts_code`` + ``trade_date`` primary keys were found."""

        return self.duplicate_rows_removed > 0


class MoneyflowCleaner:
    """Check and clean the aggregated moneyflow parquet file.

    Cleaning rules are intentionally narrow and non-destructive:

    1. Check whether ``ts_code`` or ``trade_date`` is null/empty. Rows with
       missing primary-key fields are removed from the cleaned output.
    2. Treat ``ts_code`` + ``trade_date`` as the primary key. If duplicated,
       keep the last row in the original parquet row order as the latest row.

    The original parquet file is only read; it is never overwritten.
    """

    DEFAULT_INPUT_FILE = Path("/opt/tiger/qyd/quant_llm/A_stocks_all_data/moneyflow/moneyflow.parquet")
    DEFAULT_OUTPUT_FILE = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/moneyflow/moneyflow.parquet"
    )
    DEFAULT_LOG_FILE = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/moneyflow_cleaning.log")

    KEY_COLUMNS: tuple[str, str] = ("ts_code", "trade_date")
    REQUIRED_COLUMNS: tuple[str, ...] = (
        "ts_code",
        "trade_date",
        "buy_sm_vol",
        "buy_sm_amount",
        "sell_sm_vol",
        "sell_sm_amount",
        "buy_md_vol",
        "buy_md_amount",
        "sell_md_vol",
        "sell_md_amount",
        "buy_lg_vol",
        "buy_lg_amount",
        "sell_lg_vol",
        "sell_lg_amount",
        "buy_elg_vol",
        "buy_elg_amount",
        "sell_elg_vol",
        "sell_elg_amount",
        "net_mf_vol",
        "net_mf_amount",
    )

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
            raise FileNotFoundError(f"Moneyflow input file does not exist: {self.input_file}")
        if not self.input_file.is_file():
            raise ValueError(f"Moneyflow input path is not a file: {self.input_file}")

    def process(self) -> MoneyflowProcessSummary:
        """Run quality checks and write the cleaned parquet file."""

        self.logger.info("开始处理 moneyflow 数据：input=%s, output=%s", self.input_file, self.output_file)
        self.logger.info("原始 moneyflow parquet 只读，清洗结果将单独落盘。")

        df = pd.read_parquet(self.input_file)
        self._validate_columns(df.columns)

        rows_read = len(df)
        cleaned, missing_rows_removed, duplicate_rows_removed = self._clean_dataframe(df)

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        cleaned.to_parquet(self.output_file, index=False)

        summary = MoneyflowProcessSummary(
            input_file=self.input_file,
            output_file=self.output_file,
            rows_read=rows_read,
            rows_written=len(cleaned),
            missing_rows_removed=missing_rows_removed,
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

        ts_code_empty = self._is_null_or_empty(cleaned["ts_code"])
        trade_date_empty = self._is_null_or_empty(cleaned["trade_date"])
        missing_mask = ts_code_empty | trade_date_empty
        missing_count = int(missing_mask.sum())
        if missing_count:
            self.logger.warning(
                "缺失值明细：ts_code 缺失=%d, trade_date 缺失=%d, 任一主键字段缺失=%d。",
                int(ts_code_empty.sum()),
                int(trade_date_empty.sum()),
                missing_count,
            )
            cleaned = cleaned.loc[~missing_mask].copy()

        duplicate_mask = cleaned.duplicated(subset=list(self.KEY_COLUMNS), keep="last")
        duplicate_count = int(duplicate_mask.sum())
        if duplicate_count:
            sample_duplicate_keys = (
                cleaned.loc[cleaned.duplicated(subset=list(self.KEY_COLUMNS), keep=False), list(self.KEY_COLUMNS)]
                .drop_duplicates()
                .head(10)
                .to_dict(orient="records")
            )
            self.logger.warning("唯一性问题示例（最多 10 个重复主键）：%s", sample_duplicate_keys)
            cleaned = cleaned.loc[~duplicate_mask].copy()

        return cleaned, missing_count, duplicate_count

    @staticmethod
    def _is_null_or_empty(series: pd.Series) -> pd.Series:
        return series.isna() | series.astype("string").str.strip().eq("").fillna(False)

    def _log_summary(self, summary: MoneyflowProcessSummary) -> None:
        self.logger.info(
            "moneyflow 清洗汇总：rows_read=%d, rows_written=%d, "
            "missing_rows_removed=%d, duplicate_rows_removed=%d, output=%s",
            summary.rows_read,
            summary.rows_written,
            summary.missing_rows_removed,
            summary.duplicate_rows_removed,
            summary.output_file,
        )
        if summary.has_duplicate_issue:
            self.logger.warning(
                "唯一性检查发现问题：共删除 %d 行重复 ts_code+trade_date，重复主键仅保留最后一行。",
                summary.duplicate_rows_removed,
            )
        else:
            self.logger.info("唯一性检查通过：未发现重复 ts_code+trade_date。")

        if summary.has_missing_issue:
            self.logger.warning(
                "缺失值检查发现问题：共删除 %d 行 ts_code/trade_date 为空的记录。",
                summary.missing_rows_removed,
            )
        else:
            self.logger.info("缺失值检查通过：ts_code/trade_date 均无空值。")


def configure_logging(log_level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure console and file logging for CLI usage."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    target_log_file = Path(log_file) if log_file is not None else MoneyflowCleaner.DEFAULT_LOG_FILE
    target_log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(target_log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check and clean moneyflow parquet data without modifying raw files.")
    parser.add_argument("--input-file", default=str(MoneyflowCleaner.DEFAULT_INPUT_FILE), help="Raw moneyflow parquet file.")
    parser.add_argument(
        "--output-file",
        default=str(MoneyflowCleaner.DEFAULT_OUTPUT_FILE),
        help="Cleaned moneyflow parquet file.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    parser.add_argument("--log-file", default=str(MoneyflowCleaner.DEFAULT_LOG_FILE), help="Log file path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> MoneyflowProcessSummary:
    args = parse_args(argv)
    configure_logging(log_level=args.log_level, log_file=args.log_file)
    cleaner = MoneyflowCleaner(input_file=args.input_file, output_file=args.output_file)
    return cleaner.process()


if __name__ == "__main__":
    # PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
    # python3 -m utils.dataset_cleaner.moneyflow_cleaner \
    # --input-file "/opt/tiger/qyd/quant_llm/A_stocks_all_data/moneyflow/moneyflow.parquet" \
    # --output-file "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/moneyflow/moneyflow.parquet" \
    # --log-file "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/moneyflow_cleaning.log"
    main()


__all__ = ["MoneyflowProcessSummary", "MoneyflowCleaner", "configure_logging", "main", "parse_args"]
