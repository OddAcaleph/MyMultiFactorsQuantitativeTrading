"""Generate daily suspend/resume features from Tushare suspend_d records.

The raw ``suspend_d.parquet`` file is treated as read-only. This processor
normalizes all ``S``/``R`` records into a unique daily ``trade_date`` +
``ts_code`` feature table for downstream wide-table joins.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class SuspendDProcessSummary:
    """Quality-check and output statistics for daily suspend features."""

    input_file: Path
    output_file: Path
    rows_read: int
    rows_written: int
    stocks_processed: int
    trading_days: int
    raw_duplicate_full_rows: int
    raw_duplicate_key_rows: int
    raw_conflict_key_rows: int
    raw_missing_required_rows: int
    raw_invalid_trade_date_rows: int
    raw_invalid_suspend_type_rows: int
    output_duplicate_key_rows: int
    output_missing_rows: int
    output_suspect_rows: int
    output_non_suspect_rows: int

    @property
    def has_uniqueness_issue(self) -> bool:
        """Whether raw or output uniqueness issues were found."""

        return self.raw_duplicate_key_rows > 0 or self.output_duplicate_key_rows > 0

    @property
    def has_missing_issue(self) -> bool:
        """Whether missing values were found in required raw columns or output."""

        return self.raw_missing_required_rows > 0 or self.output_missing_rows > 0


class SuspendDProcessor:
    """Create unique daily ``is_suspect`` features from suspend_d data.

    Output columns:

    - ``trade_date``: daily date in ``YYYYMMDD`` integer format.
    - ``ts_code``: stock code.
    - ``is_suspect``: ``1`` when the stock has an ``S`` suspend record on that
      date, otherwise ``0`` for an ``R`` resume-only record.

    If both ``S`` and ``R`` exist for the same ``trade_date`` + ``ts_code``, the
    output remains unique and ``S`` takes precedence, because the stock had at
    least one suspension event that day. The raw parquet file is never modified.
    """

    DEFAULT_INPUT_FILE = Path("/opt/tiger/qyd/quant_llm/A_stocks_all_data/suspend_d/suspend_d.parquet")
    DEFAULT_OUTPUT_FILE = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/suspend_d/suspend_d_daily.parquet"
    )
    DEFAULT_LOG_FILE = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/suspend_d_processing.log")

    RAW_COLUMNS: tuple[str, ...] = ("ts_code", "trade_date", "suspend_timing", "suspend_type")
    REQUIRED_RAW_COLUMNS: tuple[str, ...] = ("ts_code", "trade_date", "suspend_type")
    RAW_FULL_UNIQUE_COLUMNS: tuple[str, ...] = ("ts_code", "trade_date", "suspend_timing", "suspend_type")
    OUTPUT_KEY_COLUMNS: tuple[str, str] = ("trade_date", "ts_code")
    FEATURE_COLUMNS: tuple[str, ...] = ("is_suspect",)
    VALID_SUSPEND_TYPES: tuple[str, str] = ("S", "R")

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
            raise FileNotFoundError(f"Suspend input file does not exist: {self.input_file}")
        if not self.input_file.is_file():
            raise ValueError(f"Suspend input path is not a file: {self.input_file}")

    def process(self) -> SuspendDProcessSummary:
        """Run quality checks and write daily suspend feature parquet."""

        self.logger.info("开始处理 suspend_d 日频特征：input=%s, output=%s", self.input_file, self.output_file)
        self.logger.info("原始 suspend_d parquet 只读；输出为唯一 trade_date+ts_code 的 is_suspect 特征。")

        raw_df = pd.read_parquet(self.input_file).reset_index(names="_raw_row")
        self._validate_columns(raw_df.columns)

        rows_read = len(raw_df)
        raw_duplicate_full_rows = int(raw_df.duplicated(subset=list(self.RAW_FULL_UNIQUE_COLUMNS), keep=False).sum())
        raw_duplicate_key_rows = int(raw_df.duplicated(subset=list(self.OUTPUT_KEY_COLUMNS), keep=False).sum())
        raw_conflict_key_rows = self._count_conflict_key_rows(raw_df)
        raw_missing_required_rows = int(self._required_missing_mask(raw_df).sum())
        raw_invalid_trade_date_rows = int(
            pd.to_datetime(raw_df["trade_date"].astype("string"), format="%Y%m%d", errors="coerce").isna().sum()
        )
        raw_suspend_type = raw_df["suspend_type"].astype("string").str.strip().str.upper()
        raw_invalid_suspend_type_rows = int((~raw_suspend_type.isin(self.VALID_SUSPEND_TYPES)).sum())

        prepared_df = self._prepare_raw_dataframe(raw_df)

        feature_df = self._build_daily_features(prepared_df)

        output_duplicate_key_rows = int(feature_df.duplicated(subset=list(self.OUTPUT_KEY_COLUMNS), keep=False).sum())
        output_missing_rows = int(feature_df.loc[:, (*self.OUTPUT_KEY_COLUMNS, *self.FEATURE_COLUMNS)].isna().any(axis=1).sum())

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_parquet(self.output_file, index=False)

        summary = SuspendDProcessSummary(
            input_file=self.input_file,
            output_file=self.output_file,
            rows_read=rows_read,
            rows_written=len(feature_df),
            stocks_processed=int(feature_df["ts_code"].nunique()) if not feature_df.empty else 0,
            trading_days=int(feature_df["trade_date"].nunique()) if not feature_df.empty else 0,
            raw_duplicate_full_rows=raw_duplicate_full_rows,
            raw_duplicate_key_rows=raw_duplicate_key_rows,
            raw_conflict_key_rows=raw_conflict_key_rows,
            raw_missing_required_rows=raw_missing_required_rows,
            raw_invalid_trade_date_rows=raw_invalid_trade_date_rows,
            raw_invalid_suspend_type_rows=raw_invalid_suspend_type_rows,
            output_duplicate_key_rows=output_duplicate_key_rows,
            output_missing_rows=output_missing_rows,
            output_suspect_rows=int(feature_df["is_suspect"].sum()) if not feature_df.empty else 0,
            output_non_suspect_rows=int((feature_df["is_suspect"] == 0).sum()) if not feature_df.empty else 0,
        )
        self._log_summary(summary)
        return summary

    def _validate_columns(self, columns: Iterable[str]) -> None:
        column_set = set(columns)
        missing_columns = [col for col in self.RAW_COLUMNS if col not in column_set]
        if missing_columns:
            raise ValueError(f"{self.input_file} is missing required columns: {missing_columns}")

    def _prepare_raw_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        prepared = raw_df.copy()
        prepared["ts_code"] = prepared["ts_code"].astype("string").str.strip()
        prepared["_trade_dt"] = pd.to_datetime(prepared["trade_date"].astype("string"), format="%Y%m%d", errors="coerce")
        prepared["_suspend_type"] = prepared["suspend_type"].astype("string").str.strip().str.upper()
        prepared["is_suspect"] = (prepared["_suspend_type"] == "S").astype("int8")

        invalid_trade_date_rows = int(prepared["_trade_dt"].isna().sum())
        if invalid_trade_date_rows:
            self.logger.warning("发现 %d 行 trade_date 无法解析，这些记录不会参与日频输出。", invalid_trade_date_rows)
            prepared = prepared.loc[prepared["_trade_dt"].notna()].copy()

        invalid_suspend_type_rows = int((~prepared["_suspend_type"].isin(self.VALID_SUSPEND_TYPES)).sum())
        if invalid_suspend_type_rows:
            self.logger.warning("发现 %d 行 suspend_type 不是 S/R，这些记录不会参与日频输出。", invalid_suspend_type_rows)
            prepared = prepared.loc[prepared["_suspend_type"].isin(self.VALID_SUSPEND_TYPES)].copy()

        missing_required_rows = int(self._required_missing_mask(prepared).sum())
        if missing_required_rows:
            self.logger.warning("发现 %d 行必需字段缺失，这些记录不会参与日频输出。", missing_required_rows)
            prepared = prepared.loc[~self._required_missing_mask(prepared)].copy()

        if prepared.empty:
            raise ValueError("No valid suspend_d records remain after parsing required fields.")

        return prepared

    def _build_daily_features(self, prepared_df: pd.DataFrame) -> pd.DataFrame:
        ordered = prepared_df.sort_values(
            ["_trade_dt", "ts_code", "is_suspect", "_raw_row"],
            ascending=[True, True, True, True],
            kind="mergesort",
        )
        daily = (
            ordered.groupby(["_trade_dt", "ts_code"], as_index=False, sort=True)["is_suspect"]
            .max()
            .sort_values(["_trade_dt", "ts_code"], kind="mergesort")
        )
        output = pd.DataFrame(
            {
                "trade_date": daily["_trade_dt"].dt.strftime("%Y%m%d").astype("int32"),
                "ts_code": daily["ts_code"].astype("string"),
                "is_suspect": daily["is_suspect"].astype("int8"),
            }
        )
        return output

    def _count_conflict_key_rows(self, raw_df: pd.DataFrame) -> int:
        conflict_type_counts = raw_df.groupby(list(self.OUTPUT_KEY_COLUMNS))["suspend_type"].nunique(dropna=True)
        conflict_keys = conflict_type_counts[conflict_type_counts > 1].index
        if len(conflict_keys) == 0:
            return 0
        conflict_index = pd.MultiIndex.from_frame(raw_df.loc[:, list(self.OUTPUT_KEY_COLUMNS)])
        return int(conflict_index.isin(conflict_keys).sum())

    def _required_missing_mask(self, df: pd.DataFrame) -> pd.Series:
        missing_mask = pd.Series(False, index=df.index)
        for column in self.REQUIRED_RAW_COLUMNS:
            missing_mask = missing_mask | self._is_null_or_empty(df[column])
        return missing_mask

    @staticmethod
    def _is_null_or_empty(series: pd.Series) -> pd.Series:
        return series.isna() | series.astype("string").str.strip().eq("").fillna(False)

    def _log_summary(self, summary: SuspendDProcessSummary) -> None:
        self.logger.info(
            "suspend_d 日频特征处理汇总：rows_read=%d, rows_written=%d, stocks=%d, days=%d, "
            "raw_duplicate_full_rows=%d, raw_duplicate_key_rows=%d, raw_conflict_key_rows=%d, "
            "raw_missing_required_rows=%d, raw_invalid_trade_date_rows=%d, raw_invalid_suspend_type_rows=%d, "
            "output_duplicate_key_rows=%d, output_missing_rows=%d, suspect_rows=%d, non_suspect_rows=%d, output=%s",
            summary.rows_read,
            summary.rows_written,
            summary.stocks_processed,
            summary.trading_days,
            summary.raw_duplicate_full_rows,
            summary.raw_duplicate_key_rows,
            summary.raw_conflict_key_rows,
            summary.raw_missing_required_rows,
            summary.raw_invalid_trade_date_rows,
            summary.raw_invalid_suspend_type_rows,
            summary.output_duplicate_key_rows,
            summary.output_missing_rows,
            summary.output_suspect_rows,
            summary.output_non_suspect_rows,
            summary.output_file,
        )

        if summary.raw_duplicate_full_rows:
            self.logger.warning("原始完整行唯一性检查发现问题：%d 行存在重复 S/R 记录。", summary.raw_duplicate_full_rows)
        else:
            self.logger.info("原始完整行唯一性检查通过：未发现重复 ts_code+trade_date+suspend_timing+suspend_type。")

        if summary.raw_duplicate_key_rows:
            self.logger.warning(
                "原始日频键唯一性检查发现问题：%d 行存在重复 trade_date+ts_code；输出已按同日 S 优先聚合为唯一结果。",
                summary.raw_duplicate_key_rows,
            )
        else:
            self.logger.info("原始日频键唯一性检查通过：未发现重复 trade_date+ts_code。")

        if summary.raw_conflict_key_rows:
            self.logger.warning(
                "原始 S/R 冲突检查发现问题：%d 行在同一 trade_date+ts_code 同时存在 S 和 R；输出已置 is_suspect=1。",
                summary.raw_conflict_key_rows,
            )
        else:
            self.logger.info("原始 S/R 冲突检查通过：同一 trade_date+ts_code 未同时出现 S 和 R。")

        if summary.output_duplicate_key_rows:
            self.logger.error("输出唯一性检查失败：%d 行存在重复 trade_date+ts_code。", summary.output_duplicate_key_rows)
        else:
            self.logger.info("输出唯一性检查通过：未发现重复 trade_date+ts_code。")

        if summary.raw_missing_required_rows:
            self.logger.warning("缺失值检查发现问题：%d 行 ts_code/trade_date/suspend_type 存在缺失。", summary.raw_missing_required_rows)
        else:
            self.logger.info("缺失值检查通过：ts_code/trade_date/suspend_type 均无缺失。")

        if summary.raw_invalid_trade_date_rows:
            self.logger.warning("日期格式检查发现问题：%d 行 trade_date 无法按 YYYYMMDD 解析。", summary.raw_invalid_trade_date_rows)
        else:
            self.logger.info("日期格式检查通过：trade_date 均可按 YYYYMMDD 解析。")

        if summary.raw_invalid_suspend_type_rows:
            self.logger.warning("类型检查发现问题：%d 行 suspend_type 不是 S/R。", summary.raw_invalid_suspend_type_rows)
        else:
            self.logger.info("类型检查通过：suspend_type 均为 S/R。")

        if summary.output_missing_rows:
            self.logger.error("输出缺失值检查失败：%d 行输出特征存在缺失。", summary.output_missing_rows)
        else:
            self.logger.info("输出缺失值检查通过：trade_date/ts_code/is_suspect 均无缺失。")


def configure_logging(log_level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure console and file logging for CLI usage."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    target_log_file = Path(log_file) if log_file is not None else SuspendDProcessor.DEFAULT_LOG_FILE
    target_log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(target_log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily is_suspect features from suspend_d parquet data.")
    parser.add_argument(
        "--input-file",
        default=str(SuspendDProcessor.DEFAULT_INPUT_FILE),
        help="Raw suspend_d parquet file.",
    )
    parser.add_argument(
        "--output-file",
        default=str(SuspendDProcessor.DEFAULT_OUTPUT_FILE),
        help="Processed daily suspend feature parquet file.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    parser.add_argument("--log-file", default=str(SuspendDProcessor.DEFAULT_LOG_FILE), help="Log file path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> SuspendDProcessSummary:
    args = parse_args(argv)
    configure_logging(log_level=args.log_level, log_file=args.log_file)
    processor = SuspendDProcessor(input_file=args.input_file, output_file=args.output_file)
    return processor.process()


if __name__ == "__main__":
    # PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
    # python -m utils.dataset_processor.suspend_d_processor \
    #   --input-file "/opt/tiger/qyd/quant_llm/A_stocks_all_data/suspend_d/suspend_d.parquet" \
    #   --output-file "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/suspend_d/suspend_d_daily.parquet" \
    #   --log-file "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/suspend_d_processing.log"
    main()


__all__ = ["SuspendDProcessSummary", "SuspendDProcessor", "configure_logging", "main", "parse_args"]
