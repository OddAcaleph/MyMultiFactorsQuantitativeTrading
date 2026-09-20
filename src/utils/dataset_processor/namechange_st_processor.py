"""Generate daily ST one-hot features from stock-name-change records.

The raw ``namechange.parquet`` file is read-only. This processor expands each
stock's name-change intervals into daily rows and writes three one-hot ST state
features for downstream wide-table joins.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


@dataclass(frozen=True)
class NamechangeStProcessSummary:
    """Quality-check and output statistics for namechange ST features."""

    input_file: Path
    output_file: Path
    rows_read: int
    rows_written: int
    stocks_processed: int
    calendar_days: int
    raw_duplicate_key_rows: int
    raw_missing_required_rows: int
    raw_open_ended_rows: int
    output_duplicate_key_rows: int
    output_missing_rows: int

    @property
    def has_uniqueness_issue(self) -> bool:
        """Whether raw or output uniqueness issues were found."""

        return self.raw_duplicate_key_rows > 0 or self.output_duplicate_key_rows > 0

    @property
    def has_missing_issue(self) -> bool:
        """Whether missing values were found in required raw columns or output."""

        return self.raw_missing_required_rows > 0 or self.output_missing_rows > 0


class NamechangeStProcessor:
    """Create daily normal/ST/*ST one-hot features from namechange data.

    Output columns:

    - ``trade_date``: daily date in ``YYYYMMDD`` integer format.
    - ``ts_code``: stock code.
    - ``no_st_stock``: 1 when the stock is not ST on that date.
    - ``st_stock``: 1 when the active stock name starts with ``ST``.
    - ``star_st_stock``: 1 when the active stock name starts with ``*ST``.

    ``end_date`` missing in raw records is treated as an open-ended interval,
    not as a data-quality error. The raw parquet file is never modified.
    """

    DEFAULT_INPUT_FILE = Path("/opt/tiger/qyd/quant_llm/A_stocks_all_data/namechange/namechange.parquet")
    DEFAULT_OUTPUT_FILE = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/namechange/namechange_st_daily.parquet"
    )
    DEFAULT_LOG_FILE = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/namechange_st_processing.log")

    RAW_COLUMNS: tuple[str, ...] = ("ts_code", "name", "start_date", "end_date", "ann_date", "change_reason")
    REQUIRED_RAW_COLUMNS: tuple[str, ...] = ("ts_code", "name", "start_date", "ann_date", "change_reason")
    RAW_UNIQUE_KEY_COLUMNS: tuple[str, ...] = ("ts_code", "start_date", "end_date")
    OUTPUT_KEY_COLUMNS: tuple[str, str] = ("trade_date", "ts_code")
    FEATURE_COLUMNS: tuple[str, str, str] = ("no_st_stock", "st_stock", "star_st_stock")

    def __init__(
        self,
        input_file: str | Path | None = None,
        output_file: str | Path | None = None,
        extend_to_date: str | pd.Timestamp | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.input_file = Path(input_file or self.DEFAULT_INPUT_FILE)
        self.output_file = Path(output_file or self.DEFAULT_OUTPUT_FILE)
        self.extend_to_date = pd.Timestamp(extend_to_date) if extend_to_date is not None else None
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        if not self.input_file.exists():
            raise FileNotFoundError(f"Namechange input file does not exist: {self.input_file}")
        if not self.input_file.is_file():
            raise ValueError(f"Namechange input path is not a file: {self.input_file}")

    def process(self) -> NamechangeStProcessSummary:
        """Run quality checks and write daily ST feature parquet."""

        self.logger.info("开始处理 namechange ST 日频特征：input=%s, output=%s", self.input_file, self.output_file)
        self.logger.info("原始 namechange parquet 只读；输出为每日 normal/ST/*ST one-hot 特征。")

        raw_df = pd.read_parquet(self.input_file).reset_index(names="_raw_row")
        self._validate_columns(raw_df.columns)

        rows_read = len(raw_df)
        raw_duplicate_key_rows = int(raw_df.duplicated(subset=list(self.RAW_UNIQUE_KEY_COLUMNS), keep=False).sum())
        raw_missing_required_rows = int(self._required_missing_mask(raw_df).sum())
        raw_open_ended_rows = int(raw_df["end_date"].isna().sum())

        prepared_df = self._prepare_raw_dataframe(raw_df)
        feature_df = self._build_daily_features(prepared_df)

        output_duplicate_key_rows = int(feature_df.duplicated(subset=list(self.OUTPUT_KEY_COLUMNS), keep=False).sum())
        output_missing_rows = int(feature_df.loc[:, (*self.OUTPUT_KEY_COLUMNS, *self.FEATURE_COLUMNS)].isna().any(axis=1).sum())

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_parquet(self.output_file, index=False)

        summary = NamechangeStProcessSummary(
            input_file=self.input_file,
            output_file=self.output_file,
            rows_read=rows_read,
            rows_written=len(feature_df),
            stocks_processed=int(feature_df["ts_code"].nunique()),
            calendar_days=int(feature_df["trade_date"].nunique()),
            raw_duplicate_key_rows=raw_duplicate_key_rows,
            raw_missing_required_rows=raw_missing_required_rows,
            raw_open_ended_rows=raw_open_ended_rows,
            output_duplicate_key_rows=output_duplicate_key_rows,
            output_missing_rows=output_missing_rows,
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
        prepared["_start_dt"] = pd.to_datetime(prepared["start_date"].astype("string"), format="%Y%m%d", errors="coerce")
        prepared["_end_dt"] = pd.to_datetime(prepared["end_date"].astype("string"), format="%Y%m%d", errors="coerce")
        prepared["_ann_dt"] = pd.to_datetime(prepared["ann_date"].astype("string"), format="%Y%m%d", errors="coerce")
        prepared["_status"] = prepared["name"].map(self._classify_st_status).astype("int8")

        invalid_start_rows = int(prepared["_start_dt"].isna().sum())
        if invalid_start_rows:
            self.logger.warning("发现 %d 行 start_date 无法解析，这些记录不会参与日频展开。", invalid_start_rows)
            prepared = prepared.loc[prepared["_start_dt"].notna()].copy()

        if prepared.empty:
            raise ValueError("No valid namechange records remain after parsing start_date.")

        max_known_date = max(prepared["_start_dt"].max(), prepared["_end_dt"].max(), prepared["_ann_dt"].max())
        if self.extend_to_date is not None and self.extend_to_date > max_known_date:
            max_known_date = self.extend_to_date
        prepared["_effective_end_dt"] = prepared["_end_dt"].fillna(max_known_date)

        invalid_interval_rows = int((prepared["_effective_end_dt"] < prepared["_start_dt"]).sum())
        if invalid_interval_rows:
            self.logger.warning("发现 %d 行 end_date 早于 start_date，这些记录不会参与日频展开。", invalid_interval_rows)
            prepared = prepared.loc[prepared["_effective_end_dt"] >= prepared["_start_dt"]].copy()

        return prepared

    def _build_daily_features(self, prepared_df: pd.DataFrame) -> pd.DataFrame:
        min_date = prepared_df["_start_dt"].min()
        max_date = prepared_df["_effective_end_dt"].max()
        all_dates = pd.date_range(min_date, max_date, freq="D")
        all_stocks = pd.Index(sorted(prepared_df["ts_code"].astype("string").unique()), name="ts_code")

        base_index = pd.MultiIndex.from_product([all_dates, all_stocks], names=["_trade_dt", "ts_code"])
        daily = base_index.to_frame(index=False)
        daily["_status"] = pd.Series(0, index=daily.index, dtype="int8")

        expanded_events = self._expand_events(prepared_df)
        if not expanded_events.empty:
            expanded_events = expanded_events.sort_values(
                ["_trade_dt", "ts_code", "_start_dt", "_ann_dt", "_raw_row"], kind="mergesort"
            )
            latest_events = expanded_events.drop_duplicates(subset=["_trade_dt", "ts_code"], keep="last")
            daily = daily.merge(
                latest_events.loc[:, ["_trade_dt", "ts_code", "_status"]],
                on=["_trade_dt", "ts_code"],
                how="left",
                suffixes=("", "_event"),
            )
            daily["_status"] = daily["_status_event"].fillna(daily["_status"]).astype("int8")
            daily = daily.drop(columns=["_status_event"])

        output = pd.DataFrame(
            {
                "trade_date": daily["_trade_dt"].dt.strftime("%Y%m%d").astype("int32"),
                "ts_code": daily["ts_code"].astype("string"),
                "no_st_stock": (daily["_status"] == 0).astype("int8"),
                "st_stock": (daily["_status"] == 1).astype("int8"),
                "star_st_stock": (daily["_status"] == 2).astype("int8"),
            }
        )
        return output

    def _expand_events(self, prepared_df: pd.DataFrame) -> pd.DataFrame:
        chunks: list[pd.DataFrame] = []
        for row in prepared_df.to_dict("records"):
            date_range = pd.date_range(row["_start_dt"], row["_effective_end_dt"], freq="D")
            if date_range.empty:
                continue
            chunks.append(
                pd.DataFrame(
                    {
                        "_trade_dt": date_range,
                        "ts_code": row["ts_code"],
                        "_status": row["_status"],
                        "_start_dt": row["_start_dt"],
                        "_ann_dt": row["_ann_dt"],
                        "_raw_row": row["_raw_row"],
                    }
                )
            )
        if not chunks:
            return pd.DataFrame(columns=["_trade_dt", "ts_code", "_status", "_start_dt", "_ann_dt", "_raw_row"])
        return pd.concat(chunks, ignore_index=True)

    @staticmethod
    def _classify_st_status(name: object) -> int:
        normalized = "" if pd.isna(name) else str(name).strip()
        normalized_upper = normalized.upper()
        if normalized_upper.startswith("*ST"):
            return 2
        if normalized_upper.startswith("ST"):
            return 1
        if normalized.endswith("退"):
            return 2
        return 0

    def _required_missing_mask(self, df: pd.DataFrame) -> pd.Series:
        missing_mask = pd.Series(False, index=df.index)
        for column in self.REQUIRED_RAW_COLUMNS:
            missing_mask = missing_mask | self._is_null_or_empty(df[column])
        return missing_mask

    @staticmethod
    def _is_null_or_empty(series: pd.Series) -> pd.Series:
        return series.isna() | series.astype("string").str.strip().eq("").fillna(False)

    def _log_summary(self, summary: NamechangeStProcessSummary) -> None:
        self.logger.info(
            "namechange ST 日频特征处理汇总：rows_read=%d, rows_written=%d, stocks=%d, days=%d, "
            "raw_duplicate_key_rows=%d, raw_missing_required_rows=%d, raw_open_ended_rows=%d, "
            "output_duplicate_key_rows=%d, output_missing_rows=%d, output=%s",
            summary.rows_read,
            summary.rows_written,
            summary.stocks_processed,
            summary.calendar_days,
            summary.raw_duplicate_key_rows,
            summary.raw_missing_required_rows,
            summary.raw_open_ended_rows,
            summary.output_duplicate_key_rows,
            summary.output_missing_rows,
            summary.output_file,
        )

        if summary.raw_duplicate_key_rows:
            self.logger.warning(
                "原始唯一性检查发现问题：%d 行存在重复 ts_code+start_date+end_date；日频输出会按最新 start/ann/raw 顺序消解为唯一 ts_code+trade_date。",
                summary.raw_duplicate_key_rows,
            )
        else:
            self.logger.info("原始唯一性检查通过：未发现重复 ts_code+start_date+end_date。")

        if summary.output_duplicate_key_rows:
            self.logger.error("输出唯一性检查失败：%d 行存在重复 trade_date+ts_code。", summary.output_duplicate_key_rows)
        else:
            self.logger.info("输出唯一性检查通过：未发现重复 trade_date+ts_code。")

        if summary.raw_missing_required_rows:
            self.logger.warning("缺失值检查发现问题：%d 行必需原始字段存在缺失。", summary.raw_missing_required_rows)
        else:
            self.logger.info("缺失值检查通过：ts_code/name/start_date/ann_date/change_reason 均无缺失。")

        if summary.raw_open_ended_rows:
            self.logger.info(
                "end_date 空值说明：发现 %d 行 end_date 为空，已按开放区间处理，不作为缺失值错误。",
                summary.raw_open_ended_rows,
            )

        if summary.output_missing_rows:
            self.logger.error("输出缺失值检查失败：%d 行输出特征存在缺失。", summary.output_missing_rows)
        else:
            self.logger.info("输出缺失值检查通过：trade_date/ts_code/no_st_stock/st_stock/star_st_stock 均无缺失。")


def configure_logging(log_level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure console and file logging for CLI usage."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    target_log_file = Path(log_file) if log_file is not None else NamechangeStProcessor.DEFAULT_LOG_FILE
    target_log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(target_log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily ST one-hot features from namechange parquet data.")
    parser.add_argument(
        "--input-file",
        default=str(NamechangeStProcessor.DEFAULT_INPUT_FILE),
        help="Raw namechange parquet file.",
    )
    parser.add_argument(
        "--output-file",
        default=str(NamechangeStProcessor.DEFAULT_OUTPUT_FILE),
        help="Processed daily ST feature parquet file.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    parser.add_argument("--log-file", default=str(NamechangeStProcessor.DEFAULT_LOG_FILE), help="Log file path.")
    parser.add_argument(
        "--extend-to-date",
        default=None,
        help="Extend open-ended (end_date is null) ST records to this date (YYYYMMDD). "
             "Defaults to the max date found in namechange data itself.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> NamechangeStProcessSummary:
    args = parse_args(argv)
    configure_logging(log_level=args.log_level, log_file=args.log_file)
    processor = NamechangeStProcessor(
        input_file=args.input_file,
        output_file=args.output_file,
        extend_to_date=args.extend_to_date,
    )
    return processor.process()


if __name__ == "__main__":
    # PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \                                                                                                                                 
    # python -m utils.dataset_processor.namechange_st_processor \                                                                                                                                      
    #   --input-file "/opt/tiger/qyd/quant_llm/A_stocks_all_data/namechange/namechange.parquet" \                                                                                                      
    #   --output-file "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/namechange/namechange_st_daily.parquet" \                                                                      
    #   --log-file "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/namechange_st_processing.log"   
    main()


__all__ = ["NamechangeStProcessSummary", "NamechangeStProcessor", "configure_logging", "main", "parse_args"]
