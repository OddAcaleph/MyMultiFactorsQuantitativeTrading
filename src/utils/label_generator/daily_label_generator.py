"""Generate supervised training labels from daily wide-table bars.

The generator treats ``wide_table_daily_bars`` parquet files as read-only and
writes a separate partitioned label dataset.  Output files preserve the source
``year=YYYY/month=MM/YYYYMMDD.parquet`` layout so downstream training jobs can
load labels by trade date without touching raw or processed input data.

Leakage control:

* Forward returns are computed independently inside each ``ts_code`` time series
  after sorting by ``trade_date``.  Before calculating labels, the generator
  first builds one continuous adjusted-close history per stock with the latest
  available adjustment factor in the full dataset:
  ``adj_close(t) = close(t) * adj_factor(t) / latest_factor``.  Labels are then
  calculated from this unified adjusted-price series.
* Rank labels are computed cross-sectionally within the same ``trade_date`` only
  (``groupby("trade_date").rank(pct=True)``), so no information from future
  dates is used when creating the 0~1 rank target for a date.
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
class DailyLabelGenerateSummary:
    """Aggregate statistics for daily label generation."""

    input_dir: Path
    output_dir: Path
    files_read: int
    files_written: int
    rows_read: int
    rows_written: int
    start_date: int | None
    end_date: int | None
    label_columns: tuple[str, ...]
    rank_columns: tuple[str, ...]
    missing_source_rows: int
    zero_adj_close_rows: int
    duplicate_key_rows: int


class DailyLabelGenerator:
    """Generate forward-return labels and same-date cross-sectional rank labels.

    Labels are based on a unified full-history adjusted price series.  For each
    stock, the latest valid adjustment factor in the full input dataset is used
    as the common denominator:

    ``latest_factor = adj_factor.iloc[-1]``

    ``adj_close(t) = close(t) * adj_factor(t) / latest_factor``

    Labels are then calculated from the continuous adjusted-close series:

    ``label_hd = adj_close(t+h) / adj_close(t) - 1``

    For each horizon ``h`` in ``(1, 2, 3, 5, 10, 20)``, the generator creates:

    * ``label_{h}d = adj_close(t+h) / adj_close(t) - 1`` per stock.
    * ``label_rank_{h}d`` as same-date cross-sectional percentile rank of
      ``label_{h}d``.

    Source parquet files are never modified.  The output is written under
    ``data/generated_label/daily_labels`` by default, preserving daily date
    partitions.
    """

    DEFAULT_INPUT_DIR = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/wide_table_daily_bars")
    DEFAULT_OUTPUT_DIR = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/generated_label/daily_labels")
    DEFAULT_LOG_FILE = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/label_generate/daily_labels.log")

    KEY_COLUMNS: tuple[str, str] = ("trade_date", "ts_code")
    SOURCE_COLUMNS: tuple[str, str] = ("close", "adj_factor")
    REQUIRED_COLUMNS: tuple[str, ...] = (*KEY_COLUMNS, *SOURCE_COLUMNS)
    HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
    LABEL_COLUMNS: tuple[str, ...] = tuple(f"label_{horizon}d" for horizon in HORIZONS)
    RANK_COLUMNS: tuple[str, ...] = tuple(f"label_rank_{horizon}d" for horizon in HORIZONS)
    MAX_FORWARD_DAYS = max(HORIZONS)

    def __init__(
        self,
        input_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        start_date: str | int | None = None,
        end_date: str | int | None = None,
        output_batch_size: int = 120,
        logger: logging.Logger | None = None,
    ) -> None:
        self.input_dir = Path(input_dir or self.DEFAULT_INPUT_DIR)
        self.output_dir = Path(output_dir or self.DEFAULT_OUTPUT_DIR)
        self.start_date = self._parse_optional_date(start_date, "start_date")
        self.end_date = self._parse_optional_date(end_date, "end_date")
        self.output_batch_size = output_batch_size
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self._validate_inputs()

    def process(self) -> DailyLabelGenerateSummary:
        """Generate label parquet files and return aggregate processing stats."""

        all_files = self._list_all_wide_table_files()
        output_files = [file_path for file_path in all_files if self._is_file_date_in_output_range(file_path)]
        if not output_files:
            raise FileNotFoundError(
                f"No parquet files found under {self.input_dir}/year=*/month=*/*.parquet "
                f"for output date range [{self.start_date}, {self.end_date}]"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("开始生成训练 label：input=%s, output=%s", self.input_dir, self.output_dir)
        self.logger.info(
            "待输出交易日文件数=%d；分批输出 batch_size=%d；每批额外读取最多 %d 个未来交易日文件；所有输入数据只读。",
            len(output_files),
            self.output_batch_size,
            self.MAX_FORWARD_DAYS,
        )
        latest_adj_factor_by_stock = self._load_latest_adj_factor_by_stock(all_files)
        self.logger.info("全历史最新复权因子加载完成：stocks=%d", len(latest_adj_factor_by_stock))

        rows_read = 0
        rows_written = 0
        files_read = 0
        files_written = 0
        missing_source_rows = 0
        zero_adj_close_rows = 0
        duplicate_key_rows = 0
        output_nan_counts = dict.fromkeys((*self.LABEL_COLUMNS, *self.RANK_COLUMNS), 0)

        for batch_no, batch_output_files in enumerate(self._iter_file_batches(output_files), start=1):
            read_files = self._select_files_with_future(all_files, batch_output_files)
            output_dates = {self._date_from_file_name(file_path) for file_path in batch_output_files}
            self.logger.info(
                "开始处理第 %d 批：output_files=%d, read_files=%d, output_date_range=[%s, %s]",
                batch_no,
                len(batch_output_files),
                len(read_files),
                self._date_from_file_name(batch_output_files[0]),
                self._date_from_file_name(batch_output_files[-1]),
            )

            bars_df = self._load_wide_table_bars(read_files)
            rows_read += len(bars_df)
            files_read += len(read_files)
            batch_duplicate_key_rows = int(bars_df.duplicated(subset=list(self.KEY_COLUMNS), keep=False).sum())
            duplicate_key_rows += batch_duplicate_key_rows
            if batch_duplicate_key_rows:
                self.logger.warning("第 %d 批输入存在 %d 行重复 trade_date+ts_code，将全部保留并按原顺序计算。", batch_no, batch_duplicate_key_rows)
            missing_source_rows += int(bars_df.loc[:, list(self.SOURCE_COLUMNS)].isna().any(axis=1).sum())

            label_df = self._generate_labels(bars_df, latest_adj_factor_by_stock)
            zero_adj_close_rows += int((label_df["adj_close"] == 0).sum())
            output_df = label_df.loc[label_df["trade_date"].isin(output_dates), self._output_columns()].copy()
            output_df = output_df.sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)

            rows_written += len(output_df)
            for column in output_nan_counts:
                output_nan_counts[column] += int(output_df[column].isna().sum())
            files_written += self._write_partitioned_by_date(output_df)

        summary = DailyLabelGenerateSummary(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            files_read=files_read,
            files_written=files_written,
            rows_read=rows_read,
            rows_written=rows_written,
            start_date=self.start_date,
            end_date=self.end_date,
            label_columns=self.LABEL_COLUMNS,
            rank_columns=self.RANK_COLUMNS,
            missing_source_rows=missing_source_rows,
            zero_adj_close_rows=zero_adj_close_rows,
            duplicate_key_rows=duplicate_key_rows,
        )
        self._log_summary(summary, output_nan_counts)
        return summary

    def _validate_inputs(self) -> None:
        if not self.input_dir.exists() or not self.input_dir.is_dir():
            raise NotADirectoryError(f"Wide-table input directory does not exist or is not a directory: {self.input_dir}")
        if self.start_date is not None and self.end_date is not None and self.start_date > self.end_date:
            raise ValueError(f"start_date must be <= end_date, got {self.start_date} > {self.end_date}")
        if self.output_batch_size <= 0:
            raise ValueError(f"output_batch_size must be positive, got {self.output_batch_size}")

    def _list_all_wide_table_files(self) -> list[Path]:
        files = sorted(self.input_dir.glob("year=*/month=*/*.parquet"))
        valid_files: list[Path] = []
        for file_path in files:
            if self._date_from_file_name(file_path) is None:
                self.logger.warning("跳过无法从文件名解析交易日的 wide_table 文件：%s", file_path)
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

    def _select_files_with_future(self, all_files: Sequence[Path], output_files: Sequence[Path]) -> list[Path]:
        """Read output-date files plus up to 20 later trade-date files."""

        output_file_set = set(output_files)
        output_indices = [idx for idx, file_path in enumerate(all_files) if file_path in output_file_set]
        if not output_indices:
            return []
        first_idx = min(output_indices)
        last_idx = min(len(all_files) - 1, max(output_indices) + self.MAX_FORWARD_DAYS)
        return list(all_files[first_idx : last_idx + 1])

    def _iter_file_batches(self, files: Sequence[Path]) -> Iterable[list[Path]]:
        for start_idx in range(0, len(files), self.output_batch_size):
            yield list(files[start_idx : start_idx + self.output_batch_size])

    def _load_wide_table_bars(self, parquet_files: Sequence[Path]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file, columns=list(self.REQUIRED_COLUMNS))
            self._validate_columns(df.columns, self.REQUIRED_COLUMNS, parquet_file)
            prepared = self._prepare_wide_table_bars(df.copy(), parquet_file)
            frames.append(prepared)
            self.logger.debug("读取 wide_table 文件完成：%s rows=%d", parquet_file, len(prepared))
        if not frames:
            return pd.DataFrame(columns=list(self.REQUIRED_COLUMNS))
        combined = pd.concat(frames, ignore_index=True)
        self.logger.info("wide_table_daily_bars 读取完成：files=%d, rows=%d", len(parquet_files), len(combined))
        return combined

    def _load_latest_adj_factor_by_stock(self, parquet_files: Sequence[Path]) -> pd.Series:
        """Load the latest valid adjustment factor for every stock from full history."""

        latest_adj_factor_by_stock: dict[str, float] = {}
        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file, columns=["ts_code", "adj_factor"])
            self._validate_columns(df.columns, ("ts_code", "adj_factor"), parquet_file)
            prepared = df.copy()
            prepared["ts_code"] = prepared["ts_code"].astype("string").str.strip()
            prepared["adj_factor"] = pd.to_numeric(prepared["adj_factor"], errors="coerce")
            prepared = prepared.loc[~self._is_null_or_empty(prepared["ts_code"]) & prepared["adj_factor"].notna(), ["ts_code", "adj_factor"]]
            if prepared.empty:
                continue
            latest_adj_factor_by_stock.update(prepared.drop_duplicates("ts_code", keep="last").set_index("ts_code")["adj_factor"].to_dict())

        if not latest_adj_factor_by_stock:
            raise ValueError("No valid adj_factor found in full wide_table_daily_bars history.")
        return pd.Series(latest_adj_factor_by_stock, dtype="float64")

    def _prepare_wide_table_bars(self, daily_df: pd.DataFrame, parquet_file: Path) -> pd.DataFrame:
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
            self.logger.warning("%s 存在 %d 行 close/adj_factor 缺失或非数值，对应 label 将产生 NaN。", parquet_file, missing_source_rows)
        return prepared

    def _generate_labels(self, bars_df: pd.DataFrame, latest_adj_factor_by_stock: pd.Series) -> pd.DataFrame:
        df = bars_df.copy()
        df = df.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)

        close = df["close"].astype("float64")
        adj_factor = df["adj_factor"].astype("float64")
        latest_factor = df["ts_code"].map(latest_adj_factor_by_stock).astype("float64")
        df["adj_close"] = close.mul(adj_factor).div(latest_factor.where(latest_factor.ne(0), np.nan))
        stock_group = df.groupby("ts_code", sort=False, group_keys=False)
        for horizon in self.HORIZONS:
            label_column = f"label_{horizon}d"
            rank_column = f"label_rank_{horizon}d"
            future_adj_close = stock_group["adj_close"].shift(-horizon)
            # 先为每只股票构造统一复权价格，得到全历史连续价格序列：
            #   latest_factor = adj_factor.iloc[-1]
            #   adj_close(t) = close(t) * adj_factor(t) / latest_factor
            # 再基于连续复权价格计算 t 到 t+h 的收益率：
            #   label_hd = adj_close(t+h) / adj_close(t) - 1
            df[label_column] = future_adj_close.div(df["adj_close"].where(df["adj_close"].ne(0), np.nan)) - 1
            df[rank_column] = df.groupby("trade_date", sort=False)[label_column].rank(pct=True)

        label_nan_counts = df.loc[:, list(self.LABEL_COLUMNS)].isna().sum().to_dict()
        rank_nan_counts = df.loc[:, list(self.RANK_COLUMNS)].isna().sum().to_dict()
        self.logger.info(
            "训练 label 计算完成：rows=%d, label_nan_counts=%s, rank_nan_counts=%s",
            len(df),
            label_nan_counts,
            rank_nan_counts,
        )
        return df

    def _write_partitioned_by_date(self, output_df: pd.DataFrame) -> int:
        files_written = 0
        for trade_date, date_df in output_df.groupby("trade_date", sort=True):
            trade_date_int = int(trade_date)
            trade_date_text = f"{trade_date_int:08d}"
            output_file = self.output_dir / f"year={trade_date_text[:4]}" / f"month={trade_date_text[4:6]}" / f"{trade_date_text}.parquet"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            date_df.to_parquet(output_file, index=False)
            files_written += 1
            self.logger.info("训练 label 文件生成完成：trade_date=%s, rows=%d -> %s", trade_date_text, len(date_df), output_file)
        return files_written

    def _output_columns(self) -> list[str]:
        return [*self.KEY_COLUMNS, "adj_close", *self.LABEL_COLUMNS, *self.RANK_COLUMNS]

    def _log_summary(self, summary: DailyLabelGenerateSummary, output_nan_counts: dict[str, int]) -> None:
        self.logger.info(
            "训练 label 生成汇总：files_read=%d, files_written=%d, rows_read=%d, rows_written=%d, "
            "start_date=%s, end_date=%s, labels=%d, ranks=%d, output=%s",
            summary.files_read,
            summary.files_written,
            summary.rows_read,
            summary.rows_written,
            summary.start_date,
            summary.end_date,
            len(summary.label_columns),
            len(summary.rank_columns),
            summary.output_dir,
        )
        if summary.duplicate_key_rows:
            self.logger.warning("输入数据存在 %d 行重复 trade_date+ts_code，请关注上游宽表构建结果。", summary.duplicate_key_rows)
        if summary.missing_source_rows:
            self.logger.warning("输入数据存在 %d 行 close/adj_factor 缺失，相关 label/rank 会保留为 NaN。", summary.missing_source_rows)
        if summary.zero_adj_close_rows:
            self.logger.warning("输入数据存在 %d 行 adj_close=0，相关 label/rank 已置为 NaN。", summary.zero_adj_close_rows)
        label_nan_counts = {column: output_nan_counts[column] for column in self.LABEL_COLUMNS}
        rank_nan_counts = {column: output_nan_counts[column] for column in self.RANK_COLUMNS}
        self.logger.info("输出 label 缺失值统计：%s", label_nan_counts)
        self.logger.info("输出 rank label 缺失值统计：%s", rank_nan_counts)

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
    target_log_file = Path(log_file) if log_file is not None else DailyLabelGenerator.DEFAULT_LOG_FILE
    target_log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(target_log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate forward-return training labels from A-share daily wide-table bars.")
    parser.add_argument("--input-dir", default=str(DailyLabelGenerator.DEFAULT_INPUT_DIR), help="Processed wide_table_daily_bars directory.")
    parser.add_argument(
        "--output-dir",
        default=str(DailyLabelGenerator.DEFAULT_OUTPUT_DIR),
        help="Output directory for partitioned generated label files.",
    )
    parser.add_argument("--start-date", default=None, help="Optional inclusive output start date in YYYYMMDD format.")
    parser.add_argument("--end-date", default=None, help="Optional inclusive output end date in YYYYMMDD format.")
    parser.add_argument(
        "--output-batch-size",
        type=int,
        default=120,
        help="Number of output trade-date files per batch. Smaller values reduce memory usage.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    parser.add_argument("--log-file", default=str(DailyLabelGenerator.DEFAULT_LOG_FILE), help="Log file path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> DailyLabelGenerateSummary:
    args = parse_args(argv)
    configure_logging(log_level=args.log_level, log_file=args.log_file)
    generator = DailyLabelGenerator(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        output_batch_size=args.output_batch_size,
    )
    return generator.process()


if __name__ == "__main__":
    # 使用方法：
    # PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
    # python -m utils.label_generator.daily_label_generator \
    #   --output-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/generated_label/daily_labels" \
    #   --log-file "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/label_generate/daily_labels.log"
    main()


__all__ = [
    "DailyLabelGenerateSummary",
    "DailyLabelGenerator",
    "configure_logging",
    "main",
    "parse_args",
]
