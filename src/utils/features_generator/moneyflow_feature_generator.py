"""Generate money-flow factors from cleaned A-share moneyflow data.

The generator treats the cleaned source datasets as read-only and writes factor
files to a separate partitioned output directory.  The output preserves the
``year=YYYY/month=MM/YYYYMMDD.parquet`` layout used by ``cleaned_data/daily_bars``
so downstream factor jobs can consume data by trade date without touching
original or cleaned inputs.
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
class MoneyFlowFeatureGenerateSummary:
    """Aggregate statistics for money-flow factor generation."""

    moneyflow_input_file: Path
    daily_bars_input_dir: Path
    output_dir: Path
    daily_bar_files_read: int
    files_written: int
    moneyflow_rows_read: int
    daily_bar_rows_read: int
    rows_written: int
    start_date: int | None
    end_date: int | None
    factor_columns: tuple[str, ...]
    duplicate_key_rows: int
    missing_amount_rows: int
    zero_amount_rows: int


class MoneyFlowFeatureGenerator:
    """Generate money-flow factors from cleaned moneyflow and daily bar amount.

    Factors generated from buy/sell amount columns and daily-bar ``amount``:

    - Large-order net inflow: ``lg_net_inflow``
    - Extra-large-order net inflow: ``elg_net_inflow``
    - Main-money net inflow: ``main_net_inflow``
    - Main-money ratio: ``main_net_ratio``
    - Retail net inflow: ``retail_net_inflow``
    - Retail ratio: ``retail_ratio``
    - Main-vs-retail difference: ``main_retail_diff``
    - Main-money rolling sums: ``main_net_5``, ``main_net_10``, ``main_net_20``
    - Main-money moving averages: ``mf_ma5``, ``mf_ma20``
    - Money-flow acceleration: ``mf_acceleration``

    Rolling calculations are performed independently for each ``ts_code`` after
    sorting by ``trade_date``.  Source parquet files are never modified.  Tushare
    daily-bars ``amount`` is normally in thousand yuan while moneyflow amount
    columns are normally in ten-thousand yuan, so ``amount_unit_scale`` defaults
    to ``0.1`` before ratio calculation.  Set it to ``1.0`` if upstream data has
    already normalized units.
    """

    DEFAULT_MONEYFLOW_FILE = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/moneyflow/moneyflow.parquet"
    )
    DEFAULT_DAILY_BARS_DIR = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/daily_bars")
    DEFAULT_OUTPUT_DIR = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/features_data/moneyflow_factors")
    DEFAULT_LOG_FILE = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/features_generate/moneyflow_features.log")

    KEY_COLUMNS: tuple[str, str] = ("trade_date", "ts_code")
    MONEYFLOW_AMOUNT_COLUMNS: tuple[str, ...] = (
        "buy_sm_amount",
        "sell_sm_amount",
        "buy_md_amount",
        "sell_md_amount",
        "buy_lg_amount",
        "sell_lg_amount",
        "buy_elg_amount",
        "sell_elg_amount",
        "net_mf_amount",
    )
    MONEYFLOW_REQUIRED_COLUMNS: tuple[str, ...] = (*KEY_COLUMNS, *MONEYFLOW_AMOUNT_COLUMNS)
    DAILY_BARS_REQUIRED_COLUMNS: tuple[str, ...] = (*KEY_COLUMNS, "amount")
    FACTOR_COLUMNS: tuple[str, ...] = (
        "lg_net_inflow",
        "elg_net_inflow",
        "main_net_inflow",
        "main_net_ratio",
        "retail_net_inflow",
        "retail_ratio",
        "main_retail_diff",
        "main_net_5",
        "main_net_10",
        "main_net_20",
        "mf_ma5",
        "mf_ma20",
        "mf_acceleration",
    )
    MAX_LOOKBACK_DAYS = 20

    def __init__(
        self,
        moneyflow_file: str | Path | None = None,
        daily_bars_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        start_date: str | int | None = None,
        end_date: str | int | None = None,
        amount_unit_scale: float = 0.1,
        logger: logging.Logger | None = None,
    ) -> None:
        self.moneyflow_file = Path(moneyflow_file or self.DEFAULT_MONEYFLOW_FILE)
        self.daily_bars_dir = Path(daily_bars_dir or self.DEFAULT_DAILY_BARS_DIR)
        self.output_dir = Path(output_dir or self.DEFAULT_OUTPUT_DIR)
        self.start_date = self._parse_optional_date(start_date, "start_date")
        self.end_date = self._parse_optional_date(end_date, "end_date")
        self.amount_unit_scale = float(amount_unit_scale)
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self._validate_inputs()

    def process(self) -> MoneyFlowFeatureGenerateSummary:
        """Generate factor parquet files and return aggregate processing stats."""

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            "开始生成资金流特征因子：moneyflow=%s, daily_bars=%s, output=%s, amount_unit_scale=%s",
            self.moneyflow_file,
            self.daily_bars_dir,
            self.output_dir,
            self.amount_unit_scale,
        )
        self.logger.info("输入 moneyflow 与 daily_bars 均按只读处理，生成结果仅写入 features_data 对应目录。")

        moneyflow_df = self._load_moneyflow()
        output_dates = self._select_output_dates(moneyflow_df["trade_date"])
        if not output_dates:
            raise FileNotFoundError(
                f"No moneyflow rows found in output date range [{self.start_date}, {self.end_date}] from {self.moneyflow_file}"
            )

        read_dates = self._select_dates_with_history(sorted(moneyflow_df["trade_date"].dropna().unique()), output_dates)
        history_df = moneyflow_df.loc[moneyflow_df["trade_date"].isin(read_dates)].copy()
        daily_bar_files = self._daily_bar_files_for_dates(read_dates)
        amount_df = self._load_daily_amount(daily_bar_files)
        duplicate_key_rows = int(history_df.duplicated(subset=list(self.KEY_COLUMNS), keep=False).sum())
        if duplicate_key_rows:
            self.logger.warning("输入 moneyflow 存在 %d 行重复 trade_date+ts_code，将全部保留并按原顺序计算。", duplicate_key_rows)

        merged_df = history_df.merge(amount_df, on=list(self.KEY_COLUMNS), how="left", validate="many_to_one")
        missing_amount_rows = int(merged_df["amount"].isna().sum())
        if missing_amount_rows:
            self.logger.warning("有 %d 行资金流记录未能匹配 daily_bars.amount，资金占比因子将为 NaN。", missing_amount_rows)

        factor_df = self._generate_features(merged_df)
        output_df = factor_df.loc[factor_df["trade_date"].isin(output_dates), self._output_columns()].copy()
        output_df = output_df.sort_values(["trade_date", "ts_code"], kind="mergesort").reset_index(drop=True)
        files_written = self._write_partitioned_by_date(output_df)

        ratio_denominator = self._ratio_denominator(factor_df["amount"])
        summary = MoneyFlowFeatureGenerateSummary(
            moneyflow_input_file=self.moneyflow_file,
            daily_bars_input_dir=self.daily_bars_dir,
            output_dir=self.output_dir,
            daily_bar_files_read=len(daily_bar_files),
            files_written=files_written,
            moneyflow_rows_read=len(history_df),
            daily_bar_rows_read=len(amount_df),
            rows_written=len(output_df),
            start_date=self.start_date,
            end_date=self.end_date,
            factor_columns=self.FACTOR_COLUMNS,
            duplicate_key_rows=duplicate_key_rows,
            missing_amount_rows=missing_amount_rows,
            zero_amount_rows=int(ratio_denominator.eq(0).sum()),
        )
        self._log_summary(summary, output_df)
        return summary

    def _validate_inputs(self) -> None:
        if not self.moneyflow_file.exists() or not self.moneyflow_file.is_file():
            raise FileNotFoundError(f"Moneyflow input file does not exist: {self.moneyflow_file}")
        if not self.daily_bars_dir.exists() or not self.daily_bars_dir.is_dir():
            raise NotADirectoryError(f"Daily bars input directory does not exist or is not a directory: {self.daily_bars_dir}")
        if self.start_date is not None and self.end_date is not None and self.start_date > self.end_date:
            raise ValueError(f"start_date must be <= end_date, got {self.start_date} > {self.end_date}")
        if not np.isfinite(self.amount_unit_scale) or self.amount_unit_scale <= 0:
            raise ValueError(f"amount_unit_scale must be a positive finite number, got {self.amount_unit_scale}")

    def _load_moneyflow(self) -> pd.DataFrame:
        df = pd.read_parquet(self.moneyflow_file, columns=list(self.MONEYFLOW_REQUIRED_COLUMNS))
        self._validate_columns(df.columns, self.MONEYFLOW_REQUIRED_COLUMNS, self.moneyflow_file)
        prepared = self._prepare_moneyflow(df)
        self.logger.info("moneyflow 读取完成：rows=%d, columns=%d", len(prepared), len(prepared.columns))
        return prepared

    def _prepare_moneyflow(self, moneyflow_df: pd.DataFrame) -> pd.DataFrame:
        prepared = moneyflow_df.copy()
        prepared["trade_date"] = self._normalize_yyyymmdd(prepared["trade_date"], "trade_date", self.moneyflow_file)
        prepared["ts_code"] = prepared["ts_code"].astype("string").str.strip()
        invalid_key_mask = prepared["trade_date"].isna() | self._is_null_or_empty(prepared["ts_code"])
        invalid_key_rows = int(invalid_key_mask.sum())
        if invalid_key_rows:
            raise ValueError(f"{self.moneyflow_file} has {invalid_key_rows} invalid trade_date/ts_code rows.")

        for column in self.MONEYFLOW_AMOUNT_COLUMNS:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        missing_source_rows = int(prepared.loc[:, list(self.MONEYFLOW_AMOUNT_COLUMNS)].isna().any(axis=1).sum())
        if missing_source_rows:
            self.logger.warning("moneyflow 存在 %d 行资金流源字段缺失/非数值，相关因子将产生 NaN。", missing_source_rows)
        return prepared

    def _select_output_dates(self, all_dates: pd.Series) -> list[int]:
        unique_dates = sorted(int(date) for date in all_dates.dropna().unique())
        output_dates: list[int] = []
        for trade_date in unique_dates:
            if self.start_date is not None and trade_date < self.start_date:
                continue
            if self.end_date is not None and trade_date > self.end_date:
                continue
            output_dates.append(trade_date)
        self.logger.info("资金流待输出交易日数=%d", len(output_dates))
        return output_dates

    def _select_dates_with_history(self, all_dates: Sequence[int], output_dates: Sequence[int]) -> list[int]:
        """Read up to 20 prior moneyflow trade dates before the first output date."""

        date_to_idx = {date: idx for idx, date in enumerate(all_dates)}
        output_indices = [date_to_idx[date] for date in output_dates if date in date_to_idx]
        if not output_indices:
            return []
        first_idx = max(0, min(output_indices) - self.MAX_LOOKBACK_DAYS)
        last_idx = max(output_indices)
        read_dates = list(all_dates[first_idx : last_idx + 1])
        self.logger.info(
            "为保证 rolling 历史窗口，实际读取资金流交易日数=%d，范围=%s-%s",
            len(read_dates),
            read_dates[0] if read_dates else None,
            read_dates[-1] if read_dates else None,
        )
        return read_dates

    def _daily_bar_files_for_dates(self, trade_dates: Sequence[int]) -> list[Path]:
        files: list[Path] = []
        missing_files: list[Path] = []
        for trade_date in trade_dates:
            trade_date_text = f"{int(trade_date):08d}"
            file_path = self.daily_bars_dir / f"year={trade_date_text[:4]}" / f"month={trade_date_text[4:6]}" / f"{trade_date_text}.parquet"
            if file_path.exists():
                files.append(file_path)
            else:
                missing_files.append(file_path)
        if missing_files:
            self.logger.warning("缺少 %d 个 daily_bars 日期文件，相关日期资金占比因子可能为 NaN；示例=%s", len(missing_files), missing_files[:5])
        return files

    def _load_daily_amount(self, parquet_files: Sequence[Path]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for parquet_file in parquet_files:
            df = pd.read_parquet(parquet_file, columns=list(self.DAILY_BARS_REQUIRED_COLUMNS))
            self._validate_columns(df.columns, self.DAILY_BARS_REQUIRED_COLUMNS, parquet_file)
            prepared = self._prepare_daily_amount(df, parquet_file)
            frames.append(prepared)
            self.logger.debug("读取 daily_bars amount 文件完成：%s rows=%d", parquet_file, len(prepared))
        if not frames:
            return pd.DataFrame(columns=list(self.DAILY_BARS_REQUIRED_COLUMNS))
        combined = pd.concat(frames, ignore_index=True)
        duplicate_amount_rows = int(combined.duplicated(subset=list(self.KEY_COLUMNS), keep=False).sum())
        if duplicate_amount_rows:
            raise ValueError(f"daily_bars amount has {duplicate_amount_rows} duplicate trade_date+ts_code rows; cannot many_to_one merge.")
        self.logger.info("daily_bars amount 读取完成：files=%d, rows=%d", len(parquet_files), len(combined))
        return combined

    def _prepare_daily_amount(self, daily_df: pd.DataFrame, parquet_file: Path) -> pd.DataFrame:
        prepared = daily_df.copy()
        prepared["trade_date"] = self._normalize_yyyymmdd(prepared["trade_date"], "trade_date", parquet_file)
        prepared["ts_code"] = prepared["ts_code"].astype("string").str.strip()
        invalid_key_mask = prepared["trade_date"].isna() | self._is_null_or_empty(prepared["ts_code"])
        invalid_key_rows = int(invalid_key_mask.sum())
        if invalid_key_rows:
            raise ValueError(f"{parquet_file} has {invalid_key_rows} invalid trade_date/ts_code rows.")
        prepared["amount"] = pd.to_numeric(prepared["amount"], errors="coerce")
        missing_amount_rows = int(prepared["amount"].isna().sum())
        if missing_amount_rows:
            self.logger.warning("%s 存在 %d 行 amount 缺失/非数值，资金占比因子将产生 NaN。", parquet_file, missing_amount_rows)
        return prepared

    def _generate_features(self, moneyflow_df: pd.DataFrame) -> pd.DataFrame:
        df = moneyflow_df.copy()
        df = df.sort_values(["ts_code", "trade_date"], kind="mergesort").reset_index(drop=True)

        df["lg_net_inflow"] = df["buy_lg_amount"] - df["sell_lg_amount"]
        df["elg_net_inflow"] = df["buy_elg_amount"] - df["sell_elg_amount"]
        df["main_net_inflow"] = df["lg_net_inflow"] + df["elg_net_inflow"]
        df["retail_net_inflow"] = df["buy_sm_amount"] - df["sell_sm_amount"]

        ratio_denominator = self._ratio_denominator(df["amount"]).replace(0, np.nan)
        df["main_net_ratio"] = df["main_net_inflow"].div(ratio_denominator)
        df["retail_ratio"] = df["retail_net_inflow"].div(ratio_denominator)
        df["main_retail_diff"] = df["main_net_ratio"] - df["retail_ratio"]

        stock_group = df.groupby("ts_code", sort=False, group_keys=False)
        for window in (5, 10, 20):
            df[f"main_net_{window}"] = stock_group["main_net_inflow"].transform(
                lambda series: series.rolling(window=window, min_periods=window).sum()
            )
        for window in (5, 20):
            df[f"mf_ma{window}"] = stock_group["main_net_inflow"].transform(
                lambda series: series.rolling(window=window, min_periods=window).mean()
            )
        df["mf_acceleration"] = df["main_net_5"] - df["main_net_20"]

        nan_counts = df.loc[:, list(self.FACTOR_COLUMNS)].isna().sum().to_dict()
        self.logger.info("资金流因子计算完成：rows=%d, factor_nan_counts=%s", len(df), nan_counts)
        return df

    def _ratio_denominator(self, amount: pd.Series) -> pd.Series:
        return pd.to_numeric(amount, errors="coerce") * self.amount_unit_scale

    def _write_partitioned_by_date(self, output_df: pd.DataFrame) -> int:
        files_written = 0
        for trade_date, date_df in output_df.groupby("trade_date", sort=True):
            trade_date_int = int(trade_date)
            trade_date_text = f"{trade_date_int:08d}"
            output_file = self.output_dir / f"year={trade_date_text[:4]}" / f"month={trade_date_text[4:6]}" / f"{trade_date_text}.parquet"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            date_df.to_parquet(output_file, index=False)
            files_written += 1
            self.logger.info("资金流因子文件生成完成：trade_date=%s, rows=%d -> %s", trade_date_text, len(date_df), output_file)
        return files_written

    def _output_columns(self) -> list[str]:
        return [*self.KEY_COLUMNS, "amount", *self.MONEYFLOW_AMOUNT_COLUMNS, *self.FACTOR_COLUMNS]

    def _log_summary(self, summary: MoneyFlowFeatureGenerateSummary, output_df: pd.DataFrame) -> None:
        self.logger.info(
            "资金流因子生成汇总：moneyflow_rows_read=%d, daily_bar_files_read=%d, daily_bar_rows_read=%d, "
            "files_written=%d, rows_written=%d, start_date=%s, end_date=%s, factors=%d, output=%s",
            summary.moneyflow_rows_read,
            summary.daily_bar_files_read,
            summary.daily_bar_rows_read,
            summary.files_written,
            summary.rows_written,
            summary.start_date,
            summary.end_date,
            len(summary.factor_columns),
            summary.output_dir,
        )
        if summary.duplicate_key_rows:
            self.logger.warning("输入 moneyflow 存在 %d 行重复 trade_date+ts_code，请关注上游清洗结果。", summary.duplicate_key_rows)
        if summary.missing_amount_rows:
            self.logger.warning("输出相关计算中存在 %d 行缺失 amount，资金占比因子已置为 NaN。", summary.missing_amount_rows)
        if summary.zero_amount_rows:
            self.logger.warning("输出相关计算中存在 %d 行 amount 经单位转换后为 0，资金占比因子已置为 NaN。", summary.zero_amount_rows)
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


def configure_logging(log_level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure console and file logging for CLI usage."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    target_log_file = Path(log_file) if log_file is not None else MoneyFlowFeatureGenerator.DEFAULT_LOG_FILE
    target_log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(target_log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate money-flow factors from cleaned A-share moneyflow data.")
    parser.add_argument("--moneyflow-file", default=str(MoneyFlowFeatureGenerator.DEFAULT_MONEYFLOW_FILE), help="Cleaned moneyflow parquet file.")
    parser.add_argument("--daily-bars-dir", default=str(MoneyFlowFeatureGenerator.DEFAULT_DAILY_BARS_DIR), help="Cleaned daily_bars directory.")
    parser.add_argument(
        "--output-dir",
        default=str(MoneyFlowFeatureGenerator.DEFAULT_OUTPUT_DIR),
        help="Output directory for partitioned money-flow factors.",
    )
    parser.add_argument("--start-date", default=None, help="Optional inclusive output start date in YYYYMMDD format.")
    parser.add_argument("--end-date", default=None, help="Optional inclusive output end date in YYYYMMDD format.")
    parser.add_argument(
        "--amount-unit-scale",
        type=float,
        default=0.1,
        help="Scale applied to daily_bars.amount before ratio calculation; default 0.1 converts thousand yuan to ten-thousand yuan.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    parser.add_argument("--log-file", default=str(MoneyFlowFeatureGenerator.DEFAULT_LOG_FILE), help="Log file path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> MoneyFlowFeatureGenerateSummary:
    args = parse_args(argv)
    configure_logging(log_level=args.log_level, log_file=args.log_file)
    generator = MoneyFlowFeatureGenerator(
        moneyflow_file=args.moneyflow_file,
        daily_bars_dir=args.daily_bars_dir,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
        amount_unit_scale=args.amount_unit_scale,
    )
    return generator.process()


if __name__ == "__main__":
    # 使用方法：
    # PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
    # python -m utils.features_generator.moneyflow_feature_generator \
    #   --output-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/features_data/moneyflow_factors" \
    #   --log-file "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/features_generate/moneyflow_features.log"
    main()


__all__ = [
    "MoneyFlowFeatureGenerateSummary",
    "MoneyFlowFeatureGenerator",
    "configure_logging",
    "main",
    "parse_args",
]
