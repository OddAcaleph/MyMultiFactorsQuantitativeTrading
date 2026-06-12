"""Generate industry factors from price-volume and fundamental factors.

The generator treats all upstream datasets as read-only and writes daily
partitioned parquet files under ``data/features_data/industry_factors`` using
the same ``year=YYYY/month=MM/YYYYMMDD.parquet`` layout as the existing feature
factor directories.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class IndustryFeatureGenerateSummary:
    """Aggregate statistics for industry factor generation."""

    industry_input_file: Path
    price_volume_input_dir: Path
    fundamental_input_dir: Path
    output_dir: Path
    price_volume_files_read: int
    fundamental_files_read: int
    missing_fundamental_files: int
    files_written: int
    price_volume_rows_read: int
    fundamental_rows_read: int
    rows_written: int
    start_date: int | None
    end_date: int | None
    factor_columns: tuple[str, ...]
    l1_onehot_columns: tuple[str, ...]
    missing_l1_rows: int
    duplicate_price_volume_key_rows: int
    duplicate_fundamental_key_rows: int


class IndustryFeatureGenerator:
    """Generate L1 industry one-hot, industry returns and neutralized factors.

    Generated factor columns:

    - L1 industry one-hot columns: ``L1_xxx`` from ``L1_industry_name`` only.
    - Industry returns: ``industry_ret_1``, ``industry_ret_5`` and
      ``industry_ret_20`` as same-day L1 industry cross-sectional means.
    - Relative strength: ``relative_strength_5 = ret_5 - industry_ret_5`` and
      ``relative_strength_20 = ret_20 - industry_ret_20``.
    - Industry-neutral financial factors: ``roe_ind_neutral``,
      ``roa_ind_neutral`` and ``revenue_yoy_ind_neutral``.

    For daily return, ``ret_1`` is used when present in price-volume factors;
    otherwise ``pct_chg`` is converted from percent to decimal return. Source
    price-volume, fundamental and industry parquet files are never modified.
    """

    DEFAULT_INDUSTRY_FILE = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/industry/industry.parquet"
    )
    DEFAULT_PRICE_VOLUME_DIR = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/features_data/price_volume_factors"
    )
    DEFAULT_FUNDAMENTAL_DIR = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/features_data/fundamental_factors"
    )
    DEFAULT_OUTPUT_DIR = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/features_data/industry_factors"
    )
    DEFAULT_LOG_FILE = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/features_generate/industry_features.log")

    KEY_COLUMNS: tuple[str, str] = ("trade_date", "ts_code")
    INDUSTRY_KEY_COLUMN = "ts_code"
    INDUSTRY_LEVEL_COLUMN = "L1_industry_name"
    PRICE_RETURN_COLUMNS: tuple[str, ...] = ("ret_5", "ret_20")
    PRICE_DAILY_RETURN_CANDIDATES: tuple[str, str] = ("ret_1", "pct_chg")
    FUNDAMENTAL_COLUMNS: tuple[str, ...] = ("roe", "roa", "revenue_yoy")
    DERIVED_FACTOR_COLUMNS: tuple[str, ...] = (
        "industry_ret_1",
        "industry_ret_5",
        "industry_ret_20",
        "relative_strength_5",
        "relative_strength_20",
        "roe_ind_neutral",
        "roa_ind_neutral",
        "revenue_yoy_ind_neutral",
    )
    INDUSTRY_REQUIRED_COLUMNS: tuple[str, str] = (INDUSTRY_KEY_COLUMN, INDUSTRY_LEVEL_COLUMN)
    PRICE_REQUIRED_COLUMNS: tuple[str, ...] = (*KEY_COLUMNS, *PRICE_RETURN_COLUMNS)
    FUNDAMENTAL_REQUIRED_COLUMNS: tuple[str, ...] = (*KEY_COLUMNS, *FUNDAMENTAL_COLUMNS)

    def __init__(
        self,
        industry_file: str | Path | None = None,
        price_volume_dir: str | Path | None = None,
        fundamental_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        start_date: str | int | None = None,
        end_date: str | int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.industry_file = Path(industry_file or self.DEFAULT_INDUSTRY_FILE)
        self.price_volume_dir = Path(price_volume_dir or self.DEFAULT_PRICE_VOLUME_DIR)
        self.fundamental_dir = Path(fundamental_dir or self.DEFAULT_FUNDAMENTAL_DIR)
        self.output_dir = Path(output_dir or self.DEFAULT_OUTPUT_DIR)
        self.start_date = self._parse_optional_date(start_date, "start_date")
        self.end_date = self._parse_optional_date(end_date, "end_date")
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self._validate_inputs()

    def process(self) -> IndustryFeatureGenerateSummary:
        """Generate daily industry factor parquet files and return stats."""

        price_files = self._list_output_price_volume_files()
        if not price_files:
            raise FileNotFoundError(
                f"No price-volume parquet files found under {self.price_volume_dir}/year=*/month=*/*.parquet "
                f"for output date range [{self.start_date}, {self.end_date}]"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            "开始生成行业特征因子：industry=%s, price_volume=%s, fundamental=%s, output=%s",
            self.industry_file,
            self.price_volume_dir,
            self.fundamental_dir,
            self.output_dir,
        )
        self.logger.info("输入 industry/price_volume/fundamental parquet 均按只读处理，生成结果仅写入 features_data 对应目录。")

        industry_df, l1_onehot_columns = self._load_industry_mapping()

        files_written = 0
        fundamental_files_read = 0
        missing_fundamental_files = 0
        price_volume_rows_read = 0
        fundamental_rows_read = 0
        rows_written = 0
        missing_l1_rows = 0
        duplicate_price_volume_key_rows = 0
        duplicate_fundamental_key_rows = 0

        for price_file in price_files:
            trade_date = self._date_from_file_name(price_file)
            if trade_date is None:
                continue

            price_df = self._load_price_volume_features(price_file)
            price_volume_rows_read += len(price_df)
            duplicate_price_volume_key_rows += int(price_df.duplicated(subset=list(self.KEY_COLUMNS), keep=False).sum())

            fundamental_file = self._matching_fundamental_file(price_file)
            if fundamental_file.exists():
                fundamental_df = self._load_fundamental_features(fundamental_file)
                fundamental_files_read += 1
                fundamental_rows_read += len(fundamental_df)
                duplicate_fundamental_key_rows += int(fundamental_df.duplicated(subset=list(self.KEY_COLUMNS), keep=False).sum())
            else:
                missing_fundamental_files += 1
                self.logger.warning("未找到同日财务因子文件：%s；当日行业中性化财务因子将输出 NaN。", fundamental_file)
                fundamental_df = self._empty_fundamental_frame()

            output_df = self._generate_daily_features(price_df, fundamental_df, industry_df, l1_onehot_columns)
            self._log_daily_source_coverage(trade_date, price_df, fundamental_df, output_df)
            rows_written += len(output_df)
            missing_l1_rows += int(output_df[self.INDUSTRY_LEVEL_COLUMN].isna().sum())
            self._write_daily_file(output_df, trade_date)
            files_written += 1

        summary = IndustryFeatureGenerateSummary(
            industry_input_file=self.industry_file,
            price_volume_input_dir=self.price_volume_dir,
            fundamental_input_dir=self.fundamental_dir,
            output_dir=self.output_dir,
            price_volume_files_read=len(price_files),
            fundamental_files_read=fundamental_files_read,
            missing_fundamental_files=missing_fundamental_files,
            files_written=files_written,
            price_volume_rows_read=price_volume_rows_read,
            fundamental_rows_read=fundamental_rows_read,
            rows_written=rows_written,
            start_date=self.start_date,
            end_date=self.end_date,
            factor_columns=(*l1_onehot_columns, *self.DERIVED_FACTOR_COLUMNS),
            l1_onehot_columns=l1_onehot_columns,
            missing_l1_rows=missing_l1_rows,
            duplicate_price_volume_key_rows=duplicate_price_volume_key_rows,
            duplicate_fundamental_key_rows=duplicate_fundamental_key_rows,
        )
        self._log_summary(summary)
        return summary

    def _validate_inputs(self) -> None:
        if not self.industry_file.exists() or not self.industry_file.is_file():
            raise FileNotFoundError(f"Industry input file does not exist: {self.industry_file}")
        if not self.price_volume_dir.exists() or not self.price_volume_dir.is_dir():
            raise NotADirectoryError(f"Price-volume input directory does not exist or is not a directory: {self.price_volume_dir}")
        if not self.fundamental_dir.exists() or not self.fundamental_dir.is_dir():
            raise NotADirectoryError(f"Fundamental input directory does not exist or is not a directory: {self.fundamental_dir}")
        if self.start_date is not None and self.end_date is not None and self.start_date > self.end_date:
            raise ValueError(f"start_date must be <= end_date, got {self.start_date} > {self.end_date}")

    def _load_industry_mapping(self) -> tuple[pd.DataFrame, tuple[str, ...]]:
        df = pd.read_parquet(self.industry_file, columns=list(self.INDUSTRY_REQUIRED_COLUMNS))
        self._validate_columns(df.columns, self.INDUSTRY_REQUIRED_COLUMNS, self.industry_file)

        prepared = df.copy().reset_index(names="_raw_row")
        prepared[self.INDUSTRY_KEY_COLUMN] = prepared[self.INDUSTRY_KEY_COLUMN].astype("string").str.strip()
        prepared[self.INDUSTRY_LEVEL_COLUMN] = prepared[self.INDUSTRY_LEVEL_COLUMN].astype("string").str.strip()
        prepared.loc[prepared[self.INDUSTRY_LEVEL_COLUMN].eq(""), self.INDUSTRY_LEVEL_COLUMN] = pd.NA

        missing_ts_code_rows = int(self._is_null_or_empty(prepared[self.INDUSTRY_KEY_COLUMN]).sum())
        if missing_ts_code_rows:
            self.logger.warning("industry 映射存在 %d 行 ts_code 缺失，这些记录不会参与行业因子生成。", missing_ts_code_rows)
            prepared = prepared.loc[~self._is_null_or_empty(prepared[self.INDUSTRY_KEY_COLUMN])].copy()
        if prepared.empty:
            raise ValueError("No valid industry records remain after removing missing ts_code rows.")

        duplicate_ts_code_rows = int(prepared.duplicated(subset=[self.INDUSTRY_KEY_COLUMN], keep=False).sum())
        if duplicate_ts_code_rows:
            self.logger.warning("industry 映射存在 %d 行重复 ts_code，按原文件顺序保留最后一条。", duplicate_ts_code_rows)
            prepared = prepared.sort_values("_raw_row", kind="mergesort").drop_duplicates(
                subset=[self.INDUSTRY_KEY_COLUMN], keep="last"
            )

        l1_categories = sorted(
            category for category in prepared[self.INDUSTRY_LEVEL_COLUMN].dropna().astype("string").unique().tolist() if category
        )
        l1_onehot_columns = tuple(f"L1_{category}" for category in l1_categories)
        self.logger.info("行业 L1 映射读取完成：stocks=%d, L1_categories=%d", len(prepared), len(l1_onehot_columns))

        prepared = prepared.loc[:, [self.INDUSTRY_KEY_COLUMN, self.INDUSTRY_LEVEL_COLUMN]].sort_values(
            self.INDUSTRY_KEY_COLUMN, kind="mergesort"
        )
        return prepared.reset_index(drop=True), l1_onehot_columns

    def _list_output_price_volume_files(self) -> list[Path]:
        files = sorted(self.price_volume_dir.glob("year=*/month=*/*.parquet"))
        output_files: list[Path] = []
        for file_path in files:
            file_date = self._date_from_file_name(file_path)
            if file_date is None:
                self.logger.warning("跳过无法从文件名解析交易日的价格量因子文件：%s", file_path)
                continue
            if self.start_date is not None and file_date < self.start_date:
                continue
            if self.end_date is not None and file_date > self.end_date:
                continue
            output_files.append(file_path)
        self.logger.info("行业因子待输出交易日文件数=%d", len(output_files))
        return output_files

    def _load_price_volume_features(self, parquet_file: Path) -> pd.DataFrame:
        df = pd.read_parquet(parquet_file)
        self._validate_columns(df.columns, self.PRICE_REQUIRED_COLUMNS, parquet_file)
        if not any(column in df.columns for column in self.PRICE_DAILY_RETURN_CANDIDATES):
            raise ValueError(f"{parquet_file} is missing daily return source column: one of {self.PRICE_DAILY_RETURN_CANDIDATES}")

        columns = [*self.PRICE_REQUIRED_COLUMNS]
        daily_return_source = "ret_1" if "ret_1" in df.columns else "pct_chg"
        columns.append(daily_return_source)
        prepared = df.loc[:, columns].copy()
        prepared["trade_date"] = self._normalize_yyyymmdd(prepared["trade_date"], "trade_date", parquet_file)
        prepared["ts_code"] = prepared["ts_code"].astype("string").str.strip()
        for column in [daily_return_source, *self.PRICE_RETURN_COLUMNS]:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        prepared["ret_1"] = prepared[daily_return_source] if daily_return_source == "ret_1" else prepared[daily_return_source] / 100.0
        return prepared.loc[:, [*self.KEY_COLUMNS, "ret_1", *self.PRICE_RETURN_COLUMNS]]

    def _load_fundamental_features(self, parquet_file: Path) -> pd.DataFrame:
        df = pd.read_parquet(parquet_file, columns=list(self.FUNDAMENTAL_REQUIRED_COLUMNS))
        self._validate_columns(df.columns, self.FUNDAMENTAL_REQUIRED_COLUMNS, parquet_file)
        prepared = df.copy()
        prepared["trade_date"] = self._normalize_yyyymmdd(prepared["trade_date"], "trade_date", parquet_file)
        prepared["ts_code"] = prepared["ts_code"].astype("string").str.strip()
        for column in self.FUNDAMENTAL_COLUMNS:
            prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        return prepared.loc[:, [*self.KEY_COLUMNS, *self.FUNDAMENTAL_COLUMNS]]

    def _empty_fundamental_frame(self) -> pd.DataFrame:
        return pd.DataFrame(columns=[*self.KEY_COLUMNS, *self.FUNDAMENTAL_COLUMNS])

    def _generate_daily_features(
        self,
        price_df: pd.DataFrame,
        fundamental_df: pd.DataFrame,
        industry_df: pd.DataFrame,
        l1_onehot_columns: Sequence[str],
    ) -> pd.DataFrame:
        df = price_df.merge(fundamental_df, on=list(self.KEY_COLUMNS), how="left", validate="one_to_one")
        df = df.merge(industry_df, on="ts_code", how="left", validate="many_to_one")

        for onehot_column in l1_onehot_columns:
            category = onehot_column.removeprefix("L1_")
            df[onehot_column] = df[self.INDUSTRY_LEVEL_COLUMN].eq(category).fillna(False).astype("int8")

        industry_group = df.groupby(self.INDUSTRY_LEVEL_COLUMN, dropna=True, sort=False)
        df["industry_ret_1"] = industry_group["ret_1"].transform("mean")
        df["industry_ret_5"] = industry_group["ret_5"].transform("mean")
        df["industry_ret_20"] = industry_group["ret_20"].transform("mean")
        df["relative_strength_5"] = df["ret_5"] - df["industry_ret_5"]
        df["relative_strength_20"] = df["ret_20"] - df["industry_ret_20"]

        for source_column, neutral_column in (
            ("roe", "roe_ind_neutral"),
            ("roa", "roa_ind_neutral"),
            ("revenue_yoy", "revenue_yoy_ind_neutral"),
        ):
            industry_mean = industry_group[source_column].transform("mean")
            df[neutral_column] = df[source_column] - industry_mean

        return df.loc[:, self._output_columns(l1_onehot_columns)].sort_values(
            ["trade_date", "ts_code"], kind="mergesort"
        ).reset_index(drop=True)

    def _write_daily_file(self, output_df: pd.DataFrame, trade_date: int) -> Path:
        trade_date_text = f"{int(trade_date):08d}"
        output_file = self.output_dir / f"year={trade_date_text[:4]}" / f"month={trade_date_text[4:6]}" / f"{trade_date_text}.parquet"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_parquet(output_file, index=False)
        factor_nan_counts = output_df.loc[:, list(self.DERIVED_FACTOR_COLUMNS)].isna().sum().to_dict()
        self.logger.info(
            "行业因子文件生成完成：trade_date=%s, rows=%d, factor_nan_counts=%s -> %s",
            trade_date_text,
            len(output_df),
            factor_nan_counts,
            output_file,
        )
        return output_file

    def _log_daily_source_coverage(
        self,
        trade_date: int,
        price_df: pd.DataFrame,
        fundamental_df: pd.DataFrame,
        output_df: pd.DataFrame,
    ) -> None:
        """Log source non-null coverage so all-NaN derived factors are explainable."""

        source_nonnull = {
            "ret_1": int(price_df["ret_1"].notna().sum()),
            "ret_5": int(price_df["ret_5"].notna().sum()),
            "ret_20": int(price_df["ret_20"].notna().sum()),
            "roe": int(fundamental_df["roe"].notna().sum()) if "roe" in fundamental_df else 0,
            "roa": int(fundamental_df["roa"].notna().sum()) if "roa" in fundamental_df else 0,
            "revenue_yoy": int(fundamental_df["revenue_yoy"].notna().sum()) if "revenue_yoy" in fundamental_df else 0,
            "L1_industry_name": int(output_df[self.INDUSTRY_LEVEL_COLUMN].notna().sum()),
        }
        self.logger.info("行业因子源数据覆盖：trade_date=%s, non_null_counts=%s", trade_date, source_nonnull)

        if source_nonnull["ret_5"] == 0:
            self.logger.warning("trade_date=%s 的上游 ret_5 全为空，industry_ret_5/relative_strength_5 将全部为空。", trade_date)
        if source_nonnull["ret_20"] == 0:
            self.logger.warning("trade_date=%s 的上游 ret_20 全为空，industry_ret_20/relative_strength_20 将全部为空。", trade_date)
        for source_column, neutral_column in (
            ("roe", "roe_ind_neutral"),
            ("roa", "roa_ind_neutral"),
            ("revenue_yoy", "revenue_yoy_ind_neutral"),
        ):
            if source_nonnull[source_column] == 0:
                self.logger.warning("trade_date=%s 的上游 %s 全为空，%s 将全部为空。", trade_date, source_column, neutral_column)

    def _matching_fundamental_file(self, price_file: Path) -> Path:
        return self.fundamental_dir / price_file.relative_to(self.price_volume_dir)

    def _output_columns(self, l1_onehot_columns: Sequence[str]) -> list[str]:
        return [*self.KEY_COLUMNS, self.INDUSTRY_LEVEL_COLUMN, *l1_onehot_columns, *self.DERIVED_FACTOR_COLUMNS]

    def _log_summary(self, summary: IndustryFeatureGenerateSummary) -> None:
        self.logger.info(
            "行业因子生成汇总：price_volume_files_read=%d, fundamental_files_read=%d, missing_fundamental_files=%d, "
            "files_written=%d, price_volume_rows_read=%d, fundamental_rows_read=%d, rows_written=%d, "
            "start_date=%s, end_date=%s, L1_onehot_columns=%d, factors=%d, output=%s",
            summary.price_volume_files_read,
            summary.fundamental_files_read,
            summary.missing_fundamental_files,
            summary.files_written,
            summary.price_volume_rows_read,
            summary.fundamental_rows_read,
            summary.rows_written,
            summary.start_date,
            summary.end_date,
            len(summary.l1_onehot_columns),
            len(summary.factor_columns),
            summary.output_dir,
        )
        if summary.missing_l1_rows:
            self.logger.warning("输出中有 %d 行缺失 L1_industry_name；行业收益/中性化因子为 NaN，L1 one-hot 全 0。", summary.missing_l1_rows)
        if summary.duplicate_price_volume_key_rows:
            self.logger.warning("价格量因子输入存在 %d 行重复 trade_date+ts_code，请关注上游生成结果。", summary.duplicate_price_volume_key_rows)
        if summary.duplicate_fundamental_key_rows:
            self.logger.warning("财务因子输入存在 %d 行重复 trade_date+ts_code，请关注上游生成结果。", summary.duplicate_fundamental_key_rows)
        if summary.missing_fundamental_files:
            self.logger.warning("共有 %d 个交易日缺少同日财务因子文件，相关行业中性化财务因子已置为 NaN。", summary.missing_fundamental_files)

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
    target_log_file = Path(log_file) if log_file is not None else IndustryFeatureGenerator.DEFAULT_LOG_FILE
    target_log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(target_log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate daily L1 industry factors from A-share feature factors.")
    parser.add_argument("--industry-file", default=str(IndustryFeatureGenerator.DEFAULT_INDUSTRY_FILE), help="Cleaned industry parquet file.")
    parser.add_argument(
        "--price-volume-dir",
        default=str(IndustryFeatureGenerator.DEFAULT_PRICE_VOLUME_DIR),
        help="Input directory for partitioned price-volume factors.",
    )
    parser.add_argument(
        "--fundamental-dir",
        default=str(IndustryFeatureGenerator.DEFAULT_FUNDAMENTAL_DIR),
        help="Input directory for partitioned fundamental factors.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(IndustryFeatureGenerator.DEFAULT_OUTPUT_DIR),
        help="Output directory for partitioned industry factors.",
    )
    parser.add_argument("--start-date", default=None, help="Optional inclusive output start date in YYYYMMDD format.")
    parser.add_argument("--end-date", default=None, help="Optional inclusive output end date in YYYYMMDD format.")
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    parser.add_argument("--log-file", default=str(IndustryFeatureGenerator.DEFAULT_LOG_FILE), help="Log file path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> IndustryFeatureGenerateSummary:
    args = parse_args(argv)
    configure_logging(log_level=args.log_level, log_file=args.log_file)
    generator = IndustryFeatureGenerator(
        industry_file=args.industry_file,
        price_volume_dir=args.price_volume_dir,
        fundamental_dir=args.fundamental_dir,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    return generator.process()


if __name__ == "__main__":
    # 使用方法：
    # PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
    # python -m utils.features_generator.industry_feature_generator \
    #   --output-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/features_data/industry_factors" \
    #   --log-file "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/features_generate/industry_features.log"
    main()


__all__ = [
    "IndustryFeatureGenerateSummary",
    "IndustryFeatureGenerator",
    "configure_logging",
    "main",
    "parse_args",
]
