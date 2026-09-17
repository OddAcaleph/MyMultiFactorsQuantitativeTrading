"""Build the daily wide table for daily bars from cleaned/processed A-share data.

This module treats all source parquet files as read-only and writes enriched
daily wide-table parquet files to a separate partitioned output directory.
The output preserves the ``year=YYYY/month=MM/*.parquet`` layout of the cleaned
``daily_bars`` main table.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class WideTableDailyBarsBuildSummary:
    """Aggregate statistics for daily wide-table construction."""

    daily_bars_dir: Path
    output_dir: Path
    files_processed: int
    rows_read: int
    rows_written: int
    namechange_matched_rows: int
    suspend_matched_rows: int
    adj_factor_matched_rows: int
    fundamentals_matched_rows: int
    moneyflow_matched_rows: int
    industry_matched_rows: int
    missing_adj_factor_rows: int
    missing_fundamentals_rows: int
    missing_moneyflow_rows: int
    duplicate_daily_key_rows: int


@dataclass(frozen=True)
class _FeatureBundle:
    """Preloaded feature tables.

    Daily keyed tables are indexed by ``trade_date`` + ``ts_code`` so each
    per-day file can use cheap reindex alignment instead of repeatedly merging
    against multi-million-row feature tables.
    """

    namechange_df: pd.DataFrame
    suspend_df: pd.DataFrame
    adj_factor_df: pd.DataFrame
    fundamentals_df: pd.DataFrame
    moneyflow_df: pd.DataFrame
    industry_df: pd.DataFrame
    industry_feature_columns: tuple[str, ...]
    fundamental_columns: tuple[str, ...]


class WideTableDailyBarsBuilder:
    """Merge all cleaned/processed feature tables into daily bars.

    Merge order follows the wide-table pipeline requirement:

    1. Use ``cleaned_data/daily_bars`` as the main table.
    2. Merge processed ``namechange`` ST features and processed ``suspend_d``
       features by ``trade_date`` + ``ts_code``.
    3. Merge cleaned ``adj_factors`` by ``trade_date`` + ``ts_code``.
    4. Merge fundamentals as point-in-time features: for each daily row, use the
       latest fundamental record whose ``ann_date`` is not later than
       ``trade_date``. Then merge cleaned MoneyFlow features by ``trade_date`` +
       ``ts_code`` before processed industry one-hot features by ``ts_code`` and
       the industry validity interval ``in_date <= trade_date < out_date``.

    Source data is never modified. Output files are written under
    ``data/processd_data/wide_table_daily_bars`` by default, preserving the daily-bars
    date partition layout.
    """

    DEFAULT_DAILY_BARS_DIR = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/daily_bars")
    DEFAULT_NAMECHANGE_FILE = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/namechange/namechange_st_daily.parquet"
    )
    DEFAULT_SUSPEND_FILE = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/suspend_d/suspend_d_daily.parquet"
    )
    DEFAULT_ADJ_FACTOR_FILE = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/adj_factors/adj_factors.parquet"
    )
    DEFAULT_FUNDAMENTALS_FILE = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/fundamentals/fundamentals.parquet"
    )
    DEFAULT_MONEYFLOW_FILE = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/moneyflow/moneyflow.parquet"
    )
    DEFAULT_INDUSTRY_FILE = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/industry/daily_onehot"
    )
    DEFAULT_OUTPUT_DIR = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/wide_table_daily_bars")
    DEFAULT_LOG_FILE = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/wide_table_daily_bars_building.log")

    KEY_COLUMNS: tuple[str, str] = ("trade_date", "ts_code")
    DAILY_REQUIRED_COLUMNS: tuple[str, ...] = (
        "ts_code",
        "trade_date",
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
    NAMECHANGE_FILL_VALUES: Mapping[str, int] = {"no_st_stock": 1, "st_stock": 0, "star_st_stock": 0}
    SUSPEND_FILL_VALUES: Mapping[str, int] = {"is_suspect": 0}

    def __init__(
        self,
        daily_bars_dir: str | Path | None = None,
        namechange_file: str | Path | None = None,
        suspend_file: str | Path | None = None,
        adj_factor_file: str | Path | None = None,
        fundamentals_file: str | Path | None = None,
        moneyflow_file: str | Path | None = None,
        industry_file: str | Path | None = None,
        output_dir: str | Path | None = None,
        start_date: str | int | None = None,
        end_date: str | int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.daily_bars_dir = Path(daily_bars_dir or self.DEFAULT_DAILY_BARS_DIR)
        self.namechange_file = Path(namechange_file or self.DEFAULT_NAMECHANGE_FILE)
        self.suspend_file = Path(suspend_file or self.DEFAULT_SUSPEND_FILE)
        self.adj_factor_file = Path(adj_factor_file or self.DEFAULT_ADJ_FACTOR_FILE)
        self.fundamentals_file = Path(fundamentals_file or self.DEFAULT_FUNDAMENTALS_FILE)
        self.moneyflow_file = Path(moneyflow_file or self.DEFAULT_MONEYFLOW_FILE)
        self.industry_path = Path(industry_file or self.DEFAULT_INDUSTRY_FILE)
        self.output_dir = Path(output_dir or self.DEFAULT_OUTPUT_DIR)
        self.start_date = self._parse_optional_date(start_date, "start_date")
        self.end_date = self._parse_optional_date(end_date, "end_date")
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self._industry_is_daily = self.industry_path.is_dir()
        self._validate_inputs()

    def process(self) -> WideTableDailyBarsBuildSummary:
        """Build and write the partitioned daily wide table."""

        parquet_files = self._list_daily_bar_files()
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found under {self.daily_bars_dir}/year=*/month=*/*.parquet "
                f"for date range [{self.start_date}, {self.end_date}]"
            )

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("开始构建 daily_bars 宽表：daily_bars=%s, output=%s", self.daily_bars_dir, self.output_dir)
        self.logger.info("待处理 daily_bars 文件数：%d；所有输入数据只读。", len(parquet_files))

        features = self._load_feature_bundle()

        rows_read = 0
        rows_written = 0
        duplicate_daily_key_rows = 0
        namechange_matched_rows = 0
        suspend_matched_rows = 0
        adj_factor_matched_rows = 0
        fundamentals_matched_rows = 0
        moneyflow_matched_rows = 0
        industry_matched_rows = 0
        missing_adj_factor_rows = 0
        missing_fundamentals_rows = 0
        missing_moneyflow_rows = 0

        for parquet_file in parquet_files:
            daily_df = pd.read_parquet(parquet_file)
            self._validate_columns(daily_df.columns, self.DAILY_REQUIRED_COLUMNS, parquet_file)
            daily_df = self._prepare_daily_bars(daily_df, parquet_file)

            file_rows = len(daily_df)
            rows_read += file_rows
            file_duplicate_rows = int(daily_df.duplicated(subset=list(self.KEY_COLUMNS), keep=False).sum())
            duplicate_daily_key_rows += file_duplicate_rows
            if file_duplicate_rows:
                self.logger.warning("%s 存在 %d 行重复 trade_date+ts_code 主键。", parquet_file, file_duplicate_rows)

            enriched_df, file_namechange_matched = self._merge_daily_keyed_feature(
                daily_df,
                features.namechange_df,
                "namechange",
                fill_values=self.NAMECHANGE_FILL_VALUES,
            )
            namechange_matched_rows += file_namechange_matched

            enriched_df, file_suspend_matched = self._merge_daily_keyed_feature(
                enriched_df,
                features.suspend_df,
                "suspend_d",
                fill_values=self.SUSPEND_FILL_VALUES,
            )
            suspend_matched_rows += file_suspend_matched

            enriched_df, file_adj_factor_matched = self._merge_daily_keyed_feature(enriched_df, features.adj_factor_df, "adj_factors")
            adj_factor_matched_rows += file_adj_factor_matched
            missing_adj_factor_rows += int(enriched_df["adj_factor"].isna().sum()) if "adj_factor" in enriched_df.columns else file_rows

            enriched_df, file_fundamentals_matched = self._merge_fundamentals(enriched_df, features.fundamentals_df)
            fundamentals_matched_rows += file_fundamentals_matched
            if "ann_date" in enriched_df.columns:
                missing_fundamentals_rows += int(enriched_df["ann_date"].isna().sum())
            else:
                missing_fundamentals_rows += file_rows

            enriched_df, file_moneyflow_matched = self._merge_daily_keyed_feature(enriched_df, features.moneyflow_df, "moneyflow")
            moneyflow_matched_rows += file_moneyflow_matched
            missing_moneyflow_rows += file_rows - file_moneyflow_matched

            enriched_df, file_industry_matched = self._merge_industry(
                enriched_df,
                features.industry_df,
                features.industry_feature_columns,
            )
            industry_matched_rows += file_industry_matched

            output_file = self._output_path_for(parquet_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            enriched_df.to_parquet(output_file, index=False)
            rows_written += len(enriched_df)

            self.logger.info(
                "daily_bars 宽表文件生成完成：%s | rows=%d, namechange_match=%d, suspend_match=%d, "
                "adj_match=%d, fundamentals_match=%d, moneyflow_match=%d, industry_match=%d -> %s",
                parquet_file,
                len(enriched_df),
                file_namechange_matched,
                file_suspend_matched,
                file_adj_factor_matched,
                file_fundamentals_matched,
                file_moneyflow_matched,
                file_industry_matched,
                output_file,
            )

        summary = WideTableDailyBarsBuildSummary(
            daily_bars_dir=self.daily_bars_dir,
            output_dir=self.output_dir,
            files_processed=len(parquet_files),
            rows_read=rows_read,
            rows_written=rows_written,
            namechange_matched_rows=namechange_matched_rows,
            suspend_matched_rows=suspend_matched_rows,
            adj_factor_matched_rows=adj_factor_matched_rows,
            fundamentals_matched_rows=fundamentals_matched_rows,
            moneyflow_matched_rows=moneyflow_matched_rows,
            industry_matched_rows=industry_matched_rows,
            missing_adj_factor_rows=missing_adj_factor_rows,
            missing_fundamentals_rows=missing_fundamentals_rows,
            missing_moneyflow_rows=missing_moneyflow_rows,
            duplicate_daily_key_rows=duplicate_daily_key_rows,
        )
        self._log_summary(summary)
        return summary

    def _validate_inputs(self) -> None:
        if not self.daily_bars_dir.exists() or not self.daily_bars_dir.is_dir():
            raise NotADirectoryError(f"Daily bars input directory does not exist or is not a directory: {self.daily_bars_dir}")
        for file_path in (
            self.namechange_file,
            self.suspend_file,
            self.adj_factor_file,
            self.fundamentals_file,
            self.moneyflow_file,
        ):
            if not file_path.exists() or not file_path.is_file():
                raise FileNotFoundError(f"Required input parquet file does not exist: {file_path}")
        if not self.industry_path.exists():
            raise FileNotFoundError(f"Industry input does not exist: {self.industry_path}")
        if self.start_date is not None and self.end_date is not None and self.start_date > self.end_date:
            raise ValueError(f"start_date must be <= end_date, got {self.start_date} > {self.end_date}")

    def _list_daily_bar_files(self) -> list[Path]:
        files = sorted(self.daily_bars_dir.glob("year=*/month=*/*.parquet"))
        return [file_path for file_path in files if self._is_file_date_in_range(file_path)]

    def _is_file_date_in_range(self, file_path: Path) -> bool:
        try:
            file_date = int(file_path.stem)
        except ValueError:
            self.logger.warning("跳过无法从文件名解析交易日的 daily_bars 文件：%s", file_path)
            return False
        if self.start_date is not None and file_date < self.start_date:
            return False
        if self.end_date is not None and file_date > self.end_date:
            return False
        return True

    def _load_feature_bundle(self) -> _FeatureBundle:
        self.logger.info("加载 processed namechange 特征：%s", self.namechange_file)
        namechange_df = self._load_namechange_feature()

        self.logger.info("加载 processed suspend_d 特征：%s", self.suspend_file)
        suspend_df = self._load_daily_keyed_feature(
            self.suspend_file,
            required_columns=("trade_date", "ts_code", "is_suspect"),
        )

        self.logger.info("加载 cleaned adj_factors 特征：%s", self.adj_factor_file)
        adj_factor_df = self._load_daily_keyed_feature(
            self.adj_factor_file,
            required_columns=("trade_date", "ts_code", "adj_factor"),
        )

        self.logger.info("加载 cleaned fundamentals 特征：%s", self.fundamentals_file)
        fundamentals_df = self._load_fundamentals()
        fundamental_columns = tuple(column for column in fundamentals_df.columns if column not in {"ts_code", "_ann_dt"})

        self.logger.info("加载 cleaned MoneyFlow 特征：%s", self.moneyflow_file)
        moneyflow_df = self._load_moneyflow_feature()

        if self._industry_is_daily:
            self.logger.info("行业数据为日度分区目录，将按天加载：%s", self.industry_path)
            industry_df = pd.DataFrame()
            industry_feature_columns = self._discover_daily_industry_columns()
        else:
            self.logger.info("加载 processed industry one-hot 特征：%s", self.industry_path)
            industry_df = pd.read_parquet(self.industry_path)
            self._validate_columns(industry_df.columns, ("ts_code", "in_date", "out_date"), self.industry_path)
            industry_df = industry_df.copy()
            industry_df["ts_code"] = industry_df["ts_code"].astype("string").str.strip()
            industry_df = industry_df.loc[~self._is_null_or_empty(industry_df["ts_code"])].copy()
            industry_df["in_date"] = self._normalize_yyyymmdd(industry_df["in_date"], "in_date", self.industry_path)
            industry_df["out_date"] = self._normalize_optional_yyyymmdd(industry_df["out_date"], "out_date", self.industry_path)
            industry_df["_in_dt"] = pd.to_datetime(industry_df["in_date"].astype("string"), format="%Y%m%d", errors="coerce")
            industry_df["_out_dt"] = pd.to_datetime(industry_df["out_date"].astype("string"), format="%Y%m%d", errors="coerce")

            invalid_industry_rows = int(industry_df["_in_dt"].isna().sum())
            if invalid_industry_rows:
                self.logger.warning("industry one-hot 存在 %d 行无效 in_date，已剔除。", invalid_industry_rows)
                industry_df = industry_df.loc[industry_df["_in_dt"].notna()].copy()

            duplicate_rows = int(industry_df.duplicated(subset=["ts_code", "in_date", "out_date"], keep=False).sum())
            if duplicate_rows:
                self.logger.warning("industry one-hot 存在 %d 行重复 ts_code+in_date+out_date，保留最后一条。", duplicate_rows)
                industry_df = industry_df.drop_duplicates(subset=["ts_code", "in_date", "out_date"], keep="last")

            industry_feature_columns = tuple(column for column in industry_df.columns if column not in {"ts_code", "in_date", "out_date", "_in_dt", "_out_dt"})
            industry_df = industry_df.sort_values(["_in_dt", "ts_code", "out_date"], kind="mergesort").reset_index(drop=True)

        self.logger.info(
            "特征表加载完成：namechange_rows=%d, suspend_rows=%d, adj_rows=%d, fundamentals_rows=%d, "
            "moneyflow_rows=%d, industry_rows=%d, industry_feature_columns=%d",
            len(namechange_df),
            len(suspend_df),
            len(adj_factor_df),
            len(fundamentals_df),
            len(moneyflow_df),
            len(industry_df),
            len(industry_feature_columns),
        )

        return _FeatureBundle(
            namechange_df=namechange_df,
            suspend_df=suspend_df,
            adj_factor_df=adj_factor_df,
            fundamentals_df=fundamentals_df,
            moneyflow_df=moneyflow_df,
            industry_df=industry_df,
            industry_feature_columns=industry_feature_columns,
            fundamental_columns=fundamental_columns,
        )

    def _discover_daily_industry_columns(self) -> tuple[str, ...]:
        """Discover industry one-hot column names from the first available daily file."""
        def _walk_dirs(base_dir: Path) -> Iterable[Path]:
            year_dirs = sorted(base_dir.glob("year=*"))
            if year_dirs:
                for yd in year_dirs:
                    for md in sorted(yd.glob("month=*")):
                        for pf in sorted(md.glob("*.parquet")):
                            yield pf
            else:
                for md in sorted(base_dir.glob("month=*")):
                    for pf in sorted(md.glob("*.parquet")):
                        yield pf

        for parquet_file in _walk_dirs(self.industry_path):
            df = pd.read_parquet(parquet_file)
            cols = tuple(
                c for c in df.columns
                if c not in ("trade_date", "ts_code") and c.startswith("L1_")
            )
            if cols:
                self.logger.info("日度行业 one-hot 列数：%d", len(cols))
                return cols
        return ()

    def _load_daily_industry_for_date(self, trade_date: int) -> pd.DataFrame:
        """Load daily industry one-hot for a specific trade date."""
        date_str = str(trade_date)
        year = date_str[:4]
        month = date_str[4:6]
        # Try year=YYYY/month=MM/YYYYMMDD.parquet under industry_path
        file_path = self.industry_path / f"year={year}" / f"month={month}" / f"{date_str}.parquet"
        if not file_path.exists():
            # Fallback: industry_path is already a year directory (month=MM/YYYYMMDD.parquet)
            file_path = self.industry_path / f"month={month}" / f"{date_str}.parquet"
        if not file_path.exists():
            return pd.DataFrame(columns=["trade_date", "ts_code"])
        df = pd.read_parquet(file_path)
        df["ts_code"] = df["ts_code"].astype("string").str.strip()
        return df.set_index("ts_code", drop=True)

    def _load_moneyflow_feature(self) -> pd.DataFrame:
        """Load cleaned MoneyFlow features keyed by ``trade_date`` + ``ts_code``."""

        moneyflow_columns = (
            "trade_date",
            "ts_code",
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
        return self._load_daily_keyed_feature(self.moneyflow_file, required_columns=moneyflow_columns)

    def _load_namechange_feature(self) -> pd.DataFrame:
        """Load namechange ST features with the ``no_st_stock`` column."""

        df = pd.read_parquet(self.namechange_file)
        required_columns = ("trade_date", "ts_code", "no_st_stock", "st_stock", "star_st_stock")
        self._validate_columns(df.columns, required_columns, self.namechange_file)
        return self._prepare_daily_keyed_feature(df.loc[:, list(required_columns)].copy(), self.namechange_file)

    def _load_daily_keyed_feature(self, file_path: Path, required_columns: Sequence[str]) -> pd.DataFrame:
        df = pd.read_parquet(file_path)
        self._validate_columns(df.columns, required_columns, file_path)
        return self._prepare_daily_keyed_feature(df.loc[:, list(required_columns)].copy(), file_path)

    def _prepare_daily_keyed_feature(self, feature_df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
        feature_df["trade_date"] = self._normalize_yyyymmdd(feature_df["trade_date"], "trade_date", file_path)
        feature_df["ts_code"] = feature_df["ts_code"].astype("string").str.strip()

        invalid_key_mask = feature_df["trade_date"].isna() | self._is_null_or_empty(feature_df["ts_code"])
        invalid_key_rows = int(invalid_key_mask.sum())
        if invalid_key_rows:
            self.logger.warning("%s 存在 %d 行 trade_date/ts_code 无效，已从特征表中剔除。", file_path, invalid_key_rows)
            feature_df = feature_df.loc[~invalid_key_mask].copy()

        duplicate_rows = int(feature_df.duplicated(subset=list(self.KEY_COLUMNS), keep=False).sum())
        if duplicate_rows:
            self.logger.warning("%s 存在 %d 行重复 trade_date+ts_code，保留最后一条用于 merge。", file_path, duplicate_rows)
            feature_df = feature_df.drop_duplicates(subset=list(self.KEY_COLUMNS), keep="last")

        return feature_df.set_index(list(self.KEY_COLUMNS), drop=True).sort_index()

    def _load_fundamentals(self) -> pd.DataFrame:
        required_columns = (
            "ts_code",
            "end_date",
            "ann_date",
            "roe",
            "roa",
            "or_yoy",
            "debt_to_assets",
            "gross_margin",
            "eps",
            "bps",
        )
        raw_df = pd.read_parquet(self.fundamentals_file)
        self._validate_columns(raw_df.columns, required_columns, self.fundamentals_file)
        fundamentals = raw_df.loc[:, list(required_columns)].copy()
        fundamentals["ts_code"] = fundamentals["ts_code"].astype("string").str.strip()
        fundamentals["ann_date"] = self._normalize_yyyymmdd(fundamentals["ann_date"], "ann_date", self.fundamentals_file)
        fundamentals["end_date"] = self._normalize_yyyymmdd(fundamentals["end_date"], "end_date", self.fundamentals_file)
        fundamentals["_ann_dt"] = pd.to_datetime(fundamentals["ann_date"].astype("string"), format="%Y%m%d", errors="coerce")

        invalid_key_mask = fundamentals["_ann_dt"].isna() | self._is_null_or_empty(fundamentals["ts_code"])
        invalid_key_rows = int(invalid_key_mask.sum())
        if invalid_key_rows:
            self.logger.warning("fundamentals 存在 %d 行 ts_code/ann_date 无效，已剔除。", invalid_key_rows)
            fundamentals = fundamentals.loc[~invalid_key_mask].copy()

        duplicate_rows = int(fundamentals.duplicated(subset=["ts_code", "ann_date", "end_date"], keep=False).sum())
        if duplicate_rows:
            self.logger.warning("fundamentals 存在 %d 行重复 ts_code+ann_date+end_date，保留最后一条。", duplicate_rows)
            fundamentals = fundamentals.drop_duplicates(subset=["ts_code", "ann_date", "end_date"], keep="last")

        return fundamentals.sort_values(["_ann_dt", "ts_code", "end_date"], kind="mergesort").reset_index(drop=True)

    def _prepare_daily_bars(self, daily_df: pd.DataFrame, parquet_file: Path) -> pd.DataFrame:
        prepared = daily_df.copy()
        prepared["trade_date"] = self._normalize_yyyymmdd(prepared["trade_date"], "trade_date", parquet_file)
        prepared["ts_code"] = prepared["ts_code"].astype("string").str.strip()

        invalid_key_mask = prepared["trade_date"].isna() | self._is_null_or_empty(prepared["ts_code"])
        invalid_key_rows = int(invalid_key_mask.sum())
        if invalid_key_rows:
            raise ValueError(f"{parquet_file} has {invalid_key_rows} invalid trade_date/ts_code rows in the main table.")
        return prepared

    def _merge_daily_keyed_feature(
        self,
        daily_df: pd.DataFrame,
        feature_df: pd.DataFrame,
        feature_name: str,
        fill_values: Mapping[str, int | float] | None = None,
    ) -> tuple[pd.DataFrame, int]:
        unique_trade_dates = daily_df["trade_date"].drop_duplicates()
        if len(unique_trade_dates) == 1:
            trade_date = int(unique_trade_dates.iloc[0])
            try:
                date_features = feature_df.xs(trade_date, level="trade_date", drop_level=True)
            except KeyError:
                date_features = pd.DataFrame(columns=feature_df.columns, index=pd.Index([], name="ts_code"))
            aligned_features = date_features.reindex(daily_df["ts_code"]).reset_index(drop=True)
        else:
            key_index = pd.MultiIndex.from_frame(daily_df.loc[:, list(self.KEY_COLUMNS)])
            aligned_features = feature_df.reindex(key_index).reset_index(drop=True)
        matched_rows = int(aligned_features.notna().any(axis=1).sum()) if not aligned_features.empty else 0
        merged = pd.concat([daily_df.reset_index(drop=True), aligned_features], axis=1)

        if fill_values:
            for column, value in fill_values.items():
                if column in merged.columns:
                    merged[column] = pd.to_numeric(merged[column], errors="coerce").fillna(value).astype("int8")

        return merged, matched_rows

    def _merge_fundamentals(self, daily_df: pd.DataFrame, fundamentals_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        if fundamentals_df.empty:
            self.logger.warning("fundamentals 特征表为空，跳过 fundamentals merge。")
            daily_df.attrs["fundamentals_matched_rows"] = 0
            return daily_df, 0

        unique_trade_dates = daily_df["trade_date"].drop_duplicates()
        if len(unique_trade_dates) == 1:
            trade_date = int(unique_trade_dates.iloc[0])
            latest_fundamentals = (
                fundamentals_df.loc[fundamentals_df["ann_date"] <= trade_date]
                .drop(columns=["_ann_dt"])
                .drop_duplicates(subset=["ts_code"], keep="last")
                .set_index("ts_code", drop=True)
            )
            aligned_features = latest_fundamentals.reindex(daily_df["ts_code"]).reset_index(drop=True)
            matched_rows = int(aligned_features["ann_date"].notna().sum()) if "ann_date" in aligned_features.columns else 0
            merged = pd.concat([daily_df.reset_index(drop=True), aligned_features], axis=1)
            return merged, matched_rows

        left = daily_df.reset_index(names="_left_row").copy()
        left["_trade_dt"] = pd.to_datetime(left["trade_date"].astype("string"), format="%Y%m%d", errors="coerce")
        left_sorted = left.sort_values(["_trade_dt", "ts_code", "_left_row"], kind="mergesort")

        merged = pd.merge_asof(
            left_sorted,
            fundamentals_df,
            left_on="_trade_dt",
            right_on="_ann_dt",
            by="ts_code",
            direction="backward",
            allow_exact_matches=True,
        )
        matched_rows = int(merged["ann_date"].notna().sum()) if "ann_date" in merged.columns else 0
        merged = merged.sort_values("_left_row", kind="mergesort").drop(columns=["_left_row", "_trade_dt", "_ann_dt"])
        merged = merged.reset_index(drop=True)
        merged.attrs["fundamentals_matched_rows"] = matched_rows
        return merged, matched_rows

    def _merge_industry(
        self,
        daily_df: pd.DataFrame,
        industry_df: pd.DataFrame,
        industry_feature_columns: Sequence[str],
    ) -> tuple[pd.DataFrame, int]:
        if self._industry_is_daily:
            return self._merge_daily_industry(daily_df, industry_feature_columns)

        industry_output_columns = ["in_date", "out_date", *industry_feature_columns]
        if industry_df.empty:
            aligned_features = pd.DataFrame(index=daily_df.index, columns=industry_output_columns)
            matched_rows = 0
            merged = pd.concat([daily_df.reset_index(drop=True), aligned_features.reset_index(drop=True)], axis=1)
            for column in industry_feature_columns:
                if column in merged.columns:
                    merged[column] = merged[column].fillna(0).astype("int8")
            return merged, matched_rows

        unique_trade_dates = daily_df["trade_date"].drop_duplicates()
        if len(unique_trade_dates) == 1:
            trade_date = int(unique_trade_dates.iloc[0])
            active_mask = (industry_df["in_date"] <= trade_date) & (industry_df["out_date"].isna() | (industry_df["out_date"] > trade_date))
            active_industry = (
                industry_df.loc[active_mask]
                .drop_duplicates(subset=["ts_code"], keep="last")
                .set_index("ts_code", drop=True)
            )
            aligned_features = active_industry.reindex(daily_df["ts_code"]).reset_index(drop=True)
            aligned_features = aligned_features.loc[:, [column for column in industry_output_columns if column in aligned_features.columns]]
        else:
            left = daily_df.reset_index(names="_left_row").copy()
            left["_trade_dt"] = pd.to_datetime(left["trade_date"].astype("string"), format="%Y%m%d", errors="coerce")
            left_sorted = left.sort_values(["_trade_dt", "ts_code", "_left_row"], kind="mergesort")
            industry_sorted = industry_df.sort_values(["_in_dt", "ts_code", "out_date"], kind="mergesort")
            aligned = pd.merge_asof(
                left_sorted,
                industry_sorted,
                left_on="_trade_dt",
                right_on="_in_dt",
                by="ts_code",
                direction="backward",
                allow_exact_matches=True,
            )
            active_mask = aligned["in_date"].notna() & (aligned["out_date"].isna() | (aligned["out_date"] > aligned["trade_date"]))
            for column in industry_output_columns:
                if column in aligned.columns:
                    aligned.loc[~active_mask, column] = pd.NA
            aligned_features = aligned.sort_values("_left_row", kind="mergesort").loc[:, industry_output_columns].reset_index(drop=True)

        for column in industry_output_columns:
            if column not in aligned_features.columns:
                aligned_features[column] = pd.NA
        aligned_features = aligned_features.loc[:, industry_output_columns]
        matched_rows = int(aligned_features["in_date"].notna().sum()) if "in_date" in aligned_features.columns else 0
        merged = pd.concat([daily_df.reset_index(drop=True), aligned_features.reset_index(drop=True)], axis=1)
        for column in industry_feature_columns:
            if column in merged.columns:
                merged[column] = merged[column].fillna(0).astype("int8")
        return merged, matched_rows

    def _merge_daily_industry(
        self,
        daily_df: pd.DataFrame,
        industry_feature_columns: Sequence[str],
    ) -> tuple[pd.DataFrame, int]:
        unique_trade_dates = daily_df["trade_date"].drop_duplicates()
        if len(unique_trade_dates) == 1:
            trade_date = int(unique_trade_dates.iloc[0])
            day_industry = self._load_daily_industry_for_date(trade_date)
            ind_cols = [c for c in industry_feature_columns if c in day_industry.columns]
            if ind_cols:
                aligned = day_industry.reindex(daily_df["ts_code"])
                aligned_features = aligned[ind_cols].reset_index(drop=True)
            else:
                aligned_features = pd.DataFrame(index=range(len(daily_df)), columns=list(industry_feature_columns))
        else:
            all_dfs = []
            for td in unique_trade_dates:
                td_int = int(td)
                day_industry = self._load_daily_industry_for_date(td_int)
                day_df = daily_df[daily_df["trade_date"] == td_int].copy()
                ind_cols = [c for c in industry_feature_columns if c in day_industry.columns]
                if ind_cols:
                    aligned = day_industry.reindex(day_df["ts_code"])[ind_cols].reset_index(drop=True)
                else:
                    aligned = pd.DataFrame(index=range(len(day_df)), columns=list(industry_feature_columns))
                aligned["_row_idx"] = day_df.index
                all_dfs.append(aligned)
            if all_dfs:
                combined = pd.concat(all_dfs, ignore_index=True)
                combined = combined.set_index("_row_idx").sort_index()
                aligned_features = combined.reset_index(drop=True)
            else:
                aligned_features = pd.DataFrame(index=range(len(daily_df)), columns=list(industry_feature_columns))

        for col in industry_feature_columns:
            if col not in aligned_features.columns:
                aligned_features[col] = 0
            aligned_features[col] = pd.to_numeric(aligned_features[col], errors="coerce").fillna(0).astype("int8")

        matched_rows = int((aligned_features.sum(axis=1) > 0).sum())
        merged = pd.concat([daily_df.reset_index(drop=True), aligned_features], axis=1)
        return merged, matched_rows

    def _output_path_for(self, input_file: Path) -> Path:
        return self.output_dir / input_file.relative_to(self.daily_bars_dir)

    def _log_summary(self, summary: WideTableDailyBarsBuildSummary) -> None:
        self.logger.info(
            "daily_bars 宽表构建汇总：files=%d, rows_read=%d, rows_written=%d, "
            "namechange_match=%d, suspend_match=%d, adj_factor_match=%d, fundamentals_match=%d, "
            "moneyflow_match=%d, industry_match=%d, missing_adj_factor_rows=%d, missing_fundamentals_rows=%d, "
            "missing_moneyflow_rows=%d, duplicate_daily_key_rows=%d, output=%s",
            summary.files_processed,
            summary.rows_read,
            summary.rows_written,
            summary.namechange_matched_rows,
            summary.suspend_matched_rows,
            summary.adj_factor_matched_rows,
            summary.fundamentals_matched_rows,
            summary.moneyflow_matched_rows,
            summary.industry_matched_rows,
            summary.missing_adj_factor_rows,
            summary.missing_fundamentals_rows,
            summary.missing_moneyflow_rows,
            summary.duplicate_daily_key_rows,
            summary.output_dir,
        )
        if summary.duplicate_daily_key_rows:
            self.logger.warning("主表 daily_bars 中发现 %d 行重复 trade_date+ts_code。", summary.duplicate_daily_key_rows)
        else:
            self.logger.info("主表 daily_bars 唯一性检查通过：未发现重复 trade_date+ts_code。")
        if summary.missing_adj_factor_rows:
            self.logger.warning("adj_factor merge 后仍有 %d 行缺失 adj_factor。", summary.missing_adj_factor_rows)
        else:
            self.logger.info("adj_factor merge 完整：未发现缺失 adj_factor。")
        if summary.missing_fundamentals_rows:
            self.logger.warning("fundamentals point-in-time merge 后仍有 %d 行没有可用已披露财务数据。", summary.missing_fundamentals_rows)
        else:
            self.logger.info("fundamentals point-in-time merge 完整：所有行均匹配到已披露财务数据。")
        if summary.missing_moneyflow_rows:
            self.logger.warning("MoneyFlow merge 后仍有 %d 行没有可用资金流数据。", summary.missing_moneyflow_rows)
        else:
            self.logger.info("MoneyFlow merge 完整：所有行均匹配到资金流数据。")

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

    @classmethod
    def _normalize_optional_yyyymmdd(cls, series: pd.Series, column_name: str, file_path: Path) -> pd.Series:
        normalized = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
        empty_mask = series.isna() | normalized.isin(["", "<NA>", "nan", "NaT", "None"])
        parsed = pd.to_datetime(normalized.mask(empty_mask, pd.NA), format="%Y%m%d", errors="coerce")
        invalid_rows = int((~empty_mask & parsed.isna()).sum())
        if invalid_rows:
            raise ValueError(f"{file_path} has {invalid_rows} invalid {column_name} rows that cannot be parsed as YYYYMMDD.")
        result = pd.Series(pd.NA, index=series.index, dtype="Int32")
        result.loc[~empty_mask] = parsed.loc[~empty_mask].dt.strftime("%Y%m%d").astype("int32")
        return result

    @staticmethod
    def _is_null_or_empty(series: pd.Series) -> pd.Series:
        return series.isna() | series.astype("string").str.strip().eq("").fillna(False)

    @staticmethod
    def _parse_optional_date(value: str | int | None, name: str) -> int | None:
        if value is None or value == "":
            return None
        text = str(value).strip()
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
        if pd.isna(parsed):
            raise ValueError(f"{name} must be in YYYYMMDD format, got {value!r}")
        return int(parsed.strftime("%Y%m%d"))


def configure_logging(log_level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure console and file logging for CLI usage."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    target_log_file = Path(log_file) if log_file is not None else WideTableDailyBarsBuilder.DEFAULT_LOG_FILE
    target_log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(target_log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily wide table for daily bars from cleaned/processed A-share data.")
    parser.add_argument("--daily-bars-dir", default=str(WideTableDailyBarsBuilder.DEFAULT_DAILY_BARS_DIR), help="Cleaned daily_bars directory.")
    parser.add_argument("--namechange-file", default=str(WideTableDailyBarsBuilder.DEFAULT_NAMECHANGE_FILE), help="Processed namechange ST daily parquet file.")
    parser.add_argument("--suspend-file", default=str(WideTableDailyBarsBuilder.DEFAULT_SUSPEND_FILE), help="Processed suspend_d daily parquet file.")
    parser.add_argument("--adj-factor-file", default=str(WideTableDailyBarsBuilder.DEFAULT_ADJ_FACTOR_FILE), help="Cleaned adj_factors parquet file.")
    parser.add_argument("--fundamentals-file", default=str(WideTableDailyBarsBuilder.DEFAULT_FUNDAMENTALS_FILE), help="Cleaned fundamentals parquet file.")
    parser.add_argument("--moneyflow-file", default=str(WideTableDailyBarsBuilder.DEFAULT_MONEYFLOW_FILE), help="Cleaned MoneyFlow parquet file.")
    parser.add_argument("--industry-file", default=str(WideTableDailyBarsBuilder.DEFAULT_INDUSTRY_FILE), help="Processed industry one-hot parquet file.")
    parser.add_argument("--output-dir", default=str(WideTableDailyBarsBuilder.DEFAULT_OUTPUT_DIR), help="Output directory for partitioned daily-bars wide table.")
    parser.add_argument("--start-date", default=None, help="Optional inclusive start date in YYYYMMDD format.")
    parser.add_argument("--end-date", default=None, help="Optional inclusive end date in YYYYMMDD format.")
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    parser.add_argument("--log-file", default=str(WideTableDailyBarsBuilder.DEFAULT_LOG_FILE), help="Log file path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> WideTableDailyBarsBuildSummary:
    args = parse_args(argv)
    configure_logging(log_level=args.log_level, log_file=args.log_file)
    builder = WideTableDailyBarsBuilder(
        daily_bars_dir=args.daily_bars_dir,
        namechange_file=args.namechange_file,
        suspend_file=args.suspend_file,
        adj_factor_file=args.adj_factor_file,
        fundamentals_file=args.fundamentals_file,
        moneyflow_file=args.moneyflow_file,
        industry_file=args.industry_file,
        output_dir=args.output_dir,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    return builder.process()


if __name__ == "__main__":
    # 使用方法：
    # PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
    # python -m utils.dataset_processor.wide_table_daily_bars_builder \
    #   --output-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/wide_table_daily_bars" \
    #   --log-file "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/wide_table_daily_bars_building.log"
    main()


__all__ = ["WideTableDailyBarsBuildSummary", "WideTableDailyBarsBuilder", "configure_logging", "main", "parse_args"]
