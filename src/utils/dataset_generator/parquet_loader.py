"""Qlib DataLoader for local A-share cross-sectional parquet feature tables.

The loader keeps the original parquet data in place and exposes it in the
DataFrame shape expected by Qlib's ``DataHandlerLP`` / ``DatasetH``:

    index:   MultiIndex[datetime, instrument]
    columns: MultiIndex[feature/<feature_name>, label/<label_name>]

Features are joined by ``trade_date`` and ``ts_code`` from these processed
datasets:

* ``wide_table_daily_bars`` as the main feature table
* ``price_volume_factors``
* ``moneyflow_factors``
* ``fundamental_factors``
* ``industry_factors``
* ``generated_label/daily_labels`` as the label table
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from qlib.data.dataset.loader import DataLoader

from ..config import PROJECT_ROOT, load_loader_config


class ParquetLoader(DataLoader):
    """Load cross-sectional parquet features into Qlib Dataset-compatible format.

    Parameters
    ----------
    daily_bars_dir
        Directory of the main wide-table daily bars parquet files, e.g.
        ``data/cross_sectional_processd_data/wide_table_daily_bars``.
    feature_cols
        Feature columns to expose under the top-level ``feature`` column group.
        Columns can be raw columns from the configured parquet datasets or
        built-in derived columns.
    labels_dir
        Directory of generated daily label parquet files, e.g.
        ``data/generated_label/daily_labels``.
    label_name
        Label column to read from ``labels_dir`` under the top-level ``label``
        column group. Defaults to ``label_5d``.
    label_cols
        Label columns to read from ``labels_dir``. This can be used instead of
        ``label_name`` when multiple labels are needed.
    include_label
        Whether to add label columns to the returned DataFrame.
    dropna_label
        Whether to drop rows whose label is missing. Usually ``True`` for
        training. Set to ``False`` if you need inference rows at the tail.
    keep_original_code
        ``True`` keeps instruments as ``000001.SZ``. ``False`` converts to
        Qlib-like lower-case symbols such as ``sz000001``.
    config_path
        Optional loader JSON config path. If omitted, use the project default
        config under ``conf/parquet_loader_config.json``. Explicit constructor
        arguments always override config values.
    """

    DATA_ROOT = PROJECT_ROOT / "data" / "cross_sectional_processd_data"
    DEFAULT_LABELS_DIR = PROJECT_ROOT / "data" / "generated_label" / "daily_labels"

    MAIN_FEATURE_COLS: tuple[str, ...] = (
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "change",
        "pct_chg",
        "vol",
        "amount",
        "no_st_stock",
        "st_stock",
        "star_st_stock",
        "is_suspect",
        "adj_factor",
        "roe_cc_processed",
        "roa_cc_processed",
        "revenue_yoy_cc_processed",
        "debt_ratio_cc_processed",
        "gross_margin_cc_processed",
        "eps_cc_processed",
        "bps_cc_processed",
    )

    PRICE_VOLUME_FEATURE_COLS: tuple[str, ...] = (
        "ret_5_cc_processed",
        "ret_10_cc_processed",
        "ret_20_cc_processed",
        "ret_60_cc_processed",
        "ma5_bias_cc_processed",
        "ma10_bias_cc_processed",
        "ma20_bias_cc_processed",
        "ma60_bias_cc_processed",
        "volatility_5_cc_processed",
        "volatility_20_cc_processed",
        "volatility_60_cc_processed",
        "amplitude_cc_processed",
        "amplitude_5_cc_processed",
        "amplitude_20_cc_processed",
        "vol_ratio_5_cc_processed",
        "vol_ratio_20_cc_processed",
        "corr_price_vol_5_cc_processed",
        "corr_price_vol_20_cc_processed",
    )

    MONEYFLOW_FEATURE_COLS: tuple[str, ...] = (
        "lg_net_inflow_cc_processed",
        "elg_net_inflow_cc_processed",
        "main_net_inflow_cc_processed",
        "main_net_ratio_cc_processed",
        "retail_net_inflow_cc_processed",
        "retail_ratio_cc_processed",
        "main_retail_diff_cc_processed",
        "main_net_5_cc_processed",
        "main_net_10_cc_processed",
        "main_net_20_cc_processed",
        "mf_ma5_cc_processed",
        "mf_ma20_cc_processed",
        "mf_acceleration_cc_processed",
    )

    FUNDAMENTAL_FEATURE_COLS: tuple[str, ...] = (
        "roe_roa_gap_cc_processed",
    )

    INDUSTRY_FEATURE_COLS: tuple[str, ...] = (
        "industry_ret_1_cc_processed",
        "industry_ret_5_cc_processed",
        "industry_ret_20_cc_processed",
        "relative_strength_5_cc_processed",
        "relative_strength_20_cc_processed",
        "roe_ind_neutral_cc_processed",
        "roa_ind_neutral_cc_processed",
        "revenue_yoy_ind_neutral_cc_processed",
    )

    ENHANCED_ALPHA_FEATURE_COLS: tuple[str, ...] = (
        "macd_cc_processed",
        "rsi_14_cc_processed",
        "momentum_52w_high_dist_cc_processed",
        "gross_margin_change_cc_processed",
        "revenue_yoy_acceleration_cc_processed",
    )

    DEFAULT_FEATURE_COLS: tuple[str, ...] = (
        *MAIN_FEATURE_COLS,
        *PRICE_VOLUME_FEATURE_COLS,
        *MONEYFLOW_FEATURE_COLS,
        *FUNDAMENTAL_FEATURE_COLS,
        *INDUSTRY_FEATURE_COLS,
        *ENHANCED_ALPHA_FEATURE_COLS,
    )

    BASE_REQUIRED_COLS: tuple[str, ...] = (
        "ts_code",
        "trade_date",
    )

    DERIVED_REQUIREMENTS: Mapping[str, tuple[str, ...]] = {
        "ret_1": ("close", "pre_close"),
        "range": ("high", "low"),
        "close_open": ("close", "open"),
        "high_open": ("high", "open"),
        "low_open": ("low", "open"),
        "log_volume": ("vol",),
        "log_amount": ("amount",),
        "vwap": ("amount", "vol"),
    }

    FACTOR_SOURCES: Mapping[str, tuple[str, ...]] = {
        "price_volume": PRICE_VOLUME_FEATURE_COLS,
        "moneyflow": MONEYFLOW_FEATURE_COLS,
        "fundamental": FUNDAMENTAL_FEATURE_COLS,
        "industry": INDUSTRY_FEATURE_COLS,
        "enhanced_alpha": ENHANCED_ALPHA_FEATURE_COLS,
    }

    SUPPORTED_LABEL_COLS: tuple[str, ...] = (
        "label_1d",
        "label_2d",
        "label_3d",
        "label_5d",
        "label_10d",
        "label_20d",
        "label_rank_1d",
        "label_rank_2d",
        "label_rank_3d",
        "label_rank_5d",
        "label_rank_10d",
        "label_rank_20d",
    )

    def __init__(
        self,
        daily_bars_dir: str | Path | None = None,
        feature_cols: Sequence[str] | None = None,
        label_horizon: int | None = None,
        label_name: str | None = None,
        label_cols: Sequence[str] | None = None,
        include_label: bool | None = None,
        dropna_label: bool | None = None,
        keep_original_code: bool | None = None,
        price_volume_factors_dir: str | Path | None = None,
        moneyflow_factors_dir: str | Path | None = None,
        fundamental_factors_dir: str | Path | None = None,
        industry_factors_dir: str | Path | None = None,
        enhanced_alpha_factors_dir: str | Path | None = None,
        labels_dir: str | Path | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        loader_config = load_loader_config(config_path)
        _ = label_horizon  # Backward-compatible, labels are now read from labels_dir.

        daily_bars_dir = (
            daily_bars_dir
            or loader_config.get("daily_bars_dir")
            or self.DATA_ROOT / "wide_table_daily_bars"
        )
        feature_cols = feature_cols or loader_config.get("feature_cols") or self.DEFAULT_FEATURE_COLS
        label_cols = self._resolve_label_cols(
            label_cols=label_cols,
            label_name=label_name,
            loader_config=loader_config,
        )
        include_label = include_label if include_label is not None else loader_config.get("include_label", True)
        dropna_label = dropna_label if dropna_label is not None else loader_config.get("dropna_label", True)
        keep_original_code = keep_original_code if keep_original_code is not None else loader_config.get("keep_original_code", True)

        self.daily_bars_dir = Path(daily_bars_dir)
        self.feature_cols = list(feature_cols)
        self.label_cols = list(label_cols)
        self.label_name = self.label_cols[0] if self.label_cols else None
        self.include_label = include_label
        self.dropna_label = dropna_label
        self.keep_original_code = keep_original_code
        self.labels_dir = Path(labels_dir or loader_config.get("labels_dir") or self.DEFAULT_LABELS_DIR)
        self.factor_dirs = {
            "price_volume": Path(
                price_volume_factors_dir
                or loader_config.get("price_volume_factors_dir")
                or self.DATA_ROOT / "price_volume_factors"
            ),
            "moneyflow": Path(
                moneyflow_factors_dir
                or loader_config.get("moneyflow_factors_dir")
                or self.DATA_ROOT / "moneyflow_factors"
            ),
            "fundamental": Path(
                fundamental_factors_dir
                or loader_config.get("fundamental_factors_dir")
                or self.DATA_ROOT / "fundamental_factors"
            ),
            "industry": Path(
                industry_factors_dir
                or loader_config.get("industry_factors_dir")
                or self.DATA_ROOT / "industry_factors"
            ),
            "enhanced_alpha": Path(
                enhanced_alpha_factors_dir
                or loader_config.get("enhanced_alpha_factors_dir")
                or self.DATA_ROOT / "enhanced_alpha_factors"
            ),
        }

        if not self.daily_bars_dir.exists():
            raise FileNotFoundError(f"daily_bars_dir does not exist: {self.daily_bars_dir}")

        self._validate_feature_cols()
        self._validate_label_cols()
        self._validate_required_dirs()

    def load(self, instruments, start_time=None, end_time=None) -> pd.DataFrame:
        """Load data from parquet and return Qlib-compatible DataFrame."""

        df = self._read_parquet(start_time=start_time, end_time=end_time)
        df = self._normalize(df)
        df = self._filter_instruments(df, instruments)
        df = self._add_derived_features(df)
        df = self._filter_time(df, start_time=start_time, end_time=end_time)
        df = self._clean_for_output(df)
        return self._to_qlib_dataframe(df)

    @classmethod
    def _resolve_label_cols(
        cls,
        label_cols: Sequence[str] | None,
        label_name: str | None,
        loader_config: Mapping,
    ) -> list[str]:
        if label_cols is not None:
            return list(label_cols)

        config_label_cols = loader_config.get("label_cols")    # 多个label可以在conf中写成： "label_cols": ["label_5d", "label_rank_5d"]
        if config_label_cols is not None:
            if isinstance(config_label_cols, str):
                return [config_label_cols]
            return list(config_label_cols)

        resolved_label_name = label_name or loader_config.get("label_name") or "label_5d"
        return [resolved_label_name]

    def _validate_feature_cols(self) -> None:
        supported = set(self.DEFAULT_FEATURE_COLS) | set(self.DERIVED_REQUIREMENTS)
        unknown = sorted(set(self.feature_cols) - supported)
        if unknown:
            raise ValueError(
                "Unsupported feature_cols: "
                f"{unknown}. Supported parquet columns: {list(self.DEFAULT_FEATURE_COLS)}; "
                f"supported derived columns: {list(self.DERIVED_REQUIREMENTS)}"
            )

    def _validate_label_cols(self) -> None:
        if not self.include_label:
            return
        unknown = sorted(set(self.label_cols) - set(self.SUPPORTED_LABEL_COLS))
        if unknown:
            raise ValueError(
                f"Unsupported label columns: {unknown}. "
                f"Supported label columns: {list(self.SUPPORTED_LABEL_COLS)}"
            )
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"labels_dir does not exist: {self.labels_dir}")

    def _validate_required_dirs(self) -> None:
        for source, required_cols in self._required_factor_cols_by_source().items():
            if required_cols and not self.factor_dirs[source].exists():
                raise FileNotFoundError(f"{source} factor dir does not exist: {self.factor_dirs[source]}")

    def _required_main_cols(self) -> list[str]:
        required: set[str] = set(self.BASE_REQUIRED_COLS)

        for col in self.feature_cols:
            if col in self.MAIN_FEATURE_COLS:
                required.add(col)
            elif col in self.DERIVED_REQUIREMENTS:
                required.update(self.DERIVED_REQUIREMENTS[col])

        return [col for col in (*self.BASE_REQUIRED_COLS, *self.MAIN_FEATURE_COLS) if col in required]

    def _required_factor_cols_by_source(self) -> dict[str, list[str]]:
        needed = set(self.feature_cols)
        required_by_source: dict[str, list[str]] = {}
        for source, source_cols in self.FACTOR_SOURCES.items():
            cols = [col for col in source_cols if col in needed]
            required_by_source[source] = cols
        return required_by_source

    def _read_parquet(self, start_time=None, end_time=None) -> pd.DataFrame:
        main_df = self._read_dataset(
            self.daily_bars_dir,
            self._required_main_cols(),
            start_time=start_time,
            end_time=end_time,
        )
        if main_df.empty:
            return main_df

        for source, cols in self._required_factor_cols_by_source().items():
            if not cols:
                continue
            factor_df = self._read_dataset(
                self.factor_dirs[source],
                [*self.BASE_REQUIRED_COLS, *cols],
                start_time=start_time,
                end_time=end_time,
            )
            main_df = main_df.merge(
                factor_df,
                on=list(self.BASE_REQUIRED_COLS),
                how="left",
                validate="one_to_one",
            )

        if self.include_label:
            label_df = self._read_dataset(
                self.labels_dir,
                [*self.BASE_REQUIRED_COLS, *self.label_cols],
                start_time=start_time,
                end_time=end_time,
            )
            main_df = main_df.merge(
                label_df,
                on=list(self.BASE_REQUIRED_COLS),
                how="left",
                validate="one_to_one",
            )

        return main_df

    def _read_dataset(
        self,
        data_dir: Path,
        columns: Sequence[str],
        start_time=None,
        end_time=None,
    ) -> pd.DataFrame:
        dataset = ds.dataset(str(data_dir), format="parquet", partitioning="hive")
        self._validate_dataset_columns(data_dir, dataset.schema.names, columns)
        filter_expr = self._build_arrow_date_filter(dataset.schema, start_time=start_time, end_time=end_time)
        table = dataset.to_table(columns=list(columns), filter=filter_expr)
        return table.to_pandas()

    @staticmethod
    def _validate_dataset_columns(data_dir: Path, available_cols: Sequence[str], required_cols: Sequence[str]) -> None:
        missing = [col for col in required_cols if col not in available_cols]
        if missing:
            raise KeyError(f"Columns {missing} are missing in parquet dataset: {data_dir}")

    @staticmethod
    def _build_arrow_date_filter(schema: pa.Schema, start_time=None, end_time=None):
        trade_date_field = schema.field("trade_date")
        use_int_date = pa.types.is_integer(trade_date_field.type)

        def convert_date(value):
            date_str = pd.Timestamp(value).strftime("%Y%m%d")
            return int(date_str) if use_int_date else date_str

        expr = None
        if start_time is not None:
            expr = ds.field("trade_date") >= convert_date(start_time)
        if end_time is not None:
            end_expr = ds.field("trade_date") <= convert_date(end_time)
            expr = end_expr if expr is None else expr & end_expr
        return expr

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["datetime", "instrument", *self.feature_cols])

        df = df.copy()
        df["datetime"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
        df["instrument"] = df["ts_code"].astype(str)
        if not self.keep_original_code:
            df["instrument"] = df["instrument"].map(self.normalize_symbol)

        value_cols = [
            col
            for col in (*self.DEFAULT_FEATURE_COLS, *self.DERIVED_REQUIREMENTS.keys(), *self.SUPPORTED_LABEL_COLS)
            if col in df.columns
        ]
        for col in value_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["datetime", "instrument"])
        df = df.drop_duplicates(subset=["datetime", "instrument"], keep="last")
        return df.sort_values(["instrument", "datetime"])

    @staticmethod
    def normalize_symbol(ts_code: str) -> str:
        code, _, exchange = ts_code.partition(".")
        return f"{exchange.lower()}{code}" if exchange else ts_code.lower()

    @staticmethod
    def _filter_instruments(df: pd.DataFrame, instruments) -> pd.DataFrame:
        if df.empty or instruments is None or instruments == "all":
            return df

        if isinstance(instruments, str):
            # For market names other than "all", this loader does not have an
            # InstrumentProvider. Keep data unchanged instead of failing because
            # Qlib examples often pass market aliases.
            return df

        if isinstance(instruments, Mapping):
            allowed = set(instruments.keys())
        else:
            allowed = set(instruments)

        return df[df["instrument"].isin(allowed)]

    def _add_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        needed = set(self.feature_cols)

        if "ret_1" in needed:
            df["ret_1"] = self._safe_div(df.get("close"), df.get("pre_close")) - 1.0
        if "range" in needed:
            df["range"] = self._safe_div(df.get("high"), df.get("low")) - 1.0
        if "close_open" in needed:
            df["close_open"] = self._safe_div(df.get("close"), df.get("open")) - 1.0
        if "high_open" in needed:
            df["high_open"] = self._safe_div(df.get("high"), df.get("open")) - 1.0
        if "low_open" in needed:
            df["low_open"] = self._safe_div(df.get("low"), df.get("open")) - 1.0
        if "log_volume" in needed:
            df["log_volume"] = np.log1p(df["vol"].where(df["vol"] >= 0))
        if "log_amount" in needed:
            df["log_amount"] = np.log1p(df["amount"].where(df["amount"] >= 0))
        if "vwap" in needed:
            df["vwap"] = self._safe_div(df.get("amount"), df.get("vol"))

        return df

    @staticmethod
    def _safe_div(left: pd.Series, right: pd.Series) -> pd.Series:
        return left / right.replace(0, np.nan)

    @staticmethod
    def _filter_time(df: pd.DataFrame, start_time=None, end_time=None) -> pd.DataFrame:
        if df.empty:
            return df
        mask = pd.Series(True, index=df.index)
        if start_time is not None:
            mask &= df["datetime"] >= pd.Timestamp(start_time)
        if end_time is not None:
            mask &= df["datetime"] <= pd.Timestamp(end_time)
        return df[mask]

    def _clean_for_output(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.replace([np.inf, -np.inf], np.nan)
        if self.include_label and self.dropna_label:
            df = df.dropna(subset=self.label_cols)
        return df

    def _to_qlib_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return self._empty_qlib_dataframe()

        missing_features = [col for col in self.feature_cols if col not in df.columns]
        if missing_features:
            raise KeyError(f"Feature columns are missing after processing: {missing_features}")

        feature_df = df[self.feature_cols].astype("float32")
        blocks: dict[str, pd.DataFrame] = {"feature": feature_df}

        if self.include_label:
            missing_labels = [col for col in self.label_cols if col not in df.columns]
            if missing_labels:
                raise KeyError(f"Label columns are missing after processing: {missing_labels}")
            blocks["label"] = df[self.label_cols].astype("float32")

        out = pd.concat(blocks, axis=1)
        out.index = pd.MultiIndex.from_frame(df[["datetime", "instrument"]])
        out.index.names = ["datetime", "instrument"]
        return out.sort_index()

    def _empty_qlib_dataframe(self) -> pd.DataFrame:
        columns: list[tuple[str, str]] = [("feature", col) for col in self.feature_cols]
        if self.include_label:
            columns.extend(("label", col) for col in self.label_cols)
        return pd.DataFrame(
            columns=pd.MultiIndex.from_tuples(columns),
            index=pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"]),
        )


__all__ = ["ParquetLoader"]
