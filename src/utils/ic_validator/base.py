"""Shared utilities for Preparatory feature IC validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


DATETIME_LEVEL = "datetime"
INSTRUMENT_LEVEL = "instrument"

DEFAULT_LOCAL_FEATURE_DIR = Path(
    "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cross_sectional_processd_data/industry_factors"
)
DEFAULT_LOCAL_LABEL_DIR = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/generated_label/daily_labels")
DEFAULT_LOCAL_FEATURE_COL = "industry_ret_1_cc_processed"
DEFAULT_LOCAL_LABEL_COL = "label_rank_1d"


@dataclass(frozen=True)
class ValidationResult:
    """Container returned by every validation step."""

    metrics: dict[str, Any]
    data: dict[str, pd.DataFrame | pd.Series]


class BaseICValidationStep:
    """Base class for validators that consume Qlib ``pred_label`` data.

    Qlib model analysis utilities expect a two-level index and two canonical
    columns: ``score`` and ``label``.  This base class normalizes project data
    into that shape while allowing callers to pass arbitrary column names.
    """

    def __init__(
        self,
        score_col: str = "score",
        label_col: str = "label",
        canonical_score_col: str = "score",
        canonical_label_col: str = "label",
    ) -> None:
        self.score_col = score_col
        self.label_col = label_col
        self.canonical_score_col = canonical_score_col
        self.canonical_label_col = canonical_label_col

    def prepare_pred_label(self, pred_label: pd.DataFrame) -> pd.DataFrame:
        """Return a clean Preparatory dataframe indexed by datetime/instrument."""

        if not isinstance(pred_label, pd.DataFrame):
            raise TypeError("pred_label must be a pandas DataFrame")
        if not isinstance(pred_label.index, pd.MultiIndex):
            raise ValueError("pred_label must use a MultiIndex with datetime/instrument levels")

        df = pred_label.copy()
        df = self._ensure_index_names(df)

        missing = [col for col in (self.score_col, self.label_col) if col not in df.columns]
        if missing:
            raise KeyError(f"pred_label missing required columns: {missing}")

        df = df[[self.score_col, self.label_col]].rename(
            columns={self.score_col: self.canonical_score_col, self.label_col: self.canonical_label_col}
        )
        df = df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=[self.canonical_score_col, self.canonical_label_col]
        )
        return df.sort_index()

    @staticmethod
    def _ensure_index_names(df: pd.DataFrame) -> pd.DataFrame:
        names = list(df.index.names)
        if DATETIME_LEVEL in names and INSTRUMENT_LEVEL in names:
            if names[0] != DATETIME_LEVEL:
                df = df.reorder_levels([DATETIME_LEVEL, INSTRUMENT_LEVEL]).sort_index()
            return df

        if len(names) != 2:
            raise ValueError("pred_label index must have exactly two levels")

        # Compatible with qlib examples that use [instrument, datetime] but do
        # not always set explicit names.
        level0 = df.index.get_level_values(0)
        level1 = df.index.get_level_values(1)
        if pd.api.types.is_datetime64_any_dtype(level0):
            df.index = df.index.set_names([DATETIME_LEVEL, INSTRUMENT_LEVEL])
        elif pd.api.types.is_datetime64_any_dtype(level1):
            df.index = df.index.set_names([INSTRUMENT_LEVEL, DATETIME_LEVEL])
            df = df.reorder_levels([DATETIME_LEVEL, INSTRUMENT_LEVEL])
        else:
            # Fall back to the project's current convention.
            df.index = df.index.set_names([DATETIME_LEVEL, INSTRUMENT_LEVEL])
        return df.sort_index()

    @staticmethod
    def risk_to_dict(risk_df: pd.DataFrame | pd.Series) -> dict[str, Any]:
        """Convert Qlib ``risk_analysis`` output to a JSON-friendly dict."""

        if isinstance(risk_df, pd.Series):
            raw: Mapping[Any, Any] = risk_df.to_dict()
        elif risk_df.shape[1] == 1:
            raw = risk_df.iloc[:, 0].to_dict()
        else:
            raw = risk_df.to_dict()
        return BaseICValidationStep.json_safe(dict(raw))

    @staticmethod
    def json_safe(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): BaseICValidationStep.json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [BaseICValidationStep.json_safe(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        try:
            if pd.isna(obj):
                return None
        except (TypeError, ValueError):
            pass
        return obj

    @staticmethod
    def flatten_feature_columns(feature_cols: Iterable[str] | None, pred_label: pd.DataFrame, label_col: str) -> list[str]:
        if feature_cols is not None:
            return list(feature_cols)
        return [col for col in pred_label.columns if col != label_col]


def load_local_test_pred_label(
    feature_dir: str | Path = DEFAULT_LOCAL_FEATURE_DIR,
    label_dir: str | Path = DEFAULT_LOCAL_LABEL_DIR,
    feature_col: str = DEFAULT_LOCAL_FEATURE_COL,
    label_col: str = DEFAULT_LOCAL_LABEL_COL,
    max_files: int = 20,
) -> pd.DataFrame:
    """Load a small local feature/label sample for analyzer smoke tests.

    The project stores both feature and label data as daily parquet shards under
    ``year=YYYY/month=MM/YYYYMMDD.parquet``.  This helper loads the latest
    matching shards, joins them on ``trade_date`` and ``ts_code``, and returns a
    Preparatory dataframe with columns ``feature_col`` and ``label_col``.
    """

    feature_root = Path(feature_dir)
    label_root = Path(label_dir)
    if not feature_root.exists():
        raise FileNotFoundError(f"feature_dir does not exist: {feature_root}")
    if not label_root.exists():
        raise FileNotFoundError(f"label_dir does not exist: {label_root}")

    feature_files = {path.name: path for path in feature_root.rglob("*.parquet")}
    label_files = {path.name: path for path in label_root.rglob("*.parquet")}
    common_names = sorted(set(feature_files).intersection(label_files), reverse=True)
    if not common_names:
        raise FileNotFoundError(f"No matching parquet shards under {feature_root} and {label_root}")

    frames: list[pd.DataFrame] = []
    for name in common_names:
        feature_df = pd.read_parquet(feature_files[name], columns=["trade_date", "ts_code", feature_col])
        label_df = pd.read_parquet(label_files[name], columns=["trade_date", "ts_code", label_col])
        merged = feature_df.merge(label_df, on=["trade_date", "ts_code"], how="inner")
        merged = merged.dropna(subset=[feature_col, label_col])
        if merged.empty:
            continue
        frames.append(merged)
        if len(frames) >= max_files:
            break

    if not frames:
        raise ValueError(f"No valid non-null rows found for feature={feature_col}, label={label_col}")

    df = pd.concat(frames, ignore_index=True)
    df[DATETIME_LEVEL] = pd.to_datetime(df["trade_date"].astype(str))
    df[INSTRUMENT_LEVEL] = df["ts_code"].astype(str)
    return df.set_index([DATETIME_LEVEL, INSTRUMENT_LEVEL])[[feature_col, label_col]].sort_index()


__all__ = [
    "BaseICValidationStep",
    "ValidationResult",
    "DATETIME_LEVEL",
    "INSTRUMENT_LEVEL",
    "DEFAULT_LOCAL_FEATURE_COL",
    "DEFAULT_LOCAL_FEATURE_DIR",
    "DEFAULT_LOCAL_LABEL_COL",
    "DEFAULT_LOCAL_LABEL_DIR",
    "load_local_test_pred_label",
]
