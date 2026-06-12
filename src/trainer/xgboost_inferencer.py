"""XGBoost inferencer for loading saved models and generating predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from xgboost import XGBRegressor

from utils import ParquetLoader, load_inferencer_config, load_loader_config, load_trainer_config


class XGBoostInferencer:
    """Load a saved XGBoost model and run inference through Qlib DatasetH."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        output_dir: str | Path | None = None,
        instruments: str | Sequence[str] | Mapping[str, Any] | None = None,
        feature_cols: Sequence[str] | None = None,
        segment: str | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        label_name: str | None = None,
        label_cols: Sequence[str] | None = None,
        prefer_gpu: bool | None = None,
        config_path: str | Path | None = None,
        trainer_config_path: str | Path | None = None,
        loader_config_path: str | Path | None = None,
    ) -> None:
        self.config_path = config_path
        self.config = load_inferencer_config(config_path)

        self.trainer_config_path = trainer_config_path or self.config.get("trainer_config_path")
        self.trainer_config = load_trainer_config(self.trainer_config_path) if self.trainer_config_path else {}

        self.loader_config_path = loader_config_path or self.config.get("loader_config_path") or self.trainer_config.get(
            "loader_config_path"
        )
        self.loader_config = load_loader_config(self.loader_config_path)

        self.model_path = Path(model_path or self.config.get("model_path") or self._model_path_from_trainer_config())
        self.output_dir = Path(output_dir or self.config.get("output_dir") or self.trainer_config.get("output_dir"))
        self.instruments = instruments if instruments is not None else self.config.get(
            "instruments", self.trainer_config.get("instruments", "all")
        )
        self.feature_cols = list(feature_cols or self.loader_config.get("feature_cols") or [])
        if not self.feature_cols:
            raise ValueError("feature_cols must be provided either explicitly or in loader config feature_cols")

        self.segment = segment or self.config.get("segment", "test")
        self.start_time, self.end_time = self._resolve_infer_period(start_time=start_time, end_time=end_time)
        self.label_cols = self._resolve_label_cols(label_cols=label_cols, label_name=label_name)
        self.label_name = label_name or self.config.get("columns", {}).get("label") or self.label_cols[0]
        self.prediction_col = self.config.get("columns", {}).get("prediction", "pred")
        self.include_label = bool(self.config.get("include_label", True))
        self.prefer_gpu = prefer_gpu if prefer_gpu is not None else self.config.get("prefer_gpu", True)

        self.loader: ParquetLoader | None = None
        self.handler: DataHandlerLP | None = None
        self.dataset: DatasetH | None = None
        self.model: XGBRegressor | None = None

    def _resolve_label_cols(self, label_cols: Sequence[str] | None, label_name: str | None) -> list[str]:
        if label_cols is not None:
            return list(label_cols)
        config_label_cols = self.loader_config.get("label_cols")
        if config_label_cols is not None:
            if isinstance(config_label_cols, str):
                return [config_label_cols]
            return list(config_label_cols)
        return [label_name or self.config.get("columns", {}).get("label") or self.loader_config.get("label_name", "label_5d")]

    def _model_path_from_trainer_config(self) -> Path:
        model_dir = self.trainer_config.get("model_dir")
        model_filename = self.trainer_config.get("model_filename")
        if not model_dir or not model_filename:
            raise ValueError("model_path is missing and trainer config has no model_dir/model_filename")
        return Path(model_dir) / model_filename

    def _resolve_infer_period(self, start_time: str | None = None, end_time: str | None = None) -> tuple[str, str]:
        start = start_time or self.config.get("start_time")
        end = end_time or self.config.get("end_time")
        if start is None or end is None:
            segment_bounds = self.trainer_config.get("segments", {}).get(self.segment)
            if not segment_bounds:
                raise ValueError(f"Cannot resolve infer period for segment: {self.segment}")
            start = start or segment_bounds[0]
            end = end or segment_bounds[1]
        return str(start), str(end)

    def build_dataset(self) -> DatasetH:
        self.loader = ParquetLoader(
            daily_bars_dir=self.loader_config.get("daily_bars_dir"),
            price_volume_factors_dir=self.loader_config.get("price_volume_factors_dir"),
            moneyflow_factors_dir=self.loader_config.get("moneyflow_factors_dir"),
            fundamental_factors_dir=self.loader_config.get("fundamental_factors_dir"),
            industry_factors_dir=self.loader_config.get("industry_factors_dir"),
            labels_dir=self.loader_config.get("labels_dir"),
            feature_cols=self.feature_cols,
            label_horizon=self.loader_config.get("label_horizon", 1),
            label_name=self.label_name,
            label_cols=self.label_cols,
            include_label=self.include_label,
            dropna_label=self.include_label,
            keep_original_code=self.loader_config.get("keep_original_code", True),
            config_path=self.loader_config_path,
        )
        self.handler = DataHandlerLP(
            instruments=self.instruments,
            start_time=self.start_time,
            end_time=self.end_time,
            data_loader=self.loader,
        )
        self.dataset = DatasetH(handler=self.handler, segments={self.segment: (self.start_time, self.end_time)})
        return self.dataset

    def load_model(self) -> XGBRegressor:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file does not exist: {self.model_path}")
        self.model = XGBRegressor()
        self.model.load_model(str(self.model_path))
        self.model.set_params(device="cuda" if self.prefer_gpu and self._gpu_available() else "cpu")
        return self.model

    def prepare_segment(self) -> pd.DataFrame:
        if self.dataset is None:
            self.build_dataset()
        assert self.dataset is not None
        return self.dataset.prepare(self.segment)

    def predict(self, save: bool = False) -> pd.DataFrame:
        if self.model is None:
            self.load_model()
        assert self.model is not None
        df = self.prepare_segment()
        x, y = self.split_feature_label(df, label_name=self.label_name, include_label=self.include_label)
        pred = self.model.predict(x)
        out = pd.DataFrame({self.prediction_col: pred}, index=x.index)
        if y is not None:
            out[self.label_name] = y
        if save:
            self.save_predictions(out)
        return out

    @staticmethod
    def split_feature_label(
        df: pd.DataFrame,
        label_name: str = "label_5d",
        include_label: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        if df.empty:
            raise ValueError("Cannot run inference on an empty dataset segment")

        if isinstance(df.columns, pd.MultiIndex):
            x = df["feature"]
            y = df["label"][label_name] if include_label and "label" in df.columns.get_level_values(0) else None
        else:
            y = df[label_name] if include_label and label_name in df.columns else None
            x = df.drop(columns=[label_name]) if y is not None else df

        if y is not None:
            valid = y.notna()
            x = x.loc[valid]
            y = y.loc[valid]
        x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return x, y

    def save_predictions(self, pred_df: pd.DataFrame) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / self.config.get("prediction_filename", "pred_test.parquet")
        pred_df.to_parquet(path)
        return path

    @staticmethod
    def _gpu_available() -> bool:
        try:
            import pynvml

            pynvml.nvmlInit()
            return pynvml.nvmlDeviceGetCount() > 0
        except Exception:
            return False


__all__ = ["XGBoostInferencer"]
