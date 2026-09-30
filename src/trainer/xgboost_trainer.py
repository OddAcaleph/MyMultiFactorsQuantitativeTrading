"""XGBoost trainer built on top of Qlib DatasetH and the custom ParquetLoader."""

from __future__ import annotations

import gc
import json
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from xgboost import XGBRanker, XGBRegressor

from utils import ParquetLoader, load_loader_config, load_trainer_config


class XGBoostTrainer:
    """Train XGBoost with local parquet daily bars through Qlib DatasetH.

    The data flow is:

    ``ParquetLoader -> DataHandlerLP -> DatasetH -> XGBRegressor``.

    This class intentionally does not define a custom Qlib Dataset because
    Qlib's built-in ``DatasetH`` already handles the train/valid/test time
    segments we need for daily-bar experiments.
    """

    def __init__(
        self,
        daily_bars_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        instruments: str | Sequence[str] | Mapping[str, Any] | None = None,
        feature_cols: Sequence[str] | None = None,
        segments: Mapping[str, tuple[str, str]] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        label_horizon: int | None = None,
        label_name: str | None = None,
        label_cols: Sequence[str] | None = None,
        model_params: Mapping[str, Any] | None = None,
        keep_original_code: bool | None = None,
        price_volume_factors_dir: str | Path | None = None,
        moneyflow_factors_dir: str | Path | None = None,
        fundamental_factors_dir: str | Path | None = None,
        industry_factors_dir: str | Path | None = None,
        enhanced_alpha_factors_dir: str | Path | None = None,
        labels_dir: str | Path | None = None,
        prefer_gpu: bool | None = None,
        model_dir: str | Path | None = None,
        model_filename: str | None = None,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
        loader_config_path: str | Path | None = None,
    ) -> None:
        self.config_path = config_path
        self.config = load_trainer_config(config_path)
        trainer_config = self.config
        self.loader_config_path = loader_config_path or trainer_config.get("loader_config_path")
        self.loader_config = load_loader_config(self.loader_config_path)

        daily_bars_dir = daily_bars_dir or self.loader_config.get("daily_bars_dir")
        if daily_bars_dir is None:
            raise ValueError("daily_bars_dir must be provided either explicitly or in loader config daily_bars_dir")

        self.daily_bars_dir = Path(daily_bars_dir)
        output_dir = output_dir if output_dir is not None else trainer_config.get("output_dir")
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.model_dir = Path(model_dir or trainer_config.get("model_dir", "models"))
        self.model_filename = model_filename or trainer_config.get("model_filename", "xgboost_model.json")
        self.model_path = Path(model_path) if model_path is not None else self.model_dir / self.model_filename
        self.instruments = instruments if instruments is not None else trainer_config.get("instruments", "all")
        self.feature_cols = list(feature_cols or self.loader_config.get("feature_cols") or [])
        if not self.feature_cols:
            raise ValueError("feature_cols must be provided either explicitly or in loader config feature_cols")

        self.label_horizon = label_horizon if label_horizon is not None else self.loader_config.get("label_horizon", 1)
        effective_label_name = label_name or trainer_config.get("label_name")
        self.label_cols = self._resolve_label_cols(label_cols=label_cols, label_name=effective_label_name)
        self.label_name = effective_label_name or self.label_cols[0]
        self.keep_original_code = (
            keep_original_code if keep_original_code is not None else self.loader_config.get("keep_original_code", True)
        )
        self.price_volume_factors_dir = price_volume_factors_dir or self.loader_config.get("price_volume_factors_dir")
        self.moneyflow_factors_dir = moneyflow_factors_dir or self.loader_config.get("moneyflow_factors_dir")
        self.fundamental_factors_dir = fundamental_factors_dir or self.loader_config.get("fundamental_factors_dir")
        self.industry_factors_dir = industry_factors_dir or self.loader_config.get("industry_factors_dir")
        self.enhanced_alpha_factors_dir = enhanced_alpha_factors_dir or self.loader_config.get("enhanced_alpha_factors_dir")
        self.labels_dir = labels_dir or self.loader_config.get("labels_dir")

        self.segments = self._normalize_segments(segments or trainer_config.get("segments"))
        self.start_time = start_time or trainer_config.get("start_time") or min(start for start, _ in self.segments.values())
        self.end_time = end_time or trainer_config.get("end_time") or max(end for _, end in self.segments.values())

        params = dict(trainer_config.get("model_params", {}))
        if model_params:
            params.update(model_params)
        self.model_params = params
        self.prefer_gpu = prefer_gpu if prefer_gpu is not None else trainer_config.get("prefer_gpu", True)

        self.loader: ParquetLoader | None = None
        self.handler: DataHandlerLP | None = None
        self.dataset: DatasetH | None = None
        self.model: XGBRegressor | None = None
        self.device_used: str | None = None

        # Evaluation framework config
        eval_config = trainer_config.get("evaluation", {})
        self.evaluation_enabled = bool(eval_config.get("enabled", False))
        self.evaluation_config = dict(eval_config)

    def _resolve_label_cols(self, label_cols: Sequence[str] | None, label_name: str | None) -> list[str]:
        if label_cols is not None:
            return list(label_cols)
        if label_name is not None:
            return [label_name]
        config_label_cols = self.loader_config.get("label_cols")
        if config_label_cols is not None:
            if isinstance(config_label_cols, str):
                return [config_label_cols]
            return list(config_label_cols)
        return [label_name or self.loader_config.get("label_name", "label_5d")]

    @staticmethod
    def _normalize_segments(segments: Mapping[str, Sequence[str]] | None) -> dict[str, tuple[str, str]]:
        if not segments:
            raise ValueError("segments must be provided either explicitly or in config.trainer.segments")
        return {name: (str(bounds[0]), str(bounds[1])) for name, bounds in segments.items()}

    def build_dataset(self) -> DatasetH:
        """Build Qlib DatasetH using the custom parquet loader."""

        self.loader = ParquetLoader(
            daily_bars_dir=self.daily_bars_dir,
            price_volume_factors_dir=self.price_volume_factors_dir,
            moneyflow_factors_dir=self.moneyflow_factors_dir,
            fundamental_factors_dir=self.fundamental_factors_dir,
            industry_factors_dir=self.industry_factors_dir,
            enhanced_alpha_factors_dir=self.enhanced_alpha_factors_dir,
            labels_dir=self.labels_dir,
            feature_cols=self.feature_cols,
            label_horizon=self.label_horizon,
            label_name=self.label_name,
            label_cols=self.label_cols,
            include_label=True,
            dropna_label=True,
            keep_original_code=self.keep_original_code,
            config_path=self.loader_config_path,
        )
        self.handler = DataHandlerLP(
            instruments=self.instruments,
            start_time=self.start_time,
            end_time=self.end_time,
            data_loader=self.loader,
        )
        self.dataset = DatasetH(handler=self.handler, segments=self.segments)
        return self.dataset

    def prepare_segment(self, segment: str) -> pd.DataFrame:
        """Prepare one DatasetH segment."""

        if self.dataset is None:
            self.build_dataset()
        assert self.dataset is not None
        return self.dataset.prepare(segment)

    def prepare_xy(self, segment: str) -> tuple[pd.DataFrame, pd.Series]:
        """Prepare feature matrix and label vector for one segment.

        Loads directly from ParquetLoader to avoid Qlib DataHandlerLP caching
        the full date range in memory (which can double peak memory).
        """

        start, end = self.segments[segment]
        if self.loader is None:
            self.loader = ParquetLoader(
                daily_bars_dir=self.daily_bars_dir,
                price_volume_factors_dir=self.price_volume_factors_dir,
                moneyflow_factors_dir=self.moneyflow_factors_dir,
                fundamental_factors_dir=self.fundamental_factors_dir,
                industry_factors_dir=self.industry_factors_dir,
                enhanced_alpha_factors_dir=self.enhanced_alpha_factors_dir,
                labels_dir=self.labels_dir,
                feature_cols=self.feature_cols,
                label_horizon=self.label_horizon,
                label_name=self.label_name,
                label_cols=self.label_cols,
                include_label=True,
                dropna_label=True,
                keep_original_code=self.keep_original_code,
                config_path=self.loader_config_path,
            )
        df = self.loader.load(self.instruments, start_time=start, end_time=end)
        return self.split_feature_label(df, label_name=self.label_name)

    @staticmethod
    def split_feature_label(df: pd.DataFrame, label_name: str = "label_5d") -> tuple[pd.DataFrame, pd.Series]:
        """Split DatasetH output into X and y.

        ``DataHandlerLP`` may return either MultiIndex columns
        (``feature/open`` and ``label/label_5d``) or flattened columns
        (``open``, ``close``, ``label_5d``), so this helper supports both.
        """

        if df.empty:
            raise ValueError("Cannot split an empty dataset segment")

        if isinstance(df.columns, pd.MultiIndex):
            x = df["feature"]
            y = df["label"][label_name]
        else:
            if label_name not in df.columns:
                raise KeyError(f"Label column not found: {label_name}")
            x = df.drop(columns=[label_name])
            y = df[label_name]

        valid = y.notna()
        x = x.loc[valid].replace([np.inf, -np.inf], np.nan)
        y = y.loc[valid]
        x = x.fillna(0.0)
        return x, y

    def fit(self, verbose: bool | int = 50) -> XGBRegressor:
        """Train an XGBRegressor on train segment and validate on valid segment."""

        x_train, y_train = self.prepare_xy("train")
        x_valid, y_valid = self.prepare_xy("valid")

        self.loader = None
        gc.collect()

        fit_attempts = []
        if self.prefer_gpu and self._gpu_available():
            fit_attempts.append(("gpu", self._model_params_for_device("gpu")))
        fit_attempts.append(("cpu", self._model_params_for_device("cpu")))

        last_error: Exception | None = None
        for device_name, params in fit_attempts:
            try:
                obj = params.get("objective", "")
                if obj.startswith("rank:"):
                    self.model = XGBRanker(**params)
                    train_group = self._group_sizes(x_train)
                    # Skip eval_set for ranking with float labels: XGBoost's
                    # default ranking eval metric needs integer relevance labels.
                    self.model.fit(
                        x_train, y_train,
                        group=train_group,
                        verbose=verbose,
                    )
                else:
                    self.model = XGBRegressor(**params)
                    self.model.fit(
                        x_train, y_train,
                        eval_set=[(x_valid, y_valid)],
                        verbose=verbose,
                    )
                self.device_used = device_name
                del x_train, y_train, x_valid, y_valid
                gc.collect()
                return self.model
            except Exception as exc:
                last_error = exc
                if device_name == "gpu":
                    warnings.warn(
                        f"XGBoost GPU training failed, falling back to CPU. Original error: {exc}",
                        RuntimeWarning,
                    )
                    continue
                raise

        raise RuntimeError("XGBoost training failed on all available devices") from last_error

    @staticmethod
    def _group_sizes(x: pd.DataFrame) -> list[int]:
        """Return group sizes for ranking (one group per trading day)."""
        dates = x.index.get_level_values("datetime")
        return list(pd.Series(dates).value_counts().sort_index().values)

    @staticmethod
    def _gpu_available() -> bool:
        """Return True when at least one NVIDIA GPU is visible to the process."""

        try:
            import pynvml

            pynvml.nvmlInit()
            return pynvml.nvmlDeviceGetCount() > 0
        except Exception:
            return False

    def _model_params_for_device(self, device_name: str) -> dict[str, Any]:
        """Build XGBoost params for GPU or CPU.

        XGBoost 2.x/3.x recommends ``tree_method='hist'`` with
        ``device='cuda'`` for GPU training. CPU fallback uses the same hist
        algorithm with ``device='cpu'``.
        """

        params = dict(self.model_params)
        params["tree_method"] = "hist"
        if device_name == "gpu":
            params["device"] = "cuda"
        else:
            params["device"] = "cpu"
        return params

    def predict(self, segment: str = "test") -> pd.DataFrame:
        """Predict one segment and return index-aligned predictions."""

        if self.model is None:
            raise RuntimeError("Model is not fitted. Call fit() before predict().")

        df = self.prepare_segment(segment)
        x, y = self.split_feature_label(df, label_name=self.label_name)
        pred = self.model.predict(x)
        return pd.DataFrame({"pred": pred, self.label_name: y}, index=x.index)

    def evaluate(self, pred_df: pd.DataFrame) -> dict[str, float]:
        """Evaluate predictions with RMSE, IC and RankIC."""

        if pred_df.empty:
            return {"rmse": float("nan"), "ic": float("nan"), "rank_ic": float("nan")}

        pred = pred_df["pred"]
        label = pred_df[self.label_name]
        diff = pred - label
        rmse = float(np.sqrt(np.mean(np.square(diff))))
        ic = float(pred.corr(label))

        by_day = pred_df.groupby(level="datetime", group_keys=False)
        daily_rank_ic = by_day.apply(lambda x: x["pred"].corr(x[self.label_name], method="spearman"))
        rank_ic = float(daily_rank_ic.mean())

        return {"rmse": rmse, "ic": ic, "rank_ic": rank_ic}

    def run(self, verbose: bool | int = 50, save: bool = True) -> dict[str, Any]:
        """Fit model, predict test segment, evaluate and optionally save outputs."""

        model = self.fit(verbose=verbose)
        pred_df = self.predict("test")
        metrics = self.evaluate(pred_df)

        result = {"model": model, "pred": pred_df, "metrics": metrics}

        # Run full evaluation report if enabled
        if self.evaluation_enabled:
            try:
                eval_result = self.run_full_evaluation(pred_df)
                result["evaluation"] = eval_result
            except Exception as e:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Full evaluation report failed: {e}")
                result["evaluation"] = {"error": str(e)}

        if save:
            self.save_outputs(model=model, pred_df=pred_df, metrics=metrics)
        return result

    def run_full_evaluation(self, pred_df: pd.DataFrame) -> dict[str, Any]:
        """Run the full evaluation report on test-set predictions.

        Loads additional label columns (5d/10d/20d returns) as needed,
        builds the evaluation pred_label dataframe, and invokes
        :class:`evaluation.EvaluationReport`.
        """
        import gc

        # Free training data memory before evaluation to avoid OOM
        self.model = None
        self.dataset = None
        self.handler = None
        self.loader = None
        gc.collect()

        from evaluation import EvaluationConfig, EvaluationReport

        eval_cfg = self.evaluation_config
        score_col = eval_cfg.get("score_col", "pred")

        # Determine return horizons needed for evaluation
        return_horizons = eval_cfg.get("return_horizons", [5, 10, 20])
        return_cols = {f"{h}d": f"label_{h}d" for h in return_horizons}
        return_label_cols = list(return_cols.values())

        # Build the full pred_label dataframe with all required return columns
        eval_pred_label = self._build_eval_pred_label(pred_df, return_label_cols)

        # Primary label for IC = training label
        # Return label for hit-rate / upside / downside = raw return (first horizon)
        primary_label = self.label_name
        return_label = return_label_cols[0]

        # Backtest config
        run_backtest = bool(eval_cfg.get("run_backtest", True))
        backtest_config_path = eval_cfg.get("backtest_config_path")
        backtest_overrides = eval_cfg.get("backtest_overrides", {})

        config = EvaluationConfig(
            score_col=score_col,
            primary_label_col=primary_label,
            return_label_col=return_label,
            return_cols=return_cols,
            n_groups=int(eval_cfg.get("n_groups", 10)),
            top_ks=tuple(eval_cfg.get("top_ks", [50, 100, 200])),
            upside_realized_frac=float(eval_cfg.get("upside_realized_frac", 0.10)),
            upside_predicted_fracs=tuple(eval_cfg.get("upside_predicted_fracs", [0.10, 0.20])),
            downside_bottom_frac=float(eval_cfg.get("downside_bottom_frac", 0.10)),
            downside_crash_threshold=float(eval_cfg.get("downside_crash_threshold", -0.095)),
            run_backtest=run_backtest,
            backtest_config_path=backtest_config_path,
            backtest_overrides=backtest_overrides,
        )

        report = EvaluationReport(config=config)
        result = report.generate(eval_pred_label)

        # Save if output_dir is available
        if self.output_dir is not None:
            eval_dir = self.output_dir / "evaluation_report"
            report.save_report(eval_dir)

        return result

    def _build_eval_pred_label(self, pred_df: pd.DataFrame, return_label_cols: list[str]) -> pd.DataFrame:
        """Build evaluation dataframe with predictions + all required return labels.

        Uses pyarrow.dataset for efficient partitioned reads of label data.
        """
        if self.labels_dir is None:
            return pred_df.copy()

        missing_labels = [c for c in return_label_cols if c not in pred_df.columns]
        if not missing_labels:
            return pred_df.copy()

        from pathlib import Path
        import pyarrow.dataset as ds

        labels_dir = Path(self.labels_dir)
        if not labels_dir.exists():
            return pred_df.copy()

        test_start, test_end = self.segments.get("test", (self.start_time, self.end_time))
        test_start_int = int(pd.Timestamp(test_start).strftime("%Y%m%d"))
        test_end_int = int(pd.Timestamp(test_end).strftime("%Y%m%d"))

        try:
            dataset = ds.dataset(str(labels_dir), format="parquet", partitioning="hive")
            table = dataset.to_table(
                columns=["trade_date", "ts_code", *missing_labels],
                filter=(ds.field("trade_date") >= test_start_int)
                       & (ds.field("trade_date") <= test_end_int),
            )
            label_df = table.to_pandas()
        except Exception:
            return pred_df.copy()

        if label_df.empty:
            return pred_df.copy()

        label_df["datetime"] = pd.to_datetime(label_df["trade_date"].astype(str))
        label_df = label_df.rename(columns={"ts_code": "instrument"})
        label_df = label_df.set_index(["datetime", "instrument"]).drop(columns=["trade_date"])

        result = pred_df.join(label_df, how="left")
        return result

    def save_outputs(
        self,
        model: XGBRegressor | None = None,
        pred_df: pd.DataFrame | None = None,
        metrics: Mapping[str, float] | None = None,
    ) -> None:
        """Save model, predictions and metrics.

        Model artifacts are saved under ``model_path`` (default:
        ``/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/models`` via config).
        Predictions and metrics remain under ``output_dir`` for downstream
        evaluation/backtesting compatibility.
        """

        model = model or self.model
        if model is not None:
            self.save_model(model)

        if self.output_dir is None:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if pred_df is not None:
            pred_df.to_parquet(self.output_dir / "pred_test.parquet")
        if metrics is not None:
            with (self.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
                json.dump(dict(metrics), f, ensure_ascii=False, indent=2)

    def save_model(self, model: XGBRegressor | None = None) -> Path:
        """Save trained model to the configured models directory."""

        model = model or self.model
        if model is None:
            raise RuntimeError("No fitted model to save")
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_model(str(self.model_path))
        return self.model_path


__all__ = ["XGBoostTrainer"]
