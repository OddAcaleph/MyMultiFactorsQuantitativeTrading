"""Memory-optimized Walk-Forward XGBoost trainer for A-share daily bars.

This module provides ``WalkForwardTrainer`` which trains multiple XGBoost models
over rolling or expanding time windows.  Compared to running
``XGBoostTrainer`` in a loop, it reduces memory pressure in three ways:

1. **Shared preload**: the full feature+label panel is loaded once across the
   entire walk-forward date range, then each window slices from the cached panel.
   This avoids repeated parquet reads and repeated merge/normalize work.

2. **Per-window release**: training data (X/y arrays are freed as soon as ``fit()``
   returns; only the small model object and per-window predictions persist.

3. **Dtype discipline**: every numeric columns are kept as ``float32`` end-to-end, and
   intermediate copies (``.copy()`` are avoided wherever safe.

Typical usage::

    trainer = WalkForwardTrainer(
        config_path="conf/walk_forward_config.json",
    )
    results = trainer.run()
"""

from __future__ import annotations

import gc
import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from utils import ParquetLoader, load_walk_forward_config, load_loader_config


@dataclass
class WindowSpec:
    """Specification for one walk-forward window."""

    idx: int
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str

    @property
    def name(self) -> str:
        return (
            f"wf{self.idx:02d}_{self.train_start[:4]}-{self.train_end[:4]}"
            f"_test{self.test_start[:4]}"
        )


@dataclass
class WindowResult:
    """Result of one walk-forward window."""

    window: WindowSpec
    model_path: Path | None = None
    pred_path: Path | None = None
    metrics: dict[str, float] = field(default_factory=dict)


class WalkForwardTrainer:
    """Walk-forward XGBoost trainer with memory-efficient data preloading.

    The data flow is::

        ParquetLoader (one full load) -> cached_panel
            -> per-window slice -> XGBRegressor.fit() -> per-window predict

    Parameters
    ----------
    config_path
        Path to the walk-forward JSON config.  Defaults to
        ``conf/walk_forward_config.json``.
    loader_config_path
        Optional override for the ParquetLoader config.
    model_params
        Optional override for XGBoost model parameters.
    prefer_gpu
        Whether to try GPU training before falling back to CPU.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        loader_config_path: str | Path | None = None,
        model_params: Mapping[str, Any] | None = None,
        prefer_gpu: bool | None = None,
    ) -> None:
        self.config_path = config_path
        self.config = load_walk_forward_config(config_path)
        self.loader_config_path = loader_config_path or self.config.get("loader_config_path")
        self.loader_config = load_loader_config(self.loader_config_path)

        wf = self.config.get("walk_forward", {})
        self.mode: str = wf.get("mode", "rolling")
        self.train_window_years: int = int(wf.get("train_window_years", 5))
        self.step_years: int = int(wf.get("step_years", 1))
        self.valid_ratio: float = float(wf.get("valid_ratio", 0.15))

        self.train_start: str = wf["train_start"]
        self.train_end: str = wf["train_end"]
        self.test_start: str = wf["test_start"]
        self.test_end: str = wf["test_end"]

        params = dict(wf.get("model_params", {}))
        if model_params:
            params.update(model_params)
        self.model_params: dict[str, Any] = params
        self.prefer_gpu = prefer_gpu if prefer_gpu is not None else self.config.get("prefer_gpu", False)

        # Loss enhancement for long-side improvement
        loss_cfg = wf.get("loss_config", {}) or self.config.get("loss_config", {}) or {}
        self.loss_type: str = loss_cfg.get("type", "mse")  # mse | quantile | weighted
        self.quantile_alpha: float = float(loss_cfg.get("quantile_alpha", 0.75))
        self.label_weight_power: float = float(loss_cfg.get("label_weight_power", 2.0))
        self.label_weight_min: float = float(loss_cfg.get("label_weight_min", 0.2))

        self.label_name: str = self.config.get("label_name", "label_rank_5d")
        self.instruments = self.config.get("instruments", "all")

        output_cfg = self.config.get("output", {})
        self.model_dir = Path(output_cfg.get("model_dir", "output/walk_forward/models"))
        self.prediction_dir = Path(output_cfg.get("prediction_dir", "output/walk_forward/predictions"))
        self.backtest_dir = Path(output_cfg.get("backtest_dir", "output/walk_forward/backtest"))

        self.feature_cols: list[str] = list(self.loader_config.get("feature_cols") or [])
        if not self.feature_cols:
            raise ValueError("feature_cols must be provided in loader config")

        self.windows: list[WindowSpec] = []

    # ------------------------------------------------------------------
    # Window generation
    # ------------------------------------------------------------------

    def generate_windows(self) -> list[WindowSpec]:
        """Generate rolling or expanding window specifications."""

        ts = pd.Timestamp(self.train_start)
        te = pd.Timestamp(self.train_end)
        test_s = pd.Timestamp(self.test_start)
        test_e = pd.Timestamp(self.test_end)

        windows: list[WindowSpec] = []
        idx = 0

        if self.mode == "rolling":
            current_train_start = ts
            while True:
                current_train_end = (
                    current_train_start
                    + pd.DateOffset(years=self.train_window_years)
                    - pd.Timedelta(days=1)
                )
                current_test_start = current_train_end + pd.Timedelta(days=1)
                current_test_end = (
                    current_test_start
                    + pd.DateOffset(years=self.step_years)
                    - pd.Timedelta(days=1)
                )

                if current_test_start > test_e:
                    break
                if current_test_end < test_s:
                    current_train_start += pd.DateOffset(years=self.step_years)
                    continue

                actual_test_start = max(current_test_start, test_s)
                actual_test_end = min(current_test_end, test_e)

                train_days = (current_train_end - current_train_start).days
                valid_days = int(train_days * self.valid_ratio)
                valid_start = current_train_end - pd.Timedelta(days=valid_days) + pd.Timedelta(days=1)
                train_end_for_split = valid_start - pd.Timedelta(days=1)

                windows.append(
                    WindowSpec(
                        idx=idx,
                        train_start=current_train_start.strftime("%Y-%m-%d"),
                        train_end=train_end_for_split.strftime("%Y-%m-%d"),
                        valid_start=valid_start.strftime("%Y-%m-%d"),
                        valid_end=current_train_end.strftime("%Y-%m-%d"),
                        test_start=actual_test_start.strftime("%Y-%m-%d"),
                        test_end=actual_test_end.strftime("%Y-%m-%d"),
                    )
                )
                idx += 1
                current_train_start += pd.DateOffset(years=self.step_years)

        elif self.mode == "expanding":
            current_train_end = (
                ts + pd.DateOffset(years=self.train_window_years) - pd.Timedelta(days=1)
            )
            while current_train_end < test_e:
                current_test_start = current_train_end + pd.Timedelta(days=1)
                current_test_end = (
                    current_test_start
                    + pd.DateOffset(years=self.step_years)
                    - pd.Timedelta(days=1)
                )

                if current_test_start > test_e:
                    break
                if current_test_end < test_s:
                    current_train_end += pd.DateOffset(years=self.step_years)
                    continue

                actual_test_start = max(current_test_start, test_s)
                actual_test_end = min(current_test_end, test_e)

                train_days = (current_train_end - ts).days
                valid_days = int(train_days * self.valid_ratio)
                valid_start = current_train_end - pd.Timedelta(days=valid_days) + pd.Timedelta(days=1)
                train_end_for_split = valid_start - pd.Timedelta(days=1)

                windows.append(
                    WindowSpec(
                        idx=idx,
                        train_start=ts.strftime("%Y-%m-%d"),
                        train_end=train_end_for_split.strftime("%Y-%m-%d"),
                        valid_start=valid_start.strftime("%Y-%m-%d"),
                        valid_end=current_train_end.strftime("%Y-%m-%d"),
                        test_start=actual_test_start.strftime("%Y-%m-%d"),
                        test_end=actual_test_end.strftime("%Y-%m-%d"),
                    )
                )
                idx += 1
                current_train_end += pd.DateOffset(years=self.step_years)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.windows = windows
        return windows

    # ------------------------------------------------------------------
    # Per-window data loading (memory-optimized)
    # ------------------------------------------------------------------

    def _load_range_xy(
        self,
        start: str,
        end: str,
        dropna_label: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Load feature+label data for a date range.

        Uses direct file-path reads (not ``pyarrow.dataset``) to avoid the
        10+ GB metadata scan of the full parquet tree.  Data is loaded
        month-by-month and kept as pyarrow Tables until the final step to
        minimise peak memory.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq

        from utils.dataset_generator.parquet_loader import ParquetLoader as PL

        loader_cfg = self.loader_config
        daily_bars_dir = Path(loader_cfg["daily_bars_dir"])
        labels_dir = Path(loader_cfg["labels_dir"])

        factor_dirs = {
            "price_volume": Path(loader_cfg.get("price_volume_factors_dir", "")),
            "moneyflow": Path(loader_cfg.get("moneyflow_factors_dir", "")),
            "fundamental": Path(loader_cfg.get("fundamental_factors_dir", "")),
            "industry": Path(loader_cfg.get("industry_factors_dir", "")),
            "enhanced_alpha": Path(loader_cfg.get("enhanced_alpha_factors_dir", "")),
        }

        factor_sources = PL.FACTOR_SOURCES
        main_feature_cols = PL.MAIN_FEATURE_COLS
        base_cols = list(PL.BASE_REQUIRED_COLS)

        needed = set(self.feature_cols)
        main_cols_needed = [c for c in main_feature_cols if c in needed]

        active_factor_sources: list[tuple[str, list[str]]] = []
        for source, source_cols in factor_sources.items():
            cols_needed = [c for c in source_cols if c in needed]
            if not cols_needed:
                continue
            fdir = factor_dirs.get(source)
            if not fdir or not fdir.exists():
                continue
            active_factor_sources.append((source, cols_needed))

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        feature_cols_list = list(self.feature_cols)

        def _list_month_files(base_dir: Path, year: int, month: int) -> list[Path]:
            month_dir = base_dir / f"year={year}" / f"month={month:02d}"
            if not month_dir.exists():
                return []
            return sorted(month_dir.glob("*.parquet"))

        def _read_table(files: list[Path], columns: list[str]) -> pa.Table | None:
            if not files:
                return None
            if len(files) == 1:
                return pq.read_table(files[0], columns=columns)
            tables = [pq.read_table(f, columns=columns) for f in files]
            return pa.concat_tables(tables)

        all_tables: list[pa.Table] = []
        month_start = start_ts.replace(day=1)
        while month_start <= end_ts:
            month_end = month_start + pd.DateOffset(months=1) - pd.Timedelta(days=1)
            actual_start = max(month_start, start_ts)
            actual_end = min(month_end, end_ts)
            year = month_start.year
            month = month_start.month
            start_int = int(actual_start.strftime("%Y%m%d"))
            end_int = int(actual_end.strftime("%Y%m%d"))

            main_files = _list_month_files(daily_bars_dir, year, month)
            if not main_files:
                month_start = month_end + pd.Timedelta(days=1)
                continue

            table = _read_table(main_files, base_cols + main_cols_needed)
            if table is None:
                month_start = month_end + pd.Timedelta(days=1)
                continue

            # Filter by date if partial month
            if start_int != int(month_start.strftime("%Y%m%d")) or end_int != int(
                month_end.strftime("%Y%m%d")
            ):
                import pyarrow.compute as pc
                mask = (pc.field("trade_date") >= start_int) & (pc.field("trade_date") <= end_int)
                table = table.filter(mask)

            if table.num_rows == 0:
                month_start = month_end + pd.Timedelta(days=1)
                continue

            for source, cols_needed in active_factor_sources:
                fdir = factor_dirs[source]
                ffiles = _list_month_files(fdir, year, month)
                if not ffiles:
                    null_arr = pa.nulls(table.num_rows, type=pa.float32())
                    for col in cols_needed:
                        table = table.append_column(col, null_arr)
                    continue
                ftable = _read_table(ffiles, base_cols + cols_needed)
                if ftable is not None:
                    table = table.join(ftable, keys=base_cols, join_type="left outer")
                del ftable

            # Label
            label_files = _list_month_files(labels_dir, year, month)
            if label_files:
                ltable = _read_table(label_files, base_cols + [self.label_name])
                if ltable is not None:
                    table = table.join(ltable, keys=base_cols, join_type="left outer")
                del ltable
            else:
                table = table.append_column(
                    self.label_name, pa.nulls(table.num_rows, type=pa.float32())
                )

            all_tables.append(table)
            del table
            gc.collect()

            month_start = month_end + pd.Timedelta(days=1)

        if not all_tables:
            raise ValueError(f"No data in range {start} ~ {end}")

        # Unify schema: cast all floating-point columns to float32 to avoid
        # concat failures when different months have mixed float32/float64.
        def _unify_schema(table: pa.Table) -> pa.Table:
            fields = []
            for field in table.schema:
                if pa.types.is_float64(field.type):
                    fields.append(pa.field(field.name, pa.float32(), nullable=field.nullable))
                else:
                    fields.append(field)
            new_schema = pa.schema(fields)
            return table.cast(new_schema)

        all_tables = [_unify_schema(t) for t in all_tables]

        # Concatenate all months
        if len(all_tables) == 1:
            full_table = all_tables[0]
        else:
            full_table = pa.concat_tables(all_tables)
        del all_tables
        gc.collect()

        # Convert to pandas in one shot
        df = full_table.to_pandas()
        del full_table
        gc.collect()

        df["datetime"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d")
        df["instrument"] = df["ts_code"].astype("string")
        df.drop(columns=["trade_date", "ts_code"], inplace=True)
        df = df.dropna(subset=["datetime", "instrument"])
        df = df.drop_duplicates(subset=["datetime", "instrument"], keep="last")
        df = df.set_index(["datetime", "instrument"]).sort_index()

        # Ensure all feature columns exist and are float32
        for col in feature_cols_list:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

        if self.label_name not in df.columns:
            df[self.label_name] = np.nan
        df[self.label_name] = pd.to_numeric(df[self.label_name], errors="coerce").astype("float32")

        y = df[self.label_name]
        x = df[feature_cols_list].copy()
        del df
        gc.collect()

        x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        if dropna_label:
            valid = y.notna()
            x = x.loc[valid]
            y = y.loc[valid]

        return x, y

    # ------------------------------------------------------------------
    # Per-window training
    # ------------------------------------------------------------------

    def _compute_sample_weights(self, y: np.ndarray) -> np.ndarray | None:
        """Compute sample weights based on label rank.

        For ``weighted`` loss: high-label samples get higher weight so the model
        focuses more on identifying winning stocks.

        weight = min_weight + (label - label_min)^power, normalized to mean=1.
        """
        if self.loss_type != "weighted":
            return None
        y_min = float(np.nanmin(y))
        y_max = float(np.nanmax(y))
        if y_max - y_min < 1e-9:
            return None
        y_norm = (y - y_min) / (y_max - y_min)
        w = self.label_weight_min + np.power(y_norm, self.label_weight_power) * (1.0 - self.label_weight_min)
        w = w / np.mean(w)
        return w.astype(np.float32)

    def _model_params_for_device(self, device_name: str) -> dict[str, Any]:
        params = dict(self.model_params)
        params.setdefault("tree_method", "hist")
        if device_name == "gpu":
            params["device"] = "cuda"
        else:
            params["device"] = "cpu"

        # Inject loss type
        if self.loss_type == "quantile":
            params["objective"] = "reg:quantileerror"
            params["quantile_alpha"] = self.quantile_alpha
        elif self.loss_type == "weighted":
            # sample_weight is passed at fit time; keep default objective
            params.pop("objective", None)

        return params

    @staticmethod
    def _gpu_available() -> bool:
        try:
            import pynvml

            pynvml.nvmlInit()
            return pynvml.nvmlDeviceGetCount() > 0
        except Exception:
            return False

    def _train_one(self, x_train, y_train, x_valid, y_valid, verbose: int = 50,
                   sample_weight_train=None, sample_weight_valid=None) -> XGBRegressor:
        fit_attempts = []
        if self.prefer_gpu and self._gpu_available():
            fit_attempts.append(("gpu", self._model_params_for_device("gpu")))
        fit_attempts.append(("cpu", self._model_params_for_device("cpu")))

        last_error: Exception | None = None
        for device_name, params in fit_attempts:
            try:
                model = XGBRegressor(**params)
                fit_kwargs = {
                    "eval_set": [(x_valid, y_valid)],
                    "verbose": verbose,
                }
                if sample_weight_train is not None:
                    fit_kwargs["sample_weight"] = sample_weight_train
                if sample_weight_valid is not None:
                    fit_kwargs["sample_weight_eval_set"] = [sample_weight_valid]
                model.fit(x_train, y_train, **fit_kwargs)
                return model
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

    def train_window(self, window: WindowSpec, verbose: int = 50) -> XGBRegressor:
        """Train one window by loading train+valid in a single read.

        Train and valid are loaded together (they are contiguous in time) and
        then split by date, halving the IO and peak memory vs. loading them
        separately.
        """

        x_all, y_all = self._load_range_xy(window.train_start, window.valid_end)

        dates = x_all.index.get_level_values("datetime")
        valid_start_ts = pd.Timestamp(window.valid_start)
        train_mask = dates < valid_start_ts

        x_train = x_all[train_mask]
        y_train = y_all[train_mask]
        x_valid = x_all[~train_mask]
        y_valid = y_all[~train_mask]

        del x_all, y_all, dates, train_mask
        gc.collect()

        print(f"  train: {x_train.shape[0]:,} rows, valid: {x_valid.shape[0]:,} rows", flush=True)

        # Compute sample weights for weighted loss
        sw_train = self._compute_sample_weights(y_train)
        sw_valid = self._compute_sample_weights(y_valid)
        if sw_train is not None:
            print(f"  weighted loss: power={self.label_weight_power}, "
                  f"min_weight={self.label_weight_min}, "
                  f"weight_range=[{sw_train.min():.3f}, {sw_train.max():.3f}]", flush=True)

        # Convert to numpy to reduce memory overhead during training
        x_train_np = x_train.values
        y_train_np = y_train.values
        x_valid_np = x_valid.values
        y_valid_np = y_valid.values
        del x_train, y_train, x_valid, y_valid
        gc.collect()
        self._malloc_trim()
        self._log_mem("  before train (numpy)")

        model = self._train_one(
            x_train_np, y_train_np, x_valid_np, y_valid_np,
            sample_weight_train=sw_train,
            sample_weight_valid=sw_valid,
            verbose=verbose,
        )

        del x_train_np, y_train_np, x_valid_np, y_valid_np, sw_train, sw_valid
        gc.collect()
        self._malloc_trim()

        return model

    def predict_window(self, window: WindowSpec, model: XGBRegressor) -> pd.DataFrame:
        """Predict the test segment of one window.

        Test data is loaded in quarterly chunks to keep peak memory low.
        This is critical for later windows with more listed stocks.
        """

        model.set_params(n_jobs=1)

        test_start = pd.Timestamp(window.test_start)
        test_end = pd.Timestamp(window.test_end)

        chunks = []
        chunk_start = test_start
        while chunk_start <= test_end:
            chunk_end = min(chunk_start + pd.DateOffset(months=3) - pd.Timedelta(days=1), test_end)

            x_chunk, y_chunk = self._load_range_xy(
                chunk_start.strftime("%Y-%m-%d"),
                chunk_end.strftime("%Y-%m-%d"),
                dropna_label=False,
            )

            valid = y_chunk.notna()
            x_valid = x_chunk.loc[valid]
            y_valid = y_chunk.loc[valid]
            del x_chunk, y_chunk
            gc.collect()

            if len(x_valid) > 0:
                pred_values = model.predict(x_valid)
                chunk_df = pd.DataFrame(
                    {"pred": pred_values, self.label_name: y_valid.values},
                    index=x_valid.index,
                )
                chunks.append(chunk_df)
                del pred_values, chunk_df

            del x_valid, y_valid
            gc.collect()

            chunk_start = chunk_end + pd.Timedelta(days=1)

        if not chunks:
            return pd.DataFrame(
                columns=["pred", self.label_name],
                index=pd.MultiIndex.from_arrays([[], []], names=["datetime", "instrument"]),
            )

        result = pd.concat(chunks).sort_index()
        del chunks
        gc.collect()

        return result

    def _predict_from_file(self, window: WindowSpec, model_path: Path) -> pd.DataFrame:
        """Predict test window using a subprocess to guarantee memory cleanup.

        Training leaves glibc malloc arenas and XGBoost internal buffers that
        may not be returned to the OS.  Running prediction in a child process
        ensures all memory is released when the child exits, keeping the
        parent process footprint low for the next training window.
        """

        import multiprocessing as mp

        pred_path = self.prediction_dir / f"{window.name}_pred_tmp.parquet"
        pred_path.parent.mkdir(parents=True, exist_ok=True)

        ctx = mp.get_context("spawn")
        proc = ctx.Process(
            target=_predict_worker,
            args=(
                str(self.loader_config_path) if self.loader_config_path else "",
                str(model_path),
                window.test_start,
                window.test_end,
                self.label_name,
                str(pred_path),
            ),
        )
        proc.start()
        proc.join()

        if proc.exitcode != 0:
            raise RuntimeError(
                f"Prediction subprocess failed with exit code {proc.exitcode}"
            )

        pred_df = pd.read_parquet(pred_path)
        pred_path.unlink(missing_ok=True)
        if "datetime" in pred_df.columns and "instrument" in pred_df.columns:
            pred_df = pred_df.set_index(["datetime", "instrument"]).sort_index()
        gc.collect()

        return pred_df

    # ------------------------------------------------------------------
    # Full walk-forward run
    # ------------------------------------------------------------------

    def run(
        self,
        verbose: int = 50,
        save: bool = True,
        resume: bool = True,
    ) -> list[WindowResult]:
        """Run the full walk-forward pipeline.

        Parameters
        ----------
        verbose
            XGBoost verbosity level (0 = silent, 50 = print every 50 rounds).
        save
            Whether to save models and predictions to disk.
        resume
            If True, skip windows whose model and prediction already exist
            on disk.
        """

        if not self.windows:
            self.generate_windows()

        print(f"[WalkForward] {len(self.windows)} windows generated "
              f"(mode={self.mode}, window={self.train_window_years}y, step={self.step_years}y)",
              flush=True)

        if save:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.prediction_dir.mkdir(parents=True, exist_ok=True)

        results: list[WindowResult] = []
        failed: list[tuple[WindowSpec, str]] = []

        for i, window in enumerate(self.windows):
            print(f"\n[WalkForward] Window {i + 1}/{len(self.windows)}: {window.name}", flush=True)
            self._log_mem(f"  start of window {window.name}")

            model_path = self.model_dir / f"{window.name}.json"
            pred_path = self.prediction_dir / f"{window.name}_pred.parquet"

            result = WindowResult(window=window, model_path=model_path, pred_path=pred_path)

            pred_df: pd.DataFrame | None = None

            try:
                if resume and model_path.exists() and pred_path.exists():
                    print(f"  Skipping (already done): {window.name}", flush=True)
                    pred_df = pd.read_parquet(pred_path)
                    result.metrics = self._evaluate_pred(pred_df)
                    results.append(result)
                    continue

                if not (resume and model_path.exists()):
                    import multiprocessing as mp
                    import json

                    ctx = mp.get_context("spawn")
                    proc = ctx.Process(
                        target=_train_worker,
                        args=(
                            str(self.config_path) if self.config_path else "",
                            str(self.loader_config_path) if self.loader_config_path else "",
                            window.idx,
                            window.train_start,
                            window.train_end,
                            window.valid_start,
                            window.valid_end,
                            str(model_path),
                            self.loss_type,
                            self.quantile_alpha,
                            self.label_weight_power,
                            self.label_weight_min,
                            self.label_name,
                            json.dumps(self.model_params),
                            self.prefer_gpu,
                        ),
                    )
                    proc.start()
                    proc.join()
                    if proc.exitcode != 0:
                        raise RuntimeError(
                            f"Training subprocess failed with exit code {proc.exitcode}"
                        )
                    self._malloc_trim()
                    self._log_mem(f"  after train subprocess")

                from xgboost import XGBRegressor
                model = XGBRegressor()
                model.load_model(str(model_path))

                pred_df = self.predict_window(window, model)
                del model
                gc.collect()
                self._malloc_trim()

                if save:
                    pred_df.to_parquet(pred_path)
                    print(f"  Prediction saved: {pred_path}", flush=True)

                result.metrics = self._evaluate_pred(pred_df)
                print(f"  IC={result.metrics.get('ic', float('nan')):.4f}, "
                      f"RankIC={result.metrics.get('rank_ic', float('nan')):.4f}",
                      flush=True)

                results.append(result)

            except Exception as exc:
                import traceback
                err_msg = f"{type(exc).__name__}: {exc}"
                print(f"  ERROR in {window.name}: {err_msg}", flush=True)
                traceback.print_exc()
                failed.append((window, err_msg))

            finally:
                if pred_df is not None:
                    del pred_df
                gc.collect()
                self._log_mem(f"  end of window {window.name}")

        if save:
            self._save_summary(results)

        if failed:
            print(f"\n[WalkForward] {len(failed)} window(s) failed:", flush=True)
            for w, err in failed:
                print(f"  {w.name}: {err}", flush=True)

        return results

    @staticmethod
    def _log_mem(label: str = "") -> None:
        """Log current RSS memory usage for debugging."""
        try:
            import resource
            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            cur_kb = 0
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        cur_kb = int(line.split()[1])
                        break
            print(f"  [mem] {label}: cur_rss={cur_kb/1024:.1f} MB, max_rss={rss_kb/1024:.1f} MB", flush=True)
        except Exception:
            pass

    @staticmethod
    def _malloc_trim() -> None:
        """Try to return freed memory to the OS."""
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Evaluation & summary
    # ------------------------------------------------------------------

    def _evaluate_pred(self, pred_df: pd.DataFrame) -> dict[str, float]:
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

    def _save_summary(self, results: list[WindowResult]) -> None:
        summary = []
        for r in results:
            row = {
                "window": r.window.name,
                "train_start": r.window.train_start,
                "train_end": r.window.train_end,
                "test_start": r.window.test_start,
                "test_end": r.window.test_end,
                **r.metrics,
            }
            if r.model_path is not None:
                row["model_path"] = str(r.model_path)
            if r.pred_path is not None:
                row["pred_path"] = str(r.pred_path)
            summary.append(row)

        summary_path = self.prediction_dir / "walk_forward_summary.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n[WalkForward] Summary saved: {summary_path}", flush=True)

    def collect_all_predictions(self) -> pd.DataFrame:
        """Concatenate all window predictions into one DataFrame.

        Useful for downstream backtesting.
        """

        if not self.windows:
            self.generate_windows()

        all_preds = []
        for window in self.windows:
            pred_path = self.prediction_dir / f"{window.name}_pred.parquet"
            if pred_path.exists():
                all_preds.append(pd.read_parquet(pred_path))

        if not all_preds:
            raise FileNotFoundError("No prediction files found")

        result = pd.concat(all_preds).sort_index()
        return result[~result.index.duplicated(keep="last")]


def _predict_worker(
    loader_config_path: str,
    model_path: str,
    test_start: str,
    test_end: str,
    label_name: str,
    output_path: str,
) -> None:
    """Subprocess worker for prediction.

    Uses direct file-path reads (not ``pyarrow.dataset``) so we never scan
    the full parquet tree.  Each month we list only the relevant
    ``year=YYYY/month=MM/`` directories, which keeps metadata overhead
    proportional to the date range instead of the full dataset.

    Predictions are streamed to the output parquet file via
    ``ParquetWriter`` so peak memory stays at one month of data.
    """
    import gc
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from xgboost import XGBRegressor

    from utils import load_loader_config
    from utils.dataset_generator.parquet_loader import ParquetLoader as PL

    loader_cfg = load_loader_config(loader_config_path or None)
    feature_cols = list(loader_cfg.get("feature_cols") or [])
    daily_bars_dir = Path(loader_cfg["daily_bars_dir"])
    labels_dir = Path(loader_cfg["labels_dir"])

    factor_dirs = {
        "price_volume": Path(loader_cfg.get("price_volume_factors_dir", "")),
        "moneyflow": Path(loader_cfg.get("moneyflow_factors_dir", "")),
        "fundamental": Path(loader_cfg.get("fundamental_factors_dir", "")),
        "industry": Path(loader_cfg.get("industry_factors_dir", "")),
        "enhanced_alpha": Path(loader_cfg.get("enhanced_alpha_factors_dir", "")),
    }

    factor_sources = PL.FACTOR_SOURCES
    main_feature_cols = PL.MAIN_FEATURE_COLS
    base_cols = list(PL.BASE_REQUIRED_COLS)

    needed = set(feature_cols)
    main_cols_needed = [c for c in main_feature_cols if c in needed]

    # Pre-compute which factor sources we actually need
    active_factor_sources: list[tuple[str, list[str]]] = []
    for source, source_cols in factor_sources.items():
        cols_needed = [c for c in source_cols if c in needed]
        if not cols_needed:
            continue
        fdir = factor_dirs.get(source)
        if not fdir or not fdir.exists():
            continue
        active_factor_sources.append((source, cols_needed))

    def _list_month_files(base_dir: Path, year: int, month: int) -> list[Path]:
        """Return parquet files for one month dir, sorted by date."""
        month_dir = base_dir / f"year={year}" / f"month={month:02d}"
        if not month_dir.exists():
            return []
        return sorted(month_dir.glob("*.parquet"))

    def _read_files(files: list[Path], columns: list[str]) -> pd.DataFrame:
        """Read a list of parquet files with column selection."""
        if not files:
            return pd.DataFrame(columns=columns)
        tables = []
        for f in files:
            t = pq.read_table(f, columns=columns)
            tables.append(t)
        if len(tables) == 1:
            return tables[0].to_pandas()
        return pa.concat_tables(tables).to_pandas()

    def load_month(month_start: pd.Timestamp, month_end: pd.Timestamp) -> pd.DataFrame:
        year = month_start.year
        month = month_start.month
        start_int = int(month_start.strftime("%Y%m%d"))
        end_int = int(month_end.strftime("%Y%m%d"))

        main_files = _list_month_files(daily_bars_dir, year, month)
        if not main_files:
            return pd.DataFrame()

        df = _read_files(main_files, base_cols + main_cols_needed)

        # Filter by date range (in case month boundaries don't align)
        if start_int != int(month_start.replace(day=1).strftime("%Y%m%d")) or end_int != int(
            (month_start + pd.DateOffset(months=1) - pd.Timedelta(days=1)).strftime("%Y%m%d")
        ):
            mask = (df["trade_date"] >= start_int) & (df["trade_date"] <= end_int)
            df = df[mask]

        if df.empty:
            return pd.DataFrame()

        for source, cols_needed in active_factor_sources:
            fdir = factor_dirs[source]
            ffiles = _list_month_files(fdir, year, month)
            if not ffiles:
                for col in cols_needed:
                    df[col] = np.nan
                continue
            fdf = _read_files(ffiles, base_cols + cols_needed)
            df = df.merge(fdf, on=base_cols, how="left")
            del fdf
            gc.collect()

        label_files = _list_month_files(labels_dir, year, month)
        if label_files:
            label_df = _read_files(label_files, base_cols + [label_name])
            df = df.merge(label_df, on=base_cols, how="left")
            del label_df
            gc.collect()

        df["datetime"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d")
        df["instrument"] = df["ts_code"].astype(str)
        df.drop(columns=["trade_date", "ts_code"], inplace=True)
        df = df.dropna(subset=["datetime", "instrument"])
        df = df.drop_duplicates(subset=["datetime", "instrument"], keep="last")
        df = df.set_index(["datetime", "instrument"]).sort_index()

        for col in feature_cols + [label_name]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")

        return df

    model = XGBRegressor()
    model.load_model(model_path)
    model.set_params(n_jobs=1)

    test_start_ts = pd.Timestamp(test_start)
    test_end_ts = pd.Timestamp(test_end)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer: pq.ParquetWriter | None = None
    schema = pa.schema(
        [
            ("datetime", pa.timestamp("ns")),
            ("instrument", pa.string()),
            ("pred", pa.float32()),
            (label_name, pa.float32()),
        ]
    )

    month_start = test_start_ts.replace(day=1)
    while month_start <= test_end_ts:
        month_end = month_start + pd.DateOffset(months=1) - pd.Timedelta(days=1)
        actual_start = max(month_start, test_start_ts)
        actual_end = min(month_end, test_end_ts)

        df = load_month(actual_start, actual_end)

        if not df.empty:
            y = df[label_name]
            x = df[feature_cols]
            del df
            gc.collect()

            valid = y.notna()
            x_valid = x.loc[valid]
            y_valid = y.loc[valid]
            del x, y
            gc.collect()

            x_valid = x_valid.replace([np.inf, -np.inf], np.nan).fillna(0.0)

            pred_values = model.predict(x_valid)
            pred_df = pd.DataFrame(
                {"pred": pred_values.astype("float32"), label_name: y_valid.values},
                index=x_valid.index,
            )
            del pred_values, x_valid, y_valid
            gc.collect()

            pred_df_reset = pred_df.reset_index()
            table = pa.Table.from_pandas(pred_df_reset, schema=schema, preserve_index=False)
            del pred_df, pred_df_reset

            if writer is None:
                writer = pq.ParquetWriter(output_path, schema=schema)
            writer.write_table(table)
            del table
            gc.collect()
        else:
            del df
            gc.collect()

        month_start = month_end + pd.Timedelta(days=1)

    if writer is not None:
        writer.close()
    del model
    gc.collect()

    if not output_path.exists():
        pd.DataFrame(
            columns=["datetime", "instrument", "pred", label_name]
        ).to_parquet(output_path)


def _train_worker(
    config_path: str,
    loader_config_path: str,
    window_idx: int,
    train_start: str,
    train_end: str,
    valid_start: str,
    valid_end: str,
    model_path: str,
    loss_type: str,
    quantile_alpha: float,
    label_weight_power: float,
    label_weight_min: float,
    label_name: str,
    model_params_json: str,
    prefer_gpu: bool,
) -> None:
    """Subprocess worker for training one window.

    Runs in an isolated process so all training memory is released when
    the worker exits, keeping the parent process lean for prediction.
    """
    import json
    import gc

    import numpy as np
    from xgboost import XGBRegressor

    trainer = WalkForwardTrainer(
        config_path=config_path or None,
        loader_config_path=loader_config_path or None,
    )
    # Override loss config from args (parent already parsed them)
    trainer.loss_type = loss_type
    trainer.quantile_alpha = quantile_alpha
    trainer.label_weight_power = label_weight_power
    trainer.label_weight_min = label_weight_min
    trainer.label_name = label_name
    trainer.prefer_gpu = prefer_gpu

    # Override model params
    import json as _json
    params = _json.loads(model_params_json)
    trainer.model_params.update(params)

    # Load train+valid data
    x_all, y_all = trainer._load_range_xy(train_start, valid_end)

    dates = x_all.index.get_level_values("datetime")
    valid_start_ts = pd.Timestamp(valid_start)
    train_mask = dates < valid_start_ts

    x_train = x_all[train_mask]
    y_train = y_all[train_mask]
    x_valid = x_all[~train_mask]
    y_valid = y_all[~train_mask]
    del x_all, y_all, dates, train_mask
    gc.collect()
    WalkForwardTrainer._malloc_trim()

    # Sample weights
    sw_train = trainer._compute_sample_weights(y_train)
    sw_valid = trainer._compute_sample_weights(y_valid)

    # Convert to numpy
    x_train_np = x_train.values
    y_train_np = y_train.values
    x_valid_np = x_valid.values
    y_valid_np = y_valid.values
    del x_train, y_train, x_valid, y_valid
    gc.collect()
    WalkForwardTrainer._malloc_trim()

    # Train
    model = trainer._train_one(
        x_train_np, y_train_np, x_valid_np, y_valid_np,
        sample_weight_train=sw_train,
        sample_weight_valid=sw_valid,
        verbose=50,
    )

    model.save_model(model_path)
    print(f"  Model saved: {model_path}", flush=True)


__all__ = ["WalkForwardTrainer", "WindowSpec", "WindowResult"]
