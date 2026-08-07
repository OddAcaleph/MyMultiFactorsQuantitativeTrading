"""A-share simple backtester with executable-price simulation.

Compared with :mod:`preparatory_backtester`, this module performs a more
realistic daily portfolio simulation from parquet predictions and OHLCV bars:

* buy orders can be blocked on limit-up / limit-down days;
* share quantities are rounded to board lots (100 shares by default);
* orders can be executed at daily open or close;
* configurable slippage, commission / stamp-duty style costs and minimum fee;
* T+1 is enforced by lot, so shares bought today cannot be sold today;
* benchmark returns are read from qlib-format index data under ``features``;
* Sharpe, alpha, beta and other common performance metrics are exported.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import PROJECT_ROOT, load_simple_backtester_config, load_trainer_config


class SimpleBacktester:
    """Run a daily TopK-style A-share backtest with cash and share accounting.

    Parameters are loaded from ``conf/simple_backtester_config.json`` by default.
    Useful keys are::

        {
          "price_path": ".../data/processd_data/wide_table_daily_bars",
          "benchmark": "000300.SH",
          "benchmark_path": "/opt/tiger/qyd/qlib_data_cn/features",
          "exchange_kwargs": {
            "deal_price": "open",       # or "close"
            "open_cost": 0.0005,
            "close_cost": 0.0015,
            "slippage": 0.001,
            "min_cost": 5.0,
            "limit_threshold": 0.095,
            "forbid_buy_limit_up": true,
            "forbid_buy_limit_down": true,
            "forbid_sell_limit_down": true,
            "lot_size": 100
          }
        }
    """

    DEFAULT_PRICE_PATH = PROJECT_ROOT / "data" / "processd_data" / "wide_table_daily_bars"
    DEFAULT_BENCHMARK_PATH = Path("/opt/tiger/qyd/qlib_data_cn/features")
    _TIMESTAMP_PLACEHOLDER = "${timestamp}"

    @staticmethod
    def _resolve_output_dir(value: str | Path) -> str:
        """Replace ``${timestamp}`` placeholder with YYMMDD-HHMM."""
        s = str(value)
        if SimpleBacktester._TIMESTAMP_PLACEHOLDER in s:
            from datetime import datetime
            s = s.replace(SimpleBacktester._TIMESTAMP_PLACEHOLDER, datetime.now().strftime("%y%m%d-%H%M"))
        return s

    def __init__(self, config_path: str | Path | None = None, **overrides: Any) -> None:
        self.config_path = config_path
        self.config = self._apply_overrides(load_simple_backtester_config(config_path), overrides)

        self.trainer_config_path = self.config.get("trainer_config_path")
        self.trainer_config = load_trainer_config(self.trainer_config_path) if self.trainer_config_path else {}

        self.prediction_path = Path(self.config["prediction_path"])
        self.price_path = Path(self.config.get("price_path", self.DEFAULT_PRICE_PATH))
        self.benchmark_path = Path(self.config.get("benchmark_path", self.DEFAULT_BENCHMARK_PATH))
        self.benchmark_lookback_days = int(self.config.get("benchmark_lookback_days", 31))
        self.output_dir = Path(self._resolve_output_dir(self.config.get("output_dir", PROJECT_ROOT / "outputs" / "simple_backtest")))
        self.account = float(self.config.get("account", 1_000_000.0))
        self.freq = self.config.get("freq", "day")

        configured_benchmark = self.config.get("benchmark")
        # The simple backtester deliberately avoids market_mean as benchmark.
        self.benchmark = "000300.SH" if configured_benchmark in (None, "", "market_mean") else str(configured_benchmark)

        self.strategy_config = dict(self.config.get("strategy", {}))
        self.exchange_kwargs = dict(self.config.get("exchange_kwargs", {}))
        self.columns = dict(self.config.get("columns", {}))
        self.score_col = self.columns.get("score", "pred")
        self.save_positions = bool(self.config.get("save_positions", True))
        # Industry constraint
        self.max_industry_weight = float(self.strategy_config.get("max_industry_weight", 1.0))
        self.max_industry_count = int(self.strategy_config.get("max_industry_count", 0))  # 0 = no limit
        self.industry_data_path = Path(self.config.get("industry_data_path", ""))
        self._industry_map: dict[str, str] | None = None  # instrument -> l1_name

        # Industry-stratified sampling
        self.industry_stratified = bool(self.strategy_config.get("industry_stratified", False))
        self.industry_stratify_method = str(self.strategy_config.get("industry_stratify_method", "equal"))  # equal or proportional

        # Drop criteria: "score" (default) = sell lowest predicted score, "return" = sell worst holding return
        self.drop_criteria = self.strategy_config.get("drop_criteria", "score")

        # Stock pool filters
        self.filter_st = bool(self.strategy_config.get("filter_st", False))
        self.filter_suspend = bool(self.strategy_config.get("filter_suspend", False))
        self.filter_new_stock_days = int(self.strategy_config.get("filter_new_stock_days", 0))  # 0 = no filter
        self.min_avg_amount_20d = float(self.strategy_config.get("min_avg_amount_20d", 0))  # 0 = no filter
        self.max_buy_pct_of_avg_amount = float(self.strategy_config.get("max_buy_pct_of_avg_amount", 0))  # 0 = no limit
        self.max_single_weight = float(self.strategy_config.get("max_single_weight", 0))  # 0 = no limit
        self._list_dates: dict[str, pd.Timestamp] | None = None  # instrument -> list date
        self._avg_amount_20d: pd.Series | None = None  # (datetime, instrument) -> 20d avg amount
        self._vol_series: pd.Series | None = None  # (datetime, instrument) -> rolling vol

        # Volatility-weighted position sizing
        self.vol_weight_enabled = bool(self.strategy_config.get("vol_weight_enabled", False))
        self.vol_lookback_days = int(self.strategy_config.get("vol_lookback_days", 20))
        self.vol_weight_power = float(self.strategy_config.get("vol_weight_power", 1.0))

        # Dynamic position sizing (allows going to cash)
        self.dynamic_position_enabled = bool(self.strategy_config.get("dynamic_position_enabled", False))
        self.dyn_pos_method = str(self.strategy_config.get("dyn_pos_method", "score_threshold"))  # score_threshold / dispersion / target_vol
        self.dyn_pos_score_threshold = float(self.strategy_config.get("dyn_pos_score_threshold", 0.5))
        self.dyn_pos_min_position = float(self.strategy_config.get("dyn_pos_min_position", 0.0))
        self.dyn_pos_max_position = float(self.strategy_config.get("dyn_pos_max_position", 1.0))
        self.dyn_pos_topk_fraction = float(self.strategy_config.get("dyn_pos_topk_fraction", 0.1))
        # Target volatility method params
        self.dyn_pos_target_vol = float(self.strategy_config.get("dyn_pos_target_vol", 0.15))  # 15% annualized
        self.dyn_pos_vol_lookback = int(self.strategy_config.get("dyn_pos_vol_lookback", 60))  # trading days
        self.dyn_pos_vol_smooth = int(self.strategy_config.get("dyn_pos_vol_smooth", 5))  # smoothing days
        # Dispersion method params
        self.dyn_pos_dispersion_top_frac = float(self.strategy_config.get("dyn_pos_dispersion_top_frac", 0.1))
        self.dyn_pos_dispersion_bottom_frac = float(self.strategy_config.get("dyn_pos_dispersion_bottom_frac", 0.1))
        self.dyn_pos_dispersion_low = float(self.strategy_config.get("dyn_pos_dispersion_low", 0.1))
        self.dyn_pos_dispersion_high = float(self.strategy_config.get("dyn_pos_dispersion_high", 0.3))
        self.dyn_pos_dispersion_smooth_days = int(self.strategy_config.get("dyn_pos_dispersion_smooth_days", 5))
        # Relative dispersion (percentile-based) params
        self.dyn_pos_dispersion_lookback_days = int(self.strategy_config.get("dyn_pos_dispersion_lookback_days", 60))
        self.dyn_pos_dispersion_low_pct = float(self.strategy_config.get("dyn_pos_dispersion_low_pct", 0.3))
        self.dyn_pos_dispersion_high_pct = float(self.strategy_config.get("dyn_pos_dispersion_high_pct", 0.7))
        # Predictions are normally produced after the signal date is closed, so
        # the first executable bar is the next trading day.  Using signal t to
        # trade at t open/close is look-ahead and can materially inflate returns.
        self.signal_delay = int(self.config.get("signal_delay", self.strategy_config.get("signal_delay", 1)))
        if self.signal_delay < 0:
            raise ValueError("signal_delay must be non-negative")
        self.price_adjustment = str(self.config.get("price_adjustment", "qfq")).lower()
        if self.price_adjustment not in {"qfq", "none", "raw"}:
            raise ValueError("price_adjustment must be one of: qfq, none, raw")

        self.deal_price_col = str(self.exchange_kwargs.get("deal_price", "close")).lower()
        if self.deal_price_col not in {"open", "close"}:
            raise ValueError("exchange_kwargs.deal_price must be either 'open' or 'close'")

        self.open_cost = float(self.exchange_kwargs.get("open_cost", 0.0))
        self.close_cost = float(self.exchange_kwargs.get("close_cost", 0.0))
        self.min_cost = float(self.exchange_kwargs.get("min_cost", 0.0))
        self.slippage = float(self.exchange_kwargs.get("slippage", self.exchange_kwargs.get("impact_cost", 0.0)))
        self.limit_threshold = float(self.exchange_kwargs.get("limit_threshold", 0.095))
        self.board_specific_limits = bool(self.exchange_kwargs.get("board_specific_limits", False))
        self.lot_size = int(self.exchange_kwargs.get("lot_size", 100))

        forbid_all = bool(self.strategy_config.get("forbid_all_trade_at_limit", False))
        self.forbid_buy_limit_up = forbid_all or bool(self.exchange_kwargs.get("forbid_buy_limit_up", True))
        self.forbid_buy_limit_down = forbid_all or bool(self.exchange_kwargs.get("forbid_buy_limit_down", True))
        self.forbid_sell_limit_down = forbid_all or bool(self.exchange_kwargs.get("forbid_sell_limit_down", False))

        self.start_time, self.end_time = self._resolve_oos_period()

        # Dynamic position state
        self._dyn_pos_dispersion_history: list[float] = []  # for smoothing

    @staticmethod
    def _apply_overrides(config: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
        merged = dict(config)
        for key, value in overrides.items():
            if value is not None:
                merged[key] = value
        return merged

    def _resolve_oos_period(self) -> tuple[pd.Timestamp, pd.Timestamp]:
        start_time = self.config.get("start_time")
        end_time = self.config.get("end_time")

        if start_time is None or end_time is None:
            test_segment = self.trainer_config.get("segments", {}).get("test")
            if not test_segment:
                raise ValueError("Backtester start/end are missing and trainer config has no test segment")
            start_time = start_time or test_segment[0]
            end_time = end_time or test_segment[1]

        return pd.Timestamp(start_time), pd.Timestamp(end_time)

    def run(
        self,
        save: bool = True,
        pred_df: pd.DataFrame | None = None,
        price_df: pd.DataFrame | None = None,
        benchmark_returns: pd.Series | None = None,
    ) -> dict[str, Any]:
        """Run the backtest and optionally save artifacts."""

        topk = int(self.strategy_config.get("topk", 50))
        n_drop = int(self.strategy_config.get("n_drop", 5))
        if n_drop > topk:
            return self._skipped_result(topk=topk, n_drop=n_drop)

        pred_df = self.load_predictions() if pred_df is None else pred_df
        price_df = self.load_prices() if price_df is None else price_df
        benchmark_returns = self.load_benchmark_returns() if benchmark_returns is None else benchmark_returns
        report_df, positions_df, trades_df = self._simulate(pred_df=pred_df, price_df=price_df, benchmark_returns=benchmark_returns)
        analysis = self._analyze(report_df=report_df, trades_df=trades_df)

        if save:
            self.save_outputs(report_df=report_df, positions_df=positions_df, trades_df=trades_df, analysis=analysis)

        return {"report": report_df, "positions": positions_df, "trades": trades_df, "analysis": analysis}

    def _skipped_result(self, topk: int, n_drop: int) -> dict[str, Any]:
        """Return an empty result when the strategy parameters should be skipped."""

        skip_reason = f"strategy.n_drop ({n_drop}) is greater than strategy.topk ({topk}); skip this backtest"
        return {
            "report": pd.DataFrame(),
            "positions": pd.DataFrame(),
            "trades": pd.DataFrame(),
            "analysis": {
                "summary": {
                    "skipped": True,
                    "skip_reason": skip_reason,
                    "topk": int(topk),
                    "n_drop": int(n_drop),
                },
                "config": self.config,
            },
        }

    def load_predictions(self) -> pd.DataFrame:
        """Load OOS prediction scores with MultiIndex(datetime, instrument)."""

        if not self.prediction_path.exists():
            raise FileNotFoundError(f"Prediction file does not exist: {self.prediction_path}")

        pred_df = pd.read_parquet(self.prediction_path)
        if not isinstance(pred_df.index, pd.MultiIndex):
            raise ValueError("Prediction parquet must use MultiIndex(datetime, instrument)")
        if pred_df.index.names != ["datetime", "instrument"]:
            pred_df.index = pred_df.index.set_names(["datetime", "instrument"])
        if self.score_col not in pred_df.columns:
            raise KeyError(f"Prediction file missing required score column: {self.score_col}")

        dates = pd.to_datetime(pred_df.index.get_level_values("datetime"))
        mask = (dates >= self.start_time) & (dates <= self.end_time)
        pred_df = pred_df.loc[mask, [self.score_col]]
        pred_df = pred_df.replace([np.inf, -np.inf], np.nan).dropna(subset=[self.score_col])
        return pred_df.sort_index()

    def load_prices(self) -> pd.DataFrame:
        """Load daily OHLCV bars needed for execution and mark-to-market."""

        if not self.price_path.exists():
            raise FileNotFoundError(f"Price path does not exist: {self.price_path}")

        # For liquidity rolling we need ~60 days of lookback before start_time
        lookback_days = 0
        if self.min_avg_amount_20d > 0 or self.max_buy_pct_of_avg_amount > 0:
            lookback_days = max(lookback_days, 60)
        if self.vol_weight_enabled:
            lookback_days = max(lookback_days, self.vol_lookback_days * 2 + 10)
        load_start = self.start_time - pd.Timedelta(days=lookback_days)

        files = self._parquet_files_for_period(self.price_path, start=load_start)
        if not files:
            raise FileNotFoundError(
                f"No daily bar parquet files found in {self.price_path} for {self.start_time.date()} ~ {self.end_time.date()}"
            )

        frames = []
        required = ["trade_date", "ts_code", "open", "close", "pre_close", "pct_chg", "vol", "amount"]
        optional = ["adj_factor"]
        # Load filter-related columns if needed
        filter_cols = []
        if self.filter_st:
            filter_cols += ["st_stock", "star_st_stock"]
        if self.filter_suspend:
            filter_cols.append("is_suspect")
        for file_path in files:
            frame = pd.read_parquet(file_path)
            missing = [col for col in required if col not in frame.columns]
            if missing:
                raise KeyError(f"Daily bar file {file_path} missing required columns: {missing}")
            cols_to_load = required + [col for col in optional if col in frame.columns] + [col for col in filter_cols if col in frame.columns]
            frames.append(frame[cols_to_load])

        price_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=required)
        price_df = price_df.rename(columns={"trade_date": "datetime", "ts_code": "instrument"})
        price_df["datetime"] = pd.to_datetime(price_df["datetime"].astype(str), format="%Y%m%d", errors="coerce")
        price_df = price_df.dropna(subset=["datetime", "instrument", "open", "close"])
        numeric_cols = ["open", "close", "pre_close", "pct_chg", "vol", "amount", "adj_factor"]
        if self.filter_st:
            numeric_cols += ["st_stock", "star_st_stock"]
        if self.filter_suspend:
            numeric_cols.append("is_suspect")
        for col in numeric_cols:
            if col not in price_df.columns:
                continue
            price_df[col] = pd.to_numeric(price_df[col], errors="coerce")
        price_df = price_df.set_index(["datetime", "instrument"]).sort_index()

        # Pre-compute list dates (needs full history, loaded separately for efficiency)
        if self.filter_new_stock_days > 0:
            self._list_dates = self._load_list_dates()

        # Precompute 20d avg amount from the loaded price data (includes lookback)
        if self.min_avg_amount_20d > 0 or self.max_buy_pct_of_avg_amount > 0:
            amount_series = price_df["amount"].sort_index()
            avg_20d = amount_series.groupby(level="instrument").transform(
                lambda x: x.rolling(20, min_periods=10).mean()
            )
            self._avg_amount_20d = avg_20d

        # Precompute rolling volatility for vol-weighted position sizing
        if self.vol_weight_enabled:
            lookback = max(self.vol_lookback_days, 20)
            close_sorted = price_df["close"].sort_index()
            # Daily returns per instrument
            rets = close_sorted.groupby(level="instrument").pct_change()
            # Rolling std dev
            vol_series = rets.groupby(level="instrument").transform(
                lambda x: x.rolling(lookback, min_periods=max(5, lookback // 2)).std()
            )
            self._vol_series = vol_series
        else:
            self._vol_series = None

        # Trim to actual backtest period for the simulation
        mask = (price_df.index.get_level_values("datetime") >= self.start_time) & \
               (price_df.index.get_level_values("datetime") <= self.end_time)
        price_df = price_df.loc[mask].copy()

        return self._apply_price_adjustment(price_df)

    _list_dates_cache: dict[str, dict[str, pd.Timestamp]] = {}  # price_path -> list_dates

    def _load_list_dates(self) -> dict[str, pd.Timestamp]:
        """Load list dates (first appearance date) for all instruments.

        Priority:
        1. Pre-computed JSON file at ``price_path / ../list_dates.json``
        2. Full pyarrow dataset scan (memory-intensive, falls back to per-year scan)

        Results are cached by price_path so repeated backtests don't re-scan.
        """
        cache_key = str(self.price_path)
        if cache_key in self._list_dates_cache:
            return self._list_dates_cache[cache_key]

        import json

        # Try pre-computed JSON first (fastest, lowest memory)
        json_path = self.price_path.parent / "list_dates.json"
        if json_path.exists():
            with open(json_path) as f:
                raw = json.load(f)
            result = {k: pd.Timestamp(v) for k, v in raw.items()}
            self._list_dates_cache[cache_key] = result
            return result

        # Fall back to per-year scan to limit memory peak
        import pyarrow.dataset as ds

        all_min: dict[str, pd.Timestamp] = {}
        year_dirs = sorted(p for p in self.price_path.glob("year=*"))
        for year_dir in year_dirs:
            year_ds = ds.dataset(str(year_dir), format="parquet", partitioning=["month"])
            table = year_ds.to_table(columns=["trade_date", "ts_code"])
            df = table.to_pandas()
            df["trade_date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
            df = df.dropna(subset=["trade_date", "ts_code"])
            year_min = df.groupby("ts_code")["trade_date"].min()
            for code, date in year_min.items():
                if code not in all_min or date < all_min[code]:
                    all_min[code] = date
            del df, table, year_min

        self._list_dates_cache[cache_key] = all_min
        return all_min

    def _apply_price_adjustment(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Use 前复权价格 for return accounting when adj_factor is available.

        Tushare daily OHLC fields are raw prices.  If raw prices are used across
        ex-right / dividend adjustment dates, the share-accounting simulation can
        see artificial price jumps and report distorted returns/drawdowns.  We
        normalize each instrument's ``adj_factor`` by its first valid factor in
        the loaded backtest window, which removes corporate-action jumps without
        using future factors for the normalization base.
        """

        if self.price_adjustment in {"none", "raw"} or "adj_factor" not in price_df.columns:
            return price_df

        adjusted = price_df.copy()
        factors = adjusted["adj_factor"].replace([np.inf, -np.inf], np.nan)
        first_factor = factors.groupby(level="instrument").transform("first")
        ratio = factors / first_factor
        valid_ratio = ratio.replace([np.inf, -np.inf], np.nan)
        for col in ["open", "close", "pre_close"]:
            adjusted[col] = adjusted[col].where(valid_ratio.isna(), adjusted[col] * valid_ratio)
        return adjusted

    def _is_new_stock(self, instrument: str, trade_date: pd.Timestamp) -> bool:
        """Check if a stock is listed for fewer than filter_new_stock_days.

        If the instrument is not found in _list_dates, it means the stock
        existed before our lookback window and is definitely not new.
        """
        if self.filter_new_stock_days <= 0 or self._list_dates is None:
            return False
        list_date = self._list_dates.get(instrument)
        if list_date is None:
            return False
        days_listed = (trade_date - list_date).days
        return days_listed < self.filter_new_stock_days

    def _get_avg_amount_20d(self, instrument: str, trade_date: pd.Timestamp) -> float:
        """Get 20-day average amount for a stock on a given date."""
        if self._avg_amount_20d is None:
            return float("inf")
        try:
            val = self._avg_amount_20d.loc[(trade_date, instrument)]
            return float(val) if pd.notna(val) else 0.0
        except (KeyError, TypeError):
            return 0.0

    def _parquet_files_for_period(self, root: Path, start: pd.Timestamp | None = None) -> list[Path]:
        s = start if start is not None else self.start_time
        return self._parquet_files_between(root=root, start_time=s, end_time=self.end_time)

    def _parquet_files_between(self, root: Path, start_time: pd.Timestamp, end_time: pd.Timestamp) -> list[Path]:
        files: list[Path] = []
        for file_path in root.rglob("*.parquet"):
            try:
                file_date = pd.to_datetime(file_path.stem, format="%Y%m%d")
            except ValueError:
                files.append(file_path)
                continue
            if start_time <= file_date <= end_time:
                files.append(file_path)
        return sorted(files)

    def load_benchmark_returns(self) -> pd.Series:
        """Load benchmark returns.

        Tries qlib ``*.day.bin`` files first.  When qlib benchmark data is
        unavailable, falls back to computing an equal-weighted market return
        from the daily bar parquet files used for execution pricing.
        """

        benchmark_start_time = self.start_time - pd.Timedelta(days=self.benchmark_lookback_days)

        # --- Primary path: qlib binary format ---
        try:
            features_dir = self._resolve_qlib_features_dir(self.benchmark_path)
            benchmark_code = self._normalize_benchmark_code(self.benchmark)
            qlib_instrument = self._benchmark_to_qlib_instrument(benchmark_code)
            instrument_dir = features_dir / qlib_instrument
            if instrument_dir.exists():
                change_file = instrument_dir / "change.day.bin"
                if change_file.exists():
                    benchmark_series = self._read_qlib_day_bin(change_file, features_dir).astype(float)
                else:
                    close_file = instrument_dir / "close.day.bin"
                    if close_file.exists():
                        benchmark_series = self._read_qlib_day_bin(close_file, features_dir).astype(float).pct_change()
                    else:
                        raise FileNotFoundError(
                            f"Benchmark qlib files are missing: neither {change_file} nor {close_file} exists"
                        )
                benchmark_series = benchmark_series.replace([np.inf, -np.inf], np.nan).dropna()
                benchmark_series = benchmark_series.loc[
                    (benchmark_series.index >= benchmark_start_time) & (benchmark_series.index <= self.end_time)
                ]
                if not benchmark_series.empty:
                    benchmark_series.name = "benchmark"
                    return benchmark_series.sort_index()
        except Exception as exc:  # noqa: BLE001
            logging.warning("无法从 qlib 加载基准指数，将使用 daily bars 的市场等权收益作为基准: %s", exc)

        # --- Fallback: equal-weighted market return from daily bars ---
        import warnings
        warnings.warn(
            f"Qlib benchmark data not available for {self.benchmark!r}. "
            f"Falling back to equal-weighted market return from daily bars at {self.price_path}.",
            RuntimeWarning,
        )
        benchmark_series = self._compute_equal_weighted_market_return(benchmark_start_time, self.end_time)
        benchmark_series.name = "benchmark"
        return benchmark_series.sort_index()

    def _compute_equal_weighted_market_return(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        """Compute equal-weighted market return from daily bar parquet files."""

        files = self._parquet_files_between(self.price_path, start, end)
        if not files:
            raise FileNotFoundError(
                f"No daily bar parquet files found in {self.price_path} for market benchmark: {start.date()} ~ {end.date()}"
            )

        daily_returns = []
        for file_path in files:
            df = pd.read_parquet(file_path)
            if "pct_chg" not in df.columns:
                if "close" in df.columns and "pre_close" in df.columns:
                    df["pct_chg"] = (df["close"] / df["pre_close"] - 1.0) * 100.0
                else:
                    continue
            rets = df["pct_chg"].replace([np.inf, -np.inf], np.nan).dropna()
            if rets.empty:
                continue
            # pct_chg 单位是百分比（如 2.5 表示 2.5%），转成小数
            mean_ret = float(rets.mean()) / 100.0
            trade_date = df["trade_date"].iloc[0] if "trade_date" in df.columns else file_path.stem
            daily_returns.append((pd.Timestamp(str(trade_date)), mean_ret))

        if not daily_returns:
            raise ValueError(f"无法从 daily bars 计算市场等权收益: {start.date()} ~ {end.date()}")

        series = pd.Series(dict(daily_returns)).sort_index()
        series = series.loc[(series.index >= start) & (series.index <= end)]
        if series.empty:
            raise ValueError(f"市场等权收益序列在日期过滤后为空: {start.date()} ~ {end.date()}")
        return series

    @staticmethod
    def _resolve_qlib_features_dir(benchmark_path: Path) -> Path:
        """Return the qlib features directory from either qlib root or features path."""

        path = Path(benchmark_path)
        if not path.exists():
            raise FileNotFoundError(f"Benchmark qlib path does not exist: {path}")
        if path.name == "features":
            return path
        features_dir = path / "features"
        if features_dir.exists():
            return features_dir
        return path

    @staticmethod
    def _read_qlib_calendar(features_dir: Path) -> pd.DatetimeIndex:
        """Read qlib day calendar adjacent to the features directory."""

        calendar_path = features_dir.parent / "calendars" / "day.txt"
        if not calendar_path.exists():
            raise FileNotFoundError(f"Qlib day calendar does not exist: {calendar_path}")
        calendar = pd.read_csv(calendar_path, header=None, names=["datetime"])
        dates = pd.to_datetime(calendar["datetime"], errors="coerce")
        dates = dates.dropna()
        if dates.empty:
            raise ValueError(f"Qlib day calendar is empty or invalid: {calendar_path}")
        return pd.DatetimeIndex(dates)

    @classmethod
    def _read_qlib_day_bin(cls, file_path: Path, features_dir: Path) -> pd.Series:
        """Read a qlib daily binary field as a date-indexed float series.

        Qlib stores daily binary fields as float32 arrays.  The first element is
        the start offset into ``calendars/day.txt``; all remaining elements are
        field values aligned to consecutive calendar rows from that offset.
        """

        values = np.fromfile(file_path, dtype="<f4")
        if values.size <= 1:
            raise ValueError(f"Qlib binary file is empty or invalid: {file_path}")
        start_index = int(values[0])
        data = values[1:].astype(float)
        calendar = cls._read_qlib_calendar(features_dir)
        end_index = start_index + len(data)
        if start_index < 0 or start_index >= len(calendar):
            raise ValueError(
                f"Qlib binary start index {start_index} in {file_path} is outside calendar length {len(calendar)}"
            )
        if end_index > len(calendar):
            data = data[: len(calendar) - start_index]
            end_index = len(calendar)
        return pd.Series(data, index=calendar[start_index:end_index], name=file_path.stem.split(".")[0])

    @staticmethod
    def _normalize_benchmark_code(code: str) -> str:
        raw = code.strip().upper().replace("-", "_")
        aliases = {
            "SH000001": "000001.SH",
            "SSE": "000001.SH",
            "上证": "000001.SH",
            "上证综指": "000001.SH",
            "SH000300": "000300.SH",
            "CSI300": "000300.SH",
            "沪深300": "000300.SH",
            "SH000905": "000905.SH",
            "CSI500": "000905.SH",
            "中证500": "000905.SH",
        }
        if raw in aliases:
            return aliases[raw]
        if raw.endswith(".PARQUET"):
            raw = raw[: -len(".PARQUET")]
        raw = raw.replace("_", ".")
        if "." in raw:
            left, right = raw.split(".", 1)
            return f"{left.zfill(6)}.{right}"
        if raw.startswith("SH") and len(raw) == 8:
            return f"{raw[2:]}.SH"
        if raw.startswith("SZ") and len(raw) == 8:
            return f"{raw[2:]}.SZ"
        return raw

    @staticmethod
    def _benchmark_to_qlib_instrument(code: str) -> str:
        """Convert normalized benchmark code to qlib instrument directory name."""

        normalized = SimpleBacktester._normalize_benchmark_code(code)
        if "." in normalized:
            symbol, exchange = normalized.split(".", 1)
            return f"{exchange.lower()}{symbol.zfill(6)}"
        raw = normalized.lower()
        if len(raw) == 8 and raw[:2] in {"sh", "sz", "bj"}:
            return raw
        raise ValueError(f"Cannot convert benchmark code to qlib instrument name: {code}")

    def _simulate(
        self, pred_df: pd.DataFrame, price_df: pd.DataFrame, benchmark_returns: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        topk = int(self.strategy_config.get("topk", 50))
        n_drop = int(self.strategy_config.get("n_drop", 5))
        if topk <= 0:
            raise ValueError("strategy.topk must be positive")

        pred_df = pred_df.sort_index()
        price_df = price_df.sort_index()
        if pred_df.empty or price_df.empty:
            raise ValueError("Prediction and price data must both be non-empty for the requested period")

        cash = float(self.account)
        holdings: dict[str, list[dict[str, Any]]] = {}
        last_close: dict[str, float] = {}
        previous_account_value = float(self.account)
        reports: list[dict[str, Any]] = []
        positions: list[dict[str, Any]] = []
        trades: list[dict[str, Any]] = []
        trade_dates = pd.DatetimeIndex(price_df.index.get_level_values("datetime").unique()).sort_values()
        pred_dates = pd.DatetimeIndex(pred_df.index.get_level_values("datetime").unique()).sort_values()
        signal_date_by_trade_date = self._signal_dates_for_trade_dates(trade_dates=trade_dates, pred_dates=pred_dates)
        executable_trade_dates = [date for date in trade_dates if signal_date_by_trade_date.get(pd.Timestamp(date)) is not None]
        rebalance_dates = self._rebalance_dates(executable_trade_dates)
        benchmark_by_date = benchmark_returns.sort_index().reindex(trade_dates, method="ffill")

        for trade_date, day_price_df in price_df.groupby(level="datetime", sort=True):
            trade_date = pd.Timestamp(trade_date)
            is_rebalance_date = trade_date in rebalance_dates
            base_cols = ["open", "close", "pre_close", "pct_chg", "vol", "amount"]
            filter_cols = []
            if self.filter_st:
                filter_cols += [c for c in ["st_stock", "star_st_stock"] if c in day_price_df.columns]
            if self.filter_suspend:
                if "is_suspect" in day_price_df.columns:
                    filter_cols.append("is_suspect")
            all_cols = base_cols + filter_cols
            prices = day_price_df.droplevel("datetime")[all_cols]
            executable_prices = prices[self.deal_price_col]
            valid_price_names = set(prices.dropna(subset=[self.deal_price_col, "close"]).index)

            if is_rebalance_date:
                signal_date = signal_date_by_trade_date.get(trade_date)
                if signal_date is None:
                    day_df = pd.DataFrame(columns=[self.score_col] + all_cols)
                else:
                    day_scores = pred_df.xs(signal_date, level="datetime")[[self.score_col]]
                    day_df = day_scores.join(prices, how="inner").sort_values(self.score_col, ascending=False)
                current_value_at_deal = cash + self._stock_value(holdings, executable_prices, last_close)
                target_names = self._select_target_names(
                    day_df=day_df,
                    holdings=holdings,
                    prices=prices,
                    trade_date=trade_date,
                    topk=topk,
                    n_drop=n_drop,
                )

                # Dynamic position sizing: compute target equity ratio
                target_position_ratio = self._calc_dynamic_position_ratio(
                    day_df=day_df, topk=topk, trade_date=trade_date,
                    benchmark_returns=benchmark_returns,
                )
                target_total_value = current_value_at_deal * target_position_ratio

                # Volatility-weighted or equal-weight position sizing
                target_weights = self._calc_target_weights(
                    target_names=target_names,
                    day_df=day_df,
                    trade_date=trade_date,
                    price_df=price_df,
                )
                target_shares = self._calc_target_shares_weighted(
                    target_names=target_names,
                    target_weights=target_weights,
                    target_value=target_total_value,
                    executable_prices=executable_prices,
                    trade_date=trade_date,
                )
            else:
                day_df = pd.DataFrame(columns=[self.score_col])
                target_shares = {}

            day_buy_value = 0.0
            day_sell_value = 0.0
            day_fee = 0.0
            day_slippage = 0.0

            if is_rebalance_date:
                # Sell first.  Shares bought on trade_date are unavailable by T+1.
                for instrument in sorted(set(holdings)):
                    if instrument not in valid_price_names:
                        continue
                    current_shares = self._total_shares(holdings.get(instrument, []))
                    desired_shares = target_shares.get(instrument, 0)
                    sell_qty = self._round_lot_down(max(current_shares - desired_shares, 0))
                    if sell_qty <= 0:
                        continue
                    row = prices.loc[instrument]
                    if self._is_sell_blocked(row):
                        continue
                    available_qty = self._round_lot_down(self._available_sell_shares(holdings.get(instrument, []), trade_date))
                    sell_qty = min(sell_qty, available_qty)
                    if sell_qty <= 0:
                        continue

                    raw_price = float(row[self.deal_price_col])
                    raw_value = raw_price * sell_qty
                    exec_price = raw_price * (1.0 - self.slippage)
                    gross_cash = exec_price * sell_qty
                    fee = self._calc_fee(raw_value, self.close_cost)
                    cash += gross_cash - fee
                    self._remove_shares_fifo(holdings[instrument], sell_qty, trade_date)
                    if self._total_shares(holdings[instrument]) <= 0:
                        holdings.pop(instrument, None)
                    slippage_cost = raw_value - gross_cash
                    day_sell_value += raw_value
                    day_fee += fee
                    day_slippage += slippage_cost
                    trades.append(
                        self._trade_record(
                            trade_date, instrument, "sell", sell_qty, raw_price, exec_price, raw_value, fee, slippage_cost, cash
                        )
                    )

                # Buy in score order, so stronger names get cash first if rounding/cash binds.
                for instrument in day_df.index:
                    if instrument not in target_shares or instrument not in valid_price_names:
                        continue
                    current_shares = self._total_shares(holdings.get(instrument, []))
                    buy_qty = self._round_lot_down(max(target_shares[instrument] - current_shares, 0))
                    if buy_qty <= 0:
                        continue
                    row = prices.loc[instrument]
                    if self._is_buy_blocked(row):
                        continue
                    raw_price = float(row[self.deal_price_col])
                    affordable_qty = self._max_affordable_buy_qty(cash, raw_price, buy_qty)
                    buy_qty = min(buy_qty, affordable_qty)
                    if buy_qty <= 0:
                        continue
                    raw_value = raw_price * buy_qty
                    exec_price = raw_price * (1.0 + self.slippage)
                    gross_cash = exec_price * buy_qty
                    fee = self._calc_fee(raw_value, self.open_cost)
                    total_cash = gross_cash + fee
                    if total_cash > cash + 1e-8:
                        continue
                    cash -= total_cash
                    holdings.setdefault(instrument, []).append({"shares": int(buy_qty), "buy_date": trade_date, "cost_price": float(exec_price)})
                    slippage_cost = gross_cash - raw_value
                    day_buy_value += raw_value
                    day_fee += fee
                    day_slippage += slippage_cost
                    trades.append(
                        self._trade_record(
                            trade_date, instrument, "buy", buy_qty, raw_price, exec_price, raw_value, fee, slippage_cost, cash
                        )
                    )

            stock_value = self._stock_value(holdings, prices["close"], last_close)
            account_value = cash + stock_value

            # Update last known close prices for all holdings (suspended stocks keep previous close)
            for instrument in holdings:
                if instrument in prices.index and pd.notna(prices.loc[instrument, "close"]):
                    last_close[instrument] = float(prices.loc[instrument, "close"])
            total_cost = day_fee + day_slippage
            net_return = account_value / previous_account_value - 1.0 if previous_account_value else 0.0
            gross_return = (account_value + total_cost) / previous_account_value - 1.0 if previous_account_value else net_return
            benchmark_return = benchmark_by_date.get(pd.Timestamp(trade_date), np.nan)
            excess_return = net_return - benchmark_return if pd.notna(benchmark_return) else np.nan
            turnover = (day_buy_value + day_sell_value) / previous_account_value if previous_account_value else 0.0

            reports.append(
                {
                    "datetime": trade_date,
                    "return": float(net_return),
                    "gross_return": float(gross_return),
                    "benchmark": benchmark_return,
                    "excess_return": float(excess_return) if pd.notna(excess_return) else np.nan,
                    "cost": float(total_cost),
                    "fee": float(day_fee),
                    "slippage_cost": float(day_slippage),
                    "turnover": float(turnover),
                    "buy_value": float(day_buy_value),
                    "sell_value": float(day_sell_value),
                    "cash": float(cash),
                    "stock_value": float(stock_value),
                    "account_value": float(account_value),
                    "holdings_count": int(sum(1 for lots in holdings.values() if self._total_shares(lots) > 0)),
                    "is_rebalance_date": bool(is_rebalance_date),
                }
            )

            if self.save_positions:
                for instrument, lots in sorted(holdings.items()):
                    shares = self._total_shares(lots)
                    if shares <= 0:
                        continue
                    if instrument in prices.index and pd.notna(prices.loc[instrument, "close"]):
                        close_price = float(prices.loc[instrument, "close"])
                    elif instrument in last_close:
                        close_price = float(last_close[instrument])
                    else:
                        continue
                    market_value = shares * close_price
                    positions.append(
                        {
                            "datetime": trade_date,
                            "instrument": instrument,
                            "shares": int(shares),
                            "close": close_price,
                            "market_value": float(market_value),
                            "weight": float(market_value / account_value) if account_value else 0.0,
                            "available_shares": int(self._available_sell_shares(lots, pd.Timestamp(trade_date))),
                        }
                    )

            previous_account_value = account_value

        report_df = pd.DataFrame(reports).set_index("datetime") if reports else pd.DataFrame()
        positions_df = pd.DataFrame(positions)
        if not positions_df.empty:
            positions_df = positions_df.set_index(["datetime", "instrument"]).sort_index()
        trades_df = pd.DataFrame(trades)
        if not trades_df.empty:
            trades_df = trades_df.set_index(["datetime", "instrument", "side"]).sort_index()
        return report_df, positions_df, trades_df

    @staticmethod
    def _get_benchmark_return(benchmark_returns: pd.Series, trade_date: pd.Timestamp) -> float:
        """Return benchmark return on trade_date, or previous available value."""

        if benchmark_returns.empty:
            return np.nan
        benchmark_returns = benchmark_returns.sort_index()
        if trade_date in benchmark_returns.index and pd.notna(benchmark_returns.loc[trade_date]):
            return float(benchmark_returns.loc[trade_date])
        fallback = benchmark_returns.loc[benchmark_returns.index < trade_date]
        if fallback.empty:
            return np.nan
        return float(fallback.iloc[-1])

    def _signal_dates_for_trade_dates(
        self, trade_dates: pd.DatetimeIndex, pred_dates: pd.DatetimeIndex
    ) -> dict[pd.Timestamp, pd.Timestamp | None]:
        """Map each trade date to the prediction date that is tradable on it.

        ``signal_delay=1`` means a score generated for date t can first be used
        on the next available trading date.  This avoids buying at date-t open or
        close with a signal that would only be known after date t data is formed.
        ``signal_delay=0`` preserves the old same-day behavior for explicit
        experiments, but it is not the default because it is look-ahead prone.
        """

        pred_dates = pd.DatetimeIndex(pd.to_datetime(pred_dates)).dropna().sort_values()
        mapping: dict[pd.Timestamp, pd.Timestamp | None] = {}
        if pred_dates.empty:
            return {pd.Timestamp(date): None for date in trade_dates}

        pred_values = pred_dates.to_numpy(dtype="datetime64[ns]")
        for date in pd.DatetimeIndex(pd.to_datetime(trade_dates)).dropna().sort_values():
            if self.signal_delay == 0:
                # Latest score available at or before the trade date.
                pred_pos = int(np.searchsorted(pred_values, np.datetime64(date), side="right")) - 1
            else:
                # Strictly older scores only; delay=1 maps to the previous
                # prediction date, delay=2 to two prediction dates back, etc.
                pred_pos = int(np.searchsorted(pred_values, np.datetime64(date), side="left")) - self.signal_delay
            mapping[pd.Timestamp(date)] = pd.Timestamp(pred_dates[pred_pos]) if 0 <= pred_pos < len(pred_dates) else None
        return mapping

    def _rebalance_dates(self, dates: pd.Index | pd.Series | np.ndarray) -> set[pd.Timestamp]:
        """Return dates on which the strategy is allowed to rebalance.

        ``freq`` controls the rebalance schedule.  Daily mode rebalances on every
        trading day; weekly/monthly modes rebalance on the first available
        trading day of each week/month.  Integer trading-day intervals are also
        supported, e.g. ``10``, ``"10d"`` or ``"20days"`` means rebalance every
        10/20 available trading days starting from the first backtest date.
        """

        unique_dates = pd.DatetimeIndex(pd.to_datetime(pd.Index(dates).unique())).dropna().sort_values()
        if unique_dates.empty:
            return set()

        freq = str(self.freq).lower()
        if freq in {"day", "daily", "1d"}:
            selected = unique_dates
        elif freq in {"week", "weekly", "1w", "w"}:
            selected = unique_dates.to_series(index=unique_dates).groupby(unique_dates.to_period("W")).first()
        elif freq in {"month", "monthly", "1m", "m"}:
            selected = unique_dates.to_series(index=unique_dates).groupby(unique_dates.to_period("M")).first()
        elif (interval := self._trading_day_interval_from_freq()) is not None:
            selected = unique_dates[::interval]
        else:
            selected = unique_dates

        return {pd.Timestamp(date) for date in selected}

    def _trading_day_interval_from_freq(self) -> int | None:
        """Parse ``freq`` as an N-trading-day rebalance interval when possible."""

        if isinstance(self.freq, (int, np.integer)):
            return int(self.freq) if int(self.freq) > 0 else None

        freq = str(self.freq).strip().lower()
        if freq in {"day", "daily", "1d", "week", "weekly", "1w", "w", "month", "monthly", "1m", "m"}:
            return None
        if freq.isdigit():
            interval = int(freq)
            return interval if interval > 0 else None

        match = re.fullmatch(
            r"(?:(?:every|每)[_\s-]*)?(\d+)\s*"
            r"(?:d|day|days|td|trade[_\s-]*days?|trading[_\s-]*days?|天|个交易日)",
            freq,
        )
        if not match:
            return None
        interval = int(match.group(1))
        return interval if interval > 0 else None

    def _load_industry_map(self) -> dict[str, str]:
        """Load instrument -> industry mapping. Uses latest available industry data."""
        if self._industry_map is not None:
            return self._industry_map

        if not self.industry_data_path or not self.industry_data_path.exists():
            self._industry_map = {}
            return self._industry_map

        # Find the latest year/month available
        latest_df = None
        year_dirs = sorted([d for d in self.industry_data_path.glob("year=*")], reverse=True)
        for year_dir in year_dirs:
            month_dirs = sorted([d for d in year_dir.glob("month=*")], reverse=True)
            for month_dir in month_dirs:
                df = pd.read_parquet(month_dir)
                if "l1_name" in df.columns and "ts_code" in df.columns:
                    latest_df = df.dropna(subset=["l1_name"])
                    break
            if latest_df is not None:
                break

        if latest_df is None:
            self._industry_map = {}
        else:
            # Use the last available date's industry classification
            latest_date = latest_df["trade_date"].max()
            latest_df = latest_df[latest_df["trade_date"] == latest_date]
            self._industry_map = dict(zip(latest_df["ts_code"], latest_df["l1_name"].astype(str)))

        return self._industry_map

    def _select_target_names(
        self,
        day_df: pd.DataFrame,
        holdings: Mapping[str, list[dict[str, Any]]],
        prices: pd.DataFrame,
        trade_date: pd.Timestamp,
        topk: int,
        n_drop: int,
    ) -> list[str]:
        scores = day_df[self.score_col].dropna()
        ranked_desc = list(scores.sort_values(ascending=False).index)
        current = {inst for inst, lots in holdings.items() if self._total_shares(lots) > 0}
        valid_current = current.intersection(prices.index)

        sellable_current = {
            inst
            for inst in valid_current
            if self._available_sell_shares(holdings.get(inst, []), trade_date) > 0 and not self._is_sell_blocked(prices.loc[inst])
        }
        to_drop: set[str] = set()
        if n_drop > 0 and sellable_current:
            if self.drop_criteria == "return":
                # Sell holdings with worst holding-period return
                holding_returns = {}
                for inst in sellable_current:
                    if inst not in prices.index or inst not in scores.index:
                        continue
                    lots = holdings.get(inst, [])
                    total_shares = 0
                    total_cost = 0.0
                    for lot in lots:
                        shares = int(lot.get("shares", 0))
                        cost = float(lot.get("cost_price", 0.0))
                        if shares > 0 and cost > 0:
                            total_shares += shares
                            total_cost += shares * cost
                    if total_shares > 0 and total_cost > 0:
                        avg_cost = total_cost / total_shares
                        current_price = float(prices.loc[inst, self.deal_price_col])
                        holding_returns[inst] = current_price / avg_cost - 1.0
                if holding_returns:
                    sorted_by_return = sorted(holding_returns.items(), key=lambda x: x[1])
                    to_drop = {inst for inst, _ in sorted_by_return[:n_drop]}
            else:
                # Default: sell holdings with lowest predicted score
                scored_sellable = scores.loc[list(sellable_current.intersection(scores.index))]
                to_drop = set(scored_sellable.sort_values(ascending=True).head(n_drop).index)

        # Pre-filter: build set of stocks that pass all pool filters
        eligible = self._get_eligible_stocks(day_df, prices, trade_date)

        # Industry-stratified selection
        industry_map = self._load_industry_map()
        if self.industry_stratified and industry_map:
            return self._select_stratified(
                scores, eligible, industry_map, trade_date, topk,
                current, to_drop, holdings, prices, ranked_desc,
            )

        # Default: global ranking with industry cap
        target = [inst for inst in ranked_desc if inst in current and inst not in to_drop]
        # Keep non-sellable existing holdings even if they are not in current top scores.
        for inst in sorted(current - set(target) - to_drop):
            if inst in prices.index and self._available_sell_shares(holdings.get(inst, []), trade_date) <= 0:
                target.append(inst)

        use_industry_limit = (self.max_industry_weight < 1.0 or self.max_industry_count > 0) and industry_map

        target_set = set(target)
        industry_count: dict[str, int] = {}
        max_per_industry_count = self.max_industry_count if self.max_industry_count > 0 else topk
        if use_industry_limit and self.max_industry_weight < 1.0:
            max_by_weight = max(1, int(topk * self.max_industry_weight))
            max_per_industry_count = min(max_per_industry_count, max_by_weight)

        if use_industry_limit:
            for inst in target:
                ind = industry_map.get(inst, "unknown")
                industry_count[ind] = industry_count.get(ind, 0) + 1

        for inst in ranked_desc:
            if len(target) >= topk:
                break
            if inst in target_set:
                continue
            if inst not in eligible:
                continue
            if use_industry_limit:
                ind = industry_map.get(inst, "unknown")
                if industry_count.get(ind, 0) >= max_per_industry_count:
                    continue
                industry_count[ind] = industry_count.get(ind, 0) + 1
            target.append(inst)
            target_set.add(inst)
        return target[:topk]

    def _select_stratified(
        self,
        scores: pd.Series,
        eligible: set[str],
        industry_map: dict[str, str],
        trade_date: pd.Timestamp,
        topk: int,
        current: set[str],
        to_drop: set[str],
        holdings: Mapping[str, list[dict[str, Any]]],
        prices: pd.DataFrame,
        ranked_desc: list[str],
    ) -> list[str]:
        """Industry-stratified stock selection.

        Stocks are grouped by industry first, then within each industry the
        top-ranked by prediction score are selected.  Per-industry quotas are
        determined by ``industry_stratify_method``:

        - ``equal``: each industry gets the same number of slots (floor division,
          remainder allocated to largest industries by eligible count).
        - ``proportional``: each industry gets slots proportional to its share
          of eligible stocks.
        """
        # Group eligible stocks by industry
        industry_stocks: dict[str, list[str]] = {}
        for inst in eligible:
            ind = industry_map.get(inst, "unknown")
            industry_stocks.setdefault(ind, []).append(inst)

        # Sort each industry's stocks by score descending
        for ind in industry_stocks:
            industry_stocks[ind].sort(
                key=lambda x: scores.get(x, float("-inf")),
                reverse=True,
            )

        n_industries = len(industry_stocks)
        if n_industries == 0:
            return []

        # Compute per-industry quota
        quotas: dict[str, int] = {}
        if self.industry_stratify_method == "proportional":
            total_eligible = sum(len(v) for v in industry_stocks.values())
            remaining = topk
            sorted_inds = sorted(industry_stocks.keys(), key=lambda x: len(industry_stocks[x]), reverse=True)
            for ind in sorted_inds:
                n = len(industry_stocks[ind])
                quota = max(1, int(round(topk * n / total_eligible)))
                quota = min(quota, n, remaining)
                quotas[ind] = quota
                remaining -= quota
            # Distribute any leftover to largest industries
            if remaining > 0:
                for ind in sorted_inds:
                    if remaining <= 0:
                        break
                    if quotas[ind] < len(industry_stocks[ind]):
                        quotas[ind] += 1
                        remaining -= 1
        else:  # equal
            base = topk // n_industries
            remainder = topk % n_industries
            # Industries with more eligible stocks get the remainder
            sorted_inds = sorted(industry_stocks.keys(), key=lambda x: len(industry_stocks[x]), reverse=True)
            for i, ind in enumerate(sorted_inds):
                quota = base + (1 if i < remainder else 0)
                quotas[ind] = min(quota, len(industry_stocks[ind]))

        # Apply max industry cap if configured
        max_per_industry_count = self.max_industry_count if self.max_industry_count > 0 else topk
        if self.max_industry_weight < 1.0:
            max_by_weight = max(1, int(topk * self.max_industry_weight))
            max_per_industry_count = min(max_per_industry_count, max_by_weight)
        for ind in quotas:
            quotas[ind] = min(quotas[ind], max_per_industry_count)

        # Build target: first keep existing holdings that rank within quota
        target: list[str] = []
        target_set: set[str] = set()
        industry_filled: dict[str, int] = {ind: 0 for ind in quotas}

        # Keep non-sellable holdings first
        for inst in sorted(current - to_drop):
            ind = industry_map.get(inst, "unknown")
            if ind not in quotas:
                continue
            if inst in prices.index and self._available_sell_shares(holdings.get(inst, []), trade_date) <= 0:
                if industry_filled[ind] < quotas[ind]:
                    target.append(inst)
                    target_set.add(inst)
                    industry_filled[ind] += 1

        # Then fill each industry with top-ranked eligible stocks
        for ind in sorted(quotas.keys(), key=lambda x: -len(industry_stocks[x])):
            for inst in industry_stocks[ind]:
                if industry_filled[ind] >= quotas[ind]:
                    break
                if inst in target_set:
                    continue
                if inst in to_drop:
                    continue
                target.append(inst)
                target_set.add(inst)
                industry_filled[ind] += 1

        return target

    def _get_eligible_stocks(
        self,
        day_df: pd.DataFrame,
        prices: pd.DataFrame,
        trade_date: pd.Timestamp,
    ) -> set[str]:
        """Return set of stock symbols that pass all stock pool filters."""
        eligible = set(prices.index)

        # Must have valid price and not be buy-blocked (limit up/down, zero vol)
        buy_blocked = set()
        for inst in prices.index:
            if self._is_buy_blocked(prices.loc[inst]):
                buy_blocked.add(inst)
        eligible -= buy_blocked

        # ST filter
        if self.filter_st:
            st_set = set()
            if "st_stock" in prices.columns:
                st_set |= set(prices[prices["st_stock"] == 1].index)
            if "star_st_stock" in prices.columns:
                st_set |= set(prices[prices["star_st_stock"] == 1].index)
            eligible -= st_set

        # Suspend filter
        if self.filter_suspend and "is_suspect" in prices.columns:
            suspend_set = set(prices[prices["is_suspect"] == 1].index)
            eligible -= suspend_set

        # New stock filter
        if self.filter_new_stock_days > 0 and self._list_dates is not None:
            new_stocks = {inst for inst in eligible if self._is_new_stock(inst, trade_date)}
            eligible -= new_stocks

        # Liquidity filter: 20d avg amount >= min_avg_amount_20d
        if self.min_avg_amount_20d > 0 and self._avg_amount_20d is not None:
            low_liquidity = set()
            for inst in eligible:
                avg_amt = self._get_avg_amount_20d(inst, trade_date)
                if avg_amt < self.min_avg_amount_20d:
                    low_liquidity.add(inst)
            eligible -= low_liquidity

        return eligible

    def _calc_target_shares(self, target_names: list[str], target_value: float, executable_prices: pd.Series) -> dict[str, int]:
        if not target_names:
            return {}
        per_position_value = target_value / len(target_names)
        target_shares: dict[str, int] = {}
        for instrument in target_names:
            price = executable_prices.get(instrument, np.nan)
            if pd.isna(price) or float(price) <= 0:
                continue
            target_shares[instrument] = self._round_lot_down(int(per_position_value / float(price)))
        return target_shares

    def _calc_dynamic_position_ratio(
        self,
        day_df: pd.DataFrame,
        topk: int,
        trade_date: pd.Timestamp | None = None,
        benchmark_returns: pd.Series | None = None,
    ) -> float:
        """Compute target equity ratio based on dynamic positioning.

        Returns 1.0 if dynamic positioning is disabled.
        Methods:
        - score_threshold: uses average score of top-K stocks (fixed threshold)
        - dispersion: uses top-bottom score spread as alpha-strength proxy
        - target_vol: scales position so that expected portfolio vol matches target
        """
        if not self.dynamic_position_enabled:
            return 1.0

        if self.dyn_pos_method == "target_vol":
            return self._calc_target_vol_position_ratio(trade_date, benchmark_returns)

        scores = day_df[self.score_col].dropna()
        if len(scores) == 0:
            return self.dyn_pos_min_position

        if self.dyn_pos_method == "dispersion":
            return self._calc_dispersion_position_ratio(scores)
        else:
            return self._calc_score_threshold_position_ratio(scores)

    def _calc_target_vol_position_ratio(
        self,
        trade_date: pd.Timestamp | None,
        benchmark_returns: pd.Series | None,
    ) -> float:
        """Target volatility position sizing using benchmark index vol.

        Position ratio = target_vol / realized_vol, clamped to [min, max].
        Realized vol is the annualized std of benchmark daily returns over
        the lookback window.
        """
        if benchmark_returns is None or trade_date is None:
            return self.dyn_pos_max_position

        # Get lookback returns
        lookback_start = trade_date - pd.Timedelta(days=self.dyn_pos_vol_lookback * 2)
        mask = (benchmark_returns.index >= lookback_start) & (benchmark_returns.index < trade_date)
        hist_rets = benchmark_returns.loc[mask].dropna()

        min_periods = max(5, self.dyn_pos_vol_lookback // 2)
        if len(hist_rets) < min_periods:
            return self.dyn_pos_max_position  # not enough data, default full

        # Annualized realized volatility
        realized_vol = float(hist_rets.std() * np.sqrt(252))

        if realized_vol <= 0:
            return self.dyn_pos_max_position

        # Smooth with recent history via EMA-like approach
        if not hasattr(self, "_target_vol_history"):
            self._target_vol_history: list[float] = []
        self._target_vol_history.append(realized_vol)
        if len(self._target_vol_history) > self.dyn_pos_vol_lookback:
            self._target_vol_history.pop(0)

        smooth_n = min(self.dyn_pos_vol_smooth, len(self._target_vol_history))
        smoothed_vol = float(np.mean(self._target_vol_history[-smooth_n:]))

        # Position = target_vol / realized_vol
        ratio = self.dyn_pos_target_vol / smoothed_vol
        ratio = max(self.dyn_pos_min_position, min(self.dyn_pos_max_position, ratio))
        return ratio

    def _calc_score_threshold_position_ratio(self, scores: pd.Series) -> float:
        n_top = max(1, int(len(scores) * self.dyn_pos_topk_fraction))
        top_scores = scores.nlargest(n_top)
        avg_top_score = float(top_scores.mean())

        if avg_top_score <= self.dyn_pos_score_threshold:
            return self.dyn_pos_min_position

        score_range = 1.0 - self.dyn_pos_score_threshold
        if score_range <= 0:
            return self.dyn_pos_max_position

        ratio = (avg_top_score - self.dyn_pos_score_threshold) / score_range
        ratio = max(0.0, min(1.0, ratio))
        position = self.dyn_pos_min_position + ratio * (self.dyn_pos_max_position - self.dyn_pos_min_position)
        return max(self.dyn_pos_min_position, min(self.dyn_pos_max_position, position))

    def _calc_dispersion_position_ratio(self, scores: pd.Series) -> float:
        """Use top-bottom score dispersion as alpha-strength signal.

        Two modes via dyn_pos_method:
        - dispersion: absolute threshold (dispersion_low / dispersion_high)
        - dispersion_relative: percentile-based, uses historical lookback

        When the model can clearly distinguish good from bad stocks (high dispersion),
        we hold more equity. When dispersion is low (model can't tell apart), we reduce
        exposure since the edge is small.
        """
        n_top = max(1, int(len(scores) * self.dyn_pos_dispersion_top_frac))
        n_bottom = max(1, int(len(scores) * self.dyn_pos_dispersion_bottom_frac))

        top_mean = float(scores.nlargest(n_top).mean())
        bottom_mean = float(scores.nsmallest(n_bottom).mean())
        dispersion = top_mean - bottom_mean

        # Smooth with moving average
        self._dyn_pos_dispersion_history.append(dispersion)
        if len(self._dyn_pos_dispersion_history) > self.dyn_pos_dispersion_lookback_days:
            self._dyn_pos_dispersion_history.pop(0)

        # Short-term smoothed value for current decision
        smooth_n = min(self.dyn_pos_dispersion_smooth_days, len(self._dyn_pos_dispersion_history))
        smoothed = float(np.mean(self._dyn_pos_dispersion_history[-smooth_n:]))

        if self.dyn_pos_method == "dispersion_relative":
            return self._calc_relative_dispersion_position(smoothed)
        else:
            return self._calc_absolute_dispersion_position(smoothed)

    def _calc_absolute_dispersion_position(self, smoothed: float) -> float:
        """Absolute threshold mode."""
        if smoothed <= self.dyn_pos_dispersion_low:
            return self.dyn_pos_min_position
        if smoothed >= self.dyn_pos_dispersion_high:
            return self.dyn_pos_max_position

        spread = self.dyn_pos_dispersion_high - self.dyn_pos_dispersion_low
        if spread <= 0:
            return self.dyn_pos_max_position

        ratio = (smoothed - self.dyn_pos_dispersion_low) / spread
        ratio = max(0.0, min(1.0, ratio))
        position = self.dyn_pos_min_position + ratio * (self.dyn_pos_max_position - self.dyn_pos_min_position)
        return max(self.dyn_pos_min_position, min(self.dyn_pos_max_position, position))

    def _calc_relative_dispersion_position(self, smoothed: float) -> float:
        """Relative percentile mode: compare current dispersion to its own history."""
        hist = self._dyn_pos_dispersion_history
        # Need enough history for reliable percentile
        min_history = max(10, self.dyn_pos_dispersion_lookback_days // 3)
        if len(hist) < min_history:
            return self.dyn_pos_max_position  # default to full position when not enough data

        hist_arr = np.array(hist)
        percentile = float(np.mean(hist_arr < smoothed))  # fraction of history below current

        # Map percentile to position:
        # percentile <= low_pct -> min position
        # percentile >= high_pct -> max position
        if percentile <= self.dyn_pos_dispersion_low_pct:
            return self.dyn_pos_min_position
        if percentile >= self.dyn_pos_dispersion_high_pct:
            return self.dyn_pos_max_position

        pct_spread = self.dyn_pos_dispersion_high_pct - self.dyn_pos_dispersion_low_pct
        if pct_spread <= 0:
            return self.dyn_pos_max_position

        ratio = (percentile - self.dyn_pos_dispersion_low_pct) / pct_spread
        ratio = max(0.0, min(1.0, ratio))
        position = self.dyn_pos_min_position + ratio * (self.dyn_pos_max_position - self.dyn_pos_min_position)
        return max(self.dyn_pos_min_position, min(self.dyn_pos_max_position, position))

    def _calc_target_weights(
        self,
        target_names: list[str],
        day_df: pd.DataFrame,
        trade_date: pd.Timestamp,
        price_df: pd.DataFrame,
    ) -> dict[str, float]:
        """Compute per-instrument target weights.

        Returns equal weights if vol weighting is disabled.
        When enabled, uses inverse-volatility weighting: w_i = (1/vol_i)^power / sum.
        Volatility is pre-computed as rolling std of daily returns and looked up
        from ``self._vol_series`` for O(1) access per stock.
        """
        if not target_names:
            return {}

        if not self.vol_weight_enabled:
            w = 1.0 / len(target_names)
            return {name: w for name in target_names}

        # Fast path: look up pre-computed rolling vol
        vols: dict[str, float] = {}
        if self._vol_series is not None:
            for name in target_names:
                try:
                    v = float(self._vol_series.loc[(trade_date, name)])
                    if v == v and v > 0:
                        vols[name] = v
                except (KeyError, ValueError):
                    pass

        # Fill NaN/missing vols with median vol
        valid_vols = [v for v in vols.values() if v == v and v > 0]
        if not valid_vols:
            w = 1.0 / len(target_names)
            return {name: w for name in target_names}

        median_vol = float(np.median(valid_vols))
        inv_vols = []
        for name in target_names:
            v = vols.get(name, median_vol)
            if v != v or v <= 0:
                v = median_vol
            inv_vols.append(1.0 / (v ** self.vol_weight_power))

        total = sum(inv_vols)
        if total <= 0:
            w = 1.0 / len(target_names)
            return {name: w for name in target_names}

        return {name: inv_vols[i] / total for i, name in enumerate(target_names)}

    def _calc_target_shares_weighted(
        self,
        target_names: list[str],
        target_weights: dict[str, float],
        target_value: float,
        executable_prices: pd.Series,
        trade_date: pd.Timestamp | None = None,
    ) -> dict[str, int]:
        """Compute target share counts from per-instrument weights.

        Applies single-stock weight cap and liquidity position cap if configured.
        When caps are hit, excess weight is redistributed proportionally.
        """
        if not target_names or target_value <= 0:
            return {}

        # Step 1: compute raw position values from weights
        position_values: dict[str, float] = {}
        for instrument in target_names:
            price = executable_prices.get(instrument, np.nan)
            if pd.isna(price) or float(price) <= 0:
                continue
            weight = target_weights.get(instrument, 0.0)
            position_values[instrument] = target_value * weight

        # Step 2: apply single-stock weight cap
        if self.max_single_weight > 0:
            max_position_value = target_value * self.max_single_weight
            position_values = self._apply_value_cap(position_values, max_position_value, target_value)

        # Step 3: apply liquidity cap (max buy = avg_20d_amount * max_buy_pct)
        if self.max_buy_pct_of_avg_amount > 0 and self._avg_amount_20d is not None and trade_date is not None:
            capped_values: dict[str, float] = {}
            for inst, val in position_values.items():
                avg_amt = self._get_avg_amount_20d(inst, trade_date)
                if avg_amt > 0:
                    liq_cap = avg_amt * self.max_buy_pct_of_avg_amount
                    capped_values[inst] = min(val, liq_cap)
                else:
                    capped_values[inst] = val
            total_capped = sum(capped_values.values())
            if total_capped > 0 and total_capped < target_value * 0.99:
                position_values = self._redistribute_excess(capped_values, target_value, executable_prices, trade_date)
            else:
                position_values = capped_values

        # Step 4: convert to shares
        target_shares: dict[str, int] = {}
        for instrument, pos_value in position_values.items():
            price = executable_prices.get(instrument, np.nan)
            if pd.isna(price) or float(price) <= 0:
                continue
            target_shares[instrument] = self._round_lot_down(int(pos_value / float(price)))
        return target_shares

    def _apply_value_cap(
        self,
        position_values: dict[str, float],
        max_value: float,
        total_target: float,
    ) -> dict[str, float]:
        """Apply a per-stock value cap, redistributing excess to uncapped stocks."""
        result = dict(position_values)
        for _ in range(10):  # iterate to handle cascading caps
            capped = {k: min(v, max_value) for k, v in result.items()}
            total_capped = sum(capped.values())
            if total_capped <= 0:
                break
            excess = total_target - total_capped
            if excess <= 0:
                result = capped
                break
            # Find uncapped stocks (below cap)
            uncapped = {k: v for k, v in result.items() if v < max_value - 1e-8}
            if not uncapped:
                result = capped
                break
            uncapped_total = sum(uncapped.values())
            if uncapped_total <= 0:
                result = capped
                break
            # Distribute excess proportionally to uncapped stocks
            add_ratio = excess / uncapped_total
            for k in uncapped:
                result[k] = capped[k] * (1 + add_ratio)
            # Re-check caps after redistribution
            for k in result:
                result[k] = min(result[k], max_value)
        return result

    def _redistribute_excess(
        self,
        capped_values: dict[str, float],
        total_target: float,
        executable_prices: pd.Series,
        trade_date: pd.Timestamp,
    ) -> dict[str, float]:
        """Redistribute excess value from liquidity-capped stocks to uncapped ones."""
        result = dict(capped_values)
        for _ in range(10):
            total = sum(result.values())
            excess = total_target - total
            if excess <= 0:
                break
            # Find stocks with room (below their liquidity cap)
            room: dict[str, float] = {}
            for inst, val in result.items():
                avg_amt = self._get_avg_amount_20d(inst, trade_date)
                if avg_amt > 0:
                    liq_cap = avg_amt * self.max_buy_pct_of_avg_amount
                    if val < liq_cap - 1e-8:
                        room[inst] = liq_cap - val
            if not room:
                break
            room_total = sum(room.values())
            if room_total <= 0:
                break
            add_amount = min(excess, room_total)
            add_ratio = add_amount / room_total
            for inst in room:
                result[inst] += room[inst] * add_ratio
        return result

    def _is_buy_blocked(self, row: pd.Series) -> bool:
        if pd.isna(row.get(self.deal_price_col)) or float(row.get(self.deal_price_col)) <= 0:
            return True
        if pd.notna(row.get("vol")) and float(row.get("vol")) <= 0:
            return True
        change = self._change_rate(row)
        limit = self._limit_threshold_for(row.name if hasattr(row, 'name') else row.get("ts_code", ""), row)
        if self.forbid_buy_limit_up and change >= limit:
            return True
        if self.forbid_buy_limit_down and change <= -limit:
            return True
        return False

    def _is_sell_blocked(self, row: pd.Series) -> bool:
        if pd.isna(row.get(self.deal_price_col)) or float(row.get(self.deal_price_col)) <= 0:
            return True
        if pd.notna(row.get("vol")) and float(row.get("vol")) <= 0:
            return True
        change = self._change_rate(row)
        limit = self._limit_threshold_for(row.name if hasattr(row, 'name') else row.get("ts_code", ""), row)
        if self.forbid_sell_limit_down and change <= -limit:
            return True
        return False

    def _limit_threshold_for(self, instrument: str, row: pd.Series | None = None) -> float:
        """Return the price limit threshold for a stock based on its board and ST status."""
        if not self.board_specific_limits:
            return self.limit_threshold

        ts_code = str(instrument) if instrument else ""
        if not ts_code:
            return self.limit_threshold

        # Determine board by code prefix
        # Main board: 60*, 000, 001, 002, 003 -> 10%
        # ChiNext (创业板): 30* -> 20%
        # STAR (科创板): 68* -> 20%
        # Beijing Exchange (北交所): 8*, 92* -> 30%
        is_st = False
        if row is not None:
            is_st = bool(row.get("st_stock", 0)) or bool(row.get("star_st_stock", 0))

        # Beijing Exchange
        if ts_code.startswith(("8", "92")):
            return 0.29  # 30%, with small buffer
        # STAR board
        if ts_code.startswith("68"):
            return 0.19  # 20%
        # ChiNext board
        if ts_code.startswith("30"):
            return 0.19  # 20%
        # Main board (default)
        if is_st:
            return 0.049  # 5% for ST stocks on main board
        return 0.095  # 10% for normal main board stocks

    def _change_rate(self, row: pd.Series) -> float:
        pct_chg = row.get("pct_chg")
        if pd.notna(pct_chg):
            return float(pct_chg) / 100.0
        price = row.get(self.deal_price_col)
        pre_close = row.get("pre_close")
        if pd.notna(price) and pd.notna(pre_close) and float(pre_close) > 0:
            return float(price) / float(pre_close) - 1.0
        return 0.0

    def _calc_fee(self, raw_value: float, rate: float) -> float:
        if raw_value <= 0 or rate <= 0:
            return 0.0
        return max(raw_value * rate, self.min_cost)

    def _max_affordable_buy_qty(self, cash: float, raw_price: float, desired_qty: int) -> int:
        qty = self._round_lot_down(desired_qty)
        while qty > 0:
            raw_value = raw_price * qty
            total_cash = raw_price * (1.0 + self.slippage) * qty + self._calc_fee(raw_value, self.open_cost)
            if total_cash <= cash + 1e-8:
                return qty
            qty -= self.lot_size
        return 0

    def _round_lot_down(self, shares: int | float) -> int:
        if self.lot_size <= 1:
            return int(np.floor(shares))
        return int(np.floor(shares / self.lot_size) * self.lot_size)

    @staticmethod
    def _total_shares(lots: list[dict[str, Any]]) -> int:
        return int(sum(int(lot.get("shares", 0)) for lot in lots))

    @staticmethod
    def _available_sell_shares(lots: list[dict[str, Any]], trade_date: pd.Timestamp) -> int:
        return int(sum(int(lot.get("shares", 0)) for lot in lots if pd.Timestamp(lot.get("buy_date")) < trade_date))

    def _remove_shares_fifo(self, lots: list[dict[str, Any]], shares: int, trade_date: pd.Timestamp) -> None:
        remaining = int(shares)
        for lot in lots:
            if remaining <= 0:
                break
            if pd.Timestamp(lot.get("buy_date")) >= trade_date:
                continue
            take = min(int(lot.get("shares", 0)), remaining)
            lot["shares"] = int(lot.get("shares", 0)) - take
            remaining -= take
        lots[:] = [lot for lot in lots if int(lot.get("shares", 0)) > 0]
        if remaining > 0:
            raise RuntimeError("Attempted to sell more T+1 available shares than held")

    def _stock_value(
        self,
        holdings: Mapping[str, list[dict[str, Any]]],
        prices: pd.Series,
        last_close: Mapping[str, float] | None = None,
    ) -> float:
        value = 0.0
        for instrument, lots in holdings.items():
            price = prices.get(instrument, np.nan)
            if pd.isna(price) and last_close is not None:
                price = last_close.get(instrument, np.nan)
            if pd.isna(price):
                continue
            value += self._total_shares(lots) * float(price)
        return float(value)

    @staticmethod
    def _trade_record(
        trade_date: pd.Timestamp,
        instrument: str,
        side: str,
        shares: int,
        raw_price: float,
        exec_price: float,
        raw_value: float,
        fee: float,
        slippage_cost: float,
        cash_after: float,
    ) -> dict[str, Any]:
        return {
            "datetime": trade_date,
            "instrument": instrument,
            "side": side,
            "shares": int(shares),
            "raw_price": float(raw_price),
            "exec_price": float(exec_price),
            "raw_value": float(raw_value),
            "fee": float(fee),
            "slippage_cost": float(slippage_cost),
            "cost": float(fee + slippage_cost),
            "cash_after": float(cash_after),
        }

    def _analyze(self, report_df: pd.DataFrame, trades_df: pd.DataFrame) -> dict[str, Any]:
        if report_df.empty:
            return {}

        returns = report_df["return"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        benchmark = report_df["benchmark"].astype(float).replace([np.inf, -np.inf], np.nan)
        excess = report_df["excess_return"].astype(float).replace([np.inf, -np.inf], np.nan)
        return {
            "return": self._metric_dict(returns),
            "benchmark": self._metric_dict(benchmark.dropna()),
            "excess_return": self._metric_dict(excess.dropna()),
            "risk": self._relative_metric_dict(returns, benchmark),
            "summary": {
                "start_time": str(report_df.index.min().date()),
                "end_time": str(report_df.index.max().date()),
                "trading_days": int(len(report_df)),
                "initial_account_value": float(self.account),
                "final_account_value": float(report_df["account_value"].iloc[-1]),
                "total_return": float(report_df["account_value"].iloc[-1] / self.account - 1.0),
                "benchmark_total_return": float((1.0 + benchmark.dropna()).prod() - 1.0) if benchmark.notna().any() else None,
                "mean_turnover": float(report_df["turnover"].mean()),
                "total_cost": float(report_df["cost"].sum()),
                "total_fee": float(report_df["fee"].sum()),
                "total_slippage_cost": float(report_df["slippage_cost"].sum()),
                "trade_count": int(len(trades_df)) if trades_df is not None else 0,
                "rebalance_days": int(report_df["is_rebalance_date"].sum()) if "is_rebalance_date" in report_df else None,
                "rebalance_frequency": str(self.freq),
                "rebalance_interval_trading_days": self._trading_day_interval_from_freq(),
                "signal_delay": int(self.signal_delay),
                "price_adjustment": self.price_adjustment,
                "mean_holdings_count": float(report_df["holdings_count"].mean()),
                "deal_price": self.deal_price_col,
                "benchmark": self.benchmark,
            },
            "config": self.config,
        }

    def _metric_dict(self, returns: pd.Series) -> dict[str, float | None]:
        returns = returns.dropna()
        if returns.empty:
            return {}
        ann = self._annualization_factor()
        nav = (1.0 + returns).cumprod()
        total_return = float(nav.iloc[-1] - 1.0)
        annualized_return = float(nav.iloc[-1] ** (ann / len(returns)) - 1.0) if len(returns) > 0 and nav.iloc[-1] > 0 else None
        vol = float(returns.std(ddof=1) * np.sqrt(ann)) if len(returns) > 1 else 0.0
        sharpe = float((returns.mean() * ann) / vol) if vol > 0 else None
        downside = returns[returns < 0]
        downside_vol = float(downside.std(ddof=1) * np.sqrt(ann)) if len(downside) > 1 else 0.0
        sortino = float((returns.mean() * ann) / downside_vol) if downside_vol > 0 else None
        drawdown = nav / nav.cummax() - 1.0
        max_drawdown = float(drawdown.min())
        calmar = float(annualized_return / abs(max_drawdown)) if annualized_return is not None and max_drawdown < 0 else None
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_drawdown,
            "calmar": calmar,
            "win_rate": float((returns > 0).mean()),
            "mean_daily_return": float(returns.mean()),
            "median_daily_return": float(returns.median()),
            "best_daily_return": float(returns.max()),
            "worst_daily_return": float(returns.min()),
            "skew": float(returns.skew()) if len(returns) > 2 else None,
            "kurtosis": float(returns.kurtosis()) if len(returns) > 3 else None,
        }

    def _relative_metric_dict(self, returns: pd.Series, benchmark: pd.Series) -> dict[str, float | None]:
        aligned = pd.concat([returns.rename("return"), benchmark.rename("benchmark")], axis=1).dropna()
        if aligned.empty:
            return {}
        ann = self._annualization_factor()
        r = aligned["return"]
        b = aligned["benchmark"]
        excess = r - b
        r_vol = float(r.std(ddof=1) * np.sqrt(ann)) if len(r) > 1 else 0.0
        b_vol = float(b.std(ddof=1) * np.sqrt(ann)) if len(b) > 1 else 0.0
        benchmark_var = float(b.var(ddof=1)) if len(b) > 1 else 0.0
        beta = float(r.cov(b) / benchmark_var) if benchmark_var > 0 else None
        alpha_daily = float(r.mean() - (beta or 0.0) * b.mean()) if beta is not None else None
        alpha_annual = float(alpha_daily * ann) if alpha_daily is not None else None
        tracking_error = float(excess.std(ddof=1) * np.sqrt(ann)) if len(excess) > 1 else 0.0
        information_ratio = float((excess.mean() * ann) / tracking_error) if tracking_error > 0 else None
        return {
            "sharpe": float((r.mean() * ann) / r_vol) if r_vol > 0 else None,
            "benchmark_sharpe": float((b.mean() * ann) / b_vol) if b_vol > 0 else None,
            "alpha": alpha_annual,
            "alpha_daily": alpha_daily,
            "beta": beta,
            "excess_total_return": float((1.0 + r).prod() - (1.0 + b).prod()),
            "excess_annualized_return": float(excess.mean() * ann),
            "tracking_error": tracking_error,
            "information_ratio": information_ratio,
            "correlation": float(r.corr(b)) if len(aligned) > 1 else None,
        }

    def _annualization_factor(self) -> int:
        # The report is marked to market on every trading day regardless of the
        # rebalance frequency (``freq``).  Annualizing daily returns with 52/12
        # when the portfolio rebalances weekly/monthly would mix rebalance
        # cadence with return sampling frequency and distort Sharpe/volatility.
        return 252

    @staticmethod
    def _json_safe(obj):
        if isinstance(obj, dict):
            return {str(k): SimpleBacktester._json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [SimpleBacktester._json_safe(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, Path):
            return str(obj)
        if obj is None:
            return None
        try:
            if pd.isna(obj):
                return None
        except (TypeError, ValueError):
            pass
        return obj

    def save_outputs(
        self,
        report_df: pd.DataFrame,
        positions_df: pd.DataFrame,
        trades_df: pd.DataFrame,
        analysis: Mapping[str, Any],
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_df.to_parquet(self.output_dir / "backtest_report.parquet")
        if self.save_positions and not positions_df.empty:
            positions_df.to_parquet(self.output_dir / "positions.parquet")
        if not trades_df.empty:
            trades_df.to_parquet(self.output_dir / "trades.parquet")
        with (self.output_dir / "analysis.json").open("w", encoding="utf-8") as f:
            json.dump(self._json_safe(dict(analysis)), f, ensure_ascii=False, indent=2)


__all__ = ["SimpleBacktester"]
