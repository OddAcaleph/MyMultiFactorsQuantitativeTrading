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

    def __init__(self, config_path: str | Path | None = None, **overrides: Any) -> None:
        self.config_path = config_path
        self.config = self._apply_overrides(load_simple_backtester_config(config_path), overrides)

        self.trainer_config_path = self.config.get("trainer_config_path")
        self.trainer_config = load_trainer_config(self.trainer_config_path) if self.trainer_config_path else {}

        self.prediction_path = Path(self.config["prediction_path"])
        self.price_path = Path(self.config.get("price_path", self.DEFAULT_PRICE_PATH))
        self.benchmark_path = Path(self.config.get("benchmark_path", self.DEFAULT_BENCHMARK_PATH))
        self.benchmark_lookback_days = int(self.config.get("benchmark_lookback_days", 31))
        self.output_dir = Path(self.config.get("output_dir", PROJECT_ROOT / "outputs" / "simple_backtest"))
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
        self.lot_size = int(self.exchange_kwargs.get("lot_size", 100))

        forbid_all = bool(self.strategy_config.get("forbid_all_trade_at_limit", False))
        self.forbid_buy_limit_up = forbid_all or bool(self.exchange_kwargs.get("forbid_buy_limit_up", True))
        self.forbid_buy_limit_down = forbid_all or bool(self.exchange_kwargs.get("forbid_buy_limit_down", True))
        self.forbid_sell_limit_down = forbid_all or bool(self.exchange_kwargs.get("forbid_sell_limit_down", False))

        self.start_time, self.end_time = self._resolve_oos_period()

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

        pred_df = self.load_predictions() if pred_df is None else pred_df
        price_df = self.load_prices() if price_df is None else price_df
        benchmark_returns = self.load_benchmark_returns() if benchmark_returns is None else benchmark_returns
        report_df, positions_df, trades_df = self._simulate(pred_df=pred_df, price_df=price_df, benchmark_returns=benchmark_returns)
        analysis = self._analyze(report_df=report_df, trades_df=trades_df)

        if save:
            self.save_outputs(report_df=report_df, positions_df=positions_df, trades_df=trades_df, analysis=analysis)

        return {"report": report_df, "positions": positions_df, "trades": trades_df, "analysis": analysis}

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

        files = self._parquet_files_for_period(self.price_path)
        if not files:
            raise FileNotFoundError(
                f"No daily bar parquet files found in {self.price_path} for {self.start_time.date()} ~ {self.end_time.date()}"
            )

        frames = []
        required = ["trade_date", "ts_code", "open", "close", "pre_close", "pct_chg", "vol", "amount"]
        optional = ["adj_factor"]
        for file_path in files:
            frame = pd.read_parquet(file_path)
            missing = [col for col in required if col not in frame.columns]
            if missing:
                raise KeyError(f"Daily bar file {file_path} missing required columns: {missing}")
            frames.append(frame[required + [col for col in optional if col in frame.columns]])

        price_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=required)
        price_df = price_df.rename(columns={"trade_date": "datetime", "ts_code": "instrument"})
        price_df["datetime"] = pd.to_datetime(price_df["datetime"].astype(str), format="%Y%m%d", errors="coerce")
        price_df = price_df.dropna(subset=["datetime", "instrument", "open", "close"])
        mask = (price_df["datetime"] >= self.start_time) & (price_df["datetime"] <= self.end_time)
        price_df = price_df.loc[mask].copy()
        for col in ["open", "close", "pre_close", "pct_chg", "vol", "amount", "adj_factor"]:
            if col not in price_df.columns:
                continue
            price_df[col] = pd.to_numeric(price_df[col], errors="coerce")
        price_df = price_df.set_index(["datetime", "instrument"]).sort_index()
        return self._apply_price_adjustment(price_df)

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

    def _parquet_files_for_period(self, root: Path) -> list[Path]:
        return self._parquet_files_between(root=root, start_time=self.start_time, end_time=self.end_time)

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
        """Load benchmark returns from qlib ``*.day.bin`` files.

        Stock execution data still comes from ``wide_table_daily_bars``.  Only
        the benchmark is read from qlib format, using ``change.day.bin`` when it
        exists and falling back to ``close.day.bin`` pct-change otherwise.
        """

        features_dir = self._resolve_qlib_features_dir(self.benchmark_path)
        benchmark_code = self._normalize_benchmark_code(self.benchmark)
        qlib_instrument = self._benchmark_to_qlib_instrument(benchmark_code)
        instrument_dir = features_dir / qlib_instrument
        if not instrument_dir.exists():
            raise FileNotFoundError(
                f"Benchmark qlib directory does not exist: {instrument_dir}. "
                f"Resolved benchmark {self.benchmark!r} -> {benchmark_code} -> {qlib_instrument}. "
                "Please choose a benchmark available under the qlib features directory."
            )

        benchmark_start_time = self.start_time - pd.Timedelta(days=self.benchmark_lookback_days)
        change_file = instrument_dir / "change.day.bin"
        if change_file.exists():
            benchmark_series = self._read_qlib_day_bin(change_file, features_dir).astype(float)
        else:
            close_file = instrument_dir / "close.day.bin"
            if not close_file.exists():
                raise FileNotFoundError(
                    f"Benchmark qlib files are missing: neither {change_file} nor {close_file} exists"
                )
            benchmark_series = self._read_qlib_day_bin(close_file, features_dir).astype(float).pct_change()

        benchmark_series = benchmark_series.replace([np.inf, -np.inf], np.nan).dropna()
        benchmark_series = benchmark_series.loc[
            (benchmark_series.index >= benchmark_start_time) & (benchmark_series.index <= self.end_time)
        ]
        if benchmark_series.empty:
            raise ValueError(
                f"Benchmark {benchmark_code} ({qlib_instrument}) has no qlib return data after date filtering for "
                f"{benchmark_start_time.date()} ~ {self.end_time.date()} under {instrument_dir}"
            )
        benchmark_series.name = "benchmark"
        return benchmark_series.sort_index()

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
            prices = day_price_df.droplevel("datetime")[["open", "close", "pre_close", "pct_chg", "vol", "amount"]]
            executable_prices = prices[self.deal_price_col]
            valid_price_names = set(prices.dropna(subset=[self.deal_price_col, "close"]).index)

            if is_rebalance_date:
                signal_date = signal_date_by_trade_date.get(trade_date)
                if signal_date is None:
                    day_df = pd.DataFrame(columns=[self.score_col, "open", "close", "pre_close", "pct_chg", "vol", "amount"])
                else:
                    day_scores = pred_df.xs(signal_date, level="datetime")[[self.score_col]]
                    day_df = day_scores.join(prices, how="inner").sort_values(self.score_col, ascending=False)
                current_value_at_deal = cash + self._stock_value(holdings, executable_prices)
                target_names = self._select_target_names(
                    day_df=day_df,
                    holdings=holdings,
                    prices=prices,
                    trade_date=trade_date,
                    topk=topk,
                    n_drop=n_drop,
                )
                target_shares = self._calc_target_shares(
                    target_names=target_names,
                    target_value=current_value_at_deal,
                    executable_prices=executable_prices,
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
                    holdings.setdefault(instrument, []).append({"shares": int(buy_qty), "buy_date": trade_date})
                    slippage_cost = gross_cash - raw_value
                    day_buy_value += raw_value
                    day_fee += fee
                    day_slippage += slippage_cost
                    trades.append(
                        self._trade_record(
                            trade_date, instrument, "buy", buy_qty, raw_price, exec_price, raw_value, fee, slippage_cost, cash
                        )
                    )

            stock_value = self._stock_value(holdings, prices["close"])
            account_value = cash + stock_value
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
                    if shares <= 0 or instrument not in prices.index or pd.isna(prices.loc[instrument, "close"]):
                        continue
                    close_price = float(prices.loc[instrument, "close"])
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
            scored_sellable = scores.loc[list(sellable_current.intersection(scores.index))]
            to_drop = set(scored_sellable.sort_values(ascending=True).head(n_drop).index)

        target = [inst for inst in ranked_desc if inst in current and inst not in to_drop]
        # Keep non-sellable existing holdings even if they are not in current top scores.
        for inst in sorted(current - set(target) - to_drop):
            if inst in prices.index and self._available_sell_shares(holdings.get(inst, []), trade_date) <= 0:
                target.append(inst)

        target_set = set(target)
        for inst in ranked_desc:
            if len(target) >= topk:
                break
            if inst in target_set:
                continue
            if inst not in prices.index or self._is_buy_blocked(prices.loc[inst]):
                continue
            target.append(inst)
            target_set.add(inst)
        return target[:topk]

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

    def _is_buy_blocked(self, row: pd.Series) -> bool:
        if pd.isna(row.get(self.deal_price_col)) or float(row.get(self.deal_price_col)) <= 0:
            return True
        if pd.notna(row.get("vol")) and float(row.get("vol")) <= 0:
            return True
        if self.forbid_buy_limit_up and self._change_rate(row) >= self.limit_threshold:
            return True
        if self.forbid_buy_limit_down and self._change_rate(row) <= -self.limit_threshold:
            return True
        return False

    def _is_sell_blocked(self, row: pd.Series) -> bool:
        if pd.isna(row.get(self.deal_price_col)) or float(row.get(self.deal_price_col)) <= 0:
            return True
        if pd.notna(row.get("vol")) and float(row.get("vol")) <= 0:
            return True
        if self.forbid_sell_limit_down and self._change_rate(row) <= -self.limit_threshold:
            return True
        return False

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

    def _stock_value(self, holdings: Mapping[str, list[dict[str, Any]]], prices: pd.Series) -> float:
        value = 0.0
        for instrument, lots in holdings.items():
            price = prices.get(instrument, np.nan)
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
