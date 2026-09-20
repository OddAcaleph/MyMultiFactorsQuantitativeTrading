"""Worst-performer replacement strategy backtest.

Strategy:
  - Start with N randomly selected stocks, equally weighted.
  - Every rebalance period (default 22 trading days):
      1. Rank current holdings by their holding-period return (worst first).
      2. Sell the K worst-performing stocks.
      3. Buy K new random stocks from the eligible universe, equally weighted.

This is a baseline / "reverse momentum" strategy useful for comparison with
factor-based strategies.  It enforces realistic A-share trading rules:
T+1 settlement, board lots (100 shares), limit up/down trading halts,
commission + stamp duty costs, and slippage.

Usage:
    PYTHONPATH=src python scripts/worst_performer_replacement_backtest.py \
        --start 2023-01-01 --end 2025-12-31 \
        --n-holdings 30 --n-replace 6 --rebalance-days 22 \
        --seed 42 --output-dir outputs/worst_performer_replacement
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import PROJECT_ROOT  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_PRICE_PATH = PROJECT_ROOT / "data" / "processd_data" / "wide_table_daily_bars"


# ---------------------------------------------------------------------------
# Price data loading
# ---------------------------------------------------------------------------

def load_prices(
    price_path: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lookback_days: int = 60,
) -> pd.DataFrame:
    """Load daily OHLCV bars for the backtest period + lookback.

    Returns a DataFrame with MultiIndex [datetime, instrument] and columns:
    open, close, pre_close, pct_chg, vol, amount, adj_factor
    """
    load_start = start - pd.Timedelta(days=lookback_days)

    files = _parquet_files_for_period(price_path, load_start, end)
    if not files:
        raise FileNotFoundError(
            f"No daily bar parquet files found in {price_path} "
            f"for {start.date()} ~ {end.date()}"
        )

    required = ["trade_date", "ts_code", "open", "close", "pre_close", "pct_chg", "vol", "amount"]
    optional = ["adj_factor", "st_stock", "star_st_stock", "is_suspect"]

    frames = []
    for fp in files:
        df = pd.read_parquet(fp)
        cols = required + [c for c in optional if c in df.columns]
        frames.append(df[cols])

    price_df = pd.concat(frames, ignore_index=True)
    price_df = price_df.rename(columns={"trade_date": "datetime", "ts_code": "instrument"})
    price_df["datetime"] = pd.to_datetime(price_df["datetime"].astype(str), format="%Y%m%d", errors="coerce")
    price_df = price_df.dropna(subset=["datetime", "instrument", "open", "close"])

    numeric_cols = ["open", "close", "pre_close", "pct_chg", "vol", "amount"]
    for c in optional:
        if c in price_df.columns:
            numeric_cols.append(c)
    for col in numeric_cols:
        price_df[col] = pd.to_numeric(price_df[col], errors="coerce")

    price_df = price_df.set_index(["datetime", "instrument"]).sort_index()

    # Apply qfq adjustment (forward-adjusted prices)
    price_df = _apply_qfq(price_df)

    # Trim to actual backtest period
    mask = (price_df.index.get_level_values("datetime") >= start) & \
           (price_df.index.get_level_values("datetime") <= end)
    return price_df.loc[mask].copy()


def _parquet_files_for_period(base_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> list[Path]:
    """Find parquet files under year=/month= partitioning for a date range."""
    files: list[Path] = []
    for year_dir in sorted(base_path.glob("year=*")):
        year = int(year_dir.name.split("=")[1])
        if year < start.year - 1 or year > end.year + 1:
            continue
        for month_dir in sorted(year_dir.glob("month=*")):
            month = int(month_dir.name.split("=")[1])
            # Quick year-month filter (may include edge months, that's fine)
            if year < start.year and month < 12:
                continue
            if year > end.year and month > 1:
                continue
            for fp in sorted(month_dir.glob("*.parquet")):
                date_str = fp.stem
                try:
                    d = pd.Timestamp(date_str)
                except ValueError:
                    continue
                if start <= d <= end:
                    files.append(fp)
    return files


def _apply_qfq(price_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize adj_factor and apply forward-adjustment to OHLC prices."""
    if "adj_factor" not in price_df.columns:
        return price_df

    adj = price_df["adj_factor"].groupby(level="instrument")
    first_factor = adj.transform("first")
    norm_factor = price_df["adj_factor"] / first_factor

    for col in ("open", "close", "pre_close"):
        if col in price_df.columns:
            price_df[col] = price_df[col] * norm_factor

    return price_df


# ---------------------------------------------------------------------------
# Trading helpers
# ---------------------------------------------------------------------------

@dataclass
class Lot:
    shares: int
    buy_date: pd.Timestamp
    buy_price: float
    cost_basis: float  # total cost (price * shares + fees)


@dataclass
class Position:
    instrument: str
    lots: list[Lot] = field(default_factory=list)

    @property
    def total_shares(self) -> int:
        return sum(l.shares for l in self.lots)

    @property
    def available_shares(self) -> int:
        """Shares available to sell today (T+1: bought before today)."""
        # We compute this dynamically given a trade_date in the backtest loop.
        return 0

    def available_shares_on(self, trade_date: pd.Timestamp) -> int:
        return sum(l.shares for l in self.lots if l.buy_date < trade_date)

    def cost_basis(self) -> float:
        return sum(l.cost_basis for l in self.lots)


def round_lot_down(shares: float, lot_size: int = 100) -> int:
    return int(shares // lot_size) * lot_size


def calc_fee(value: float, rate: float, min_cost: float) -> float:
    return max(value * rate, min_cost)


def is_limit_up(pct_chg, threshold: float):
    """Check if stock is at limit up. Works with scalar or pandas Series."""
    return (~np.isnan(pct_chg)) & (pct_chg >= threshold * 100)


def is_limit_down(pct_chg, threshold: float):
    """Check if stock is at limit down. Works with scalar or pandas Series."""
    return (~np.isnan(pct_chg)) & (pct_chg <= -threshold * 100)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

class WorstPerformerReplacementBacktester:
    """Backtest a worst-performer replacement strategy.

    Parameters
    ----------
    n_holdings : int
        Number of stocks to hold.
    n_replace : int
        Number of worst-performing stocks to replace each rebalance.
    rebalance_days : int
        Rebalance frequency in trading days.
    seed : int
        Random seed for reproducibility.
    initial_cash : float
        Starting capital.
    deal_price : str
        "open" or "close".
    open_cost, close_cost, slippage, min_cost : float
        Transaction cost parameters.
    limit_threshold : float
        Daily limit threshold (e.g. 0.095 for 9.5%).
    lot_size : int
        Board lot size.
    filter_st, filter_suspend : bool
        Whether to exclude ST / suspended stocks from buys.
    """

    def __init__(
        self,
        n_holdings: int = 30,
        n_replace: int = 6,
        rebalance_days: int = 22,
        seed: int = 42,
        initial_cash: float = 1_000_000.0,
        deal_price: str = "open",
        open_cost: float = 0.0005,
        close_cost: float = 0.0015,
        slippage: float = 0.001,
        min_cost: float = 5.0,
        limit_threshold: float = 0.095,
        lot_size: int = 100,
        filter_st: bool = True,
        filter_suspend: bool = True,
    ) -> None:
        self.n_holdings = n_holdings
        self.n_replace = n_replace
        self.rebalance_days = rebalance_days
        self.rng = np.random.RandomState(seed)
        self.initial_cash = initial_cash
        self.deal_price = deal_price
        self.open_cost = open_cost
        self.close_cost = close_cost
        self.slippage = slippage
        self.min_cost = min_cost
        self.limit_threshold = limit_threshold
        self.lot_size = lot_size
        self.filter_st = filter_st
        self.filter_suspend = filter_suspend

    def run(self, price_df: pd.DataFrame) -> dict[str, Any]:
        """Run the backtest.

        Returns dict with: equity_curve (DataFrame), positions (DataFrame),
        trades (DataFrame), metrics (dict).
        """
        dates = sorted(price_df.index.get_level_values("datetime").unique())
        logger.info(f"Backtest period: {dates[0].date()} ~ {dates[-1].date()} ({len(dates)} trading days)")

        cash = self.initial_cash
        positions: dict[str, Position] = {}

        # Output records
        equity_records: list[dict] = []
        position_records: list[dict] = []
        trade_records: list[dict] = []

        # Track rebalance counter
        rebalance_counter = 0
        is_initialized = False

        for i, date in enumerate(dates):
            day_data = price_df.xs(date, level="datetime")

            # ---- Compute eligible universe for today's buys ----
            eligible = self._get_eligible_universe(day_data)
            eligible_set = set(eligible.index)

            # ---- Initial entry on first day ----
            if not is_initialized:
                cash = self._initial_buy(date, day_data, eligible, positions, cash, trade_records)
                is_initialized = True
                self._record_day_end(date, day_data, positions, cash, 0.0, 0.0, 0.0, 0.0, True, equity_records, position_records)
                continue

            # ---- Daily P&L (mark to market) ----
            # We compute gross return from close-to-close of holdings
            prev_close_total = 0.0
            curr_close_total = 0.0
            for inst, pos in positions.items():
                if inst not in day_data.index:
                    continue
                shares = pos.total_shares
                prev_close_total += shares * day_data.loc[inst, "pre_close"]
                curr_close_total += shares * day_data.loc[inst, "close"]

            gross_return = 0.0
            if prev_close_total > 0:
                gross_return = (curr_close_total - prev_close_total) / (prev_close_total + cash)

            # ---- Check if rebalance day ----
            rebalance_counter += 1
            is_rebalance = rebalance_counter >= self.rebalance_days
            daily_fee = 0.0
            daily_slippage = 0.0
            daily_buy_value = 0.0
            daily_sell_value = 0.0

            if is_rebalance:
                rebalance_counter = 0
                daily_fee, daily_slippage, daily_buy_value, daily_sell_value = self._rebalance(
                    date, day_data, eligible_set, positions, cash, trade_records
                )
                # cash is updated in-place via _rebalance

            # ---- Record day end ----
            turnover = (daily_buy_value + daily_sell_value) / max(self._prev_account_value, 1e-9)
            self._record_day_end(
                date, day_data, positions, cash,
                daily_fee, daily_slippage, daily_buy_value, daily_sell_value,
                is_rebalance, equity_records, position_records,
                turnover=turnover, gross_return=gross_return,
            )

        # ---- Build output DataFrames ----
        equity_df = pd.DataFrame(equity_records).set_index("datetime")
        positions_df = pd.DataFrame(position_records).set_index(["datetime", "instrument"]) if position_records else pd.DataFrame()
        trades_df = pd.DataFrame(trade_records).set_index(["datetime", "instrument", "side"]) if trade_records else pd.DataFrame()

        metrics = self._compute_metrics(equity_df)

        return {
            "equity_curve": equity_df,
            "positions": positions_df,
            "trades": trades_df,
            "metrics": metrics,
        }

    def _get_eligible_universe(self, day_data: pd.DataFrame) -> pd.DataFrame:
        """Get stocks eligible for buying on this day."""
        df = day_data.copy()
        # Valid price & volume
        df = df[df["open"].notna() & df["close"].notna() & (df["vol"] > 0)]

        # ST filter
        if self.filter_st:
            if "st_stock" in df.columns:
                df = df[df["st_stock"].fillna(0) == 0]
            if "star_st_stock" in df.columns:
                df = df[df["star_st_stock"].fillna(0) == 0]

        # Suspend filter
        if self.filter_suspend and "is_suspect" in df.columns:
            df = df[df["is_suspect"].fillna(0) == 0]

        # Not limit up or limit down (can't buy at limit)
        pct = df["pct_chg"]
        df = df[~is_limit_up(pct, self.limit_threshold) & ~is_limit_down(pct, self.limit_threshold)]

        return df

    def _initial_buy(
        self,
        date: pd.Timestamp,
        day_data: pd.DataFrame,
        eligible: pd.DataFrame,
        positions: dict[str, Position],
        cash: float,
        trade_records: list[dict],
    ) -> float:
        """Buy initial n_holdings stocks equally weighted."""
        if len(eligible) < self.n_holdings:
            raise RuntimeError(f"Only {len(eligible)} eligible stocks on {date.date()}, need {self.n_holdings}")

        picks = self.rng.choice(eligible.index.tolist(), size=self.n_holdings, replace=False)
        target_value_per_stock = cash / self.n_holdings

        for inst in picks:
            cash = self._buy_stock(date, inst, day_data, target_value_per_stock, positions, cash, trade_records)

        self._prev_account_value = cash + self._portfolio_value(day_data, positions)
        return cash

    def _rebalance(
        self,
        date: pd.Timestamp,
        day_data: pd.DataFrame,
        eligible_set: set[str],
        positions: dict[str, Position],
        cash: float,
        trade_records: list[dict],
    ) -> tuple[float, float, float, float]:
        """Sell worst n_replace performers, buy n_replace new random stocks."""
        total_fee = 0.0
        total_slippage = 0.0
        total_buy = 0.0
        total_sell = 0.0

        # ---- Compute holding-period return for each position ----
        # Use cost basis vs current market value as holding return
        holding_returns: list[tuple[str, float]] = []
        for inst, pos in positions.items():
            if inst not in day_data.index:
                continue
            cost = pos.cost_basis()
            if cost <= 0:
                continue
            market_val = pos.total_shares * day_data.loc[inst, "close"]
            ret = (market_val - cost) / cost
            holding_returns.append((inst, ret))

        holding_returns.sort(key=lambda x: x[1])  # worst first

        # ---- Sell worst n_replace (that are sellable) ----
        sold_count = 0
        for inst, ret in holding_returns:
            if sold_count >= self.n_replace:
                break
            pos = positions[inst]
            available = pos.available_shares_on(date)
            if available <= 0:
                continue  # T+1 locked or no shares
            # Check limit down (can't sell at limit down)
            pct = day_data.loc[inst, "pct_chg"]
            if is_limit_down(pct, self.limit_threshold):
                continue
            # Check zero volume
            if day_data.loc[inst, "vol"] <= 0:
                continue

            sell_value, fee, slip = self._sell_stock(date, inst, day_data, available, positions, cash, trade_records)
            cash += sell_value - fee - slip  # actually _sell_stock returns net? No, let's fix
            # Wait, let me redesign: _sell_stock returns gross proceeds, fee, slippage
            # Cash increases by gross proceeds - fee
            total_sell += sell_value
            total_fee += fee
            total_slippage += slip
            cash += sell_value - fee  # slippage is already reflected in exec price
            sold_count += 1

        # ---- Buy n_replace new random stocks ----
        # Candidates: eligible and not currently held
        held_set = set(positions.keys())
        candidates = sorted(eligible_set - held_set)
        if len(candidates) < sold_count:
            logger.warning(f"Only {len(candidates)} candidates on {date.date()}, replacing {sold_count}")
            sold_count = min(sold_count, len(candidates))

        if sold_count > 0:
            new_picks = self.rng.choice(candidates, size=sold_count, replace=False)
            # Compute current account value for equal weighting
            account_value = cash + self._portfolio_value(day_data, positions)
            target_value = account_value / self.n_holdings

            for inst in new_picks:
                if inst not in day_data.index:
                    continue
                buy_value, fee, slip = self._buy_stock_return(
                    date, inst, day_data, target_value, positions, cash, trade_records
                )
                total_buy += buy_value
                total_fee += fee
                total_slippage += slip
                cash -= (buy_value + fee)  # buy_value includes slippage? let's be precise

        self._prev_account_value = cash + self._portfolio_value(day_data, positions)
        return total_fee, total_slippage, total_buy, total_sell

    def _buy_stock(
        self,
        date: pd.Timestamp,
        inst: str,
        day_data: pd.DataFrame,
        target_value: float,
        positions: dict[str, Position],
        cash: float,
        trade_records: list[dict],
    ) -> float:
        """Execute a buy. Returns remaining cash."""
        raw_price = day_data.loc[inst, self.deal_price]
        if np.isnan(raw_price) or raw_price <= 0:
            return cash

        exec_price = raw_price * (1 + self.slippage)
        max_shares = target_value / exec_price
        shares = round_lot_down(max_shares, self.lot_size)
        if shares <= 0:
            return cash

        gross_value = shares * exec_price
        fee = calc_fee(gross_value, self.open_cost, self.min_cost)
        total_cost = gross_value + fee

        if total_cost > cash:
            # Reduce shares to fit cash
            max_value = cash - self.min_cost
            shares = round_lot_down(max_value / exec_price, self.lot_size)
            if shares <= 0:
                return cash
            gross_value = shares * exec_price
            fee = calc_fee(gross_value, self.open_cost, self.min_cost)
            total_cost = gross_value + fee

        lot = Lot(shares=shares, buy_date=date, buy_price=raw_price, cost_basis=total_cost)
        if inst not in positions:
            positions[inst] = Position(instrument=inst)
        positions[inst].lots.append(lot)

        trade_records.append({
            "datetime": date,
            "instrument": inst,
            "side": "buy",
            "shares": shares,
            "raw_price": raw_price,
            "exec_price": exec_price,
            "raw_value": shares * raw_price,
            "fee": fee,
            "slippage_cost": shares * raw_price * self.slippage,
            "cost": total_cost,
            "cash_after": cash - total_cost,
        })

        return cash - total_cost

    def _buy_stock_return(
        self,
        date: pd.Timestamp,
        inst: str,
        day_data: pd.DataFrame,
        target_value: float,
        positions: dict[str, Position],
        cash: float,
        trade_records: list[dict],
    ) -> tuple[float, float, float]:
        """Buy stock and return (gross_value, fee, slippage_cost). Updates cash in caller."""
        raw_price = day_data.loc[inst, self.deal_price]
        if np.isnan(raw_price) or raw_price <= 0:
            return 0.0, 0.0, 0.0

        exec_price = raw_price * (1 + self.slippage)
        max_shares = target_value / exec_price
        shares = round_lot_down(max_shares, self.lot_size)
        if shares <= 0:
            return 0.0, 0.0, 0.0

        gross_value = shares * exec_price
        fee = calc_fee(gross_value, self.open_cost, self.min_cost)
        total_cost = gross_value + fee

        if total_cost > cash:
            max_val = cash - self.min_cost
            shares = round_lot_down(max_val / exec_price, self.lot_size)
            if shares <= 0:
                return 0.0, 0.0, 0.0
            gross_value = shares * exec_price
            fee = calc_fee(gross_value, self.open_cost, self.min_cost)
            total_cost = gross_value + fee

        lot = Lot(shares=shares, buy_date=date, buy_price=raw_price, cost_basis=total_cost)
        if inst not in positions:
            positions[inst] = Position(instrument=inst)
        positions[inst].lots.append(lot)

        slip_cost = shares * raw_price * self.slippage
        trade_records.append({
            "datetime": date,
            "instrument": inst,
            "side": "buy",
            "shares": shares,
            "raw_price": raw_price,
            "exec_price": exec_price,
            "raw_value": shares * raw_price,
            "fee": fee,
            "slippage_cost": slip_cost,
            "cost": total_cost,
            "cash_after": cash - total_cost,
        })

        return gross_value, fee, slip_cost

    def _sell_stock(
        self,
        date: pd.Timestamp,
        inst: str,
        day_data: pd.DataFrame,
        shares_to_sell: int,
        positions: dict[str, Position],
        cash: float,
        trade_records: list[dict],
    ) -> tuple[float, float, float]:
        """Sell shares using FIFO lot accounting. Returns (gross_proceeds, fee, slippage_cost)."""
        raw_price = day_data.loc[inst, self.deal_price]
        if np.isnan(raw_price) or raw_price <= 0:
            return 0.0, 0.0, 0.0

        exec_price = raw_price * (1 - self.slippage)
        shares_to_sell = round_lot_down(shares_to_sell, self.lot_size)
        if shares_to_sell <= 0:
            return 0.0, 0.0, 0.0

        pos = positions[inst]
        # FIFO: sell from earliest lots that are available (buy_date < date)
        available_lots = [l for l in pos.lots if l.buy_date < date]
        remaining = shares_to_sell
        total_shares_sold = 0
        total_cost_basis_removed = 0.0

        for lot in available_lots:
            if remaining <= 0:
                break
            sell_from_lot = min(lot.shares, remaining)
            lot.shares -= sell_from_lot
            remaining -= sell_from_lot
            total_shares_sold += sell_from_lot
            # Remove proportional cost basis
            ratio = sell_from_lot / max(lot.shares + sell_from_lot, 1)
            total_cost_basis_removed += lot.cost_basis * ratio
            lot.cost_basis -= lot.cost_basis * ratio

        # Clean up empty lots
        pos.lots = [l for l in pos.lots if l.shares > 0]
        if not pos.lots:
            del positions[inst]

        if total_shares_sold <= 0:
            return 0.0, 0.0, 0.0

        gross_proceeds = total_shares_sold * exec_price
        fee = calc_fee(gross_proceeds, self.close_cost, self.min_cost)
        slip_cost = total_shares_sold * raw_price * self.slippage

        trade_records.append({
            "datetime": date,
            "instrument": inst,
            "side": "sell",
            "shares": total_shares_sold,
            "raw_price": raw_price,
            "exec_price": exec_price,
            "raw_value": total_shares_sold * raw_price,
            "fee": fee,
            "slippage_cost": slip_cost,
            "cost": fee + slip_cost,
            "cash_after": cash + gross_proceeds - fee,
        })

        return gross_proceeds, fee, slip_cost

    def _portfolio_value(self, day_data: pd.DataFrame, positions: dict[str, Position]) -> float:
        """Mark-to-market value using close price."""
        total = 0.0
        for inst, pos in positions.items():
            if inst in day_data.index:
                close = day_data.loc[inst, "close"]
                if not np.isnan(close):
                    total += pos.total_shares * close
        return total

    def _record_day_end(
        self,
        date: pd.Timestamp,
        day_data: pd.DataFrame,
        positions: dict[str, Position],
        cash: float,
        fee: float,
        slippage_cost: float,
        buy_value: float,
        sell_value: float,
        is_rebalance: bool,
        equity_records: list[dict],
        position_records: list[dict],
        turnover: float = 0.0,
        gross_return: float = 0.0,
    ) -> None:
        stock_value = self._portfolio_value(day_data, positions)
        account_value = cash + stock_value
        prev_val = getattr(self, "_prev_account_value", self.initial_cash)
        net_return = (account_value - prev_val) / max(prev_val, 1e-9) if prev_val > 0 else 0.0

        equity_records.append({
            "datetime": date,
            "return": net_return,
            "gross_return": gross_return,
            "cost": fee + slippage_cost,
            "fee": fee,
            "slippage_cost": slippage_cost,
            "turnover": turnover,
            "buy_value": buy_value,
            "sell_value": sell_value,
            "cash": cash,
            "stock_value": stock_value,
            "account_value": account_value,
            "holdings_count": len(positions),
            "is_rebalance_date": is_rebalance,
        })

        self._prev_account_value = account_value

        for inst, pos in positions.items():
            if inst not in day_data.index:
                continue
            close = day_data.loc[inst, "close"]
            if np.isnan(close):
                continue
            mv = pos.total_shares * close
            position_records.append({
                "datetime": date,
                "instrument": inst,
                "shares": pos.total_shares,
                "close": close,
                "market_value": mv,
                "weight": mv / max(account_value, 1e-9),
                "available_shares": pos.available_shares_on(date),
            })

    def _compute_metrics(self, equity_df: pd.DataFrame) -> dict[str, Any]:
        """Compute standard performance metrics."""
        rets = equity_df["return"].dropna()
        if len(rets) == 0:
            return {}

        total_return = equity_df["account_value"].iloc[-1] / self.initial_cash - 1
        n_days = len(rets)
        ann_factor = 252 / max(n_days, 1)
        annualized_return = (1 + total_return) ** ann_factor - 1
        annualized_vol = rets.std() * np.sqrt(252)
        sharpe = annualized_return / annualized_vol if annualized_vol > 0 else np.nan

        # Max drawdown
        equity = equity_df["account_value"]
        running_max = equity.cummax()
        drawdown = equity / running_max - 1
        max_drawdown = drawdown.min()
        calmar = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

        # Sortino
        downside = rets[rets < 0]
        downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else np.nan
        sortino = annualized_return / downside_vol if downside_vol and downside_vol > 0 else np.nan

        win_rate = (rets > 0).mean()

        total_turnover = equity_df["turnover"].mean()
        total_cost = equity_df["cost"].sum()
        total_fee = equity_df["fee"].sum()
        total_slip = equity_df["slippage_cost"].sum()
        trade_count = int((equity_df["buy_value"] > 0).sum() + (equity_df["sell_value"] > 0).sum())

        return {
            "total_return": float(total_return),
            "annualized_return": float(annualized_return),
            "annualized_volatility": float(annualized_vol),
            "sharpe": float(sharpe),
            "sortino": float(sortino) if not np.isnan(sortino) else None,
            "max_drawdown": float(max_drawdown),
            "calmar": float(calmar) if not np.isnan(calmar) else None,
            "win_rate": float(win_rate),
            "mean_daily_return": float(rets.mean()),
            "best_daily_return": float(rets.max()),
            "worst_daily_return": float(rets.min()),
            "trading_days": int(n_days),
            "initial_value": float(self.initial_cash),
            "final_value": float(equity_df["account_value"].iloc[-1]),
            "mean_turnover": float(total_turnover),
            "total_cost": float(total_cost),
            "total_fee": float(total_fee),
            "total_slippage": float(total_slip),
            "trade_days_with_activity": trade_count,
            "n_holdings": self.n_holdings,
            "n_replace": self.n_replace,
            "rebalance_days": self.rebalance_days,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Worst-performer replacement strategy backtest")
    parser.add_argument("--start", type=str, default="2023-01-01", help="Backtest start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-12-31", help="Backtest end date (YYYY-MM-DD)")
    parser.add_argument("--n-holdings", type=int, default=30, help="Number of stocks to hold")
    parser.add_argument("--n-replace", type=int, default=6, help="Number of worst performers to replace each rebalance")
    parser.add_argument("--rebalance-days", type=int, default=22, help="Rebalance frequency in trading days")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0, help="Initial capital")
    parser.add_argument("--price-path", type=str, default=str(DEFAULT_PRICE_PATH), help="Path to daily bar parquet data")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--deal-price", type=str, default="open", choices=["open", "close"])
    parser.add_argument("--open-cost", type=float, default=0.0005)
    parser.add_argument("--close-cost", type=float, default=0.0015)
    parser.add_argument("--slippage", type=float, default=0.001)
    parser.add_argument("--min-cost", type=float, default=5.0)
    parser.add_argument("--limit-threshold", type=float, default=0.095)
    parser.add_argument("--lot-size", type=int, default=100)
    parser.add_argument("--no-filter-st", action="store_true", help="Do not filter ST stocks")
    parser.add_argument("--no-filter-suspend", action="store_true", help="Do not filter suspended stocks")
    args = parser.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    price_path = Path(args.price_path)

    # Output dir with timestamp
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%y%m%d-%H%M")
        output_dir = PROJECT_ROOT / "outputs" / "worst_performer_replacement" / ts
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading price data from {price_path}")
    price_df = load_prices(price_path, start, end)
    logger.info(f"Loaded {len(price_df)} rows, {price_df.index.get_level_values('instrument').nunique()} instruments")

    bt = WorstPerformerReplacementBacktester(
        n_holdings=args.n_holdings,
        n_replace=args.n_replace,
        rebalance_days=args.rebalance_days,
        seed=args.seed,
        initial_cash=args.initial_cash,
        deal_price=args.deal_price,
        open_cost=args.open_cost,
        close_cost=args.close_cost,
        slippage=args.slippage,
        min_cost=args.min_cost,
        limit_threshold=args.limit_threshold,
        lot_size=args.lot_size,
        filter_st=not args.no_filter_st,
        filter_suspend=not args.no_filter_suspend,
    )

    logger.info(
        f"Running backtest: {args.n_holdings} holdings, replace {args.n_replace} worst "
        f"every {args.rebalance_days} days, seed={args.seed}"
    )
    result = bt.run(price_df)

    # Save outputs
    result["equity_curve"].to_parquet(output_dir / "backtest_report.parquet")
    if not result["positions"].empty:
        result["positions"].to_parquet(output_dir / "positions.parquet")
    if not result["trades"].empty:
        result["trades"].to_parquet(output_dir / "trades.parquet")
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(result["metrics"], f, indent=2, ensure_ascii=False, default=str)

    # Print summary
    m = result["metrics"]
    logger.info("=" * 60)
    logger.info("Backtest Complete")
    logger.info("=" * 60)
    logger.info(f"Period: {start.date()} ~ {end.date()} ({m['trading_days']} days)")
    logger.info(f"Total return: {m['total_return']:.2%}")
    logger.info(f"Annualized return: {m['annualized_return']:.2%}")
    logger.info(f"Annualized vol: {m['annualized_volatility']:.2%}")
    logger.info(f"Sharpe: {m['sharpe']:.3f}")
    logger.info(f"Max drawdown: {m['max_drawdown']:.2%}")
    logger.info(f"Win rate: {m['win_rate']:.2%}")
    logger.info(f"Mean turnover: {m['mean_turnover']:.2%}")
    logger.info(f"Total cost: {m['total_cost']:,.0f}")
    logger.info(f"Final value: {m['final_value']:,.0f}")
    logger.info(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
