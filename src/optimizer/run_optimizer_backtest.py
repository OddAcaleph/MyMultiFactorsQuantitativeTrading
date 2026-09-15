"""Portfolio Optimizer backtest — full V1 (Phases 1-5).

Runs mean-variance optimization over walk-forward predictions with real
risk model data, and computes portfolio returns using daily bar data.

Supports:
- Alpha standardization
- Candidate pool with liquidity/ST/suspend/new-stock filters
- Risk model (X, F, D)
- Industry constraints
- Style factor constraints
- Turnover penalty (L1)
- Risk attribution
- Full diagnostics output

Memory-efficient: processes one year at a time.

Usage:
    PYTHONPATH=src python src/optimizer/run_optimizer_backtest.py \\
        --pred-path output/walk_forward_optimized/predictions/all_predictions_2005_2025.parquet \\
        --risk-dir outputs/risk_model/v1 \\
        --bars-dir data/cleaned_data/daily_bars \\
        --output-dir output/optimizer_backtest/v1_full \\
        --pool-size 100 --max-weight 0.03 --risk-aversion 0.1 \\
        --turnover-penalty 0.01 --rebalance-freq 5 \\
        --industry-max-weight 0.20 \\
        --min-avg-amount-20d 50000 \\
        --filter-st
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from optimizer import (
    FactorTiming,
    OptimizationDiagnostics,
    PortfolioOptimizer,
    RiskAttribution,
    RiskInterface,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Portfolio Optimizer Backtest (Full V1)")
    p.add_argument("--pred-path", required=True, help="Predictions parquet path")
    p.add_argument("--risk-dir", required=True, help="Risk model output dir")
    p.add_argument("--bars-dir", required=True, help="Daily bars data dir")
    p.add_argument("--output-dir", required=True, help="Output directory")

    # Candidate pool
    p.add_argument("--pool-size", type=int, default=100)
    p.add_argument("--min-avg-amount-20d", type=float, default=0,
                   help="Min 20-day avg amount (in 千元). 0 = no filter.")
    p.add_argument("--filter-st", action="store_true", help="Filter ST stocks")
    p.add_argument("--filter-suspend", action="store_true", help="Filter suspended stocks")
    p.add_argument("--filter-new-stock-days", type=int, default=0,
                   help="Filter stocks listed fewer than N days")

    # Objective
    p.add_argument("--risk-aversion", type=float, default=0.1)
    p.add_argument("--turnover-penalty", type=float, default=0.01)

    # Transaction costs
    p.add_argument("--buy-cost", type=float, default=0.0005,
                   help="Buy-side commission rate (default 0.0005 = 5bp)")
    p.add_argument("--sell-cost", type=float, default=0.0015,
                   help="Sell-side commission + stamp duty rate (default 0.0015 = 15bp)")
    p.add_argument("--slippage", type=float, default=0.001,
                   help="Slippage rate per trade (default 0.001 = 10bp, applied both sides)")

    # Board lot
    p.add_argument("--board-lot", type=int, default=0,
                   help="Board lot size (round shares to integer multiples). 0 = disabled. A-share = 100.")

    # Constraints
    p.add_argument("--max-weight", type=float, default=0.03)
    p.add_argument("--industry-max-weight", type=float, default=0.0,
                   help="Max industry weight. 0 = no constraint.")
    p.add_argument("--style-bounds", type=str, default="",
                   help="Style factor bounds as JSON: '{\"SIZE\":[-0.5,0.5], ...}'")
    p.add_argument("--max-vol", type=float, default=0.0,
                   help="Max annualized portfolio volatility (hard constraint, fully invested). "
                        "0 = no constraint. e.g. 0.20 = 20% annualized vol cap.")
    p.add_argument("--target-vol", type=float, default=0.0,
                   help="Target annualized volatility for dynamic position sizing. "
                        "0 = no targeting (always fully invested). "
                        "Scales position up/down with cash to hit target vol. "
                        "e.g. 0.20 = target 20% annualized vol.")

    # Alpha processing
    p.add_argument("--alpha-industry-neutral", action="store_true",
                   help="Industry-neutralize alpha scores before optimization.")
    p.add_argument("--alpha-rank", action="store_true",
                   help="Use rank-based alpha standardization instead of z-score.")
    p.add_argument("--alpha-smooth-span", type=int, default=0,
                   help="EMA smoothing span for alpha scores (per-stock, time-series). "
                        "0 = disabled. e.g. 5 = 5-day EMA, reduces turnover.")

    # Rebalance
    p.add_argument("--rebalance-freq", type=int, default=5,
                   help="Rebalance every N trading days")
    p.add_argument("--signal-delay", type=int, default=1,
                   help="Number of days to delay signals (T-day signal -> T+delay trade). "
                        "Default 1 = no look-ahead, realistic execution.")

    # Trend MA dynamic position sizing (MA50 bear market filter)
    p.add_argument("--trend-ma-days", type=int, default=0,
                   help="Number of days for trend MA filter. 0 = disabled. "
                        "e.g. 50 = MA50: if market index < MA50, reduce position to trend-ma-bear-position.")
    p.add_argument("--trend-ma-bear-position", type=float, default=0.3,
                   help="Position ratio when market is below trend MA (bear market). "
                        "Default 0.3 = 30% invested, 70% cash.")
    p.add_argument("--trend-ma-smooth", action="store_true",
                   help="Use smooth transition around MA instead of binary on/off.")
    p.add_argument("--trend-ma-smooth-band", type=float, default=0.05,
                   help="Smooth band around MA (fraction of MA). Default 0.05 = +/-5%%.")
    p.add_argument("--trend-ma2-days", type=int, default=0,
                   help="Number of days for second (long-term) trend MA filter. 0 = disabled. "
                        "e.g. 200 = MA200: if market below MA200, reduce to trend-ma2-bear-position.")
    p.add_argument("--trend-ma2-bear-position", type=float, default=0.15,
                   help="Position ratio when market is below both MA1 and MA2. "
                        "Default 0.15 = 15% invested.")

    # Strategy mode
    p.add_argument("--strategy", type=str, default="mean_variance",
                   choices=["mean_variance", "risk_parity"],
                   help="Portfolio strategy: mean_variance (QP optimizer) or risk_parity.")

    # Risk-parity specific parameters
    p.add_argument("--rp-alpha-tilt", type=float, default=0.3,
                   help="Risk-parity alpha tilt: 0=pure RP, 1=pure alpha-weighted.")
    p.add_argument("--rp-industry-neutral-selection", action="store_true",
                   help="Use industry-stratified stock selection in risk-parity.")
    p.add_argument("--rp-max-weight", type=float, default=0.10,
                   help="Max single-stock weight for risk-parity.")
    p.add_argument("--rp-industry-max-weight", type=float, default=0.20,
                   help="Max industry weight for risk-parity.")

    # Date range
    p.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")

    # Factor timing (regime-based alpha scaling)
    p.add_argument("--factor-timing", action="store_true",
                   help="Enable factor timing (volatility regime + alpha momentum scaling).")
    p.add_argument("--ft-vol-lookback", type=int, default=20,
                   help="Factor timing: lookback days for volatility regime.")
    p.add_argument("--ft-vol-low-threshold", type=float, default=0.012,
                   help="Factor timing: low vol threshold (daily). Below = scale up.")
    p.add_argument("--ft-vol-high-threshold", type=float, default=0.025,
                   help="Factor timing: high vol threshold (daily). Above = scale down.")
    p.add_argument("--ft-vol-low-scale", type=float, default=1.3,
                   help="Factor timing: alpha scale in low-vol regime.")
    p.add_argument("--ft-vol-high-scale", type=float, default=0.5,
                   help="Factor timing: alpha scale in high-vol regime.")
    p.add_argument("--ft-alpha-momentum", action="store_true",
                   help="Factor timing: enable alpha momentum scaling.")
    p.add_argument("--ft-alpha-mom-lookback", type=int, default=20,
                   help="Factor timing: lookback days for alpha momentum.")
    p.add_argument("--ft-alpha-mom-min-scale", type=float, default=0.3,
                   help="Factor timing: min alpha scale for momentum (worst IC).")
    p.add_argument("--ft-alpha-mom-max-scale", type=float, default=1.5,
                   help="Factor timing: max alpha scale for momentum (best IC).")
    p.add_argument("--ft-alpha-mom-ic-threshold", type=float, default=0.0,
                   help="Factor timing: IC threshold for normalization (IC below this = min scale).")
    p.add_argument("--ft-alpha-mom-ic-scale", type=float, default=0.05,
                   help="Factor timing: IC scale range (IC threshold + scale = max scale).")

    return p.parse_args()


def load_year_bars(bars_dir: str, year: int) -> pd.DataFrame:
    """Load all daily bars for a given year.

    Uses wide_table_daily_bars when available (has ST/suspend columns),
    falls back to daily_bars otherwise.
    """
    # Try wide_table first (has ST/suspend columns)
    wide_dir = Path(str(bars_dir).replace("daily_bars", "wide_table_daily_bars"))
    use_wide = wide_dir.exists()

    source_dir = wide_dir if use_wide else Path(bars_dir)
    year_dir = source_dir / f"year={year}"
    if not year_dir.exists():
        return pd.DataFrame()

    all_dfs = []
    base_cols = ["trade_date", "ts_code", "pct_chg", "close", "open", "pre_close", "amount", "vol"]
    extra_cols = ["st_stock", "star_st_stock", "is_suspect"] if use_wide else []

    for month_dir in sorted(year_dir.iterdir()):
        if not month_dir.is_dir():
            continue
        for f in sorted(month_dir.glob("*.parquet")):
            df = pd.read_parquet(f)
            cols = [c for c in base_cols + extra_cols if c in df.columns]
            all_dfs.append(df[cols])

    if not all_dfs:
        return pd.DataFrame()

    full = pd.concat(all_dfs, ignore_index=True)
    full["ret"] = full["pct_chg"] / 100.0
    full["trade_date"] = full["trade_date"].astype(int)
    full["ret"] = full["ret"].clip(lower=-0.35, upper=0.6)

    # Gap return (pre_close -> open) and intraday return (open -> close)
    # For open-price execution model:
    #   - gap_ret = (open - pre_close) / pre_close  (overnight jump)
    #   - intraday_ret = (close - open) / open       (daytime move)
    full["gap_ret"] = np.where(
        (full["pre_close"].notna()) & (full["pre_close"] > 0) & (full["open"].notna()) & (full["open"] > 0),
        (full["open"] - full["pre_close"]) / full["pre_close"],
        0.0,
    )
    full["intraday_ret"] = np.where(
        (full["open"].notna()) & (full["open"] > 0) & (full["close"].notna()) & (full["close"] > 0),
        (full["close"] - full["open"]) / full["open"],
        0.0,
    )
    full["gap_ret"] = full["gap_ret"].clip(lower=-0.35, upper=0.6)
    full["intraday_ret"] = full["intraday_ret"].clip(lower=-0.35, upper=0.6)

    # Open-gap pct (for limit-up/down judgment at open)
    full["open_gap_pct"] = full["gap_ret"] * 100.0

    # Compute 20-day avg amount for liquidity filtering
    full = full.sort_values(["ts_code", "trade_date"])
    full["avg_amount_20d"] = full.groupby("ts_code")["amount"].transform(
        lambda x: x.rolling(20, min_periods=10).mean()
    )

    # Derive is_st flag
    if "st_stock" in full.columns and "star_st_stock" in full.columns:
        full["is_st"] = ((full["st_stock"] == 1) | (full["star_st_stock"] == 1)).astype(int)
    elif "st_stock" in full.columns:
        full["is_st"] = (full["st_stock"] == 1).astype(int)
    else:
        full["is_st"] = 0

    if "is_suspect" not in full.columns:
        full["is_suspect"] = 0

    return full


def load_list_dates(bars_dir: str) -> dict[str, pd.Timestamp]:
    """Load list dates for all stocks from list_dates.json."""
    import json

    candidates = [
        Path(str(bars_dir).replace("daily_bars", "wide_table_daily_bars")).parent / "list_dates.json",
        Path(bars_dir).parent / "list_dates.json",
        Path(bars_dir).parent.parent / "processd_data" / "list_dates.json",
    ]

    for p in candidates:
        if p.exists():
            with open(p) as f:
                raw = json.load(f)
            logger.info("Loaded list_dates from %s (%d stocks)", p, len(raw))
            return {k: pd.Timestamp(str(v)) for k, v in raw.items()}

    logger.warning("list_dates.json not found, new-stock filter disabled (tried: %s)",
                   ", ".join(str(p) for p in candidates))
    return {}


def _limit_threshold_for(instrument: str, is_st: bool = False) -> float:
    """Return price limit threshold (as decimal, e.g. 0.10 for 10%)."""
    ts_code = str(instrument)
    if ts_code.startswith(("8", "92")):
        return 0.29
    if ts_code.startswith("68"):
        return 0.19
    if ts_code.startswith("30"):
        return 0.19
    if is_st:
        return 0.049
    return 0.095


def _is_buy_blocked(pct_chg: float, limit: float, vol: float) -> bool:
    """Check if buying is blocked (limit up or zero volume)."""
    if pd.isna(vol) or vol <= 0:
        return True
    if pd.isna(pct_chg):
        return True
    return pct_chg >= limit


def _is_sell_blocked(pct_chg: float, limit: float, vol: float) -> bool:
    """Check if selling is blocked (limit down or zero volume)."""
    if pd.isna(vol) or vol <= 0:
        return True
    if pd.isna(pct_chg):
        return True
    return pct_chg <= -limit


def _apply_board_lot(
    target_weights: pd.Series,
    current_weights: pd.Series,
    price: pd.Series,
    nav: float,
    board_lot: int,
    reference_nav: float = 10_000_000.0,
) -> pd.Series:
    """Round target weights to integer board-lot shares.

    Uses a reference NAV (default 10M) to compute share counts, then converts
    back to weights. This avoids artifacts from the normalized nav=1.0 scale —
    board-lot rounding error is proportional to lot value / position value,
    which is realistic at scale.

    The residual cash (uninvested due to rounding) is left as cash.
    """
    if board_lot <= 0 or len(target_weights) == 0:
        return target_weights

    all_codes = target_weights.index.union(current_weights.index)
    tgt = target_weights.reindex(all_codes, fill_value=0.0)

    prices = price.reindex(all_codes)
    valid = prices.notna() & (prices > 0) & (tgt > 0)

    # Compute shares at reference NAV scale
    tgt_value_ref = tgt[valid] * reference_nav
    shares_ref = np.floor(tgt_value_ref / prices[valid] / board_lot) * board_lot

    # Convert back to weights at current NAV
    new_value = shares_ref * prices[valid]
    result = new_value / reference_nav

    result = result[result > 1e-8]
    return result


def load_pred_year(pred: pd.DataFrame, year: int) -> pd.Series:
    """Extract predictions for a given year."""
    dates = pred.index.get_level_values(0)
    mask = (dates.year == year)
    return pred.loc[mask, "pred"]


def run_backtest(
    pred_full: pd.DataFrame,
    risk_interface: RiskInterface,
    bars_dir: str,
    config: dict,
    rebalance_freq: int = 5,
    start_date: str | None = None,
    end_date: str | None = None,
    list_dates: dict[str, pd.Timestamp] | None = None,
    enable_limit_constraints: bool = True,
    buy_cost: float = 0.0,
    sell_cost: float = 0.0,
    slippage: float = 0.0,
    board_lot: int = 0,
    target_vol: float = 0.0,
    signal_delay: int = 1,
    trend_ma_days: int = 0,
    trend_ma_bear_position: float = 0.3,
    trend_ma_smooth: bool = False,
    trend_ma_smooth_band: float = 0.05,
    trend_ma2_days: int = 0,
    trend_ma2_bear_position: float = 0.15,
    factor_timing: FactorTiming | None = None,
) -> dict:
    """Run optimizer backtest year by year.

    Parameters
    ----------
    pred_full : pd.DataFrame
        Full predictions with MultiIndex (datetime, instrument).
    risk_interface : RiskInterface
    bars_dir : str
        Path to daily bars parquet directory.
    config : dict
        Optimizer configuration.
    rebalance_freq : int
        Rebalance every N trading days.

    Returns
    -------
    dict with nav, returns, holdings, diagnostics, factor_exposure, risk_attribution
    """
    all_dates = pd.to_datetime(pred_full.index.get_level_values(0).unique()).sort_values()
    if start_date:
        all_dates = all_dates[all_dates >= pd.to_datetime(start_date)]
    if end_date:
        all_dates = all_dates[all_dates <= pd.to_datetime(end_date)]

    # Shift predictions by signal_delay days to avoid look-ahead bias.
    # T-day signal is generated at T close, trades execute at T+signal_delay open.
    # We shift predictions forward so that pred.loc[dt] reflects the signal
    # available for trading on day dt.
    if signal_delay > 0:
        pred_full = pred_full.copy()
        pred_dates = pd.to_datetime(pred_full.index.get_level_values(0))
        unique_dates = pd.Series(pred_dates.unique()).sort_values().reset_index(drop=True)
        # Build a mapping from original date -> shifted date
        date_map = {}
        for i, d in enumerate(unique_dates):
            shifted_idx = i + signal_delay
            if shifted_idx < len(unique_dates):
                date_map[d] = unique_dates[shifted_idx]
        # Only keep predictions that have a valid shifted date
        mask = pred_dates.isin(date_map.keys())
        pred_full = pred_full.loc[mask].copy()
        # Map dates forward
        new_dates = pred_dates[mask].map(date_map)
        # Rebuild MultiIndex
        instruments = pred_full.index.get_level_values(1)
        pred_full.index = pd.MultiIndex.from_arrays(
            [new_dates, instruments],
            names=pred_full.index.names,
        )
        logger.info("Applied signal_delay=%d: predictions shifted forward by %d trading days",
                    signal_delay, signal_delay)

    years = sorted(all_dates.year.unique())
    logger.info("Backtest: %s to %s, %d years, %d trading days",
                all_dates[0].strftime('%Y-%m-%d'),
                all_dates[-1].strftime('%Y-%m-%d'),
                len(years), len(all_dates))

    # Build market index for trend MA filter (equal-weight cross-sectional mean return)
    # We compute the index NAV from daily bar returns across all stocks.
    market_nav: pd.Series | None = None
    market_nav2: pd.Series | None = None
    if trend_ma_days > 0 or trend_ma2_days > 0:
        logger.info("Building market index for trend MA filter")
        market_returns = []
        for year in years:
            bars_df = load_year_bars(bars_dir, year)
            if bars_df.empty:
                continue
            ret_pivot = bars_df.pivot(index="trade_date", columns="ts_code", values="ret")
            mkt_ret = ret_pivot.mean(axis=1)
            market_returns.append(mkt_ret)
            del bars_df, ret_pivot
            gc.collect()
        if market_returns:
            mkt_ret_series = pd.concat(market_returns).sort_index()
            mkt_ret_series.index = pd.to_datetime(mkt_ret_series.index.astype(str))
            market_nav = (1.0 + mkt_ret_series).cumprod()
            market_nav2 = market_nav  # same base for MA2
            logger.info("Market index built: %d trading days", len(market_nav))
            del mkt_ret_series
            gc.collect()

    strategy_mode = config.get("strategy", "mean_variance")
    if strategy_mode == "risk_parity":
        from optimizer.risk_parity_strategy import RiskParityStrategy
        opt = RiskParityStrategy(config, risk_interface)
    else:
        opt = PortfolioOptimizer(config, risk_interface)
    attribution = RiskAttribution()
    diagnostics = OptimizationDiagnostics()

    # State variables
    current_weights = pd.Series(dtype=float)  # actual executed weights
    target_weights = pd.Series(dtype=float)   # optimizer target weights
    prev_weights = pd.Series(dtype=float)     # weights before today's rebalance (for cost calc)
    nav = 1.0
    day_count = 0
    total_fee = 0.0
    total_slippage = 0.0

    # Output collectors
    nav_list = []
    holdings_list = []
    factor_exp_list = []
    risk_attr_list = []

    n_optimized = 0
    n_skipped = 0

    # Factor timing state
    _prev_alpha_for_ft: pd.Series | None = None

    for year in years:
        logger.info("Processing year %d", year)

        bars_df = load_year_bars(bars_dir, year)
        if bars_df.empty:
            logger.warning("No bar data for year %d, skipping", year)
            continue

        # Pivot returns (close-to-close)
        ret_pivot = bars_df.pivot(index="trade_date", columns="ts_code", values="ret")

        # Pivot gap return (pre_close -> open) and intraday return (open -> close)
        gap_ret_pivot = bars_df.pivot(index="trade_date", columns="ts_code", values="gap_ret")
        intraday_ret_pivot = bars_df.pivot(index="trade_date", columns="ts_code", values="intraday_ret")

        # Pivot avg_amount_20d for liquidity filtering
        amt_pivot = bars_df.pivot(index="trade_date", columns="ts_code", values="avg_amount_20d")

        # Pivot ST/suspend flags
        st_pivot = bars_df.pivot(index="trade_date", columns="ts_code", values="is_st") if "is_st" in bars_df.columns else None
        suspend_pivot = bars_df.pivot(index="trade_date", columns="ts_code", values="is_suspect") if "is_suspect" in bars_df.columns else None

        # Pivot open-gap pct and vol for limit-up/down constraints at open
        open_gap_pivot = bars_df.pivot(index="trade_date", columns="ts_code", values="open_gap_pct")
        vol_pivot = bars_df.pivot(index="trade_date", columns="ts_code", values="vol")

        # Pivot open price for board-lot rounding (execution at open)
        open_pivot = bars_df.pivot(index="trade_date", columns="ts_code", values="open") if board_lot > 0 else None

        year_dates = all_dates[all_dates.year == year]

        for dt in year_dates:
            date_int = int(dt.strftime("%Y%m%d"))

            if date_int not in ret_pivot.index:
                nav_list.append((dt, nav))
                day_count += 1
                continue

            today_ret = ret_pivot.loc[date_int]
            today_gap_ret = gap_ret_pivot.loc[date_int] if date_int in gap_ret_pivot.index else pd.Series(dtype=float)
            today_intraday_ret = intraday_ret_pivot.loc[date_int] if date_int in intraday_ret_pivot.index else pd.Series(dtype=float)

            # Update factor timing with today's market return
            if factor_timing is not None:
                mkt_ret_today = today_ret.mean()
                if not np.isnan(mkt_ret_today):
                    factor_timing.update_market(mkt_ret_today)

            # Apply execution constraints (limit-up/down at open) before today's returns.
            # Rebalance decisions made on day t-1 close execute on day t open.
            # If a stock opens limit-up on day t, you can't buy it; if opens limit-down,
            # you can't sell it. We adjust target_weights -> actual_weights here.
            rebalanced_today = False
            if enable_limit_constraints and len(target_weights) > 0:
                # Check if target differs from current (i.e. there's a rebalance to execute)
                all_codes_check = target_weights.index.union(current_weights.index)
                if len(all_codes_check) > 0:
                    tgt = target_weights.reindex(all_codes_check, fill_value=0.0)
                    cur = current_weights.reindex(all_codes_check, fill_value=0.0)
                    if not (abs(tgt - cur) < 1e-8).all():
                        rebalanced_today = True

                if rebalanced_today:
                    prev_w = current_weights.copy()
                    day_open_gap = open_gap_pivot.loc[date_int] if date_int in open_gap_pivot.index else pd.Series(dtype=float)
                    day_vol = vol_pivot.loc[date_int] if date_int in vol_pivot.index else pd.Series(dtype=float)
                    day_st = st_pivot.loc[date_int] if (st_pivot is not None and date_int in st_pivot.index) else pd.Series(dtype=float)

                    actual = current_weights.copy()

                    # Process each stock in target_weights
                    all_codes = target_weights.index.union(current_weights.index)
                    for code in all_codes:
                        tgt_w = target_weights.get(code, 0.0)
                        curr_w = current_weights.get(code, 0.0)

                        gap_pct = day_open_gap.get(code, np.nan)
                        vol_val = day_vol.get(code, np.nan)
                        is_st = bool(day_st.get(code, 0)) if code in day_st.index else False
                        limit = _limit_threshold_for(code, is_st)
                        gap_dec = gap_pct / 100.0 if pd.notna(gap_pct) else 0.0

                        if tgt_w > curr_w:
                            # Want to buy more — check if open is limit-up
                            if _is_buy_blocked(gap_dec, limit, vol_val):
                                # Can't buy, keep current weight
                                actual[code] = curr_w
                            else:
                                actual[code] = tgt_w
                        elif tgt_w < curr_w:
                            # Want to sell — check if open is limit-down
                            if _is_sell_blocked(gap_dec, limit, vol_val):
                                # Can't sell, keep current weight
                                actual[code] = curr_w
                            else:
                                actual[code] = tgt_w
                        else:
                            actual[code] = tgt_w

                    # Filter out tiny weights
                    actual = actual[actual > 1e-8]

                    # Fallback allocation: if some buys are blocked by limit-up at open,
                    # redistribute the unspent budget to other buyable stocks
                    # proportional to their target weights.
                    blocked_buy_budget = 0.0
                    for code in all_codes:
                        tgt_w = target_weights.get(code, 0.0)
                        curr_w = current_weights.get(code, 0.0)
                        if tgt_w > curr_w:
                            act_w = actual.get(code, 0.0)
                            if act_w < tgt_w - 1e-8:
                                blocked_buy_budget += (tgt_w - act_w)

                    if blocked_buy_budget > 1e-6:
                        # Stocks that can still receive more (not at target, not blocked)
                        eligible = []
                        for code in actual.index:
                            tgt_w = target_weights.get(code, 0.0)
                            act_w = actual[code]
                            if act_w < tgt_w - 1e-8:
                                # Already at target (not blocked) — can't add more
                                continue
                            # Check if this stock is buy-blocked at open
                            gap_pct = day_open_gap.get(code, np.nan)
                            vol_val = day_vol.get(code, np.nan)
                            is_st_code = bool(day_st.get(code, 0)) if code in day_st.index else False
                            limit_code = _limit_threshold_for(code, is_st_code)
                            gap_dec_code = gap_pct / 100.0 if pd.notna(gap_pct) else 0.0
                            if _is_buy_blocked(gap_dec_code, limit_code, vol_val):
                                continue  # Still blocked
                            # Can add more — use remaining headroom to target weight
                            headroom = tgt_w - act_w
                            if headroom > 1e-8:
                                eligible.append((code, headroom, tgt_w))

                        # Distribute blocked budget proportionally to target weight
                        if eligible:
                            total_tgt = sum(t for _, _, t in eligible)
                            if total_tgt > 0:
                                remaining = blocked_buy_budget
                                for code, headroom, tgt_w in sorted(eligible, key=lambda x: -x[2]):
                                    share = (tgt_w / total_tgt) * blocked_buy_budget
                                    add = min(share, headroom, remaining)
                                    if add > 1e-8:
                                        actual[code] = actual.get(code, 0.0) + add
                                        remaining -= add
                                    if remaining < 1e-8:
                                        break

                    # Board-lot rounding: convert weights -> shares -> weights
                    # Uses open price since execution happens at open
                    if board_lot > 0 and open_pivot is not None and date_int in open_pivot.index:
                        actual = _apply_board_lot(
                            actual, prev_w,
                            open_pivot.loc[date_int],
                            nav, board_lot,
                        )

                    w_sum = actual.sum()
                    if w_sum > 0 and abs(w_sum - 1.0) > 0.001:
                        pass  # keep as-is, non-invested portion earns 0

                    current_weights = actual
                    # Update target to match actual for next day (constraint persists)
                    target_weights = actual.copy()

                    # Deduct transaction costs (based on executed value at open price)
                    if buy_cost > 0 or sell_cost > 0 or slippage > 0:
                        all_cost_codes = current_weights.index.union(prev_w.index)
                        curr_cost = current_weights.reindex(all_cost_codes, fill_value=0.0)
                        prev_cost = prev_w.reindex(all_cost_codes, fill_value=0.0)
                        diff = curr_cost - prev_cost

                        buy_amount = float(diff[diff > 0].sum()) * nav
                        sell_amount = float((-diff[diff < 0]).sum()) * nav

                        fee_cost = buy_amount * buy_cost + sell_amount * sell_cost
                        slip_cost = (buy_amount + sell_amount) * slippage
                        total_cost = fee_cost + slip_cost

                        if total_cost > 0:
                            nav -= total_cost
                            total_fee += fee_cost
                            total_slippage += slip_cost
            elif not enable_limit_constraints and len(target_weights) > 0:
                # No limit constraints but target changed — still execute rebalance
                all_codes_check = target_weights.index.union(current_weights.index)
                if len(all_codes_check) > 0:
                    tgt = target_weights.reindex(all_codes_check, fill_value=0.0)
                    cur = current_weights.reindex(all_codes_check, fill_value=0.0)
                    if not (abs(tgt - cur) < 1e-8).all():
                        rebalanced_today = True
                        prev_w = current_weights.copy()
                        current_weights = target_weights.copy()

                        # Board-lot rounding (at open price)
                        if board_lot > 0 and open_pivot is not None and date_int in open_pivot.index:
                            current_weights = _apply_board_lot(
                                current_weights, prev_w,
                                open_pivot.loc[date_int],
                                nav, board_lot,
                            )

                        # Deduct transaction costs
                        if buy_cost > 0 or sell_cost > 0 or slippage > 0:
                            all_cost_codes = current_weights.index.union(prev_w.index)
                            curr_cost = current_weights.reindex(all_cost_codes, fill_value=0.0)
                            prev_cost = prev_w.reindex(all_cost_codes, fill_value=0.0)
                            diff = curr_cost - prev_cost

                            buy_amount = float(diff[diff > 0].sum()) * nav
                            sell_amount = float((-diff[diff < 0]).sum()) * nav

                            fee_cost = buy_amount * buy_cost + sell_amount * sell_cost
                            slip_cost = (buy_amount + sell_amount) * slippage
                            total_cost = fee_cost + slip_cost

                            if total_cost > 0:
                                nav -= total_cost
                                total_fee += fee_cost
                                total_slippage += slip_cost

            # Apply today's returns to current holdings.
            # Open-price execution model:
            #   - New buys: bought at open, earn intraday return (open -> close)
            #   - Continued holds: held all day, earn full-day return (pre_close -> close)
            #   - Sells: sold at open, earn only gap return (pre_close -> open)
            # We track prev_weights before rebalance to distinguish buys from holds.
            if len(current_weights) > 0:
                if rebalanced_today and len(prev_w) > 0:
                    # Need to categorize each position: buy, hold, or sell
                    all_ret_codes = current_weights.index.union(prev_w.index)
                    curr_w_ret = current_weights.reindex(all_ret_codes, fill_value=0.0)
                    prev_w_ret = prev_w.reindex(all_ret_codes, fill_value=0.0)

                    # Intersect with available return data
                    valid_codes = curr_w_ret.index.intersection(today_gap_ret.index).intersection(today_intraday_ret.index)
                    curr_w_ret = curr_w_ret.loc[valid_codes]
                    prev_w_ret = prev_w_ret.loc[valid_codes]

                    if len(valid_codes) > 0:
                        gap = today_gap_ret.loc[valid_codes]
                        intra = today_intraday_ret.loc[valid_codes]

                        # For each stock, decompose weight into:
                        #   hold_weight = min(prev, curr) -> earns full day return (gap + intra, compounded)
                        #   buy_weight = max(0, curr - prev) -> earns intraday return only
                        #   sell_weight = max(0, prev - curr) -> earns gap return only
                        hold_w = np.minimum(prev_w_ret, curr_w_ret)
                        buy_w = (curr_w_ret - prev_w_ret).clip(lower=0)
                        sell_w = (prev_w_ret - curr_w_ret).clip(lower=0)

                        # Full-day return for hold portion (compounded: (1+gap)*(1+intra) - 1)
                        # But gap and intra are already computed such that:
                        #   close = pre_close * (1+gap) * (1+intra)
                        #   full_ret = (1+gap)*(1+intra) - 1
                        full_day_ret = (1.0 + gap) * (1.0 + intra) - 1.0

                        day_ret = float(
                            (hold_w * full_day_ret).sum()
                            + (buy_w * intra).sum()
                            + (sell_w * gap).sum()
                        )
                        nav *= (1.0 + day_ret)
                    else:
                        day_ret = 0.0
                else:
                    # No rebalance today — all positions are holds, earn full-day return
                    common = current_weights.index.intersection(today_ret.index)
                    if len(common) > 0:
                        w = current_weights.loc[common]
                        day_ret = float((w * today_ret.loc[common]).sum())
                        nav *= (1.0 + day_ret)
                    else:
                        day_ret = 0.0
            else:
                day_ret = 0.0

            nav_list.append((dt, nav))

            is_rebalance = (day_count % rebalance_freq == 0)
            day_count += 1

            if not is_rebalance:
                continue

            # Get alpha for today
            try:
                day_pred = pred_full.xs(dt, level=0)["pred"]
            except KeyError:
                n_skipped += 1
                continue

            if len(day_pred) < 30:
                n_skipped += 1
                continue

            # Note: factor timing is applied as a position sizing overlay after optimization
            # (not alpha scaling, since uniform alpha scaling doesn't change the optimal
            # portfolio under fully-invested + long-only constraints).

            # Update factor timing alpha momentum with realized returns from previous period.
            # We use today's return as the realized forward return for the previous alpha.
            if factor_timing is not None and factor_timing.alpha_momentum_enabled:
                if _prev_alpha_for_ft is not None and len(today_ret) > 0:
                    factor_timing.update_alpha_performance(_prev_alpha_for_ft, today_ret)
                _prev_alpha_for_ft = day_pred.copy()

            # Build market_data for candidate pool filtering
            market_data = None
            if date_int in amt_pivot.index:
                md_dict = {}
                day_amt = amt_pivot.loc[date_int]
                md_dict["avg_amount_20d"] = day_amt.reindex(day_pred.index).fillna(0)

                if st_pivot is not None and date_int in st_pivot.index:
                    md_dict["is_st"] = st_pivot.loc[date_int].reindex(day_pred.index).fillna(0).astype(bool)

                if suspend_pivot is not None and date_int in suspend_pivot.index:
                    md_dict["is_suspended"] = suspend_pivot.loc[date_int].reindex(day_pred.index).fillna(0).astype(bool)

                if list_dates and config.get("candidate_pool", {}).get("filter_new_stock_days", 0) > 0:
                    new_stock_days = config["candidate_pool"]["filter_new_stock_days"]
                    list_days = pd.Series(
                        [
                            (dt - list_dates.get(code, pd.Timestamp("1990-01-01"))).days
                            for code in day_pred.index
                        ],
                        index=day_pred.index,
                    )
                    md_dict["list_days"] = list_days

                market_data = pd.DataFrame(md_dict, index=day_pred.index)

            # Run optimization
            try:
                # For volatility targeting, pass normalized (full-investment) weights
                # to the optimizer so turnover penalty is computed correctly.
                opt_prev_weights = current_weights
                if target_vol > 0 and len(current_weights) > 0:
                    w_sum = current_weights.sum()
                    if w_sum > 0 and abs(w_sum - 1.0) > 1e-6:
                        opt_prev_weights = current_weights / w_sum

                result = opt.optimize(
                    date_int, day_pred,
                    current_weights=opt_prev_weights,
                    market_data=market_data,
                )
            except Exception as e:
                logger.debug("Optimization failed for %d: %s", date_int, e)
                n_skipped += 1
                continue

            if strategy_mode == "mean_variance":
                if "optimal" not in result.solver_status and "optimal_inaccurate" not in result.solver_status and "user_limit" not in result.solver_status:
                    n_skipped += 1
                    continue
            else:
                if result.solver_status == "no_candidates":
                    n_skipped += 1
                    continue

            n_optimized += 1
            target_weights = result.weights

            # Volatility targeting: scale position up/down to hit target vol.
            # If predicted vol > target, hold cash; if < target, stay fully invested.
            if target_vol > 0 and result.portfolio_volatility > 0:
                # result.portfolio_volatility is daily; convert target to daily
                target_vol_daily = target_vol / np.sqrt(252.0)
                scale = min(1.0, target_vol_daily / result.portfolio_volatility)
                if scale < 1.0:
                    target_weights = target_weights * scale

            # Trend MA dynamic position sizing (bear market filter)
            # If market index is below its MA, reduce position to bear_position.
            if trend_ma_days > 0 and market_nav is not None and dt in market_nav.index:
                idx = market_nav.index.get_loc(dt)
                if idx >= trend_ma_days:
                    ma_val = market_nav.iloc[idx - trend_ma_days + 1:idx + 1].mean()
                    current_val = market_nav.iloc[idx]
                    if trend_ma_smooth:
                        band = ma_val * trend_ma_smooth_band
                        if band > 0:
                            normalized = (current_val - ma_val) / band
                            t = max(0.0, min(1.0, (normalized + 1.0) / 2.0))
                            trend_position = trend_ma_bear_position + (1.0 - trend_ma_bear_position) * t
                        else:
                            trend_position = 1.0 if current_val >= ma_val else trend_ma_bear_position
                    else:
                        trend_position = 1.0 if current_val >= ma_val else trend_ma_bear_position

                    if trend_position < 1.0:
                        target_weights = target_weights * trend_position

            # Second (long-term) MA filter
            if trend_ma2_days > 0 and market_nav2 is not None and dt in market_nav2.index:
                idx = market_nav2.index.get_loc(dt)
                if idx >= trend_ma2_days:
                    ma2_val = market_nav2.iloc[idx - trend_ma2_days + 1:idx + 1].mean()
                    current_val = market_nav2.iloc[idx]
                    if current_val < ma2_val:
                        # Additional reduction: scale to ma2_bear_position from current
                        current_sum = target_weights.sum()
                        if current_sum > trend_ma2_bear_position:
                            scale = trend_ma2_bear_position / current_sum if current_sum > 0 else 0
                            target_weights = target_weights * scale

            # Factor timing: position sizing overlay
            # Scale portfolio exposure up/down based on vol regime / alpha momentum.
            # scale < 1 means hold cash; scale > 1 is capped at 1.0 (no leverage).
            if factor_timing is not None:
                ft_scale = factor_timing.get_scale(dt)
                if ft_scale < 1.0:
                    target_weights = target_weights * ft_scale

            # Risk attribution
            risk_data = risk_interface.get_day_risk_data(
                date_int, result.weights.index.tolist(),
            )
            attr_result = attribution.compute(
                result.weights.values,
                risk_data.exposures,
                risk_data.factor_cov,
                risk_data.specific_variance,
                risk_data.factor_names,
            )

            # Record diagnostics
            diagnostics.record(date_int, result, attr_result)

            # Record holdings (all positive weights)
            for code, weight in result.weights.items():
                if weight > 1e-6:
                    holdings_list.append({
                        "trade_date": date_int,
                        "ts_code": code,
                        "weight": float(weight),
                    })

            # Record factor exposure
            if len(result.factor_exposure) > 0:
                exp_row = {"trade_date": date_int}
                for name, val in result.factor_exposure.items():
                    exp_row[name] = float(val)
                factor_exp_list.append(exp_row)

            # Record risk attribution
            attr_row = {
                "trade_date": date_int,
                "total_variance": attr_result.total_variance,
                "factor_variance": attr_result.factor_variance,
                "specific_variance": attr_result.specific_variance,
                "specific_contribution_pct": attr_result.specific_contribution_pct,
            }
            for name, val in attr_result.factor_contribution_pct.items():
                attr_row[f"rc_{name}"] = float(val)
            risk_attr_list.append(attr_row)

        del bars_df, ret_pivot, gap_ret_pivot, intraday_ret_pivot, amt_pivot
        gc.collect()

        logger.info("Year %d done: nav=%.4f, optimized=%d, skipped=%d",
                    year, nav, n_optimized, n_skipped)

    # Build output DataFrames
    nav_df = pd.DataFrame(nav_list, columns=["date", "nav"]).set_index("date")
    nav_df["daily_return"] = nav_df["nav"].pct_change().fillna(0.0)

    diag_df = diagnostics.to_dataframe()
    holdings_df = pd.DataFrame(holdings_list)
    factor_exp_df = pd.DataFrame(factor_exp_list)
    risk_attr_df = pd.DataFrame(risk_attr_list)

    return {
        "nav": nav_df,
        "diagnostics": diag_df,
        "holdings": holdings_df,
        "factor_exposure": factor_exp_df,
        "risk_attribution": risk_attr_df,
        "n_optimized": n_optimized,
        "n_skipped": n_skipped,
        "total_fee": total_fee,
        "total_slippage": total_slippage,
    }


def compute_metrics(nav_df: pd.DataFrame, diag_df: pd.DataFrame) -> dict:
    """Compute backtest performance metrics."""
    if nav_df.empty:
        return {}

    nav = nav_df["nav"]
    returns = nav_df["daily_return"]

    total_return = nav.iloc[-1] / nav.iloc[0] - 1.0
    n_years = (nav.index[-1] - nav.index[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / n_years) - 1.0 if n_years > 0 else 0.0

    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

    # Max drawdown
    rolling_max = nav.cummax()
    drawdown = nav / rolling_max - 1.0
    max_dd = drawdown.min()

    # Sortino
    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 1 else annual_vol
    sortino = annual_return / downside_vol if downside_vol > 0 else 0.0

    # Calmar
    calmar = annual_return / abs(max_dd) if max_dd < 0 else 0.0

    win_rate = (returns > 0).mean()

    metrics = {
        "total_return": round(float(total_return), 4),
        "annual_return": round(float(annual_return), 4),
        "annual_volatility": round(float(annual_vol), 4),
        "sharpe_ratio": round(float(sharpe), 3),
        "sortino_ratio": round(float(sortino), 3),
        "calmar_ratio": round(float(calmar), 3),
        "max_drawdown": round(float(max_dd), 4),
        "win_rate": round(float(win_rate), 4),
        "n_trading_days": len(nav),
    }

    if not diag_df.empty:
        metrics.update({
            "avg_holdings": round(float(diag_df["n_holdings"].mean()), 1),
            "avg_turnover": round(float(diag_df["turnover"].mean()), 4),
            "avg_portfolio_vol_daily": round(float(diag_df["portfolio_volatility"].mean()), 6),
            "avg_alpha_retention": round(float(diag_df["alpha_retention"].mean()), 4),
            "avg_solve_time_ms": round(float(diag_df["solve_time_ms"].mean()), 2),
            "factor_risk_pct": round(float(diag_df.get("factor_risk_pct", pd.Series([0])).mean()), 4),
            "specific_risk_pct": round(float(diag_df.get("specific_risk_pct", pd.Series([0])).mean()), 4),
        })

    return metrics


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions
    logger.info("Loading predictions...")
    pred_full = pd.read_parquet(args.pred_path)
    pred_full.index = pred_full.index.set_names(["datetime", "instrument"])
    logger.info("Predictions loaded: %d rows", len(pred_full))

    # Risk interface
    risk_if = RiskInterface({"output_dir": args.risk_dir})

    # Parse style bounds
    style_bounds = {}
    if args.style_bounds:
        import json as _json
        style_bounds = _json.loads(args.style_bounds)

    # Build optimizer config
    constraints = {
        "long_only": True,
        "fully_invested": True,
        "max_weight": args.max_weight,
    }
    if args.industry_max_weight > 0:
        constraints["industry"] = {"max_weight": args.industry_max_weight}
    if style_bounds:
        constraints["style"] = style_bounds
    if args.max_vol > 0:
        constraints["max_vol"] = args.max_vol

    config = {
        "strategy": args.strategy,
        "alpha": {"method": "zscore", "winsorize": True, "winsorize_quantile": 0.01,
                  "industry_neutral": args.alpha_industry_neutral,
                  "rank": args.alpha_rank,
                  "smooth_span": args.alpha_smooth_span},
        "candidate_pool": {
            "pool_size": args.pool_size,
            "min_avg_amount_20d": args.min_avg_amount_20d,
            "filter_st": args.filter_st,
            "filter_suspend": args.filter_suspend,
            "filter_new_stock_days": args.filter_new_stock_days,
        },
        "objective": {
            "risk_aversion": args.risk_aversion,
            "turnover_penalty": args.turnover_penalty,
        },
        "constraints": constraints,
        "risk_parity": {
            "alpha_tilt": args.rp_alpha_tilt,
            "industry_neutral_selection": args.rp_industry_neutral_selection,
            "max_weight": args.rp_max_weight,
            "industry_max_weight": args.rp_industry_max_weight,
        },
        "target_vol": args.target_vol,
        "solver": {"solver": "OSQP", "verbose": False, "max_iter": 4000},
    }

    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    # Initialize factor timing
    factor_timing = None
    if args.factor_timing:
        ft_config = {
            "enabled": True,
            "vol_lookback": args.ft_vol_lookback,
            "vol_low_threshold": args.ft_vol_low_threshold,
            "vol_high_threshold": args.ft_vol_high_threshold,
            "vol_low_scale": args.ft_vol_low_scale,
            "vol_high_scale": args.ft_vol_high_scale,
            "alpha_momentum_enabled": args.ft_alpha_momentum,
            "alpha_mom_lookback": args.ft_alpha_mom_lookback,
            "alpha_mom_min_scale": args.ft_alpha_mom_min_scale,
            "alpha_mom_max_scale": args.ft_alpha_mom_max_scale,
            "alpha_mom_ic_threshold": args.ft_alpha_mom_ic_threshold,
            "alpha_mom_ic_scale": args.ft_alpha_mom_ic_scale,
        }
        factor_timing = FactorTiming(ft_config)
        logger.info("Factor timing enabled: vol_lookback=%d, low_thresh=%.4f, high_thresh=%.4f",
                    args.ft_vol_lookback, args.ft_vol_low_threshold, args.ft_vol_high_threshold)

    # Load list dates for new-stock filter
    list_dates = {}
    if args.filter_new_stock_days > 0:
        list_dates = load_list_dates(args.bars_dir)
        logger.info("Loaded list dates for %d stocks", len(list_dates))

    # Run backtest
    t0 = time.time()
    results = run_backtest(
        pred_full, risk_if, args.bars_dir, config,
        rebalance_freq=args.rebalance_freq,
        start_date=args.start_date,
        end_date=args.end_date,
        list_dates=list_dates,
        enable_limit_constraints=True,
        buy_cost=args.buy_cost,
        sell_cost=args.sell_cost,
        slippage=args.slippage,
        board_lot=args.board_lot,
        target_vol=args.target_vol,
        signal_delay=args.signal_delay,
        trend_ma_days=args.trend_ma_days,
        trend_ma_bear_position=args.trend_ma_bear_position,
        trend_ma_smooth=args.trend_ma_smooth,
        trend_ma_smooth_band=args.trend_ma_smooth_band,
        trend_ma2_days=args.trend_ma2_days,
        trend_ma2_bear_position=args.trend_ma2_bear_position,
        factor_timing=factor_timing if args.factor_timing else None,
    )
    elapsed = time.time() - t0

    # Compute metrics
    metrics = compute_metrics(results["nav"], results["diagnostics"])
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["n_optimized"] = results["n_optimized"]
    metrics["n_skipped"] = results["n_skipped"]
    metrics["total_fee"] = round(results.get("total_fee", 0.0), 6)
    metrics["total_slippage"] = round(results.get("total_slippage", 0.0), 6)
    metrics["total_transaction_cost"] = round(
        results.get("total_fee", 0.0) + results.get("total_slippage", 0.0), 6
    )

    # Save outputs
    results["nav"].to_parquet(output_dir / "nav.parquet")
    if not results["diagnostics"].empty:
        results["diagnostics"].to_parquet(output_dir / "diagnostics.parquet")
    if not results["holdings"].empty:
        results["holdings"].to_parquet(output_dir / "holdings.parquet")
    if not results["factor_exposure"].empty:
        results["factor_exposure"].to_parquet(output_dir / "factor_exposure.parquet")
    if not results["risk_attribution"].empty:
        results["risk_attribution"].to_parquet(output_dir / "risk_attribution.parquet")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Print summary
    print("\n" + "=" * 60)
    print("  Optimizer Backtest Results (Full V1)")
    print("=" * 60)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:>12.4f}")
        else:
            print(f"  {k:30s}: {v}")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
