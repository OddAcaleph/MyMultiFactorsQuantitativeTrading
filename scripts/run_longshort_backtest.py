"""Long-short dollar-neutral backtest (50% long / 50% short) - careful version.

Uses industry-neutral 27-model ensemble alpha.
Long top N / short bottom N, equal weight within each leg.
Signal delay = 1 day (no look-ahead).
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json

BASE = Path('/mnt/bn/mixinfer-server-cn/qyd/MyMultiFactorsQuantitativeTrading')

# Load predictions
print('Loading predictions...')
pred = pd.read_parquet(
    BASE / 'output/walk_forward_ensembles/predictions/industry_v2_icweighted_27model/walk_forward_full_pred.parquet'
)
pred = pred.rename_axis(index={'instrument': 'ts_code'})
print(f'  shape: {pred.shape}, date range: {pred.index.get_level_values(0).min()} to {pred.index.get_level_values(0).max()}')

# Load daily returns (label_1d = forward 1-day return from t to t+1)
print('Loading daily returns...')
all_years = sorted(pred.index.get_level_values(0).year.unique())
returns_list = []
for year in all_years:
    df = pd.read_parquet(
        BASE / f'data/generated_label/daily_labels/year={year}/',
        columns=['trade_date', 'ts_code', 'label_1d']
    )
    df['datetime'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
    df = df.set_index(['datetime', 'ts_code'])
    returns_list.append(df['label_1d'].astype(np.float32))

returns = pd.concat(returns_list)
del returns_list
print(f'  shape: {len(returns)}')

# Shift predictions by 1 day (signal delay)
print('Applying signal delay (1 day)...')
pred = pred.rename(columns={'pred': 'alpha'})
# pred on day t is used to trade on day t+1, earning return from t+1 to t+2
# label_1d on day t is the return from t to t+1
# So we need: pred on day t-1 pairs with label_1d on day t

# Align: for each date, the alpha used is from the previous trading day
# We'll do this by shifting alpha forward by 1 day per stock
pred_shifted = pred.copy()
pred_shifted['alpha'] = pred_shifted.groupby(level='ts_code')['alpha'].shift(1)
pred_shifted = pred_shifted.dropna()
print(f'  After shift: {len(pred_shifted)} rows')

# Merge
print('Merging...')
merged = pred_shifted.join(returns.rename('ret'), how='inner')
del pred, pred_shifted, returns
print(f'  Merged: {merged.shape}')

# Parameters
N_LONG = 50
N_SHORT = 50
LONG_W = 0.5 / N_LONG
SHORT_W = 0.5 / N_SHORT
REBAL_FREQ = 2

# Run backtest
print(f'\nRunning long-short backtest (long top {N_LONG} / short bottom {N_SHORT}, rebal={REBAL_FREQ}d)...')

dates = sorted(merged.index.get_level_values(0).unique())
print(f'  Trading days: {len(dates)}')

nav = 1.0
nav_list = []
day_count = 0
current_long = set()
current_short = set()
long_w_map = {}  # ts_code -> weight
short_w_map = {}

for i, dt in enumerate(dates):
    day_data = merged.xs(dt, level=0)
    
    # Compute day return from current positions
    day_ret = 0.0
    n_long_valid = 0
    n_short_valid = 0
    if current_long or current_short:
        long_stocks = [s for s in current_long if s in day_data.index]
        short_stocks = [s for s in current_short if s in day_data.index]

        if long_stocks:
            long_rets = day_data.loc[long_stocks, 'ret'].values
            long_rets = np.clip(np.nan_to_num(long_rets, nan=0.0, posinf=0.0, neginf=0.0), -0.5, 0.5)
            long_contrib = float(np.nanmean(long_rets)) if np.sum(~np.isnan(long_rets)) > 0 else 0.0
            n_long_valid = int(np.sum(np.isfinite(day_data.loc[long_stocks, 'ret'].values)))
        else:
            long_contrib = 0.0

        if short_stocks:
            short_rets = day_data.loc[short_stocks, 'ret'].values
            short_rets = np.clip(np.nan_to_num(short_rets, nan=0.0, posinf=0.0, neginf=0.0), -0.5, 0.5)
            short_contrib = float(np.nanmean(short_rets)) if np.sum(~np.isnan(short_rets)) > 0 else 0.0
            n_short_valid = int(np.sum(np.isfinite(day_data.loc[short_stocks, 'ret'].values)))
        else:
            short_contrib = 0.0

        day_ret = (long_contrib - short_contrib) * 0.5  # 50% long, 50% short
        nav *= (1.0 + day_ret)
    
    nav_list.append((dt, nav, day_ret))

    # Debug first 10 days
    if i < 10:
        print(f'  Day {i} {dt.strftime("%Y-%m-%d")}: ret={day_ret:.4f}, nav={nav:.6f}, '
              f'long={len(current_long)}({n_long_valid}), short={len(current_short)}({n_short_valid})')
    
    # Rebalance
    day_count += 1
    if day_count % REBAL_FREQ != 0:
        continue
    
    if len(day_data) < N_LONG + N_SHORT + 10:
        continue
    
    # Rank by alpha
    ranked = day_data['alpha'].sort_values(ascending=False)
    current_long = set(ranked.head(N_LONG).index)
    current_short = set(ranked.tail(N_SHORT).index)

print(f'\nFinal nav: {nav:.6f}')

# Compute metrics
nav_df = pd.DataFrame(nav_list, columns=['date', 'nav', 'daily_return']).set_index('date')

total_ret = nav_df['nav'].iloc[-1] / nav_df['nav'].iloc[0] - 1.0
n_years = (nav_df.index[-1] - nav_df.index[0]).days / 365.25
annual_ret = (1 + total_ret) ** (1 / n_years) - 1
annual_vol = nav_df['daily_return'].std() * np.sqrt(252)
sharpe = annual_ret / annual_vol if annual_vol > 0 else 0

neg_rets = nav_df['daily_return'][nav_df['daily_return'] < 0]
downside_vol = neg_rets.std() * np.sqrt(252) if len(neg_rets) > 0 else 0
sortino = annual_ret / downside_vol if downside_vol > 0 else 0

rolling_max = nav_df['nav'].cummax()
drawdown = nav_df['nav'] / rolling_max - 1.0
max_dd = drawdown.min()
calmar = annual_ret / abs(max_dd) if max_dd != 0 else 0

win_rate = (nav_df['daily_return'] > 0).mean()

print(f'\n=== Long-Short Dollar-Neutral Backtest (top50/bottom50, rebal2d) ===')
print(f'  Total return:     {total_ret:.2%}')
print(f'  Annual return:    {annual_ret:.2%}')
print(f'  Annual vol:       {annual_vol:.2%}')
print(f'  Sharpe ratio:     {sharpe:.3f}')
print(f'  Sortino ratio:    {sortino:.3f}')
print(f'  Calmar ratio:     {calmar:.3f}')
print(f'  Max drawdown:     {max_dd:.2%}')
print(f'  Win rate:         {win_rate:.2%}')
print(f'  Trading days:     {len(nav_df)}')

# Save
out_dir = BASE / 'output/optimizer_backtest/industry_v2_27model_indneutral_longshort'
out_dir.mkdir(parents=True, exist_ok=True)
nav_df[['nav', 'daily_return']].to_parquet(out_dir / 'nav.parquet')

metrics = {
    'total_return': float(total_ret),
    'annual_return': float(annual_ret),
    'annual_volatility': float(annual_vol),
    'sharpe_ratio': float(sharpe),
    'sortino_ratio': float(sortino),
    'calmar_ratio': float(calmar),
    'max_drawdown': float(max_dd),
    'win_rate': float(win_rate),
    'n_trading_days': len(nav_df),
    'n_long': N_LONG,
    'n_short': N_SHORT,
    'rebalance_freq': REBAL_FREQ,
    'signal_delay': 1,
}
with open(out_dir / 'metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print(f'\nResults saved to {out_dir}')
