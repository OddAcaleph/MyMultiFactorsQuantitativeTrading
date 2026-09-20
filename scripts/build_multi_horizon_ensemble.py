"""Build multi-horizon ensemble - ultra memory efficient, process one year at a time."""
import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path('/mnt/bn/mixinfer-server-cn/qyd/MyMultiFactorsQuantitativeTrading')
GRID_DIR = BASE / 'output/wf_grid_search/full/predictions'
OUT_DIR = BASE / 'output/walk_forward_multi_horizon/predictions'
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = 'est200_d8_lr003'
horizons = ['5d', '10d', '20d']
years = list(range(2005, 2026))

# Step 1: Build ensemble by processing one year at a time
print('Step 1: Building ensemble predictions (year-by-year)...')

# Load all predictions first (they're small ~127MB each)
preds = {}
for h in horizons:
    path = GRID_DIR / f'wf_rolling_tw5y_step1y_label_rank_{h}_{CONFIG}/walk_forward_full_pred.parquet'
    df = pd.read_parquet(path)
    df.columns = [f'pred_{h}']
    dates = pd.to_datetime(df.index.get_level_values(0))
    mask = (dates >= '2005-01-01') & (dates <= '2025-12-31')
    df = df[mask]
    preds[h] = df
    print(f'  Loaded pred_{h}: {df.shape}')

# Merge
combined = preds['5d'].join(preds['10d'], how='inner').join(preds['20d'], how='inner')
del preds
print(f'Combined: {combined.shape}')

# Compute rank average ensemble
print('Computing rank-average ensemble...')
r5 = combined.groupby(level=0)['pred_5d'].rank(pct=True)
r10 = combined.groupby(level=0)['pred_10d'].rank(pct=True)
r20 = combined.groupby(level=0)['pred_20d'].rank(pct=True)
ensemble = ((r5 + r10 + r20) / 3).astype(np.float32).to_frame('pred')
del r5, r10, r20

ensemble.to_parquet(OUT_DIR / 'walk_forward_full_pred.parquet')
print(f'Saved ensemble: {ensemble.shape}')
del ensemble

# Step 2: IC analysis - load labels year by year
print()
print('Step 2: IC analysis (year-by-year)...')

ic_all = {h: [] for h in horizons + ['ens5', 'ens10']}

for year in years:
    # Load labels for this year
    labels = pd.read_parquet(
        BASE / f'data/generated_label/daily_labels/year={year}/',
        columns=['trade_date', 'ts_code', 'label_5d', 'label_10d', 'label_20d']
    )
    labels['datetime'] = pd.to_datetime(labels['trade_date'], format='%Y%m%d')
    labels = labels.set_index(['datetime', 'ts_code'])
    labels = labels[['label_5d', 'label_10d', 'label_20d']].astype(np.float32)
    
    # Get predictions for this year
    dates = pd.to_datetime(combined.index.get_level_values(0))
    mask = dates.year == year
    pred_year = combined[mask]
    
    # Merge
    df_year = pred_year.join(labels, how='inner')
    del labels
    
    # Compute IC for each horizon
    for h in horizons:
        ic = df_year.groupby(level=0).apply(
            lambda x: x[f'pred_{h}'].corr(x[f'label_{h}']), include_groups=False)
        ic_all[h].append(ic)
    
    # Ensemble IC (recompute for this year)
    r5 = df_year.groupby(level=0)['pred_5d'].rank(pct=True)
    r10 = df_year.groupby(level=0)['pred_10d'].rank(pct=True)
    r20 = df_year.groupby(level=0)['pred_20d'].rank(pct=True)
    ens = (r5 + r10 + r20) / 3
    
    ic_ens5 = df_year.assign(ens=ens).groupby(level=0).apply(
        lambda x: x['ens'].corr(x['label_5d']), include_groups=False)
    ic_ens10 = df_year.assign(ens=ens).groupby(level=0).apply(
        lambda x: x['ens'].corr(x['label_10d']), include_groups=False)
    ic_all['ens5'].append(ic_ens5)
    ic_all['ens10'].append(ic_ens10)
    
    print(f'  {year}: 5d IC={ic_all["5d"][-1].mean():.4f} | 10d={ic_all["10d"][-1].mean():.4f} | 20d={ic_all["20d"][-1].mean():.4f} | ens5={ic_ens5.mean():.4f}')

print()
print('=== Full Period IC Summary ===')
for key in ['5d', '10d', '20d', 'ens5', 'ens10']:
    ic = pd.concat(ic_all[key])
    print(f'{key:6s}: mean IC={ic.mean():.4f}, ICIR={ic.mean()/ic.std():.2f}, win={(ic>0).mean():.2%}')

# Correlation between predictions (sample from one year)
print()
print('=== Prediction Correlations (2020 sample) ===')
dates = pd.to_datetime(combined.index.get_level_values(0))
sample = combined[dates.year == 2020]
print(sample[['pred_5d', 'pred_10d', 'pred_20d']].corr())

print()
print('Done!')
