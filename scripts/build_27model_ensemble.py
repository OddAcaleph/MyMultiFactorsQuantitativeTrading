"""Build 27-model IC-IR weighted ensemble - ultra memory efficient, year-by-year."""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys

BASE = Path('/mnt/bn/mixinfer-server-cn/qyd/MyMultiFactorsQuantitativeTrading')
ENH_DIR = BASE / 'output/wf_grid_search/industry_v2_ensemble/predictions'
TOP16_DIR = BASE / 'output/wf_grid_search/industry_v2_27model/predictions'
BASE_FS_DIR = BASE / 'output/wf_grid_search/industry_v2_27model/predictions'
OUT_DIR = BASE / 'output/walk_forward_ensembles/predictions/industry_v2_icweighted_27model'
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ['5d', '10d', '20d']
PARAMS = {
    'fast': 'est100_d6_lr005',
    'bal': 'est200_d8_lr003',
    'deep': 'est300_d10_lr001',
}
FEATURE_SETS = {
    'enh': (ENH_DIR, ''),
    'top16': (TOP16_DIR, '_top16'),
    'base': (BASE_FS_DIR, '_base'),
}

model_names = []
model_paths = []
model_labs = []
for fs, (fdir, fs_suffix) in FEATURE_SETS.items():
    for lab in LABELS:
        for pname, psuffix in PARAMS.items():
            name = f'{fs}_{lab}_{pname}'
            path = fdir / f'wf_rolling_tw5y_step1y_label_rank_{lab}_{psuffix}{fs_suffix}/walk_forward_full_pred.parquet'
            if not path.exists():
                print(f'WARNING: {name} not found at {path}')
                continue
            model_names.append(name)
            model_paths.append(path)
            model_labs.append(lab)

print(f'{len(model_names)} models found', flush=True)

# Step 1: Compute IC-IR weights year-by-year
print('\nStep 1: Computing IC-IR for each model...', flush=True)
years = list(range(2005, 2026))
ic_by_model = {name: [] for name in model_names}

for year in years:
    # Load all labels for this year
    labels = pd.read_parquet(
        BASE / f'data/generated_label/daily_labels/year={year}/',
        columns=['trade_date', 'ts_code', 'label_rank_5d', 'label_rank_10d', 'label_rank_20d']
    )
    labels['datetime'] = pd.to_datetime(labels['trade_date'], format='%Y%m%d')
    labels = labels.set_index(['datetime', 'ts_code'])
    labels = labels[['label_rank_5d', 'label_rank_10d', 'label_rank_20d']].astype(np.float32)
    
    for name, path, lab in zip(model_names, model_paths, model_labs):
        label_col = f'label_rank_{lab}'
        
        # Load prediction, filter to this year
        pred = pd.read_parquet(path)
        pred = pred.rename_axis(index={'instrument': 'ts_code'})
        dates = pd.to_datetime(pred.index.get_level_values(0))
        mask = (dates.year == year)
        pred_year = pred[mask]
        del pred
        
        if len(pred_year) == 0:
            continue
        
        # Merge with labels
        merged = pred_year.join(labels[[label_col]], how='inner')
        del pred_year
        
        if len(merged) == 0:
            continue
        
        # Compute daily IC using numpy (fast)
        daily_ics = []
        for dt, group in merged.groupby(level=0):
            if len(group) < 10:
                continue
            ic = np.corrcoef(group['pred'].values, group[label_col].values)[0, 1]
            daily_ics.append(ic)
        
        if daily_ics:
            ic_by_model[name].append(pd.Series(daily_ics))
        del merged, daily_ics
    
    del labels
    print(f'  {year}: done', flush=True)

# Compute summary stats
ic_summary = {}
weights = {}
for name in model_names:
    if ic_by_model[name]:
        ic_all = pd.concat(ic_by_model[name])
        mean_ic = float(ic_all.mean())
        std_ic = float(ic_all.std())
        ir = mean_ic / std_ic if std_ic > 0 else 0
        win_rate = float((ic_all > 0).mean())
        ic_summary[name] = {'mean': mean_ic, 'std': std_ic, 'ir': ir, 'win_rate': win_rate}
        weights[name] = max(ir, 0.01)
        print(f'  {name:20s}: IC={mean_ic:.4f}, IR={ir:.2f}, win={win_rate:.2%}', flush=True)
    else:
        ic_summary[name] = {'mean': 0, 'std': 0, 'ir': 0, 'win_rate': 0}
        weights[name] = 0.01

total_w = sum(weights.values())
weights_norm = {k: v / total_w for k, v in weights.items()}

print('\n=== Weights (IC-IR weighted) ===', flush=True)
for name, w in sorted(weights_norm.items(), key=lambda x: -x[1]):
    print(f'  {name:20s}: {w:.4f}', flush=True)

with open(OUT_DIR / 'weights.json', 'w') as f:
    json.dump(weights_norm, f, indent=2)
with open(OUT_DIR / 'ic_summary.json', 'w') as f:
    json.dump(ic_summary, f, indent=2)

# Step 2: Build ensemble year-by-year
print('\nStep 2: Building ensemble predictions (year-by-year)...', flush=True)
ensemble_parts = []

for year in years:
    # Load all predictions for this year
    preds_year = {}
    common_idx = None
    for name, path in zip(model_names, model_paths):
        pred = pd.read_parquet(path)
        pred = pred.rename_axis(index={'instrument': 'ts_code'})
        dates = pd.to_datetime(pred.index.get_level_values(0))
        mask = dates.year == year
        pred_year = pred[mask]
        del pred
        
        if len(pred_year) == 0:
            continue
        
        preds_year[name] = pred_year
        if common_idx is None:
            common_idx = pred_year.index
        else:
            common_idx = common_idx.intersection(pred_year.index)
    
    if common_idx is None or len(common_idx) == 0:
        print(f'  {year}: no data', flush=True)
        continue
    
    # Compute ranks and weighted ensemble
    ens = np.zeros(len(common_idx), dtype=np.float32)
    for name in preds_year:
        vals = preds_year[name].loc[common_idx, 'pred'].values
        # Rank per date
        dates_arr = common_idx.get_level_values(0)
        unique_dates = pd.unique(dates_arr)
        ranks = np.zeros(len(vals), dtype=np.float32)
        for dt in unique_dates:
            dt_mask = dates_arr == dt
            r = pd.Series(vals[dt_mask]).rank(pct=True).values
            ranks[dt_mask] = r
        ens += ranks * weights_norm[name]
        del ranks
    
    ens_df = pd.DataFrame({'pred': ens}, index=common_idx)
    ensemble_parts.append(ens_df)
    del preds_year, ens, ens_df
    print(f'  {year}: {len(common_idx)} rows', flush=True)

print('\nConcatenating all years...', flush=True)
ensemble = pd.concat(ensemble_parts)
print(f'Final ensemble: {ensemble.shape}', flush=True)

ensemble.to_parquet(OUT_DIR / 'walk_forward_full_pred.parquet')
print(f'Saved to {OUT_DIR / "walk_forward_full_pred.parquet"}', flush=True)

# Step 3: Ensemble IC
print('\nStep 3: Ensemble IC...', flush=True)
for lab in LABELS:
    label_col = f'label_rank_{lab}'
    ics = []
    for year in years:
        dates = pd.to_datetime(ensemble.index.get_level_values(0))
        mask = dates.year == year
        if not mask.any():
            continue
        
        ens_year = ensemble[mask]
        
        labels = pd.read_parquet(
            BASE / f'data/generated_label/daily_labels/year={year}/',
            columns=['trade_date', 'ts_code', label_col]
        )
        labels['datetime'] = pd.to_datetime(labels['trade_date'], format='%Y%m%d')
        labels = labels.set_index(['datetime', 'ts_code'])[label_col].astype(np.float32)
        
        merged = ens_year.join(labels, how='inner')
        if len(merged) == 0:
            continue
        
        daily_ics = []
        for dt, group in merged.groupby(level=0):
            if len(group) < 10:
                continue
            ic = np.corrcoef(group['pred'].values, group[label_col].values)[0, 1]
            daily_ics.append(ic)
        
        if daily_ics:
            ics.append(pd.Series(daily_ics))
        del merged
    
    if ics:
        ic_all = pd.concat(ics)
        print(f'  vs {lab}: IC={ic_all.mean():.4f}, IR={ic_all.mean()/ic_all.std():.2f}, win={(ic_all>0).mean():.2%}', flush=True)

print('\nDone!', flush=True)
