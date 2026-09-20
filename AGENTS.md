# AGENTS.md — MyMultiFactorsQuantitativeTrading

## 1. Project Overview

**MyMultiFactorsQuantitativeTrading** is an end-to-end A-share multi-factor quantitative trading research platform. It implements a full pipeline from data fetching, cleaning, feature engineering, XGBoost model training, to realistic backtesting with A-share trading rules (T+1, board lots, price limits, etc.).

**Version**: v0.1.0
**Core algorithm**: XGBoost gradient boosted trees on top of Qlib's data infrastructure
**Market coverage**: A-shares (Tushare data source)
**Key new features in v0.1.0**: Walk-forward training, enhanced alpha factors (20+), full A-share backtest constraints (industry, liquidity, ST/suspend, new stock filter), volatility-weighted position sizing, dynamic position management

### Architecture

The project follows a **linear pipeline** architecture. Each stage reads from a data directory and writes to the next, never mutating raw inputs.

```
data_fetch → data_clean → data_process → features_generate → cross_sectional_process → label_generate → model_train → backtest/ic_validate
```

Key modules under `src/`:

| Module | Path | Responsibility |
|--------|------|----------------|
| Trainer | `src/trainer/` | XGBoost training, inference, grid search, walk-forward training |
| Backtester | `src/backtester/` | Preparatory (fast) and simple (realistic) backtests with A-share constraints |
| Dataset Fetcher | `src/utils/dataset_fetcher/` | Tushare API data download |
| Dataset Cleaner | `src/utils/dataset_cleaner/` | Deduplication, OHLC validation |
| Dataset Processor | `src/utils/dataset_processor/` | Wide-table joins (ST, suspend, adj, fundamentals, industry) |
| Features Generator | `src/utils/features_generator/` | 66+ factors across price/volume, fundamental, moneyflow, industry, enhanced alpha |
| Cross-sectional Processor | `src/utils/cross_sectional_processor/` | Winsorization + Z-score per trading day |
| Label Generator | `src/utils/label_generator/` | Forward return and rank labels (1d–20d horizons) |
| Dataset Generator | `src/utils/dataset_generator/` | ParquetLoader implementing Qlib DataLoader interface |
| IC Validator | `src/utils/ic_validator/` | IC analysis, group returns, long-short backtest, turnover |
| Config | `src/utils/config.py` | JSON config loading with `${PROJECT_ROOT}` expansion and deep merge |

Data flows through directories under `data/`: `raw_data` → `cleaned_data` → `processd_data` → `features_data` → `cross_sectional_processd_data` → `generated_label`.

### Entry Points

Each module has a `run_*.py` CLI entry point:
- `src/trainer/run_xgboost_training.py`
- `src/trainer/run_xgboost_inference.py`
- `src/trainer/run_xgboost_train_backtest_grid_search.py`
- `src/trainer/run_walk_forward_training.py` — walk-forward rolling training
- `src/trainer/run_walk_forward_backtest.py` — walk-forward backtest (stitched predictions)
- `src/backtester/run_simple_backtest.py`
- `src/backtester/run_preparatory_backtest.py`
- `src/backtester/run_simple_backtest_grid_search.py`
- `src/utils/ic_validator/run_ic_validation.py` (run as `python -m utils.ic_validator.run_ic_validation`)

Shell pipeline scripts are in `scripts/data_pipeline/`.

Reusable utility scripts are in `scripts/`. One-off experiment scripts are in `scripts/experiments/` and are not committed.
Experiment config variants are in `conf/experiments/`.

---

## 2. Build & Commands

### Environment Setup

```bash
pip install -r requirements.txt
export PYTHONPATH="/path/to/project/src:$PYTHONPATH"
export TUSHARE_TOKEN="your_token"   # or echo to ~/.tushare_token
```

Key dependencies: `xgboost==3.2.0`, `pyqlib==0.9.7`, `pandas`, `numpy`, `pyarrow`, `tushare`, `matplotlib`.

### Data Pipeline

```bash
# Full daily pipeline
bash scripts/data_pipeline/daily_data_pipeline.sh [start_date] [end_date]

# Individual stages
bash scripts/data_pipeline/data_fetch_pipeline.sh YYYYMMDD YYYYMMDD
bash scripts/data_pipeline/data_process_pipeline.sh YYYYMMDD YYYYMMDD
bash scripts/data_pipeline/factor_calculation_pipeline.sh YYYYMMDD YYYYMMDD
bash scripts/data_pipeline/label_calculation_pipeline.sh YYYYMMDD YYYYMMDD
```

Environment variables control pipeline stages: `RUN_FETCH`, `RUN_PROCESS`, `INFERENCE_CPU`, `MASTER_CONTINUE_ON_ERROR`.

### Training & Inference

```bash
# Full training (uses conf/xgboost_trainer_config.json)
python src/trainer/run_xgboost_training.py [--cpu] [--model-path PATH]

# Inference
python src/trainer/run_xgboost_inference.py [--start-time ... --end-time ...] [--no-label]
```

Default train/valid/test split: 2000–2020 / 2021–2022 / 2023–2025.

### Walk-Forward Training

```bash
# Walk-forward training + backtest (uses conf/walk_forward_config.json)
python scripts/run_walk_forward.py --config conf/walk_forward_config.json

# Walk-forward grid search (parallel with scheduler + workers)
python scripts/start_wf_grid_search.py --config conf/walk_forward_config.json --n-workers 4
```

Walk-forward supports rolling and expanding windows. Each window trains independently and test-set predictions are stitched together for a single continuous backtest.

### Enhanced Alpha Factors

```bash
# Generate enhanced alpha factors (momentum, volume-price, moneyflow, fundamental momentum)
bash scripts/data_pipeline/enhanced_alpha_factors_pipeline.sh [start_date] [end_date]
```

Adds ~20 offensive alpha factors on top of the base 61 factors: MACD, RSI, 52-week high distance, OBV, revenue acceleration, gross margin change, etc.

### Backtesting

```bash
python src/backtester/run_simple_backtest.py \
  --config conf/simple_backtester_config.json \
  --prediction-path outputs/.../pred_test.parquet \
  --topk 50 --n-drop 5 --plot
```

**Backtest constraint features** (all configurable via `strategy` in config):
- **Stock pool filters**: ST filter (`filter_st`), suspend filter (`filter_suspend`), new stock filter (`filter_new_stock_days`), liquidity filter (`min_avg_amount_20d`)
- **Industry constraints**: max industry weight (`max_industry_weight`), max industry count (`max_industry_count`), industry-stratified sampling
- **Position sizing**: equal weight (default), volatility-weighted (`vol_weight_enabled`), single-stock weight cap (`max_single_weight`)
- **Dynamic position**: target-vol, score-threshold, and dispersion-based methods that can reduce exposure to cash
- **Board-specific price limits**: different limit-up/down thresholds for main board (10%), ChiNext/STAR (20%), BSE (30%)
- **Sell criteria**: lowest predicted score (default) or worst holding return (`drop_criteria`)

### Grid Search

```bash
python src/trainer/run_xgboost_train_backtest_grid_search.py \
  --config conf/xgboost_train_backtest_grid_search_phase2.json
```

Supports random search over hyperparameters + backtest parameters, with resume capability.

### IC Validation

```bash
python -m utils.ic_validator.run_ic_validation \
  --pred-path outputs/.../pred_test.parquet \
  --save-dir outputs/ic_validation
```

### Tests

```bash
PYTHONPATH="src" pytest test/ -v
```

Three test files: `test_xgboost_trainer.py`, `test_preparatory_backtester.py`, `test_parquet_loader.py`.

---

## 3. Code Style

- **Language**: Python 3.7+ with `from __future__ import annotations`
- **Docstrings**: Google-style docstrings on classes and public functions
- **Imports**: Standard library first, then third-party, then local; relative imports within packages
- **Path handling**: Use `pathlib.Path` throughout. Resolve paths via `src/utils/config.py:resolve_path()` which supports `${PROJECT_ROOT}` placeholder
- **Config pattern**: JSON config files in `conf/`. Load via `load_*_config()` helpers in `src/utils/config.py`. CLI overrides are merged with `deep_merge()`
- **Class naming**: `PascalCase` for classes (e.g., `XGBoostTrainer`, `SimpleBacktester`, `ParquetLoader`)
- **File naming**: `snake_case.py` for modules, `run_*.py` for CLI entry points
- **Data immutability**: Raw data is never modified in place. Each processing stage writes to a new directory under `data/`
- **Type hints**: Use `typing` annotations on public APIs (`str | Path`, `Mapping[str, Any]`, `Sequence[str]`)
- **Qlib integration**: The custom `ParquetLoader` implements Qlib's `DataLoader` interface so it plugs directly into `DataHandlerLP` and `DatasetH`

---

## 4. Testing

- **Framework**: pytest
- **Test directory**: `test/` at project root
- **Conventions**:
  - Tests set `PYTHONPATH` to `src/` and import from package roots (`from trainer import XGBoostTrainer`)
  - Smoke tests use small instrument lists (e.g., `["000003.SZ", "000005.SZ"]`) and short date ranges to run quickly
  - Test fixtures define minimal model params (small `n_estimators`, shallow `max_depth`)
  - Tests read from local `data/` directories; no network calls in tests
- **Running tests**:
  ```bash
  PYTHONPATH="src" pytest test/ -v
  PYTHONPATH="src" pytest test/test_xgboost_trainer.py -v
  ```

---

## 5. Security

- **Tushare token**: Never commit tokens. Use `TUSHARE_TOKEN` env var or `~/.tushare_token`. The `.gitignore` excludes `.env`, `secrets.json`, and `config.ini`
- **Data safety**: "Raw data read-only" principle — all processing stages write to separate output directories and never mutate source data
- **Look-ahead bias prevention**:
  - Labels use the latest full-history adj factor for unified forward-return calculation
  - Backtest enforces `signal_delay=1` (t-day predictions trade on t+1)
  - Point-in-time fundamental data uses only disclosed reports as of each date
  - Feature and label computations are in separate modules to prevent leakage
- **Gitignore**: `data/`, `outputs/`, `output/`, `models/` are all gitignored to avoid committing large data files and model artifacts

---

## 6. Configuration

All configuration is JSON-based in `conf/`. Key files in `conf/` root:

| Config file | Purpose |
|-------------|---------|
| `parquet_loader_config.json` | Data source paths, 66+ feature columns, label selection |
| `xgboost_trainer_config.json` | Train/valid/test segments, model params, output paths, GPU preference |
| `xgboost_trainer_enhanced_alpha_config.json` | Training config with enhanced alpha factors |
| `xgboost_inferencer_config.json` | Inference model path, time range, output path |
| `simple_backtester_config.json` | Backtest params: topk, n_drop, costs, slippage, limit rules, lot size |
| `preparatory_backtester_config.json` | Fast preparatory backtest config |
| `simple_backtest_grid_search_config.json` | Grid search over backtest parameters |
| `xgboost_train_backtest_grid_search_config.json` | Combined train + backtest grid search |
| `xgboost_train_backtest_grid_search_phase2.json` | Phase 2 random search (200 trials) config |
| `xgboost_train_backtest_grid_search_phase3.json` | Phase 3 random search config |
| `walk_forward_config.json` | Walk-forward training + backtest (rolling windows) |

Experiment-specific config variants live in `conf/experiments/` (walk-forward variants, full-constraint backtests, top16 factor configs, etc.).

### Config System Details

- **`${PROJECT_ROOT}` placeholder**: Automatically expanded to the project root directory by `load_config()` in `src/utils/config.py`
- **`${timestamp}` placeholder**: Expanded at runtime in backtest output directories (format: `YYMMDD-HHMM`)
- **Deep merge**: `deep_merge(base, override)` recursively merges config dicts without mutation
- **Config loading pattern**: Each module has a `load_<module>_config()` helper that defaults to its JSON file in `conf/`
- **CLI overrides**: `run_*.py` scripts accept `--config` plus individual parameter flags that override loaded config values

### Switching Training Targets

To change the prediction label (e.g., from `label_5d` to `label_rank_5d`), update three files:
1. `parquet_loader_config.json` → `label_name`
2. `xgboost_trainer_config.json` → `model_filename` and `label_name`
3. `xgboost_inferencer_config.json` → `model_path` and label column

### Adding New Features

1. Add computation in the appropriate `src/utils/features_generator/*_feature_generator.py`
2. Add cross-sectional processing in `src/utils/cross_sectional_processor/*_cross_sectional_processor.py`
3. Add the column name to `feature_cols` in `parquet_loader_config.json`
