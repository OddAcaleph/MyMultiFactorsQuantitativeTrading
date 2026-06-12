from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from trainer import XGBoostInferencer, XGBoostTrainer


DATA_ROOT = PROJECT_ROOT / "data"
CROSS_SECTIONAL_DATA_DIR = DATA_ROOT / "cross_sectional_processd_data"
DAILY_BARS_DIR = CROSS_SECTIONAL_DATA_DIR / "wide_table_daily_bars"
PRICE_VOLUME_FACTORS_DIR = CROSS_SECTIONAL_DATA_DIR / "price_volume_factors"
MONEYFLOW_FACTORS_DIR = CROSS_SECTIONAL_DATA_DIR / "moneyflow_factors"
FUNDAMENTAL_FACTORS_DIR = CROSS_SECTIONAL_DATA_DIR / "fundamental_factors"
INDUSTRY_FACTORS_DIR = CROSS_SECTIONAL_DATA_DIR / "industry_factors"
LABELS_DIR = DATA_ROOT / "generated_label" / "daily_labels"
TEST_INSTRUMENTS = ["000003.SZ", "000005.SZ"]
TEST_SEGMENTS = {
    "train": ("2000-01-04", "2000-01-10"),
    "valid": ("2000-01-11", "2000-01-14"),
    "test": ("2000-01-17", "2000-01-21"),
}
TEST_MODEL_PARAMS = {
    "n_estimators": 5,
    "max_depth": 2,
    "learning_rate": 0.1,
    "n_jobs": 1,
    "tree_method": "hist",
    "objective": "reg:squarederror",
    "random_state": 42,
}


def build_small_trainer(**kwargs) -> XGBoostTrainer:
    params = {
        "daily_bars_dir": DAILY_BARS_DIR,
        "price_volume_factors_dir": PRICE_VOLUME_FACTORS_DIR,
        "moneyflow_factors_dir": MONEYFLOW_FACTORS_DIR,
        "fundamental_factors_dir": FUNDAMENTAL_FACTORS_DIR,
        "industry_factors_dir": INDUSTRY_FACTORS_DIR,
        "labels_dir": LABELS_DIR,
        "output_dir": None,
        "instruments": TEST_INSTRUMENTS,
        "feature_cols": ["open", "high", "low", "close", "ret_1", "range", "ret_5_cc_processed"],
        "label_name": "label_5d",
        "segments": TEST_SEGMENTS,
        "start_time": "2000-01-04",
        "end_time": "2000-01-21",
        "model_params": TEST_MODEL_PARAMS,
        "prefer_gpu": False,
    }
    params.update(kwargs)
    return XGBoostTrainer(
        **params,
    )


def test_trainer_can_prepare_xy_from_dataset():
    trainer = build_small_trainer()

    x_train, y_train = trainer.prepare_xy("train")

    assert not x_train.empty
    assert not y_train.empty
    assert list(x_train.columns) == ["open", "high", "low", "close", "ret_1", "range", "ret_5_cc_processed"]
    assert y_train.name == "label_5d"
    assert x_train.index.equals(y_train.index)
    assert set(x_train.index.get_level_values("instrument")) <= set(TEST_INSTRUMENTS)


def test_trainer_fit_predict_and_evaluate_smoke():
    trainer = build_small_trainer()

    model = trainer.fit(verbose=False)
    pred_df = trainer.predict("test")
    metrics = trainer.evaluate(pred_df)

    assert model is trainer.model
    assert not pred_df.empty
    assert list(pred_df.columns) == ["pred", "label_5d"]
    assert pred_df.index.names == ["datetime", "instrument"]
    assert np.isfinite(pred_df["pred"]).all()

    assert set(metrics) == {"rmse", "ic", "rank_ic"}
    assert np.isfinite(metrics["rmse"])


def test_split_feature_label_supports_flat_dataframe():
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2020-01-01"), "000001.SZ"), (pd.Timestamp("2020-01-02"), "000001.SZ")],
        names=["datetime", "instrument"],
    )
    df = pd.DataFrame(
        {
            "open": [1.0, np.nan],
            "close": [1.1, 1.2],
            "label_5d": [0.01, 0.02],
        },
        index=index,
    )

    x, y = XGBoostTrainer.split_feature_label(df)

    assert list(x.columns) == ["open", "close"]
    assert y.name == "label_5d"
    assert x.index.equals(y.index)
    assert x.isna().sum().sum() == 0


def test_trainer_saves_model_and_inferencer_loads_it(tmp_path):
    model_path = tmp_path / "xgboost_test_model.json"
    output_dir = tmp_path / "infer_outputs"
    trainer = build_small_trainer(model_path=model_path)

    trainer.fit(verbose=False)
    saved_path = trainer.save_model()

    assert saved_path == model_path
    assert model_path.exists()

    inferencer = XGBoostInferencer(
        model_path=model_path,
        output_dir=output_dir,
        instruments=TEST_INSTRUMENTS,
        feature_cols=["open", "high", "low", "close", "ret_1", "range", "ret_5_cc_processed"],
        label_name="label_5d",
        segment="test",
        start_time="2000-01-17",
        end_time="2000-01-21",
        prefer_gpu=False,
    )
    pred_df = inferencer.predict(save=True)

    assert not pred_df.empty
    assert list(pred_df.columns) == ["pred", "label_5d"]
    assert (output_dir / "pred_test.parquet").exists()
