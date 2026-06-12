from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils import ParquetLoader


DATA_ROOT = PROJECT_ROOT / "data"
CROSS_SECTIONAL_DATA_DIR = DATA_ROOT / "cross_sectional_processd_data"
DAILY_BARS_DIR = CROSS_SECTIONAL_DATA_DIR / "wide_table_daily_bars"
PRICE_VOLUME_FACTORS_DIR = CROSS_SECTIONAL_DATA_DIR / "price_volume_factors"
MONEYFLOW_FACTORS_DIR = CROSS_SECTIONAL_DATA_DIR / "moneyflow_factors"
FUNDAMENTAL_FACTORS_DIR = CROSS_SECTIONAL_DATA_DIR / "fundamental_factors"
INDUSTRY_FACTORS_DIR = CROSS_SECTIONAL_DATA_DIR / "industry_factors"
LABELS_DIR = DATA_ROOT / "generated_label" / "daily_labels"


def test_loader_returns_qlib_compatible_dataframe():
    loader = ParquetLoader(
        daily_bars_dir=DAILY_BARS_DIR,
        price_volume_factors_dir=PRICE_VOLUME_FACTORS_DIR,
        moneyflow_factors_dir=MONEYFLOW_FACTORS_DIR,
        fundamental_factors_dir=FUNDAMENTAL_FACTORS_DIR,
        industry_factors_dir=INDUSTRY_FACTORS_DIR,
        labels_dir=LABELS_DIR,
        feature_cols=[
            "open",
            "high",
            "low",
            "close",
            "ret_1",
            "range",
            "vol",
            "log_volume",
            "ret_5_cc_processed",
            "main_net_inflow_cc_processed",
            "roe_roa_gap_cc_processed",
            "industry_ret_1_cc_processed",
        ],
        label_name="label_5d",
        dropna_label=False,
    )

    df = loader.load(
        instruments=["000003.SZ", "000005.SZ"],
        start_time="2020-01-04",
        end_time="2020-01-10",
    )

    import pdb; pdb.set_trace()

    assert not df.empty
    assert isinstance(df.index, pd.MultiIndex)
    assert df.index.names == ["datetime", "instrument"]

    assert ("feature", "open") in df.columns
    assert ("feature", "ret_1") in df.columns
    assert ("feature", "range") in df.columns
    assert ("feature", "log_volume") in df.columns
    assert ("feature", "ret_5_cc_processed") in df.columns
    assert ("feature", "main_net_inflow_cc_processed") in df.columns
    assert ("feature", "roe_roa_gap_cc_processed") in df.columns
    assert ("feature", "industry_ret_1_cc_processed") in df.columns
    assert ("label", "label_5d") in df.columns

    assert set(df.index.get_level_values("instrument")) <= {"000003.SZ", "000005.SZ"}


def test_loader_supports_dataset_like_output_shape():
    loader = ParquetLoader(
        daily_bars_dir=DAILY_BARS_DIR,
        labels_dir=LABELS_DIR,
        feature_cols=["open", "high", "low", "close"],
        label_name="label_5d",
        dropna_label=False,
    )

    df = loader.load(
        instruments=["000003.SZ"],
        start_time="2000-01-04",
        end_time="2000-01-10",
    )

    feature_df = df["feature"]
    label_s = df["label"]["label_5d"]

    assert list(feature_df.columns) == ["open", "high", "low", "close"]
    assert label_s.name == "label_5d"
    assert feature_df.shape[0] == label_s.shape[0]
    assert feature_df.index.equals(label_s.index)


def test_loader_supports_multiple_configurable_labels():
    loader = ParquetLoader(
        daily_bars_dir=DAILY_BARS_DIR,
        labels_dir=LABELS_DIR,
        feature_cols=["open"],
        label_cols=["label_1d", "label_rank_5d"],
        dropna_label=False,
    )

    df = loader.load(
        instruments=["000003.SZ"],
        start_time="2000-01-04",
        end_time="2000-01-10",
    )

    assert list(df["feature"].columns) == ["open"]
    assert list(df["label"].columns) == ["label_1d", "label_rank_5d"]
    assert df["feature"].index.equals(df["label"].index)

if __name__ == "__main__":
    test_loader_returns_qlib_compatible_dataframe()