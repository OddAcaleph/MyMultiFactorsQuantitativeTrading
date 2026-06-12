from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backtester import PreparatoryBacktester


def _write_daily_label(label_dir: Path, trade_date: str, returns: dict[str, float]) -> None:
    output_file = label_dir / f"year={trade_date[:4]}" / f"month={trade_date[4:6]}" / f"{trade_date}.parquet"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "trade_date": [int(trade_date)] * len(returns),
            "ts_code": list(returns),
            "label_1d": list(returns.values()),
        }
    ).to_parquet(output_file, index=False)


def test_preparatory_backtester_runs_with_prediction_and_daily_labels(tmp_path):
    label_dir = tmp_path / "daily_labels"
    prediction_path = tmp_path / "pred_test.parquet"
    output_dir = tmp_path / "preparatory_backtest"

    _write_daily_label(label_dir, "20230103", {"000001.SZ": 0.01, "000002.SZ": 0.02, "000003.SZ": -0.01})
    _write_daily_label(label_dir, "20230104", {"000001.SZ": -0.02, "000002.SZ": 0.03, "000003.SZ": 0.01})

    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2023-01-03"), "000001.SZ"),
            (pd.Timestamp("2023-01-03"), "000002.SZ"),
            (pd.Timestamp("2023-01-03"), "000003.SZ"),
            (pd.Timestamp("2023-01-04"), "000001.SZ"),
            (pd.Timestamp("2023-01-04"), "000002.SZ"),
            (pd.Timestamp("2023-01-04"), "000003.SZ"),
        ],
        names=["datetime", "instrument"],
    )
    pd.DataFrame({"pred": [0.3, 0.2, 0.1, 0.1, 0.4, 0.3]}, index=index).to_parquet(prediction_path)

    backtester = PreparatoryBacktester(
        config_path=PROJECT_ROOT / "conf" / "preparatory_backtester_config.json",
        prediction_path=prediction_path,
        label_path=label_dir,
        output_dir=output_dir,
        start_time="2023-01-03",
        end_time="2023-01-04",
        account=100.0,
        strategy={"topk": 2, "n_drop": 1},
        exchange_kwargs={"open_cost": 0.0, "close_cost": 0.0, "impact_cost": 0.0},
    )

    result = backtester.run(save=True)

    report = result["report"]
    positions = result["positions"]
    assert list(report.columns) == [
        "return",
        "gross_return",
        "benchmark",
        "excess_return",
        "cost",
        "turnover",
        "account_value",
        "holdings_count",
    ]
    assert len(report) == 2
    assert len(positions) == 4
    assert np.isclose(report.loc[pd.Timestamp("2023-01-03"), "return"], 0.015)
    assert np.isclose(report.loc[pd.Timestamp("2023-01-04"), "return"], 0.02)
    assert (output_dir / "backtest_report.parquet").exists()
    assert (output_dir / "positions.parquet").exists()
    assert (output_dir / "analysis.json").exists()
