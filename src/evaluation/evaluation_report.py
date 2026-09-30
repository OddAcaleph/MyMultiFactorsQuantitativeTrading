"""Unified evaluation report that combines all metrics and backtest results.

The :class:`EvaluationReport` orchestrates:

1. Overall IC / Rank IC / ICIR
2. Long-side IC
3. Top-quantile return (decile)
4. Top-K hit rate
5. Upside capture
6. Downside filter score
7. Return-decile within-group IC
8. Long-backtest metrics (via SimpleBacktester)

It produces a JSON-serializable summary plus optional per-metric dataframes.
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation.metrics import (
    calc_downside_filter_score,
    calc_long_side_ic,
    calc_overall_ic,
    calc_return_decile_ic,
    calc_top_k_hit_rate,
    calc_top_quantile_return,
    calc_upside_capture,
)
from utils import load_simple_backtester_config, PROJECT_ROOT


logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for the evaluation report.

    Attributes
    ----------
    score_col
        Column name of the prediction score in *pred_label*.
    primary_label_col
        Primary label column used for IC / hit-rate calculations.
    return_cols
        Mapping of horizon name -> return column name, e.g.
        ``{"5d": "label_5d", "10d": "label_10d", "20d": "label_20d"}``.
    n_groups
        Number of quantile groups for top-quantile return (default 10).
    top_ks
        List of top-K sizes for hit-rate analysis.
    upside_realized_frac
        Fraction of top-realized-return stocks for upside capture.
    upside_predicted_fracs
        Predicted top fractions to check for upside capture.
    downside_bottom_frac
        Bottom fraction for downside filter analysis.
    downside_crash_threshold
        Return threshold defining a "crash".
    run_backtest
        Whether to run SimpleBacktester for long-backtest metrics.
    backtest_config_path
        Path to simple_backtester config JSON.  If None, uses project default.
    backtest_overrides
        Overrides for the backtest config (e.g. prediction_path, output_dir).
    benchmark_returns
        Optional daily benchmark return series for hit-rate benchmark comparison.
    """

    score_col: str = "pred"
    primary_label_col: str = "label_5d"
    return_label_col: str | None = None
    return_cols: Mapping[str, str] = field(
        default_factory=lambda: {"5d": "label_5d", "10d": "label_10d", "20d": "label_20d"}
    )
    n_groups: int = 10
    top_ks: Sequence[int] = (50, 100, 200)
    upside_realized_frac: float = 0.10
    upside_predicted_fracs: Sequence[float] = (0.10, 0.20)
    downside_bottom_frac: float = 0.10
    downside_crash_threshold: float = -0.095
    return_decile_n_groups: int = 10
    run_backtest: bool = True
    backtest_config_path: str | Path | None = None
    backtest_overrides: Mapping[str, Any] | None = None
    benchmark_returns: pd.Series | None = None


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy / pandas types to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    if isinstance(obj, pd.Series):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (pd.Timestamp, Path)):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, bool)):
        return obj
    return str(obj)


class EvaluationReport:
    """Generate a comprehensive evaluation report from predictions.

    Parameters
    ----------
    config
        Evaluation configuration.
    """

    def __init__(self, config: EvaluationConfig | None = None) -> None:
        self.config = config or EvaluationConfig()
        self._result: dict[str, Any] | None = None
        self._dataframes: dict[str, pd.DataFrame | pd.Series] = {}

    def generate(self, pred_label: pd.DataFrame) -> dict[str, Any]:
        """Run all evaluation metrics and return the full report.

        Parameters
        ----------
        pred_label
            DataFrame indexed by (datetime, instrument) containing the score
            column and all required label/return columns.

        Returns
        -------
        dict
            Nested dictionary with ``summary`` (JSON-safe scalars) and ``details``
            (intermediate series / frames).
        """
        cfg = self.config
        score_col = cfg.score_col
        primary_label = cfg.primary_label_col
        # Return label for hit-rate / upside / downside (raw return; falls back to primary_label)
        return_label = cfg.return_label_col or cfg.primary_label_col

        # Validate required columns
        required = {score_col, primary_label}
        missing = required - set(pred_label.columns)
        if missing:
            raise ValueError(f"pred_label missing required columns: {missing}")

        result: dict[str, Any] = {"summary": {}, "details": {}}

        # 1. Overall IC
        logger.info("Computing overall IC...")
        overall_ic = calc_overall_ic(pred_label, score_col=score_col, label_col=primary_label)
        result["summary"]["overall_ic"] = {
            "ic_mean": overall_ic["ic"]["ic_mean"],
            "ic_std": overall_ic["ic"]["ic_std"],
            "icir": overall_ic["ic"]["icir"],
            "ic_positive_rate": overall_ic["ic"]["positive_rate"],
            "rank_ic_mean": overall_ic["rank_ic"]["ic_mean"],
            "rank_ic_std": overall_ic["rank_ic"]["ic_std"],
            "rank_icir": overall_ic["rank_ic"]["icir"],
            "rank_ic_positive_rate": overall_ic["rank_ic"]["positive_rate"],
            "ic_days": overall_ic["ic"]["count"],
        }
        self._dataframes["daily_ic"] = overall_ic["daily_ic"]
        self._dataframes["daily_rank_ic"] = overall_ic["daily_rank_ic"]

        # 2. Long-side IC
        logger.info("Computing long-side IC...")
        long_ic = calc_long_side_ic(pred_label, score_col=score_col, label_col=primary_label)
        result["summary"]["long_side_ic"] = {
            "ic_mean": long_ic["ic"]["ic_mean"],
            "ic_std": long_ic["ic"]["ic_std"],
            "icir": long_ic["ic"]["icir"],
            "rank_ic_mean": long_ic["rank_ic"]["ic_mean"],
            "rank_ic_std": long_ic["rank_ic"]["ic_std"],
            "rank_icir": long_ic["rank_ic"]["icir"],
            "ic_days": long_ic["ic"]["count"],
        }
        self._dataframes["daily_long_side_ic"] = long_ic["daily_long_side_ic"]
        self._dataframes["daily_long_side_rank_ic"] = long_ic["daily_long_side_rank_ic"]

        # 3. Top quantile return
        logger.info("Computing top quantile return...")
        top_q = calc_top_quantile_return(
            pred_label,
            score_col=score_col,
            return_cols=dict(cfg.return_cols),
            n_groups=cfg.n_groups,
        )
        result["summary"]["top_quantile_return"] = {}
        for horizon, hdata in top_q.get("horizons", {}).items():
            result["summary"]["top_quantile_return"][horizon] = {
                "top_group_mean_return": hdata.get("top_group_mean_return"),
                "all_groups_mean_return": hdata.get("all_groups_mean_return", {}),
            }
            if "top_group_daily_return" in hdata:
                self._dataframes[f"top_group_daily_return_{horizon}"] = hdata["top_group_daily_return"]
        result["summary"]["top_quantile_return"]["n_groups"] = top_q.get("n_groups", cfg.n_groups)

        # 4. Top-K hit rate
        logger.info("Computing top-K hit rate...")
        hit_rate = calc_top_k_hit_rate(
            pred_label,
            score_col=score_col,
            return_col=return_label,
            ks=cfg.top_ks,
            benchmark_returns=cfg.benchmark_returns,
        )
        result["summary"]["top_k_hit_rate"] = {}
        for k, kdata in hit_rate.items():
            if not kdata:
                continue
            summary_k = {
                    "beat_median_mean": kdata.get("beat_median_mean"),
                    "positive_mean": kdata.get("positive_mean"),
                    "mean_return": kdata.get("mean_return"),
                }
            if "beat_benchmark_mean" in kdata:
                summary_k["beat_benchmark_mean"] = kdata["beat_benchmark_mean"]
            result["summary"]["top_k_hit_rate"][k] = summary_k
            if "daily" in kdata:
                self._dataframes[f"top_k_hit_rate_daily_k{k}"] = kdata["daily"]

        # 5. Upside capture
        logger.info("Computing upside capture...")
        upside = calc_upside_capture(
            pred_label,
            score_col=score_col,
            return_col=return_label,
            realized_top_frac=cfg.upside_realized_frac,
            predicted_top_fracs=cfg.upside_predicted_fracs,
        )
        result["summary"]["upside_capture"] = {}
        for frac in cfg.upside_predicted_fracs:
            key = f"{int(frac * 100)}%"
            result["summary"]["upside_capture"][f"pred_top_{key}"] = upside.get(key)
        result["summary"]["upside_capture"]["realized_top_frac"] = cfg.upside_realized_frac
        if "daily" in upside:
            for key, series in upside["daily"].items():
                self._dataframes[f"upside_capture_daily_{key}"] = series

        # 6. Downside filter score
        logger.info("Computing downside filter score...")
        downside = calc_downside_filter_score(
            pred_label,
            score_col=score_col,
            return_col=return_label,
            bottom_pred_frac=cfg.downside_bottom_frac,
            crash_threshold=cfg.downside_crash_threshold,
        )
        result["summary"]["downside_filter_score"] = {
            "crash_hit_rate": downside.get("crash_hit_rate"),
            "crash_precision": downside.get("crash_precision"),
            "crash_recall": downside.get("crash_recall"),
            "bottom_frac": downside.get("bottom_frac"),
            "crash_threshold": downside.get("crash_threshold"),
        }
        if "daily_hit_rate" in downside:
            self._dataframes["downside_daily_hit_rate"] = downside["daily_hit_rate"]
        if "daily_precision" in downside:
            self._dataframes["downside_daily_precision"] = downside["daily_precision"]
        if "daily_recall" in downside:
            self._dataframes["downside_daily_recall"] = downside["daily_recall"]

        # 7. Return-decile within-group IC
        logger.info("Computing return-decile within-group IC...")
        ret_decile_ic = calc_return_decile_ic(
            pred_label,
            score_col=score_col,
            return_col=return_label,
            n_groups=cfg.return_decile_n_groups,
        )
        result["summary"]["return_decile_ic"] = {
            "per_group_ic": ret_decile_ic.get("per_group_ic", {}),
            "n_groups": ret_decile_ic.get("n_groups", cfg.return_decile_n_groups),
        }
        if not ret_decile_ic.get("daily_per_group_ic", pd.DataFrame()).empty:
            self._dataframes["daily_return_decile_ic"] = ret_decile_ic["daily_per_group_ic"]

        # 8. Long backtest metrics
        if cfg.run_backtest:
            logger.info("Running long backtest...")
            backtest_result = self._run_backtest(pred_label)
            result["summary"]["long_backtest_metrics"] = backtest_result["summary"]
            if "report" in backtest_result:
                self._dataframes["backtest_report"] = backtest_result["report"]
            if "positions" in backtest_result:
                self._dataframes["backtest_positions"] = backtest_result["positions"]

        self._result = result
        return result

    def _run_backtest(self, pred_label: pd.DataFrame) -> dict[str, Any]:
        """Run SimpleBacktester and extract key metrics.

        The backtester reads predictions from disk (prediction_path).  We write the
        pred_df to a temp location if needed, or pass it in-memory.
        """
        from backtester import SimpleBacktester

        cfg = self.config
        overrides = dict(cfg.backtest_overrides or {})

        # Build backtester instance
        bt = SimpleBacktester(
            config_path=cfg.backtest_config_path,
            **overrides,
        )

        # Run backtest — pass pred_df directly to avoid writing to disk
        bt_result = bt.run(save=False, pred_df=pred_label)

        analysis = bt_result.get("analysis", {})
        summary = analysis.get("summary", {})
        return_metrics = analysis.get("return", {})

        long_metrics = {
            "total_return": return_metrics.get("total_return"),
            "annualized_return": return_metrics.get("annualized_return"),
            "annualized_volatility": return_metrics.get("annualized_volatility"),
            "sharpe": return_metrics.get("sharpe"),
            "max_drawdown": return_metrics.get("max_drawdown"),
            "calmar": return_metrics.get("calmar"),
            "sortino": return_metrics.get("sortino"),
            "mean_turnover": summary.get("mean_turnover"),
            "total_cost": summary.get("total_cost"),
            "trading_days": summary.get("trading_days"),
            "final_account_value": summary.get("final_account_value"),
            "benchmark_total_return": summary.get("benchmark_total_return"),
            "start_time": summary.get("start_time"),
            "end_time": summary.get("end_time"),
            "deal_price": summary.get("deal_price"),
            "benchmark": summary.get("benchmark"),
        }

        return {
            "summary": long_metrics,
            "report": bt_result.get("report", pd.DataFrame()),
            "positions": bt_result.get("positions", pd.DataFrame()),
            "trades": bt_result.get("trades", pd.DataFrame()),
        }

    def save_report(self, output_dir: str | Path) -> Path:
        """Save the evaluation report to disk.

        Saves:
        - ``evaluation_report.json`` — JSON-safe summary metrics
        - ``evaluation_data.h5`` — intermediate dataframes (daily IC, hit rates, etc.)
        - ``backtest_report.csv`` — backtest daily report (if backtest ran)
        """
        if self._result is None:
            raise RuntimeError("No report generated yet. Call generate() first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # JSON summary
        summary_path = output_dir / "evaluation_report.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(self._result["summary"]), f, ensure_ascii=False, indent=2)
        logger.info(f"Evaluation report saved to {summary_path}")

        # Intermediate dataframes as parquet
        data_dir = output_dir / "evaluation_data"
        data_dir.mkdir(exist_ok=True)
        for name, df in self._dataframes.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                safe_name = name.replace("/", "_")
                df.to_parquet(data_dir / f"{safe_name}.parquet")
            elif isinstance(df, pd.Series) and not df.empty:
                safe_name = name.replace("/", "_")
                df.to_frame().to_parquet(data_dir / f"{safe_name}.parquet")

        logger.info(f"Evaluation data saved to {data_dir}")
        return summary_path


def run_evaluation(
    pred_label: pd.DataFrame,
    config: EvaluationConfig | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Convenience function: generate report and optionally save it.

    Parameters
    ----------
    pred_label
        Prediction + label DataFrame indexed by (datetime, instrument).
    config
        Evaluation configuration.
    output_dir
        If provided, save report artifacts to this directory.

    Returns
    -------
    dict
        Full evaluation result (same as ``EvaluationReport.generate()``).
    """
    report = EvaluationReport(config=config)
    result = report.generate(pred_label)
    if output_dir is not None:
        report.save_report(output_dir)
    return result
