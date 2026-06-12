"""Synchronous and asynchronous orchestration for feature IC validation."""

from __future__ import annotations

import asyncio
import copy
import json
from concurrent.futures import Executor
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .base import BaseICValidationStep, ValidationResult
from .group_return import GroupReturnAnalyzer
from .ic_analysis import DailyICAnalyzer
from .long_short_backtest import LongShortBacktestAnalyzer
from .score_distribution import ScoreDistributionAnalyzer
from .turnover import TurnoverAnalyzer


class ICValidator:
    """Validate feature IC with Preparatory analysis steps.

    Parameters
    ----------
    label_col
        Realized return column used as Qlib ``label``.
    feature_cols
        Feature/score columns to validate.  If omitted, all non-label columns
        are validated one by one.
    groups
        Number of quantile groups for stratified return and turnover analysis.
    topk
        Number of names in long/short baskets.
    freq
        Frequency passed to Qlib ``risk_analysis``.
    """

    def __init__(
        self,
        label_col: str = "label",
        feature_cols: Sequence[str] | None = None,
        groups: int = 5,
        topk: int = 50,
        turnover_lag: int = 1,
        freq: str = "day",
        reverse: bool = False,
        use_qlib_exchange: bool = False,
        qlib_backtest_kwargs: Mapping[str, Any] | None = None,
        steps: Mapping[str, BaseICValidationStep] | None = None,
    ) -> None:
        self.label_col = label_col
        self.feature_cols = list(feature_cols) if feature_cols is not None else None
        self.groups = groups
        self.topk = topk
        self.turnover_lag = turnover_lag
        self.freq = freq
        self.reverse = reverse
        self.use_qlib_exchange = use_qlib_exchange
        self.qlib_backtest_kwargs = dict(qlib_backtest_kwargs or {})
        self.steps = dict(steps) if steps is not None else self._default_steps()

    def _default_steps(self) -> dict[str, BaseICValidationStep]:
        common = {"label_col": self.label_col}
        return {
            "ic": DailyICAnalyzer(freq=self.freq, **common),
            "group_return": GroupReturnAnalyzer(groups=self.groups, reverse=self.reverse, freq=self.freq, **common),
            "long_short": LongShortBacktestAnalyzer(
                topk=self.topk,
                freq=self.freq,
                use_qlib_exchange=self.use_qlib_exchange,
                qlib_backtest_kwargs=self.qlib_backtest_kwargs,
                **common,
            ),
            "turnover": TurnoverAnalyzer(groups=self.groups, lag=self.turnover_lag, **common),
            "score_distribution": ScoreDistributionAnalyzer(**common),
        }

    def validate(self, pred_label: pd.DataFrame, save_dir: str | Path | None = None) -> dict[str, Any]:
        """Synchronously compute all validation metrics for each feature."""

        features = BaseICValidationStep.flatten_feature_columns(self.feature_cols, pred_label, self.label_col)
        results = {feature: self.validate_feature(pred_label, feature) for feature in features}
        output = {"features": results, "summary": self._summary(results)}
        if save_dir is not None:
            self.save_results(output, save_dir)
        return output

    def validate_feature(self, pred_label: pd.DataFrame, feature_col: str) -> dict[str, Any]:
        """Synchronously compute all steps for a single feature/score column."""

        step_results: dict[str, Any] = {}
        local_steps = copy.deepcopy(self.steps)
        for name, step in local_steps.items():
            step.score_col = feature_col
            step.label_col = self.label_col
            result = step.validate(pred_label)
            step_results[name] = {"metrics": result.metrics, "data": result.data}
        return step_results

    async def async_validate(
        self,
        pred_label: pd.DataFrame,
        save_dir: str | Path | None = None,
        executor: Executor | None = None,
    ) -> dict[str, Any]:
        """Asynchronously compute validation metrics for all features."""

        features = BaseICValidationStep.flatten_feature_columns(self.feature_cols, pred_label, self.label_col)
        loop = asyncio.get_running_loop()
        tasks = [loop.run_in_executor(executor, self.validate_feature, pred_label, feature) for feature in features]
        feature_results = await asyncio.gather(*tasks)
        results = dict(zip(features, feature_results))
        output = {"features": results, "summary": self._summary(results)}
        if save_dir is not None:
            await loop.run_in_executor(executor, self.save_results, output, save_dir)
        return output

    async def async_validate_steps(self, pred_label: pd.DataFrame, feature_col: str) -> dict[str, Any]:
        """Asynchronously compute each validation step for one feature."""

        local_steps = copy.deepcopy(self.steps)

        async def _run_step(name: str, step: BaseICValidationStep) -> tuple[str, ValidationResult]:
            step.score_col = feature_col
            step.label_col = self.label_col
            return name, await asyncio.to_thread(step.validate, pred_label)

        pairs = await asyncio.gather(*[_run_step(name, step) for name, step in local_steps.items()])
        return {name: {"metrics": result.metrics, "data": result.data} for name, result in pairs}

    @staticmethod
    def _summary(results: Mapping[str, Any]) -> dict[str, Any]:
        rows: list[dict[str, Any]] = []
        for feature, feature_result in results.items():
            ic_metrics = feature_result.get("ic", {}).get("metrics", {}).get("daily", {})
            rank_ic = ic_metrics.get("Rank IC", {})
            ic = ic_metrics.get("IC", {})
            group_metrics = feature_result.get("group_return", {}).get("metrics", {})
            long_short_metrics = feature_result.get("long_short", {}).get("metrics", {})
            rows.append(
                {
                    "feature": feature,
                    "ic_mean": ic.get("ic_mean"),
                    "ic_std": ic.get("ic_std"),
                    "icir": ic.get("icir"),
                    "rank_ic_mean": rank_ic.get("ic_mean"),
                    "rank_icir": rank_ic.get("icir"),
                    "group_top_minus_bottom_mean": group_metrics.get("monotonicity", {}).get("top_minus_bottom_mean"),
                    "long_short_mean": long_short_metrics.get("mean_return", {}).get("long_short"),
                }
            )
        return {"table": rows}

    @classmethod
    def save_results(cls, results: Mapping[str, Any], save_dir: str | Path) -> None:
        """Save metrics JSON and dataframe artifacts as CSV for every feature/step."""

        root = Path(save_dir)
        root.mkdir(parents=True, exist_ok=True)
        metrics_only = cls._strip_dataframes(results)
        with (root / "metrics.json").open("w", encoding="utf-8") as f:
            json.dump(BaseICValidationStep.json_safe(metrics_only), f, ensure_ascii=False, indent=2)

        for feature, feature_result in results.get("features", {}).items():
            feature_dir = root / str(feature)
            feature_dir.mkdir(parents=True, exist_ok=True)
            for step_name, step_result in feature_result.items():
                for data_name, data in step_result.get("data", {}).items():
                    if isinstance(data, pd.Series):
                        data.to_frame(name=data.name or "value").to_csv(feature_dir / f"{step_name}_{data_name}.csv")
                    elif isinstance(data, pd.DataFrame):
                        data.to_csv(feature_dir / f"{step_name}_{data_name}.csv")

    @classmethod
    def _strip_dataframes(cls, obj: Any) -> Any:
        if isinstance(obj, Mapping):
            return {k: cls._strip_dataframes(v) for k, v in obj.items() if k != "data"}
        if isinstance(obj, list):
            return [cls._strip_dataframes(v) for v in obj]
        if isinstance(obj, (pd.DataFrame, pd.Series)):
            return None
        return obj


__all__ = ["ICValidator"]
