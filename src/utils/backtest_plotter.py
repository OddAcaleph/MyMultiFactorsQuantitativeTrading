"""Plot utilities for visualizing backtest reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class BacktestPlotter:
    """Create subplot charts from backtest report data.

    Expected report columns are compatible with ``PreparatoryBacktester`` output:

    - ``return``: strategy daily net return
    - ``benchmark``: benchmark daily return
    - ``account_value``: strategy account value

    The generated figure includes account value, cumulative return vs.
    benchmark, daily return, and drawdown with the maximum drawdown interval
    highlighted.
    """

    DEFAULT_REPORT_PATH = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/preparatory_backtest/backtest_report.parquet"
    )
    DEFAULT_OUTPUT_PATH = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/preparatory_backtest/backtest_plots.png"
    )

    def __init__(
        self,
        report_path: str | Path | None = None,
        output_path: str | Path | None = None,
        figsize: tuple[int, int] = (16, 12),
    ) -> None:
        self.report_path = Path(report_path) if report_path is not None else self.DEFAULT_REPORT_PATH
        self.output_path = Path(output_path) if output_path is not None else self.DEFAULT_OUTPUT_PATH
        self.figsize = figsize

    def load_report(self, report_path: str | Path | None = None) -> pd.DataFrame:
        """Load a parquet or pickle backtest report."""

        path = Path(report_path) if report_path is not None else self.report_path
        if not path.exists():
            raise FileNotFoundError(f"Backtest report does not exist: {path}")

        if path.suffix in {".pkl", ".pickle"}:
            report = pd.read_pickle(path)
        elif path.suffix == ".parquet":
            report = pd.read_parquet(path)
        else:
            raise ValueError(f"Unsupported report file suffix: {path.suffix}")

        return self._normalize_report(report)

    def plot(
        self,
        report: pd.DataFrame | None = None,
        output_path: str | Path | None = None,
        title: str = "Backtest Performance",
        show: bool = False,
    ) -> dict[str, Any]:
        """Plot backtest curves and save the figure.

        Returns
        -------
        dict
            Plot metadata including output path and max drawdown interval.
        """

        report = self._normalize_report(report) if report is not None else self.load_report()
        curves = self.compute_curves(report)
        dd_info = self.max_drawdown(curves["account_value"])

        fig, axes = plt.subplots(4, 1, figsize=self.figsize, sharex=True)
        fig.suptitle(title, fontsize=16)

        self._plot_account_value(axes[0], curves, dd_info)
        self._plot_cumulative_return(axes[1], curves, dd_info)
        self._plot_daily_return(axes[2], curves, dd_info)
        self._plot_drawdown(axes[3], curves, dd_info)

        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")

        fig.tight_layout(rect=[0, 0, 1, 0.97])

        path = Path(output_path) if output_path is not None else self.output_path
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

        return {
            "output_path": str(path),
            "max_drawdown": dd_info,
            "start_time": str(report.index.min().date()),
            "end_time": str(report.index.max().date()),
            "trading_days": int(len(report)),
        }

    @staticmethod
    def compute_curves(report: pd.DataFrame) -> pd.DataFrame:
        """Compute account, return, benchmark and drawdown curves."""

        report = BacktestPlotter._normalize_report(report)
        curves = pd.DataFrame(index=report.index)

        if "account_value" in report.columns:
            account_value = report["account_value"].astype(float)
            if "return" in report.columns and not report.empty:
                first_return = float(report["return"].iloc[0])
                initial_account = account_value.iloc[0] / (1.0 + first_return) if first_return != -1 else account_value.iloc[0]
            else:
                initial_account = account_value.iloc[0]
        else:
            initial_account = 1.0
            account_value = initial_account * (1.0 + report["return"].astype(float)).cumprod()

        curves["account_value"] = account_value
        curves["strategy_cum_return"] = account_value / initial_account - 1.0
        curves["daily_return"] = report["return"].astype(float)

        if "benchmark" in report.columns:
            benchmark_daily = report["benchmark"].astype(float)
            curves["benchmark_daily_return"] = benchmark_daily
            curves["benchmark_cum_return"] = (1.0 + benchmark_daily).cumprod() - 1.0
            curves["benchmark_value"] = initial_account * (1.0 + curves["benchmark_cum_return"])
        else:
            curves["benchmark_daily_return"] = np.nan
            curves["benchmark_cum_return"] = np.nan
            curves["benchmark_value"] = np.nan

        running_max = account_value.cummax()
        curves["drawdown"] = account_value / running_max - 1.0
        return curves

    @staticmethod
    def max_drawdown(account_value: pd.Series) -> dict[str, Any]:
        """Find the maximum drawdown interval from an account value curve."""

        if account_value.empty:
            return {"start": None, "end": None, "drawdown": np.nan}

        running_max = account_value.cummax()
        drawdown = account_value / running_max - 1.0
        trough = drawdown.idxmin()
        peak = account_value.loc[:trough].idxmax()
        return {
            "start": peak,
            "end": trough,
            "drawdown": float(drawdown.loc[trough]),
            "peak_value": float(account_value.loc[peak]),
            "trough_value": float(account_value.loc[trough]),
        }

    @staticmethod
    def _normalize_report(report: pd.DataFrame) -> pd.DataFrame:
        if report is None or report.empty:
            raise ValueError("Backtest report is empty")
        report = report.copy()
        if not isinstance(report.index, pd.DatetimeIndex):
            if "datetime" in report.columns:
                report["datetime"] = pd.to_datetime(report["datetime"])
                report = report.set_index("datetime")
            else:
                report.index = pd.to_datetime(report.index)
        report = report.sort_index()
        if "return" not in report.columns:
            raise KeyError("Backtest report must contain a 'return' column")
        return report

    @staticmethod
    def _highlight_drawdown(ax, dd_info: dict[str, Any]) -> None:
        if dd_info.get("start") is not None and dd_info.get("end") is not None:
            ax.axvspan(dd_info["start"], dd_info["end"], color="red", alpha=0.12, label="Max DD Interval")

    def _plot_account_value(self, ax, curves: pd.DataFrame, dd_info: dict[str, Any]) -> None:
        ax.plot(curves.index, curves["account_value"], label="Account Value", color="tab:blue")
        if curves["benchmark_value"].notna().any():
            ax.plot(curves.index, curves["benchmark_value"], label="Benchmark Value", color="tab:orange", alpha=0.85)
        self._highlight_drawdown(ax, dd_info)
        ax.set_ylabel("Amount")
        ax.set_title("Account Value")

    def _plot_cumulative_return(self, ax, curves: pd.DataFrame, dd_info: dict[str, Any]) -> None:
        ax.plot(curves.index, curves["strategy_cum_return"], label="Strategy Cum Return", color="tab:green")
        if curves["benchmark_cum_return"].notna().any():
            ax.plot(curves.index, curves["benchmark_cum_return"], label="Benchmark Cum Return", color="tab:orange")
        self._highlight_drawdown(ax, dd_info)
        ax.set_ylabel("Cumulative Return")
        ax.set_title("Strategy vs Benchmark")

    def _plot_daily_return(self, ax, curves: pd.DataFrame, dd_info: dict[str, Any]) -> None:
        ax.plot(curves.index, curves["daily_return"], label="Strategy Daily Return", color="tab:purple", alpha=0.8)
        if curves["benchmark_daily_return"].notna().any():
            ax.plot(curves.index, curves["benchmark_daily_return"], label="Benchmark Daily Return", color="tab:gray", alpha=0.65)
        self._highlight_drawdown(ax, dd_info)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_ylabel("Daily Return")
        ax.set_title("Daily Return")

    def _plot_drawdown(self, ax, curves: pd.DataFrame, dd_info: dict[str, Any]) -> None:
        ax.plot(curves.index, curves["drawdown"], label="Drawdown", color="tab:red")
        self._highlight_drawdown(ax, dd_info)
        if dd_info.get("end") is not None:
            ax.scatter([dd_info["end"]], [dd_info["drawdown"]], color="red", zorder=5, label="Max DD Trough")
            ax.annotate(
                f"Max DD: {dd_info['drawdown']:.2%}",
                xy=(dd_info["end"], dd_info["drawdown"]),
                xytext=(10, 20),
                textcoords="offset points",
                arrowprops={"arrowstyle": "->", "color": "red"},
            )
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_ylabel("Drawdown")
        ax.set_title("Drawdown")


__all__ = ["BacktestPlotter"]
