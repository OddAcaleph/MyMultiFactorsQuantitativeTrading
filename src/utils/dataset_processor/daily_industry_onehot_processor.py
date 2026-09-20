"""Generate daily one-hot encoded industry features from raw daily snapshots.

Reads partitioned daily industry snapshots (``year=YYYY/month=MM/YYYYMMDD.parquet``)
and writes daily one-hot parquet files in the same partition layout.

Output columns per day:
- ``trade_date``: trading date as int (YYYYMMDD).
- ``ts_code``: stock code.
- ``L1_<industry>``: one-hot columns for each L1 industry (int8, 0 or 1).

Stocks with no industry classification on a given date have all L1 columns
equal to 0 — they are NOT included in the output (zero rows for those stocks).
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


@dataclass
class DailyIndustryOneHotSummary:
    """Statistics for daily industry one-hot processing."""

    input_dir: Path
    output_dir: Path
    days_processed: int = 0
    days_skipped: int = 0
    days_failed: int = 0
    total_rows_written: int = 0
    industries: list[str] = field(default_factory=list)

    @property
    def n_industries(self) -> int:
        return len(self.industries)


class DailyIndustryOneHotProcessor:
    """Build daily industry one-hot wide tables from raw daily snapshots.

    The input directory must contain partitioned snapshots as produced by
    ``IndustryFetcher``::

        <input_dir>/year=YYYY/month=MM/YYYYMMDD.parquet

    The output uses the same partition layout with one-hot columns::

        <output_dir>/year=YYYY/month=MM/YYYYMMDD.parquet
    """

    DEFAULT_INPUT_DIR = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/raw_data/industry"
    )
    DEFAULT_OUTPUT_DIR = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/industry/daily_onehot"
    )
    DEFAULT_LOG_FILE = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/daily_industry_onehot_processing.log"
    )

    def __init__(
        self,
        input_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        level: str = "L1",
        logger: logging.Logger | None = None,
    ) -> None:
        self.input_dir = Path(input_dir or self.DEFAULT_INPUT_DIR)
        self.output_dir = Path(output_dir or self.DEFAULT_OUTPUT_DIR)
        self.level = level
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Industry input directory does not exist: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise ValueError(f"Industry input path is not a directory: {self.input_dir}")

    def discover_industries(self) -> list[str]:
        """Discover all industry categories across all snapshots."""
        industries: set[str] = set()
        for parquet_path in self._iter_snapshot_files():
            df = pd.read_parquet(parquet_path, columns=["industry", "level"])
            if self.level in df["level"].values:
                subset = df.loc[df["level"] == self.level, "industry"]
            else:
                subset = df["industry"]
            industries.update(subset.dropna().astype(str).unique())
        return sorted(industries)

    def process(
        self,
        force: bool = False,
        industries: list[str] | None = None,
    ) -> DailyIndustryOneHotSummary:
        """Process all daily snapshots into one-hot format."""
        summary = DailyIndustryOneHotSummary(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
        )

        if industries is None:
            self.logger.info("扫描全量行业分类...")
            industries = self.discover_industries()
        summary.industries = list(industries)
        self.logger.info("行业分类数（%s）：%d", self.level, len(industries))

        onehot_cols = [f"{self.level}_{ind}" for ind in industries]

        for parquet_path in self._iter_snapshot_files():
            rel = parquet_path.relative_to(self.input_dir)
            out_path = self.output_dir / rel

            if not force and out_path.exists():
                summary.days_skipped += 1
                continue

            try:
                df = pd.read_parquet(parquet_path)
                onehot_df = self._build_onehot_for_day(df, industries, onehot_cols)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                onehot_df.to_parquet(out_path, compression="snappy", index=False)
                summary.total_rows_written += len(onehot_df)
                summary.days_processed += 1
            except Exception as exc:
                self.logger.warning("处理 %s 失败：%s", parquet_path.name, exc)
                summary.days_failed += 1
                continue

            if summary.days_processed % 100 == 0:
                self.logger.info(
                    "进度：已处理 %d 天，跳过 %d 天，失败 %d 天，累计行数 %d",
                    summary.days_processed,
                    summary.days_skipped,
                    summary.days_failed,
                    summary.total_rows_written,
                )

        self._log_summary(summary)
        return summary

    def _build_onehot_for_day(
        self,
        df: pd.DataFrame,
        industries: list[str],
        onehot_cols: list[str],
    ) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["trade_date", "ts_code"] + onehot_cols)

        level_df = df[df["level"] == self.level].copy() if "level" in df.columns else df.copy()
        level_df = level_df[level_df["industry"].notna()].copy()

        if level_df.empty:
            return pd.DataFrame(columns=["trade_date", "ts_code"] + onehot_cols)

        level_df["_val"] = 1
        pivot = level_df.pivot_table(
            index=["trade_date", "ts_code"],
            columns="industry",
            values="_val",
            fill_value=0,
            aggfunc="max",
        ).astype("int8")
        pivot.columns.name = None
        pivot = pivot.reset_index()

        pivot["trade_date"] = pivot["trade_date"].astype(int)
        pivot["ts_code"] = pivot["ts_code"].astype(str)

        for col in onehot_cols:
            ind_name = col[len(self.level) + 1 :]
            if ind_name not in pivot.columns:
                pivot[col] = 0
            else:
                pivot[col] = pivot[ind_name].astype("int8")

        keep_cols = ["trade_date", "ts_code"] + onehot_cols
        for c in keep_cols:
            if c not in pivot.columns:
                pivot[c] = 0
        return pivot[keep_cols]

    def _iter_snapshot_files(self) -> Iterable[Path]:
        # Support both flat input_dir (year=*/month=*/*.parquet) and
        # input_dir being a specific year (month=*/*.parquet)
        year_dirs = sorted(self.input_dir.glob("year=*"))
        if year_dirs:
            for year_dir in year_dirs:
                if not year_dir.is_dir():
                    continue
                for month_dir in sorted(year_dir.glob("month=*")):
                    if not month_dir.is_dir():
                        continue
                    for parquet_file in sorted(month_dir.glob("*.parquet")):
                        yield parquet_file
        else:
            for month_dir in sorted(self.input_dir.glob("month=*")):
                if not month_dir.is_dir():
                    continue
                for parquet_file in sorted(month_dir.glob("*.parquet")):
                    yield parquet_file

    def _log_summary(self, summary: DailyIndustryOneHotSummary) -> None:
        self.logger.info("=" * 60)
        self.logger.info("日度行业 one-hot 处理完成")
        self.logger.info("输入目录：%s", summary.input_dir)
        self.logger.info("输出目录：%s", summary.output_dir)
        self.logger.info("行业分类数：%d", summary.n_industries)
        self.logger.info("已处理天数：%d", summary.days_processed)
        self.logger.info("跳过天数：%d", summary.days_skipped)
        self.logger.info("失败天数：%d", summary.days_failed)
        self.logger.info("累计写入行数：%d", summary.total_rows_written)
        self.logger.info("=" * 60)


def configure_logging(log_level: str = "INFO", log_file: str | Path | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    target_log_file = Path(log_file) if log_file is not None else DailyIndustryOneHotProcessor.DEFAULT_LOG_FILE
    target_log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(target_log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从日度行业快照生成日度 one-hot 行业特征。"
    )
    parser.add_argument(
        "--input-dir",
        default=str(DailyIndustryOneHotProcessor.DEFAULT_INPUT_DIR),
        help="原始日度行业快照目录（year=YYYY/month=MM 分区）。",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DailyIndustryOneHotProcessor.DEFAULT_OUTPUT_DIR),
        help="日度 one-hot 输出目录。",
    )
    parser.add_argument(
        "--level",
        default="L1",
        choices=["L1", "L2", "L3"],
        help="行业级别，默认 L1。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新处理所有日期，覆盖已有输出。",
    )
    parser.add_argument("--log-level", default="INFO", help="日志级别。")
    parser.add_argument(
        "--log-file",
        default=str(DailyIndustryOneHotProcessor.DEFAULT_LOG_FILE),
        help="日志文件路径。",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> DailyIndustryOneHotSummary:
    args = parse_args(argv)
    configure_logging(log_level=args.log_level, log_file=args.log_file)
    processor = DailyIndustryOneHotProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        level=args.level,
    )
    return processor.process(force=args.force)


if __name__ == "__main__":
    main()


__all__ = [
    "DailyIndustryOneHotSummary",
    "DailyIndustryOneHotProcessor",
    "configure_logging",
    "main",
    "parse_args",
]
