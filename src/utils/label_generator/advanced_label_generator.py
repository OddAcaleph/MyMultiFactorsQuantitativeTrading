"""Advanced label generator — industry-excess and classification labels.

Reads existing daily label parquet files (with forward returns) and daily
industry data, then computes additional label types:

1. ``excess_industry_rank_{h}d`` — rank of (future_return - industry_mean_return)
2. ``excess_market_rank_{h}d`` — rank of (future_return - market_mean_return)
3. ``up_top20_cls_{h}d`` — binary: 1 if future return rank >= 80th percentile
4. ``excess_industry_top20_cls_{h}d`` — binary: 1 if industry-excess rank >= 80th percentile

Output is written to a separate directory so existing label data is never modified.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@dataclass
class AdvancedLabelStats:
    input_dir: Path
    industry_dir: Path
    output_dir: Path
    files_written: int
    rows_written: int
    horizons: tuple[int, ...]
    label_columns: tuple[str, ...]


class AdvancedLabelGenerator:
    """Generate industry-excess and classification labels from existing labels.

    Parameters
    ----------
    input_dir
        Path to existing daily label parquet files (year=YYYY/month=MM/*.parquet).
    industry_dir
        Path to daily industry snapshot parquet files (same partition structure).
    output_dir
        Output directory for new label files (separate from input).
    horizons
        Forward-return horizons in days to compute labels for.
    industry_level
        Industry level column to use (e.g. "L1").
    top_frac
        Fraction threshold for "top X%" classification labels (default 0.20).
    start_date, end_date
        Optional date range filter (YYYYMMDD int or string).
    """

    KEY_COLUMNS: tuple[str, str] = ("trade_date", "ts_code")

    def __init__(
        self,
        input_dir: str | Path,
        industry_dir: str | Path,
        output_dir: str | Path,
        horizons: Sequence[int] = (5, 10, 20),
        industry_level: str = "L1",
        top_frac: float = 0.20,
        start_date: str | int | None = None,
        end_date: str | int | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.input_dir = Path(input_dir)
        self.industry_dir = Path(industry_dir)
        self.output_dir = Path(output_dir)
        self.horizons = tuple(horizons)
        self.industry_level = industry_level
        self.top_frac = float(top_frac)
        self.start_date = self._parse_date(start_date, "start_date")
        self.end_date = self._parse_date(end_date, "end_date")
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self._validate_inputs()

    @staticmethod
    def _parse_date(value: str | int | None, name: str) -> int | None:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        s = str(value).replace("-", "")
        try:
            return int(s)
        except ValueError:
            raise ValueError(f"{name} must be YYYYMMDD, got {value!r}")

    def _validate_inputs(self) -> None:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input label directory not found: {self.input_dir}")
        if not self.industry_dir.exists():
            raise FileNotFoundError(f"Industry directory not found: {self.industry_dir}")
        if self.start_date is not None and self.end_date is not None and self.start_date > self.end_date:
            raise ValueError(f"start_date must be <= end_date")
        if not (0 < self.top_frac < 1):
            raise ValueError(f"top_frac must be in (0, 1), got {self.top_frac}")

    def _list_label_files(self) -> list[Path]:
        files = sorted(self.input_dir.glob("year=*/month=*/*.parquet"))
        result = []
        for f in files:
            try:
                date_int = int(f.stem)
            except ValueError:
                continue
            if self.start_date is not None and date_int < self.start_date:
                continue
            if self.end_date is not None and date_int > self.end_date:
                continue
            result.append(f)
        return result

    def _list_industry_files(self) -> list[Path]:
        files = sorted(self.industry_dir.glob("year=*/month=*/*.parquet"))
        result = []
        for f in files:
            try:
                date_int = int(f.stem)
            except ValueError:
                continue
            if self.start_date is not None and date_int < self.start_date:
                continue
            if self.end_date is not None and date_int > self.end_date:
                continue
            result.append(f)
        return result

    def _load_industry_map(self, files: list[Path]) -> pd.DataFrame:
        """Load industry classification for all dates in *files*.

        Returns DataFrame with columns [trade_date, ts_code, industry].
        """
        dfs = []
        for f in files:
            df = pd.read_parquet(f)
            # Filter to the desired level
            if "level" in df.columns:
                df = df[df["level"] == self.industry_level].copy()
            if "industry" not in df.columns or "ts_code" not in df.columns:
                continue
            subset = df[["trade_date", "ts_code", "industry"]].copy()
            dfs.append(subset)
        if not dfs:
            return pd.DataFrame(columns=["trade_date", "ts_code", "industry"])
        result = pd.concat(dfs, ignore_index=True)
        # Normalize trade_date to int for consistent comparison
        if not result.empty:
            result["trade_date"] = result["trade_date"].astype(int)
        return result

    def _compute_labels_for_day(self, label_df: pd.DataFrame, industry_series: pd.Series) -> dict[str, pd.Series]:
        """Compute all advanced labels for a single cross-section.

        Parameters
        ----------
        label_df
            DataFrame for one trade_date, indexed by ts_code, with label columns.
        industry_series
            Series mapping ts_code -> industry name.

        Returns
        -------
        dict
            Mapping of label column name -> Series (indexed by ts_code).
        """
        result: dict[str, pd.Series] = {}
        n_stocks = len(label_df)
        if n_stocks == 0:
            return result

        for h in self.horizons:
            ret_col = f"label_{h}d"
            if ret_col not in label_df.columns:
                continue

            returns = label_df[ret_col].astype(float)
            valid_mask = returns.replace([np.inf, -np.inf], np.nan).notna()
            valid_returns = returns[valid_mask]
            if valid_returns.empty:
                result[f"excess_industry_rank_{h}d"] = pd.Series(np.nan, index=returns.index)
                result[f"excess_market_rank_{h}d"] = pd.Series(np.nan, index=returns.index)
                result[f"up_top20_cls_{h}d"] = pd.Series(np.nan, index=returns.index)
                result[f"excess_industry_top20_cls_{h}d"] = pd.Series(np.nan, index=returns.index)
                continue

            # 1. Market excess return rank
            market_mean = valid_returns.mean()
            excess_market = valid_returns - market_mean
            excess_market_rank = excess_market.rank(pct=True)
            full_excess_market_rank = pd.Series(np.nan, index=returns.index)
            full_excess_market_rank.loc[valid_mask] = excess_market_rank
            result[f"excess_market_rank_{h}d"] = full_excess_market_rank

            # 3. Up top20 classification (based on raw return rank)
            return_rank = valid_returns.rank(pct=True)
            up_cls = (return_rank >= (1 - self.top_frac)).astype(float)
            full_up_cls = pd.Series(np.nan, index=returns.index)
            full_up_cls.loc[valid_mask] = up_cls
            result[f"up_top20_cls_{h}d"] = full_up_cls

            # 2 & 4. Industry excess return (need industry mapping)
            # Build industry-aligned series
            aligned_industry = industry_series.reindex(returns.index)
            valid_industry_mask = valid_mask & aligned_industry.notna()
            if valid_industry_mask.sum() > 0:
                ind_returns = pd.DataFrame({
                    "ret": returns[valid_industry_mask],
                    "industry": aligned_industry[valid_industry_mask],
                })
                industry_mean_ret = ind_returns.groupby("industry")["ret"].transform("mean")
                excess_industry = ind_returns["ret"] - industry_mean_ret
                excess_industry_rank = excess_industry.rank(pct=True)

                full_excess_ind_rank = pd.Series(np.nan, index=returns.index)
                full_excess_ind_rank.loc[valid_industry_mask] = excess_industry_rank
                result[f"excess_industry_rank_{h}d"] = full_excess_ind_rank

                # 4. Industry excess top20 classification
                excess_ind_cls = (excess_industry_rank >= (1 - self.top_frac)).astype(float)
                full_excess_ind_cls = pd.Series(np.nan, index=returns.index)
                full_excess_ind_cls.loc[valid_industry_mask] = excess_ind_cls
                result[f"excess_industry_top20_cls_{h}d"] = full_excess_ind_cls
            else:
                result[f"excess_industry_rank_{h}d"] = pd.Series(np.nan, index=returns.index)
                result[f"excess_industry_top20_cls_{h}d"] = pd.Series(np.nan, index=returns.index)

        return result

    def process(self, batch_size: int = 250) -> AdvancedLabelStats:
        """Run the full advanced label generation pipeline.

        Parameters
        ----------
        batch_size
            Number of label files to process per batch.  Industry data is
            loaded per-batch to keep memory usage bounded.
        """

        label_files = self._list_label_files()
        if not label_files:
            raise FileNotFoundError(f"No label files found in {self.input_dir} for the date range")

        industry_files = self._list_industry_files()
        industry_file_map = {int(f.stem): f for f in industry_files}

        self.logger.info("开始生成高级 label: input=%s, industry=%s, output=%s",
                         self.input_dir, self.industry_dir, self.output_dir)
        self.logger.info("Label 文件数: %d, Industry 文件数: %d, Horizons: %s, batch_size=%d",
                         len(label_files), len(industry_files), self.horizons, batch_size)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        label_columns = []
        for h in self.horizons:
            label_columns.extend([
                f"excess_industry_rank_{h}d",
                f"excess_market_rank_{h}d",
                f"up_top20_cls_{h}d",
                f"excess_industry_top20_cls_{h}d",
            ])

        files_written = 0
        rows_written = 0
        total_batches = (len(label_files) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(label_files))
            batch_files = label_files[start:end]

            # Load industry data for this batch's date range
            batch_dates = [int(f.stem) for f in batch_files]
            batch_ind_files = [industry_file_map[d] for d in batch_dates if d in industry_file_map]
            if batch_ind_files:
                industry_df = self._load_industry_map(batch_ind_files)
                self.logger.info("第 %d/%d 批: label_files=%d, industry_files=%d",
                                 batch_idx + 1, total_batches, len(batch_files), len(batch_ind_files))
            else:
                industry_df = pd.DataFrame(columns=["trade_date", "ts_code", "industry"])
                self.logger.info("第 %d/%d 批: label_files=%d, 无匹配行业数据",
                                 batch_idx + 1, total_batches, len(batch_files))

            for label_file in batch_files:
                date_int = int(label_file.stem)
                label_df = pd.read_parquet(label_file)

                if label_df.empty:
                    continue

                # Get industry for this date
                if not industry_df.empty:
                    day_industry = industry_df[industry_df["trade_date"] == date_int]
                    if day_industry.empty:
                        industry_series = pd.Series(dtype=str)
                    else:
                        industry_series = day_industry.set_index("ts_code")["industry"]
                else:
                    industry_series = pd.Series(dtype=str)

                # Compute labels
                label_indexed = label_df.set_index("ts_code")
                new_labels = self._compute_labels_for_day(label_indexed, industry_series)

                # Build output
                output_data = {"trade_date": label_df["trade_date"].values, "ts_code": label_df["ts_code"].values}
                for col_name in label_columns:
                    if col_name in new_labels:
                        output_data[col_name] = new_labels[col_name].values
                    else:
                        output_data[col_name] = np.nan

                output_df = pd.DataFrame(output_data)
                rows_written += len(output_df)

                # Write partitioned by date
                date_str = f"{date_int:08d}"
                output_file = self.output_dir / f"year={date_str[:4]}" / f"month={date_str[4:6]}" / f"{date_str}.parquet"
                output_file.parent.mkdir(parents=True, exist_ok=True)
                output_df.to_parquet(output_file, index=False)
                files_written += 1

        self.logger.info("高级 label 生成完成: files_written=%d, rows_written=%d", files_written, rows_written)

        return AdvancedLabelStats(
            input_dir=self.input_dir,
            industry_dir=self.industry_dir,
            output_dir=self.output_dir,
            files_written=files_written,
            rows_written=rows_written,
            horizons=self.horizons,
            label_columns=tuple(label_columns),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate advanced labels (industry-excess, classification).")
    parser.add_argument("--input-dir", required=True, help="现有 daily label parquet 目录")
    parser.add_argument("--industry-dir", required=True, help="日度行业快照 parquet 目录")
    parser.add_argument("--output-dir", required=True, help="新 label 输出目录（独立目录，不修改原数据）")
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 10, 20],
                        help="收益周期（天数），默认 5 10 20")
    parser.add_argument("--industry-level", default="L1", help="行业层级，默认 L1")
    parser.add_argument("--top-frac", type=float, default=0.20, help="Top X% 分类阈值，默认 0.20")
    parser.add_argument("--start-date", default=None, help="起始日期 YYYYMMDD")
    parser.add_argument("--end-date", default=None, help="结束日期 YYYYMMDD")
    return parser.parse_args()


def main() -> AdvancedLabelStats:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    generator = AdvancedLabelGenerator(
        input_dir=args.input_dir,
        industry_dir=args.industry_dir,
        output_dir=args.output_dir,
        horizons=args.horizons,
        industry_level=args.industry_level,
        top_frac=args.top_frac,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    stats = generator.process()
    print(f"生成完成: {stats.files_written} 个文件, {stats.rows_written} 行")
    print(f"输出目录: {stats.output_dir}")
    print(f"Label 列: {stats.label_columns}")
    return stats


if __name__ == "__main__":
    main()
