"""股票回购 fetcher.

按公告日期 (ann_date) 分区存储。

数据目录: ``<raw_data_dir>/repurchase/year=YYYY/month=MM/YYYYMMDD.parquet``
进度文件: ``repurchase_fetch_progress.json``
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .base_daily_fetcher import (
    BaseDailyFetcher,
    FetchSummary,
    add_common_args,
    setup_logging,
)


class RepurchaseFetcher(BaseDailyFetcher):
    """股票回购数据 fetcher."""

    data_subdir = "repurchase"
    progress_key = "repurchase_completed_dates"
    progress_file_name = "repurchase_fetch_progress.json"
    date_column = "ann_date"

    def _fetch_one_date(self, trade_date: str) -> pd.DataFrame | None:
        df = self.pro.repurchase(ann_date=trade_date)
        return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="拉取 Tushare 股票回购 repurchase")
    add_common_args(parser)
    args = parser.parse_args(argv)

    logger = setup_logging("repurchase_fetcher", args.log_level)
    fetcher = RepurchaseFetcher(
        token=args.token,
        output_dir=args.output_dir,
        http_url=args.http_url,
        logger=logger,
    )
    summary: FetchSummary = fetcher.fetch(
        start_date=args.start_date,
        end_date=args.end_date,
        batch_size=args.batch_size,
        sleep_time=args.sleep_time,
        retry_wait_seconds=args.retry_wait_seconds,
        max_retries=args.max_retries,
        force_fetch=args.force_fetch,
    )
    return 1 if summary.failed_dates else 0


if __name__ == "__main__":
    raise SystemExit(main())
