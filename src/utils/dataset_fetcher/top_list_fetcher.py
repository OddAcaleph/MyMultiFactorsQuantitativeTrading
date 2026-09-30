"""龙虎榜每日明细 fetcher.

数据目录: ``<raw_data_dir>/top_list/year=YYYY/month=MM/YYYYMMDD.parquet``
进度文件: ``top_list_fetch_progress.json``
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


class TopListFetcher(BaseDailyFetcher):
    """龙虎榜每日明细数据 fetcher."""

    data_subdir = "top_list"
    progress_key = "top_list_completed_dates"
    progress_file_name = "top_list_fetch_progress.json"
    date_column = "trade_date"

    def _fetch_one_date(self, trade_date: str) -> pd.DataFrame | None:
        df = self.pro.top_list(trade_date=trade_date)
        return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="拉取 Tushare 龙虎榜每日明细 top_list")
    add_common_args(parser)
    args = parser.parse_args(argv)

    logger = setup_logging("top_list_fetcher", args.log_level)
    fetcher = TopListFetcher(
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
