"""股东增减持 fetcher.

按公告日期 (ann_date) 分区存储。

数据目录: ``<raw_data_dir>/stk_holdertrade/year=YYYY/month=MM/YYYYMMDD.parquet``
进度文件: ``stk_holdertrade_fetch_progress.json``
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


class StkHolderTradeFetcher(BaseDailyFetcher):
    """股东增减持数据 fetcher."""

    data_subdir = "stk_holdertrade"
    progress_key = "stk_holdertrade_completed_dates"
    progress_file_name = "stk_holdertrade_fetch_progress.json"
    date_column = "ann_date"

    def _fetch_one_date(self, trade_date: str) -> pd.DataFrame | None:
        df = self.pro.stk_holdertrade(ann_date=trade_date)
        return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="拉取 Tushare 股东增减持 stk_holdertrade")
    add_common_args(parser)
    args = parser.parse_args(argv)

    logger = setup_logging("stk_holdertrade_fetcher", args.log_level)
    fetcher = StkHolderTradeFetcher(
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
