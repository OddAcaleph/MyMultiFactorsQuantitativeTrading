"""概念板块行情 fetcher (东方财富 DC 概念板块).

按交易日期分区存储。

数据目录: ``<raw_data_dir>/dc_index/year=YYYY/month=MM/YYYYMMDD.parquet``
进度文件: ``dc_index_fetch_progress.json``
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


class DcIndexFetcher(BaseDailyFetcher):
    """东方财富概念/行业/地域板块行情 fetcher."""

    data_subdir = "dc_index"
    progress_key = "dc_index_completed_dates"
    progress_file_name = "dc_index_fetch_progress.json"
    date_column = "trade_date"

    def __init__(self, idx_type: str = "概念板块", **kwargs) -> None:
        super().__init__(**kwargs)
        self.idx_type = idx_type

    def _fetch_one_date(self, trade_date: str) -> pd.DataFrame | None:
        df = self.pro.dc_index(trade_date=trade_date, idx_type=self.idx_type)
        return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="拉取 Tushare 概念板块行情 dc_index")
    add_common_args(parser)
    parser.add_argument(
        "--idx_type",
        type=str,
        default="概念板块",
        help="板块类型：概念板块/行业板块/地域板块，默认概念板块",
    )
    args = parser.parse_args(argv)

    logger = setup_logging("dc_index_fetcher", args.log_level)
    fetcher = DcIndexFetcher(
        idx_type=args.idx_type,
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
