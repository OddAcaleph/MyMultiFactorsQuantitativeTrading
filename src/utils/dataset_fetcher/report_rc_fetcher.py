"""卖方盈利预测 (研报) fetcher.

按报告日期 (report_date) 分区存储。

数据目录: ``<raw_data_dir>/report_rc/year=YYYY/month=MM/YYYYMMDD.parquet``
进度文件: ``report_rc_fetch_progress.json``
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


class ReportRcFetcher(BaseDailyFetcher):
    """卖方盈利预测数据 fetcher.

    注意：report_rc 按 report_date（研报发布日期）分区，非交易日也可能有数据。
    这里仍用交易日历驱动，但非交易日也可能返回数据（周末也可能发研报）。
    """

    data_subdir = "report_rc"
    progress_key = "report_rc_completed_dates"
    progress_file_name = "report_rc_fetch_progress.json"
    date_column = "report_date"

    def _fetch_one_date(self, trade_date: str) -> pd.DataFrame | None:
        df = self.pro.report_rc(report_date=trade_date)
        return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="拉取 Tushare 卖方盈利预测 report_rc")
    add_common_args(parser)
    args = parser.parse_args(argv)

    logger = setup_logging("report_rc_fetcher", args.log_level)
    fetcher = ReportRcFetcher(
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
