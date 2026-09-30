"""概念板块成分股 fetcher (东方财富 DC 板块成分).

按交易日期分区存储。每个交易日保存全部板块的成分股数据。

数据目录: ``<raw_data_dir>/dc_member/year=YYYY/month=MM/YYYYMMDD.parquet``
进度文件: ``dc_member_fetch_progress.json``
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from .base_daily_fetcher import (
    BaseDailyFetcher,
    FetchSummary,
    add_common_args,
    setup_logging,
)


class DcMemberFetcher(BaseDailyFetcher):
    """东方财富板块成分股 fetcher.

    每个交易日先获取当日所有板块列表 (dc_index)，再逐个板块拉取成分股。
    注意：数据最早从 2024-12-20 开始。
    """

    data_subdir = "dc_member"
    progress_key = "dc_member_completed_dates"
    progress_file_name = "dc_member_fetch_progress.json"
    date_column = "trade_date"

    def __init__(self, idx_type: str = "概念板块", sleep_per_concept: float = 0.15, **kwargs) -> None:
        super().__init__(**kwargs)
        self.idx_type = idx_type
        self.sleep_per_concept = sleep_per_concept

    def _fetch_one_date(self, trade_date: str) -> pd.DataFrame | None:
        # 先获取当日板块列表
        idx_df = self.pro.dc_index(trade_date=trade_date, idx_type=self.idx_type)
        if idx_df is None or idx_df.empty:
            return pd.DataFrame()

        all_frames: list[pd.DataFrame] = []
        for _, row in idx_df.iterrows():
            ts_code = row["ts_code"]
            try:
                member_df = self.pro.dc_member(ts_code=ts_code, trade_date=trade_date)
                if member_df is not None and not member_df.empty:
                    all_frames.append(member_df)
            except Exception as exc:
                self.logger.warning("获取 %s 成分股失败：%s", ts_code, exc)
            time.sleep(self.sleep_per_concept)

        if not all_frames:
            return pd.DataFrame()
        return pd.concat(all_frames, ignore_index=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="拉取 Tushare 概念板块成分股 dc_member")
    add_common_args(parser)
    parser.add_argument(
        "--idx_type",
        type=str,
        default="概念板块",
        help="板块类型：概念板块/行业板块/地域板块，默认概念板块",
    )
    parser.add_argument(
        "--sleep_per_concept",
        type=float,
        default=0.15,
        help="每个板块请求之间的等待秒数，默认 0.15",
    )
    args = parser.parse_args(argv)

    logger = setup_logging("dc_member_fetcher", args.log_level)
    fetcher = DcMemberFetcher(
        idx_type=args.idx_type,
        sleep_per_concept=args.sleep_per_concept,
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
