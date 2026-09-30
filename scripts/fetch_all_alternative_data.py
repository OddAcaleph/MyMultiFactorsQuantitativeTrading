"""批量拉取所有新增特色数据。

依次拉取：
- top_list       龙虎榜每日明细
- top_inst        龙虎榜机构明细
- block_trade     大宗交易
- forecast        业绩预告
- repurchase      股票回购
- stk_holdertrade 股东增减持
- report_rc       卖方盈利预测
- dc_index        概念板块行情
- dc_member       概念板块成分股

用法:
    python scripts/fetch_all_alternative_data.py --start_date 20240101 --end_date 20250901
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.dataset_fetcher.base_daily_fetcher import (
    DEFAULT_HTTP_URL,
    DEFAULT_RAW_DATA_DIR,
    FetchSummary,
    setup_logging,
)
from utils.dataset_fetcher.top_list_fetcher import TopListFetcher
from utils.dataset_fetcher.top_inst_fetcher import TopInstFetcher
from utils.dataset_fetcher.block_trade_fetcher import BlockTradeFetcher
from utils.dataset_fetcher.forecast_fetcher import ForecastFetcher
from utils.dataset_fetcher.repurchase_fetcher import RepurchaseFetcher
from utils.dataset_fetcher.stk_holdertrade_fetcher import StkHolderTradeFetcher
from utils.dataset_fetcher.report_rc_fetcher import ReportRcFetcher
from utils.dataset_fetcher.dc_index_fetcher import DcIndexFetcher
from utils.dataset_fetcher.dc_member_fetcher import DcMemberFetcher


FETCHER_CLASSES = [
    ("top_list", TopListFetcher, {}),
    ("top_inst", TopInstFetcher, {}),
    ("block_trade", BlockTradeFetcher, {}),
    ("forecast", ForecastFetcher, {}),
    ("repurchase", RepurchaseFetcher, {}),
    ("stk_holdertrade", StkHolderTradeFetcher, {}),
    ("report_rc", ReportRcFetcher, {}),
    ("dc_index", DcIndexFetcher, {"idx_type": "概念板块"}),
    ("dc_member", DcMemberFetcher, {"idx_type": "概念板块", "sleep_per_concept": 0.1}),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量拉取 Tushare 特色数据")
    parser.add_argument("--token", type=str, default=None, help="Tushare Token")
    parser.add_argument("--start_date", type=str, required=True, help="开始日期 YYYYMMDD")
    parser.add_argument("--end_date", type=str, default=None, help="结束日期 YYYYMMDD")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_RAW_DATA_DIR),
        help=f"raw_data 根目录，默认 {DEFAULT_RAW_DATA_DIR}",
    )
    parser.add_argument("--http_url", type=str, default=DEFAULT_HTTP_URL, help="Tushare 代理地址")
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="只拉取指定数据集（逗号分隔），如 top_list,block_trade",
    )
    parser.add_argument("--batch_size", type=int, default=20, help="批次大小")
    parser.add_argument("--sleep_time", type=float, default=0.3, help="每次请求后等待秒数")
    parser.add_argument("--force_fetch", action="store_true", help="强制重新拉取")
    parser.add_argument("--log_level", type=str, default=os.environ.get("LOG_LEVEL", "INFO"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logging("fetch_all_alternative", args.log_level)

    only_set = set(s.strip() for s in args.only.split(",")) if args.only else None

    logger.info("=" * 70)
    logger.info("批量拉取特色数据")
    logger.info("时间范围：%s ~ %s", args.start_date, args.end_date or "今天")
    logger.info("输出目录：%s", args.output_dir)
    if only_set:
        logger.info("仅拉取：%s", ", ".join(sorted(only_set)))
    logger.info("=" * 70)

    all_summaries: list[FetchSummary] = []
    failed_names: list[str] = []

    for name, cls, extra_kwargs in FETCHER_CLASSES:
        if only_set and name not in only_set:
            logger.info("跳过 %s（不在 --only 列表中）", name)
            continue

        logger.info("")
        logger.info(">>> 开始拉取：%s", name)
        try:
            fetcher = cls(
                token=args.token,
                output_dir=args.output_dir,
                http_url=args.http_url,
                logger=logger,
                **extra_kwargs,
            )
            summary = fetcher.fetch(
                start_date=args.start_date,
                end_date=args.end_date,
                batch_size=args.batch_size,
                sleep_time=args.sleep_time,
                force_fetch=args.force_fetch,
            )
            all_summaries.append(summary)
            if summary.failed_dates:
                failed_names.append(name)
        except Exception as exc:
            logger.error("拉取 %s 异常终止：%s", name, exc, exc_info=True)
            failed_names.append(name)

        time.sleep(1)  # 不同数据之间稍作间隔

    # 汇总
    logger.info("")
    logger.info("=" * 70)
    logger.info("全部拉取完成 - 汇总")
    logger.info("=" * 70)
    for s in all_summaries:
        status = "❌" if s.failed_dates else "✅"
        logger.info(
            "  %s %-18s  成功=%d  跳过=%d  失败=%d  行数=%d",
            status,
            s.data_name,
            s.fetched_dates,
            s.skipped_dates,
            len(s.failed_dates),
            s.rows_written,
        )

    if failed_names:
        logger.info("")
        logger.warning("存在失败的数据集：%s", ", ".join(failed_names))
        return 1

    logger.info("")
    logger.info("全部数据集拉取成功！")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
