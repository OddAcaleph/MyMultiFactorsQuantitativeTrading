"""Fetch broad-market index daily OHLCV data from Tushare.

Fetches index daily bars (open/high/low/close/volume/amount etc.) for a
configurable set of broad-market indices via ``pro.index_daily`` and writes
them as date-partitioned parquet files, one directory per index:

```
<output_dir>/index_daily/<index_code>/year=YYYY/month=MM/YYYYMMDD.parquet
```

Each daily parquet file contains a single row for that index on that trading
day, with columns:

- trade_date
- ts_code       (index code, e.g. 000300.SH)
- close
- open
- high
- low
- pre_close
- change
- pct_chg
- vol
- amount

Progress is saved under ``<output_dir>/index_daily_fetch_progress.json``
so an interrupted run can be resumed safely.

Existing data under other raw_data subdirectories is never touched.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


DEFAULT_RAW_DATA_DIR = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/raw_data")
DEFAULT_HTTP_URL = os.environ.get("DEFAULT_HTTP_URL") or os.environ.get("TUSHARE_HTTP_URL") or "http://jiaoch.site"
PROGRESS_FILE_NAME = "index_daily_fetch_progress.json"
DIR_NAME = "index_daily"

# Default broad-market indices to fetch.
DEFAULT_INDICES: dict[str, str] = {
    "000016.SH": "上证50",
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000906.SH": "中证800",
    "000852.SH": "中证1000",
    "000010.SH": "上证180",
    "399006.SZ": "创业板指",
    "000688.SH": "科创50",
    "399001.SZ": "深证成指",
    "000001.SH": "上证指数",
}


@dataclass
class IndexDailyFetchSummary:
    start_date: str = ""
    end_date: str = ""
    total_trade_dates: int = 0
    fetched_trade_dates: int = 0
    skipped_trade_dates: int = 0
    failed_trade_dates: list[str] = field(default_factory=list)
    rows_written: int = 0
    output_dir: str = ""
    indices_fetched: dict[str, int] = field(default_factory=dict)


class IndexDailyFetcher:
    """Fetch broad-market index daily OHLCV data from Tushare."""

    def __init__(
        self,
        token: str | None = None,
        output_dir: str | Path = DEFAULT_RAW_DATA_DIR,
        http_url: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.token = token or self._load_token()
        self.output_dir = Path(output_dir)
        self.base_dir = self.output_dir / DIR_NAME
        self.progress_file = self.output_dir / PROGRESS_FILE_NAME
        self.http_url = http_url or DEFAULT_HTTP_URL
        self.logger = logger or logging.getLogger("index_daily_fetcher")
        self.pro = self._init_tushare_client()

    def fetch(
        self,
        start_date: str,
        end_date: str,
        indices: dict[str, str] | None = None,
        sleep_time: float = 0.2,
        retry_wait_seconds: float = 2.0,
        max_retries: int = 3,
        force_fetch: bool = False,
    ) -> IndexDailyFetchSummary:
        """Fetch and write daily index OHLCV data.

        Parameters
        ----------
        start_date, end_date : str
            Date range in YYYYMMDD format.
        indices : dict[str, str] | None
            Mapping of index_code -> index_name.  Defaults to
            :data:`DEFAULT_INDICES`.
        sleep_time : float
            Seconds to sleep after each successful API call.
        retry_wait_seconds : float
            Base wait for retries; multiplied by retry index.
        max_retries : int
            Maximum retries per API call.
        force_fetch : bool
            If True, overwrite existing daily parquet files.
        """
        self._validate_date(start_date, "start_date")
        self._validate_date(end_date, "end_date")

        if indices is None:
            indices = DEFAULT_INDICES

        summary = IndexDailyFetchSummary(
            start_date=start_date,
            end_date=end_date,
            output_dir=str(self.base_dir),
        )

        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: fetch index_daily for each index (full date range per index)
        all_data = self._fetch_all_index_daily(
            indices=indices,
            start_date=start_date,
            end_date=end_date,
            sleep_time=sleep_time,
            retry_wait_seconds=retry_wait_seconds,
            max_retries=max_retries,
        )
        if all_data.empty:
            self.logger.error("未获取到任何指数日线数据，终止。")
            return summary

        for idx_code in sorted(indices.keys()):
            sub = all_data[all_data["ts_code"] == idx_code]
            if not sub.empty:
                summary.indices_fetched[idx_code] = len(sub)

        # Step 2: get trade calendar (for progress reporting)
        trade_dates = self._get_trade_dates(start_date, end_date, retry_wait_seconds)
        summary.total_trade_dates = len(trade_dates)
        self.logger.info("交易日总数：%d（%s ~ %s）", len(trade_dates), start_date, end_date)

        # Step 3: write daily parquets per index
        progress = self._load_progress()
        completed = set(progress.get("completed_dates", []))

        for trade_date in trade_dates:
            if not force_fetch and trade_date in completed and self._all_indices_files_exist(trade_date, all_data):
                summary.skipped_trade_dates += 1
                continue

            try:
                rows = self._write_daily_snapshots(
                    all_data=all_data,
                    trade_date=trade_date,
                    force_fetch=force_fetch,
                )
                if rows == 0:
                    completed.add(trade_date)
                    progress["completed_dates"] = sorted(completed)
                    self._save_progress(progress)
                    summary.skipped_trade_dates += 1
                    continue

                summary.rows_written += rows
                summary.fetched_trade_dates += 1
                completed.add(trade_date)
                progress["completed_dates"] = sorted(completed)
                self._save_progress(progress)

                if summary.fetched_trade_dates % 200 == 0:
                    self.logger.info(
                        "进度：已处理 %d / %d 个交易日（本次新增 %d，跳过 %d）",
                        summary.fetched_trade_dates + summary.skipped_trade_dates,
                        summary.total_trade_dates,
                        summary.fetched_trade_dates,
                        summary.skipped_trade_dates,
                    )
            except Exception as exc:
                self.logger.error("写入 %s 数据失败：%s", trade_date, exc, exc_info=True)
                summary.failed_trade_dates.append(trade_date)

        self._log_summary(summary)
        return summary

    def _fetch_all_index_daily(
        self,
        indices: dict[str, str],
        start_date: str,
        end_date: str,
        sleep_time: float,
        retry_wait_seconds: float,
        max_retries: int,
    ) -> pd.DataFrame:
        """Fetch index_daily for all requested indices."""
        frames: list[pd.DataFrame] = []

        for idx_code, idx_name in indices.items():
            self.logger.info("获取指数日线：%s (%s)", idx_code, idx_name)
            df = self._fetch_one_index_daily(
                ts_code=idx_code,
                start_date=start_date,
                end_date=end_date,
                sleep_time=sleep_time,
                retry_wait_seconds=retry_wait_seconds,
                max_retries=max_retries,
            )
            if df is not None and not df.empty:
                frames.append(df)
                self.logger.info("  %s: %d 条记录", idx_name, len(df))
            else:
                self.logger.warning("  %s: 未获取到数据，跳过。", idx_name)

        if not frames:
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        result["trade_date"] = result["trade_date"].astype(str).str.replace("-", "", regex=False)
        result["ts_code"] = result["ts_code"].astype(str)
        result = result.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
        self.logger.info("合计获取指数日线记录数：%d", len(result))
        return result

    def _fetch_one_index_daily(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        sleep_time: float,
        retry_wait_seconds: float,
        max_retries: int,
    ) -> pd.DataFrame | None:
        for retry_idx in range(1, max_retries + 1):
            try:
                df = self.pro.index_daily(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                )
                if df is None:
                    df = pd.DataFrame()
                time.sleep(sleep_time)
                return df
            except Exception as exc:
                self.logger.warning(
                    "获取 index_daily %s [%s~%s] 失败（第 %d/%d 次）：%s",
                    ts_code, start_date, end_date, retry_idx, max_retries, exc,
                )
                if retry_idx < max_retries:
                    time.sleep(retry_wait_seconds * retry_idx)

        self.logger.warning(
            "获取 index_daily %s [%s~%s] 最终失败。",
            ts_code, start_date, end_date,
        )
        return None

    def _write_daily_snapshots(
        self,
        all_data: pd.DataFrame,
        trade_date: str,
        force_fetch: bool,
    ) -> int:
        """Write one parquet per index for this date.  Returns total rows written."""
        day_data = all_data[all_data["trade_date"] == trade_date]
        if day_data.empty:
            return 0

        total_rows = 0
        for idx_code, group in day_data.groupby("ts_code"):
            out_path = self._index_date_path(idx_code, trade_date)
            if not force_fetch and out_path.exists():
                continue

            df = group.copy().reset_index(drop=True)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(out_path, compression="snappy", index=False)
            total_rows += len(df)

        return total_rows

    def _index_date_path(self, index_code: str, trade_date: str) -> Path:
        return (
            self.base_dir
            / index_code
            / f"year={trade_date[:4]}"
            / f"month={trade_date[4:6]}"
            / f"{trade_date}.parquet"
        )

    def _all_indices_files_exist(self, trade_date: str, all_data: pd.DataFrame) -> bool:
        day_data = all_data[all_data["trade_date"] == trade_date]
        if day_data.empty:
            return True
        for idx_code in day_data["ts_code"].unique():
            if not self._index_date_path(idx_code, trade_date).exists():
                return False
        return True

    def _get_trade_dates(self, start_date: str, end_date: str, retry_wait_seconds: float) -> list[str]:
        for retry_idx in range(1, 4):
            try:
                df = self.pro.trade_cal(exchange="SSE", start_date=start_date, end_date=end_date, is_open="1")
                if df is None or df.empty:
                    return []
                return sorted(df["cal_date"].astype(str).tolist())
            except Exception as exc:
                self.logger.warning("获取交易日历失败（第 %d/3 次）：%s", retry_idx, exc)
                if retry_idx < 3:
                    time.sleep(retry_wait_seconds * retry_idx)

        self.logger.warning("交易日历接口连续失败，退化为逐自然日生成。")
        return self._calendar_dates(start_date, end_date)

    @staticmethod
    def _calendar_dates(start_date: str, end_date: str) -> list[str]:
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        dates: list[str] = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y%m%d"))
            current += timedelta(days=1)
        return dates

    def _init_tushare_client(self):
        try:
            import tushare as ts
        except ImportError as exc:
            raise ImportError("未安装 tushare，请先执行：pip install tushare") from exc

        # Skip ts.set_token() which writes to ~/tk.csv (may be read-only).
        pro = ts.pro_api(self.token)
        if self.http_url:
            pro._DataApi__http_url = self.http_url
        pro._DataApi__token = self.token
        return pro

    @staticmethod
    def _load_token() -> str:
        token = os.environ.get("TUSHARE_TOKEN")
        if token:
            return token.strip()

        token_file = Path.home() / ".tushare_token"
        if token_file.exists():
            token = token_file.read_text(encoding="utf-8").strip()
            if token:
                return token

        raise ValueError("请通过 --token、环境变量 TUSHARE_TOKEN 或 ~/.tushare_token 提供 Tushare Token")

    @staticmethod
    def _validate_date(value: str, name: str) -> None:
        try:
            datetime.strptime(value, "%Y%m%d")
        except ValueError as exc:
            raise ValueError(f"{name} must use YYYYMMDD format, got {value!r}") from exc

    def _load_progress(self) -> dict:
        if not self.progress_file.exists():
            return {"completed_dates": []}
        with self.progress_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save_progress(self, progress: dict) -> None:
        with self.progress_file.open("w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

    def _log_summary(self, summary: IndexDailyFetchSummary) -> None:
        self.logger.info("=" * 60)
        self.logger.info("指数日线数据拉取完成")
        self.logger.info("时间范围：%s ~ %s", summary.start_date, summary.end_date)
        self.logger.info("交易日总数：%d", summary.total_trade_dates)
        self.logger.info("本次成功写入交易日：%d", summary.fetched_trade_dates)
        self.logger.info("本次跳过交易日：%d", summary.skipped_trade_dates)
        self.logger.info("失败交易日：%s", list(summary.failed_trade_dates))
        self.logger.info("写入行数：%d", summary.rows_written)
        self.logger.info("输出目录：%s", summary.output_dir)
        if summary.indices_fetched:
            self.logger.info("各指数记录数：")
            for idx_code, count in sorted(summary.indices_fetched.items()):
                self.logger.info("  %s: %d 条", idx_code, count)
        self.logger.info("=" * 60)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("index_daily_fetcher")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="拉取 Tushare 宽基指数日线行情（点数）")
    parser.add_argument("--token", type=str, default=None, help="Tushare Token；默认读取 TUSHARE_TOKEN 或 ~/.tushare_token")
    parser.add_argument("--start_date", type=str, required=True, help="开始日期，格式 YYYYMMDD")
    parser.add_argument("--end_date", type=str, default=None, help="结束日期，格式 YYYYMMDD；默认今天")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_RAW_DATA_DIR),
        help=f"raw_data 根目录；默认 {DEFAULT_RAW_DATA_DIR}",
    )
    parser.add_argument("--http_url", type=str, default=DEFAULT_HTTP_URL, help="可选 Tushare 兼容代理地址")
    parser.add_argument(
        "--indices",
        type=str,
        default="",
        help="逗号分隔的指数代码列表，如 000300.SH,000905.SH；默认拉取全部宽基指数",
    )
    parser.add_argument("--sleep_time", type=float, default=0.2, help="每次成功请求后的等待秒数")
    parser.add_argument(
        "--retry_wait_seconds",
        type=float,
        default=2.0,
        help="请求失败后重试的基础等待秒数；实际等待为 retry_wait_seconds * retry_idx",
    )
    parser.add_argument("--max_retries", type=int, default=3, help="单个接口最大重试次数")
    parser.add_argument("--force_fetch", action="store_true", help="忽略进度文件，强制重新拉取并覆盖同名 parquet")
    parser.add_argument("--log_level", type=str, default=os.environ.get("LOG_LEVEL", "INFO"), help="日志级别")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = setup_logging(args.log_level)

    indices: dict[str, str] | None = None
    if args.indices:
        codes = [c.strip() for c in args.indices.split(",") if c.strip()]
        indices = {c: DEFAULT_INDICES.get(c, c) for c in codes}

    if args.end_date is None:
        args.end_date = datetime.now().strftime("%Y%m%d")

    fetcher = IndexDailyFetcher(
        token=args.token,
        output_dir=args.output_dir,
        http_url=args.http_url,
        logger=logger,
    )
    summary = fetcher.fetch(
        start_date=args.start_date,
        end_date=args.end_date,
        indices=indices,
        sleep_time=args.sleep_time,
        retry_wait_seconds=args.retry_wait_seconds,
        max_retries=args.max_retries,
        force_fetch=args.force_fetch,
    )
    return 1 if summary.failed_trade_dates else 0


if __name__ == "__main__":
    raise SystemExit(main())
