"""Fetch raw A-share daily suspension/resumption data from Tushare.

The fetcher writes raw daily ``suspend_d`` snapshots into a date-partitioned
parquet layout, mirroring the suspend/resume part of
``/opt/tiger/qyd/quant_llm/scripts/preprocess/run_fetch_all_ashares.py``:

```
<output_dir>/suspend_d/year=YYYY/month=MM/YYYYMMDD.parquet
```

It fetches data by trading date, optionally filters by the full A-share
universe, and keeps a progress file so interrupted runs can resume.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


DEFAULT_RAW_DATA_DIR = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/raw_data")
DEFAULT_HTTP_URL = os.environ.get("DEFAULT_HTTP_URL") or os.environ.get("TUSHARE_HTTP_URL") or "http://jiaoch.site"
PROGRESS_FILE_NAME = "suspend_d_fetch_progress.json"


@dataclass(frozen=True)
class SuspendDFetchSummary:
    """Summary returned after a ``suspend_d`` fetch run."""

    start_date: str
    end_date: str
    total_trade_dates: int
    fetched_trade_dates: int
    skipped_trade_dates: int
    failed_trade_dates: tuple[str, ...]
    rows_written: int
    output_dir: Path


class SuspendDFetcher:
    """Fetch raw daily suspension/resumption records for all A-shares.

    Parameters
    ----------
    token
        Tushare token. If omitted, it is loaded from ``TUSHARE_TOKEN`` or
        ``~/.tushare_token``.
    output_dir
        Base raw-data directory. Daily ``suspend_d`` files are written into its
        ``suspend_d`` subdirectory. Defaults to
        ``/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/raw_data``.
    http_url
        Optional custom Tushare-compatible API endpoint, useful when using a
        proxy service.
    """

    SUSPEND_D_SUBDIR = "suspend_d"
    COMPLETED_PROGRESS_KEY = "suspend_d_completed_dates"

    def __init__(
        self,
        token: str | None = None,
        output_dir: str | Path | None = None,
        http_url: str | None = DEFAULT_HTTP_URL,
        logger: logging.Logger | None = None,
    ) -> None:
        self.token = token or self._load_token()
        self.output_dir = Path(output_dir or DEFAULT_RAW_DATA_DIR)
        self.suspend_d_dir = self.output_dir / self.SUSPEND_D_SUBDIR
        self.progress_file = self.output_dir / PROGRESS_FILE_NAME
        self.http_url = http_url
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.pro = self._init_tushare_client()

    def fetch(
        self,
        start_date: str,
        end_date: str | None = None,
        batch_size: int = 20,
        sleep_time: float = 0.2,
        retry_wait_seconds: float = 2.0,
        max_retries: int = 3,
        force_fetch: bool = False,
        filter_ashares: bool = False,
    ) -> SuspendDFetchSummary:
        """Fetch ``suspend_d`` data between ``start_date`` and ``end_date``.

        Dates must use ``YYYYMMDD`` format. ``end_date`` defaults to today.
        When ``force_fetch`` is false, dates already recorded in the progress
        file and already present on disk are skipped so the command can be
        safely resumed.
        """

        end_date = end_date or datetime.now().strftime("%Y%m%d")
        self._validate_date(start_date, "start_date")
        self._validate_date(end_date, "end_date")
        if start_date > end_date:
            raise ValueError(f"start_date must be <= end_date, got {start_date} > {end_date}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_retries <= 0:
            raise ValueError("max_retries must be positive")
        if sleep_time < 0:
            raise ValueError("sleep_time must be non-negative")
        if retry_wait_seconds < 0:
            raise ValueError("retry_wait_seconds must be non-negative")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.suspend_d_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("开始拉取 raw suspend_d：%s ~ %s", start_date, end_date)
        trade_dates = self.get_trade_dates(start_date, end_date, retry_wait_seconds=retry_wait_seconds)
        if not trade_dates:
            raise RuntimeError(f"未获取到交易日历：{start_date} ~ {end_date}")
        self.logger.info("交易日数量：%d", len(trade_dates))

        ts_codes_set: set[str] | None = None
        if filter_ashares:
            ts_codes_set = set(self.get_all_ashare_codes())
            self.logger.info("全量 A 股股票数量（含上市/退市/暂停上市）：%d", len(ts_codes_set))

        progress = self._load_progress()
        completed_dates = set(progress.get(self.COMPLETED_PROGRESS_KEY, []))
        pending_dates = (
            trade_dates
            if force_fetch
            else [d for d in trade_dates if d not in completed_dates or not self._date_output_file_exists(d)]
        )
        skipped_dates = 0 if force_fetch else len(trade_dates) - len(pending_dates)

        if not pending_dates:
            self.logger.info("suspend_d 已全部拉取完成，无需重复拉取。")
            return SuspendDFetchSummary(
                start_date=start_date,
                end_date=end_date,
                total_trade_dates=len(trade_dates),
                fetched_trade_dates=0,
                skipped_trade_dates=skipped_dates,
                failed_trade_dates=(),
                rows_written=0,
                output_dir=self.suspend_d_dir,
            )

        rows_written = 0
        fetched_dates: list[str] = []
        failed_dates: list[str] = []
        total_batches = (len(pending_dates) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            batch_dates = pending_dates[batch_idx * batch_size : (batch_idx + 1) * batch_size]
            batch_frames: list[pd.DataFrame] = []
            batch_success_dates: list[str] = []

            for trade_date in batch_dates:
                df = self._fetch_one_trade_date(
                    trade_date=trade_date,
                    ts_codes_set=ts_codes_set,
                    max_retries=max_retries,
                    sleep_time=sleep_time,
                    retry_wait_seconds=retry_wait_seconds,
                )
                if df is None:
                    failed_dates.append(trade_date)
                    continue

                if not df.empty:
                    batch_frames.append(df)
                batch_success_dates.append(trade_date)
                fetched_dates.append(trade_date)

            if batch_frames:
                combined = pd.concat(batch_frames, ignore_index=True)
                rows_written += self._save_partitioned_by_date(combined)

            if batch_success_dates:
                completed_dates.update(batch_success_dates)
                progress[self.COMPLETED_PROGRESS_KEY] = sorted(completed_dates)
                self._save_progress(progress)

            self.logger.info(
                "suspend_d 批次 %d/%d 完成：成功日期=%d，失败日期累计=%d，累计写入行数=%d",
                batch_idx + 1,
                total_batches,
                len(batch_success_dates),
                len(failed_dates),
                rows_written,
            )

        summary = SuspendDFetchSummary(
            start_date=start_date,
            end_date=end_date,
            total_trade_dates=len(trade_dates),
            fetched_trade_dates=len(fetched_dates),
            skipped_trade_dates=skipped_dates,
            failed_trade_dates=tuple(failed_dates),
            rows_written=rows_written,
            output_dir=self.suspend_d_dir,
        )
        self._log_summary(summary)
        return summary

    def get_trade_dates(self, start_date: str, end_date: str, retry_wait_seconds: float = 2.0) -> list[str]:
        """Return open SSE trading dates in ascending order."""

        for retry_idx in range(1, 4):
            try:
                df = self.pro.trade_cal(exchange="SSE", start_date=start_date, end_date=end_date, is_open="1")
                if df is None or df.empty:
                    return []
                return sorted(df["cal_date"].astype(str).tolist())
            except Exception as exc:  # pragma: no cover - depends on remote API
                self.logger.warning(
                    "获取交易日历失败（第 %d/3 次）：%s",
                    retry_idx,
                    exc,
                )
                if retry_idx < 3:
                    time.sleep(retry_wait_seconds * retry_idx)

        fallback_dates = self._calendar_dates(start_date, end_date)
        self.logger.warning(
            "交易日历接口连续失败，将退化为逐自然日尝试 suspend_d 拉取，日期数=%d。",
            len(fallback_dates),
        )
        return fallback_dates

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

    def get_all_ashare_codes(self) -> list[str]:
        """Return all A-share ts_code values, including delisted/suspended ones."""

        frames: list[pd.DataFrame] = []
        for list_status in ("L", "D", "P"):
            try:
                df = self.pro.stock_basic(list_status=list_status)
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception as exc:  # pragma: no cover - depends on remote API
                self.logger.warning("获取 list_status=%s 股票列表失败：%s", list_status, exc)
            time.sleep(0.2)

        if not frames:
            raise RuntimeError("无法获取 A 股股票列表")

        stocks = pd.concat(frames, ignore_index=True)
        return sorted(stocks["ts_code"].dropna().astype(str).unique().tolist())

    def _fetch_one_trade_date(
        self,
        trade_date: str,
        ts_codes_set: set[str] | None,
        max_retries: int,
        sleep_time: float,
        retry_wait_seconds: float,
    ) -> pd.DataFrame | None:
        for retry_idx in range(1, max_retries + 1):
            try:
                df = self.pro.suspend_d(trade_date=trade_date)
                if df is None:
                    df = pd.DataFrame()
                if not df.empty:
                    if "trade_date" not in df.columns:
                        df = df.copy()
                        df["trade_date"] = trade_date
                    else:
                        df = df.copy()
                        df["trade_date"] = df["trade_date"].astype(str)
                    if ts_codes_set is not None and "ts_code" in df.columns:
                        df = df[df["ts_code"].astype(str).isin(ts_codes_set)].copy()
                time.sleep(sleep_time)
                return df
            except Exception as exc:  # pragma: no cover - depends on remote API
                self.logger.warning(
                    "获取 suspend_d %s 失败（第 %d/%d 次）：%s",
                    trade_date,
                    retry_idx,
                    max_retries,
                    exc,
                )
                if retry_idx < max_retries:
                    time.sleep(retry_wait_seconds * retry_idx)

        self.logger.error("获取 suspend_d %s 最终失败，跳过该日期。", trade_date)
        return None

    def _save_partitioned_by_date(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        if "trade_date" not in df.columns:
            raise ValueError("suspend_d response does not contain required column: trade_date")

        rows_written = 0
        for trade_date, group in df.groupby("trade_date"):
            date_str = str(trade_date)
            year = date_str[:4]
            month = date_str[4:6]
            partition_dir = self.suspend_d_dir / f"year={year}" / f"month={month}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            file_path = partition_dir / f"{date_str}.parquet"
            group.to_parquet(file_path, compression="snappy", index=False)
            rows_written += len(group)
        return rows_written

    def _date_output_file_exists(self, trade_date: str) -> bool:
        date_str = str(trade_date)
        return (
            self.suspend_d_dir
            / f"year={date_str[:4]}"
            / f"month={date_str[4:6]}"
            / f"{date_str}.parquet"
        ).exists()

    def _init_tushare_client(self):
        try:
            import tushare as ts
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("未安装 tushare，请先执行：pip install tushare") from exc

        ts.set_token(self.token)
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

    def _load_progress(self) -> dict[str, list[str]]:
        if not self.progress_file.exists():
            return {}
        with self.progress_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save_progress(self, progress: dict[str, list[str]]) -> None:
        with self.progress_file.open("w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

    def _log_summary(self, summary: SuspendDFetchSummary) -> None:
        self.logger.info("=" * 60)
        self.logger.info("suspend_d 拉取完成")
        self.logger.info("时间范围：%s ~ %s", summary.start_date, summary.end_date)
        self.logger.info("交易日总数：%d", summary.total_trade_dates)
        self.logger.info("本次成功拉取交易日：%d", summary.fetched_trade_dates)
        self.logger.info("本次跳过交易日：%d", summary.skipped_trade_dates)
        self.logger.info("失败交易日：%s", list(summary.failed_trade_dates))
        self.logger.info("写入行数：%d", summary.rows_written)
        self.logger.info("输出目录：%s", summary.output_dir)
        self.logger.info("=" * 60)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("suspend_d_fetcher")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="拉取 Tushare 原始每日停复牌数据 suspend_d")
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
    parser.add_argument("--batch_size", type=int, default=20, help="按交易日批量保存的批次大小")
    parser.add_argument("--sleep_time", type=float, default=0.2, help="每次成功请求后的等待秒数")
    parser.add_argument(
        "--retry_wait_seconds",
        type=float,
        default=2.0,
        help="请求失败后重试的基础等待秒数；实际等待为 retry_wait_seconds * retry_idx",
    )
    parser.add_argument("--max_retries", type=int, default=3, help="单个交易日最大重试次数")
    parser.add_argument("--force_fetch", action="store_true", help="忽略进度文件，强制重新拉取并覆盖同名 parquet")
    parser.add_argument(
        "--filter_ashares",
        action="store_true",
        help="额外按全量 A 股股票列表过滤 suspend_d 返回结果；默认不启用，避免 stock_basic 临时失败影响落盘",
    )
    parser.add_argument("--no_filter_ashares", action="store_true", help="兼容旧参数：不额外按全量 A 股股票列表过滤返回结果")
    parser.add_argument("--log_level", type=str, default=os.environ.get("LOG_LEVEL", "INFO"), help="日志级别")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = setup_logging(args.log_level)
    fetcher = SuspendDFetcher(
        token=args.token,
        output_dir=args.output_dir,
        http_url=args.http_url,
        logger=logger,
    )
    summary = fetcher.fetch(
        start_date=args.start_date,
        end_date=args.end_date,
        batch_size=args.batch_size,
        sleep_time=args.sleep_time,
        retry_wait_seconds=args.retry_wait_seconds,
        max_retries=args.max_retries,
        force_fetch=args.force_fetch,
        filter_ashares=args.filter_ashares and not args.no_filter_ashares,
    )
    return 1 if summary.failed_trade_dates else 0


if __name__ == "__main__":
    raise SystemExit(main())
