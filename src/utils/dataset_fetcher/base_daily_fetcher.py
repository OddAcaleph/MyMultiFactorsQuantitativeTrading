"""按交易日拉取 Tushare 数据的通用基类。

各特色数据 fetcher 继承本类，只需指定：
- 子目录名 (``data_subdir``)
- 进度文件 key (``progress_key``)
- 进度文件名 (``progress_file_name``)
- 单日期拉取方法 (``_fetch_one_date``)
- 日期列名 (``date_column``)，默认 trade_date

输出格式与 moneyflow_fetcher 一致：
``<raw_data_dir>/<subdir>/year=YYYY/month=MM/YYYYMMDD.parquet``
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


DEFAULT_RAW_DATA_DIR = Path(
    os.environ.get("DEFAULT_RAW_DATA_DIR")
    or "/opt/tiger/qyd/MyMultiFactorsQuantitativeTrading/data/raw_data"
)
DEFAULT_HTTP_URL = (
    os.environ.get("DEFAULT_HTTP_URL")
    or os.environ.get("TUSHARE_HTTP_URL")
    or "http://jiaoch.site"
)


@dataclass(frozen=True)
class FetchSummary:
    """数据拉取结果汇总。"""

    data_name: str
    start_date: str
    end_date: str
    total_dates: int
    fetched_dates: int
    skipped_dates: int
    failed_dates: tuple[str, ...]
    rows_written: int
    output_dir: Path


class BaseDailyFetcher(ABC):
    """按交易日拉取 Tushare 数据的基类。

    子类必须实现：
    - ``data_subdir``: 数据子目录名（如 ``top_list``）
    - ``progress_key``: 进度文件中的 key
    - ``progress_file_name``: 进度文件名
    - ``_fetch_one_date``: 拉取单个日期数据
    """

    data_subdir: str = ""
    progress_key: str = ""
    progress_file_name: str = ""
    date_column: str = "trade_date"  # 用于分区的日期列名

    def __init__(
        self,
        token: str | None = None,
        output_dir: str | Path | None = None,
        http_url: str | None = DEFAULT_HTTP_URL,
        logger: logging.Logger | None = None,
    ) -> None:
        if not self.data_subdir:
            raise ValueError("子类必须设置 data_subdir")
        if not self.progress_key:
            raise ValueError("子类必须设置 progress_key")
        if not self.progress_file_name:
            raise ValueError("子类必须设置 progress_file_name")

        self.token = token or self._load_token()
        self.output_dir = Path(output_dir or DEFAULT_RAW_DATA_DIR)
        self.data_dir = self.output_dir / self.data_subdir
        self.progress_file = self.output_dir / self.progress_file_name
        self.http_url = http_url
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.pro = self._init_tushare_client()

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------
    def fetch(
        self,
        start_date: str,
        end_date: str | None = None,
        batch_size: int = 20,
        sleep_time: float = 0.3,
        retry_wait_seconds: float = 2.0,
        max_retries: int = 3,
        force_fetch: bool = False,
    ) -> FetchSummary:
        end_date = end_date or datetime.now().strftime("%Y%m%d")
        self._validate_date(start_date, "start_date")
        self._validate_date(end_date, "end_date")
        if start_date > end_date:
            raise ValueError(f"start_date must be <= end_date, got {start_date} > {end_date}")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_retries <= 0:
            raise ValueError("max_retries must be positive")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("开始拉取 %s：%s ~ %s", self.data_subdir, start_date, end_date)
        trade_dates = self._get_trade_dates(start_date, end_date, retry_wait_seconds)
        if not trade_dates:
            self.logger.warning("未获取到交易日历：%s ~ %s", start_date, end_date)
            return FetchSummary(
                data_name=self.data_subdir,
                start_date=start_date,
                end_date=end_date,
                total_dates=0,
                fetched_dates=0,
                skipped_dates=0,
                failed_dates=(),
                rows_written=0,
                output_dir=self.data_dir,
            )
        self.logger.info("交易日数量：%d", len(trade_dates))

        progress = self._load_progress()
        completed_dates = set(progress.get(self.progress_key, []))
        pending_dates = (
            trade_dates
            if force_fetch
            else [d for d in trade_dates if d not in completed_dates or not self._date_file_exists(d)]
        )
        skipped_dates = 0 if force_fetch else len(trade_dates) - len(pending_dates)

        if not pending_dates:
            self.logger.info("%s 已全部拉取完成，无需重复拉取。", self.data_subdir)
            return FetchSummary(
                data_name=self.data_subdir,
                start_date=start_date,
                end_date=end_date,
                total_dates=len(trade_dates),
                fetched_dates=0,
                skipped_dates=skipped_dates,
                failed_dates=(),
                rows_written=0,
                output_dir=self.data_dir,
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
                df = self._fetch_one_date_with_retry(
                    trade_date=trade_date,
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
                rows_written += self._save_partitioned(combined)

            if batch_success_dates:
                completed_dates.update(batch_success_dates)
                progress[self.progress_key] = sorted(completed_dates)
                self._save_progress(progress)

            self.logger.info(
                "%s 批次 %d/%d 完成：成功=%d，失败累计=%d，累计写入行数=%d",
                self.data_subdir,
                batch_idx + 1,
                total_batches,
                len(batch_success_dates),
                len(failed_dates),
                rows_written,
            )

        summary = FetchSummary(
            data_name=self.data_subdir,
            start_date=start_date,
            end_date=end_date,
            total_dates=len(trade_dates),
            fetched_dates=len(fetched_dates),
            skipped_dates=skipped_dates,
            failed_dates=tuple(failed_dates),
            rows_written=rows_written,
            output_dir=self.data_dir,
        )
        self._log_summary(summary)
        return summary

    # ------------------------------------------------------------------
    # 子类需实现
    # ------------------------------------------------------------------
    @abstractmethod
    def _fetch_one_date(self, trade_date: str) -> pd.DataFrame | None:
        """拉取单个交易日的数据。失败抛异常，成功返回 DataFrame。"""
        ...

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------
    def _fetch_one_date_with_retry(
        self,
        trade_date: str,
        max_retries: int,
        sleep_time: float,
        retry_wait_seconds: float,
    ) -> pd.DataFrame | None:
        for retry_idx in range(1, max_retries + 1):
            try:
                df = self._fetch_one_date(trade_date)
                if df is None:
                    df = pd.DataFrame()
                time.sleep(sleep_time)
                return df
            except Exception as exc:
                self.logger.warning(
                    "获取 %s %s 失败（第 %d/%d 次）：%s",
                    self.data_subdir,
                    trade_date,
                    retry_idx,
                    max_retries,
                    exc,
                )
                if retry_idx < max_retries:
                    time.sleep(retry_wait_seconds * retry_idx)

        self.logger.error("获取 %s %s 最终失败，跳过。", self.data_subdir, trade_date)
        return None

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

        fallback = self._calendar_dates(start_date, end_date)
        self.logger.warning("交易日历接口连续失败，退化为逐自然日尝试，日期数=%d。", len(fallback))
        return fallback

    @staticmethod
    def _calendar_dates(start_date: str, end_date: str) -> list[str]:
        s = datetime.strptime(start_date, "%Y%m%d")
        e = datetime.strptime(end_date, "%Y%m%d")
        dates: list[str] = []
        cur = s
        while cur <= e:
            dates.append(cur.strftime("%Y%m%d"))
            cur += timedelta(days=1)
        return dates

    def _save_partitioned(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        if self.date_column not in df.columns:
            raise ValueError(f"{self.data_subdir} 响应缺少列: {self.date_column}")

        rows_written = 0
        for date_val, group in df.groupby(self.date_column):
            date_str = str(date_val)
            year = date_str[:4]
            month = date_str[4:6]
            partition_dir = self.data_dir / f"year={year}" / f"month={month}"
            partition_dir.mkdir(parents=True, exist_ok=True)
            file_path = partition_dir / f"{date_str}.parquet"
            group.to_parquet(file_path, compression="snappy", index=False)
            rows_written += len(group)
        return rows_written

    def _date_file_exists(self, trade_date: str) -> bool:
        date_str = str(trade_date)
        return (
            self.data_dir
            / f"year={date_str[:4]}"
            / f"month={date_str[4:6]}"
            / f"{date_str}.parquet"
        ).exists()

    def _init_tushare_client(self):
        try:
            import tushare as ts
        except ImportError as exc:
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

        raise ValueError("请通过环境变量 TUSHARE_TOKEN 或 ~/.tushare_token 提供 Tushare Token")

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

    def _log_summary(self, summary: FetchSummary) -> None:
        self.logger.info("=" * 60)
        self.logger.info("%s 拉取完成", summary.data_name)
        self.logger.info("时间范围：%s ~ %s", summary.start_date, summary.end_date)
        self.logger.info("交易日总数：%d", summary.total_dates)
        self.logger.info("本次成功拉取：%d", summary.fetched_dates)
        self.logger.info("本次跳过：%d", summary.skipped_dates)
        self.logger.info("失败日期：%s", list(summary.failed_dates))
        self.logger.info("写入行数：%d", summary.rows_written)
        self.logger.info("输出目录：%s", summary.output_dir)
        self.logger.info("=" * 60)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """为 fetcher CLI 添加通用参数。"""
    parser.add_argument("--token", type=str, default=None, help="Tushare Token")
    parser.add_argument("--start_date", type=str, required=True, help="开始日期 YYYYMMDD")
    parser.add_argument("--end_date", type=str, default=None, help="结束日期 YYYYMMDD，默认今天")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_RAW_DATA_DIR),
        help=f"raw_data 根目录，默认 {DEFAULT_RAW_DATA_DIR}",
    )
    parser.add_argument("--http_url", type=str, default=DEFAULT_HTTP_URL, help="Tushare 兼容代理地址")
    parser.add_argument("--batch_size", type=int, default=20, help="批次大小")
    parser.add_argument("--sleep_time", type=float, default=0.3, help="每次请求后等待秒数")
    parser.add_argument("--retry_wait_seconds", type=float, default=2.0, help="重试基础等待秒数")
    parser.add_argument("--max_retries", type=int, default=3, help="最大重试次数")
    parser.add_argument("--force_fetch", action="store_true", help="强制重新拉取")
    parser.add_argument("--log_level", type=str, default=os.environ.get("LOG_LEVEL", "INFO"), help="日志级别")


def setup_logging(name: str, log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(name)
