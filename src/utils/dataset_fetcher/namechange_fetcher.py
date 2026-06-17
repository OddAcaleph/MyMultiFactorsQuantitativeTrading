"""Fetch raw A-share name-change/ST history data from Tushare.

The fetcher follows the ``namechange`` part of
``/opt/tiger/qyd/quant_llm/scripts/preprocess/run_fetch_all_ashares.py``:
it first gets the full A-share universe, then pulls Tushare ``namechange`` by
``ts_code`` and writes one raw parquet file per stock:

```
<output_dir>/namechange/000001.SZ.parquet
```

Tushare's ``namechange`` interface returns historical name-change records. ST
and *ST periods can be identified from these raw records by downstream cleaning
logic. Progress is saved under ``<output_dir>/namechange_fetch_progress.json``
so interrupted runs can resume safely.

After fetching, use ``namechange_merger.py`` to merge these per-stock parquet
files into one large ``namechange.parquet`` if needed.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd


DEFAULT_RAW_DATA_DIR = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/raw_data")
DEFAULT_HTTP_URL = os.environ.get("DEFAULT_HTTP_URL") or os.environ.get("TUSHARE_HTTP_URL") or "http://jiaoch.site"
PROGRESS_FILE_NAME = "namechange_fetch_progress.json"
FAILED_CODES_FILE_NAME = "namechange_failed_codes.json"


@dataclass(frozen=True)
class NamechangeFetchSummary:
    """Summary returned after a ``namechange`` fetch run."""

    start_date: str
    end_date: str
    total_codes: int
    fetched_codes: int
    skipped_codes: int
    failed_codes: tuple[str, ...]
    rows_written: int
    output_dir: Path


class NamechangeFetcher:
    """Fetch raw Tushare ``namechange`` data for all A-shares.

    Parameters
    ----------
    token
        Tushare token. If omitted, it is loaded from ``TUSHARE_TOKEN`` or
        ``~/.tushare_token``.
    output_dir
        Base raw-data directory. Name-change/ST history files are written into
        its ``namechange`` subdirectory, one parquet per stock. Defaults to
        ``/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/raw_data``.
    http_url
        Optional custom Tushare-compatible API endpoint, useful when using a
        proxy service.
    logger
        Optional logger. If omitted, a class-named logger is used.
    """

    NAMECHANGE_SUBDIR = "namechange"
    COMPLETED_PROGRESS_KEY = "namechange_completed_codes"

    def __init__(
        self,
        token: str | None = None,
        output_dir: str | Path | None = None,
        http_url: str | None = DEFAULT_HTTP_URL,
        logger: logging.Logger | None = None,
    ) -> None:
        self.token = token or self._load_token()
        self.output_dir = Path(output_dir or DEFAULT_RAW_DATA_DIR)
        self.namechange_dir = self.output_dir / self.NAMECHANGE_SUBDIR
        self.progress_file = self.output_dir / PROGRESS_FILE_NAME
        self.failed_codes_file = self.output_dir / FAILED_CODES_FILE_NAME
        self.http_url = http_url
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.pro = self._init_tushare_client()

    def fetch(
        self,
        start_date: str,
        end_date: str | None = None,
        ts_codes: list[str] | None = None,
        sleep_time: float = 0.2,
        retry_wait_seconds: float = 2.0,
        max_retries: int = 3,
        force_fetch: bool = False,
    ) -> NamechangeFetchSummary:
        """Fetch raw ``namechange`` data between ``start_date`` and ``end_date``.

        Dates must use ``YYYYMMDD`` format. ``end_date`` defaults to today.
        Tushare's ``namechange`` date range filters by name-change date.
        When ``force_fetch`` is false, stock codes already recorded in the
        progress file and already present on disk are skipped.
        """

        end_date = end_date or datetime.now().strftime("%Y%m%d")
        self._validate_date(start_date, "start_date")
        self._validate_date(end_date, "end_date")
        if start_date > end_date:
            raise ValueError(f"start_date must be <= end_date, got {start_date} > {end_date}")
        if max_retries <= 0:
            raise ValueError("max_retries must be positive")
        if sleep_time < 0:
            raise ValueError("sleep_time must be non-negative")
        if retry_wait_seconds < 0:
            raise ValueError("retry_wait_seconds must be non-negative")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.namechange_dir.mkdir(parents=True, exist_ok=True)

        codes = sorted(set(ts_codes or self.get_all_ashare_codes()))
        if not codes:
            raise RuntimeError("未获取到待拉取的 A 股股票列表")

        self.logger.info("开始拉取 raw namechange/ST 历史：%s ~ %s", start_date, end_date)
        self.logger.info("待处理 A 股股票数量：%d", len(codes))

        progress = self._load_progress()
        progress_key = f"{self.COMPLETED_PROGRESS_KEY}:{start_date}:{end_date}"
        completed_codes = set(progress.get(progress_key, []))
        pending_codes = codes if force_fetch else [code for code in codes if code not in completed_codes]
        skipped_codes = 0 if force_fetch else len(codes) - len(pending_codes)

        if not pending_codes:
            self.logger.info("namechange 已全部拉取完成，无需重复拉取。")
            self._save_failed_codes([], start_date, end_date)
            return NamechangeFetchSummary(
                start_date=start_date,
                end_date=end_date,
                total_codes=len(codes),
                fetched_codes=0,
                skipped_codes=skipped_codes,
                failed_codes=(),
                rows_written=0,
                output_dir=self.namechange_dir,
            )

        rows_written = 0
        fetched_codes: list[str] = []
        failed_codes: list[str] = []

        for index, ts_code in enumerate(pending_codes, start=1):
            df = self._fetch_one_code(
                ts_code=ts_code,
                start_date=start_date,
                end_date=end_date,
                max_retries=max_retries,
                sleep_time=sleep_time,
                retry_wait_seconds=retry_wait_seconds,
            )
            if df is None:
                failed_codes.append(ts_code)
                continue

            rows_written += self._save_one_code(ts_code, df)
            fetched_codes.append(ts_code)
            completed_codes.add(ts_code)

            if index % 50 == 0 or index == len(pending_codes):
                progress[progress_key] = sorted(completed_codes)
                self._save_progress(progress)
                self.logger.info(
                    "namechange 进度：本次 %d/%d，累计完成 %d/%d，失败累计=%d，累计写入行数=%d",
                    index,
                    len(pending_codes),
                    len(completed_codes),
                    len(codes),
                    len(failed_codes),
                    rows_written,
                )

        progress[progress_key] = sorted(completed_codes)
        self._save_progress(progress)
        self._save_failed_codes(failed_codes, start_date, end_date)

        summary = NamechangeFetchSummary(
            start_date=start_date,
            end_date=end_date,
            total_codes=len(codes),
            fetched_codes=len(fetched_codes),
            skipped_codes=skipped_codes,
            failed_codes=tuple(failed_codes),
            rows_written=rows_written,
            output_dir=self.namechange_dir,
        )
        self._log_summary(summary)
        return summary

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

    def _fetch_one_code(
        self,
        ts_code: str,
        start_date: str,
        end_date: str,
        max_retries: int,
        sleep_time: float,
        retry_wait_seconds: float,
    ) -> pd.DataFrame | None:
        for retry_idx in range(1, max_retries + 1):
            try:
                df = self.pro.namechange(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if df is None:
                    df = pd.DataFrame()
                time.sleep(sleep_time)
                return df
            except Exception as exc:  # pragma: no cover - depends on remote API
                self.logger.warning(
                    "获取 namechange/ST 历史 %s 失败（第 %d/%d 次）：%s",
                    ts_code,
                    retry_idx,
                    max_retries,
                    exc,
                )
                if retry_idx < max_retries:
                    time.sleep(retry_wait_seconds * retry_idx)

        self.logger.error("获取 namechange/ST 历史 %s 最终失败，跳过该股票。", ts_code)
        return None

    def _save_one_code(self, ts_code: str, df: pd.DataFrame) -> int:
        file_path = self.namechange_dir / f"{ts_code}.parquet"
        if df.empty:
            if not file_path.exists():
                df.to_parquet(file_path, compression="snappy", index=False)
            return 0

        rows_written = len(df)
        if file_path.exists():
            existing = pd.read_parquet(file_path)
            if not existing.empty:
                df = pd.concat([existing, df], ignore_index=True).drop_duplicates().reset_index(drop=True)

        df.to_parquet(file_path, compression="snappy", index=False)
        return rows_written

    def _code_output_file_exists(self, ts_code: str) -> bool:
        return (self.namechange_dir / f"{ts_code}.parquet").exists()

    def _save_failed_codes(self, failed_codes: list[str], start_date: str, end_date: str) -> None:
        if not failed_codes:
            if self.failed_codes_file.exists():
                self.failed_codes_file.unlink()
            return

        payload = {
            "start_date": start_date,
            "end_date": end_date,
            "failed_codes": failed_codes,
            "failed_count": len(failed_codes),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with self.failed_codes_file.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        self.logger.error("本次 namechange 拉取失败股票数：%d，失败列表：%s", len(failed_codes), failed_codes)
        self.logger.error("失败股票列表已保存：%s", self.failed_codes_file)

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

    def _load_progress(self) -> dict[str, object]:
        if not self.progress_file.exists():
            return {}
        with self.progress_file.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _save_progress(self, progress: dict[str, object]) -> None:
        with self.progress_file.open("w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)

    def _log_summary(self, summary: NamechangeFetchSummary) -> None:
        self.logger.info("=" * 60)
        self.logger.info("namechange/ST 历史拉取完成")
        self.logger.info("时间范围：%s ~ %s", summary.start_date, summary.end_date)
        self.logger.info("股票总数：%d", summary.total_codes)
        self.logger.info("本次成功拉取股票：%d", summary.fetched_codes)
        self.logger.info("本次跳过股票：%d", summary.skipped_codes)
        self.logger.info("失败股票：%s", list(summary.failed_codes))
        self.logger.info("写入行数：%d", summary.rows_written)
        self.logger.info("输出目录：%s", summary.output_dir)
        self.logger.info("=" * 60)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("namechange_fetcher")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="拉取 Tushare 原始 ST/namechange 历史数据")
    parser.add_argument("--token", type=str, default=None, help="Tushare Token；默认读取 TUSHARE_TOKEN 或 ~/.tushare_token")
    parser.add_argument("--start_date", type=str, required=True, help="开始日期，格式 YYYYMMDD；对应 namechange 变更日期范围")
    parser.add_argument("--end_date", type=str, default=None, help="结束日期，格式 YYYYMMDD；默认今天")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_RAW_DATA_DIR),
        help=f"raw_data 根目录；默认 {DEFAULT_RAW_DATA_DIR}；按股票输出到 <output_dir>/namechange/*.parquet",
    )
    parser.add_argument("--http_url", type=str, default=DEFAULT_HTTP_URL, help="可选 Tushare 兼容代理地址")
    parser.add_argument(
        "--ts_codes",
        nargs="+",
        default=None,
        help="可选股票代码列表，如 000001.SZ 600000.SH；默认拉取全量 A 股（含上市/退市/暂停上市）",
    )
    parser.add_argument("--sleep_time", type=float, default=0.2, help="每次成功请求后的等待秒数")
    parser.add_argument(
        "--retry_wait_seconds",
        type=float,
        default=2.0,
        help="请求失败后重试的基础等待秒数；实际等待为 retry_wait_seconds * retry_idx",
    )
    parser.add_argument("--max_retries", type=int, default=3, help="单只股票最大重试次数")
    parser.add_argument("--force_fetch", action="store_true", help="忽略进度文件，强制重新拉取并覆盖同名 parquet")
    parser.add_argument("--log_level", type=str, default=os.environ.get("LOG_LEVEL", "INFO"), help="日志级别")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = setup_logging(args.log_level)
    fetcher = NamechangeFetcher(
        token=args.token,
        output_dir=args.output_dir,
        http_url=args.http_url,
        logger=logger,
    )
    summary = fetcher.fetch(
        start_date=args.start_date,
        end_date=args.end_date,
        ts_codes=args.ts_codes,
        sleep_time=args.sleep_time,
        retry_wait_seconds=args.retry_wait_seconds,
        max_retries=args.max_retries,
        force_fetch=args.force_fetch,
    )
    return 1 if summary.failed_codes else 0


if __name__ == "__main__":
    raise SystemExit(main())
