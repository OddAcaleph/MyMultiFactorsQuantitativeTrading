"""Fetch raw A-share fundamentals data from Tushare.

The fetcher follows the fundamentals part of
``/opt/tiger/qyd/quant_llm/scripts/preprocess/run_fetch_all_ashares.py``:
it first gets the full A-share universe, then pulls ``fina_indicator`` by
``ts_code`` and writes one raw parquet file per stock:

```
<output_dir>/fundamentals/000001.SZ.parquet
```

``FundamentalsMerger`` can then combine those per-stock files into one large
parquet table for downstream preprocessing:

```
<output_dir>/fundamentals/*.parquet  ->  <output_dir>/fundamentals/fundamentals.parquet
```

Progress is saved under ``<output_dir>/fundamentals_fetch_progress.json`` so an
interrupted run can be resumed safely.
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
DEFAULT_FUNDAMENTALS_DIR = DEFAULT_RAW_DATA_DIR / "fundamentals"
DEFAULT_MERGED_OUTPUT_FILE_NAME = "fundamentals.parquet"
DEFAULT_HTTP_URL = os.environ.get("DEFAULT_HTTP_URL") or os.environ.get("TUSHARE_HTTP_URL") or "http://jiaoch.site"
PROGRESS_FILE_NAME = "fundamentals_fetch_progress.json"


@dataclass(frozen=True)
class FundamentalsFetchSummary:
    """Summary returned after a fundamentals fetch run."""

    start_date: str
    end_date: str
    total_codes: int
    fetched_codes: int
    skipped_codes: int
    failed_codes: tuple[str, ...]
    rows_written: int
    output_dir: Path


@dataclass(frozen=True)
class FundamentalsMergeSummary:
    """Summary returned after merging per-stock fundamentals parquet files."""

    input_dir: Path
    output_file: Path
    total_files: int
    merged_files: int
    failed_files: tuple[str, ...]
    rows_written: int


class FundamentalsFetcher:
    """Fetch raw Tushare ``fina_indicator`` data for all A-shares.

    Parameters
    ----------
    token
        Tushare token. If omitted, it is loaded from ``TUSHARE_TOKEN`` or
        ``~/.tushare_token``.
    output_dir
        Base raw-data directory. Fundamentals are written into its
        ``fundamentals`` subdirectory. Defaults to
        ``/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/raw_data``.
    http_url
        Optional custom Tushare-compatible API endpoint, useful when using a
        proxy service.
    """

    FUNDAMENTALS_SUBDIR = "fundamentals"
    COMPLETED_PROGRESS_KEY = "financial_completed_codes"
    CODE_WATERMARKS_KEY = "financial_code_watermarks"
    DATE_WATERMARK_KEY = "financial_date_watermark"
    DATE_COLUMN = "ann_date"

    def __init__(
        self,
        token: str | None = None,
        output_dir: str | Path | None = None,
        http_url: str | None = DEFAULT_HTTP_URL,
        logger: logging.Logger | None = None,
    ) -> None:
        self.token = token or self._load_token()
        self.output_dir = Path(output_dir or DEFAULT_RAW_DATA_DIR)
        self.fundamentals_dir = self.output_dir / self.FUNDAMENTALS_SUBDIR
        self.progress_file = self.output_dir / PROGRESS_FILE_NAME
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
    ) -> FundamentalsFetchSummary:
        """Fetch raw fundamentals between ``start_date`` and ``end_date``.

        Dates must use ``YYYYMMDD`` format. ``end_date`` defaults to today.
        Tushare's ``fina_indicator`` date range filters by announcement date.
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
        self.fundamentals_dir.mkdir(parents=True, exist_ok=True)

        codes = sorted(set(ts_codes or self.get_all_ashare_codes()))
        if not codes:
            raise RuntimeError("未获取到待拉取的 A 股股票列表")

        self.logger.info("开始拉取 raw fundamentals/fina_indicator：%s ~ %s", start_date, end_date)
        self.logger.info("待处理 A 股股票数量：%d", len(codes))

        progress = self._load_progress()

        if not force_fetch and ts_codes is None and self._has_existing_history():
            market_start_date = self._resolve_market_incremental_start_date(progress, start_date)
            if market_start_date > end_date:
                self.logger.info(
                    "fundamentals 历史数据已覆盖到 %s，无需拉取 %s ~ %s。",
                    self._previous_date(market_start_date),
                    start_date,
                    end_date,
                )
                return FundamentalsFetchSummary(
                    start_date=start_date,
                    end_date=end_date,
                    total_codes=len(codes),
                    fetched_codes=0,
                    skipped_codes=len(codes),
                    failed_codes=(),
                    rows_written=0,
                    output_dir=self.fundamentals_dir,
                )

            market_summary = self._try_fetch_market_incremental(
                requested_start_date=start_date,
                effective_start_date=market_start_date,
                end_date=end_date,
                total_codes=len(codes),
                sleep_time=sleep_time,
                retry_wait_seconds=retry_wait_seconds,
                max_retries=max_retries,
                progress=progress,
            )
            if market_summary is not None:
                return market_summary

        code_watermarks = self._progress_mapping(progress, self.CODE_WATERMARKS_KEY)
        pending_codes: list[tuple[str, str]] = []
        skipped_codes = 0
        if force_fetch:
            pending_codes = [(code, start_date) for code in codes]
        else:
            for code in codes:
                effective_start_date = self._resolve_code_incremental_start_date(code, code_watermarks, start_date)
                if effective_start_date > end_date:
                    skipped_codes += 1
                    continue
                pending_codes.append((code, effective_start_date))

        if not pending_codes:
            self.logger.info("fundamentals 已全部拉取完成，无需重复拉取。")
            return FundamentalsFetchSummary(
                start_date=start_date,
                end_date=end_date,
                total_codes=len(codes),
                fetched_codes=0,
                skipped_codes=skipped_codes,
                failed_codes=(),
                rows_written=0,
                output_dir=self.fundamentals_dir,
            )

        rows_written = 0
        fetched_codes: list[str] = []
        failed_codes: list[str] = []

        for index, (ts_code, effective_start_date) in enumerate(pending_codes, start=1):
            df = self._fetch_one_code(
                ts_code=ts_code,
                start_date=effective_start_date,
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
            code_watermarks[ts_code] = end_date

            if index % 50 == 0 or index == len(pending_codes):
                progress[self.CODE_WATERMARKS_KEY] = dict(sorted(code_watermarks.items()))
                self._save_progress(progress)
                self.logger.info(
                    "fundamentals 进度：本次 %d/%d，累计完成 %d/%d，失败累计=%d，累计写入行数=%d",
                    index,
                    len(pending_codes),
                    len(codes) - skipped_codes - len(failed_codes),
                    len(codes),
                    len(failed_codes),
                    rows_written,
                )

        summary = FundamentalsFetchSummary(
            start_date=start_date,
            end_date=end_date,
            total_codes=len(codes),
            fetched_codes=len(fetched_codes),
            skipped_codes=skipped_codes,
            failed_codes=tuple(failed_codes),
            rows_written=rows_written,
            output_dir=self.fundamentals_dir,
        )
        self._log_summary(summary)
        return summary

    def _try_fetch_market_incremental(
        self,
        requested_start_date: str,
        effective_start_date: str,
        end_date: str,
        total_codes: int,
        sleep_time: float,
        retry_wait_seconds: float,
        max_retries: int,
        progress: dict[str, object],
    ) -> FundamentalsFetchSummary | None:
        """Fetch recent fundamentals for all stocks in one date-range request.

        For daily updates this avoids querying every stock individually.  If the
        Tushare-compatible endpoint does not support market-wide
        ``fina_indicator`` queries for the requested range, return ``None`` so
        the caller can fall back to per-code incremental fetching.
        """

        self.logger.info(
            "检测到已有 fundamentals 历史数据，尝试按公告日期增量拉取全市场：%s ~ %s",
            effective_start_date,
            end_date,
        )
        df = self._fetch_market_range(
            start_date=effective_start_date,
            end_date=end_date,
            sleep_time=sleep_time,
            retry_wait_seconds=retry_wait_seconds,
            max_retries=max_retries,
        )
        if df is None:
            self.logger.warning("全市场 fundamentals 增量拉取不可用，将回退到逐股票增量拉取。")
            return None

        try:
            rows_written = self._save_market_frame_by_code(df)
        except ValueError as exc:
            self.logger.warning("全市场 fundamentals 增量结果不可用：%s，将回退到逐股票增量拉取。", exc)
            return None
        progress[self.DATE_WATERMARK_KEY] = end_date
        self._save_progress(progress)
        fetched_codes = tuple(sorted(df["ts_code"].dropna().astype(str).unique().tolist())) if not df.empty and "ts_code" in df.columns else ()
        summary = FundamentalsFetchSummary(
            start_date=requested_start_date,
            end_date=end_date,
            total_codes=total_codes,
            fetched_codes=len(fetched_codes),
            skipped_codes=total_codes - len(fetched_codes),
            failed_codes=(),
            rows_written=rows_written,
            output_dir=self.fundamentals_dir,
        )
        self._log_summary(summary)
        return summary

    def _fetch_market_range(
        self,
        start_date: str,
        end_date: str,
        sleep_time: float,
        retry_wait_seconds: float,
        max_retries: int,
    ) -> pd.DataFrame | None:
        for retry_idx in range(1, max_retries + 1):
            try:
                df = self.pro.fina_indicator(start_date=start_date, end_date=end_date)
                if df is None:
                    df = pd.DataFrame()
                time.sleep(sleep_time)
                return df
            except Exception as exc:  # pragma: no cover - depends on remote API
                self.logger.warning(
                    "全市场 fundamentals/fina_indicator %s~%s 拉取失败（第 %d/%d 次）：%s",
                    start_date,
                    end_date,
                    retry_idx,
                    max_retries,
                    exc,
                )
                if retry_idx < max_retries:
                    time.sleep(retry_wait_seconds * retry_idx)
        return None

    def _save_market_frame_by_code(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        if "ts_code" not in df.columns:
            raise ValueError("fina_indicator market response does not contain required column: ts_code")
        rows_written = 0
        for ts_code, group in df.groupby("ts_code"):
            rows_written += self._save_one_code(str(ts_code), group.reset_index(drop=True))
        return rows_written

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
                df = self.pro.fina_indicator(ts_code=ts_code, start_date=start_date, end_date=end_date)
                if df is None:
                    df = pd.DataFrame()
                time.sleep(sleep_time)
                return df
            except Exception as exc:  # pragma: no cover - depends on remote API
                self.logger.warning(
                    "获取 fundamentals/fina_indicator %s 失败（第 %d/%d 次）：%s",
                    ts_code,
                    retry_idx,
                    max_retries,
                    exc,
                )
                if retry_idx < max_retries:
                    time.sleep(retry_wait_seconds * retry_idx)

        self.logger.error("获取 fundamentals/fina_indicator %s 最终失败，跳过该股票。", ts_code)
        return None

    def _save_one_code(self, ts_code: str, df: pd.DataFrame) -> int:
        file_path = self.fundamentals_dir / f"{ts_code}.parquet"
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

    def _has_existing_history(self) -> bool:
        merged_file = self.fundamentals_dir / DEFAULT_MERGED_OUTPUT_FILE_NAME
        if merged_file.exists():
            return True
        return any(
            file_path.is_file() and file_path.name != DEFAULT_MERGED_OUTPUT_FILE_NAME
            for file_path in self.fundamentals_dir.glob("*.parquet")
        )

    def _resolve_market_incremental_start_date(self, progress: dict[str, object], requested_start_date: str) -> str:
        watermark = progress.get(self.DATE_WATERMARK_KEY)
        max_existing_date = str(watermark) if self._is_valid_yyyymmdd(watermark) else self._max_existing_date()
        if not max_existing_date:
            return requested_start_date
        return max(requested_start_date, self._next_date(max_existing_date))

    def _resolve_code_incremental_start_date(
        self,
        ts_code: str,
        code_watermarks: dict[str, str],
        requested_start_date: str,
    ) -> str:
        max_existing_date = code_watermarks.get(ts_code) or self._max_existing_code_date(ts_code)
        if not max_existing_date:
            return requested_start_date
        return max(requested_start_date, self._next_date(max_existing_date))

    def _max_existing_date(self) -> str | None:
        merged_file = self.fundamentals_dir / DEFAULT_MERGED_OUTPUT_FILE_NAME
        candidates: list[str] = []
        if merged_file.exists():
            max_date = self._max_date_in_file(merged_file, self.DATE_COLUMN)
            if max_date:
                candidates.append(max_date)
        if not candidates:
            for file_path in self.fundamentals_dir.glob("*.parquet"):
                if file_path.name == DEFAULT_MERGED_OUTPUT_FILE_NAME:
                    continue
                max_date = self._max_date_in_file(file_path, self.DATE_COLUMN)
                if max_date:
                    candidates.append(max_date)
        return max(candidates) if candidates else None

    def _max_existing_code_date(self, ts_code: str) -> str | None:
        return self._max_date_in_file(self.fundamentals_dir / f"{ts_code}.parquet", self.DATE_COLUMN)

    @staticmethod
    def _max_date_in_file(file_path: Path, date_column: str) -> str | None:
        if not file_path.exists():
            return None
        try:
            df = pd.read_parquet(file_path, columns=[date_column])
        except Exception:
            return None
        if df.empty or date_column not in df.columns:
            return None
        dates = df[date_column].dropna().astype(str).str.replace("-", "", regex=False)
        dates = dates[dates.str.fullmatch(r"\d{8}")]
        if dates.empty:
            return None
        return str(dates.max())

    @staticmethod
    def _progress_mapping(progress: dict[str, object], key: str) -> dict[str, str]:
        value = progress.get(key, {})
        if not isinstance(value, dict):
            return {}
        return {str(k): str(v) for k, v in value.items() if FundamentalsFetcher._is_valid_yyyymmdd(v)}

    @staticmethod
    def _next_date(value: str) -> str:
        return (datetime.strptime(value, "%Y%m%d") + timedelta(days=1)).strftime("%Y%m%d")

    @staticmethod
    def _previous_date(value: str) -> str:
        return (datetime.strptime(value, "%Y%m%d") - timedelta(days=1)).strftime("%Y%m%d")

    @staticmethod
    def _is_valid_yyyymmdd(value: object) -> bool:
        if not isinstance(value, str):
            return False
        try:
            datetime.strptime(value, "%Y%m%d")
            return True
        except ValueError:
            return False

    def _code_output_file_exists(self, ts_code: str) -> bool:
        return (self.fundamentals_dir / f"{ts_code}.parquet").exists()

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

    def _log_summary(self, summary: FundamentalsFetchSummary) -> None:
        self.logger.info("=" * 60)
        self.logger.info("fundamentals 拉取完成")
        self.logger.info("时间范围：%s ~ %s", summary.start_date, summary.end_date)
        self.logger.info("股票总数：%d", summary.total_codes)
        self.logger.info("本次成功拉取股票：%d", summary.fetched_codes)
        self.logger.info("本次跳过股票：%d", summary.skipped_codes)
        self.logger.info("失败股票：%s", list(summary.failed_codes))
        self.logger.info("写入行数：%d", summary.rows_written)
        self.logger.info("输出目录：%s", summary.output_dir)
        self.logger.info("=" * 60)


class FundamentalsMerger:
    """Merge per-stock raw ``fina_indicator`` parquet files into one parquet.

    Parameters
    ----------
    input_dir
        Directory containing per-stock parquet files such as
        ``000001.SZ.parquet``. Defaults to
        ``/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/raw_data/fundamentals``.
    output_file
        Target merged parquet. Defaults to ``<input_dir>/fundamentals.parquet``.
        The output file is excluded from source files automatically, so the
        command can be safely rerun.
    logger
        Optional logger. If omitted, a class-named logger is used.
    """

    def __init__(
        self,
        input_dir: str | Path | None = None,
        output_file: str | Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.input_dir = Path(input_dir or DEFAULT_FUNDAMENTALS_DIR)
        self.output_file = Path(output_file) if output_file else self.input_dir / DEFAULT_MERGED_OUTPUT_FILE_NAME
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def merge(
        self,
        pattern: str = "*.parquet",
        drop_duplicates: bool = True,
        sort_by: list[str] | None = None,
        add_ts_code_from_filename: bool = True,
    ) -> FundamentalsMergeSummary:
        """Merge per-stock fundamentals parquet files.

        Parameters
        ----------
        pattern
            Glob pattern used inside ``input_dir``. The default reads all
            parquet files and excludes ``output_file`` automatically.
        drop_duplicates
            Whether to drop fully duplicated rows before writing output.
        sort_by
            Optional column names used for stable sorting when present. If not
            provided, the merger tries ``ts_code``, ``end_date``, ``ann_date``
            and ``period``.
        add_ts_code_from_filename
            If a source parquet lacks ``ts_code``, add it from the file stem
            such as ``000001.SZ``. Normal Tushare output already contains this
            column, so this is mainly a safety net.
        """

        if not self.input_dir.exists():
            raise FileNotFoundError(f"fundamentals 输入目录不存在：{self.input_dir}")

        source_files = self._list_source_files(pattern)
        self.logger.info("开始合并 fundamentals parquet：输入目录=%s，源文件数=%d", self.input_dir, len(source_files))

        frames: list[pd.DataFrame] = []
        failed_files: list[str] = []
        for file_path in source_files:
            try:
                df = pd.read_parquet(file_path)
                if df.empty:
                    continue
                if add_ts_code_from_filename and "ts_code" not in df.columns:
                    df = df.copy()
                    df.insert(0, "ts_code", file_path.stem)
                frames.append(df)
            except Exception as exc:
                failed_files.append(str(file_path))
                self.logger.warning("读取 fundamentals parquet 失败：%s，错误：%s", file_path, exc)

        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not combined.empty:
            if drop_duplicates:
                combined = combined.drop_duplicates().reset_index(drop=True)
            sort_columns = self._existing_sort_columns(combined, sort_by)
            if sort_columns:
                combined = combined.sort_values(sort_columns).reset_index(drop=True)

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(self.output_file, compression="snappy", index=False)

        summary = FundamentalsMergeSummary(
            input_dir=self.input_dir,
            output_file=self.output_file,
            total_files=len(source_files),
            merged_files=len(source_files) - len(failed_files),
            failed_files=tuple(failed_files),
            rows_written=len(combined),
        )
        self._log_summary(summary)
        return summary

    def _list_source_files(self, pattern: str) -> list[Path]:
        output_file = self.output_file.resolve()
        files: list[Path] = []
        for file_path in self.input_dir.glob(pattern):
            if not file_path.is_file():
                continue
            if file_path.resolve() == output_file:
                continue
            if file_path.name == DEFAULT_MERGED_OUTPUT_FILE_NAME:
                continue
            files.append(file_path)
        return sorted(files)

    @staticmethod
    def _existing_sort_columns(df: pd.DataFrame, sort_by: list[str] | None) -> list[str]:
        candidates = sort_by or ["ts_code", "end_date", "ann_date", "period"]
        return [column for column in candidates if column in df.columns]

    def _log_summary(self, summary: FundamentalsMergeSummary) -> None:
        self.logger.info("=" * 60)
        self.logger.info("fundamentals parquet 合并完成")
        self.logger.info("输入目录：%s", summary.input_dir)
        self.logger.info("输出文件：%s", summary.output_file)
        self.logger.info("源文件数：%d", summary.total_files)
        self.logger.info("成功合并文件数：%d", summary.merged_files)
        self.logger.info("失败文件：%s", list(summary.failed_files))
        self.logger.info("写入行数：%d", summary.rows_written)
        self.logger.info("=" * 60)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("fundamentals_fetcher")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="拉取并/或合并 Tushare 原始财务指标 fundamentals/fina_indicator")
    parser.add_argument("--token", type=str, default=None, help="Tushare Token；默认读取 TUSHARE_TOKEN 或 ~/.tushare_token")
    parser.add_argument("--start_date", type=str, default=None, help="开始日期，格式 YYYYMMDD；对应 fina_indicator 公告日期范围")
    parser.add_argument("--end_date", type=str, default=None, help="结束日期，格式 YYYYMMDD；默认今天")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_RAW_DATA_DIR),
        help=f"raw_data 根目录；默认 {DEFAULT_RAW_DATA_DIR}",
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
        default=5.0,
        help="请求失败后重试的基础等待秒数；实际等待为 retry_wait_seconds * retry_idx",
    )
    parser.add_argument("--max_retries", type=int, default=5, help="单只股票最大重试次数")
    parser.add_argument("--force_fetch", action="store_true", help="忽略进度文件，强制重新拉取并覆盖同名 parquet")
    parser.add_argument("--merge_only", action="store_true", help="只合并已有的按股票 parquet，不执行 Tushare 拉取")
    parser.add_argument("--merge_after_fetch", action="store_true", help="拉取完成后继续合并为一个大的 fundamentals parquet")
    parser.add_argument(
        "--merged_output_file",
        type=str,
        default=None,
        help="合并后的 parquet 文件；默认 <output_dir>/fundamentals/fundamentals.parquet",
    )
    parser.add_argument("--merge_pattern", type=str, default="*.parquet", help="合并输入文件 glob pattern；默认 *.parquet")
    parser.add_argument("--no_drop_duplicates", action="store_true", help="合并时不对完整重复行去重")
    parser.add_argument(
        "--sort_by",
        nargs="+",
        default=None,
        help="合并后可选排序字段列表；默认尝试 ts_code end_date ann_date period",
    )
    parser.add_argument("--log_level", type=str, default=os.environ.get("LOG_LEVEL", "INFO"), help="日志级别")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = setup_logging(args.log_level)

    if args.merge_only:
        merger = FundamentalsMerger(
            input_dir=Path(args.output_dir) / FundamentalsFetcher.FUNDAMENTALS_SUBDIR,
            output_file=args.merged_output_file,
            logger=logger,
        )
        summary = merger.merge(
            pattern=args.merge_pattern,
            drop_duplicates=not args.no_drop_duplicates,
            sort_by=args.sort_by,
        )
        return 1 if summary.failed_files else 0

    if not args.start_date:
        raise ValueError("非 --merge_only 模式下必须提供 --start_date")

    fetcher = FundamentalsFetcher(
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
    if summary.failed_codes:
        return 1

    if args.merge_after_fetch:
        merger = FundamentalsMerger(
            input_dir=fetcher.fundamentals_dir,
            output_file=args.merged_output_file,
            logger=logger,
        )
        merge_summary = merger.merge(
            pattern=args.merge_pattern,
            drop_duplicates=not args.no_drop_duplicates,
            sort_by=args.sort_by,
        )
        return 1 if merge_summary.failed_files else 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
