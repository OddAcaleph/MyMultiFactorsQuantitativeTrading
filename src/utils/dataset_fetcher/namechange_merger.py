"""Merge per-stock Tushare namechange parquet files into one large parquet.

``namechange_fetcher.py`` writes raw ST/name-change history as one parquet per
stock to make network failures easy to retry. This utility combines those files
into a single table for downstream preprocessing:

```
<raw_data_dir>/namechange/*.parquet  ->  <raw_data_dir>/namechange/namechange.parquet
```
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_RAW_DATA_DIR = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/raw_data")
DEFAULT_NAMECHANGE_DIR = DEFAULT_RAW_DATA_DIR / "namechange"
DEFAULT_OUTPUT_FILE_NAME = "namechange.parquet"


@dataclass(frozen=True)
class NamechangeMergeSummary:
    """Summary returned after merging per-stock namechange parquet files."""

    input_dir: Path
    output_file: Path
    total_files: int
    merged_files: int
    failed_files: tuple[str, ...]
    rows_written: int


class NamechangeMerger:
    """Merge per-stock raw ``namechange`` parquet files into one parquet.

    Parameters
    ----------
    input_dir
        Directory containing per-stock parquet files such as
        ``000001.SZ.parquet``. Defaults to
        ``/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/raw_data/namechange``.
    output_file
        Target merged parquet. Defaults to ``<input_dir>/namechange.parquet``.
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
        self.input_dir = Path(input_dir or DEFAULT_NAMECHANGE_DIR)
        self.output_file = Path(output_file) if output_file else self.input_dir / DEFAULT_OUTPUT_FILE_NAME
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    def merge(
        self,
        pattern: str = "*.parquet",
        drop_duplicates: bool = True,
        sort_by: list[str] | None = None,
    ) -> NamechangeMergeSummary:
        """Merge per-stock parquet files.

        Parameters
        ----------
        pattern
            Glob pattern used inside ``input_dir``. The default reads all
            parquet files and excludes ``output_file`` automatically.
        drop_duplicates
            Whether to drop fully duplicated rows before writing output.
        sort_by
            Optional column names used for stable sorting when present. If not
            provided, the merger tries ``ts_code``, ``start_date``, ``end_date``.
        """

        if not self.input_dir.exists():
            raise FileNotFoundError(f"namechange 输入目录不存在：{self.input_dir}")

        source_files = self._list_source_files(pattern)
        self.logger.info("开始合并 namechange parquet：输入目录=%s，源文件数=%d", self.input_dir, len(source_files))

        frames: list[pd.DataFrame] = []
        failed_files: list[str] = []
        for file_path in source_files:
            try:
                df = pd.read_parquet(file_path)
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                failed_files.append(str(file_path))
                self.logger.warning("读取 namechange parquet 失败：%s，错误：%s", file_path, exc)

        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if not combined.empty:
            if drop_duplicates:
                combined = combined.drop_duplicates().reset_index(drop=True)
            sort_columns = self._existing_sort_columns(combined, sort_by)
            if sort_columns:
                combined = combined.sort_values(sort_columns).reset_index(drop=True)

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(self.output_file, compression="snappy", index=False)

        summary = NamechangeMergeSummary(
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
            files.append(file_path)
        return sorted(files)

    @staticmethod
    def _existing_sort_columns(df: pd.DataFrame, sort_by: list[str] | None) -> list[str]:
        candidates = sort_by or ["ts_code", "start_date", "end_date"]
        return [column for column in candidates if column in df.columns]

    def _log_summary(self, summary: NamechangeMergeSummary) -> None:
        self.logger.info("=" * 60)
        self.logger.info("namechange parquet 合并完成")
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
    return logging.getLogger("namechange_merger")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="合并按股票保存的 Tushare namechange parquet 为一个大表")
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(DEFAULT_NAMECHANGE_DIR),
        help=f"按股票保存的 namechange parquet 目录；默认 {DEFAULT_NAMECHANGE_DIR}",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="合并后的 parquet 文件；默认 <input_dir>/namechange.parquet",
    )
    parser.add_argument("--pattern", type=str, default="*.parquet", help="输入文件 glob pattern；默认 *.parquet")
    parser.add_argument("--no_drop_duplicates", action="store_true", help="不对合并后的完整重复行去重")
    parser.add_argument(
        "--sort_by",
        nargs="+",
        default=None,
        help="可选排序字段列表；默认尝试 ts_code start_date end_date",
    )
    parser.add_argument("--log_level", type=str, default=os.environ.get("LOG_LEVEL", "INFO"), help="日志级别")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger = setup_logging(args.log_level)
    merger = NamechangeMerger(
        input_dir=args.input_dir,
        output_file=args.output_file,
        logger=logger,
    )
    summary = merger.merge(
        pattern=args.pattern,
        drop_duplicates=not args.no_drop_duplicates,
        sort_by=args.sort_by,
    )
    return 1 if summary.failed_files else 0


if __name__ == "__main__":
    raise SystemExit(main())
