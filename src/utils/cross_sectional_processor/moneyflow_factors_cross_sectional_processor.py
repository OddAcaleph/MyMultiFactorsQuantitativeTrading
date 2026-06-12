"""Cross-sectional processing for moneyflow_factors.

The processor treats source moneyflow feature parquet files as read-only,
applies cross-sectional winsorization followed by z-score standardization to
selected moneyflow factor columns, and writes enriched parquet files into a
separate output directory while preserving the original ``year=*/month=*``
daily partition layout.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd

try:  # Support running with PYTHONPATH as project root: import src.utils...
    from src.utils.cross_sectional_processor.data_winsorize_util import winsorize_by_trade_date
    from src.utils.cross_sectional_processor.data_z_score_util import zscore_by_trade_date
except ModuleNotFoundError:  # Support running with PYTHONPATH=.../src: import utils...
    from utils.cross_sectional_processor.data_winsorize_util import winsorize_by_trade_date
    from utils.cross_sectional_processor.data_z_score_util import zscore_by_trade_date


@dataclass(frozen=True)
class MoneyflowFactorsCrossSectionalSummary:
    """Processing statistics for moneyflow_factors cross-sectional features."""

    input_dir: Path
    output_dir: Path
    files_found: int
    files_processed: int
    rows_read: int
    rows_written: int
    factor_columns: tuple[str, ...]
    missing_required_files: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    failed_files: Mapping[str, str] = field(default_factory=dict)

    @property
    def processed_columns(self) -> tuple[str, ...]:
        """Names of generated cross-sectionally processed columns."""

        return tuple(f"{column}_cc_processed" for column in self.factor_columns)

    @property
    def has_errors(self) -> bool:
        """Whether any file was skipped or failed."""

        return bool(self.missing_required_files or self.failed_files)


class MoneyflowFactorsCrossSectionalProcessor:
    """Apply winsorize then z-score processing to moneyflow_factors.

    Output keeps all original columns and appends one ``{factor}_cc_processed``
    column for each configured factor. Source parquet files are never modified.
    """

    DEFAULT_INPUT_DIR = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/features_data/moneyflow_factors"
    )
    DEFAULT_OUTPUT_DIR = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cross_sectional_processd_data/moneyflow_factors"
    )
    DEFAULT_LOG_FILE = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/cross_sectional_process/"
        "moneyflow_factors_cross_sectional_processor.log"
    )

    DATE_COLUMN = "trade_date"
    KEY_COLUMNS: tuple[str, str] = ("trade_date", "ts_code")
    FACTOR_COLUMNS: tuple[str, ...] = (
        "lg_net_inflow",
        "elg_net_inflow",
        "main_net_inflow",
        "main_net_ratio",
        "retail_net_inflow",
        "retail_ratio",
        "main_retail_diff",
        "main_net_5",
        "main_net_10",
        "main_net_20",
        "mf_ma5",
        "mf_ma20",
        "mf_acceleration",
    )

    def __init__(
        self,
        input_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        factor_columns: Sequence[str] | None = None,
        logger: logging.Logger | None = None,
        fail_fast: bool = False,
    ) -> None:
        self.input_dir = Path(input_dir or self.DEFAULT_INPUT_DIR)
        self.output_dir = Path(output_dir or self.DEFAULT_OUTPUT_DIR)
        self.factor_columns = tuple(factor_columns or self.FACTOR_COLUMNS)
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.fail_fast = fail_fast

        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise ValueError(f"Input path is not a directory: {self.input_dir}")

    def process(self) -> MoneyflowFactorsCrossSectionalSummary:
        """Process every daily parquet file and preserve relative output paths."""

        input_files = self._list_input_files()
        self.logger.info(
            "开始横截面处理 moneyflow_factors：input=%s, output=%s, files=%d, factors=%s",
            self.input_dir,
            self.output_dir,
            len(input_files),
            ",".join(self.factor_columns),
        )
        self.logger.info("原始 moneyflow_factors parquet 只读；输出文件写入新的 cross_sectional_processd_data 路径。")

        rows_read = 0
        rows_written = 0
        files_processed = 0
        missing_required_files: dict[str, tuple[str, ...]] = {}
        failed_files: dict[str, str] = {}

        for file_index, input_file in enumerate(input_files, start=1):
            try:
                output_file = self._output_file_for(input_file)
                processed_df = self._process_one_file(input_file)
                rows_read += len(processed_df)
                output_file.parent.mkdir(parents=True, exist_ok=True)
                processed_df.to_parquet(output_file, index=False)
                rows_written += len(processed_df)
                files_processed += 1

                if file_index == 1 or file_index % 250 == 0 or file_index == len(input_files):
                    self.logger.info(
                        "处理进度：%d/%d files, latest_input=%s, latest_output=%s, rows=%d",
                        file_index,
                        len(input_files),
                        input_file,
                        output_file,
                        len(processed_df),
                    )
            except MissingRequiredColumnsError as exc:
                missing_required_files[str(input_file)] = tuple(exc.missing_columns)
                self.logger.error("文件缺少必需列，已跳过：file=%s, missing_columns=%s", input_file, exc.missing_columns)
                if self.fail_fast:
                    raise
            except Exception as exc:  # noqa: BLE001 - log per-file errors and continue by default.
                failed_files[str(input_file)] = repr(exc)
                self.logger.exception("文件处理失败，已跳过：file=%s", input_file)
                if self.fail_fast:
                    raise

        summary = MoneyflowFactorsCrossSectionalSummary(
            input_dir=self.input_dir,
            output_dir=self.output_dir,
            files_found=len(input_files),
            files_processed=files_processed,
            rows_read=rows_read,
            rows_written=rows_written,
            factor_columns=self.factor_columns,
            missing_required_files=missing_required_files,
            failed_files=failed_files,
        )
        self._log_summary(summary)
        return summary

    def _list_input_files(self) -> list[Path]:
        files = sorted(self.input_dir.glob("year=*/month=*/*.parquet"))
        if not files:
            raise FileNotFoundError(f"No daily parquet files found under: {self.input_dir}")
        return files

    def _output_file_for(self, input_file: Path) -> Path:
        return self.output_dir / input_file.relative_to(self.input_dir)

    def _process_one_file(self, input_file: Path) -> pd.DataFrame:
        df = pd.read_parquet(input_file)
        self._validate_columns(df.columns)

        processed = df.copy()
        for factor in self.factor_columns:
            temp_factor = f"__{factor}_winsorized"
            processed[temp_factor] = pd.to_numeric(processed[factor], errors="coerce")
            processed[temp_factor] = winsorize_by_trade_date(processed, temp_factor, date_column=self.DATE_COLUMN)
            processed[f"{factor}_cc_processed"] = zscore_by_trade_date(
                processed,
                temp_factor,
                date_column=self.DATE_COLUMN,
            )
            processed = processed.drop(columns=[temp_factor])

        return processed

    def _validate_columns(self, columns: Iterable[str]) -> None:
        column_set = set(columns)
        required_columns = (*self.KEY_COLUMNS, *self.factor_columns)
        missing_columns = tuple(column for column in required_columns if column not in column_set)
        if missing_columns:
            raise MissingRequiredColumnsError(missing_columns)

    def _log_summary(self, summary: MoneyflowFactorsCrossSectionalSummary) -> None:
        self.logger.info(
            "moneyflow_factors 横截面处理汇总：files_found=%d, files_processed=%d, rows_read=%d, rows_written=%d, "
            "factors=%s, processed_columns=%s, output=%s",
            summary.files_found,
            summary.files_processed,
            summary.rows_read,
            summary.rows_written,
            ",".join(summary.factor_columns),
            ",".join(summary.processed_columns),
            summary.output_dir,
        )
        if summary.missing_required_files:
            self.logger.error("存在 %d 个文件缺少必需列。", len(summary.missing_required_files))
        else:
            self.logger.info("必需列检查通过：所有已处理文件均包含 trade_date/ts_code 与目标因子列。")

        if summary.failed_files:
            self.logger.error("存在 %d 个文件处理失败。", len(summary.failed_files))
        else:
            self.logger.info("文件处理检查通过：未发生非预期处理失败。")


class MissingRequiredColumnsError(ValueError):
    """Raised when an input parquet file misses required columns."""

    def __init__(self, missing_columns: Sequence[str]) -> None:
        self.missing_columns = tuple(missing_columns)
        super().__init__(f"Missing required columns: {self.missing_columns}")


def configure_logging(log_level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure console and file logging for CLI usage."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    target_log_file = Path(log_file) if log_file is not None else MoneyflowFactorsCrossSectionalProcessor.DEFAULT_LOG_FILE
    target_log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(target_log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Winsorize then z-score selected moneyflow_factors by trade_date cross-section."
    )
    parser.add_argument(
        "--input-dir",
        default=str(MoneyflowFactorsCrossSectionalProcessor.DEFAULT_INPUT_DIR),
        help="Input moneyflow_factors directory with year=*/month=* daily parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(MoneyflowFactorsCrossSectionalProcessor.DEFAULT_OUTPUT_DIR),
        help="Output directory for cross-sectionally processed daily parquet files.",
    )
    parser.add_argument(
        "--factor-columns",
        nargs="+",
        default=list(MoneyflowFactorsCrossSectionalProcessor.FACTOR_COLUMNS),
        help="Factor columns to process. Each output column is named {factor}_cc_processed.",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop immediately when any input file fails.")
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    parser.add_argument(
        "--log-file",
        default=str(MoneyflowFactorsCrossSectionalProcessor.DEFAULT_LOG_FILE),
        help="Log file path.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> MoneyflowFactorsCrossSectionalSummary:
    args = parse_args(argv)
    configure_logging(log_level=args.log_level, log_file=args.log_file)
    processor = MoneyflowFactorsCrossSectionalProcessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        factor_columns=args.factor_columns,
        fail_fast=args.fail_fast,
    )
    return processor.process()


if __name__ == "__main__":
    # 使用方法
    # PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab" \
    # python -m src.utils.cross_sectional_processor.moneyflow_factors_cross_sectional_processor \
    # --log-file "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/cross_sectional_process/moneyflow_factors_cross_sectional_processor.log" \
    # --input-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/features_data/moneyflow_factors" \
    # --output-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cross_sectional_processd_data/moneyflow_factors"
    main()


__all__ = [
    "MissingRequiredColumnsError",
    "MoneyflowFactorsCrossSectionalProcessor",
    "MoneyflowFactorsCrossSectionalSummary",
    "configure_logging",
    "main",
    "parse_args",
]
