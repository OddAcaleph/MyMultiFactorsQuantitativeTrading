"""Generate one-hot encoded wide industry features.

The cleaned ``industry.parquet`` file is treated as read-only. This processor
checks uniqueness and missing values for the industry classification fields,
then writes a wide table with ``in_date``/``out_date`` validity columns and
one-hot columns for ``industry``, ``L1``, ``L2`` and ``L3`` categories.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


@dataclass(frozen=True)
class IndustryOneHotProcessSummary:
    """Quality-check and output statistics for industry one-hot features."""

    input_file: Path
    output_file: Path
    rows_read: int
    rows_written: int
    stocks_processed: int
    feature_columns: int
    raw_duplicate_ts_code_rows: int
    raw_conflict_ts_code_rows: int
    raw_missing_rows: int
    raw_missing_by_column: Mapping[str, int]
    output_duplicate_ts_code_rows: int
    output_missing_rows: int

    @property
    def has_uniqueness_issue(self) -> bool:
        """Whether raw or output uniqueness issues were found."""

        return self.raw_duplicate_ts_code_rows > 0 or self.output_duplicate_ts_code_rows > 0

    @property
    def has_missing_issue(self) -> bool:
        """Whether missing values were found in raw category fields or output."""

        return self.raw_missing_rows > 0 or self.output_missing_rows > 0


class IndustryOneHotProcessor:
    """Create a unique ``ts_code`` keyed industry one-hot wide table.

    Output columns:

    - ``ts_code``: stock code.
    - ``in_date``: industry classification effective date.
    - ``out_date``: industry classification invalid date; empty if still active.
    - ``industry_<category>``: one-hot columns from the cleaned ``industry`` field.
    - ``L1_<category>``: one-hot columns from ``l1_name``.
    - ``L2_<category>``: one-hot columns from ``l2_name``.
    - ``L3_<category>``: one-hot columns from ``l3_name``.

    Missing category values are not imputed and do not create a dummy category;
    their corresponding level's one-hot columns are all zero. The input parquet
    file is never modified.
    """

    DEFAULT_INPUT_FILE = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/industry/industry.parquet"
    )
    DEFAULT_OUTPUT_FILE = Path(
        "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/industry/industry_onehot.parquet"
    )
    DEFAULT_LOG_FILE = Path("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/industry_onehot_processing.log")

    KEY_COLUMN = "ts_code"
    CATEGORY_COLUMNS: Mapping[str, str] = {
        "industry": "industry",
        "L1": "l1_name",
        "L2": "l2_name",
        "L3": "l3_name",
    }
    VALIDITY_COLUMNS: tuple[str, str] = ("in_date", "out_date")
    REQUIRED_COLUMNS: tuple[str, ...] = (KEY_COLUMN, *CATEGORY_COLUMNS.values(), *VALIDITY_COLUMNS)
    MISSING_CHECK_COLUMNS: tuple[str, ...] = (KEY_COLUMN, *CATEGORY_COLUMNS.values())

    def __init__(
        self,
        input_file: str | Path | None = None,
        output_file: str | Path | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.input_file = Path(input_file or self.DEFAULT_INPUT_FILE)
        self.output_file = Path(output_file or self.DEFAULT_OUTPUT_FILE)
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        if not self.input_file.exists():
            raise FileNotFoundError(f"Industry input file does not exist: {self.input_file}")
        if not self.input_file.is_file():
            raise ValueError(f"Industry input path is not a file: {self.input_file}")

    def process(self) -> IndustryOneHotProcessSummary:
        """Run quality checks and write the one-hot encoded wide parquet."""

        self.logger.info("开始处理 industry one-hot 宽表：input=%s, output=%s", self.input_file, self.output_file)
        self.logger.info("原始/清洗后 industry parquet 只读；输出为 ts_code 唯一的行业 one-hot 宽表。")

        raw_df = pd.read_parquet(self.input_file).reset_index(names="_raw_row")
        self._validate_columns(raw_df.columns)

        rows_read = len(raw_df)
        raw_duplicate_ts_code_rows = int(raw_df.duplicated(subset=[self.KEY_COLUMN], keep=False).sum())
        raw_conflict_ts_code_rows = self._count_conflict_ts_code_rows(raw_df)
        raw_missing_by_column = {column: int(self._is_null_or_empty(raw_df[column]).sum()) for column in self.MISSING_CHECK_COLUMNS}
        raw_missing_rows = int(self._required_missing_mask(raw_df).sum())

        prepared_df = self._prepare_raw_dataframe(raw_df)
        feature_df = self._build_one_hot_features(prepared_df)

        output_duplicate_ts_code_rows = int(feature_df.duplicated(subset=[self.KEY_COLUMN], keep=False).sum())
        output_missing_rows = int(feature_df.isna().any(axis=1).sum())

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_parquet(self.output_file, index=False)

        summary = IndustryOneHotProcessSummary(
            input_file=self.input_file,
            output_file=self.output_file,
            rows_read=rows_read,
            rows_written=len(feature_df),
            stocks_processed=int(feature_df[self.KEY_COLUMN].nunique()) if not feature_df.empty else 0,
            feature_columns=max(len(feature_df.columns) - 1, 0),
            raw_duplicate_ts_code_rows=raw_duplicate_ts_code_rows,
            raw_conflict_ts_code_rows=raw_conflict_ts_code_rows,
            raw_missing_rows=raw_missing_rows,
            raw_missing_by_column=raw_missing_by_column,
            output_duplicate_ts_code_rows=output_duplicate_ts_code_rows,
            output_missing_rows=output_missing_rows,
        )
        self._log_summary(summary)
        return summary

    def _validate_columns(self, columns: Iterable[str]) -> None:
        column_set = set(columns)
        missing_columns = [column for column in self.REQUIRED_COLUMNS if column not in column_set]
        if missing_columns:
            raise ValueError(f"{self.input_file} is missing required columns: {missing_columns}")

    def _prepare_raw_dataframe(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        prepared = raw_df.loc[:, ("_raw_row", *self.REQUIRED_COLUMNS)].copy()
        prepared[self.KEY_COLUMN] = prepared[self.KEY_COLUMN].astype("string").str.strip()
        for column in self.CATEGORY_COLUMNS.values():
            prepared[column] = prepared[column].astype("string").str.strip()
            prepared.loc[prepared[column].eq(""), column] = pd.NA
        for column in self.VALIDITY_COLUMNS:
            prepared[column] = prepared[column].astype("string").str.strip().fillna("")
            prepared.loc[prepared[column].isin(["<NA>", "nan", "NaT", "None"]), column] = ""

        missing_ts_code_rows = int(self._is_null_or_empty(prepared[self.KEY_COLUMN]).sum())
        if missing_ts_code_rows:
            self.logger.warning("发现 %d 行 ts_code 缺失，这些记录不会参与 one-hot 输出。", missing_ts_code_rows)
            prepared = prepared.loc[~self._is_null_or_empty(prepared[self.KEY_COLUMN])].copy()

        if prepared.empty:
            raise ValueError("No valid industry records remain after removing missing ts_code rows.")

        if prepared.duplicated(subset=[self.KEY_COLUMN], keep=False).any():
            self.logger.warning("发现重复 ts_code，输出会按原文件顺序保留最后一条记录以保证宽表唯一性。")
            prepared = prepared.sort_values("_raw_row", kind="mergesort").drop_duplicates(subset=[self.KEY_COLUMN], keep="last")

        return prepared.sort_values(self.KEY_COLUMN, kind="mergesort").reset_index(drop=True)

    def _build_one_hot_features(self, prepared_df: pd.DataFrame) -> pd.DataFrame:
        output = pd.DataFrame(
            {
                self.KEY_COLUMN: prepared_df[self.KEY_COLUMN].astype("string"),
                "in_date": prepared_df["in_date"].astype("string"),
                "out_date": prepared_df["out_date"].astype("string"),
            }
        )

        for prefix, source_column in self.CATEGORY_COLUMNS.items():
            dummy_df = pd.get_dummies(prepared_df[source_column], prefix=prefix, prefix_sep="_", dummy_na=False, dtype="int8")
            dummy_df = dummy_df.reindex(sorted(dummy_df.columns), axis=1)
            self.logger.info(
                "%s one-hot 编码完成：source_column=%s, categories=%d",
                prefix,
                source_column,
                len(dummy_df.columns),
            )
            output = pd.concat([output, dummy_df], axis=1)

        return output

    def _count_conflict_ts_code_rows(self, raw_df: pd.DataFrame) -> int:
        conflict_counts = (
            raw_df.groupby(self.KEY_COLUMN, dropna=False)[list(self.CATEGORY_COLUMNS.values())]
            .nunique(dropna=True)
            .max(axis=1)
        )
        conflict_keys = conflict_counts[conflict_counts > 1].index
        if len(conflict_keys) == 0:
            return 0
        return int(raw_df[self.KEY_COLUMN].isin(conflict_keys).sum())

    def _required_missing_mask(self, df: pd.DataFrame) -> pd.Series:
        missing_mask = pd.Series(False, index=df.index)
        for column in self.MISSING_CHECK_COLUMNS:
            missing_mask = missing_mask | self._is_null_or_empty(df[column])
        return missing_mask

    @staticmethod
    def _is_null_or_empty(series: pd.Series) -> pd.Series:
        return series.isna() | series.astype("string").str.strip().eq("").fillna(False)

    def _log_summary(self, summary: IndustryOneHotProcessSummary) -> None:
        self.logger.info(
            "industry one-hot 宽表处理汇总：rows_read=%d, rows_written=%d, stocks=%d, feature_columns=%d, "
            "raw_duplicate_ts_code_rows=%d, raw_conflict_ts_code_rows=%d, raw_missing_rows=%d, "
            "output_duplicate_ts_code_rows=%d, output_missing_rows=%d, output=%s",
            summary.rows_read,
            summary.rows_written,
            summary.stocks_processed,
            summary.feature_columns,
            summary.raw_duplicate_ts_code_rows,
            summary.raw_conflict_ts_code_rows,
            summary.raw_missing_rows,
            summary.output_duplicate_ts_code_rows,
            summary.output_missing_rows,
            summary.output_file,
        )

        if summary.raw_duplicate_ts_code_rows:
            self.logger.warning(
                "原始唯一性检查发现问题：%d 行存在重复 ts_code；输出已保留最后一条以保证 ts_code 唯一。",
                summary.raw_duplicate_ts_code_rows,
            )
        else:
            self.logger.info("原始唯一性检查通过：未发现重复 ts_code。")

        if summary.raw_conflict_ts_code_rows:
            self.logger.warning(
                "原始分类冲突检查发现问题：%d 行重复 ts_code 下存在不同 industry/L1/L2/L3 分类。",
                summary.raw_conflict_ts_code_rows,
            )
        else:
            self.logger.info("原始分类冲突检查通过：同一 ts_code 未出现不同 industry/L1/L2/L3 分类。")

        if summary.output_duplicate_ts_code_rows:
            self.logger.error("输出唯一性检查失败：%d 行存在重复 ts_code。", summary.output_duplicate_ts_code_rows)
        else:
            self.logger.info("输出唯一性检查通过：未发现重复 ts_code。")

        missing_detail = ", ".join(f"{column}={count}" for column, count in summary.raw_missing_by_column.items())
        if summary.raw_missing_rows:
            self.logger.warning(
                "缺失值检查发现问题：%d 行 industry/L1/L2/L3 或 ts_code 存在缺失；明细：%s。缺失分类对应 one-hot 列保持全 0。",
                summary.raw_missing_rows,
                missing_detail,
            )
        else:
            self.logger.info("缺失值检查通过：ts_code/industry/L1/L2/L3 均无缺失；明细：%s。", missing_detail)

        if summary.output_missing_rows:
            self.logger.error("输出缺失值检查失败：%d 行输出宽表存在缺失。", summary.output_missing_rows)
        else:
            self.logger.info("输出缺失值检查通过：ts_code 和所有 one-hot 特征均无缺失。")


def configure_logging(log_level: str = "INFO", log_file: str | Path | None = None) -> None:
    """Configure console and file logging for CLI usage."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    target_log_file = Path(log_file) if log_file is not None else IndustryOneHotProcessor.DEFAULT_LOG_FILE
    target_log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(target_log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate industry/L1/L2/L3 one-hot wide features from industry parquet data.")
    parser.add_argument(
        "--input-file",
        default=str(IndustryOneHotProcessor.DEFAULT_INPUT_FILE),
        help="Cleaned industry parquet file.",
    )
    parser.add_argument(
        "--output-file",
        default=str(IndustryOneHotProcessor.DEFAULT_OUTPUT_FILE),
        help="Processed industry one-hot wide parquet file.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level, e.g. INFO or DEBUG.")
    parser.add_argument("--log-file", default=str(IndustryOneHotProcessor.DEFAULT_LOG_FILE), help="Log file path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> IndustryOneHotProcessSummary:
    args = parse_args(argv)
    configure_logging(log_level=args.log_level, log_file=args.log_file)
    processor = IndustryOneHotProcessor(input_file=args.input_file, output_file=args.output_file)
    return processor.process()


if __name__ == "__main__":
    # PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
    # python -m utils.dataset_processor.industry_onehot_processor \
    #   --input-file "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/industry/industry.parquet" \
    #   --output-file "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/industry/industry_onehot.parquet" \
    #   --log-file "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/industry_onehot_processing.log"
    main()


__all__ = ["IndustryOneHotProcessSummary", "IndustryOneHotProcessor", "configure_logging", "main", "parse_args"]
