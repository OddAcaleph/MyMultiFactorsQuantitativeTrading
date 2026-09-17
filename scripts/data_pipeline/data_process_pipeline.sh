#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# A 股数据清洗与处理流水线
# ============================================================================
#
# 前置条件：
#   已通过 data_fetch_pipeline.sh 将原始数据拉取到 data/raw_data 下。
#
# 这个脚本负责从 raw_data 进入 cleaned_data / processd_data：
#   1. 准备 raw 聚合大表
#      - fetch 阶段按日期分区落盘的数据会先合并成单 parquet，供现有 cleaner/processor 读取；
#      - fundamentals/namechange 的按股票 parquet 也会重新合并，保证下游读到最新大表。
#   2. 清洗数据
#      - 使用 src/utils/dataset_cleaner 下的清洗工具；
#      - 原始 raw parquet 不会被覆盖，清洗结果写到 data/cleaned_data。
#   3. 处理数据
#      - 使用 src/utils/dataset_processor 下的处理工具；
#      - 生成 ST 日频、停牌日频、行业 one-hot、最终 daily_bars 宽表等中间/下游数据。
#
# 用法：
#   bash scripts/data_pipeline/data_process_pipeline.sh [start_date] [end_date]
#
# 参数：
#   start_date/end_date 仅传给 wide_table_daily_bars_builder，用于限制最终宽表日期范围；
#   不传则处理 cleaned daily_bars 中的全量日期。
#
# 示例：
#   # 清洗和处理全量数据
#   bash scripts/data_pipeline/data_process_pipeline.sh
#
#   # 只构建指定日期区间的最终宽表，其他清洗/处理仍基于当前 raw/cleaned 全量输入
#   bash scripts/data_pipeline/data_process_pipeline.sh 20200101 20260618
#
# 常用环境变量：
#   CONTINUE_ON_ERROR=0        遇到第一个失败步骤立即退出；默认 1，继续执行后续独立步骤
#   PREPARE_RAW_AGGREGATES=0   跳过 raw 聚合大表准备；默认 1
#   BUILD_WIDE_TABLE=0         跳过最终 wide_table_daily_bars 构建；默认 1
#   LOG_DIR=/path              覆盖 pipeline 日志目录；默认 log/data_pipeline
#   LOG_LEVEL=DEBUG            传给 Python cleaner/processor 的日志级别；默认 INFO
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATASET_FETCHER_PATH="${PROJECT_ROOT}/src/utils/dataset_fetcher"
DATASET_CLEANER_PATH="${PROJECT_ROOT}/src/utils/dataset_cleaner"
DATASET_PROCESSOR_PATH="${PROJECT_ROOT}/src/utils/dataset_processor"

RAW_DATA_DIR="${PROJECT_ROOT}/data/raw_data"
CLEANED_DATA_DIR="${PROJECT_ROOT}/data/cleaned_data"
PROCESSED_DATA_DIR="${PROJECT_ROOT}/data/processd_data"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/log/data_pipeline}"

normalize_date_compact() {
  local raw_date="$1"
  local compact_date

  compact_date="${raw_date//-/}"
  if [[ ! "${compact_date}" =~ ^[0-9]{8}$ ]]; then
    echo "日期格式错误：${raw_date}，请使用 YYYYMMDD 或 YYYY-MM-DD" >&2
    return 1
  fi

  local hyphen_date="${compact_date:0:4}-${compact_date:4:2}-${compact_date:6:2}"
  if [[ "$(date -d "${hyphen_date}" +%Y%m%d 2>/dev/null || true)" != "${compact_date}" ]]; then
    echo "非法日期：${raw_date}" >&2
    return 1
  fi

  echo "${compact_date}"
}

START_DATE=""
END_DATE=""
if [[ -n "${1:-}" ]]; then
  START_DATE="$(normalize_date_compact "$1")"
fi
if [[ -n "${2:-}" ]]; then
  END_DATE="$(normalize_date_compact "$2")"
else
  END_DATE="${START_DATE}"
fi
if [[ -n "${START_DATE}" && -n "${END_DATE}" && "${START_DATE}" > "${END_DATE}" ]]; then
  echo "日期区间错误：start_date=${START_DATE} 晚于 end_date=${END_DATE}" >&2
  exit 1
fi
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
PREPARE_RAW_AGGREGATES="${PREPARE_RAW_AGGREGATES:-1}"
BUILD_WIDE_TABLE="${BUILD_WIDE_TABLE:-1}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

mkdir -p "${RAW_DATA_DIR}" "${CLEANED_DATA_DIR}" "${PROCESSED_DATA_DIR}" "${LOG_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
FAILED_STEPS=()
declare -A STEP_STATUS=()

WIDE_DATE_ARGS=()
if [[ -n "${START_DATE}" ]]; then
  WIDE_DATE_ARGS+=(--start-date "${START_DATE}")
fi
if [[ -n "${END_DATE}" ]]; then
  WIDE_DATE_ARGS+=(--end-date "${END_DATE}")
fi

# 统一步骤执行器：保持 pipeline 可观测、可续跑。
run_step() {
  local step_name="$1"
  shift
  local log_file="${LOG_DIR}/${RUN_ID}_${step_name}.log"

  echo "[$(date '+%F %T')] START ${step_name}，日志：${log_file}"
  set +e
  "$@" 2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e

  STEP_STATUS["${step_name}"]="${status}"
  if [[ ${status} -ne 0 ]]; then
    FAILED_STEPS+=("${step_name}")
    echo "[$(date '+%F %T')] FAIL  ${step_name}，退出码：${status}"
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit "${status}"
    fi
    return 0
  fi

  echo "[$(date '+%F %T')] OK    ${step_name}"
}

# 将 raw_data/<dataset>/year=*/month=*/*.parquet 合并成 raw_data/<dataset>/<output_name>。
# 当前 adj_factors/moneyflow/industry/suspend_d 的 cleaner/processor 读取单个聚合 parquet，
# 因此这里提供轻量 raw 聚合步骤。输出文件不会参与输入扫描，避免自我递归合并。
merge_partitioned_raw() {
  local dataset_name="$1"
  local output_name="$2"
  local input_dir="${RAW_DATA_DIR}/${dataset_name}"
  local output_file="${input_dir}/${output_name}"

  python3 - "${input_dir}" "${output_file}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

input_dir = Path(sys.argv[1])
output_file = Path(sys.argv[2])

if not input_dir.exists():
    raise FileNotFoundError(f"raw partition directory does not exist: {input_dir}")

source_files = sorted(input_dir.glob("year=*/month=*/*.parquet"))
if not source_files:
    raise FileNotFoundError(f"no partition parquet files found under {input_dir}/year=*/month=*/*.parquet")

frames = []
rows_read = 0
for file_path in source_files:
    df = pd.read_parquet(file_path)
    rows_read += len(df)
    if not df.empty:
        frames.append(df)

combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
if not combined.empty:
    combined = combined.drop_duplicates().reset_index(drop=True)

output_file.parent.mkdir(parents=True, exist_ok=True)
combined.to_parquet(output_file, compression="snappy", index=False)
print(
    f"merged raw partitions: input_dir={input_dir}, files={len(source_files)}, "
    f"rows_read={rows_read}, rows_written={len(combined)}, output={output_file}"
)
PY
}

echo "data process pipeline run_id=${RUN_ID}, raw_data=${RAW_DATA_DIR}, cleaned_data=${CLEANED_DATA_DIR}, processd_data=${PROCESSED_DATA_DIR}"

if [[ "${PREPARE_RAW_AGGREGATES}" == "1" ]]; then
  # fundamentals/namechange 是按股票文件保存，复用 fetcher/merger 自带合并逻辑。
  run_step raw_fundamentals_merge \
    python3 "${DATASET_FETCHER_PATH}/fundamentals_fetcher.py" \
      --merge_only \
      --output_dir "${RAW_DATA_DIR}"

  run_step raw_namechange_merge \
    python3 "${DATASET_FETCHER_PATH}/namechange_merger.py" \
      --input_dir "${RAW_DATA_DIR}/namechange" \
      --output_file "${RAW_DATA_DIR}/namechange/namechange.parquet"

  # 以下几个数据集 fetch 阶段是按日期分区保存；这里合并成 cleaner/processor 需要的单 parquet。
  run_step raw_adj_factors_merge merge_partitioned_raw "adj_factors" "adj_factors.parquet"
  run_step raw_moneyflow_merge merge_partitioned_raw "moneyflow" "moneyflow.parquet"
  run_step raw_industry_merge merge_partitioned_raw "industry" "industry.parquet"
  run_step raw_suspend_d_merge merge_partitioned_raw "suspend_d" "suspend_d.parquet"
else
  echo "跳过 raw 聚合大表准备：PREPARE_RAW_AGGREGATES=${PREPARE_RAW_AGGREGATES}"
fi

# ---------------------------------------------------------------------------
# 清洗阶段：raw_data -> cleaned_data
# ---------------------------------------------------------------------------

run_step clean_daily_bars \
  python3 "${DATASET_CLEANER_PATH}/daily_bars_cleaner.py" \
    --input-dir "${RAW_DATA_DIR}/daily_bars" \
    --output-dir "${CLEANED_DATA_DIR}/daily_bars" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_daily_bars_cleaning_inner.log"

run_step clean_adj_factors \
  python3 "${DATASET_CLEANER_PATH}/adj_factors_cleaner.py" \
    --input-file "${RAW_DATA_DIR}/adj_factors/adj_factors.parquet" \
    --output-file "${CLEANED_DATA_DIR}/adj_factors/adj_factors.parquet" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_adj_factors_cleaning_inner.log"

run_step clean_moneyflow \
  python3 "${DATASET_CLEANER_PATH}/moneyflow_cleaner.py" \
    --input-file "${RAW_DATA_DIR}/moneyflow/moneyflow.parquet" \
    --output-file "${CLEANED_DATA_DIR}/moneyflow/moneyflow.parquet" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_moneyflow_cleaning_inner.log"

run_step clean_fundamentals \
  python3 "${DATASET_CLEANER_PATH}/fundamentals_cleaner.py" \
    --input-file "${RAW_DATA_DIR}/fundamentals/fundamentals.parquet" \
    --output-file "${CLEANED_DATA_DIR}/fundamentals/fundamentals.parquet" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_fundamentals_cleaning_inner.log"

run_step clean_industry \
  python3 "${DATASET_CLEANER_PATH}/industry_cleaner.py" \
    --input-file "${RAW_DATA_DIR}/industry/industry.parquet" \
    --output-file "${CLEANED_DATA_DIR}/industry/industry.parquet" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_industry_cleaning_inner.log"

# ---------------------------------------------------------------------------
# 处理阶段：raw_data / cleaned_data -> processd_data
# ---------------------------------------------------------------------------

run_step process_namechange_st \
  python3 "${DATASET_PROCESSOR_PATH}/namechange_st_processor.py" \
    --input-file "${RAW_DATA_DIR}/namechange/namechange.parquet" \
    --output-file "${PROCESSED_DATA_DIR}/namechange/namechange_st_daily.parquet" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_namechange_st_processing_inner.log"

run_step process_suspend_d \
  python3 "${DATASET_PROCESSOR_PATH}/suspend_d_processor.py" \
    --input-file "${RAW_DATA_DIR}/suspend_d/suspend_d.parquet" \
    --output-file "${PROCESSED_DATA_DIR}/suspend_d/suspend_d_daily.parquet" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_suspend_d_processing_inner.log"

run_step process_daily_industry_onehot \
  python3 "${DATASET_PROCESSOR_PATH}/daily_industry_onehot_processor.py" \
    --input-dir "${CLEANED_DATA_DIR}/industry" \
    --output-dir "${PROCESSED_DATA_DIR}/industry/daily_onehot" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_daily_industry_onehot_processing_inner.log"

run_step process_industry_onehot \
  python3 "${DATASET_PROCESSOR_PATH}/industry_onehot_processor.py" \
    --input-file "${CLEANED_DATA_DIR}/industry/industry.parquet" \
    --output-file "${PROCESSED_DATA_DIR}/industry/industry_onehot.parquet" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_industry_onehot_processing_inner.log"

if [[ "${BUILD_WIDE_TABLE}" == "1" ]]; then
  run_step build_wide_table_daily_bars \
    python3 "${DATASET_PROCESSOR_PATH}/wide_table_daily_bars_builder.py" \
      --daily-bars-dir "${CLEANED_DATA_DIR}/daily_bars" \
      --namechange-file "${PROCESSED_DATA_DIR}/namechange/namechange_st_daily.parquet" \
      --suspend-file "${PROCESSED_DATA_DIR}/suspend_d/suspend_d_daily.parquet" \
      --adj-factor-file "${CLEANED_DATA_DIR}/adj_factors/adj_factors.parquet" \
      --fundamentals-file "${CLEANED_DATA_DIR}/fundamentals/fundamentals.parquet" \
      --moneyflow-file "${CLEANED_DATA_DIR}/moneyflow/moneyflow.parquet" \
      --industry-file "${PROCESSED_DATA_DIR}/industry/daily_onehot" \
      --output-dir "${PROCESSED_DATA_DIR}/wide_table_daily_bars" \
      "${WIDE_DATE_ARGS[@]}" \
      --log-level "${LOG_LEVEL}" \
      --log-file "${LOG_DIR}/${RUN_ID}_wide_table_daily_bars_building_inner.log"
else
  echo "跳过 wide_table_daily_bars 构建：BUILD_WIDE_TABLE=${BUILD_WIDE_TABLE}"
fi

if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
  echo "data process pipeline 完成但存在失败步骤：${FAILED_STEPS[*]}"
  echo "请查看 ${LOG_DIR}/${RUN_ID}_*.log 定位问题。修复后可重新执行同一条命令；清洗/处理输出会覆盖同名结果文件。"
  exit 1
fi

echo "data process pipeline 全部步骤成功完成。"
