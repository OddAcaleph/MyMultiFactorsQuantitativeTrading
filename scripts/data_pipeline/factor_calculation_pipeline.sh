#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# A 股特征因子生成与横截面处理流水线
# ============================================================================
#
# 前置条件：
#   已通过 data_fetch_pipeline.sh / data_process_pipeline.sh 准备好 cleaned_data。
#
# 这个脚本负责从 cleaned_data / features_data 进入：
#   1. features_data
#      - 使用 src/utils/features_generator 下的脚本生成特征因子；
#      - 价格量、财务、资金流因子直接从 cleaned_data 生成；
#      - 行业因子依赖已生成的价格量与财务因子。
#   2. cross_sectional_processd_data
#      - 使用 src/utils/cross_sectional_processor 下的工具；
#      - 对本脚本生成的各类特征因子做横截面 winsorize + z-score 处理；
#      - 同时补齐 processd_data/wide_table_daily_bars，并处理成训练依赖的
#        cross_sectional_processd_data/wide_table_daily_bars。
#
# 用法：
#   bash scripts/data_pipeline/factor_calculation_pipeline.sh [start_date] [end_date]
#
# 参数：
#   start_date/end_date 可选，格式 YYYYMMDD；不传则处理可用的全量日期。
#
# 示例：
#   # 全量生成并横截面处理
#   bash scripts/data_pipeline/factor_calculation_pipeline.sh
#
#   # 只处理指定日期区间
#   bash scripts/data_pipeline/factor_calculation_pipeline.sh 20200101 20260618
#
# 常用环境变量：
#   CONTINUE_ON_ERROR=0             遇到第一个失败步骤立即退出；默认 1，继续执行后续独立步骤
#   GENERATE_FEATURES=0             跳过特征因子生成；默认 1
#   BUILD_WIDE_TABLE_DAILY_BARS=0   跳过 processd_data/wide_table_daily_bars 构建；默认 1
#   CROSS_SECTIONAL_PROCESS=0       跳过横截面处理；默认 1
#   CROSS_SECTIONAL_WIDE_TABLE=0    跳过 wide_table_daily_bars 横截面处理；默认 1
#   PROCESS_FAIL_FAST=1             横截面处理遇到单文件错误立即失败；默认 0
#   AMOUNT_UNIT_SCALE=0.1           资金流因子中 daily_bars.amount 的单位缩放；默认 0.1
#   LOG_DIR=/path                   覆盖 pipeline 日志目录；默认 log/data_pipeline
#   LOG_LEVEL=DEBUG                 传给 Python generator/processor 的日志级别；默认 INFO
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

FEATURE_GENERATOR_PATH="${PROJECT_ROOT}/src/utils/features_generator"
CROSS_SECTIONAL_PROCESSOR_PATH="${PROJECT_ROOT}/src/utils/cross_sectional_processor"
DATASET_PROCESSOR_PATH="${PROJECT_ROOT}/src/utils/dataset_processor"

CLEANED_DATA_DIR="${PROJECT_ROOT}/data/cleaned_data"
FEATURES_DATA_DIR="${PROJECT_ROOT}/data/features_data"
PROCESSED_DATA_DIR="${PROJECT_ROOT}/data/processd_data"
CROSS_SECTIONAL_DATA_DIR="${PROJECT_ROOT}/data/cross_sectional_processd_data"
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
GENERATE_FEATURES="${GENERATE_FEATURES:-1}"
BUILD_WIDE_TABLE_DAILY_BARS="${BUILD_WIDE_TABLE_DAILY_BARS:-1}"
CROSS_SECTIONAL_PROCESS="${CROSS_SECTIONAL_PROCESS:-1}"
CROSS_SECTIONAL_WIDE_TABLE="${CROSS_SECTIONAL_WIDE_TABLE:-1}"
PROCESS_FAIL_FAST="${PROCESS_FAIL_FAST:-0}"
AMOUNT_UNIT_SCALE="${AMOUNT_UNIT_SCALE:-0.1}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

mkdir -p "${FEATURES_DATA_DIR}" "${CROSS_SECTIONAL_DATA_DIR}" "${LOG_DIR}"

# 让以脚本路径直接运行的 cross_sectional_processor 也能稳定 import src.utils / utils。
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
FAILED_STEPS=()
declare -A STEP_STATUS=()

DATE_ARGS=()
if [[ -n "${START_DATE}" ]]; then
  DATE_ARGS+=(--start-date "${START_DATE}")
fi
if [[ -n "${END_DATE}" ]]; then
  DATE_ARGS+=(--end-date "${END_DATE}")
fi

PROCESS_FAIL_FAST_ARGS=()
if [[ "${PROCESS_FAIL_FAST}" == "1" ]]; then
  PROCESS_FAIL_FAST_ARGS+=(--fail-fast)
fi

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

echo "factor calculation pipeline run_id=${RUN_ID}, start_date=${START_DATE:-ALL}, end_date=${END_DATE:-ALL}, features_data=${FEATURES_DATA_DIR}, cross_sectional_data=${CROSS_SECTIONAL_DATA_DIR}"

# ---------------------------------------------------------------------------
# 特征因子生成阶段：cleaned_data -> features_data
# ---------------------------------------------------------------------------

if [[ "${GENERATE_FEATURES}" == "1" ]]; then
  run_step generate_price_volume_factors \
    python3 "${FEATURE_GENERATOR_PATH}/price_volume_feature_generator.py" \
      --input-dir "${CLEANED_DATA_DIR}/daily_bars" \
      --output-dir "${FEATURES_DATA_DIR}/price_volume_factors" \
      "${DATE_ARGS[@]}" \
      --log-level "${LOG_LEVEL}" \
      --log-file "${LOG_DIR}/${RUN_ID}_price_volume_features_inner.log"

  run_step generate_fundamental_factors \
    python3 "${FEATURE_GENERATOR_PATH}/fundamental_feature_generator.py" \
      --fundamentals-file "${CLEANED_DATA_DIR}/fundamentals/fundamentals.parquet" \
      --daily-bars-dir "${CLEANED_DATA_DIR}/daily_bars" \
      --output-dir "${FEATURES_DATA_DIR}/fundamental_factors" \
      "${DATE_ARGS[@]}" \
      --log-level "${LOG_LEVEL}" \
      --log-file "${LOG_DIR}/${RUN_ID}_fundamental_features_inner.log"

  run_step generate_moneyflow_factors \
    python3 "${FEATURE_GENERATOR_PATH}/moneyflow_feature_generator.py" \
      --moneyflow-file "${CLEANED_DATA_DIR}/moneyflow/moneyflow.parquet" \
      --daily-bars-dir "${CLEANED_DATA_DIR}/daily_bars" \
      --output-dir "${FEATURES_DATA_DIR}/moneyflow_factors" \
      --amount-unit-scale "${AMOUNT_UNIT_SCALE}" \
      "${DATE_ARGS[@]}" \
      --log-level "${LOG_LEVEL}" \
      --log-file "${LOG_DIR}/${RUN_ID}_moneyflow_features_inner.log"

  # 行业因子依赖 price_volume_factors 与 fundamental_factors。
  if [[ "${STEP_STATUS[generate_price_volume_factors]:-1}" == "0" && "${STEP_STATUS[generate_fundamental_factors]:-1}" == "0" ]]; then
    run_step generate_industry_factors \
      python3 "${FEATURE_GENERATOR_PATH}/industry_feature_generator.py" \
        --industry-file "${CLEANED_DATA_DIR}/industry/industry.parquet" \
        --price-volume-dir "${FEATURES_DATA_DIR}/price_volume_factors" \
        --fundamental-dir "${FEATURES_DATA_DIR}/fundamental_factors" \
        --output-dir "${FEATURES_DATA_DIR}/industry_factors" \
        "${DATE_ARGS[@]}" \
        --log-level "${LOG_LEVEL}" \
        --log-file "${LOG_DIR}/${RUN_ID}_industry_features_inner.log"
  else
    echo "跳过 generate_industry_factors：依赖的 price_volume/fundamental 因子生成未全部成功。"
  fi
else
  echo "跳过特征因子生成：GENERATE_FEATURES=${GENERATE_FEATURES}"
fi

# ---------------------------------------------------------------------------
# Daily 宽表补齐阶段：cleaned_data/processd_data -> processd_data/wide_table_daily_bars
# ---------------------------------------------------------------------------

if [[ "${BUILD_WIDE_TABLE_DAILY_BARS}" == "1" ]]; then
  run_step build_wide_table_daily_bars \
    python3 "${DATASET_PROCESSOR_PATH}/wide_table_daily_bars_builder.py" \
      --daily-bars-dir "${CLEANED_DATA_DIR}/daily_bars" \
      --namechange-file "${PROCESSED_DATA_DIR}/namechange/namechange_st_daily.parquet" \
      --suspend-file "${PROCESSED_DATA_DIR}/suspend_d/suspend_d_daily.parquet" \
      --adj-factor-file "${CLEANED_DATA_DIR}/adj_factors/adj_factors.parquet" \
      --fundamentals-file "${CLEANED_DATA_DIR}/fundamentals/fundamentals.parquet" \
      --moneyflow-file "${CLEANED_DATA_DIR}/moneyflow/moneyflow.parquet" \
      --industry-file "${PROCESSED_DATA_DIR}/industry/industry_onehot.parquet" \
      --output-dir "${PROCESSED_DATA_DIR}/wide_table_daily_bars" \
      "${DATE_ARGS[@]}" \
      --log-level "${LOG_LEVEL}" \
      --log-file "${LOG_DIR}/${RUN_ID}_wide_table_daily_bars_building_inner.log"
else
  echo "跳过 processd_data/wide_table_daily_bars 构建：BUILD_WIDE_TABLE_DAILY_BARS=${BUILD_WIDE_TABLE_DAILY_BARS}"
fi

# ---------------------------------------------------------------------------
# 横截面处理阶段：features_data -> cross_sectional_processd_data
# ---------------------------------------------------------------------------

if [[ "${CROSS_SECTIONAL_PROCESS}" == "1" ]]; then
  if [[ "${CROSS_SECTIONAL_WIDE_TABLE}" == "1" ]]; then
    run_step cross_sectional_wide_table_daily_bars \
      python3 "${CROSS_SECTIONAL_PROCESSOR_PATH}/wide_table_daily_bars_cross_sectional_processor.py" \
        --input-dir "${PROCESSED_DATA_DIR}/wide_table_daily_bars" \
        --output-dir "${CROSS_SECTIONAL_DATA_DIR}/wide_table_daily_bars" \
        "${DATE_ARGS[@]}" \
        "${PROCESS_FAIL_FAST_ARGS[@]}" \
        --log-level "${LOG_LEVEL}" \
        --log-file "${LOG_DIR}/${RUN_ID}_wide_table_daily_bars_cross_sectional_inner.log"
  else
    echo "跳过 wide_table_daily_bars 横截面处理：CROSS_SECTIONAL_WIDE_TABLE=${CROSS_SECTIONAL_WIDE_TABLE}"
  fi

  if [[ "${GENERATE_FEATURES}" != "1" || "${STEP_STATUS[generate_price_volume_factors]:-0}" == "0" ]]; then
    run_step cross_sectional_price_volume_factors \
      python3 "${CROSS_SECTIONAL_PROCESSOR_PATH}/price_volume_factors_cross_sectional_processor.py" \
        --input-dir "${FEATURES_DATA_DIR}/price_volume_factors" \
        --output-dir "${CROSS_SECTIONAL_DATA_DIR}/price_volume_factors" \
        "${DATE_ARGS[@]}" \
        "${PROCESS_FAIL_FAST_ARGS[@]}" \
        --log-level "${LOG_LEVEL}" \
        --log-file "${LOG_DIR}/${RUN_ID}_price_volume_factors_cross_sectional_inner.log"
  else
    echo "跳过 cross_sectional_price_volume_factors：price_volume 因子生成未成功。"
  fi

  if [[ "${GENERATE_FEATURES}" != "1" || "${STEP_STATUS[generate_fundamental_factors]:-0}" == "0" ]]; then
    run_step cross_sectional_fundamental_factors \
      python3 "${CROSS_SECTIONAL_PROCESSOR_PATH}/fundamental_factors_cross_sectional_processor.py" \
        --input-dir "${FEATURES_DATA_DIR}/fundamental_factors" \
        --output-dir "${CROSS_SECTIONAL_DATA_DIR}/fundamental_factors" \
        "${DATE_ARGS[@]}" \
        "${PROCESS_FAIL_FAST_ARGS[@]}" \
        --log-level "${LOG_LEVEL}" \
        --log-file "${LOG_DIR}/${RUN_ID}_fundamental_factors_cross_sectional_inner.log"
  else
    echo "跳过 cross_sectional_fundamental_factors：fundamental 因子生成未成功。"
  fi

  if [[ "${GENERATE_FEATURES}" != "1" || "${STEP_STATUS[generate_moneyflow_factors]:-0}" == "0" ]]; then
    run_step cross_sectional_moneyflow_factors \
      python3 "${CROSS_SECTIONAL_PROCESSOR_PATH}/moneyflow_factors_cross_sectional_processor.py" \
        --input-dir "${FEATURES_DATA_DIR}/moneyflow_factors" \
        --output-dir "${CROSS_SECTIONAL_DATA_DIR}/moneyflow_factors" \
        "${DATE_ARGS[@]}" \
        "${PROCESS_FAIL_FAST_ARGS[@]}" \
        --log-level "${LOG_LEVEL}" \
        --log-file "${LOG_DIR}/${RUN_ID}_moneyflow_factors_cross_sectional_inner.log"
  else
    echo "跳过 cross_sectional_moneyflow_factors：moneyflow 因子生成未成功。"
  fi

  if [[ "${GENERATE_FEATURES}" != "1" || "${STEP_STATUS[generate_industry_factors]:-0}" == "0" ]]; then
    run_step cross_sectional_industry_factors \
      python3 "${CROSS_SECTIONAL_PROCESSOR_PATH}/industry_factors_cross_sectional_processor.py" \
        --input-dir "${FEATURES_DATA_DIR}/industry_factors" \
        --output-dir "${CROSS_SECTIONAL_DATA_DIR}/industry_factors" \
        "${DATE_ARGS[@]}" \
        "${PROCESS_FAIL_FAST_ARGS[@]}" \
        --log-level "${LOG_LEVEL}" \
        --log-file "${LOG_DIR}/${RUN_ID}_industry_factors_cross_sectional_inner.log"
  else
    echo "跳过 cross_sectional_industry_factors：industry 因子生成未成功。"
  fi
else
  echo "跳过横截面处理：CROSS_SECTIONAL_PROCESS=${CROSS_SECTIONAL_PROCESS}"
fi

if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
  echo "factor calculation pipeline 完成但存在失败步骤：${FAILED_STEPS[*]}"
  echo "请查看 ${LOG_DIR}/${RUN_ID}_*.log 定位问题。修复后可重新执行同一条命令；特征与横截面输出会覆盖同名结果文件。"
  exit 1
fi

echo "factor calculation pipeline 全部步骤成功完成。"
