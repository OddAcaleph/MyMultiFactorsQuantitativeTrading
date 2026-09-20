#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 增强 Alpha 因子生成与横截面处理流水线
# ============================================================================
#
# 生成进攻型 alpha 因子（动量突破、量价、资金流动量、基本面动量等），
# 并做横截面 winsorize + z-score 处理。
#
# 前置条件：
#   cleaned_data/daily_bars, cleaned_data/moneyflow, cleaned_data/fundamentals
#   已通过 data_fetch_pipeline.sh / data_process_pipeline.sh 准备好。
#
# 输出：
#   data/features_data/enhanced_alpha_factors/        原始因子
#   data/cross_sectional_processd_data/enhanced_alpha_factors/  横截面处理后
#
# 用法：
#   bash scripts/data_pipeline/enhanced_alpha_factors_pipeline.sh [start_date] [end_date]
#
# 常用环境变量：
#   GENERATE_ENHANCED=0      跳过因子生成，只做横截面处理；默认 1
#   CROSS_SECTIONAL_ENHANCED=0  跳过横截面处理；默认 1
#   PROCESS_FAIL_FAST=1      横截面处理单文件失败立即退出；默认 0
#   LOG_DIR=/path            覆盖日志目录
#   LOG_LEVEL=DEBUG          日志级别；默认 INFO
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

FEATURE_GENERATOR_PATH="${PROJECT_ROOT}/src/utils/features_generator"
CROSS_SECTIONAL_PROCESSOR_PATH="${PROJECT_ROOT}/src/utils/cross_sectional_processor"

CLEANED_DATA_DIR="${PROJECT_ROOT}/data/cleaned_data"
FEATURES_DATA_DIR="${PROJECT_ROOT}/data/features_data"
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

GENERATE_ENHANCED="${GENERATE_ENHANCED:-1}"
CROSS_SECTIONAL_ENHANCED="${CROSS_SECTIONAL_ENHANCED:-1}"
PROCESS_FAIL_FAST="${PROCESS_FAIL_FAST:-0}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

mkdir -p "${FEATURES_DATA_DIR}" "${CROSS_SECTIONAL_DATA_DIR}" "${LOG_DIR}"
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

FAIL_FAST_ARGS=()
if [[ "${PROCESS_FAIL_FAST}" == "1" ]]; then
  FAIL_FAST_ARGS+=(--fail-fast)
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

echo "enhanced alpha factors pipeline run_id=${RUN_ID}, start_date=${START_DATE:-ALL}, end_date=${END_DATE:-ALL}"

# ---------------------------------------------------------------------------
# 阶段一：增强 alpha 因子生成
# ---------------------------------------------------------------------------

if [[ "${GENERATE_ENHANCED}" == "1" ]]; then
  run_step generate_enhanced_alpha_factors \
    python3 "${FEATURE_GENERATOR_PATH}/enhanced_alpha_feature_generator.py" \
      --input-dir "${CLEANED_DATA_DIR}/daily_bars" \
      --moneyflow-file "${CLEANED_DATA_DIR}/moneyflow/moneyflow.parquet" \
      --fundamentals-file "${CLEANED_DATA_DIR}/fundamentals/fundamentals.parquet" \
      --industry-file "${CLEANED_DATA_DIR}/industry/industry.parquet" \
      --output-dir "${FEATURES_DATA_DIR}/enhanced_alpha_factors" \
      "${DATE_ARGS[@]}" \
      --log-level "${LOG_LEVEL}" \
      --log-file "${LOG_DIR}/${RUN_ID}_enhanced_alpha_factors_inner.log"
else
  echo "跳过增强alpha因子生成：GENERATE_ENHANCED=${GENERATE_ENHANCED}"
fi

# ---------------------------------------------------------------------------
# 阶段二：横截面处理
# ---------------------------------------------------------------------------

if [[ "${CROSS_SECTIONAL_ENHANCED}" == "1" ]]; then
  if [[ "${GENERATE_ENHANCED}" != "1" || "${STEP_STATUS[generate_enhanced_alpha_factors]:-0}" == "0" ]]; then
    run_step cross_sectional_enhanced_alpha_factors \
      python3 "${CROSS_SECTIONAL_PROCESSOR_PATH}/enhanced_alpha_factors_cross_sectional_processor.py" \
        --input-dir "${FEATURES_DATA_DIR}/enhanced_alpha_factors" \
        --output-dir "${CROSS_SECTIONAL_DATA_DIR}/enhanced_alpha_factors" \
        --industry-file "${CLEANED_DATA_DIR}/industry/industry.parquet" \
        "${DATE_ARGS[@]}" \
        "${FAIL_FAST_ARGS[@]}" \
        --log-level "${LOG_LEVEL}" \
        --log-file "${LOG_DIR}/${RUN_ID}_enhanced_alpha_cross_sectional_inner.log"
  else
    echo "跳过横截面处理：增强alpha因子生成未成功。"
  fi
else
  echo "跳过横截面处理：CROSS_SECTIONAL_ENHANCED=${CROSS_SECTIONAL_ENHANCED}"
fi

if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
  echo "enhanced alpha factors pipeline 完成但存在失败步骤：${FAILED_STEPS[*]}"
  echo "请查看 ${LOG_DIR}/${RUN_ID}_*.log 定位问题。"
  exit 1
fi

echo "enhanced alpha factors pipeline 全部步骤成功完成。"
echo "因子输出：${FEATURES_DATA_DIR}/enhanced_alpha_factors"
echo "横截面输出：${CROSS_SECTIONAL_DATA_DIR}/enhanced_alpha_factors"
