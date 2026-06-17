#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATASET_FETCHER_PATH="${SCRIPT_DIR}/../../src/utils/dataset_fetcher"
OUTPUT_DIR="${SCRIPT_DIR}/../../data/raw_data"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/../../log/data_pipeline}"

# 用法：
#   data_pipeline.sh [start_date] [end_date]
#
# - 只传 start_date 时，只拉取当天。
# - 不传参数时，默认拉取今天。
# - 默认遇到单个 fetcher 失败会继续执行后续独立 fetcher，最后汇总失败步骤并返回非 0。
# - 失败后重新执行同一条命令即可补齐：各 fetcher 会根据 progress 文件 + 已落盘 parquet 跳过已完成分区/股票。
#
# 可选环境变量：
#   CONTINUE_ON_ERROR=0  遇到第一个失败步骤立即退出
#   FORCE_FETCH=1        给支持的 fetcher 追加 --force_fetch，强制重拉并覆盖同名 parquet
#   LOG_DIR=/path        覆盖 pipeline 日志目录

START_DATE="${1:-$(date +%Y%m%d)}"
END_DATE="${2:-${START_DATE}}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
FORCE_FETCH="${FORCE_FETCH:-0}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
FAILED_STEPS=()
declare -A STEP_STATUS=()

FORCE_ARGS=()
if [[ "${FORCE_FETCH}" == "1" ]]; then
  FORCE_ARGS+=(--force_fetch)
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

echo "data pipeline run_id=${RUN_ID}, start_date=${START_DATE}, end_date=${END_DATE}, output_dir=${OUTPUT_DIR}"

run_step daily_bars_fetch \
  python3 "${DATASET_FETCHER_PATH}/daily_bars_fetcher.py" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --output_dir "${OUTPUT_DIR}" \
    "${FORCE_ARGS[@]}"

run_step adj_factors_fetch \
  python3 "${DATASET_FETCHER_PATH}/adj_factors_fetcher.py" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --output_dir "${OUTPUT_DIR}" \
    "${FORCE_ARGS[@]}"

run_step moneyflow_fetch \
  python3 "${DATASET_FETCHER_PATH}/moneyflow_fetcher.py" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --output_dir "${OUTPUT_DIR}" \
    "${FORCE_ARGS[@]}"

run_step fundamentals_fetch \
  python3 "${DATASET_FETCHER_PATH}/fundamentals_fetcher.py" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --output_dir "${OUTPUT_DIR}" \
    "${FORCE_ARGS[@]}"

if [[ "${STEP_STATUS[fundamentals_fetch]:-1}" == "0" ]]; then
  run_step fundamentals_merge \
    python3 "${DATASET_FETCHER_PATH}/fundamentals_fetcher.py" \
      --merge_only \
      --output_dir "${OUTPUT_DIR}"
else
  echo "跳过 fundamentals_merge：fundamentals_fetch 未成功，避免生成不完整的合并文件。"
fi

run_step industry_fetch \
  python3 "${DATASET_FETCHER_PATH}/industry_fetcher.py" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --output_dir "${OUTPUT_DIR}" \
    "${FORCE_ARGS[@]}"

run_step suspend_d_fetch \
  python3 "${DATASET_FETCHER_PATH}/suspend_d_fetcher.py" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --output_dir "${OUTPUT_DIR}" \
    "${FORCE_ARGS[@]}"

run_step namechange_fetch \
  python3 "${DATASET_FETCHER_PATH}/namechange_fetcher.py" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --output_dir "${OUTPUT_DIR}" \
    "${FORCE_ARGS[@]}"

if [[ "${STEP_STATUS[namechange_fetch]:-1}" == "0" ]]; then
  run_step namechange_merge \
    python3 "${DATASET_FETCHER_PATH}/namechange_merger.py" \
      --input_dir "${OUTPUT_DIR}/namechange" \
      --output_dir "${OUTPUT_DIR}/namechange"
else
  echo "跳过 namechange_merge：namechange_fetch 未成功，避免生成不完整的合并文件。"
fi

if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
  echo "data pipeline 完成但存在失败步骤：${FAILED_STEPS[*]}"
  echo "可重新执行同一条命令补齐；默认不会重拉已完成的日期/股票。如需强制重拉，使用：FORCE_FETCH=1 $0 ${START_DATE} ${END_DATE}"
  exit 1
fi

echo "data pipeline 全部步骤成功完成。"
