#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# A 股监督学习标签生成流水线
# ============================================================================
#
# 前置条件：
#   已通过 data_fetch_pipeline.sh / data_process_pipeline.sh 准备好
#   data/processd_data/wide_table_daily_bars。
#
# 这个脚本负责从 processd_data 进入 generated_label：
#   - 使用 src/utils/label_generator 下的脚本生成标签；
#   - 当前生成 daily_labels，包括未来 1/2/3/5/10/20 个交易日收益率 label
#     以及同日横截面 percentile rank label；
#   - 输出保持 year=YYYY/month=MM/YYYYMMDD.parquet 分区结构，便于下游训练按日期加载。
#
# 用法：
#   bash scripts/data_pipeline/label_calculation_pipeline.sh [start_date] [end_date]
#
# 参数：
#   start_date/end_date 可选，格式 YYYYMMDD；不传则生成可用的全量日期。
#
# 示例：
#   # 全量生成标签
#   bash scripts/data_pipeline/label_calculation_pipeline.sh
#
#   # 只生成指定日期区间的标签
#   bash scripts/data_pipeline/label_calculation_pipeline.sh 20200101 20260618
#
# 常用环境变量：
#   CONTINUE_ON_ERROR=0        遇到第一个失败步骤立即退出；默认 1，继续执行后续独立步骤
#   GENERATE_DAILY_LABELS=0    跳过 daily_labels 生成；默认 1
#   OUTPUT_BATCH_SIZE=120      每批输出交易日文件数；调小可降低内存占用
#   LOG_DIR=/path              覆盖 pipeline 日志目录；默认 log/data_pipeline
#   LOG_LEVEL=DEBUG            传给 Python label generator 的日志级别；默认 INFO
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

LABEL_GENERATOR_PATH="${PROJECT_ROOT}/src/utils/label_generator"

PROCESSED_DATA_DIR="${PROJECT_ROOT}/data/processd_data"
GENERATED_LABEL_DIR="${PROJECT_ROOT}/data/generated_label"
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
GENERATE_DAILY_LABELS="${GENERATE_DAILY_LABELS:-1}"
OUTPUT_BATCH_SIZE="${OUTPUT_BATCH_SIZE:-120}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

mkdir -p "${GENERATED_LABEL_DIR}" "${LOG_DIR}"

# 让以脚本路径直接运行的 label_generator 也能稳定 import src.utils / utils。
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

echo "label calculation pipeline run_id=${RUN_ID}, start_date=${START_DATE:-ALL}, end_date=${END_DATE:-ALL}, generated_label=${GENERATED_LABEL_DIR}"

# ---------------------------------------------------------------------------
# 标签生成阶段：processd_data/wide_table_daily_bars -> generated_label/daily_labels
# ---------------------------------------------------------------------------

if [[ "${GENERATE_DAILY_LABELS}" == "1" ]]; then
  run_step generate_daily_labels \
    python3 "${LABEL_GENERATOR_PATH}/daily_label_generator.py" \
      --input-dir "${PROCESSED_DATA_DIR}/wide_table_daily_bars" \
      --output-dir "${GENERATED_LABEL_DIR}/daily_labels" \
      "${DATE_ARGS[@]}" \
      --output-batch-size "${OUTPUT_BATCH_SIZE}" \
      --log-level "${LOG_LEVEL}" \
      --log-file "${LOG_DIR}/${RUN_ID}_daily_labels_inner.log"
else
  echo "跳过 daily_labels 生成：GENERATE_DAILY_LABELS=${GENERATE_DAILY_LABELS}"
fi

if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
  echo "label calculation pipeline 完成但存在失败步骤：${FAILED_STEPS[*]}"
  echo "请查看 ${LOG_DIR}/${RUN_ID}_*.log 定位问题。修复后可重新执行同一条命令；标签输出会覆盖同名结果文件。"
  exit 1
fi

echo "label calculation pipeline 全部步骤成功完成。"
