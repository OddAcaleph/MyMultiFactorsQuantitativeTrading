#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# A 股日常端到端数据流水线
# ============================================================================
#
# 这个脚本串联每日数据生产的完整链路：
#   1. data_fetch_pipeline.sh             raw_data 拉取/合并
#   2. data_process_pipeline.sh           cleaned_data / processd_data 生成
#   3. factor_calculation_pipeline.sh     features_data / 横截面数据生成
#   4. label_calculation_pipeline.sh      generated_label 生成
#   5. run_xgboost_inference.py           运行最新模型推理
#
# 用法：
#   bash scripts/data_pipeline/daily_data_pipeline.sh [start_date] [end_date]
#
# 日期格式：
#   - 支持 YYYYMMDD 或 YYYY-MM-DD；脚本会自动规范化。
#   - 不传日期时默认跑今天。
#   - 只传 start_date 时默认 start_date=end_date，适合每日增量。
#
# 示例：
#   # 跑今天的完整链路
#   bash scripts/data_pipeline/daily_data_pipeline.sh
#
#   # 跑单日
#   bash scripts/data_pipeline/daily_data_pipeline.sh 20260623
#
#   # 跑闭区间
#   bash scripts/data_pipeline/daily_data_pipeline.sh 20260601 20260623
#
# 常用环境变量：
#   MASTER_CONTINUE_ON_ERROR=1  总控脚本单阶段失败后继续跑后续阶段；默认 0，失败即停
#   RUN_FETCH=0                 跳过数据拉取；默认 1
#   RUN_PROCESS=0               跳过清洗处理；默认 1
#   RUN_FACTORS=0               跳过因子计算/横截面处理；默认 1
#   RUN_LABELS=0                跳过标签生成；默认 1
#   RUN_INFERENCE=0             跳过 XGBoost 推理；默认 1
#   INFERENCE_START_DATE=日期   额外给推理传 --start-time；默认不传，仅覆盖 --end-time
#   INFERENCE_CONFIG=/path      覆盖推理配置文件
#   INFERENCE_MODEL_PATH=/path  覆盖模型路径
#   INFERENCE_OUTPUT_DIR=/path  覆盖推理输出目录
#   INFERENCE_SEGMENT=test      覆盖推理 segment
#   INFERENCE_CPU=1             强制 CPU 推理
#   INFERENCE_NO_LABEL=1        推理时不加载 label
#   LOG_DIR=/path               覆盖日志目录；默认 log/data_pipeline
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/log/data_pipeline}"

MASTER_CONTINUE_ON_ERROR="${MASTER_CONTINUE_ON_ERROR:-0}"
RUN_FETCH="${RUN_FETCH:-1}"
RUN_PROCESS="${RUN_PROCESS:-1}"
RUN_FACTORS="${RUN_FACTORS:-1}"
RUN_LABELS="${RUN_LABELS:-1}"
RUN_INFERENCE="${RUN_INFERENCE:-1}"

mkdir -p "${LOG_DIR}"
export LOG_DIR

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
FAILED_STAGES=()
STAGES_RUN=()
declare -A STAGE_STATUS=()
STAGE_INDEX=0
TOTAL_STAGES=0

for enabled in "${RUN_FETCH}" "${RUN_PROCESS}" "${RUN_FACTORS}" "${RUN_LABELS}" "${RUN_INFERENCE}"; do
  if [[ "${enabled}" == "1" ]]; then
    TOTAL_STAGES=$((TOTAL_STAGES + 1))
  fi
done

usage() {
  sed -n '1,55p' "$0" | sed 's/^# \{0,1\}//'
}

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

to_hyphen_date() {
  local compact_date="$1"
  echo "${compact_date:0:4}-${compact_date:4:2}-${compact_date:6:2}"
}

format_command() {
  printf '%q ' "$@"
  printf '\n'
}

run_stage() {
  local stage_name="$1"
  shift
  local log_file="${LOG_DIR}/${RUN_ID}_${stage_name}.log"
  local command_text
  local start_ts
  local end_ts
  local elapsed

  STAGE_INDEX=$((STAGE_INDEX + 1))
  STAGES_RUN+=("${stage_name}")
  command_text="$(format_command "$@")"
  start_ts="$(date +%s)"

  echo ""
  echo "============================================================================"
  echo "[$(date '+%F %T')] 当前阶段：[${STAGE_INDEX}/${TOTAL_STAGES}] ${stage_name}"
  echo "阶段日志：${log_file}"
  echo "日志目录：${LOG_DIR}"
  echo "执行命令：${command_text}"
  echo "============================================================================"
  set +e
  "$@" 2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e
  end_ts="$(date +%s)"
  elapsed=$((end_ts - start_ts))

  STAGE_STATUS["${stage_name}"]="${status}"
  if [[ ${status} -ne 0 ]]; then
    FAILED_STAGES+=("${stage_name}")
    echo "[$(date '+%F %T')] FAIL  [${STAGE_INDEX}/${TOTAL_STAGES}] ${stage_name}，退出码：${status}，耗时：${elapsed}s"
    echo "失败阶段日志：${log_file}"
    if [[ "${MASTER_CONTINUE_ON_ERROR}" != "1" ]]; then
      echo "MASTER_CONTINUE_ON_ERROR=${MASTER_CONTINUE_ON_ERROR}，总控脚本停止。"
      exit "${status}"
    fi
    return 0
  fi

  echo "[$(date '+%F %T')] OK    [${STAGE_INDEX}/${TOTAL_STAGES}] ${stage_name}，耗时：${elapsed}s"
  echo "阶段日志已保存：${log_file}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

START_DATE="$(normalize_date_compact "${1:-$(date +%Y%m%d)}")"
END_DATE="$(normalize_date_compact "${2:-${START_DATE}}")"
END_TIME="$(to_hyphen_date "${END_DATE}")"

if [[ "${START_DATE}" > "${END_DATE}" ]]; then
  echo "日期区间错误：start_date=${START_DATE} 晚于 end_date=${END_DATE}" >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

echo ""
echo "============================================================================"
echo "daily data pipeline 启动"
echo "run_id: ${RUN_ID}"
echo "项目根目录: ${PROJECT_ROOT}"
echo "日期范围: ${START_DATE} ~ ${END_DATE}"
echo "推理 end-time: ${END_TIME}"
echo "日志目录: ${LOG_DIR}"
echo "阶段日志命名: ${LOG_DIR}/${RUN_ID}_<stage>.log"
echo "启用阶段数: ${TOTAL_STAGES}"
echo "阶段开关: RUN_FETCH=${RUN_FETCH}, RUN_PROCESS=${RUN_PROCESS}, RUN_FACTORS=${RUN_FACTORS}, RUN_LABELS=${RUN_LABELS}, RUN_INFERENCE=${RUN_INFERENCE}"
echo "失败策略: MASTER_CONTINUE_ON_ERROR=${MASTER_CONTINUE_ON_ERROR}"
echo "============================================================================"

if [[ "${RUN_FETCH}" == "1" ]]; then
  run_stage data_fetch \
    bash "${SCRIPT_DIR}/data_fetch_pipeline.sh" "${START_DATE}" "${END_DATE}"
else
  echo "跳过阶段 data_fetch：RUN_FETCH=${RUN_FETCH}，日志目录仍为 ${LOG_DIR}"
fi

if [[ "${RUN_PROCESS}" == "1" ]]; then
  run_stage data_process \
    bash "${SCRIPT_DIR}/data_process_pipeline.sh" "${START_DATE}" "${END_DATE}"
else
  echo "跳过阶段 data_process：RUN_PROCESS=${RUN_PROCESS}，日志目录仍为 ${LOG_DIR}"
fi

if [[ "${RUN_FACTORS}" == "1" ]]; then
  run_stage factor_calculation \
    bash "${SCRIPT_DIR}/factor_calculation_pipeline.sh" "${START_DATE}" "${END_DATE}"
else
  echo "跳过阶段 factor_calculation：RUN_FACTORS=${RUN_FACTORS}，日志目录仍为 ${LOG_DIR}"
fi

if [[ "${RUN_LABELS}" == "1" ]]; then
  run_stage label_calculation \
    bash "${SCRIPT_DIR}/label_calculation_pipeline.sh" "${START_DATE}" "${END_DATE}"
else
  echo "跳过阶段 label_calculation：RUN_LABELS=${RUN_LABELS}，日志目录仍为 ${LOG_DIR}"
fi

if [[ "${RUN_INFERENCE}" == "1" ]]; then
  INFERENCE_ARGS=(
    python3 "${PROJECT_ROOT}/src/trainer/run_xgboost_inference.py"
    --end-time "${END_TIME}"
  )

  if [[ -n "${INFERENCE_START_DATE:-}" ]]; then
    INFERENCE_START_COMPACT="$(normalize_date_compact "${INFERENCE_START_DATE}")"
    INFERENCE_ARGS+=(--start-time "$(to_hyphen_date "${INFERENCE_START_COMPACT}")")
  fi
  if [[ -n "${INFERENCE_CONFIG:-}" ]]; then
    INFERENCE_ARGS+=(--config "${INFERENCE_CONFIG}")
  fi
  if [[ -n "${INFERENCE_MODEL_PATH:-}" ]]; then
    INFERENCE_ARGS+=(--model-path "${INFERENCE_MODEL_PATH}")
  fi
  if [[ -n "${INFERENCE_OUTPUT_DIR:-}" ]]; then
    INFERENCE_ARGS+=(--output-dir "${INFERENCE_OUTPUT_DIR}")
  fi
  if [[ -n "${INFERENCE_SEGMENT:-}" ]]; then
    INFERENCE_ARGS+=(--segment "${INFERENCE_SEGMENT}")
  fi
  if [[ "${INFERENCE_CPU:-0}" == "1" ]]; then
    INFERENCE_ARGS+=(--cpu)
  fi
  if [[ "${INFERENCE_NO_LABEL:-0}" == "1" ]]; then
    INFERENCE_ARGS+=(--no-label)
  fi

  run_stage xgboost_inference "${INFERENCE_ARGS[@]}"
else
  echo "跳过阶段 xgboost_inference：RUN_INFERENCE=${RUN_INFERENCE}，日志目录仍为 ${LOG_DIR}"
fi

echo ""
echo "============================================================================"
echo "daily data pipeline 阶段汇总"
echo "run_id: ${RUN_ID}"
echo "日志目录: ${LOG_DIR}"
if [[ ${#STAGES_RUN[@]} -eq 0 ]]; then
  echo "本次没有启用任何阶段。"
else
  for stage_name in "${STAGES_RUN[@]}"; do
    echo "- ${stage_name}: exit_code=${STAGE_STATUS[${stage_name}]:-unknown}, log=${LOG_DIR}/${RUN_ID}_${stage_name}.log"
  done
fi
echo "============================================================================"

if [[ ${#FAILED_STAGES[@]} -gt 0 ]]; then
  echo "daily data pipeline 完成但存在失败阶段：${FAILED_STAGES[*]}"
  echo "请查看 ${LOG_DIR}/${RUN_ID}_*.log 定位问题。修复后可重新执行同一条命令；各子流水线具备幂等/覆盖能力。"
  exit 1
fi

echo "daily data pipeline 全部阶段成功完成。"
