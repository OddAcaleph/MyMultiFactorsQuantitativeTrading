#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# A 股原始数据拉取流水线
# ============================================================================
#
# 这个脚本只负责「raw_data」层的数据拉取与必要的 raw 合并，不做清洗、特征处理
# 或宽表构建。后续清洗/处理请运行同目录下的 data_process_pipeline.sh。
#
# 数据落盘约定：
#   - 日频/按交易日接口：按 year=YYYY/month=MM/YYYYMMDD.parquet 分区落盘
#       daily_bars、adj_factors、moneyflow、industry、suspend_d
#   - 按股票接口：先按 <ts_code>.parquet 落盘，再合并成大表
#       fundamentals -> fundamentals/fundamentals.parquet
#       namechange   -> namechange/namechange.parquet
#
# 幂等/断点续跑：
#   - fetcher 会结合 progress json 与已落盘 parquet 跳过已完成日期/股票。
#   - fundamentals/namechange 支持基于历史最大日期的增量更新。
#   - 如需完全重拉，可设置 FORCE_FETCH=1。
#
# 用法：
#   bash scripts/data_pipeline/data_fetch_pipeline.sh [start_date] [end_date]
#
# 示例：
#   # 默认拉取今天
#   bash scripts/data_pipeline/data_fetch_pipeline.sh
#
#   # 拉取单日
#   bash scripts/data_pipeline/data_fetch_pipeline.sh 20260618
#
#   # 拉取日期区间
#   bash scripts/data_pipeline/data_fetch_pipeline.sh 20200101 20260618
#
# 常用环境变量：
#   CONTINUE_ON_ERROR=0  遇到第一个失败步骤立即退出；默认 1，继续执行后续独立步骤
#   FORCE_FETCH=1        给支持的 fetcher 追加 --force_fetch，强制重拉并覆盖同名 parquet
#   LOG_DIR=/path        覆盖 pipeline 日志目录；默认 log/data_pipeline
# ============================================================================

# 脚本所在目录。后续路径均基于该目录拼接，保证无论从哪个工作目录执行都能定位项目文件。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Python fetcher 所在目录。
DATASET_FETCHER_PATH="${PROJECT_ROOT}/src/utils/dataset_fetcher"

# raw_data 根目录。所有原始数据都写到这里，后续清洗脚本也默认从这里读取。
OUTPUT_DIR="${PROJECT_ROOT}/data/raw_data"

# 每个步骤会把 stdout/stderr tee 到独立日志文件，便于失败后定位问题。
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

# 日期参数：
#   - 不传：默认今天
#   - 只传 start_date：只拉取当天
#   - 传 start_date end_date：拉取闭区间
START_DATE="$(normalize_date_compact "${1:-$(date +%Y%m%d)}")"
END_DATE="$(normalize_date_compact "${2:-${START_DATE}}")"
if [[ "${START_DATE}" > "${END_DATE}" ]]; then
  echo "日期区间错误：start_date=${START_DATE} 晚于 end_date=${END_DATE}" >&2
  exit 1
fi

# 失败控制：默认继续跑后续步骤，最终统一返回失败；适合长流水线尽量多产出可用数据。
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"

# 强制重拉控制：默认 0，复用各 fetcher 的断点续跑能力。
FORCE_FETCH="${FORCE_FETCH:-0}"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

# run_id 用于给一轮流水线的所有日志打同一个时间戳前缀。
RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
FAILED_STEPS=()
declare -A STEP_STATUS=()

# FORCE_ARGS 会按需传给各 fetcher。用数组避免空参数/空字符串被错误解析。
FORCE_ARGS=()
if [[ "${FORCE_FETCH}" == "1" ]]; then
  FORCE_ARGS+=(--force_fetch)
fi

# 统一的步骤执行器：
#   1. 打印开始/结束日志；
#   2. 将命令输出 tee 到步骤日志；
#   3. 捕获被 tee 包裹命令的真实退出码；
#   4. 根据 CONTINUE_ON_ERROR 决定继续还是立即退出。
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

echo "data fetch pipeline run_id=${RUN_ID}, start_date=${START_DATE}, end_date=${END_DATE}, output_dir=${OUTPUT_DIR}"

# 1) 日线行情：按交易日拉取全市场 daily，输出到 raw_data/daily_bars/year=*/month=*/*.parquet。
run_step daily_bars_fetch \
  python3 "${DATASET_FETCHER_PATH}/daily_bars_fetcher.py" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --output_dir "${OUTPUT_DIR}" \
    "${FORCE_ARGS[@]}"

# 2) 复权因子：按交易日拉取 adj_factor，输出到 raw_data/adj_factors/year=*/month=*/*.parquet。
run_step adj_factors_fetch \
  python3 "${DATASET_FETCHER_PATH}/adj_factors_fetcher.py" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --output_dir "${OUTPUT_DIR}" \
    "${FORCE_ARGS[@]}"

# 3) 资金流：按交易日拉取 moneyflow，输出到 raw_data/moneyflow/year=*/month=*/*.parquet。
run_step moneyflow_fetch \
  python3 "${DATASET_FETCHER_PATH}/moneyflow_fetcher.py" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --output_dir "${OUTPUT_DIR}" \
    "${FORCE_ARGS[@]}"

# 4) 财务指标：fina_indicator。优先用历史水位做增量，按股票文件落盘。
run_step fundamentals_fetch \
  python3 "${DATASET_FETCHER_PATH}/fundamentals_fetcher.py" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --output_dir "${OUTPUT_DIR}" \
    "${FORCE_ARGS[@]}"

# 4.1) 财务指标 raw 合并：只有拉取成功才合并，避免将不完整增量误合并给下游使用。
if [[ "${STEP_STATUS[fundamentals_fetch]:-1}" == "0" ]]; then
  run_step fundamentals_merge \
    python3 "${DATASET_FETCHER_PATH}/fundamentals_fetcher.py" \
      --merge_only \
      --output_dir "${OUTPUT_DIR}"
else
  echo "跳过 fundamentals_merge：fundamentals_fetch 未成功，避免生成不完整的合并文件。"
fi

# 5) 行业分类：生成每日行业快照，输出到 raw_data/industry/year=*/month=*/*.parquet。
run_step industry_fetch \
  python3 "${DATASET_FETCHER_PATH}/industry_fetcher.py" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --output_dir "${OUTPUT_DIR}" \
    "${FORCE_ARGS[@]}"

# 6) 停复牌：按交易日拉取 suspend_d，输出到 raw_data/suspend_d/year=*/month=*/*.parquet。
run_step suspend_d_fetch \
  python3 "${DATASET_FETCHER_PATH}/suspend_d_fetcher.py" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --output_dir "${OUTPUT_DIR}" \
    "${FORCE_ARGS[@]}"

# 7) 名称变更/ST 历史：优先用历史水位做增量，按股票文件落盘。
run_step namechange_fetch \
  python3 "${DATASET_FETCHER_PATH}/namechange_fetcher.py" \
    --start_date "${START_DATE}" \
    --end_date "${END_DATE}" \
    --output_dir "${OUTPUT_DIR}" \
    "${FORCE_ARGS[@]}"

# 7.1) namechange raw 合并：只有拉取成功才合并，避免下游读到不完整大表。
if [[ "${STEP_STATUS[namechange_fetch]:-1}" == "0" ]]; then
  run_step namechange_merge \
    python3 "${DATASET_FETCHER_PATH}/namechange_merger.py" \
      --input_dir "${OUTPUT_DIR}/namechange" \
      --output_file "${OUTPUT_DIR}/namechange/namechange.parquet"
else
  echo "跳过 namechange_merge：namechange_fetch 未成功，避免生成不完整的合并文件。"
fi

# 汇总失败步骤。CONTINUE_ON_ERROR=1 时，单步失败不会中断流水线，但最终会返回非 0。
if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
  echo "data fetch pipeline 完成但存在失败步骤：${FAILED_STEPS[*]}"
  echo "可重新执行同一条命令补齐；默认不会重拉已完成的日期/股票。如需强制重拉，使用：FORCE_FETCH=1 $0 ${START_DATE} ${END_DATE}"
  exit 1
fi

echo "data fetch pipeline 全部步骤成功完成。"
