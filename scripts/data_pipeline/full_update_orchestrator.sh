#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 全量数据更新自动化编排脚本
# ============================================================================
#
# 功能：
#   1. 补充剩余小数据集（adj_factors / suspend_d / fundamentals / industry）
#   2. 等待特色数据（8个）拉取完成
#   3. 运行 data_process_pipeline.sh（清洗 + 加工 + 宽表）
#   4. 运行 factor_calculation_pipeline.sh（特征 + 横截面）
#   5. 运行 label_calculation_pipeline.sh（标签）
#
# 用法：
#   bash scripts/data_pipeline/full_update_orchestrator.sh
#
# 环境变量：
#   TUSHARE_TOKEN          Tushare token（必需）
#   SKIP_FETCH=1           跳过数据拉取，直接从处理开始（默认 0）
#   SKIP_PROCESS=1         跳过处理阶段（默认 0）
#   SKIP_FACTORS=1         跳过因子和横截面阶段（默认 0）
#   SKIP_LABELS=1          跳过标签阶段（默认 0）
#   CONTINUE_ON_ERROR=1    遇到错误继续（默认 1）
#   LOG_DIR                日志目录（默认 log/full_update）
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RAW_DATA_DIR="${PROJECT_ROOT}/data/raw_data"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/log/full_update}"
mkdir -p "${LOG_DIR}"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"
export TUSHARE_TOKEN="${TUSHARE_TOKEN:-}"

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"
FAILED_STEPS=()
declare -A STEP_STATUS=()

log() {
    echo "[$(date '+%F %T')] $*"
}

run_step() {
    local step_name="$1"
    shift
    local log_file="${LOG_DIR}/${RUN_ID}_${step_name}.log"

    log "START ${step_name}，日志：${log_file}"
    set +e
    "$@" 2>&1 | tee "${log_file}"
    local status=${PIPESTATUS[0]}
    set -e

    STEP_STATUS["${step_name}"]="${status}"
    if [[ ${status} -ne 0 ]]; then
        FAILED_STEPS+=("${step_name}")
        log "FAIL  ${step_name}，退出码：${status}"
        if [[ "${CONTINUE_ON_ERROR:-1}" != "1" ]]; then
            exit "${status}"
        fi
        return 0
    fi
    log "OK    ${step_name}"
}

# 检查特色数据是否全部完成（通过检查进度文件）
wait_for_alternative_data() {
    local datasets=("top_list" "top_inst" "block_trade" "forecast" "repurchase" "stk_holdertrade" "report_rc" "dc_index")
    local target_date="20250602"  # 历史补全的结束日期

    log "等待特色数据拉取完成..."

    while true; do
        local all_done=1
        for ds in "${datasets[@]}"; do
            local progress_file="${RAW_DATA_DIR}/${ds}_fetch_progress.json"
            if [[ -f "${progress_file}" ]]; then
                local last_date
                # 兼容两种进度文件格式：BaseDailyFetcher 的 completed_dates 列表，或旧版的 last_fetched_date
                last_date=$(python3 -c "
import json
d = json.load(open('${progress_file}'))
# 格式1: {dataset}_completed_dates 列表
key = '${ds}_completed_dates'
if key in d and d[key]:
    print(max(d[key]))
else:
    # 格式2: last_fetched_date
    print(d.get('last_fetched_date', ''))
" 2>/dev/null || echo "")
                if [[ -z "${last_date}" || "${last_date}" < "${target_date}" ]]; then
                    all_done=0
                    log "  ${ds}: 进度 ${last_date} / ${target_date}（未完成）"
                fi
            else
                all_done=0
                log "  ${ds}: 无进度文件（可能还在跑第一个数据集）"
            fi
        done

        if [[ ${all_done} -eq 1 ]]; then
            log "所有特色数据拉取完成！"
            return 0
        fi

        log "等待 120 秒后重试..."
        sleep 120
    done
}

log "========================================"
log "全量数据更新编排启动 run_id=${RUN_ID}"
log "========================================"

# ============================================================================
# 阶段 1：补充剩余小数据集
# ============================================================================
if [[ "${SKIP_FETCH:-0}" != "1" ]]; then
    log ""
    log "========== 阶段 1：补充小数据集 =========="

    # adj_factors 增量
    run_step fetch_adj_factors \
        python3 "${PROJECT_ROOT}/src/utils/dataset_fetcher/adj_factors_fetcher.py" \
            --start_date 20260715 \
            --end_date 20260921 \
            --output_dir "${RAW_DATA_DIR}" \
            --sleep_time 0.3

    # suspend_d 增量
    run_step fetch_suspend_d \
        python3 "${PROJECT_ROOT}/src/utils/dataset_fetcher/suspend_d_fetcher.py" \
            --start_date 20260715 \
            --end_date 20260921 \
            --output_dir "${RAW_DATA_DIR}" \
            --sleep_time 0.3

    # fundamentals 增量
    run_step fetch_fundamentals \
        python3 "${PROJECT_ROOT}/src/utils/dataset_fetcher/fundamentals_fetcher.py" \
            --start_date 20260501 \
            --end_date 20260921 \
            --output_dir "${RAW_DATA_DIR}" \
            --sleep_time 0.3

    # industry 增量
    run_step fetch_industry \
        python3 "${PROJECT_ROOT}/src/utils/dataset_fetcher/industry_fetcher.py" \
            --start_date 20260916 \
            --end_date 20260921 \
            --output_dir "${RAW_DATA_DIR}" \
            --sleep_time 0.3

    # 等待特色数据完成
    run_step wait_alternative_data wait_for_alternative_data

    # 验证 raw 数据完整性
    run_step validate_raw \
        python3 "${PROJECT_ROOT}/scripts/data_pipeline/validate_data_integrity.py" \
            --stage raw \
            --output "${LOG_DIR}/${RUN_ID}_validate_raw.json"

    log "阶段 1 完成：所有原始数据已就绪并通过验证"
else
    log "跳过数据拉取阶段：SKIP_FETCH=1"
fi

# ============================================================================
# 阶段 2：数据清洗 + 加工（data_process_pipeline）
# ============================================================================
if [[ "${SKIP_PROCESS:-0}" != "1" ]]; then
    log ""
    log "========== 阶段 2：数据清洗 + 加工 =========="

    run_step data_process_pipeline \
        bash "${PROJECT_ROOT}/scripts/data_pipeline/data_process_pipeline.sh"

    # 验证 cleaned + processed 数据完整性
    run_step validate_cleaned \
        python3 "${PROJECT_ROOT}/scripts/data_pipeline/validate_data_integrity.py" \
            --stage cleaned \
            --output "${LOG_DIR}/${RUN_ID}_validate_cleaned.json"

    run_step validate_processed \
        python3 "${PROJECT_ROOT}/scripts/data_pipeline/validate_data_integrity.py" \
            --stage processed \
            --output "${LOG_DIR}/${RUN_ID}_validate_processed.json"

    log "阶段 2 完成：cleaned_data + processd_data 已更新并通过验证"
else
    log "跳过处理阶段：SKIP_PROCESS=1"
fi

# ============================================================================
# 阶段 3：特征生成 + 横截面处理（factor_calculation_pipeline）
# ============================================================================
if [[ "${SKIP_FACTORS:-0}" != "1" ]]; then
    log ""
    log "========== 阶段 3：特征 + 横截面处理 =========="

    run_step factor_calculation_pipeline \
        bash "${PROJECT_ROOT}/scripts/data_pipeline/factor_calculation_pipeline.sh"

    # 验证 features + cross_sectional 数据完整性
    run_step validate_features \
        python3 "${PROJECT_ROOT}/scripts/data_pipeline/validate_data_integrity.py" \
            --stage features \
            --output "${LOG_DIR}/${RUN_ID}_validate_features.json"

    run_step validate_cross_sectional \
        python3 "${PROJECT_ROOT}/scripts/data_pipeline/validate_data_integrity.py" \
            --stage cross_sectional \
            --output "${LOG_DIR}/${RUN_ID}_validate_cross_sectional.json"

    log "阶段 3 完成：features_data + cross_sectional_processd_data 已更新并通过验证"
else
    log "跳过因子阶段：SKIP_FACTORS=1"
fi

# ============================================================================
# 阶段 4：标签生成（label_calculation_pipeline）
# ============================================================================
if [[ "${SKIP_LABELS:-0}" != "1" ]]; then
    log ""
    log "========== 阶段 4：标签生成 =========="

    run_step label_calculation_pipeline \
        bash "${PROJECT_ROOT}/scripts/data_pipeline/label_calculation_pipeline.sh"

    # 验证标签数据完整性
    run_step validate_labels \
        python3 "${PROJECT_ROOT}/scripts/data_pipeline/validate_data_integrity.py" \
            --stage labels \
            --output "${LOG_DIR}/${RUN_ID}_validate_labels.json"

    log "阶段 4 完成：generated_label 已更新并通过验证"
else
    log "跳过标签阶段：SKIP_LABELS=1"
fi

# ============================================================================
# 总结
# ============================================================================
log ""
log "========================================"
if [[ ${#FAILED_STEPS[@]} -gt 0 ]]; then
    log "全量更新完成，但存在失败步骤：${FAILED_STEPS[*]}"
    log "请查看 ${LOG_DIR}/${RUN_ID}_*.log 定位问题"
    exit 1
else
    log "全量更新全部完成！"
    log "日志目录：${LOG_DIR}"
fi
log "========================================"
