#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# 增量更新流水线
# ============================================================================
#
# 自动检测数据最新日期，只拉取和处理新增数据，几分钟即可完成每日更新。
#
# 用法：
#   bash scripts/data_pipeline/incremental_update.sh [target_end_date]
#
# 参数：
#   target_end_date  目标结束日期，默认今天（YYYYMMDD 格式）
#
# 常用环境变量：
#   SKIP_FETCH=1          跳过 raw 数据拉取，直接从清洗开始
#   SKIP_LABELS=1         跳过标签生成
#   FETCH_ONLY=1          只拉取 raw 数据，不做下游处理
#   LOG_DIR=/path         覆盖日志目录
#   LOG_LEVEL=DEBUG       日志级别
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RAW_DATA_DIR="${PROJECT_ROOT}/data/raw_data"
CLEANED_DATA_DIR="${PROJECT_ROOT}/data/cleaned_data"
PROCESSED_DATA_DIR="${PROJECT_ROOT}/data/processd_data"
FEATURES_DATA_DIR="${PROJECT_ROOT}/data/features_data"
CROSS_SECTIONAL_DIR="${PROJECT_ROOT}/data/cross_sectional_processd_data"
LABEL_DIR="${PROJECT_ROOT}/data/generated_label"

LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/log/incremental_update}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"

mkdir -p "${LOG_DIR}"
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

RUN_ID="$(date +%Y%m%d_%H%M%S)_$$"

# 目标结束日期（默认今天）
TARGET_END_DATE="${1:-$(date +%Y%m%d)}"

# 因子滚动窗口回溯天数（最大 252 天，来自 enhanced_alpha）
LOOKBACK_DAYS=260
# 标签未来窗口（20 天）
LABEL_LOOKAHEAD_DAYS=20

echo "========================================"
echo "  增量更新流水线"
echo "  目标日期：${TARGET_END_DATE}"
echo "  Run ID：${RUN_ID}"
echo "========================================"

# 统一步骤执行器
run_step() {
  local step_name="$1"
  shift
  local log_file="${LOG_DIR}/${RUN_ID}_${step_name}.log"

  echo "[$(date '+%F %T')] START ${step_name}，日志：${log_file}"
  set +e
  "$@" 2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e

  if [[ ${status} -ne 0 ]]; then
    echo "[$(date '+%F %T')] FAIL  ${step_name}，退出码：${status}"
    exit "${status}"
  fi

  echo "[$(date '+%F %T')] OK    ${step_name}"
}

# 检测 raw_data/daily_bars 的最新日期
echo ""
echo "--- 检测数据最新日期 ---"

RAW_DAILY_LATEST=$(python3 -c "
from pathlib import Path
dates = [f.stem for f in Path('${RAW_DATA_DIR}/daily_bars').rglob('*.parquet')
         if len(f.stem) == 8 and f.stem.isdigit()]
print(max(dates) if dates else 'NONE')
" 2>/dev/null || echo "NONE")

echo "raw_data/daily_bars 最新日期：${RAW_DAILY_LATEST}"

if [[ "${RAW_DAILY_LATEST}" == "NONE" ]]; then
  echo "错误：找不到 raw daily_bars 数据，请先执行全量初始化。"
  exit 1
fi

if [[ ! "${RAW_DAILY_LATEST}" < "${TARGET_END_DATE}" ]]; then
  echo "数据已经是最新的（${RAW_DAILY_LATEST} >= ${TARGET_END_DATE}），无需更新。"
  exit 0
fi

# 计算增量日期范围
# 新增交易日（不含最新日期，从下一天开始）
INCREMENT_START=$(python3 -c "
import sys
from pathlib import Path

# 从 daily_bars 获取交易日历
dates = sorted([f.stem for f in Path('${RAW_DATA_DIR}/daily_bars').rglob('*.parquet')
                if len(f.stem) == 8 and f.stem.isdigit()])
latest = '${RAW_DAILY_LATEST}'
idx = dates.index(latest) if latest in dates else len(dates) - 1
# 下一个交易日
if idx + 1 < len(dates):
    print(dates[idx + 1])
else:
    print(latest)
" 2>/dev/null)

echo "增量起始日期：${INCREMENT_START}"
echo "增量结束日期：${TARGET_END_DATE}"

# 因子计算需要向前回溯的日期（用于滚动窗口）
FACTOR_START=$(python3 -c "
import sys
from pathlib import Path

dates = sorted([f.stem for f in Path('${RAW_DATA_DIR}/daily_bars').rglob('*.parquet')
                if len(f.stem) == 8 and f.stem.isdigit()])
latest = '${RAW_DAILY_LATEST}'
lookback = ${LOOKBACK_DAYS}
idx = dates.index(latest) if latest in dates else len(dates) - 1
start_idx = max(0, idx - lookback)
print(dates[start_idx])
" 2>/dev/null)

echo "因子计算起始（含回溯）：${FACTOR_START}"

# 标签需要重算的起始（最后 20 天 + 新增）
LABEL_START=$(python3 -c "
import sys
from pathlib import Path

dates = sorted([f.stem for f in Path('${RAW_DATA_DIR}/daily_bars').rglob('*.parquet')
                if len(f.stem) == 8 and f.stem.isdigit()])
latest = '${RAW_DAILY_LATEST}'
lookahead = ${LABEL_LOOKAHEAD_DAYS}
idx = dates.index(latest) if latest in dates else len(dates) - 1
start_idx = max(0, idx - lookahead)
print(dates[start_idx])
" 2>/dev/null)

echo "标签重算起始（含未来窗口）：${LABEL_START}"

# ============================================================================
# 阶段 1：raw 数据拉取
# ============================================================================
if [[ "${SKIP_FETCH:-0}" != "1" ]]; then
  echo ""
  echo "========== 阶段 1：原始数据拉取 =========="

  # 日行情 + 资金流 + 指数
  run_step fetch_daily_bars \
    python3 "${PROJECT_ROOT}/src/utils/dataset_fetcher/daily_bars_fetcher.py" \
      --start_date "${INCREMENT_START}" \
      --end_date "${TARGET_END_DATE}" \
      --output_dir "${RAW_DATA_DIR}"

  run_step fetch_moneyflow \
    python3 "${PROJECT_ROOT}/src/utils/dataset_fetcher/moneyflow_fetcher.py" \
      --start_date "${INCREMENT_START}" \
      --end_date "${TARGET_END_DATE}" \
      --output_dir "${RAW_DATA_DIR}"

  run_step fetch_index_daily \
    python3 "${PROJECT_ROOT}/src/utils/dataset_fetcher/index_daily_fetcher.py" \
      --start_date "${INCREMENT_START}" \
      --end_date "${TARGET_END_DATE}" \
      --output_dir "${RAW_DATA_DIR}"

  # 特色数据（8个）
  for ds in top_list top_inst block_trade forecast repurchase stk_holdertrade report_rc dc_index; do
    run_step "fetch_${ds}" \
      python3 "${PROJECT_ROOT}/src/utils/dataset_fetcher/${ds}_fetcher.py" \
        --start_date "${INCREMENT_START}" \
        --end_date "${TARGET_END_DATE}" \
        --output_dir "${RAW_DATA_DIR}"
  done

  # 基本面 + 行业 + 复权因子 + 停牌（全量重拉或增量，取决于 fetcher 实现）
  run_step fetch_adj_factors \
    python3 "${PROJECT_ROOT}/src/utils/dataset_fetcher/adj_factors_fetcher.py" \
      --start_date "${INCREMENT_START}" \
      --end_date "${TARGET_END_DATE}" \
      --output_dir "${RAW_DATA_DIR}"

  run_step fetch_suspend_d \
    python3 "${PROJECT_ROOT}/src/utils/dataset_fetcher/suspend_d_fetcher.py" \
      --start_date "${INCREMENT_START}" \
      --end_date "${TARGET_END_DATE}" \
      --output_dir "${RAW_DATA_DIR}"

  run_step fetch_industry \
    python3 "${PROJECT_ROOT}/src/utils/dataset_fetcher/industry_fetcher.py" \
      --output_dir "${RAW_DATA_DIR}"

  run_step fetch_fundamentals \
    python3 "${PROJECT_ROOT}/src/utils/dataset_fetcher/fundamentals_fetcher.py" \
      --output_dir "${RAW_DATA_DIR}"
else
  echo ""
  echo "跳过 raw 数据拉取：SKIP_FETCH=1"
fi

if [[ "${FETCH_ONLY:-0}" == "1" ]]; then
  echo ""
  echo "FETCH_ONLY=1，只拉取数据，退出。"
  exit 0
fi

# ============================================================================
# 阶段 2：清洗 + 加工
# ============================================================================
echo ""
echo "========== 阶段 2：清洗 + 加工 =========="

# 2.1 raw 聚合（全量单文件数据集，每次重新合并，很快）
echo "--- 2.1 raw 聚合 ---"

run_step raw_fundamentals_merge \
  python3 "${PROJECT_ROOT}/src/utils/dataset_fetcher/fundamentals_fetcher.py" \
    --merge_only \
    --output_dir "${RAW_DATA_DIR}"

run_step raw_namechange_merge \
  python3 "${PROJECT_ROOT}/src/utils/dataset_fetcher/namechange_merger.py" \
    --input_dir "${RAW_DATA_DIR}/namechange" \
    --output_file "${RAW_DATA_DIR}/namechange/namechange.parquet"

# 合并按日期分区的 raw 数据为单文件（供 cleaner 使用）
merge_partitioned() {
  local ds="$1"
  python3 - "${RAW_DATA_DIR}/${ds}" "${RAW_DATA_DIR}/${ds}/${ds}.parquet" <<'PY'
from pathlib import Path
import sys
import pandas as pd

input_dir = Path(sys.argv[1])
output_file = Path(sys.argv[2])

source_files = sorted(input_dir.glob("year=*/month=*/*.parquet"))
frames = [pd.read_parquet(f) for f in source_files]
combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
if not combined.empty:
    combined = combined.drop_duplicates().reset_index(drop=True)
output_file.parent.mkdir(parents=True, exist_ok=True)
combined.to_parquet(output_file, compression="snappy", index=False)
print(f"merged {len(source_files)} files, {len(combined)} rows -> {output_file}")
PY
}

run_step raw_adj_factors_merge merge_partitioned "adj_factors"
run_step raw_moneyflow_merge merge_partitioned "moneyflow"
run_step raw_industry_merge merge_partitioned "industry"
run_step raw_suspend_d_merge merge_partitioned "suspend_d"

# 2.2 清洗
echo "--- 2.2 数据清洗 ---"

# daily_bars 增量清洗（只处理新增日期）
run_step clean_daily_bars_incremental \
  python3 -m utils.incremental.daily_bars_incremental_cleaner \
    --input-dir "${RAW_DATA_DIR}/daily_bars" \
    --output-dir "${CLEANED_DATA_DIR}/daily_bars" \
    --start-date "${INCREMENT_START}" \
    --end-date "${TARGET_END_DATE}" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_clean_daily_bars.log"

# 全量单文件的清洗（数据量小，全量重跑也很快）
run_step clean_adj_factors \
  python3 "${PROJECT_ROOT}/src/utils/dataset_cleaner/adj_factors_cleaner.py" \
    --input-file "${RAW_DATA_DIR}/adj_factors/adj_factors.parquet" \
    --output-file "${CLEANED_DATA_DIR}/adj_factors/adj_factors.parquet" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_clean_adj_factors.log"

run_step clean_moneyflow \
  python3 "${PROJECT_ROOT}/src/utils/dataset_cleaner/moneyflow_cleaner.py" \
    --input-file "${RAW_DATA_DIR}/moneyflow/moneyflow.parquet" \
    --output-file "${CLEANED_DATA_DIR}/moneyflow/moneyflow.parquet" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_clean_moneyflow.log"

run_step clean_fundamentals \
  python3 "${PROJECT_ROOT}/src/utils/dataset_cleaner/fundamentals_cleaner.py" \
    --input-file "${RAW_DATA_DIR}/fundamentals/fundamentals.parquet" \
    --output-file "${CLEANED_DATA_DIR}/fundamentals/fundamentals.parquet" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_clean_fundamentals.log"

run_step clean_industry \
  python3 "${PROJECT_ROOT}/src/utils/dataset_cleaner/industry_cleaner.py" \
    --input-file "${RAW_DATA_DIR}/industry/industry.parquet" \
    --output-file "${CLEANED_DATA_DIR}/industry/industry.parquet" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_clean_industry.log"

# 2.3 加工处理
echo "--- 2.3 加工处理 ---"

run_step process_namechange_st \
  python3 "${PROJECT_ROOT}/src/utils/dataset_processor/namechange_st_processor.py" \
    --input-file "${RAW_DATA_DIR}/namechange/namechange.parquet" \
    --output-file "${PROCESSED_DATA_DIR}/namechange/namechange_st_daily.parquet" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_namechange_st.log"

run_step process_suspend_d \
  python3 "${PROJECT_ROOT}/src/utils/dataset_processor/suspend_d_processor.py" \
    --input-file "${RAW_DATA_DIR}/suspend_d/suspend_d.parquet" \
    --output-file "${PROCESSED_DATA_DIR}/suspend_d/suspend_d_daily.parquet" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_suspend_d.log"

run_step process_daily_industry_onehot \
  python3 "${PROJECT_ROOT}/src/utils/dataset_processor/daily_industry_onehot_processor.py" \
    --input-dir "${CLEANED_DATA_DIR}/industry" \
    --output-dir "${PROCESSED_DATA_DIR}/industry/daily_onehot" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_daily_industry_onehot.log"

# 2.4 宽表（增量构建）
echo "--- 2.4 宽表构建 ---"

run_step build_wide_table_incremental \
  python3 "${PROJECT_ROOT}/src/utils/dataset_processor/wide_table_daily_bars_builder.py" \
    --daily-bars-dir "${CLEANED_DATA_DIR}/daily_bars" \
    --namechange-file "${PROCESSED_DATA_DIR}/namechange/namechange_st_daily.parquet" \
    --suspend-file "${PROCESSED_DATA_DIR}/suspend_d/suspend_d_daily.parquet" \
    --adj-factor-file "${CLEANED_DATA_DIR}/adj_factors/adj_factors.parquet" \
    --fundamentals-file "${CLEANED_DATA_DIR}/fundamentals/fundamentals.parquet" \
    --moneyflow-file "${CLEANED_DATA_DIR}/moneyflow/moneyflow.parquet" \
    --industry-file "${PROCESSED_DATA_DIR}/industry/daily_onehot" \
    --output-dir "${PROCESSED_DATA_DIR}/wide_table_daily_bars" \
    --start-date "${INCREMENT_START}" \
    --end-date "${TARGET_END_DATE}" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_wide_table.log"

# ============================================================================
# 阶段 3：因子生成 + 横截面处理
# ============================================================================
echo ""
echo "========== 阶段 3：因子生成 + 横截面处理 =========="

# 3.1 因子生成（从 FACTOR_START 开始，含回溯窗口）
echo "--- 3.1 因子生成 ---"

run_step gen_price_volume_factors \
  python3 "${PROJECT_ROOT}/src/utils/features_generator/price_volume_feature_generator.py" \
    --daily-bars-dir "${CLEANED_DATA_DIR}/daily_bars" \
    --adj-factor-file "${CLEANED_DATA_DIR}/adj_factors/adj_factors.parquet" \
    --output-dir "${FEATURES_DATA_DIR}/price_volume_factors" \
    --start-date "${FACTOR_START}" \
    --end-date "${TARGET_END_DATE}" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_price_volume_factors.log"

run_step gen_fundamental_factors \
  python3 "${PROJECT_ROOT}/src/utils/features_generator/fundamental_feature_generator.py" \
    --daily-bars-dir "${CLEANED_DATA_DIR}/daily_bars" \
    --fundamentals-file "${CLEANED_DATA_DIR}/fundamentals/fundamentals.parquet" \
    --output-dir "${FEATURES_DATA_DIR}/fundamental_factors" \
    --start-date "${FACTOR_START}" \
    --end-date "${TARGET_END_DATE}" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_fundamental_factors.log"

run_step gen_moneyflow_factors \
  python3 "${PROJECT_ROOT}/src/utils/features_generator/moneyflow_feature_generator.py" \
    --moneyflow-file "${CLEANED_DATA_DIR}/moneyflow/moneyflow.parquet" \
    --daily-bars-dir "${CLEANED_DATA_DIR}/daily_bars" \
    --output-dir "${FEATURES_DATA_DIR}/moneyflow_factors" \
    --start-date "${FACTOR_START}" \
    --end-date "${TARGET_END_DATE}" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_moneyflow_factors.log"

run_step gen_industry_factors \
  python3 "${PROJECT_ROOT}/src/utils/features_generator/industry_feature_generator.py" \
    --daily-bars-dir "${CLEANED_DATA_DIR}/daily_bars" \
    --industry-file "${CLEANED_DATA_DIR}/industry/industry.parquet" \
    --fundamentals-file "${CLEANED_DATA_DIR}/fundamentals/fundamentals.parquet" \
    --output-dir "${FEATURES_DATA_DIR}/industry_factors" \
    --start-date "${FACTOR_START}" \
    --end-date "${TARGET_END_DATE}" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_industry_factors.log"

# 3.2 横截面处理（只处理新增日期）
echo "--- 3.2 横截面处理 ---"

run_step cs_wide_table \
  python3 "${PROJECT_ROOT}/src/utils/cross_sectional_processor/wide_table_daily_bars_cross_sectional_processor.py" \
    --input-dir "${PROCESSED_DATA_DIR}/wide_table_daily_bars" \
    --output-dir "${CROSS_SECTIONAL_DIR}/wide_table_daily_bars" \
    --start-date "${INCREMENT_START}" \
    --end-date "${TARGET_END_DATE}" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_cs_wide_table.log"

run_step cs_price_volume \
  python3 "${PROJECT_ROOT}/src/utils/cross_sectional_processor/price_volume_factors_cross_sectional_processor.py" \
    --input-dir "${FEATURES_DATA_DIR}/price_volume_factors" \
    --output-dir "${CROSS_SECTIONAL_DIR}/price_volume_factors" \
    --start-date "${INCREMENT_START}" \
    --end-date "${TARGET_END_DATE}" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_cs_price_volume.log"

run_step cs_fundamental \
  python3 "${PROJECT_ROOT}/src/utils/cross_sectional_processor/fundamental_factors_cross_sectional_processor.py" \
    --input-dir "${FEATURES_DATA_DIR}/fundamental_factors" \
    --output-dir "${CROSS_SECTIONAL_DIR}/fundamental_factors" \
    --start-date "${INCREMENT_START}" \
    --end-date "${TARGET_END_DATE}" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_cs_fundamental.log"

run_step cs_moneyflow \
  python3 "${PROJECT_ROOT}/src/utils/cross_sectional_processor/moneyflow_factors_cross_sectional_processor.py" \
    --input-dir "${FEATURES_DATA_DIR}/moneyflow_factors" \
    --output-dir "${CROSS_SECTIONAL_DIR}/moneyflow_factors" \
    --start-date "${INCREMENT_START}" \
    --end-date "${TARGET_END_DATE}" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_cs_moneyflow.log"

run_step cs_industry \
  python3 "${PROJECT_ROOT}/src/utils/cross_sectional_processor/industry_factors_cross_sectional_processor.py" \
    --input-dir "${FEATURES_DATA_DIR}/industry_factors" \
    --output-dir "${CROSS_SECTIONAL_DIR}/industry_factors" \
    --start-date "${INCREMENT_START}" \
    --end-date "${TARGET_END_DATE}" \
    --log-level "${LOG_LEVEL}" \
    --log-file "${LOG_DIR}/${RUN_ID}_cs_industry.log"

# ============================================================================
# 阶段 4：标签生成
# ============================================================================
if [[ "${SKIP_LABELS:-0}" != "1" ]]; then
  echo ""
  echo "========== 阶段 4：标签生成 =========="

  run_step gen_daily_labels \
    python3 "${PROJECT_ROOT}/src/utils/label_generator/daily_label_generator.py" \
      --input-dir "${PROCESSED_DATA_DIR}/wide_table_daily_bars" \
      --output-dir "${LABEL_DIR}/daily_labels" \
      --start-date "${LABEL_START}" \
      --end-date "${TARGET_END_DATE}" \
      --output-batch-size 120 \
      --log-level "${LOG_LEVEL}" \
      --log-file "${LOG_DIR}/${RUN_ID}_daily_labels.log"
else
  echo "跳过标签生成：SKIP_LABELS=1"
fi

# ============================================================================
# 完成
# ============================================================================
echo ""
echo "========================================"
echo "  增量更新完成！"
echo "  数据已更新至：${TARGET_END_DATE}"
echo "  日志目录：${LOG_DIR}"
echo "========================================"
