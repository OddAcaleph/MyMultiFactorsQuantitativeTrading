#!/usr/bin/env bash

# 批量计算 xgboost_grid_search_models 下所有 grid 模型的 IC / ICIR / RankIC。
# 输出目录固定在 /opt/tiger/qyd/tmp/xgb_model_ic_results，便于直接查看汇总结果。

set -uo pipefail

MODEL_ROOT="${MODEL_ROOT:-/mnt/bn/aigc-algorithm-group/qyd/pp/xgboost_grid_search_models}"
CALCULATOR="${CALCULATOR:-/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/utils/ic_validator/xgb_ic_metrics.py}"
OUT_ROOT="${OUT_ROOT:-/opt/tiger/qyd/tmp}"
RESULT_DIR="${RESULT_DIR:-${OUT_ROOT}/xgb_model_ic_results}"
PER_MODEL_DIR="${RESULT_DIR}/per_model"
ERROR_LOG="${RESULT_DIR}/failed_models.log"
SUMMARY_CSV="${RESULT_DIR}/xgb_model_ic_summary.csv"
SUMMARY_TSV="${RESULT_DIR}/xgb_model_ic_summary.tsv"
SUMMARY_TXT="${RESULT_DIR}/xgb_model_ic_summary.txt"

mkdir -p "${PER_MODEL_DIR}"
: > "${ERROR_LOG}"

if [[ ! -d "${MODEL_ROOT}" ]]; then
  echo "模型根目录不存在: ${MODEL_ROOT}" >&2
  exit 1
fi

if [[ ! -f "${CALCULATOR}" ]]; then
  echo "IC 计算脚本不存在: ${CALCULATOR}" >&2
  exit 1
fi

echo "========== XGB 模型 IC 批量计算 =========="
echo "模型根目录: ${MODEL_ROOT}"
echo "IC 计算脚本: ${CALCULATOR}"
echo "输出目录: ${RESULT_DIR}"
echo

mapfile -d '' PRED_FILES < <(MODEL_ROOT="${MODEL_ROOT}" python - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["MODEL_ROOT"])
paths = sorted(root.rglob("grid0*/train_outputs/pred_test.parquet"))
for path in paths:
    print(str(path), end="\0")
PY
)

if [[ ${#PRED_FILES[@]} -eq 0 ]]; then
  echo "没有找到匹配的 pred_test.parquet: ${MODEL_ROOT}/**/grid0*/train_outputs/pred_test.parquet" >&2
  exit 1
fi

echo "找到 ${#PRED_FILES[@]} 个 pred_test.parquet，开始逐个计算..."
echo

success_count=0
failed_count=0

for pred_path in "${PRED_FILES[@]}"; do
  model_dir="$(basename "$(dirname "$(dirname "${pred_path}")")")"
  model_out_dir="${PER_MODEL_DIR}/${model_dir}"
  mkdir -p "${model_out_dir}"

  echo "[RUN] ${model_dir}"
  echo "      pred: ${pred_path}"

  if python "${CALCULATOR}" "${pred_path}" --output-dir "${model_out_dir}" \
    > "${model_out_dir}/stdout_metrics.json" \
    2> "${model_out_dir}/stderr.log"; then
    success_count=$((success_count + 1))
    echo "      ok: ${model_out_dir}/xgb_ic_metrics.json"
  else
    failed_count=$((failed_count + 1))
    {
      echo "[FAILED] ${model_dir}"
      echo "pred_path=${pred_path}"
      echo "stderr=${model_out_dir}/stderr.log"
      echo
    } >> "${ERROR_LOG}"
    echo "      failed: 详情见 ${model_out_dir}/stderr.log"
  fi
done

echo
echo "计算完成: success=${success_count}, failed=${failed_count}"
echo "开始生成汇总文件..."

RESULT_DIR="${RESULT_DIR}" MODEL_ROOT="${MODEL_ROOT}" python - <<'PY'
import csv
import json
import math
import os
from pathlib import Path

result_dir = Path(os.environ["RESULT_DIR"])
per_model_dir = result_dir / "per_model"
summary_csv = result_dir / "xgb_model_ic_summary.csv"
summary_tsv = result_dir / "xgb_model_ic_summary.tsv"
summary_txt = result_dir / "xgb_model_ic_summary.txt"
error_log = result_dir / "failed_models.log"

rows = []
for metrics_path in sorted(per_model_dir.glob("grid0*/xgb_ic_metrics.json")):
    model_name = metrics_path.parent.name
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    rows.append(
        {
            "model": model_name,
            "label_col": metrics.get("label_col"),
            "ic": metrics.get("ic"),
            "ic_std": metrics.get("ic_std"),
            "icir": metrics.get("icir"),
            "rank_ic": metrics.get("rank_ic"),
            "rank_ic_std": metrics.get("rank_ic_std"),
            "rank_icir": metrics.get("rank_icir"),
            "row_count": metrics.get("row_count"),
            "date_count": metrics.get("date_count"),
            "valid_ic_days": metrics.get("valid_ic_days"),
            "valid_rank_ic_days": metrics.get("valid_rank_ic_days"),
            "start_date": metrics.get("start_date"),
            "end_date": metrics.get("end_date"),
            "metrics_json": str(metrics_path),
            "daily_ic_csv": str(metrics_path.parent / "xgb_daily_ic.csv"),
        }
    )

def numeric_value(value):
    if value is None:
        return float("-inf")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("-inf")
    if math.isnan(value):
        return float("-inf")
    return value

rows_by_model = sorted(rows, key=lambda row: row["model"])
rows_by_icir = sorted(rows, key=lambda row: numeric_value(row["icir"]), reverse=True)

fieldnames = [
    "model",
    "label_col",
    "ic",
    "ic_std",
    "icir",
    "rank_ic",
    "rank_ic_std",
    "rank_icir",
    "row_count",
    "date_count",
    "valid_ic_days",
    "valid_rank_ic_days",
    "start_date",
    "end_date",
    "metrics_json",
    "daily_ic_csv",
]

with summary_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_by_model)

with summary_tsv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
    writer.writeheader()
    writer.writerows(rows_by_model)

def fmt_float(value):
    if value is None:
        return "NA"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return "NA"
    return f"{value:.6f}"

def table_lines(table_rows):
    headers = ["rank", "model", "label", "IC", "ICIR", "RankIC", "RankICIR", "days", "rows"]
    body = []
    for idx, row in enumerate(table_rows, 1):
        body.append(
            [
                str(idx),
                row["model"],
                str(row["label_col"]),
                fmt_float(row["ic"]),
                fmt_float(row["icir"]),
                fmt_float(row["rank_ic"]),
                fmt_float(row["rank_icir"]),
                str(row["valid_ic_days"]),
                str(row["row_count"]),
            ]
        )
    widths = [len(h) for h in headers]
    for line in body:
        widths = [max(w, len(cell)) for w, cell in zip(widths, line)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*headers), fmt.format(*["-" * w for w in widths])]
    lines.extend(fmt.format(*line) for line in body)
    return lines

failed_text = error_log.read_text(encoding="utf-8").strip() if error_log.exists() else ""
with summary_txt.open("w", encoding="utf-8") as f:
    f.write("XGB 模型 IC 汇总\n")
    f.write("================\n")
    f.write(f"模型根目录: {os.environ['MODEL_ROOT']}\n")
    f.write(f"成功模型数: {len(rows)}\n")
    f.write(f"失败模型数: {failed_text.count('[FAILED]')}\n")
    f.write(f"CSV 明细: {summary_csv}\n")
    f.write(f"TSV 明细: {summary_tsv}\n")
    f.write(f"失败日志: {error_log}\n")
    f.write("\n按 ICIR 从高到低排序：\n")
    f.write("\n".join(table_lines(rows_by_icir)))
    f.write("\n\n按模型名排序：\n")
    f.write("\n".join(table_lines(rows_by_model)))
    if failed_text:
        f.write("\n\n失败模型：\n")
        f.write(failed_text)
        f.write("\n")

print(f"已生成汇总 TXT: {summary_txt}")
print(f"已生成汇总 CSV: {summary_csv}")
print(f"已生成汇总 TSV: {summary_tsv}")
PY

echo
echo "========== 汇总结果预览 =========="
cat "${SUMMARY_TXT}"

echo
echo "所有输出已保存到: ${RESULT_DIR}"
