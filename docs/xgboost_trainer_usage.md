# XGBoost Trainer 操作方案

本文档说明如何使用当前项目中的 `XGBoostTrainer` 和 `XGBoostInferencer`，基于新版 `ParquetLoader` 读取 61 个特征和 `generated_label/daily_labels` 中的 label 进行训练、评估和推理。

## 1. 当前数据读取方案

训练数据由 `ParquetLoader` 负责读取，配置文件是：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/parquet_loader_config.json
```

当前默认读取以下数据源：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cross_sectional_processd_data/wide_table_daily_bars
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cross_sectional_processd_data/price_volume_factors
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cross_sectional_processd_data/moneyflow_factors
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cross_sectional_processd_data/fundamental_factors
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cross_sectional_processd_data/industry_factors
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/generated_label/daily_labels
```

默认特征为配置文件中的 61 个 `feature_cols`。

默认训练目标为：

```json
"label_name": "label_5d"
```

即从 `daily_labels` 中读取 `label_5d`，不再由 trainer 或 loader 临时计算 label。

## 2. 训练配置文件

XGBoost 训练配置文件是：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/xgboost_trainer_config.json
```

当前关键配置如下：

```json
{
  "loader_config_path": "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/parquet_loader_config.json",
  "instruments": "all",
  "start_time": "2000-01-01",
  "end_time": "2025-12-31",
  "output_dir": "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset",
  "model_dir": "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/models",
  "model_filename": "xgboost_label_5d.json",
  "prefer_gpu": true,
  "segments": {
    "train": ["2000-01-01", "2020-12-31"],
    "valid": ["2021-01-01", "2022-12-31"],
    "test": ["2023-01-01", "2025-12-31"]
  }
}
```

训练完成后默认会输出：

```text
模型文件：/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/models/xgboost_label_5d.json
测试集预测：/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet
评估指标：/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/metrics.json
```

## 3. 推荐先做小样本试跑

全量训练时间较长，建议先用少量股票和短时间区间做 smoke test。

项目已经提供训练入口脚本：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/trainer/run_xgboost_training.py
```

可以先查看参数：

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/trainer/run_xgboost_training.py" --help
```

如果只是使用配置文件里的全量配置，直接看第 4 节即可。

小样本试跑由于需要临时覆盖股票池、时间段和模型参数，推荐继续使用 Python 片段方式执行：

在任意目录执行：

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" python - <<'PY'
from trainer import XGBoostTrainer

trainer = XGBoostTrainer(
    config_path="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/xgboost_trainer_config.json",
    instruments=["000003.SZ", "000005.SZ"],
    segments={
        "train": ("2020-01-02", "2020-03-31"),
        "valid": ("2020-04-01", "2020-04-30"),
        "test": ("2020-05-01", "2020-05-29"),
    },
    start_time="2020-01-02",
    end_time="2020-05-29",
    model_params={
        "n_estimators": 20,
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_jobs": 1,
        "objective": "reg:squarederror",
        "random_state": 42,
    },
    prefer_gpu=False,
    output_dir="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_smoke_test",
    model_path="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/models/xgboost_smoke_test.json",
)

result = trainer.run(verbose=False, save=True)

print("小样本训练完成")
print("模型路径:", trainer.model_path)
print("评估指标:", result["metrics"])
PY
```

如果输出了 `metrics`，并且模型文件、预测文件正常生成，就说明训练链路可用。

## 4. 启动全量训练

确认小样本没问题后，执行全量训练：

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/trainer/run_xgboost_training.py"
```

说明：

- `prefer_gpu=true` 时会优先尝试 GPU。
- 如果 GPU 不可用或训练失败，当前实现会 fallback 到 CPU。
- 默认使用 `/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/xgboost_trainer_config.json`。
- 默认 `--verbose 50`，表示训练过程中每隔一定轮次输出一次 XGBoost 日志。

常用参数：

```bash
# 指定训练配置
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/trainer/run_xgboost_training.py" \
  --config "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/xgboost_trainer_config.json"

# 强制 CPU 训练
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/trainer/run_xgboost_training.py" --cpu

# 不打印 XGBoost 训练日志
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/trainer/run_xgboost_training.py" --verbose false

# 覆盖模型输出路径
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/trainer/run_xgboost_training.py" \
  --model-path "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/models/xgboost_custom.json"

# 覆盖输出目录
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/trainer/run_xgboost_training.py" \
  --output-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_custom_run"
```

## 5. 推理 / 重新生成预测

推理配置文件是：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/xgboost_inferencer_config.json
```

全量训练完成后，可以用保存好的模型重新生成预测：

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/trainer/run_xgboost_inference.py"
```

默认推理输出：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet
```

推理脚本路径：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/trainer/run_xgboost_inference.py
```

查看参数：

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/trainer/run_xgboost_inference.py" --help
```

常用参数：

```bash
# 指定推理配置
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/trainer/run_xgboost_inference.py" \
  --config "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/xgboost_inferencer_config.json"

# 指定模型路径
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/trainer/run_xgboost_inference.py" \
  --model-path "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/models/xgboost_label_5d.json"

# 指定推理时间区间
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/trainer/run_xgboost_inference.py" \
  --start-time "2024-01-01" \
  --end-time "2024-12-31"

# 推理时不读取 label，只输出 pred
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/trainer/run_xgboost_inference.py" --no-label

# 强制 CPU 推理
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/trainer/run_xgboost_inference.py" --cpu
```

## 6. 切换训练 label

当前支持的 label 包括：

```text
label_1d
label_2d
label_3d
label_5d
label_10d
label_20d
label_rank_1d
label_rank_2d
label_rank_3d
label_rank_5d
label_rank_10d
label_rank_20d
```

如果要把训练目标从 `label_5d` 切到 `label_rank_5d`，需要同步修改三个地方。

### 6.1 修改 loader label

文件：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/parquet_loader_config.json
```

修改：

```json
"label_name": "label_rank_5d"
```

### 6.2 修改 trainer 模型文件名

文件：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/xgboost_trainer_config.json
```

建议修改模型文件名，避免覆盖旧模型：

```json
"model_filename": "xgboost_label_rank_5d.json"
```

### 6.3 修改 inferencer 配置

文件：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/xgboost_inferencer_config.json
```

修改：

```json
"model_path": "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/models/xgboost_label_rank_5d.json",
"columns": {
  "prediction": "pred",
  "label": "label_rank_5d"
}
```

然后重新执行训练和推理脚本。

## 7. 常见检查命令

### 7.1 检查 trainer 相关测试

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" pytest "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/test/test_xgboost_trainer.py" -q
```

### 7.2 检查 parquet loader 测试

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" pytest "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/test/test_parquet_loader.py" -q
```

### 7.3 检查配置 JSON 格式

```bash
python -m json.tool "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/parquet_loader_config.json" >/dev/null
python -m json.tool "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/xgboost_trainer_config.json" >/dev/null
python -m json.tool "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/xgboost_inferencer_config.json" >/dev/null
```

## 8. 训练结果后续使用

训练输出的 `pred_test.parquet` 包含：

```text
pred
label_5d  # 或当前配置的 label_name
```

索引是：

```text
datetime
instrument
```

这个文件可以继续接后续 IC 分析、分组收益分析或回测模块使用。
