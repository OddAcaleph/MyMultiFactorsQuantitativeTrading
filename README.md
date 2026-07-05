# MyMultiFactorsQuantitativeTrading

一个完整的A股多因子量化交易系统，基于机器学习（XGBoost）进行股票预测和回测。项目实现了从数据获取、清洗、特征工程、模型训练到回测验证的完整量化投研流水线。

---

## 目录

- [项目简介](#项目简介)
- [项目特性](#项目特性)
- [目录结构](#目录结构)
- [技术栈](#技术栈)
- [快速开始](#快速开始)
  - [环境准备](#环境准备)
  - [数据流水线](#数据流水线)
  - [模型训练](#模型训练)
  - [模型推理](#模型推理)
  - [回测验证](#回测验证)
  - [IC验证](#ic验证)
- [核心模块说明](#核心模块说明)
  - [数据获取模块](#数据获取模块)
  - [数据清洗模块](#数据清洗模块)
  - [数据处理模块](#数据处理模块)
  - [特征生成模块](#特征生成模块)
  - [横截面处理模块](#横截面处理模块)
  - [标签生成模块](#标签生成模块)
  - [数据加载模块](#数据加载模块)
  - [模型训练模块](#模型训练模块)
  - [回测模块](#回测模块)
  - [IC验证模块](#ic验证模块)
- [配置文件说明](#配置文件说明)
- [特征工程](#特征工程)
- [常见问题](#常见问题)
- [注意事项](#注意事项)
- [许可证](#许可证)

---

## 项目简介

**MyMultiFactorsQuantitativeTrading** 是一个端到端的A股量化投研平台，支持多因子模型构建、XGBoost监督学习、以及贴近真实交易规则的回测验证。

- **版本**：v0.0.1
- **定位**：完整的量化投研流水线
- **支持市场**：A股
- **核心算法**：XGBoost 梯度提升树

### 版本历史

- **v0.0.1**：基础组件和流水线（数据获取、清洗、处理、特征生成、标签生成、XGBoost训练预测、简单回测）
- **最新版本**：多因子量化交易系统初始化版本

---

## 项目特性

1. **完整的端到端流水线**：从数据获取到回测验证的完整量化投研流程
2. **贴近真实的回测引擎**：考虑A股T+1、涨跌停、100股整数倍等特有规则
3. **丰富的因子库**：61个特征，涵盖价量、基本面、资金流、行业等多个维度
4. **严格的泄露控制**：标签计算、回测信号延迟等机制避免未来函数
5. **Qlib生态兼容**：自定义数据加载器无缝对接Qlib框架
6. **完善的文档**：每个模块都有详细的使用文档和设计说明
7. **自动化流水线**：Shell脚本支持每日自动数据更新和模型推理
8. **可配置性强**：所有关键参数通过JSON配置文件管理
9. **断点续传**：数据获取、处理等步骤都支持断点续传，提高效率
10. **数据只读原则**：所有处理步骤都不修改原始数据，结果写入新目录

---

## 目录结构

```
/opt/tiger/qyd/qyd/
├── conf/                          # 配置文件目录
│   ├── parquet_loader_config.json          # Parquet数据加载器配置
│   ├── xgboost_trainer_config.json         # XGBoost训练器配置
│   ├── xgboost_inferencer_config.json      # XGBoost推理器配置
│   ├── simple_backtester_config.json       # 简单回测器配置
│   ├── preparatory_backtester_config.json  # 预备回测器配置
│   ├── simple_backtest_grid_search_config.json      # 回测网格搜索配置
│   └── xgboost_train_backtest_grid_search_config.json # 训练+回测网格搜索配置
│
├── src/                           # 源代码目录
│   ├── trainer/                   # 模型训练模块
│   │   ├── xgboost_trainer.py              # XGBoost训练器核心类
│   │   ├── xgboost_inferencer.py           # XGBoost推理器核心类
│   │   ├── run_xgboost_training.py         # 训练命令行入口
│   │   ├── run_xgboost_inference.py        # 推理命令行入口
│   │   └── run_xgboost_train_backtest_grid_search.py  # 训练+回测网格搜索
│   │
│   ├── backtester/                # 回测模块
│   │   ├── simple_backtester.py             # 简单回测器（真实交易规则）
│   │   ├── preparatory_backtester.py        # 预备回测器（快速验证）
│   │   ├── run_simple_backtest.py          # 简单回测命令行入口
│   │   ├── run_preparatory_backtest.py      # 预备回测命令行入口
│   │   └── run_simple_backtest_grid_search.py  # 回测网格搜索入口
│   │
│   └── utils/                    # 工具模块
│       ├── config.py                       # 配置加载工具
│       ├── backtest_plotter.py             # 回测结果绘图工具
│       ├── dataset_fetcher/                # 数据获取模块
│       ├── dataset_cleaner/                # 数据清洗模块
│       ├── dataset_processor/              # 数据处理模块
│       ├── features_generator/             # 特征生成模块
│       ├── cross_sectional_processor/      # 横截面处理模块
│       ├── label_generator/                # 标签生成模块
│       ├── dataset_generator/              # 数据集生成模块
│       └── ic_validator/                   # IC验证模块
│
├── data/                          # 数据目录
│   ├── raw_data/                  # 原始数据（Tushare获取）
│   ├── cleaned_data/              # 清洗后数据
│   ├── processd_data/             # 处理后数据
│   ├── features_data/             # 特征数据
│   ├── cross_sectional_processd_data/  # 横截面处理后数据
│   └── generated_label/           # 生成的标签数据
│
├── scripts/                       # 脚本目录
│   ├── data_pipeline/             # 数据流水线脚本
│   ├── ic_validate/               # IC验证脚本
│   ├── search/                    # 网格搜索脚本
│   └── design/                    # 设计文档
│
├── docs/                          # 文档目录
├── models/                        # 模型目录
├── outputs/                       # 输出目录
├── log/                           # 日志目录
├── test/                          # 测试目录
└── requirements.txt               # Python依赖
```

---

## 技术栈

| 类别 | 技术/框架 | 版本 | 用途 |
|------|----------|------|------|
| **编程语言** | Python | 3.x | 主要开发语言 |
| **机器学习** | XGBoost | 0.3.2 | 梯度提升树模型 |
| **量化框架** | pyqlib | 0.9.7 | 微软开源量化框架 |
| **数据处理** | pandas | - | 数据处理和分析 |
| | numpy | - | 数值计算 |
| **数据存储** | pyarrow | - | Parquet文件读写 |
| **数据源** | tushare | - | A股数据API |
| **可视化** | matplotlib | - | 回测结果绘图 |

### 系统要求

- Python 3.7+
- 内存：建议16GB以上（处理全市场数据时）
- 磁盘：建议100GB以上（存储历史数据）
- GPU（可选）：支持CUDA的NVIDIA GPU可加速训练

---

## 快速开始

### 环境准备

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置Tushare Token（三选一）
export TUSHARE_TOKEN="your_token"
# 或
echo "your_token" > ~/.tushare_token
# 或在命令行参数中指定

# 3. 设置Python路径
export PYTHONPATH="/opt/tiger/qyd/qyd/src:$PYTHONPATH"
```

### 数据流水线

```bash
# 每日完整流水线（推荐）
bash scripts/data_pipeline/daily_data_pipeline.sh

# 指定日期范围
bash scripts/data_pipeline/daily_data_pipeline.sh 20260601 20260630

# 单独运行各阶段
bash scripts/data_pipeline/data_fetch_pipeline.sh 20260601 20260630
bash scripts/data_pipeline/data_process_pipeline.sh 20260601 20260630
bash scripts/data_pipeline/factor_calculation_pipeline.sh 20260601 20260630
bash scripts/data_pipeline/label_calculation_pipeline.sh 20260601 20260630
```

#### 流水线环境变量控制

```bash
# 跳过某些阶段
RUN_FETCH=0 RUN_PROCESS=1 bash scripts/data_pipeline/daily_data_pipeline.sh

# 强制CPU推理
INFERENCE_CPU=1 bash scripts/data_pipeline/daily_data_pipeline.sh

# 失败后继续
MASTER_CONTINUE_ON_ERROR=1 bash scripts/data_pipeline/daily_data_pipeline.sh
```

### 模型训练

```bash
# 小样本试跑（推荐先做）
PYTHONPATH="/opt/tiger/qyd/qyd/src" python - <<'PY'
from trainer import XGBoostTrainer

trainer = XGBoostTrainer(
    config_path="/opt/tiger/qyd/qyd/conf/xgboost_trainer_config.json",
    instruments=["000003.SZ", "000005.SZ"],
    segments={
        "train": ("2020-01-02", "2020-03-31"),
        "valid": ("2020-04-01", "2020-04-30"),
        "test": ("2020-05-01", "2020-05-29"),
    },
    model_params={
        "n_estimators": 20,
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_jobs": 1,
    },
    prefer_gpu=False,
    output_dir="/opt/tiger/qyd/qyd/outputs/xgb_smoke_test",
    model_path="/opt/tiger/qyd/qyd/models/xgboost_smoke_test.json",
)

result = trainer.run(verbose=False, save=True)
print("评估指标:", result["metrics"])
PY

# 全量训练
PYTHONPATH="/opt/tiger/qyd/qyd/src" \
python "/opt/tiger/qyd/qyd/src/trainer/run_xgboost_training.py"

# 强制CPU训练
PYTHONPATH="/opt/tiger/qyd/qyd/src" \
python "/opt/tiger/qyd/qyd/src/trainer/run_xgboost_training.py" --cpu

# 自定义模型输出
PYTHONPATH="/opt/tiger/qyd/qyd/src" \
python "/opt/tiger/qyd/qyd/src/trainer/run_xgboost_training.py" \
  --model-path "/opt/tiger/qyd/qyd/models/xgboost_custom.json"
```

#### 默认数据集划分

- **训练集**：2000-01-01 ~ 2020-12-31
- **验证集**：2021-01-01 ~ 2022-12-31
- **测试集**：2023-01-01 ~ 2025-12-31

### 模型推理

```bash
# 使用训练好的模型进行推理
PYTHONPATH="/opt/tiger/qyd/qyd/src" \
python "/opt/tiger/qyd/qyd/src/trainer/run_xgboost_inference.py"

# 指定推理时间范围
PYTHONPATH="/opt/tiger/qyd/qyd/src" \
python "/opt/tiger/qyd/qyd/src/trainer/run_xgboost_inference.py" \
  --start-time "2024-01-01" \
  --end-time "2024-12-31"

# 推理时不加载label（只输出预测）
PYTHONPATH="/opt/tiger/qyd/qyd/src" \
python "/opt/tiger/qyd/qyd/src/trainer/run_xgboost_inference.py" --no-label
```

### 回测验证

```bash
# 使用默认配置回测
PYTHONPATH="/opt/tiger/qyd/qyd/src" \
python "/opt/tiger/qyd/qyd/src/backtester/run_simple_backtest.py"

# 完整参数回测（推荐）
PYTHONPATH="/opt/tiger/qyd/qyd/src" \
python "/opt/tiger/qyd/qyd/src/backtester/run_simple_backtest.py" \
  --config "/opt/tiger/qyd/qyd/conf/simple_backtester_config.json" \
  --prediction-path "/opt/tiger/qyd/qyd/outputs/xgb_cross_sectional_dataset/pred_test.parquet" \
  --price-path "/opt/tiger/qyd/qyd/data/processd_data/wide_table_daily_bars" \
  --benchmark-path "/opt/tiger/qyd/qlib_data_cn/features" \
  --output-dir "/opt/tiger/qyd/qyd/outputs/simple_backtest/hs300_open" \
  --start-time "2023-01-01" \
  --end-time "2025-12-31" \
  --benchmark "000300.SH" \
  --deal-price "open" \
  --topk 50 \
  --n-drop 5 \
  --account 1000000 \
  --open-cost 0.0005 \
  --close-cost 0.0015 \
  --slippage 0.001 \
  --min-cost 5 \
  --limit-threshold 0.095 \
  --lot-size 100 \
  --forbid-sell-limit-down \
  --plot
```

### IC验证

```bash
PYTHONPATH="/opt/tiger/qyd/qyd/src" \
python -m utils.ic_validator.run_ic_validation \
  --pred-path "/opt/tiger/qyd/qyd/outputs/xgb_cross_sectional_dataset/pred_test.parquet" \
  --save-dir "/opt/tiger/qyd/qyd/outputs/ic_validation"
```

### 运行测试

```bash
# 运行所有测试
PYTHONPATH="/opt/tiger/qyd/qyd/src" pytest "/opt/tiger/qyd/qyd/test/" -v

# 单独运行测试
PYTHONPATH="/opt/tiger/qyd/qyd/src" pytest "/opt/tiger/qyd/qyd/test/test_xgboost_trainer.py" -v
PYTHONPATH="/opt/tiger/qyd/qyd/src" pytest "/opt/tiger/qyd/qyd/test/test_parquet_loader.py" -v
PYTHONPATH="/opt/tiger/qyd/qyd/src" pytest "/opt/tiger/qyd/qyd/test/test_preparatory_backtester.py" -v
```

---

## 核心模块说明

### 数据获取模块

**位置**：`src/utils/dataset_fetcher/`

**核心文件**：
- `daily_bars_fetcher.py` - 日线数据获取器，从Tushare获取A股日线行情
- `adj_factors_fetcher.py` - 复权因子获取器
- `fundamentals_fetcher.py` - 财务数据获取器
- `moneyflow_fetcher.py` - 资金流数据获取器
- `industry_fetcher.py` - 行业数据获取器
- `namechange_fetcher.py` - 名称变更（ST标记）获取器
- `suspend_d_fetcher.py` - 停牌数据获取器

**功能特点**：
- 支持断点续传，通过`*_fetch_progress.json`记录已完成日期
- 按交易日批量获取，避免内存溢出
- 自动重试机制，应对API不稳定
- 数据按`year=YYYY/month=MM/YYYYMMDD.parquet`分区存储

### 数据清洗模块

**位置**：`src/utils/dataset_cleaner/`

**清洗规则**：
- 检查`ts_code + trade_date`唯一性，去重
- 检查OHLC（开高低收）缺失值，剔除异常行
- 允许`pre_close`、`change`、`pct_chg`缺失
- 原始数据只读，清洗结果写入独立目录

### 数据处理模块

**位置**：`src/utils/dataset_processor/`

**宽表构建流程**：
1. 主表：清洗后的日线数据
2. 合并ST标记（`no_st_stock`, `st_stock`, `star_st_stock`）
3. 合并停牌标记（`is_suspect`）
4. 合并复权因子（`adj_factor`）
5. 合并财务数据（Point-in-Time，使用最新已披露数据）
6. 合并资金流数据
7. 合并行业One-Hot编码

### 特征生成模块

**位置**：`src/utils/features_generator/`

**价量因子（18个）**：
- 动量因子：`ret_5`, `ret_10`, `ret_20`, `ret_60`
- 均线偏离：`ma5_bias`, `ma10_bias`, `ma20_bias`, `ma60_bias`
- 波动率：`volatility_5`, `volatility_20`, `volatility_60`
- 振幅：`amplitude`, `amplitude_5`, `amplitude_20`
- 量比：`vol_ratio_5`, `vol_ratio_20`
- 价量相关性：`corr_price_vol_5`, `corr_price_vol_20`

**基本面因子**：
- 直接因子：`roe`, `roa`, `or_yoy`, `gross_margin`, `debt_to_assets`, `eps`, `bps`
- 衍生因子：`roe_roa_gap`, `roe_rank`, `roa_rank`, `revenue_yoy_rank`

**资金流因子（13个）**：
- `lg_net_inflow`, `elg_net_inflow`, `main_net_inflow`, `main_net_ratio`
- `retail_net_inflow`, `retail_ratio`, `main_retail_diff`
- `main_net_5`, `main_net_10`, `main_net_20`
- `mf_ma5`, `mf_ma20`, `mf_acceleration`

**行业因子（8个）**：
- `industry_ret_1`, `industry_ret_5`, `industry_ret_20`
- `relative_strength_5`, `relative_strength_20`
- `roe_ind_neutral`, `roa_ind_neutral`, `revenue_yoy_ind_neutral`

### 横截面处理模块

**位置**：`src/utils/cross_sectional_processor/`

**处理流程**：
1. 按交易日分组，对每个因子进行缩尾处理（Winsorization，去除极端值）
2. 按交易日分组，对每个因子进行Z-Score标准化
3. 输出列命名为`{factor}_cc_processed`

### 标签生成模块

**位置**：`src/utils/label_generator/`

**支持的标签（12种）**：
- 收益率标签：`label_1d`, `label_2d`, `label_3d`, `label_5d`, `label_10d`, `label_20d`
- 排名标签：`label_rank_1d`, `label_rank_2d`, `label_rank_3d`, `label_rank_5d`, `label_rank_10d`, `label_rank_20d`

**标签计算方法**：
1. 使用全历史最新复权因子构建统一复权价格序列
2. 计算远期收益率：`label_hd = adj_close(t+h) / adj_close(t) - 1`
3. 排名标签为同日横截面百分位排名

### 数据加载模块

**位置**：`src/utils/dataset_generator/`

**功能**：
- 实现Qlib的`DataLoader`接口，可直接用于Qlib的`DataHandlerLP`和`DatasetH`
- 支持从多个Parquet数据源合并特征和标签
- 支持61个特征列的灵活配置
- 输出格式：MultiIndex(datetime, instrument)，MultiIndex列(feature/*, label/*)

### 模型训练模块

**位置**：`src/trainer/`

**训练流程**：
```
ParquetLoader → DataHandlerLP → DatasetH → XGBRegressor
```

**关键特性**：
- 自动GPU检测和回退（GPU失败自动切换CPU）
- 支持训练/验证/测试三段时间划分
- 评估指标：RMSE、IC（信息系数）、RankIC（秩相关系数）
- 模型参数可配置（`n_estimators`, `max_depth`, `learning_rate`等）

### 回测模块

**位置**：`src/backtester/`

#### SimpleBacktester 核心特性

**交易规则模拟**：
- T+1交易规则（按lot记录买入日期，当日买入不可当日卖出）
- 100股整数倍交易（可配置`lot_size`）
- 涨跌停限制（默认禁止买入涨跌停股票）
- 支持`open`或`close`作为成交价

**成本模拟**：
- 买入手续费：`open_cost`（默认0.05%）
- 卖出手续费/印花税：`close_cost`（默认0.15%）
- 单笔最低费用：`min_cost`（默认5元）
- 滑点：`slippage`（默认0.1%）

**策略类型**：
- TopK策略：选择预测分数最高的K只股票
- Dropout策略：每次调仓卖出最弱的N只，买入新的高分股票
- 调仓频率：日/周/月/每N个交易日

**绩效指标**：
- 收益率：总收益、年化收益、日胜率
- 风险指标：年化波动率、最大回撤、Sharpe、Sortino、Calmar
- 相对指标：Alpha、Beta、Tracking Error、Information Ratio
- 交易统计：换手率、交易次数、总成本

### IC验证模块

**位置**：`src/utils/ic_validator/`

**验证步骤**：
1. IC分析：计算每日IC、IC均值、IC标准差、ICIR
2. 分组收益：按因子分位数分组，计算各组收益
3. 多空回测：做多高分组，做空低分组
4. 换手率分析：计算各组调仓换手率
5. 分数分布：分析预测分数的分布特征

---

## 配置文件说明

### Parquet加载器配置

**文件**：`conf/parquet_loader_config.json`

```json
{
  "daily_bars_dir": "/opt/tiger/qyd/qyd/data/cross_sectional_processd_data/wide_table_daily_bars",
  "price_volume_factors_dir": "/opt/tiger/qyd/qyd/data/cross_sectional_processd_data/price_volume_factors",
  "moneyflow_factors_dir": "/opt/tiger/qyd/qyd/data/cross_sectional_processd_data/moneyflow_factors",
  "fundamental_factors_dir": "/opt/tiger/qyd/qyd/data/cross_sectional_processd_data/fundamental_factors",
  "industry_factors_dir": "/opt/tiger/qyd/qyd/data/cross_sectional_processd_data/industry_factors",
  "labels_dir": "/opt/tiger/qyd/qyd/data/generated_label/daily_labels",
  "feature_cols": [...],  // 61个特征列
  "label_name": "label_5d",
  "include_label": true,
  "dropna_label": true,
  "keep_original_code": true
}
```

### XGBoost训练器配置

**文件**：`conf/xgboost_trainer_config.json`

```json
{
  "loader_config_path": "/opt/tiger/qyd/qyd/conf/parquet_loader_config.json",
  "instruments": "all",
  "start_time": "2000-01-01",
  "end_time": "2025-12-31",
  "output_dir": "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset",
  "model_dir": "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/models",
  "model_filename": "xgboost_label_5d.json",
  "label_name": "label_5d",
  "prefer_gpu": true,
  "segments": {
    "train": ["2000-01-01", "2020-12-31"],
    "valid": ["2021-01-01", "2022-12-31"],
    "test": ["2023-01-01", "2025-12-31"]
  },
  "model_params": {
    "n_estimators": 100,
    "max_depth": 12,
    "learning_rate": 0.01,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": 8
  }
}
```

### 简单回测器配置

**文件**：`conf/simple_backtester_config.json`

```json
{
  "prediction_path": "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet",
  "price_path": "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/wide_table_daily_bars",
  "benchmark_path": "/opt/tiger/qyd/qlib_data_cn/features",
  "output_dir": "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/simple_backtest/0611_2",
  "start_time": "2023-01-01",
  "end_time": "2025-12-31",
  "account": 1000000.0,
  "benchmark": "000300.SH",
  "freq": "5d",
  "signal_delay": 1,
  "price_adjustment": "qfq",
  "strategy": {
    "topk": 50,
    "n_drop": 2
  },
  "exchange_kwargs": {
    "deal_price": "open",
    "open_cost": 0.0005,
    "close_cost": 0.0015,
    "min_cost": 5.0,
    "slippage": 0.0010,
    "limit_threshold": 0.095,
    "lot_size": 100,
    "forbid_buy_limit_up": true,
    "forbid_buy_limit_down": true,
    "forbid_sell_limit_down": true
  }
}
```

---

## 特征工程

### 特征总数：61个

| 类别 | 数量 | 说明 |
|------|------|------|
| 基础行情 | 14 | open, high, low, close, pre_close, change, pct_chg, vol, amount, adj_factor, ST标记(3), is_suspect |
| 基本面 | 7 | roe, roa, revenue_yoy, debt_ratio, gross_margin, eps, bps |
| 价量因子 | 18 | 动量、均线偏离、波动率、振幅、量比、价量相关性 |
| 资金流因子 | 13 | 主力净流入、散户净流入、资金流加速度等 |
| 行业因子 | 8 | 行业收益、相对强弱、行业中性化指标 |
| 衍生因子 | 1 | roe_roa_gap |

### 特征处理流程

```
原始特征 → 横截面缩尾(Winsorize) → 横截面Z-Score标准化 → 训练特征
```

---

## 常见问题

### Q: 如何切换训练目标？

A: 修改三个配置文件：
1. `parquet_loader_config.json`：`"label_name": "label_rank_5d"`
2. `xgboost_trainer_config.json`：`"model_filename": "xgboost_label_rank_5d.json"`
3. `xgboost_inferencer_config.json`：`"model_path"`和`"columns.label"`

### Q: 如何添加新特征？

A:
1. 在对应的`*_feature_generator.py`中添加因子计算逻辑
2. 在`*_cross_sectional_processor.py`中添加横截面处理
3. 在`parquet_loader_config.json`的`feature_cols`中添加新列名

### Q: 回测时买入股票数量明显少于topk？

A: 可能原因：
- 涨跌停限制过滤了部分股票
- 股票价格较高，按100股取整后现金不足
- 手续费和滑点导致可买股数下降
- 当天行情缺失或成交量为0

---

## 注意事项

### 数据安全
- 所有数据处理步骤都遵循"原始数据只读"原则
- 处理结果写入独立目录，不会覆盖原始数据
- 建议定期备份`data/raw_data`目录

### 标签泄露控制
- 标签计算使用全历史最新复权因子统一复权，避免未来函数
- 排名标签仅使用同日横截面数据
- 特征和标签计算分离，避免训练时引入未来信息

### 回测真实性
- `signal_delay=1`：t日的预测信号只能用于t+1日及以后的交易，避免未来函数
- 支持前复权（qfq）价格调整，消除分红配股影响
- T+1规则严格执行，按lot记录买入日期

### 性能优化
- Parquet分区存储，支持按日期快速筛选
- PyArrow列式存储，读取效率高
- 数据处理支持分批处理，降低内存占用
- XGBoost支持GPU加速

### 可扩展性
- 新增因子：在`features_generator`中添加新的生成器类
- 新增模型：在`trainer`中添加新的训练器类
- 新增回测规则：在`backtester`中扩展`SimpleBacktester`

### 实验管理
- 建议每次实验使用独立的`output_dir`
- 配置文件随结果一起保存，便于复现
- 日志文件记录完整的处理过程

---

## 下一步建议

1. **扩展因子库**：可以添加更多技术指标、宏观因子、情绪因子等
2. **模型优化**：尝试LightGBM、CatBoost、深度学习等其他模型
3. **风险模型**：加入风险平价、最大分散化等投资组合优化
4. **实盘对接**：开发交易接口，实现从回测到实盘的过渡
5. **监控告警**：添加模型性能监控、数据质量监控
6. **参数优化**：使用贝叶斯优化等方法进行超参数调优
7. **多策略融合**：开发多个策略并进行组合优化

---

## 许可证

本项目仅供学习和研究使用，不构成任何投资建议。投资有风险，入市需谨慎。
