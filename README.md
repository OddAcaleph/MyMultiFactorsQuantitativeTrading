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
  - [Walk-Forward滚动训练](#walk-forward滚动训练)
  - [模型推理](#模型推理)
  - [回测验证](#回测验证)
  - [IC验证](#ic验证)
  - [统一评估](#统一评估)
  - [组合优化](#组合优化)
  - [风险建模](#风险建模)
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
  - [统一评估框架](#统一评估框架)
  - [投资组合优化器](#投资组合优化器)
  - [风险模型](#风险模型)
- [配置文件说明](#配置文件说明)
- [特征工程](#特征工程)
- [常见问题](#常见问题)
- [注意事项](#注意事项)
- [许可证](#许可证)

---

## 项目简介

**MyMultiFactorsQuantitativeTrading** 是一个端到端的A股量化投研平台，支持多因子模型构建、XGBoost监督学习、以及贴近真实交易规则的回测验证。

- **版本**：v0.2.0
- **定位**：完整的量化投研流水线
- **支持市场**：A股
- **核心算法**：XGBoost 梯度提升树 + 均值-方差优化

### 版本历史

- **v0.0.1**：基础组件和流水线（数据获取、清洗、处理、特征生成、标签生成、XGBoost训练预测、简单回测）
- **v0.1.0**：Walk-Forward滚动训练、增强Alpha因子库（20+）、完整A股回测约束体系（行业/流动性/ST/次新股）、波动率加权、动态仓位管理、板块差异化涨跌停
- **v0.2.0**：高级标签体系（行业/市场超额Rank、Top20分类）、统一多维度评估框架（8大指标）、Barra风格风险模型、投资组合优化器（QP/OSQP）、风险平价策略、风险归因分析

---

## 项目特性

1. **完整的端到端流水线**：从数据获取到回测验证的完整量化投研流程
2. **贴近真实的回测引擎**：考虑A股T+1、涨跌停、100股整数倍等特有规则
3. **丰富的因子库**：66+个特征，涵盖价量、基本面、资金流、行业、增强Alpha等多个维度
4. **严格的泄露控制**：标签计算、回测信号延迟等机制避免未来函数
5. **Qlib生态兼容**：自定义数据加载器无缝对接Qlib框架
6. **完善的文档**：每个模块都有详细的使用文档和设计说明
7. **自动化流水线**：Shell脚本支持每日自动数据更新和模型推理
8. **可配置性强**：所有关键参数通过JSON配置文件管理
9. **断点续传**：数据获取、处理等步骤都支持断点续传，提高效率
10. **数据只读原则**：所有处理步骤都不修改原始数据，结果写入新目录
11. **Walk-Forward滚动训练**：支持滚动/扩展窗口训练，更贴近实盘的模型评估
12. **完整约束体系**：行业约束、流动性过滤、ST/停牌/次新股过滤、个股权重上限
13. **高级仓位管理**：波动率加权、目标波动率动态仓位、离散度动态仓位
14. **板块差异化涨跌停**：主板/创业板/科创板/北交所不同涨跌幅限制
15. **高级标签体系**：行业超额Rank、市场超额Rank、Top20上涨分类、行业超额Top20分类，覆盖5d/10d/20d三周期
16. **统一评估框架**：IC/Rank IC/ICIR、多头IC、分层收益、Top-K命中率、上涨捕获率、风险识别、组内IC、多头回测 8大维度
17. **Barra风格风险模型**：8个风格因子+行业因子，WLS因子收益估计，Ledoit-Wolf协方差缩减
18. **投资组合优化器**：基于CVXPY的QP/SOCP优化，支持行业/风格/波动/换手约束，OSQP/CLARABEL求解器
19. **风险平价策略**：风险贡献均衡的资产配置方法
20. **风险归因分析**：因子暴露、风格暴露、行业暴露、alpha留存率等完整归因指标

---

## 目录结构

```
MyMultiFactorsQuantitativeTrading/
├── conf/                          # 配置文件目录
│   ├── parquet_loader_config.json          # Parquet数据加载器配置
│   ├── xgboost_trainer_config.json         # XGBoost训练器配置
│   ├── xgboost_trainer_enhanced_alpha_config.json  # 增强Alpha训练配置
│   ├── xgboost_inferencer_config.json      # XGBoost推理器配置
│   ├── simple_backtester_config.json       # 简单回测器配置
│   ├── preparatory_backtester_config.json  # 预备回测器配置
│   ├── simple_backtest_grid_search_config.json      # 回测网格搜索配置
│   ├── xgboost_train_backtest_grid_search_config.json # 训练+回测网格搜索
│   ├── xgboost_train_backtest_grid_search_phase2.json # 第二阶段网格搜索
│   ├── xgboost_train_backtest_grid_search_phase3.json # 第三阶段网格搜索
│   ├── walk_forward_config.json            # Walk-Forward训练配置
│   ├── risk_model/                         # 风险模型配置
│   │   ├── risk_model_v1.json
│   │   └── risk_model_v2.json
│   ├── optimizer/                          # 优化器配置
│   │   ├── optimizer_v1_full.json
│   │   └── optimizer_v1_phase1.json
│   └── experiments/                        # 实验变体配置（不纳入主版本）
│       └── advanced_labels/                # 高级标签实验配置
│
├── src/                           # 源代码目录
│   ├── trainer/                   # 模型训练模块
│   │   ├── xgboost_trainer.py              # XGBoost训练器核心类
│   │   ├── xgboost_inferencer.py           # XGBoost推理器核心类
│   │   ├── walk_forward_trainer.py         # Walk-Forward滚动训练器
│   │   ├── run_xgboost_training.py         # 训练命令行入口
│   │   ├── run_xgboost_inference.py        # 推理命令行入口
│   │   ├── run_walk_forward_training.py    # Walk-Forward训练入口
│   │   ├── run_walk_forward_backtest.py    # Walk-Forward回测入口
│   │   └── run_xgboost_train_backtest_grid_search.py  # 训练+回测网格搜索
│   │
│   ├── backtester/                # 回测模块
│   │   ├── simple_backtester.py             # 简单回测器（真实交易规则+约束体系）
│   │   ├── preparatory_backtester.py        # 预备回测器（快速验证）
│   │   ├── run_simple_backtest.py          # 简单回测命令行入口
│   │   ├── run_preparatory_backtest.py      # 预备回测命令行入口
│   │   └── run_simple_backtest_grid_search.py  # 回测网格搜索入口
│   │
│   ├── evaluation/                # 统一评估框架
│   │   ├── evaluation_report.py            # 评估报告生成器（8大维度）
│   │   ├── metrics.py                      # 评估指标计算函数
│   │   └── run_evaluation.py               # 评估命令行入口
│   │
│   ├── optimizer/                 # 投资组合优化器
│   │   ├── portfolio_optimizer.py          # 优化器门面类
│   │   ├── objective_builder.py            # 优化目标构建器
│   │   ├── constraint_builder.py           # 约束构建器
│   │   ├── qp_solver.py                    # QP/SOCP求解器（OSQP/CLARABEL）
│   │   ├── risk_interface.py               # 风险数据接口
│   │   ├── risk_attribution.py             # 风险归因分析
│   │   ├── alpha_processor.py              # Alpha预处理（中性化等）
│   │   ├── candidate_pool.py               # 候选池筛选
│   │   ├── factor_timing.py                # 因子择时
│   │   ├── optimization_result.py          # 优化结果数据类
│   │   ├── diagnostics.py                  # 优化诊断工具
│   │   ├── risk_parity_strategy.py         # 风险平价策略
│   │   └── run_optimizer_backtest.py       # 优化器回测入口
│   │
│   ├── risk/                      # 风险模型
│   │   ├── risk_model.py                   # 风险模型门面类
│   │   ├── risk_exposure.py                # 风险暴露构建（8风格+行业）
│   │   ├── factor_return.py                # 因子收益估计（WLS横截面回归）
│   │   ├── factor_covariance.py            # 因子协方差（Ledoit-Wolf缩减）
│   │   ├── specific_risk.py                # 特异风险估计
│   │   ├── covariance_builder.py           # 股票协方差构建
│   │   ├── run_risk_model.py               # 风险模型命令行入口
│   │   └── validators/                     # 风险模型验证器
│   │       ├── exposure_validator.py
│   │       ├── covariance_validator.py
│   │       └── risk_forecast_validator.py
│   │
│   └── utils/                    # 工具模块
│       ├── config.py                       # 配置加载工具
│       ├── backtest_plotter.py             # 回测结果绘图工具
│       ├── dataset_fetcher/                # 数据获取模块
│       ├── dataset_cleaner/                # 数据清洗模块
│       ├── dataset_processor/              # 数据处理模块
│       ├── features_generator/             # 特征生成模块（含增强Alpha）
│       ├── cross_sectional_processor/      # 横截面处理模块
│       ├── label_generator/                # 标签生成模块
│       │   ├── daily_label_generator.py    # 基础日度标签（收益+排名）
│       │   └── advanced_label_generator.py # 高级标签（超额Rank+分类）
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
│       ├── daily_labels/          # 基础标签（12种）
│       └── advanced_labels/       # 高级标签（12种：超额Rank+分类）
│
├── scripts/                       # 脚本目录（可复用工具）
│   ├── data_pipeline/             # 数据流水线脚本
│   ├── experiments/               # 一次性实验脚本（不提交）
│   ├── run_walk_forward.py        # Walk-Forward训练+回测主脚本
│   ├── wf_grid_search.py          # WF网格搜索主框架
│   ├── wf_grid_search_scheduler.py # WF网格搜索并行调度器
│   ├── wf_grid_search_worker.py   # WF网格搜索Worker
│   ├── start_wf_grid_search.py    # WF网格搜索启动器
│   ├── analyze_enhanced_alpha_ic.py # 增强Alpha IC分析工具
│   ├── worst_performer_replacement_backtest.py # 最差表现替换策略
│   └── update_wide_table_st.py    # 宽表ST标记更新工具
│
├── docs/                          # 文档目录
│   └── reports/                   # 研究报告与使用说明
├── models/                        # 模型目录
├── output/                        # 实验输出目录
├── log/                           # 日志目录
├── test/                          # 测试目录
├── AGENTS.md                      # Agent开发指南
└── requirements.txt               # Python依赖
```

---

## 技术栈

| 类别 | 技术/框架 | 版本 | 用途 |
|------|----------|------|------|
| **编程语言** | Python | 3.x | 主要开发语言 |
| **机器学习** | XGBoost | 3.2.0 | 梯度提升树模型 |
| **量化框架** | pyqlib | 0.9.7 | 微软开源量化框架 |
| **优化求解** | CVXPY | - | 凸优化建模 |
| | OSQP | - | QP问题求解器 |
| | CLARABEL | - | SOCP问题求解器 |
| **数据处理** | pandas | - | 数据处理和分析 |
| | numpy | - | 数值计算 |
| **数据存储** | pyarrow | - | Parquet文件读写 |
| **数据源** | tushare | - | A股数据API |
| **可视化** | matplotlib | - | 回测结果绘图 |
| **测试框架** | pytest | - | 单元测试 |

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
bash scripts/data_pipeline/enhanced_alpha_factors_pipeline.sh 20260601 20260630
bash scripts/data_pipeline/label_calculation_pipeline.sh 20260601 20260630
```

#### 生成高级标签

```bash
# 生成行业超额Rank、市场超额Rank、Top20分类等高级标签
PYTHONPATH="src" python - <<'PY'
from utils.label_generator import AdvancedLabelGenerator

gen = AdvancedLabelGenerator(
    label_dir="data/generated_label/daily_labels",
    industry_dir="data/processd_data/wide_table_daily_bars",
    output_dir="data/generated_label/advanced_labels",
    horizons=(5, 10, 20),
)
stats = gen.process()
print(f"生成 {stats.output_files} 个文件，共 {stats.total_rows} 行")
print(f"标签列: {stats.columns}")
PY
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

### Walk-Forward滚动训练

```bash
# Walk-Forward训练 + 回测（推荐，更贴近实盘评估）
PYTHONPATH="src" python scripts/run_walk_forward.py \
  --config conf/walk_forward_config.json

# Walk-Forward网格搜索（并行）
PYTHONPATH="src" python scripts/start_wf_grid_search.py \
  --config conf/walk_forward_config.json \
  --n-workers 4
```

**特点**：
- 支持滚动窗口（rolling）和扩展窗口（expanding）两种模式
- 内存优化：一次性加载全量特征，按窗口切片训练
- 各窗口测试集预测拼接后进行连续回测
- 支持断点续跑和结果缓存

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

### 统一评估

```bash
# 8维度综合评估（IC + 分层收益 + Top-K命中率 + 上涨捕获 + 风险识别 + 回测）
PYTHONPATH="src" python src/evaluation/run_evaluation.py \
  --pred-path "output/.../pred_test.parquet" \
  --label-dir "data/generated_label/daily_labels" \
  --label-name "label_5d" \
  --output-dir "output/evaluation_report" \
  --run-backtest \
  --price-path "data/processd_data/wide_table_daily_bars" \
  --topk 50
```

**评估指标一览**：

| 维度 | 核心指标 | 说明 |
|------|---------|------|
| 整体IC | ic_mean, rank_ic, icir, positive_rate | 全样本信息系数 |
| 多头IC | long_side_ic_mean | 仅正收益股票中的排序能力 |
| 分层收益 | top_group_mean_return | 十分位分组最高组收益 |
| Top-K命中率 | beat_median_mean, positive_mean | Top50/100/200表现 |
| 上涨捕获率 | upside_capture_rate | 真实涨幅前10%被捕获比例 |
| 风险识别 | downside_hit_rate, precision, recall | 低分股暴跌识别能力 |
| 组内IC | return_decile_ic | 各收益分位内的排序能力 |
| 多头回测 | sharpe, max_drawdown, annualized_return | 真实交易规则下回测绩效 |

### 组合优化

```bash
# 运行优化器回测
PYTHONPATH="src" python src/optimizer/run_optimizer_backtest.py \
  --config "conf/optimizer/optimizer_v1_full.json"
```

### 风险建模

```bash
# 构建风险模型
PYTHONPATH="src" python src/risk/run_risk_model.py \
  --config "conf/risk_model/risk_model_v2.json"
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

**核心文件**：
- `daily_label_generator.py` - 基础日度标签生成器
- `advanced_label_generator.py` - 高级标签生成器（超额收益 + 分类）

#### 基础标签（12种）

- 收益率标签：`label_1d`, `label_2d`, `label_3d`, `label_5d`, `label_10d`, `label_20d`
- 排名标签：`label_rank_1d`, `label_rank_2d`, `label_rank_3d`, `label_rank_5d`, `label_rank_10d`, `label_rank_20d`

**标签计算方法**：
1. 使用全历史最新复权因子构建统一复权价格序列
2. 计算远期收益率：`label_hd = adj_close(t+h) / adj_close(t) - 1`
3. 排名标签为同日横截面百分位排名

#### 高级标签（12种）

基于基础收益标签和行业数据，生成更贴近选股目标的高级标签，独立存储于 `data/generated_label/advanced_labels/`：

| 标签名称格式 | 类型 | 说明 | 取值范围 |
|-------------|------|------|---------|
| `excess_industry_rank_{h}d` | 排序 | 行业超额收益全市场百分位排名 | [0, 1] |
| `excess_market_rank_{h}d` | 排序 | 市场超额收益全市场百分位排名 | [0, 1] |
| `up_top20_cls_{h}d` | 分类 | 未来涨幅前20%标记为1 | {0, 1} |
| `excess_industry_top20_cls_{h}d` | 分类 | 行业超额前20%标记为1 | {0, 1} |

> h ∈ {5, 10, 20}，共 4 类 × 3 周期 = 12 个高级标签。

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

**A股约束体系**（可配置）：
- 股票池过滤：ST股过滤、停牌过滤、次新股过滤、流动性过滤（20日均成交额）
- 行业约束：单行业权重上限、行业数量上限、行业分层抽样
- 仓位管理：等权/波动率加权、单票权重上限、成交额占比限制
- 动态仓位：目标波动率、分数阈值、截面离散度三种方法，可持有现金
- 板块涨跌停：主板(10%)/创业板科创板(20%)/北交所(30%)差异化处理

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

### 统一评估框架

**位置**：`src/evaluation/`

**核心类**：`EvaluationReport`（评估报告生成器）、`metrics.py`（指标计算函数集）

**8大评估维度**：

| 维度 | 计算函数 | 核心输出指标 |
|------|---------|------------|
| 整体IC表现 | `calc_overall_ic` | ic_mean, rank_ic_mean, icir, positive_rate |
| 多头IC | `calc_long_side_ic` | long_side_ic_mean（仅正收益样本） |
| 分层收益 | `calc_top_quantile_return` | top_group_mean_return, 各十分组均值 |
| Top-K命中率 | `calc_top_k_hit_rate` | beat_median_mean, positive_mean（K=50/100/200） |
| 上涨捕获率 | `calc_upside_capture` | upside_capture_rate（真实Top10%被预测捕获比例） |
| 风险识别 | `calc_downside_filter_score` | hit_rate, precision, recall（尾部风险识别） |
| 收益分位组内IC | `calc_return_decile_ic` | 各真实收益分位内的排序能力 |
| 多头回测（可选） | `_run_backtest` | sharpe, max_drawdown, annualized_return 等 |

**输出格式**：
- `evaluation_report.json` - 汇总指标（JSON格式）
- `evaluation_data/*.parquet` - 日度序列明细数据
- 可选回测报告与持仓明细

### 投资组合优化器

**位置**：`src/optimizer/`

**核心类**：`PortfolioOptimizer`（门面）、`ObjectiveBuilder`、`ConstraintBuilder`、`QPSolver`

**优化目标**（均值-方差框架）：
```
max  αᵀw - λ wᵀΣw - γ ||w - w_prev||₁
```

**约束类型**：
- **基础约束**：全投资约束（可选）、多空边界、单股权重上限
- **换手约束**：L1 换手率约束（辅助变量法）
- **行业约束**：单行业权重上限
- **风格约束**：风格因子暴露上下限
- **波动约束**：组合波动率上限（SOCP，wᵀΣw ≤ σ²）

**求解器**：
- 默认 OSQP（QP问题）
- 含波动约束时自动切换 CLARABEL（SOCP问题）

**输出结果**（`OptimizationResult`）：
- weights, trade_list（交易清单）
- expected_alpha, portfolio_volatility, turnover
- 风险归因：factor_exposure, style_exposure, industry_exposure
- alpha_retention（相对Top-N等权组合alpha留存率）

### 风险模型

**位置**：`src/risk/`

**核心类**：`RiskModel`（门面）、`RiskExposureBuilder`、`FactorReturnEstimator`、`FactorCovarianceEstimator`、`SpecificRiskEstimator`、`CovarianceBuilder`

**架构流程**：
```
风险暴露 → 因子收益估计 → 因子协方差 → 特异风险 → 股票协方差
```

**8个风格因子**：

| 因子 | 说明 |
|------|------|
| SIZE | 规模（对数市值） |
| BETA | 市场贝塔（252日回归） |
| MOMENTUM | 动量（12月-1月） |
| VALUE | 价值（B/P、E/P综合） |
| GROWTH | 成长（营收同比、ROE、毛利率变动） |
| VOLATILITY | 波动率（20/60日波动） |
| LIQUIDITY | 流动性（Amihud非流动性） |
| LEVERAGE | 杠杆（资产负债率） |

**关键技术**：
- WLS横截面回归估计因子收益（权重为20日均成交额平方根）
- Ledoit-Wolf缩减估计因子协方差
- 行业-规模分组中位数收缩特异风险
- PSD修正确保协方差矩阵正定性

**验证器**（`src/risk/validators/`）：暴露验证、协方差验证、风险预测验证

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

### 特征总数：66+个

| 类别 | 数量 | 说明 |
|------|------|------|
| 基础行情 | 14 | open, high, low, close, pre_close, change, pct_chg, vol, amount, adj_factor, ST标记(3), is_suspect |
| 基本面 | 7 | roe, roa, revenue_yoy, debt_ratio, gross_margin, eps, bps |
| 价量因子 | 18 | 动量、均线偏离、波动率、振幅、量比、价量相关性 |
| 资金流因子 | 13 | 主力净流入、散户净流入、资金流加速度等 |
| 行业因子 | 8 | 行业收益、相对强弱、行业中性化指标 |
| 衍生因子 | 1 | roe_roa_gap |
| 增强Alpha因子 | 5+ | MACD、RSI、52周新高距离、毛利率变化、营收同比加速度（完整20+因子在增强Alpha模块） |

### 增强Alpha因子

**位置**：`src/utils/features_generator/enhanced_alpha_feature_generator.py`

提供进攻性Alpha因子，补充原有防御性因子体系：

- **动量类**：12月/6月动量（skip 1月/2周）、52周新高距离、RSI、Williams %R、MACD、连涨连跌天数
- **量价类**：OBV能量潮、量价趋势、换手率、成交量动量
- **资金流类**：主力净流入5日变化、20日趋势、价量背离
- **基本面动量**：ROE环比变化、营收同比加速度、毛利率变化

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

1. **扩展因子库**：可以添加更多技术指标、宏观因子、情绪因子、另类数据因子等
2. **模型优化**：尝试LightGBM、CatBoost、深度学习（Transformer/MLP）等其他模型
3. **多模型集成**：Stacking、Bagging、加权融合等集成学习方法
4. **风险模型深化**：更多风格因子、非线性风险、条件风险预测
5. **优化器增强**：鲁棒优化、贝叶斯优化、在线学习、交易成本优化
6. **实盘对接**：开发交易接口，实现从回测到实盘的过渡
7. **监控告警**：添加模型性能监控、数据质量监控、风险监控
8. **参数优化**：使用贝叶斯优化等方法进行超参数调优
9. **多策略融合**：开发多个策略并进行组合优化、策略轮动
10. **Walk-Forward深化**：更多窗口配置、自适应窗口、在线学习
11. **约束体系完善**：更多交易规则模拟、融券做空、打新收益、融资融券
12. **高级标签扩展**：更多标签类型（分位数回归、排序学习、多任务学习）
13. **归因分析完善**：Brinson归因、因子收益归因、行业归因
14. **可视化平台**：Web界面的因子分析、回测分析、风险监控仪表盘

---

## 许可证

本项目仅供学习和研究使用，不构成任何投资建议。投资有风险，入市需谨慎。
