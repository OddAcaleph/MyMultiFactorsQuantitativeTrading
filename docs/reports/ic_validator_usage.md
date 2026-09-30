# IC Validator 使用说明

本文档说明 `/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/utils/ic_validator` 中的 IC 验证工具代码结构、运行方式，以及样例输出目录 `/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/ic_validation_csv_test` 中各文件的含义。

## 1. 工具目标

`ic_validator` 用于对单个或多个特征进行 qlib 风格的因子有效性验证。当前默认串联以下 5 类验证：

1. 每日 IC / Rank IC / ICIR
2. 分层收益分析（Group Return）
3. Long-Short 分析
4. Turnover 分析
5. Score Distribution 分析

统一入口为 `ICValidator`，CLI 脚本为 `run_ic_validation.py`。

## 2. 代码结构

代码目录：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/utils/ic_validator
├── __init__.py
├── base.py
├── group_return.py
├── ic_analysis.py
├── long_short_backtest.py
├── run_ic_validation.py
├── score_distribution.py
├── turnover.py
└── validator.py
```

### 2.1 `base.py`

核心职责：

- 定义 `ValidationResult`。
- 定义 `BaseICValidationStep`。
- 将输入数据规范化为 qlib 常用结构：`MultiIndex(datetime, instrument)`。
- 提供本地测试数据加载函数 `load_local_test_pred_label`。

关键逻辑：

- `BaseICValidationStep.prepare_pred_label` 会检查输入是否为 `MultiIndex`，并抽取 `score_col` 和 `label_col` 两列。
- `load_local_test_pred_label` 会从分区 parquet 目录读取特征和标签，按 `trade_date + ts_code` 合并，并转换成：

```text
index: MultiIndex(datetime, instrument)
columns: [feature_col, label_col]
```

### 2.2 `ic_analysis.py`

类：`DailyICAnalyzer`

功能：

- 计算每日 IC。
- 计算每日 Rank IC。
- 汇总 `ic_mean`、`ic_std`、`icir`、`positive_rate`、`count`。
- 对每日 IC 序列调用 qlib 的 `risk_analysis`。

qlib API 使用：

```python
from qlib.contrib.eva.alpha import calc_ic
from qlib.contrib.evaluate import risk_analysis
```

IC 计算方式：

```python
ic, rank_ic = calc_ic(pred, label, date_col="datetime", dropna=False)
```

`risk_analysis` 对每日 IC 序列计算：

```text
mean = daily_ic.mean()
std = daily_ic.std(ddof=1)
annualized_return = mean * 238       # freq="day" 时 qlib 使用 238
information_ratio = mean / std * sqrt(238)
max_drawdown = min(cumsum(IC) - cummax(cumsum(IC)))
```

注意：这里的 `risk_analysis` 原本是收益序列分析函数；用于 IC 序列时，含义是累计 IC 曲线上的 qlib 风险统计。

### 2.3 `group_return.py`

类：`GroupReturnAnalyzer`

功能：

- 每天按 score 从高到低排序。
- 分成 `Group1 ~ GroupN`。
- 计算每组平均 label。
- 计算：
  - `long-short = Group1 - GroupN`
  - `long-average = Group1 - 当天全市场平均 label`
- 对 `long-short` 和 `long-average` 调用 qlib `risk_analysis`。

说明：qlib 0.9.7 中完整 N 组分层收益逻辑主要位于私有 report helper：

```python
qlib.contrib.report.analysis_model.analysis_model_performance._group_return
```

但该函数偏画图用途，依赖 plotly，返回 figure 而不是原始 `Group1 ~ GroupN` DataFrame。因此本工具保留了与 qlib `_group_return` 公式一致的本地 DataFrame 计算，以便后续保存 CSV 和做汇总分析。

### 2.4 `long_short_backtest.py`

类：`LongShortBacktestAnalyzer`

功能：

- 默认通过 qlib 公共 alpha API 计算 long-short：

```python
from qlib.contrib.eva.alpha import calc_long_short_return
```

- qlib 的 `calc_long_short_return` 返回 `(long - short) / 2` 和市场平均收益，因此本工具会将第一项乘以 2，输出常规 long-short spread。
- 如果设置 `--use-qlib-exchange`，则尝试调用：

```python
qlib.contrib.evaluate.long_short_backtest
```

该模式需要 qlib quote provider / exchange 环境已经正确初始化。

### 2.5 `turnover.py`

类：`TurnoverAnalyzer`

功能：

- 计算 Top 组和 Bottom 组的换手率。
- 当前公式遵循 qlib report 中 `_pred_turnover` 的逻辑：

```text
turnover = 1 - 当前组与上一期同组重合数量 / 当前组数量
```

输出：

- `Top`
- `Bottom`

### 2.6 `score_distribution.py`

类：`ScoreDistributionAnalyzer`

功能：

- 统计特征整体分布。
- 统计每日 score 分布。
- 输出 histogram。

主要输出字段：

- `overall`
- `daily_mean_summary`
- `daily_std_summary`

### 2.7 `validator.py`

类：`ICValidator`

这是所有分析类的统一编排入口。

默认串联步骤在 `ICValidator._default_steps` 中定义：

```python
return {
    "ic": DailyICAnalyzer(...),
    "group_return": GroupReturnAnalyzer(...),
    "long_short": LongShortBacktestAnalyzer(...),
    "turnover": TurnoverAnalyzer(...),
    "score_distribution": ScoreDistributionAnalyzer(...),
}
```

主要接口：

```python
validator.validate(pred_label, save_dir=output_dir)
validator.async_validate(pred_label, save_dir=output_dir)
```

保存逻辑：

- `metrics.json`：所有 metrics 汇总。
- `summary.csv`：核心指标摘要表，由 CLI 额外保存。
- 各步骤明细数据：保存为 CSV，不保存 parquet。

### 2.8 `run_ic_validation.py`

命令行入口脚本。

输入参数：

- `--feature-path`：特征 parquet 文件或分区目录。
- `--feature-col`：特征列名。
- `--label-path`：标签 parquet 文件或分区目录。
- `--label-col`：标签列名。
- `--output-dir`：输出目录。
- `--max-files`：读取多少个最新匹配日度 parquet 分片；`<=0` 表示读取全部。
- `--groups`：分层数量，默认 5。
- `--topk`：Long-Short topk，默认 50。
- `--turnover-lag`：turnover lag，默认 1。
- `--freq`：传给 qlib `risk_analysis` 的频率，默认 `day`。
- `--reverse`：是否反转 score 排序方向。
- `--use-qlib-exchange`：是否使用 qlib exchange 版本的 long-short backtest。
- `--async-run`：是否异步执行。

## 3. 运行示例

样例特征：

```text
feature_path = /opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cross_sectional_processd_data/industry_factors
feature_col  = industry_ret_1_cc_processed
```

样例标签：

```text
label_path = /opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/generated_label/daily_labels
label_col  = label_rank_1d
```

运行命令：

```bash
python /opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/utils/ic_validator/run_ic_validation.py \
  --feature-path /opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cross_sectional_processd_data/industry_factors \
  --feature-col industry_ret_1_cc_processed \
  --label-path /opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/generated_label/daily_labels \
  --label-col label_rank_1d \
  --output-dir /opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/ic_validation_csv_test \
  --topk 50 \
  --groups 5
```

如果只想读取最近 N 个匹配分片，例如最近 20 个交易日：

```bash
--max-files 20
```

如果读取全部历史：

```bash
--max-files 0
```

## 4. 样例输出目录结构

样例输出路径：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/ic_validation_csv_test/industry_ret_1_cc_processed
```

目录结构：

```text
industry_ret_1_cc_processed/
├── metrics.json
├── summary.csv
└── industry_ret_1_cc_processed/
    ├── group_return_cumulative.csv
    ├── group_return_group_returns.csv
    ├── ic_daily_ic.csv
    ├── ic_monthly_ic.csv
    ├── long_short_cumulative.csv
    ├── long_short_long_short_returns.csv
    ├── score_distribution_daily_distribution.csv
    ├── score_distribution_histogram.csv
    └── turnover_turnover.csv
```

说明：当前输出目录中有一层 feature 名称子目录，这是 `ICValidator.save_results` 按 feature 维度保存明细数据的结果。对于单特征验证，路径形如：

```text
<output-dir>/<feature-col>/<feature-col>/*.csv
```

## 5. `summary.csv` 说明

样例文件：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/ic_validation_csv_test/industry_ret_1_cc_processed/summary.csv
```

样例内容：

```csv
feature,ic_mean,ic_std,icir,rank_ic_mean,rank_icir,group_top_minus_bottom_mean,long_short_mean
industry_ret_1_cc_processed,0.0015204358429582285,0.140470581624577,0.010823873763274929,0.003501002084602079,0.025541981014473348,0.0031661385274577847,-0.0025983959573820546
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `feature` | 特征名称 |
| `ic_mean` | 日度 IC 均值 |
| `ic_std` | 日度 IC 样本标准差 |
| `icir` | `ic_mean / ic_std`，非年化 ICIR |
| `rank_ic_mean` | 日度 Rank IC 均值 |
| `rank_icir` | `rank_ic_mean / rank_ic_std`，非年化 Rank ICIR |
| `group_top_minus_bottom_mean` | 分层收益中 `Group1 - GroupN` 的均值 |
| `long_short_mean` | Long-Short 序列均值 |

## 6. `metrics.json` 说明

样例文件：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/ic_validation_csv_test/industry_ret_1_cc_processed/metrics.json
```

该文件包含所有 validator 的 metrics，结构大致为：

```json
{
  "features": {
    "industry_ret_1_cc_processed": {
      "ic": {"metrics": {...}},
      "group_return": {"metrics": {...}},
      "long_short": {"metrics": {...}},
      "turnover": {"metrics": {...}},
      "score_distribution": {"metrics": {...}}
    }
  },
  "summary": {
    "table": [...]
  }
}
```

### 6.1 `ic.metrics.daily`

样例：

```json
"IC": {
  "ic_mean": 0.0015204358429582285,
  "ic_std": 0.140470581624577,
  "icir": 0.010823873763274929,
  "positive_rate": 0.5053999325008437,
  "count": 5926
}
```

含义：

- `ic_mean`：每日 IC 均值。
- `ic_std`：每日 IC 样本标准差。
- `icir`：非年化 ICIR，即 `ic_mean / ic_std`。
- `positive_rate`：每日 IC 大于 0 的比例。
- `count`：有效 IC 天数。

### 6.2 `ic.metrics.risk_analysis`

这是对每日 IC / Rank IC 序列调用 qlib `risk_analysis` 后得到的结果。

在 `freq="day"` 下，qlib 使用的年化因子是 238：

```text
mean = daily_ic.mean()
std = daily_ic.std(ddof=1)
annualized_return = mean * 238
information_ratio = mean / std * sqrt(238)
max_drawdown = min(cumsum(IC) - cummax(cumsum(IC)))
```

注意：`risk_analysis` 原本用于收益序列；用于 IC 序列时，`max_drawdown` 是累计 IC 曲线的绝对回撤，不是百分比回撤。

### 6.3 `group_return.metrics`

主要字段：

- `mean_return_by_group`：每组平均 label。
- `cumulative_return_by_group`：每组 label 累计和。
- `monotonicity.is_decreasing_from_group1`：组收益均值是否从 Group1 开始单调递减。
- `monotonicity.top_minus_bottom_mean`：`Group1 - GroupN` 均值。
- `long-short_risk`：对 `long-short` 序列调用 qlib `risk_analysis`。
- `long-average_risk`：对 `long-average` 序列调用 qlib `risk_analysis`。

### 6.4 `long_short.metrics`

主要字段：

- `backend`：当前使用的计算后端。
  - `label_based_fallback` 表示使用 qlib alpha API `calc_long_short_return` 对 label 序列做 long-short 分析。
  - 如果启用 `--use-qlib-exchange` 且 qlib provider 正常，则可使用 qlib exchange backtest。
- `risk_analysis.long_short`：对 long-short 序列调用 qlib `risk_analysis`。
- `risk_analysis.market_average`：对市场平均 label 序列调用 qlib `risk_analysis`。
- `mean_return`：long-short 和 market_average 的均值。
- `cumulative_return`：long-short 和 market_average 的累计和。

### 6.5 `turnover.metrics`

主要字段：

- `mean_turnover.Top`：Top 组平均换手。
- `mean_turnover.Bottom`：Bottom 组平均换手。
- `std_turnover.Top`：Top 组换手标准差。
- `std_turnover.Bottom`：Bottom 组换手标准差。
- `latest_turnover`：最后一个交易日的换手。
- `count`：有效 turnover 天数。

### 6.6 `score_distribution.metrics`

主要字段：

- `overall`：全样本 score 分布统计。
- `daily_mean_summary`：每日 score 均值序列的分布统计。
- `daily_std_summary`：每日 score 标准差序列的分布统计。

## 7. 明细 CSV 文件说明

明细 CSV 位于：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/ic_validation_csv_test/industry_ret_1_cc_processed/industry_ret_1_cc_processed
```

| 文件 | 内容 |
| --- | --- |
| `ic_daily_ic.csv` | 每日 IC 和 Rank IC |
| `ic_monthly_ic.csv` | 月度 IC 和 Rank IC 均值 |
| `group_return_group_returns.csv` | 每日 Group1~GroupN、long-short、long-average |
| `group_return_cumulative.csv` | 上述分层收益的累计和 |
| `long_short_long_short_returns.csv` | 每日 long_short 和 market_average |
| `long_short_cumulative.csv` | long_short 和 market_average 的累计和 |
| `turnover_turnover.csv` | 每日 Top / Bottom turnover |
| `score_distribution_daily_distribution.csv` | 每日 score 的 describe 统计 |
| `score_distribution_histogram.csv` | score 直方图分桶统计 |

## 8. Python API 用法

除了命令行脚本，也可以直接使用 Python API。

```python
from utils.ic_validator import ICValidator
from utils.ic_validator.base import load_local_test_pred_label

pred_label = load_local_test_pred_label(
    feature_dir="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cross_sectional_processd_data/industry_factors",
    label_dir="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/generated_label/daily_labels",
    feature_col="industry_ret_1_cc_processed",
    label_col="label_rank_1d",
    max_files=20,
)

validator = ICValidator(
    label_col="label_rank_1d",
    feature_cols=["industry_ret_1_cc_processed"],
    groups=5,
    topk=50,
)

result = validator.validate(
    pred_label,
    save_dir="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/ic_validation_example/industry_ret_1_cc_processed",
)
```

异步版本：

```python
import asyncio

result = asyncio.run(
    validator.async_validate(
        pred_label,
        save_dir="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/ic_validation_example/industry_ret_1_cc_processed",
    )
)
```

## 9. 输入数据要求

CLI 当前假设输入为日度 parquet 分片目录或 parquet 文件集合，且每个分片至少包含：

特征文件：

```text
trade_date
ts_code
<feature_col>
```

标签文件：

```text
trade_date
ts_code
<label_col>
```

加载逻辑会按 `trade_date + ts_code` inner join。合并后空值会在各 validator 的 `prepare_pred_label` 中被清理。

## 10. 注意事项

1. `--max-files 0` 会读取全部匹配分片，数据量较大时运行时间和内存占用都会明显增加。
2. `risk_analysis` 的 day 年化因子是 qlib 默认的 238，不是 252。
3. `label_rank_1d` 是截面排名标签，数值通常在 0~1 附近；如果改用真实收益标签，Group Return / Long-Short 的数值尺度会变化。
4. `GroupReturnAnalyzer` 的完整分组收益表是按 qlib 私有 `_group_return` 的公式实现的本地 raw table 版本，因为 qlib 私有函数本身返回 plotly figure，不适合直接保存为指标 CSV。
5. 默认输出明细为 CSV，不再保存 parquet。

