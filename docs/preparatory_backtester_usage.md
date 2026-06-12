# PreparatoryBacktester 使用说明

本文档说明如何使用：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/backtester/preparatory_backtester.py
```

该模块提供 `PreparatoryBacktester` 类，用于读取样本外预测结果 `pred_test.parquet`，按 qlib 风格的 `TopkDropoutStrategy` 思路做日频组合回测，并输出收益报表、持仓和风险分析结果。

## 1. 功能概览

`PreparatoryBacktester` 的核心逻辑位于 `src/backtester/preparatory_backtester.py`：

- 默认配置文件：`/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/preparatory_backtester_config.json`
- 读取预测文件：`prediction_path`
- 筛选 OOS 区间：优先使用 backtester 配置中的 `start_time` / `end_time`，为空时读取 trainer 配置中的 `segments.test`
- 策略逻辑：每天持有预测分数最高的 `topk` 只股票，每次调仓卖出得分最弱的 `n_drop` 只，再补入高分股票
- 收益计算：用预测文件中的 label 列作为下一期实现收益
- 交易成本：使用 `exchange_kwargs.open_cost + close_cost + impact_cost`
- 风险分析：调用 `qlib.contrib.evaluate.risk_analysis`

代码入口：

```python
from backtester import PreparatoryBacktester

backtester = PreparatoryBacktester()
result = backtester.run(save=True)
```

## 2. 输入数据要求

回测输入是一个 parquet 预测文件，默认由 XGBoost 训练或推理流程生成。

预测文件要求：

1. index 必须是 `MultiIndex(datetime, instrument)`。
2. 至少包含两列：
   - `pred`：模型预测分数，默认由 `columns.score` 指定。
   - `label`：实际收益或标签，默认由 `columns.label` 指定。
3. `datetime` 会被过滤到 OOS 回测区间内。

如果你的预测文件 label 列名不是 `label`，例如是 `label_5d`，需要修改配置：

```json
"columns": {
  "score": "pred",
  "label": "label_5d"
}
```

## 3. 默认配置文件

配置文件路径：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/preparatory_backtester_config.json
```

关键配置示例：

```json
{
  "trainer_config_path": "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/xgboost_trainer_config.json",
  "prediction_path": "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_daily_bars/pred_test.parquet",
  "output_dir": "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/preparatory_backtest",
  "start_time": null,
  "end_time": null,
  "account": 100000000.0,
  "benchmark": "market_mean",
  "freq": "day",
  "strategy": {
    "topk": 50,
    "n_drop": 5
  },
  "exchange_kwargs": {
    "open_cost": 0.0005,
    "close_cost": 0.0015,
    "impact_cost": 0.0050
  },
  "columns": {
    "score": "pred",
    "label": "label"
  },
  "save_positions": true
}
```

注意：如果默认 `prediction_path` 不存在，请改成实际预测文件路径。例如当前训练文档中常见输出是：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet
```

## 4. 回测 run 脚本

项目提供了命令行入口脚本：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/backtester/run_preparatory_backtest.py
```

查看参数：

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/backtester/run_preparatory_backtest.py" --help
```

### 4.1 使用默认配置运行

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/backtester/run_preparatory_backtest.py"
```

默认会读取：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/preparatory_backtester_config.json
```

并输出：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/preparatory_backtest/backtest_report.parquet
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/preparatory_backtest/positions.parquet
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/preparatory_backtest/analysis.json
```

### 4.2 指定预测文件和 label 列运行

如果默认配置中的 `prediction_path` 不存在，或预测文件中的 label 列不是 `label`，可以用参数覆盖：

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/backtester/run_preparatory_backtest.py" \
  --prediction-path "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet" \
  --output-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/preparatory_backtest" \
  --score-col "pred" \
  --label-col "label_5d"
```

### 4.3 指定回测时间区间

`start_time` 和 `end_time` 默认可从 trainer 配置的 `segments.test` 推导；如果需要指定区间：

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/backtester/run_preparatory_backtest.py" \
  --prediction-path "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet" \
  --start-time "2023-01-01" \
  --end-time "2025-12-31"
```

### 4.4 覆盖策略参数并生成回测图

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/backtester/run_preparatory_backtest.py" \
  --prediction-path "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet" \
  --output-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/preparatory_backtest_top100" \
  --score-col "pred" \
  --label-col "label_5d" \
  --topk 100 \
  --n-drop 10 \
  --plot
```

生成图表默认保存到：

```text
<output-dir>/backtest_plots.png
```

也可以用 `--plot-output` 指定图片路径。

### 4.5 常用参数

| 参数 | 含义 |
| --- | --- |
| `--config` | 指定 backtester 配置 JSON，默认是 `conf/preparatory_backtester_config.json` |
| `--trainer-config` | 指定 trainer 配置 JSON；当未设置回测起止日期时用于读取 `segments.test` |
| `--prediction-path` | 指定预测 parquet 文件 |
| `--output-dir` | 指定回测输出目录 |
| `--start-time` / `--end-time` | 指定回测起止日期 |
| `--score-col` | 指定预测分数字段，默认来自 `columns.score` |
| `--label-col` | 指定收益 / 标签字段，默认来自 `columns.label` |
| `--topk` | 持仓股票数量，覆盖 `strategy.topk` |
| `--n-drop` | 每次调仓卖出的弱势持仓数量，覆盖 `strategy.n_drop` |
| `--account` | 初始资金，覆盖 `account` |
| `--benchmark` | 基准，默认 `market_mean` |
| `--freq` | 传给 qlib `risk_analysis` 的频率，默认 `day` |
| `--no-save` | 只运行不保存文件 |
| `--no-positions` | 不保存 `positions.parquet` |
| `--plot` | 回测后生成 `backtest_plots.png` |
| `--plot-output` | 指定图表输出路径 |

## 5. Python API 运行方式

如果需要在 notebook 或其他 Python 代码中调用，可以直接使用 `PreparatoryBacktester` 类。

### 5.1 使用默认配置运行

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" python - <<'PY'
from backtester import PreparatoryBacktester

backtester = PreparatoryBacktester(
    config_path="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/preparatory_backtester_config.json",
)

result = backtester.run(save=True)

print("回测完成")
print("输出目录:", backtester.output_dir)
print("回测交易日数:", result["analysis"].get("summary", {}).get("trading_days"))
print("最终账户权益:", result["analysis"].get("summary", {}).get("final_account_value"))
PY
```

### 5.2 临时覆盖预测文件、输出目录和 label 列

如果你不想修改 JSON 配置，可以在初始化时通过 keyword 参数覆盖顶层配置：

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" python - <<'PY'
from backtester import PreparatoryBacktester

backtester = PreparatoryBacktester(
    config_path="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/preparatory_backtester_config.json",
    prediction_path="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet",
    output_dir="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/preparatory_backtest_custom",
    columns={"score": "pred", "label": "label_5d"},
)

result = backtester.run(save=True)

print("回测完成")
print("输出目录:", backtester.output_dir)
print("绩效摘要:", result["analysis"].get("summary", {}))
PY
```

### 5.3 临时覆盖回测时间区间

`start_time` 和 `end_time` 默认可从 trainer 配置的 `segments.test` 推导；如果需要指定区间：

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" python - <<'PY'
from backtester import PreparatoryBacktester

backtester = PreparatoryBacktester(
    config_path="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/preparatory_backtester_config.json",
    prediction_path="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet",
    start_time="2023-01-01",
    end_time="2025-12-31",
)

result = backtester.run(save=True)
print(result["analysis"].get("summary", {}))
PY
```

## 6. 输出文件说明

默认输出目录：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/preparatory_backtest
```

`run(save=True)` 会保存：

| 文件 | 内容 |
| --- | --- |
| `backtest_report.parquet` | 每日回测报表 |
| `positions.parquet` | 每日持仓权重；仅当 `save_positions=true` 且有持仓时保存 |
| `analysis.json` | qlib `risk_analysis` 结果和摘要指标 |

`backtest_report.parquet` 主要列：

| 列名 | 含义 |
| --- | --- |
| `return` | 扣除成本后的策略日收益 |
| `gross_return` | 未扣成本的持仓平均 label |
| `benchmark` | 基准收益；默认是当天全市场 label 均值 |
| `excess_return` | `return - benchmark` |
| `cost` | 当天调仓成本 |
| `turnover` | 当天换手率 |
| `account_value` | 账户权益曲线 |
| `holdings_count` | 当天有效持仓数量 |

`analysis.json` 主要包含：

- `return`：策略收益风险分析。
- `benchmark`：基准收益风险分析。
- `excess_return`：超额收益风险分析。
- `summary`：起止日期、交易日数、最终账户权益、平均换手、平均成本、平均持仓数。
- `config`：本次回测使用的配置。

### 6.1 `run()` 返回结果结构说明

除了落盘文件，`PreparatoryBacktester.run()` 本身还会返回一个字典：

```python
{
    "report": report_df,
    "positions": positions_df,
    "analysis": analysis,
}
```

含义如下：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `report` | `pd.DataFrame` | 每日回测报表，内容与 `backtest_report.parquet` 一致 |
| `positions` | `pd.DataFrame` | 每日持仓权重表，内容与 `positions.parquet` 一致 |
| `analysis` | `dict` | 风险分析和摘要信息，内容与 `analysis.json` 一致 |

因此如果你是在 Python 里调用：

```python
result = backtester.run(save=True)
```

可以直接通过：

```python
result["report"]
result["positions"]
result["analysis"]
```

拿到内存中的回测结果，而不必重新读文件。

### 6.2 `backtest_report.parquet` 数据结构说明

当前样例输出中，`backtest_report.parquet` 的结构是：

- index：`DatetimeIndex`，索引名为 `datetime`
- columns：

```text
return
gross_return
benchmark
excess_return
cost
turnover
account_value
holdings_count
```

字段解释：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `return` | `float` | 当日策略净收益率，等于 `gross_return - cost` |
| `gross_return` | `float` | 当日持仓股票 label 的等权平均值，表示扣成本前收益 |
| `benchmark` | `float` | 当日基准收益；默认 `market_mean` 表示全市场 label 均值 |
| `excess_return` | `float` | 当日超额收益，等于 `return - benchmark` |
| `cost` | `float` | 当日总成本，按换手率乘成本率估算 |
| `turnover` | `float` | 当日换手率，基于新旧权重差绝对值求和 |
| `account_value` | `float` | 按日复利累计后的账户净值 |
| `holdings_count` | `int` | 当日有效持仓数量 |

计算关系可简化理解为：

```text
gross_return = 当日选中股票的 label 均值
cost = turnover × (open_cost + close_cost + impact_cost)
return = gross_return - cost
account_value[t] = account_value[t-1] × (1 + return)
excess_return = return - benchmark
```

当前项目真实样例中，该文件的形态类似：

```text
index name: datetime
shape: (723, 8)
```

样例前 3 行可理解为：

```text
datetime    return   gross_return  benchmark  excess_return  cost    turnover  account_value   holdings_count
2023-01-03  0.0170   0.0240        0.0133     0.0037         0.0070  1.0       1.017009e+08    50
2023-01-04  0.0048   0.0062        0.0007     0.0041         0.0014  0.2       1.021883e+08    50
2023-01-05  0.0014   0.0028       -0.0052     0.0066         0.0014  0.2       1.023298e+08    50
```

其中：

- `2023-01-03` 是首个调仓日，所以 `turnover=1.0`，表示从空仓切到满仓。
- 后续如果只是局部调仓，`turnover` 会明显低于 1。
- `holdings_count=50` 对应默认 `topk=50`。

### 6.3 `positions.parquet` 数据结构说明

`positions.parquet` 用于记录每日持仓明细。当前样例结构为：

- index：`MultiIndex(datetime, instrument)`
- columns：

```text
weight
```

字段说明：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `datetime` | index | 交易日 |
| `instrument` | index | 股票代码 |
| `weight` | `float` | 当日该股票的组合权重 |

当前实现使用等权持仓，因此默认 `topk=50` 时，每只股票的权重通常约为：

```text
1 / 50 = 0.02
```

样例前几行类似：

```text
datetime    instrument   weight
2023-01-03  000032.SZ    0.02
2023-01-03  000045.SZ    0.02
2023-01-03  000419.SZ    0.02
```

如果某天因为可用 label 的股票不足，实际权重会自动按当日有效持仓数量重新均分。

### 6.4 `analysis.json` 数据结构说明

`analysis.json` 是最终汇总文件，顶层通常包含：

```json
{
  "return": {...},
  "benchmark": {...},
  "excess_return": {...},
  "summary": {...},
  "config": {...}
}
```

其中：

#### 6.4.1 `summary`

`summary` 是最常用的总览信息，当前样例类似：

```json
{
  "start_time": "2023-01-03",
  "end_time": "2025-12-31",
  "trading_days": 723,
  "final_account_value": 113245964976.88585,
  "mean_turnover": 0.20597510373443986,
  "mean_cost": 0.0014418257261410788,
  "mean_holdings_count": 50.0
}
```

字段说明：

| 字段 | 含义 |
| --- | --- |
| `start_time` | 实际回测起始交易日 |
| `end_time` | 实际回测结束交易日 |
| `trading_days` | 有效回测日数 |
| `final_account_value` | 回测结束时账户净值 |
| `mean_turnover` | 平均日换手率 |
| `mean_cost` | 平均日成本 |
| `mean_holdings_count` | 平均持仓股票数 |

#### 6.4.2 `return` / `benchmark` / `excess_return`

这三部分都是对收益序列调用 qlib `risk_analysis` 后的结果。通常会包含类似字段：

- `mean`
- `std`
- `annualized_return`
- `information_ratio`
- `max_drawdown`

它们分别表示：

| 区块 | 含义 |
| --- | --- |
| `return` | 对策略净收益序列做风险分析 |
| `benchmark` | 对基准收益序列做风险分析 |
| `excess_return` | 对超额收益序列做风险分析 |

#### 6.4.3 `config`

`config` 会原样记录本次回测使用的配置，便于追溯：

- 用了哪个 `prediction_path`
- 输出目录是什么
- `topk` / `n_drop` 是多少
- 成本参数如何设置
- `score` 和 `label` 字段分别是什么

建议在比较不同回测实验时，把 `analysis.json` 和对应的配置一起保存，便于后续复盘。

## 7. 生成回测图

项目还提供了 `BacktestPlotter`，可以基于 `backtest_report.parquet` 生成收益曲线图。

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" python - <<'PY'
from utils.backtest_plotter import BacktestPlotter

plotter = BacktestPlotter(
    report_path="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/preparatory_backtest/backtest_report.parquet",
    output_path="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/preparatory_backtest/backtest_plots.png",
)

info = plotter.plot(title="PreparatoryBacktester Performance")
print("图表已生成:", info["output_path"])
print("最大回撤信息:", info["max_drawdown"])
PY
```

输出图片：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/preparatory_backtest/backtest_plots.png
```

## 8. 常见问题排查

### 8.1 `Prediction file does not exist`

说明 `prediction_path` 配置的预测文件不存在。处理方式：

1. 先运行 XGBoost 训练或推理，生成 `pred_test.parquet`。
2. 或将 backtester 配置里的 `prediction_path` 改成实际文件路径。

### 8.2 `Prediction parquet must use MultiIndex(datetime, instrument)`

说明输入 parquet 的索引结构不符合要求。需要确保预测文件保存时使用：

```text
index: MultiIndex(datetime, instrument)
columns: pred, label
```

### 8.3 `Prediction file missing required columns`

说明 `columns.score` 或 `columns.label` 指向的列不存在。检查：

```bash
python - <<'PY'
import pandas as pd

path = "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet"
df = pd.read_parquet(path)
print(df.index.names)
print(df.columns.tolist())
PY
```

然后同步修改配置中的：

```json
"columns": {
  "score": "pred",
  "label": "实际存在的label列名"
}
```

### 8.4 `Backtester start/end are missing and trainer config has no test segment`

说明 backtester 配置里没有 `start_time` / `end_time`，且 trainer 配置里也没有 `segments.test`。处理方式二选一：

1. 在 backtester 配置里直接设置 `start_time` 和 `end_time`。
2. 确认 `trainer_config_path` 指向的 JSON 中存在 `segments.test`。

## 9. 推荐完整流程

1. 训练或推理生成预测文件：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet
```

2. 检查预测文件列名：

```bash
python - <<'PY'
import pandas as pd
path = "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet"
df = pd.read_parquet(path)
print("index names:", df.index.names)
print("columns:", df.columns.tolist())
print(df.head())
PY
```

3. 根据实际列名设置 `columns.label`。

4. 运行回测脚本：

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/backtester/run_preparatory_backtest.py" \
  --prediction-path "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet" \
  --score-col "pred" \
  --label-col "实际存在的label列名" \
  --plot
```

5. 查看：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/preparatory_backtest/analysis.json
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/preparatory_backtest/backtest_report.parquet
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/preparatory_backtest/positions.parquet
```

6. 可选：使用 `BacktestPlotter` 生成 `backtest_plots.png`。
