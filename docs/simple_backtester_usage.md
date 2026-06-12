# SimpleBacktester 使用说明

本文档说明如何使用更细化的 A 股回测模块：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/backtester/simple_backtester.py
```

以及对应命令行入口：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/backtester/run_simple_backtest.py
```

`SimpleBacktester` 基于当前项目已有的预测结果 parquet 和 `data/processd_data/wide_table_daily_bars` 日线宽表 parquet 做日频组合回测。相比 `PreparatoryBacktester`，它不再只用 label 均值近似收益，而是引入了更接近真实交易的现金、持仓、成交价、手续费、滑点、100 股整数倍和 T+1 约束；benchmark 仅从 `/opt/tiger/qyd/qlib_data_cn/features` 下的 qlib 格式指数数据读取，例如 `000300.SH` 会映射到 `features/sh000300`。

## 1. 功能概览

核心代码位于：

```text
src/backtester/simple_backtester.py
```

主要特性：

1. **TopK / Dropout 调仓**
   - 在调仓日根据预测分数排序；非调仓日只做持仓估值和收益记录，不买卖。
   - 目标持仓数量由 `strategy.topk` 控制。
   - 每次最多卖出弱势持仓 `strategy.n_drop` 只，再补入高分股票。
   - 调仓频率由 `freq` 控制，支持 `day` / `week` / `month`，也支持 `10d`、`20d` 这类每 N 个交易日调仓。

2. **真实行情成交**
   - 从 `data/processd_data/wide_table_daily_bars` 读取股票日线宽表。
   - 支持用 `open` 或 `close` 作为成交价。
   - 每天用收盘价做持仓市值和账户权益估值。

3. **涨跌停交易限制**
   - 默认禁止买入涨停股票。
   - 默认禁止买入跌停股票。
   - 可选禁止卖出跌停股票。
   - 涨跌停阈值由 `exchange_kwargs.limit_threshold` 控制，默认可设为 `0.095`。

4. **A 股 100 股整数倍**
   - 买入和卖出数量都会按 `lot_size` 向下取整。
   - 默认 `lot_size=100`。

5. **T+1 交易**
   - 每笔买入按 lot 记录 `buy_date`。
   - 当天买入的股票当天不可卖出。
   - `positions.parquet` 中会输出 `available_shares`，表示当天可卖股数。

6. **手续费和滑点**
   - 买入手续费：`open_cost`
   - 卖出手续费 / 印花税等：`close_cost`
   - 单笔最低费用：`min_cost`
   - 单边滑点：`slippage`
   - 买入成交价按 `raw_price * (1 + slippage)`。
   - 卖出成交价按 `raw_price * (1 - slippage)`。

7. **从 qlib features 直接读取 benchmark 指数**
   - 不再使用 `market_mean`。
   - 不再从 `data/A_stocks_all_data/index_data` 读取指数文件。
   - benchmark 从 `/opt/tiger/qyd/qlib_data_cn/features/<instrument>` 读取 `change.day.bin`。
   - 股票成交行情仍然从 `data/processd_data/wide_table_daily_bars` 读取。
   - 如果配置中仍是 `market_mean`，`SimpleBacktester` 会自动切换到 `000300.SH`。

8. **量化指标输出**
   - 收益、年化收益、年化波动率。
   - Sharpe、Sortino、Calmar。
   - 最大回撤、胜率。
   - Alpha、Beta、Tracking Error、Information Ratio、相关系数等。

## 2. 输入数据要求

### 2.1 预测文件 `prediction_path`

预测文件默认来自训练或推理流程，例如：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet
```

要求：

1. parquet index 必须是 `MultiIndex(datetime, instrument)`。
2. 至少包含预测分数字段，默认是 `pred`。
3. `datetime` 会被过滤到回测区间 `start_time` ~ `end_time`。

示例结构：

```text
index: MultiIndex(datetime, instrument)
columns: pred, label_5d, ...
```

检查预测文件：

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

### 2.2 股票日线行情 `price_path`

默认路径：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/wide_table_daily_bars
```

目录下通常是按日期分区的 parquet 文件：

```text
wide_table_daily_bars/
  year=2023/
    month=01/
      20230103.parquet
      20230104.parquet
```

每个日线 parquet 至少需要以下字段：

| 字段 | 含义 |
| --- | --- |
| `trade_date` | 交易日期，格式如 `20230103` |
| `ts_code` | 股票代码，如 `000001.SZ` |
| `open` | 开盘价 |
| `close` | 收盘价 |
| `pre_close` | 前收盘价 |
| `pct_chg` | 当日涨跌幅，单位是百分比，例如 `9.98` 表示 `9.98%` |
| `vol` | 成交量；小于等于 0 时视为不可交易 |
| `amount` | 成交额 |

### 2.3 benchmark 数据 `benchmark_path`

默认目录：

```text
/opt/tiger/qyd/qlib_data_cn/features
```

`benchmark_path` 指向 qlib 的 `features` 目录。股票交易行情仍然来自 `wide_table_daily_bars`，只有 benchmark 使用 qlib 格式数据。benchmark 代码会映射到 qlib instrument 目录，例如 `000300.SH` → `sh000300`、`000905.SH` → `sh000905`。至少需要文件：

| 文件 | 含义 |
| --- | --- |
| `features/<instrument>/change.day.bin` | benchmark 日收益率，qlib float32 二进制格式 |
| `features/<instrument>/close.day.bin` | benchmark 收盘价；当 `change.day.bin` 不存在时用于 `pct_change()` 回退 |
| `calendars/day.txt` | qlib 日频交易日历，通常位于 `features` 目录的上一级 |

benchmark 收益优先使用 qlib 的：

```text
features/<instrument>/change.day.bin
```

如果 `change.day.bin` 不存在，则回退为：

```text
features/<instrument>/close.day.bin 的 pct_change()
```

不会再用股票行做 `amount` 加权聚合，也不会用“排名前 300 / 第 301~800”之类的近似方式。如果某个回测交易日没有对应 benchmark 数据，会使用该交易日前最近一个可用 benchmark 收益；为了支持首个回测日缺失，加载 benchmark 时会额外向前读取 `benchmark_lookback_days` 天的数据。如果向前查找后仍没有任何可用 benchmark 数据，程序会直接报错，避免产出误导性 benchmark。

支持的 benchmark 写法包括：

| 写法 | 解释 |
| --- | --- |
| `000300.SH` | 沪深300 |
| `SH000300` | 沪深300 |
| `沪深300` | 沪深300 |
| `000001.SH` | 上证综指（仅当 qlib features 下存在 `sh000001` 时可用） |
| `SH000001` | 上证综指（仅当 qlib features 下存在 `sh000001` 时可用） |
| `上证综指` | 上证综指（仅当 qlib features 下存在 `sh000001` 时可用） |
| `000905.SH` | 中证500 |
| `SH000905` | 中证500 |
| `中证500` | 中证500 |

## 3. 配置参数说明

`SimpleBacktester` 使用自己的默认配置文件：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/simple_backtester_config.json
```

配置示例：

```json
{
  "prediction_path": "/path/to/pred_test.parquet",
  "price_path": "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/wide_table_daily_bars",
  "benchmark_path": "/opt/tiger/qyd/qlib_data_cn/features",
  "output_dir": "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/simple_backtest",
  "start_time": "2023-01-01",
  "end_time": "2025-12-31",
  "account": 1000000.0,
  "benchmark": "000300.SH",
  "benchmark_lookback_days": 31,
  "freq": "10d",
  "strategy": {
    "topk": 10,
    "n_drop": 3,
    "forbid_all_trade_at_limit": true
  },
  "exchange_kwargs": {
    "deal_price": "open",
    "open_cost": 0.0005,
    "close_cost": 0.0015,
    "min_cost": 5.0,
    "slippage": 0.001,
    "limit_threshold": 0.095,
    "lot_size": 100,
    "forbid_buy_limit_up": true,
    "forbid_buy_limit_down": true,
    "forbid_sell_limit_down": true
  },
  "columns": {
    "score": "pred"
  },
  "save_positions": true
}
```

关键参数：

| 参数 | 含义 |
| --- | --- |
| `prediction_path` | 预测 parquet 文件 |
| `price_path` | 股票日线宽表 parquet 根目录 |
| `benchmark_path` | benchmark 使用的 qlib features 目录，例如 `/opt/tiger/qyd/qlib_data_cn/features` |
| `output_dir` | 输出目录 |
| `start_time` / `end_time` | 回测区间 |
| `account` | 初始资金 |
| `benchmark` | benchmark 名称，默认回退为 `000300.SH` |
| `benchmark_lookback_days` | 首个回测日 benchmark 缺失时，向前查找可用 benchmark 的自然日窗口，默认 `31` |
| `strategy.topk` | 目标持仓数量 |
| `strategy.n_drop` | 每次优先卖出的弱势持仓数量 |
| `exchange_kwargs.deal_price` | 成交价字段，`open` 或 `close` |
| `exchange_kwargs.open_cost` | 买入费率 |
| `exchange_kwargs.close_cost` | 卖出费率 |
| `exchange_kwargs.min_cost` | 单笔最低费用 |
| `exchange_kwargs.slippage` | 单边滑点 |
| `exchange_kwargs.limit_threshold` | 涨跌停判断阈值 |
| `exchange_kwargs.lot_size` | 交易股数单位，A 股通常为 `100` |
| `columns.score` | 预测分数字段 |
| `save_positions` | 是否保存每日持仓明细 |

## 4. run 脚本使用方法

run 脚本路径：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/backtester/run_simple_backtest.py
```

查看帮助：

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/backtester/run_simple_backtest.py" --help
```

### 4.1 使用默认配置运行

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/backtester/run_simple_backtest.py"
```

默认会读取：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/simple_backtester_config.json
```

默认 `price_path` 指向：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/wide_table_daily_bars
```

默认 `benchmark_path` 指向：

```text
/opt/tiger/qyd/qlib_data_cn/features
```

### 4.2 推荐完整运行命令

下面命令使用：

- 沪深300作为 benchmark；
- 在 open 成交；
- 买入费率 0.05%；
- 卖出费率 0.15%；
- 单边滑点 0.10%；
- 单笔最低费用 5 元；
- 100 股整数倍；
- 禁止买入涨停 / 跌停；
- 禁止卖出跌停；
- 同时生成图表。

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/backtester/run_simple_backtest.py" \
  --config "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/simple_backtester_config.json" \
  --prediction-path "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet" \
  --price-path "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/wide_table_daily_bars" \
  --benchmark-path "/opt/tiger/qyd/qlib_data_cn/features" \
  --output-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/simple_backtest/hs300_open" \
  --start-time "2023-01-01" \
  --end-time "2025-12-31" \
  --benchmark "000300.SH" \
  --deal-price "open" \
  --topk 10 \
  --n-drop 3 \
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

### 4.3 使用 close 成交

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/backtester/run_simple_backtest.py" \
  --prediction-path "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet" \
  --output-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/simple_backtest/hs300_close" \
  --benchmark "000300.SH" \
  --deal-price "close" \
  --topk 10 \
  --n-drop 3
```

### 4.4 使用上证综指作为 benchmark

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/backtester/run_simple_backtest.py" \
  --prediction-path "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet" \
  --output-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/simple_backtest/sse" \
  --benchmark "000001.SH"
```

也可以写：

```bash
--benchmark "上证综指"
```

### 4.5 允许买入涨停或跌停

默认禁止买入涨停 / 跌停。如果要放开：

```bash
--allow-buy-limit-up
--allow-buy-limit-down
```

一般不建议放开这两个限制，因为实际成交概率很低。

### 4.6 不保存持仓或只试跑

不保存 `positions.parquet`：

```bash
--no-positions
```

只在内存中运行，不落盘：

```bash
--no-save
```

注意：如果使用 `--no-save`，同时指定 `--plot` 会跳过绘图，因为绘图依赖已保存的 `backtest_report.parquet`。

### 4.7 常用 CLI 参数

| 参数 | 含义 |
| --- | --- |
| `--config` | 配置 JSON，默认 `conf/simple_backtester_config.json` |
| `--trainer-config` | trainer 配置；当回测起止日期为空时用于读取 `segments.test` |
| `--prediction-path` | 预测 parquet 文件 |
| `--price-path` | 股票日线宽表 parquet 根目录 |
| `--benchmark-path` | benchmark 使用的 qlib features 目录，例如 `/opt/tiger/qyd/qlib_data_cn/features` |
| `--output-dir` | 输出目录 |
| `--start-time` / `--end-time` | 回测起止日期 |
| `--score-col` | 预测分数字段 |
| `--topk` | 目标持仓数量 |
| `--n-drop` | 每次卖出弱势持仓数量 |
| `--account` | 初始资金 |
| `--benchmark` | benchmark 指数 |
| `--freq` | 调仓频率，同时用于年化口径；支持 `day`、`week`、`month`、`10d`、`20d`、`每10天`、`10个交易日` 等 |
| `--deal-price` | `open` 或 `close` |
| `--open-cost` | 买入费率 |
| `--close-cost` | 卖出费率 |
| `--min-cost` | 单笔最低费用 |
| `--slippage` | 单边滑点 |
| `--limit-threshold` | 涨跌停阈值 |
| `--lot-size` | 交易股数单位 |
| `--allow-buy-limit-up` | 允许买入涨停 |
| `--allow-buy-limit-down` | 允许买入跌停 |
| `--forbid-sell-limit-down` | 禁止卖出跌停 |
| `--no-save` | 不保存文件 |
| `--no-positions` | 不保存持仓文件 |
| `--plot` | 生成回测图 |
| `--plot-output` | 指定回测图输出路径 |

## 5. Python API 使用方法

也可以在 Python / notebook 中直接调用：

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" python - <<'PY'
from backtester import SimpleBacktester

backtester = SimpleBacktester(
    config_path="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/simple_backtester_config.json",
    prediction_path="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet",
    price_path="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/wide_table_daily_bars",
    benchmark_path="/opt/tiger/qyd/qlib_data_cn/features",
    output_dir="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/simple_backtest/python_api",
    benchmark="000300.SH",
    start_time="2023-01-01",
    end_time="2025-12-31",
    strategy={"topk": 10, "n_drop": 3},
    exchange_kwargs={
        "deal_price": "open",
        "open_cost": 0.0005,
        "close_cost": 0.0015,
        "slippage": 0.001,
        "min_cost": 5.0,
        "limit_threshold": 0.095,
        "lot_size": 100,
        "forbid_buy_limit_up": True,
        "forbid_buy_limit_down": True,
        "forbid_sell_limit_down": True,
    },
)

result = backtester.run(save=True)
print("输出目录:", backtester.output_dir)
print("摘要:", result["analysis"].get("summary", {}))
print("风险指标:", result["analysis"].get("risk", {}))
PY
```

`run()` 返回：

```python
{
    "report": report_df,
    "positions": positions_df,
    "trades": trades_df,
    "analysis": analysis,
}
```

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `report` | `pd.DataFrame` | 每日账户、收益、成本和 benchmark 报表 |
| `positions` | `pd.DataFrame` | 每日持仓明细 |
| `trades` | `pd.DataFrame` | 每笔成交明细 |
| `analysis` | `dict` | 汇总指标和风险指标 |

## 6. 输出文件说明

如果 `save=True` 且未指定 `--no-save`，输出目录中会生成：

| 文件 | 内容 |
| --- | --- |
| `backtest_report.parquet` | 每日回测报表 |
| `positions.parquet` | 每日持仓明细；当 `save_positions=true` 且有持仓时保存 |
| `trades.parquet` | 每笔成交明细 |
| `analysis.json` | 回测摘要、收益指标、风险指标和配置快照 |
| `backtest_plots.png` | 可选；使用 `--plot` 时生成 |

### 6.1 `backtest_report.parquet`

索引：

```text
DatetimeIndex，索引名 datetime
```

主要字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `return` | `float` | 当日策略净收益率，已反映行情涨跌、手续费和滑点 |
| `gross_return` | `float` | 近似扣费前收益率，即把当日成本加回后的收益 |
| `benchmark` | `float` | 从日线宽表中指定 benchmark 标的行读取得到的当日收益 |
| `excess_return` | `float` | 当日超额收益，等于 `return - benchmark` |
| `cost` | `float` | 当日总交易成本，等于 `fee + slippage_cost`，金额单位 |
| `fee` | `float` | 当日手续费合计，金额单位 |
| `slippage_cost` | `float` | 当日滑点成本合计，金额单位 |
| `turnover` | `float` | 当日成交额 / 前一日账户权益 |
| `buy_value` | `float` | 当日买入原始成交额，不含滑点和手续费 |
| `sell_value` | `float` | 当日卖出原始成交额，不含滑点和手续费 |
| `cash` | `float` | 当日收盘后的现金余额 |
| `stock_value` | `float` | 当日收盘持仓市值 |
| `account_value` | `float` | 当日账户总权益，等于 `cash + stock_value` |
| `holdings_count` | `int` | 当日持仓股票数量 |

核心计算关系：

```text
买入执行价 = 原始成交价 × (1 + slippage)
卖出执行价 = 原始成交价 × (1 - slippage)
fee = max(raw_value × cost_rate, min_cost)  # cost_rate > 0 时
account_value = cash + 持仓股数 × 当日 close
return[t] = account_value[t] / account_value[t-1] - 1
turnover[t] = (buy_value[t] + sell_value[t]) / account_value[t-1]
excess_return[t] = return[t] - benchmark[t]
```

### 6.2 `positions.parquet`

索引：

```text
MultiIndex(datetime, instrument)
```

字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `shares` | `int` | 当日收盘持股数量，按 `lot_size` 约束后的实际股数 |
| `close` | `float` | 当日收盘价 |
| `market_value` | `float` | 持仓市值，等于 `shares × close` |
| `weight` | `float` | 持仓市值占账户权益比例 |
| `available_shares` | `int` | T+1 规则下当天可卖股数；当天买入不计入可卖 |

说明：

- `shares` 不一定刚好等权，因为买入数量要按 100 股取整，并受现金、费用和股价影响。
- `available_shares` 可用于检查 T+1 限制是否生效。

### 6.3 `trades.parquet`

索引：

```text
MultiIndex(datetime, instrument, side)
```

其中：

- `side=buy` 表示买入。
- `side=sell` 表示卖出。

字段：

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `shares` | `int` | 成交股数，按 `lot_size` 取整 |
| `raw_price` | `float` | 原始成交价，即当日 `open` 或 `close` |
| `exec_price` | `float` | 含滑点后的实际执行价 |
| `raw_value` | `float` | 原始成交额，等于 `shares × raw_price` |
| `fee` | `float` | 该笔交易手续费 |
| `slippage_cost` | `float` | 该笔交易滑点成本 |
| `cost` | `float` | 该笔总成本，等于 `fee + slippage_cost` |
| `cash_after` | `float` | 该笔成交后现金余额 |

买入成本：

```text
cash -= shares × raw_price × (1 + slippage) + fee
```

卖出回款：

```text
cash += shares × raw_price × (1 - slippage) - fee
```

### 6.4 `analysis.json`

顶层结构：

```json
{
  "return": {...},
  "benchmark": {...},
  "excess_return": {...},
  "risk": {...},
  "summary": {...},
  "config": {...}
}
```

#### 6.4.1 `summary`

| 字段 | 含义 |
| --- | --- |
| `start_time` | 实际回测起始交易日 |
| `end_time` | 实际回测结束交易日 |
| `trading_days` | 有效回测交易日数量 |
| `initial_account_value` | 初始资金 |
| `final_account_value` | 回测结束账户权益 |
| `total_return` | 策略总收益率 |
| `benchmark_total_return` | benchmark 总收益率 |
| `mean_turnover` | 平均日换手率 |
| `total_cost` | 回测期间总交易成本 |
| `total_fee` | 回测期间总手续费 |
| `total_slippage_cost` | 回测期间总滑点成本 |
| `trade_count` | 成交笔数 |
| `rebalance_days` | 实际调仓日数量 |
| `rebalance_frequency` | 本次使用的调仓频率配置，例如 `week`、`10d` |
| `rebalance_interval_trading_days` | 如果使用每 N 个交易日调仓，则为 N；否则为 `null` |
| `mean_holdings_count` | 平均持仓股票数量 |
| `deal_price` | 本次使用的成交价字段，`open` 或 `close` |
| `benchmark` | 本次使用的 benchmark |

#### 6.4.2 `return` / `benchmark` / `excess_return`

这三个区块分别对策略收益、benchmark 收益和超额收益计算单序列指标。

| 字段 | 含义 |
| --- | --- |
| `total_return` | 区间总收益 |
| `annualized_return` | 年化收益 |
| `annualized_volatility` | 年化波动率 |
| `sharpe` | Sharpe 比率，使用 0 无风险利率近似 |
| `sortino` | Sortino 比率 |
| `max_drawdown` | 最大回撤 |
| `calmar` | Calmar 比率 |
| `win_rate` | 日胜率 |
| `mean_daily_return` | 平均日收益 |
| `median_daily_return` | 中位数日收益 |
| `best_daily_return` | 最好单日收益 |
| `worst_daily_return` | 最差单日收益 |
| `skew` | 收益偏度 |
| `kurtosis` | 收益峰度 |

#### 6.4.3 `risk`

`risk` 是策略相对 benchmark 的指标。

| 字段 | 含义 |
| --- | --- |
| `alpha` | 年化 alpha |
| `alpha_daily` | 日 alpha |
| `beta` | 策略收益相对 benchmark 的 beta |
| `excess_total_return` | 策略累计收益与 benchmark 累计收益差 |
| `excess_annualized_return` | 年化超额收益，按平均日超额收益年化 |
| `tracking_error` | 年化跟踪误差 |
| `information_ratio` | 信息比率 |
| `correlation` | 策略收益和 benchmark 收益相关系数 |

#### 6.4.4 `config`

`config` 保存本次回测使用的配置快照，便于复现实验，包括：

- 预测文件路径。
- 行情路径。
- benchmark。
- 输出路径。
- `topk` / `n_drop`。
- 成交价、手续费、滑点、涨跌停阈值等参数。

## 7. 回测逻辑细节

### 7.1 每日流程

每个交易日按以下顺序处理：

1. 读取当天所有有预测分数且有行情的股票。
2. 根据预测分数从高到低排序。
3. 从当前持仓中挑选最多 `n_drop` 个弱势股票作为卖出候选。
4. 对卖出候选检查：
   - 是否有行情；
   - 是否停牌或成交量为 0；
   - 是否受跌停卖出限制；
   - T+1 可卖股数是否大于 0。
5. 先卖出，再买入。
6. 买入时跳过涨停 / 跌停 / 无成交量股票。
7. 买入股数按 `lot_size` 向下取整。
8. 用当日 `close` 计算持仓市值和账户权益。
9. 保存日收益、成本、持仓、成交和风险指标。

### 7.2 调仓频率 `freq`

`freq` 现在不只是年化参数，也会控制实际买卖发生的调仓日。回测仍然每天输出净值、收益、持仓市值和 benchmark；但只有调仓日会执行卖出 / 买入，非调仓日不会产生 `trades.parquet` 记录。

支持的写法：

| `freq` 写法 | 含义 |
| --- | --- |
| `day` / `daily` / `1d` | 每个交易日调仓 |
| `week` / `weekly` / `1w` / `w` | 每周第一个可交易日调仓 |
| `month` / `monthly` / `1m` / `m` | 每月第一个可交易日调仓 |
| `10` / `10d` / `10day` / `10days` | 从回测第一个交易日开始，每 10 个有效交易日调仓一次 |
| `20d` / `20td` / `20trading_days` | 从回测第一个交易日开始，每 20 个有效交易日调仓一次 |
| `每10天` / `10个交易日` | 中文写法，同样表示每 10 个交易日调仓一次 |

例如每 10 个交易日调仓：

```json
{
  "freq": "10d"
}
```

命令行覆盖：

```bash
python src/backtester/run_simple_backtest.py --freq 20d
```

输出的 `backtest_report.parquet` 中会包含 `is_rebalance_date` 字段，便于检查哪些日期发生了调仓；`analysis.json` 的 `summary.rebalance_days` 会记录总调仓日数量。

### 7.3 为什么先卖后买

先卖后买可以释放现金，使调仓更接近真实账户逻辑。当天卖出获得的现金可用于当天买入；但当天买入的股票受 T+1 约束，不能当天卖出。

### 7.4 涨跌停判断

优先使用行情字段：

```text
pct_chg / 100
```

当涨跌幅：

```text
>= limit_threshold
```

视为涨停。

当涨跌幅：

```text
<= -limit_threshold
```

视为跌停。

如果 `pct_chg` 缺失，则使用：

```text
deal_price / pre_close - 1
```

作为回退。

### 7.5 open 和 close 成交的区别

使用 `--deal-price open`：

- 买卖都按当天开盘价成交。
- 收盘时按当天收盘价估值。
- 更适合模拟“盘前生成信号，开盘调仓”。

使用 `--deal-price close`：

- 买卖都按当天收盘价成交。
- 收盘时仍按当天收盘价估值。
- 更适合模拟“收盘前或收盘价附近调仓”。

注意：如果你的预测信号实际上是在当天收盘后才能得到，那么用当天 close 成交可能存在未来函数风险；这种情况下应考虑将信号滞后一日或使用次日 open 成交。

## 8. 生成回测图

命令行中加入：

```bash
--plot
```

默认输出：

```text
<output-dir>/backtest_plots.png
```

图中包括：

1. 账户权益曲线。
2. 策略累计收益 vs benchmark 累计收益。
3. 策略日收益 vs benchmark 日收益。
4. 回撤曲线，并标注最大回撤区间。

也可以单独使用 `BacktestPlotter`：

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" python - <<'PY'
from utils import BacktestPlotter

plotter = BacktestPlotter(
    report_path="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/simple_backtest/hs300_open/backtest_report.parquet",
    output_path="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/simple_backtest/hs300_open/backtest_plots.png",
)
info = plotter.plot(title="SimpleBacktester Performance")
print(info)
PY
```

## 9. 常见问题排查

### 9.1 `Prediction file does not exist`

说明 `prediction_path` 指向的预测文件不存在。请先运行训练 / 推理生成 `pred_test.parquet`，或用 `--prediction-path` 指向实际路径。

### 9.2 `Prediction parquet must use MultiIndex(datetime, instrument)`

说明预测文件索引不符合要求。需要保证：

```text
index names = ["datetime", "instrument"]
```

### 9.3 `No overlapping prediction and price records`

说明预测文件和日线行情在股票代码或日期上没有交集。检查：

1. `prediction_path` 的日期区间。
2. `price_path` 是否包含对应日期。
3. 股票代码格式是否一致，例如是否都是 `000001.SZ`。
4. `start_time` / `end_time` 是否过窄。

### 9.4 `Benchmark qlib path does not exist` / `Benchmark qlib directory does not exist`

说明 `benchmark_path` 指向的 qlib features 目录不存在，或指定 benchmark 映射后的 instrument 目录不存在。当前版本 benchmark 默认应指向：

```text
/opt/tiger/qyd/qlib_data_cn/features
```

请检查：

```bash
ls /opt/tiger/qyd/qlib_data_cn/features/sh000300
```

或在命令行显式传入：

```bash
--benchmark-path "/opt/tiger/qyd/qlib_data_cn/features"
```

### 9.5 买入股票数量明显少于 topk

可能原因：

1. 涨停 / 跌停限制过滤了部分股票。
2. 股票价格较高，按 100 股取整后现金不足。
3. 手续费和滑点导致可买股数下降。
4. 当天行情缺失或成交量为 0。

### 9.6 `trades.parquet` 没有生成

如果整个回测期间没有任何成交，`trades.parquet` 不会保存。常见原因是预测和行情没有交集，或者所有候选股票都被交易限制过滤。

## 10. 推荐实验流程

1. 生成或确认预测文件：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet
```

2. 检查预测文件结构：

```bash
python - <<'PY'
import pandas as pd
path = "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet"
df = pd.read_parquet(path)
print(df.index.names)
print(df.columns.tolist())
print(df.head())
PY
```

3. 运行 open 成交回测：

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/backtester/run_simple_backtest.py" \
  --prediction-path "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet" \
  --output-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/simple_backtest/open" \
  --benchmark "000300.SH" \
  --deal-price "open" \
  --topk 10 \
  --n-drop 3 \
  --plot
```

4. 运行 close 成交回测做对照：

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/backtester/run_simple_backtest.py" \
  --prediction-path "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet" \
  --output-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/simple_backtest/close" \
  --benchmark "000300.SH" \
  --deal-price "close" \
  --topk 10 \
  --n-drop 3 \
  --plot
```

5. 对比两个输出目录中的：

```text
analysis.json
backtest_report.parquet
trades.parquet
backtest_plots.png
```

重点关注：

- `summary.total_return`
- `return.sharpe`
- `risk.alpha`
- `risk.beta`
- `risk.information_ratio`
- `summary.total_cost`
- `summary.total_slippage_cost`
- `summary.trade_count`
