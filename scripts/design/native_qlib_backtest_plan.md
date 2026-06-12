# 原生 Qlib 回测改造方案

## 1. 背景

当前项目中的回测入口主要是：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/backtester/preparatory_backtester.py
```

当前实现虽然命名为 `PreparatoryBacktester`，但并没有调用 qlib 原生回测引擎。现有逻辑是：

1. 读取 XGBoost 训练 / 推理生成的 `pred_test.parquet`。
2. 按日根据 `pred` 排序。
3. 本地模拟一个简化版 `TopkDropoutStrategy`。
4. 用预测文件中的 `label` / `label_5d` 均值作为组合收益。
5. 用 qlib 的 `risk_analysis` 做收益统计。

也就是说，当前回测更准确地说是：

```text
parquet prediction + label-based portfolio simulation
```

而不是：

```text
qlib provider + qlib strategy + qlib executor + qlib exchange
```

如果要改成真实 qlib 回测，需要引入 qlib 原生的：

```python
from qlib.backtest import backtest
from qlib.backtest.executor import SimulatorExecutor
from qlib.contrib.strategy import TopkDropoutStrategy
```

并且必须先准备 qlib 可读取的数据 provider。

---

## 2. 改造目标

目标是将当前回测能力扩展为双 backend：

```text
backend = "parquet"  # 保留当前轻量回测
backend = "qlib"     # 新增原生 qlib 回测
```

### 2.1 保留 parquet backend

保留当前实现的原因：

1. 不依赖 qlib provider。
2. 可以快速验证模型 OOS 预测效果。
3. 可直接消费当前 `pred_test.parquet`。
4. 适合做研究阶段的轻量验证。

### 2.2 新增 qlib backend

新增 qlib backend 后，真实收益不再来自 label 均值，而来自 qlib exchange 根据行情价格、交易成本、交易约束模拟出的组合净值。

目标流程：

```text
qlib provider 行情数据
    + prediction signal
    -> qlib.init(provider_uri=...)
    -> qlib.contrib.strategy.TopkDropoutStrategy
    -> qlib.backtest.executor.SimulatorExecutor
    -> qlib.backtest.backtest()
    -> portfolio_metric / indicator_metric
```

---

## 3. 前置条件：准备 qlib provider

真实 qlib 回测依赖 qlib 数据 provider。当前项目目录下暂未发现已经落好的 provider，例如：

```text
/home/tiger/.qlib/qlib_data/cn_data
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/qlib_cn_data
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/qlib_data
```

因此第一步需要把现有 parquet 行情数据转换为 qlib binary provider。

项目中已有相关设计文档：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/scripts/design/qlib_ashare_conversion_plan.md
```

目标 provider 目录建议为：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/qlib_cn_data
```

目标结构大致如下：

```text
qlib_cn_data/
├── calendars/
│   └── day.txt
├── instruments/
│   └── all.txt
└── features/
    ├── sh600000/
    │   ├── open.day.bin
    │   ├── close.day.bin
    │   ├── high.day.bin
    │   ├── low.day.bin
    │   ├── volume.day.bin
    │   ├── amount.day.bin
    │   └── factor.day.bin
    └── sz000001/
        ├── open.day.bin
        ├── close.day.bin
        └── ...
```

完成后应能执行：

```python
import qlib

qlib.init(
    provider_uri="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/qlib_cn_data",
    region="cn",
)
```

并能通过 qlib 数据接口读取行情，例如：

```python
from qlib.data import D

df = D.features(
    instruments=["sz000001"],
    fields=["$open", "$close", "$volume", "$factor"],
    start_time="2023-01-01",
    end_time="2023-01-31",
    freq="day",
)
print(df.head())
```

---

## 4. 配置文件改造

配置文件：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/preparatory_backtester_config.json
```

建议增加 qlib backend 相关字段。

### 4.1 推荐配置示例

```json
{
  "backend": "qlib",
  "provider_uri": "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/qlib_cn_data",
  "region": "cn",
  "market": "all",
  "trainer_config_path": "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/conf/xgboost_trainer_config.json",
  "prediction_path": "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet",
  "output_dir": "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/qlib_native_backtest",
  "start_time": null,
  "end_time": null,
  "account": 1000000.0,
  "benchmark": "sh000300",
  "freq": "day",
  "instrument_format": "qlib_lower",
  "strategy": {
    "class": "TopkDropoutStrategy",
    "topk": 10,
    "n_drop": 3,
    "method_sell": "bottom",
    "method_buy": "top",
    "hold_thresh": 1,
    "only_tradable": true,
    "forbid_all_trade_at_limit": true,
    "risk_degree": 0.95
  },
  "executor": {
    "time_per_step": "day",
    "generate_portfolio_metrics": true,
    "verbose": false
  },
  "exchange_kwargs": {
    "freq": "day",
    "deal_price": "close",
    "open_cost": 0.0005,
    "close_cost": 0.0015,
    "min_cost": 5.0,
    "impact_cost": 0.0,
    "limit_threshold": 0.095
  },
  "columns": {
    "score": "pred",
    "label": "label_5d"
  },
  "save_positions": true
}
```

### 4.2 关键配置说明

| 字段 | 含义 |
| --- | --- |
| `backend` | `parquet` 表示当前轻量回测；`qlib` 表示原生 qlib 回测 |
| `provider_uri` | qlib provider 数据目录 |
| `region` | qlib 初始化 region，A 股一般为 `cn` |
| `market` | qlib universe，可先用 `all` |
| `benchmark` | qlib benchmark instrument，必须存在于 provider 中 |
| `instrument_format` | 预测文件股票代码转 qlib 代码的方式 |
| `strategy.topk` | 持仓股票数 |
| `strategy.n_drop` | 每次调仓替换股票数量 |
| `strategy.only_tradable` | 是否只交易 qlib 判定可交易的股票 |
| `executor.time_per_step` | 回测步长，日频用 `day` |
| `exchange_kwargs.deal_price` | 成交价字段，例如 `close` |

注意：真实 qlib 回测中，`label_5d` 不再用于计算收益。它只用于兼容当前 prediction parquet 的列检查。真实收益来自 qlib provider 中的价格序列。

---

## 5. 股票代码格式转换

当前项目预测文件中的 instrument 通常是：

```text
000001.SZ
600000.SH
```

qlib provider 中常见格式是：

```text
sz000001
sh600000
```

所以 qlib backend 需要将预测 signal 的 instrument 做转换。

### 5.1 推荐转换函数

```python
@staticmethod
def _to_qlib_lower_instrument(code: str) -> str:
    if "." not in code:
        return code.lower()
    symbol, exchange = code.split(".", 1)
    return f"{exchange.lower()}{symbol}"
```

示例：

```text
000001.SZ -> sz000001
600000.SH -> sh600000
```

如果后续 provider 使用大写格式，则可以增加：

```text
instrument_format = "qlib_upper"
```

对应：

```text
000001.SZ -> SZ000001
600000.SH -> SH600000
```

---

## 6. `preparatory_backtester.py` 改造方案

文件路径：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/backtester/preparatory_backtester.py
```

### 6.1 改造 `run()`

当前 `run()` 直接调用本地 `_simulate_topk_dropout()`。建议改为 backend 分发：

```python
def run(self, save: bool = True) -> dict[str, Any]:
    backend = self.config.get("backend", "parquet")
    if backend == "parquet":
        return self.run_parquet_backtest(save=save)
    if backend == "qlib":
        return self.run_preparatory_backtest(save=save)
    raise ValueError(f"Unsupported backtest backend: {backend}")
```

### 6.2 将当前逻辑迁移到 `run_parquet_backtest()`

```python
def run_parquet_backtest(self, save: bool = True) -> dict[str, Any]:
    pred_df = self.load_predictions()
    report_df, positions_df = self._simulate_topk_dropout(pred_df)
    analysis = self._analyze(report_df)

    if save:
        self.save_outputs(report_df=report_df, positions_df=positions_df, analysis=analysis)

    return {"report": report_df, "positions": positions_df, "analysis": analysis}
```

这样可以保持旧行为不变。

### 6.3 新增 `run_preparatory_backtest()`

核心逻辑：

```python
def run_preparatory_backtest(self, save: bool = True) -> dict[str, Any]:
    import qlib
    from qlib.backtest import backtest
    from qlib.backtest.executor import SimulatorExecutor
    from qlib.contrib.strategy import TopkDropoutStrategy

    provider_uri = self.config["provider_uri"]
    region = self.config.get("region", "cn")
    qlib.init(provider_uri=provider_uri, region=region)

    pred_df = self.load_predictions()
    signal = self._prepare_qlib_signal(pred_df)

    strategy = TopkDropoutStrategy(
        signal=signal,
        topk=int(self.strategy_config.get("topk", 50)),
        n_drop=int(self.strategy_config.get("n_drop", 5)),
        method_sell=self.strategy_config.get("method_sell", "bottom"),
        method_buy=self.strategy_config.get("method_buy", "top"),
        hold_thresh=int(self.strategy_config.get("hold_thresh", 1)),
        only_tradable=bool(self.strategy_config.get("only_tradable", True)),
        forbid_all_trade_at_limit=bool(self.strategy_config.get("forbid_all_trade_at_limit", True)),
        risk_degree=float(self.strategy_config.get("risk_degree", 0.95)),
    )

    executor_config = self.config.get("executor", {})
    executor = SimulatorExecutor(
        time_per_step=executor_config.get("time_per_step", "day"),
        generate_portfolio_metrics=bool(executor_config.get("generate_portfolio_metrics", True)),
        verbose=bool(executor_config.get("verbose", False)),
    )

    portfolio_metric, indicator_metric = backtest(
        start_time=self.start_time,
        end_time=self.end_time,
        strategy=strategy,
        executor=executor,
        benchmark=self.benchmark,
        account=self.account,
        exchange_kwargs=self.exchange_kwargs,
    )

    result = {
        "portfolio_metric": portfolio_metric,
        "indicator_metric": indicator_metric,
        "config": self.config,
    }

    if save:
        self.save_qlib_outputs(result)

    return result
```

### 6.4 新增 `_prepare_qlib_signal()`

```python
def _prepare_qlib_signal(self, pred_df: pd.DataFrame) -> pd.Series:
    signal = pred_df[self.score_col].copy().sort_index()

    instrument_format = self.config.get("instrument_format", "raw")
    if instrument_format == "qlib_lower":
        signal = signal.rename(index=self._to_qlib_lower_instrument, level="instrument")
    elif instrument_format == "qlib_upper":
        signal = signal.rename(index=self._to_qlib_upper_instrument, level="instrument")

    signal.index = signal.index.set_names(["datetime", "instrument"])
    return signal
```

### 6.5 新增 instrument 转换函数

```python
@staticmethod
def _to_qlib_lower_instrument(code: str) -> str:
    if "." not in code:
        return code.lower()
    symbol, exchange = code.split(".", 1)
    return f"{exchange.lower()}{symbol}"


@staticmethod
def _to_qlib_upper_instrument(code: str) -> str:
    if "." not in code:
        return code.upper()
    symbol, exchange = code.split(".", 1)
    return f"{exchange.upper()}{symbol}"
```

### 6.6 新增 qlib 输出保存函数

qlib 返回对象可能包含 DataFrame、Series、dict 或复杂对象。建议同时保存结构化文件和 pickle。

```python
def save_qlib_outputs(self, result: Mapping[str, Any]) -> None:
    self.output_dir.mkdir(parents=True, exist_ok=True)

    pd.to_pickle(result, self.output_dir / "qlib_backtest_result.pkl")

    portfolio_metric = result.get("portfolio_metric")
    indicator_metric = result.get("indicator_metric")

    if isinstance(portfolio_metric, dict):
        for key, value in portfolio_metric.items():
            self._save_metric_item(key, value)

    if isinstance(indicator_metric, dict):
        for key, value in indicator_metric.items():
            self._save_metric_item(f"indicator_{key}", value)

    with (self.output_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(self._json_safe(self.config), f, ensure_ascii=False, indent=2)


def _save_metric_item(self, name: str, value: Any) -> None:
    path_prefix = self.output_dir / str(name)
    if isinstance(value, pd.DataFrame):
        value.to_parquet(path_prefix.with_suffix(".parquet"))
    elif isinstance(value, pd.Series):
        value.to_frame(name=str(name)).to_parquet(path_prefix.with_suffix(".parquet"))
    else:
        with path_prefix.with_suffix(".json").open("w", encoding="utf-8") as f:
            json.dump(self._json_safe(value), f, ensure_ascii=False, indent=2)
```

---

## 7. `run_preparatory_backtest.py` 改造方案

文件路径：

```text
/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/backtester/run_preparatory_backtest.py
```

### 7.1 增加 CLI 参数

```python
parser.add_argument(
    "--backend",
    choices=["parquet", "qlib"],
    default=None,
    help="Backtest backend. Use qlib for native qlib backtest.",
)
parser.add_argument(
    "--provider-uri",
    default=None,
    help="Qlib provider uri for native qlib backtest.",
)
parser.add_argument(
    "--region",
    default=None,
    help="Qlib region, e.g. cn.",
)
parser.add_argument(
    "--instrument-format",
    choices=["raw", "qlib_lower", "qlib_upper"],
    default=None,
    help="Convert instrument code to qlib format, e.g. 000001.SZ -> sz000001.",
)
```

### 7.2 在 `build_overrides()` 中加入覆盖字段

```python
overrides: dict[str, Any] = {
    "backend": args.backend,
    "provider_uri": args.provider_uri,
    "region": args.region,
    "instrument_format": args.instrument_format,
    "trainer_config_path": args.trainer_config,
    "prediction_path": args.prediction_path,
    "output_dir": args.output_dir,
    "start_time": args.start_time,
    "end_time": args.end_time,
    "account": args.account,
    "benchmark": args.benchmark,
    "freq": args.freq,
}
```

### 7.3 qlib backend 下的运行命令示例

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/backtester/run_preparatory_backtest.py" \
  --backend qlib \
  --provider-uri "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/qlib_cn_data" \
  --prediction-path "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet" \
  --output-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/qlib_native_backtest" \
  --score-col "pred" \
  --label-col "label_5d" \
  --start-time "2023-01-01" \
  --end-time "2025-12-31" \
  --benchmark "sh000300" \
  --topk 10 \
  --n-drop 3 \
  --instrument-format qlib_lower
```

---

## 8. 真实 qlib 回测与当前 parquet 回测的差异

| 项目 | 当前 parquet 回测 | 原生 qlib 回测 |
| --- | --- | --- |
| 数据来源 | `pred_test.parquet` 中的 `pred + label` | qlib provider 行情 + prediction signal |
| 收益来源 | 选中股票 label 均值 | qlib exchange 根据真实行情撮合计算 |
| 买卖实现 | 本地集合模拟持仓变化 | qlib `TopkDropoutStrategy` 生成订单，`SimulatorExecutor` 执行 |
| 成本 | `turnover × cost_rate` 简化估算 | qlib exchange 根据交易订单和成本配置计算 |
| 停牌 / 涨跌停 | 未真实处理 | 可由 qlib exchange / provider 处理 |
| benchmark | 默认 `market_mean` | provider 中的指数，如 `sh000300` |
| label 作用 | 直接作为收益 | 不直接参与收益，只可用于预测评估 |

---

## 9. 实施顺序建议

### 阶段 1：落地 qlib provider

1. 基于现有 parquet 行情数据生成 qlib provider。
2. 确保目录中包含：
   - calendars
   - instruments
   - features
   - benchmark 指数行情
3. 用 `qlib.init(provider_uri=...)` 验证可初始化。
4. 用 `D.features()` 验证可以读取股票和指数行情。

### 阶段 2：扩展 backtester backend

1. 在配置文件中增加 `backend`、`provider_uri`、`instrument_format` 等字段。
2. 将当前 `run()` 拆成 `run_parquet_backtest()`。
3. 新增 `run_preparatory_backtest()`。
4. 新增 signal instrument 转换函数。
5. 新增 qlib 结果保存函数。

### 阶段 3：扩展 CLI

1. 给 `run_preparatory_backtest.py` 增加 qlib backend 参数。
2. 使用短区间、小股票池 smoke test。
3. 检查 qlib 输出结构。
4. 再跑完整 OOS 区间。

### 阶段 4：更新文档

1. 更新 `docs/preparatory_backtester_usage.md`。
2. 区分说明 `parquet backend` 与 `qlib backend`。
3. 补充 qlib provider 前置条件。
4. 补充原生 qlib 输出文件说明。

---

## 10. 验收标准

### 10.1 provider 验收

```python
import qlib
from qlib.data import D

qlib.init(
    provider_uri="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/qlib_cn_data",
    region="cn",
)

df = D.features(
    instruments=["sz000001"],
    fields=["$open", "$close", "$volume"],
    start_time="2023-01-01",
    end_time="2023-01-31",
    freq="day",
)
assert not df.empty
```

### 10.2 signal 验收

```python
pred_df = pd.read_parquet("/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet")
signal = backtester._prepare_qlib_signal(pred_df)
assert signal.index.names == ["datetime", "instrument"]
assert signal.index.get_level_values("instrument")[0].startswith(("sh", "sz"))
```

### 10.3 qlib backtest 验收

短区间运行：

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/backtester/run_preparatory_backtest.py" \
  --backend qlib \
  --provider-uri "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/qlib_cn_data" \
  --prediction-path "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/xgb_cross_sectional_dataset/pred_test.parquet" \
  --output-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/outputs/qlib_native_backtest_smoke" \
  --score-col pred \
  --label-col label_5d \
  --start-time "2023-01-03" \
  --end-time "2023-01-31" \
  --benchmark "sh000300" \
  --topk 10 \
  --n-drop 3 \
  --instrument-format qlib_lower
```

验收条件：

1. qlib 初始化成功。
2. signal 与 provider instrument 能对齐。
3. `qlib.backtest.backtest()` 正常返回 `portfolio_metric` 和 `indicator_metric`。
4. 输出目录下存在 qlib 原生回测结果文件。
5. 不再使用 label 均值作为收益来源。

---

## 11. 风险与注意事项

1. **provider 是关键路径**：没有 qlib provider，原生回测无法运行。
2. **instrument 格式必须一致**：预测 signal 和 provider 的股票代码必须完全一致。
3. **benchmark 必须存在**：`benchmark` 不能再用 `market_mean`，必须是 provider 中的指数代码。
4. **label 不再代表收益**：真实 qlib 回测收益来自行情成交价，不来自 `label_5d`。
5. **成本模型会变化**：qlib 原生 exchange 会按订单和成本配置计算成本，和当前简化换手成本不完全一致。
6. **返回结构可能随 qlib 版本变化**：建议保存 pickle，同时兼容 DataFrame / Series / dict。
7. **短区间 smoke test 很重要**：先验证 1 个月左右的数据，再跑完整 2023-2025 OOS。

---

## 12. 最终建议

推荐先不要直接删除当前轻量回测，而是保留为 `backend="parquet"`。

新增原生 qlib 回测作为 `backend="qlib"`，等 qlib provider、signal 对齐、benchmark 和输出结构全部验证稳定后，再将文档和默认配置逐步切到 qlib backend。

