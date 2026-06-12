# daily_bars Parquet 对接 Qlib 并完成最简 XGBoost 训练方案

## 1. 目标

先只使用 `/opt/tiger/qyd/quant_llm/A_stocks_all_data/daily_bars` 里的基础日线字段，完成一次最简单的 XGBoost 训练闭环。

第一版只使用这些最普通的字段：

- `open`
- `high`
- `low`
- `close`
- `vol`
- `amount`

暂时不接入：

- 复权因子
- 基本面
- 资金流
- 行业
- 停牌/ST/退市过滤

第一版目标是：**先把 Parquet 数据读起来，构造简单特征和标签，能训练出一个 XGBoost 模型，并能在验证集上输出预测结果和基础指标。**

---

## 2. 当前环境确认

本地环境已确认：

```text
qlib: OK, version=0.9.7
xgboost: OK, version=3.2.0
pandas: OK, version=2.3.3
pyarrow: OK, version=23.0.1
```

另外检查到：

1. 本地可以导入 `qlib.data.dataset.DatasetH`。
2. 本地可以导入 `qlib.data.dataset.handler.DataHandlerLP`。
3. 本地 `qlib.contrib.model.gbdt` 里没有可直接导入的 `XGBModel`。

所以第一版不建议强行依赖 Qlib 内置的 XGBoost wrapper，而是采用：

> **Qlib 负责数据集抽象 / 数据切分接口，XGBoost 使用原生 `xgboost.XGBRegressor` 完成训练。**

这样最稳，也最容易排查问题。

---

## 3. daily_bars 数据格式回顾

代表性文件：

```text
/opt/tiger/qyd/quant_llm/A_stocks_all_data/daily_bars/year=2000/month=01/20000104.parquet
```

字段：

```text
ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount
```

样例：

```text
     ts_code trade_date  open  high   low  close  pre_close  change  pct_chg      vol      amount
0  000003.SZ   20000104  5.48  5.85  5.40   5.74       5.47    0.27     4.94  19073.0  10787.1201
1  000005.SZ   20000104  6.10  6.27  6.00   6.24       6.04    0.20     3.31   8365.0   5132.4196
2  000007.SZ   20000104  8.20  8.28  7.88   8.28       8.00    0.28     3.50   3368.0   2739.0356
```

---

## 4. 推荐路线

有两种可行路线。

### 路线 A：最小可行路线，直接 Parquet -> pandas -> XGBoost

这是最快跑通训练的路线，不强制使用 Qlib。

流程：

```text
daily_bars parquet
  -> pandas/pyarrow 读取
  -> 统一成 MultiIndex(date, instrument)
  -> 构造 feature 和 label
  -> 时间切分 train/valid/test
  -> xgboost.XGBRegressor 训练
  -> 输出 IC/RMSE/预测文件
```

优点：

- 最快验证数据是否可训练；
- 不需要先转换 Qlib bin；
- 失败点少，方便排查字段、日期、标签问题。

缺点：

- 还没有完全接入 Qlib 的 Dataset/Workflow/Recorder；
- 后续做 Qlib 回测时还需要再封装一次。

### 路线 B：推荐落地路线，Parquet -> 自定义 Qlib DataLoader -> DatasetH -> XGBoost

这是我建议采用的路线。

流程：

```text
daily_bars parquet
  -> 自定义 ParquetLoader
  -> Qlib DataHandlerLP
  -> Qlib DatasetH
  -> dataset.prepare("train" / "valid" / "test")
  -> 原生 xgboost.XGBRegressor 训练
```

优点：

- 不需要先把 Parquet 转成 Qlib bin；
- 仍然可以使用 Qlib 的 `DatasetH` 分段能力；
- 后续可以比较自然地扩展到 Qlib workflow；
- 也方便未来替换成 LightGBM、线性模型或深度模型。

缺点：

- 需要写一个很薄的自定义 `DataLoader`；
- 第一版需要自己处理数据列分组，即 `feature` 和 `label`。

### 路线 C：Parquet -> Qlib bin provider -> Qlib 标准表达式

这是长期更标准的路线，但不建议作为第一版。

流程：

```text
daily_bars parquet
  -> 转成 Qlib 标准 calendar / instruments / features bin
  -> qlib.init(provider_uri=...)
  -> Alpha158 或自定义 Handler
  -> DatasetH
  -> 模型训练
```

优点：

- 最符合 Qlib 官方数据形态；
- 后续做回测、表达式特征、缓存等最方便。

缺点：

- 前置转换工作较多；
- 复权、日历、股票生命周期都要先处理；
- 对当前“只用 daily_bars 简单训练一次”的目标来说偏重。

---

## 5. 本次建议采用的方案

建议第一版采用 **路线 B**：

> **保持原始 daily_bars Parquet 不动，写一个 Qlib 自定义 DataLoader，返回 Qlib DatasetH 所需的 DataFrame，然后用原生 XGBoost 训练。**

这样既能对接 Qlib 的 Dataset，又不需要先做复杂的数据转换。

---

## 6. 数据读取设计

### 6.1 原始目录

```text
/opt/tiger/qyd/quant_llm/A_stocks_all_data/daily_bars
```

目录形态：

```text
daily_bars/
  year=2000/
    month=01/
      20000104.parquet
      ...
  year=2001/
  ...
```

### 6.2 读取方式

第一版建议使用 `pyarrow.dataset` 读取整个分区目录：

```python
import pyarrow.dataset as ds

dataset = ds.dataset(daily_bars_dir, format="parquet", partitioning="hive")
table = dataset.to_table(columns=[
    "ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount"
])
df = table.to_pandas()
```

如果全量一次读取太大，则先支持参数限制年份：

```bash
--start-date 2018-01-01 --end-date 2024-12-31
```

第一版建议默认只跑一个较短区间，比如 2018-2024，避免一上来读取 2000-2026 全量数据导致内存压力过大。

---

## 7. 证券代码与日期规范

### 7.1 日期

原始日期：

```text
20000104
```

转换为：

```text
2000-01-04
```

代码：

```python
df["datetime"] = pd.to_datetime(df["trade_date"], format="%Y%m%d")
```

### 7.2 股票代码

原始代码：

```text
000001.SZ
600000.SH
```

建议第一版先保持原始 `ts_code` 作为 instrument：

```text
000001.SZ
600000.SH
```

原因：

1. 当前只是训练，不做 Qlib bin provider；
2. 可以减少一次转换带来的歧义；
3. 后续如果要转成 Qlib 标准 bin，再统一改成 `sz000001` / `sh600000`。

---

## 8. 特征设计

第一版只使用非常简单的日线特征。

### 8.1 原始特征

```text
open
high
low
close
vol
amount
```

### 8.2 衍生特征

建议额外加几个简单且稳定的特征：

```text
ret_1       = close / pre_close - 1
range       = high / low - 1
close_open  = close / open - 1
high_open   = high / open - 1
low_open    = low / open - 1
log_volume  = log1p(vol)
log_amount  = log1p(amount)
```

如果你希望“只用 open/high/low/close”，也可以先只保留：

```text
open, high, low, close
```

但从训练稳定性看，建议至少加上 `vol` 和 `amount`。

---

## 9. 标签设计

第一版做一个最普通的日频截面回归任务：预测下一交易日收益率。

标签：

```text
label = next_close / close - 1
```

按股票分组计算：

```python
df = df.sort_values(["instrument", "datetime"])
df["label"] = df.groupby("instrument")["close"].shift(-1) / df["close"] - 1
```

注意：

1. 标签必须按股票内时间序列 shift；
2. 每只股票最后一天没有下一日 close，要丢弃；
3. 暂时不做复权，因此如果遇到除权日，标签会有噪声；这在第一版可以接受，但后续正式训练必须接入 `adj_factors`。

---

## 10. Qlib DatasetH 对接设计

### 10.1 自定义 DataLoader

Qlib 的 `DataLoader` 只要求实现：

```python
load(self, instruments, start_time=None, end_time=None) -> pd.DataFrame
```

返回值需要是一个 DataFrame：

- index：`MultiIndex(datetime, instrument)`
- columns：两层列索引，第一层是 `feature` / `label`

目标结构示意：

```text
columns:
  feature/open
  feature/high
  feature/low
  feature/close
  feature/vol
  feature/amount
  feature/ret_1
  feature/range
  feature/close_open
  label/label

index:
  datetime, instrument
```

### 10.2 DataHandlerLP

自定义 loader 接入 `DataHandlerLP`：

```python
from qlib.data.dataset.handler import DataHandlerLP

handler = DataHandlerLP(
    instruments="all",
    start_time="2018-01-01",
    end_time="2024-12-31",
    data_loader=ParquetLoader(
        daily_bars_dir="/opt/tiger/qyd/quant_llm/A_stocks_all_data/daily_bars"
    ),
)
```

### 10.3 DatasetH

用 Qlib DatasetH 管理训练、验证、测试切分：

```python
from qlib.data.dataset import DatasetH

dataset = DatasetH(
    handler=handler,
    segments={
        "train": ("2018-01-01", "2021-12-31"),
        "valid": ("2022-01-01", "2022-12-31"),
        "test":  ("2023-01-01", "2024-12-31"),
    },
)
```

然后：

```python
train_df = dataset.prepare("train")
valid_df = dataset.prepare("valid")
test_df = dataset.prepare("test")
```

---

## 11. XGBoost 训练设计

由于本地 Qlib 里没有直接可用的 `XGBModel`，第一版直接用原生 XGBoost：

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    n_jobs=8,
)
```

从 Qlib DatasetH 输出里取特征和标签：

```python
X_train = train_df["feature"]
y_train = train_df["label"]["label"]

X_valid = valid_df["feature"]
y_valid = valid_df["label"]["label"]
```

训练：

```python
model.fit(
    X_train,
    y_train,
    eval_set=[(X_valid, y_valid)],
    verbose=50,
)
```

预测：

```python
pred = model.predict(test_df["feature"])
```

输出结果建议保存为：

```text
./outputs/daily_bars_xgb/pred_test.parquet
./outputs/daily_bars_xgb/model.json
./outputs/daily_bars_xgb/metrics.json
```

---

## 12. 评估指标

第一版建议用三个指标即可：

### 12.1 RMSE

衡量回归误差：

```text
rmse = sqrt(mean((pred - label)^2))
```

### 12.2 IC

计算预测值和真实下一日收益的相关系数：

```text
IC = corr(pred, label)
```

### 12.3 Rank IC

计算截面排序相关性：

```text
RankIC = spearman_corr(pred, label)
```

建议按天计算 IC / RankIC，然后取均值：

```python
test_result.groupby("datetime").apply(lambda x: x["pred"].corr(x["label"]))
```

---

## 13. 第一版脚本建议

建议新增脚本：

```text
train_daily_bars_xgboost.py
```

建议参数：

```bash
python train_daily_bars_xgboost.py \
  --daily-bars-dir /opt/tiger/qyd/quant_llm/A_stocks_all_data/daily_bars \
  --start-date 2018-01-01 \
  --end-date 2024-12-31 \
  --train-start 2018-01-01 \
  --train-end 2021-12-31 \
  --valid-start 2022-01-01 \
  --valid-end 2022-12-31 \
  --test-start 2023-01-01 \
  --test-end 2024-12-31 \
  --output-dir ./outputs/daily_bars_xgb
```

### 13.1 脚本内部模块

建议拆成这些函数/类：

```text
ParquetLoader
normalize_daily_bars()
build_features_and_label()
build_qlib_dataset()
extract_xy()
train_xgboost()
evaluate_predictions()
save_outputs()
```

---

## 14. 数据泄漏注意事项

即使是最简单版本，也必须避免以下问题：

1. **标签不能用未来数据构造特征**：只能用当日及历史字段预测下一日收益。
2. **时间切分不能随机切分**：必须按日期切分 train/valid/test。
3. **shift 必须在股票内完成**：不能全市场直接 shift。
4. **测试集日期不能参与训练标准化**：第一版如果不做标准化，可以避开这个问题。
5. **未复权价格会带来除权噪声**：第一版接受，第二版必须接入 `adj_factors`。

---

## 15. 为什么不先转 Qlib bin

当前目标是“先只用 daily_bars 的几个普通字段做一次 XGBoost 训练”。

如果一开始转 Qlib bin，需要先处理：

1. calendar
2. instruments
3. 股票代码规范
4. 每只股票 feature bin
5. 复权因子
6. 上市/退市时间

这些会把任务复杂度明显提高。

所以第一版更推荐：

```text
Parquet -> 自定义 Qlib DataLoader -> DatasetH -> XGBoost
```

等这个最小训练闭环跑通以后，再做：

```text
Parquet -> Qlib bin provider -> 标准 Qlib workflow
```

---

## 16. 后续升级路线

第一版跑通后，建议按这个顺序升级：

### 第二版：接入复权因子

把 `adj_factors` 接进来，构造：

```text
open_adj, high_adj, low_adj, close_adj
```

然后用 `close_adj` 构造 label，减少除权除息造成的异常收益。

### 第三版：加入更多历史窗口特征

例如：

```text
ret_5, ret_10, ret_20
vol_mean_5, vol_mean_20
amount_mean_5, amount_mean_20
close_ma_5_ratio, close_ma_20_ratio
```

### 第四版：转换为标准 Qlib bin

等字段、标签、训练流程都稳定后，再把数据转成 Qlib provider 格式，之后可以直接使用 Qlib 表达式、Alpha158、回测模块等能力。

---

## 17. 最终建议

我建议下一步直接实现一个最小脚本：

```text
train_daily_bars_xgboost.py
```

第一版只保证：

1. 能读取 `daily_bars` Parquet；
2. 能构造基础 feature；
3. 能构造下一日收益 label；
4. 能通过 Qlib `DatasetH` 做时间切分；
5. 能用原生 XGBoost 训练；
6. 能输出预测结果和基础指标。

这个方案最适合当前阶段，因为它既对接了 Qlib 的 Dataset 思路，又避免了提前陷入 Qlib bin 数据转换的复杂细节。
