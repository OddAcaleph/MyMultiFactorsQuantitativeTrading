# Dataset Utils 说明文档

本文档说明 `src/utils/dataset_cleaner`、`src/utils/dataset_processor`、`src/utils/features_generator`、`src/utils/label_generator`、`src/utils/cross_sectional_processor` 五个目录下每个类的作用、典型使用方式、生成数据样例、字段含义以及运行注意事项。

> 约定：所有清洗、加工、因子生成、label 生成和横截面处理工具均以只读方式读取上游 parquet，不会原地覆盖原始文件；输出会写入 `data/cleaned_data`、`data/processd_data`、`data/features_data`、`data/generated_label` 或 `data/cross_sectional_processd_data` 下的独立目录。路径中的 `processd_data` / `cross_sectional_processd_data` 是当前代码里使用的目录名，请按现有代码保持一致。

## 1. 整体数据流

```text
原始数据 /opt/tiger/qyd/quant_llm/A_stocks_all_data
  ├─ daily_bars       -> DailyBarsCleaner       -> data/cleaned_data/daily_bars/year=YYYY/month=MM/YYYYMMDD.parquet
  ├─ adj_factors      -> AdjFactorsCleaner      -> data/cleaned_data/adj_factors/adj_factors.parquet
  ├─ fundamentals     -> FundamentalsCleaner    -> data/cleaned_data/fundamentals/fundamentals.parquet
  ├─ industry         -> IndustryCleaner        -> data/cleaned_data/industry/industry.parquet
  ├─ moneyflow        -> MoneyflowCleaner       -> data/cleaned_data/moneyflow/moneyflow.parquet
  ├─ namechange       -> NamechangeStProcessor  -> data/processd_data/namechange/namechange_st_daily.parquet
  └─ suspend_d        -> SuspendDProcessor      -> data/processd_data/suspend_d/suspend_d_daily.parquet

cleaned_data / processd_data
  ├─ industry.parquet -> IndustryOneHotProcessor -> data/processd_data/industry/industry_onehot.parquet
  └─ 多张清洗/加工表 -> WideTableDailyBarsBuilder -> data/processd_data/wide_table_daily_bars/year=YYYY/month=MM/YYYYMMDD.parquet

cleaned_data / features_data
  ├─ daily_bars       -> PriceVolumeFeatureGenerator  -> data/features_data/price_volume_factors/...
  ├─ moneyflow+bars   -> MoneyFlowFeatureGenerator    -> data/features_data/moneyflow_factors/...
  ├─ fundamentals+bars-> FundamentalFeatureGenerator  -> data/features_data/fundamental_factors/...
  └─ industry+price/fundamental -> IndustryFeatureGenerator -> data/features_data/industry_factors/...

features_data / processd_data
  ├─ price_volume_factors  -> PriceVolumeFactorsCrossSectionalProcessor  -> data/cross_sectional_processd_data/price_volume_factors/...
  ├─ moneyflow_factors     -> MoneyflowFactorsCrossSectionalProcessor    -> data/cross_sectional_processd_data/moneyflow_factors/...
  ├─ fundamental_factors   -> FundamentalFactorsCrossSectionalProcessor  -> data/cross_sectional_processd_data/fundamental_factors/...
  ├─ industry_factors      -> IndustryFactorsCrossSectionalProcessor     -> data/cross_sectional_processd_data/industry_factors/...
  └─ wide_table_daily_bars -> WideTableDailyBarsCrossSectionalProcessor  -> data/cross_sectional_processd_data/wide_table_daily_bars/...

processd_data
  └─ wide_table_daily_bars -> DailyLabelGenerator -> data/generated_label/daily_labels/year=YYYY/month=MM/YYYYMMDD.parquet
```

通用运行方式：

```bash
export PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src"
python -m utils.dataset_cleaner.daily_bars_cleaner --log-file /opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/daily_bars_cleaning.log
```

也可以在 Python 中直接调用：

```python
from utils.dataset_cleaner.daily_bars_cleaner import DailyBarsCleaner

summary = DailyBarsCleaner().process()
print(summary.rows_read, summary.rows_written)
```

---

## 2. `dataset_cleaner`：原始数据清洗类

### 2.1 daily_bars_cleaner.py

#### 2.1.1 `DailyBarsProcessSummary`

- **所在文件**：`src/utils/dataset_cleaner/daily_bars_cleaner.py`
- **作用**：`DailyBarsCleaner.process()` 的聚合统计返回值。
- **主要字段**：
  - `files_processed`：处理的日行情 parquet 文件数。
  - `rows_read` / `rows_written`：读取/写出的总行数。
  - `missing_rows_removed`：因必需字段缺失删除的行数。
  - `duplicate_rows_removed`：因 `ts_code + trade_date` 重复删除的行数。
  - `files_with_missing` / `files_with_duplicates`：出现缺失/重复问题的文件数。
  - `has_missing_issue` / `has_duplicate_issue`：便捷布尔属性。

#### 2.1.2 `DailyBarsCleaner`

- **作用**：清洗分区日行情数据，输入目录形如 `year=YYYY/month=MM/YYYYMMDD.parquet`，输出保持相同分区结构。
- **默认输入**：`/opt/tiger/qyd/quant_llm/A_stocks_all_data/daily_bars`
- **默认输出**：`/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/daily_bars`
- **核心规则**：
  1. 校验必需列：`ts_code, trade_date, open, high, low, close, pre_close, change, pct_chg, vol, amount`。
  2. 删除任一必需字段为空的行。
  3. 以 `ts_code + trade_date` 为主键去重；跨文件也会去重，按排序后的文件顺序保留首次出现。

使用示例：

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python -m utils.dataset_cleaner.daily_bars_cleaner \
  --input-dir "/opt/tiger/qyd/quant_llm/A_stocks_all_data/daily_bars" \
  --output-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/daily_bars" \
  --log-file "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/daily_bars_cleaning.log"
```

输出数据样例：

| ts_code | trade_date | open | high | low | close | pre_close | change | pct_chg | vol | amount |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 000001.SZ | 20240102 | 9.39 | 9.48 | 9.20 | 9.30 | 9.39 | -0.09 | -0.9585 | 412345.67 | 385000.12 |

字段含义：

| 字段 | 含义 |
|---|---|
| `ts_code` | 股票代码，Tushare 格式，如 `000001.SZ`。 |
| `trade_date` | 交易日期，`YYYYMMDD`。 |
| `open/high/low/close` | 开盘/最高/最低/收盘价。 |
| `pre_close` | 前收盘价。 |
| `change` | 涨跌额，通常为 `close - pre_close`。 |
| `pct_chg` | 涨跌幅百分数。 |
| `vol` | 成交量。 |
| `amount` | 成交额，Tushare 日行情通常为千元。 |

### 2.2 adj_factors_cleaner.py

#### 2.2.1 `AdjFactorsProcessSummary`

- **作用**：`AdjFactorsCleaner.process()` 的统计返回值。
- **字段**：`input_file`、`output_file`、`rows_read`、`rows_written`、`missing_rows_removed`、`duplicate_rows_removed`，以及 `has_missing_issue`、`has_duplicate_issue`。

#### 2.2.2 `AdjFactorsCleaner`

- **作用**：清洗复权因子聚合文件。
- **默认输入**：`/opt/tiger/qyd/quant_llm/A_stocks_all_data/adj_factors/adj_factors.parquet`
- **默认输出**：`data/cleaned_data/adj_factors/adj_factors.parquet`
- **核心规则**：
  1. 必需列：`ts_code, trade_date, adj_factor`。
  2. 删除 `ts_code` 或 `trade_date` 为空的行。
  3. 按 `ts_code + trade_date` 去重，保留原文件顺序中的最后一行。

使用示例：

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python -m utils.dataset_cleaner.adj_factors_cleaner \
  --output-file "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data/adj_factors/adj_factors.parquet"
```

输出样例：

| ts_code | trade_date | adj_factor |
|---|---:|---:|
| 000001.SZ | 20240102 | 123.4567 |

字段含义：

| 字段 | 含义 |
|---|---|
| `ts_code` | 股票代码。 |
| `trade_date` | 交易日期。 |
| `adj_factor` | 复权因子，用于前复权/后复权价格计算。 |

### 2.3 fundamentals_cleaner.py

#### 2.3.1 `FundamentalsProcessSummary`

- **作用**：`FundamentalsCleaner.process()` 的统计返回值。
- **字段**：
  - `missing_rows_found`：发现必需字段缺失的行数；注意该清洗器不会因为财务因子缺失删除记录。
  - `invalid_date_rows_removed`：删除的无效日期行数。
  - `duplicate_rows_removed`：删除的重复主键行数。
  - 其他为输入/输出路径和行数统计。

#### 2.3.2 `FundamentalsCleaner`

- **作用**：清洗财务指标聚合文件。
- **默认输入**：`/opt/tiger/qyd/quant_llm/A_stocks_all_data/fundamentals/fundamentals.parquet`
- **默认输出**：`data/cleaned_data/fundamentals/fundamentals.parquet`
- **核心规则**：
  1. 必需列：`ts_code, ann_date, end_date, roe, roa, revenue_yoy, debt_ratio, gross_margin, eps, bps`。
  2. 财务因子缺失只记录不删除。
  3. 删除 `ann_date` 或 `end_date` 无法按 `YYYYMMDD` 解析，或 `ann_date <= end_date` 的记录。
  4. 以 `ts_code + ann_date + end_date` 去重，保留最后一行。

输出样例：

| ts_code | ann_date | end_date | roe | roa | revenue_yoy | debt_ratio | gross_margin | eps | bps |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 000001.SZ | 20240430 | 20240331 | 3.20 | 0.25 | 6.10 | 91.2 | 52.3 | 0.45 | 21.8 |

字段含义：

| 字段 | 含义 |
|---|---|
| `ts_code` | 股票代码。 |
| `ann_date` | 公告日期，即市场可见日期。后续点时合并只使用 `ann_date <= trade_date` 的记录，避免未来函数。 |
| `end_date` | 财报报告期截止日。 |
| `roe` | 净资产收益率。 |
| `roa` | 总资产收益率。 |
| `revenue_yoy` | 营业收入同比增速。 |
| `debt_ratio` | 资产负债率。 |
| `gross_margin` | 毛利率。 |
| `eps` | 每股收益。 |
| `bps` | 每股净资产。 |

### 2.4 industry_cleaner.py

#### 2.4.1 `IndustryProcessSummary`

- **作用**：`IndustryCleaner.process()` 的统计返回值。
- **字段**：输入/输出路径、读取/写出行数、`duplicate_rows_removed`，以及 `has_duplicate_issue`。

#### 2.4.2 `IndustryCleaner`

- **作用**：清洗行业分类聚合文件。
- **默认输入**：`/opt/tiger/qyd/quant_llm/A_stocks_all_data/industry/industry.parquet`
- **默认输出**：`data/cleaned_data/industry/industry.parquet`
- **核心规则**：仅删除完全重复行，保留最后一行；不做缺失值、日期或 schema 校验。

输出样例：

| ts_code | industry | L1_industry_name | L2_industry_name | L3_industry_name |
|---|---|---|---|---|
| 000001.SZ | 银行 | 金融 | 银行 | 股份制银行 |

字段含义：

| 字段 | 含义 |
|---|---|
| `ts_code` | 股票代码。 |
| `industry` | 原始行业名称。 |
| `L1_industry_name` | 一级行业名称。 |
| `L2_industry_name` | 二级行业名称。 |
| `L3_industry_name` | 三级行业名称。 |

### 2.5 moneyflow_cleaner.py

#### 2.5.1 `MoneyflowProcessSummary`

- **作用**：`MoneyflowCleaner.process()` 的统计返回值。
- **字段**：输入/输出路径、读取/写出行数、`missing_rows_removed`、`duplicate_rows_removed`，以及问题布尔属性。

#### 2.5.2 `MoneyflowCleaner`

- **作用**：清洗资金流聚合文件。
- **默认输入**：`/opt/tiger/qyd/quant_llm/A_stocks_all_data/moneyflow/moneyflow.parquet`
- **默认输出**：`data/cleaned_data/moneyflow/moneyflow.parquet`
- **核心规则**：
  1. 校验所有资金流必需列。
  2. 删除 `ts_code` 或 `trade_date` 为空的行。
  3. 以 `ts_code + trade_date` 去重，保留最后一行。

输出样例：

| ts_code | trade_date | buy_sm_vol | buy_sm_amount | sell_sm_vol | sell_sm_amount | buy_lg_amount | sell_lg_amount | buy_elg_amount | sell_elg_amount | net_mf_amount |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 000001.SZ | 20240102 | 10000 | 1200.5 | 9500 | 1100.2 | 5000.0 | 4300.0 | 2200.0 | 1800.0 | 1500.3 |

完整字段含义：

| 字段 | 含义 |
|---|---|
| `ts_code` | 股票代码。 |
| `trade_date` | 交易日期。 |
| `buy_sm_vol` / `buy_sm_amount` | 小单买入量/金额。 |
| `sell_sm_vol` / `sell_sm_amount` | 小单卖出量/金额。 |
| `buy_md_vol` / `buy_md_amount` | 中单买入量/金额。 |
| `sell_md_vol` / `sell_md_amount` | 中单卖出量/金额。 |
| `buy_lg_vol` / `buy_lg_amount` | 大单买入量/金额。 |
| `sell_lg_vol` / `sell_lg_amount` | 大单卖出量/金额。 |
| `buy_elg_vol` / `buy_elg_amount` | 特大单买入量/金额。 |
| `sell_elg_vol` / `sell_elg_amount` | 特大单卖出量/金额。 |
| `net_mf_vol` / `net_mf_amount` | 净流入量/金额。 |

---

## 3. `dataset_processor`：中间加工类

### 3.1 industry_onehot_processor.py

#### 3.1.1 `IndustryOneHotProcessSummary`

- **作用**：`IndustryOneHotProcessor.process()` 的统计返回值。
- **字段**：包括输入/输出路径、读取/写出行数、股票数、one-hot 特征列数、原始重复/冲突/缺失行数、输出重复/缺失行数。

#### 3.1.2 `IndustryOneHotProcessor`

- **作用**：将清洗后的行业表转换为每只股票一行的 one-hot 宽表。
- **默认输入**：`data/cleaned_data/industry/industry.parquet`
- **默认输出**：`data/processd_data/industry/industry_onehot.parquet`
- **规则**：
  1. 必需列：`ts_code, industry, L1_industry_name, L2_industry_name, L3_industry_name`。
  2. `ts_code` 重复时保留原文件顺序最后一条，保证输出 `ts_code` 唯一。
  3. 分类缺失不单独建缺失类别，对应层级 one-hot 全为 0。

输出样例（动态 one-hot 列完整清单见第 7 节）：

| ts_code | industry_银行 | industry_证券 | L1_非银金融 | L1_食品饮料 | L2_国有大型银行Ⅱ | L3_证券Ⅲ |
|---|---:|---:|---:|---:|---:|---:|
| 600030.SH | 0 | 1 | 1 | 0 | 0 | 1 |

字段含义：

| 字段 | 含义 |
|---|---|
| `ts_code` | 股票代码，唯一键。 |
| `industry_<category>` | 原始 `industry` 的类别哑变量，属于该类别为 1，否则 0。 |
| `L1_<category>` | 一级行业哑变量。 |
| `L2_<category>` | 二级行业哑变量。 |
| `L3_<category>` | 三级行业哑变量。 |

### 3.2 namechange_st_processor.py

#### 3.2.1 `NamechangeStProcessSummary`

- **作用**：`NamechangeStProcessor.process()` 的统计返回值。
- **字段**：输入/输出路径、读取/写出行数、股票数、展开日历天数、原始重复键行数、缺失必需字段行数、开放区间行数、输出重复/缺失行数。

#### 3.2.2 `NamechangeStProcessor`

- **作用**：根据名称变更记录展开生成每日 ST 状态 one-hot 特征。
- **默认输入**：`/opt/tiger/qyd/quant_llm/A_stocks_all_data/namechange/namechange.parquet`
- **默认输出**：`data/processd_data/namechange/namechange_st_daily.parquet`
- **规则**：
  1. 原始列：`ts_code, name, start_date, end_date, ann_date, change_reason`。
  2. `end_date` 为空视作开放区间，不作为错误；开放区间终点使用数据中已知最大日期。
  3. 名称以 `*ST` 开头记为 `star_st_stock=1`；以 `ST` 开头记为 `st_stock=1`；否则 `no_st_stock=1`。
  4. 同一 `trade_date + ts_code` 多条事件时，按 `_start_dt, _ann_dt, _raw_row` 排序保留最后一条。

输出样例：

| trade_date | ts_code | no_st_stock | st_stock | star_st_stock |
|---:|---|---:|---:|---:|
| 20240102 | 000001.SZ | 1 | 0 | 0 |
| 20240102 | 000002.SZ | 0 | 1 | 0 |

字段含义：

| 字段 | 含义 |
|---|---|
| `trade_date` | 日频日期，`YYYYMMDD` 整数。 |
| `ts_code` | 股票代码。 |
| `no_st_stock` | 当日非 ST 状态为 1。 |
| `st_stock` | 当日 `ST` 状态为 1。 |
| `star_st_stock` | 当日 `*ST` 状态为 1。 |

### 3.3 suspend_d_processor.py

#### 3.3.1 `SuspendDProcessSummary`

- **作用**：`SuspendDProcessor.process()` 的统计返回值。
- **字段**：输入/输出路径、读取/写出行数、股票数、交易日数、原始完整重复行数、日频键重复行数、S/R 冲突行数、缺失/非法日期/非法类型行数、输出重复/缺失行数，以及输出停牌/非停牌行数。

#### 3.3.2 `SuspendDProcessor`

- **作用**：将 Tushare `suspend_d` 停复牌记录规范化为每日停牌特征。
- **默认输入**：`/opt/tiger/qyd/quant_llm/A_stocks_all_data/suspend_d/suspend_d.parquet`
- **默认输出**：`data/processd_data/suspend_d/suspend_d_daily.parquet`
- **规则**：
  1. 必需列：`ts_code, trade_date, suspend_timing, suspend_type`。
  2. 仅保留 `suspend_type` 为 `S` 或 `R` 的记录。
  3. 输出以 `trade_date + ts_code` 唯一；同日同时存在 `S` 和 `R` 时，`S` 优先，`is_suspect=1`。

输出样例：

| trade_date | ts_code | is_suspect |
|---:|---|---:|
| 20240102 | 000001.SZ | 0 |
| 20240102 | 000002.SZ | 1 |

字段含义：

| 字段 | 含义 |
|---|---|
| `trade_date` | 交易日期。 |
| `ts_code` | 股票代码。 |
| `is_suspect` | 代码中当前字段名，语义是当日是否存在停牌 `S` 记录；1 表示停牌，0 表示只有复牌/非停牌记录。 |

### 3.4 wide_table_daily_bars_builder.py

#### 3.4.1 `WideTableDailyBarsBuildSummary`

- **作用**：`WideTableDailyBarsBuilder.process()` 的统计返回值。
- **字段**：主表路径、输出路径、文件数、读取/写出行数，各来源匹配行数，`missing_adj_factor_rows`、`missing_fundamentals_rows`、`missing_moneyflow_rows`，以及主表重复键行数。

#### 3.4.2 `_FeatureBundle`

- **作用**：`WideTableDailyBarsBuilder` 内部使用的预加载特征包 dataclass。
- **包含内容**：`namechange_df`、`suspend_df`、`adj_factor_df`、`fundamentals_df`、`moneyflow_df`、`industry_df`，以及行业/财务列名元信息。
- **使用方式**：内部类，不建议外部直接使用。

#### 3.4.3 `WideTableDailyBarsBuilder`

- **作用**：把清洗后的日行情主表与 ST、停牌、复权因子、财务、资金流、行业 one-hot 合并成日频宽表。
- **默认主表**：`data/cleaned_data/daily_bars`
- **默认输出**：`data/processd_data/wide_table_daily_bars`
- **合并顺序**：
  1. `cleaned_data/daily_bars` 作为主表。
  2. 按 `trade_date + ts_code` 合并 `namechange_st_daily`、`suspend_d_daily`。
  3. 按 `trade_date + ts_code` 合并 `adj_factor`。
  4. 财务表按 point-in-time 逻辑合并：取 `ann_date <= trade_date` 的最新记录。
  5. 按 `trade_date + ts_code` 合并 MoneyFlow。
  6. 按 `ts_code` 合并行业 one-hot。
- **默认填充**：ST 缺失填为非 ST（`no_st_stock=1, st_stock=0, star_st_stock=0`）；停牌缺失填 `is_suspect=0`；行业 one-hot 缺失填 0。

使用示例：

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python -m utils.dataset_processor.wide_table_daily_bars_builder \
  --start-date 20240101 \
  --end-date 20240131 \
  --output-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/wide_table_daily_bars"
```

完整输出样例（下表完整展示全部固定字段，并展示动态行业 one-hot 的取值规则；当前环境 293 个实际输出列的完整顺序见第 7 节）：

| ts_code | trade_date | open | high | low | close | pre_close | change | pct_chg | vol | amount | no_st_stock | st_stock | star_st_stock | is_suspect | adj_factor | end_date | ann_date | roe | roa | revenue_yoy | debt_ratio | gross_margin | eps | bps | buy_sm_vol | buy_sm_amount | sell_sm_vol | sell_sm_amount | buy_md_vol | buy_md_amount | sell_md_vol | sell_md_amount | buy_lg_vol | buy_lg_amount | sell_lg_vol | sell_lg_amount | buy_elg_vol | buy_elg_amount | sell_elg_vol | sell_elg_amount | net_mf_vol | net_mf_amount | industry_银行 | industry_证券 | L1_非银金融 | L1_食品饮料 | L2_国有大型银行Ⅱ | L3_高速公路 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 000001.SZ | 20240102 | 9.39 | 9.48 | 9.20 | 9.30 | 9.39 | -0.09 | -0.9585 | 412345.67 | 385000.12 | 1 | 0 | 0 | 0 | 123.4567 | 20230930 | 20231030 | 3.20 | 0.25 | 6.10 | 91.2 | 52.3 | 0.45 | 21.8 | 10000 | 1200.5 | 9500 | 1100.2 | 8000 | 980.0 | 7600 | 910.0 | 5000 | 5000.0 | 4300 | 4300.0 | 2200 | 2200.0 | 1800 | 1800.0 | 1500 | 1500.3 | 1 | 0 | 0 | 0 | 1 | 0 |

> 动态行业 one-hot 列遵循统一规则：`industry_<category>`、`L1_<category>`、`L2_<category>`、`L3_<category>` 均为 0/1，属于该类别为 1，否则为 0；未命中的所有动态 one-hot 列均为 0。当前环境的全部动态列名已在第 7 节按实际 parquet schema 完整列出。

完整固定字段含义：

| 字段 | 含义 |
|---|---|
| `ts_code` | 股票代码。 |
| `trade_date` | 交易日期，日频主键之一。 |
| `open` | 当日开盘价。 |
| `high` | 当日最高价。 |
| `low` | 当日最低价。 |
| `close` | 当日收盘价。 |
| `pre_close` | 前一交易日收盘价。 |
| `change` | 涨跌额，通常为 `close - pre_close`。 |
| `pct_chg` | 涨跌幅百分数。 |
| `vol` | 成交量。 |
| `amount` | 成交额，Tushare 日行情通常为千元。 |
| `no_st_stock` | 当日正常/非 ST 状态为 1。 |
| `st_stock` | 当日 `ST` 状态为 1。 |
| `star_st_stock` | 当日 `*ST` 状态为 1。 |
| `is_suspect` | 当日是否存在停牌 `S` 记录；1 表示停牌，0 表示未停牌或只有复牌记录。 |
| `adj_factor` | 当日复权因子。 |
| `end_date` | 该日可用的最新财报报告期截止日。 |
| `ann_date` | 该日可用的最新财报公告日，满足 `ann_date <= trade_date`。 |
| `roe` | 净资产收益率。 |
| `roa` | 总资产收益率。 |
| `revenue_yoy` | 营业收入同比增速。 |
| `debt_ratio` | 资产负债率。 |
| `gross_margin` | 毛利率。 |
| `eps` | 每股收益。 |
| `bps` | 每股净资产。 |
| `buy_sm_vol` | 小单买入量。 |
| `buy_sm_amount` | 小单买入金额。 |
| `sell_sm_vol` | 小单卖出量。 |
| `sell_sm_amount` | 小单卖出金额。 |
| `buy_md_vol` | 中单买入量。 |
| `buy_md_amount` | 中单买入金额。 |
| `sell_md_vol` | 中单卖出量。 |
| `sell_md_amount` | 中单卖出金额。 |
| `buy_lg_vol` | 大单买入量。 |
| `buy_lg_amount` | 大单买入金额。 |
| `sell_lg_vol` | 大单卖出量。 |
| `sell_lg_amount` | 大单卖出金额。 |
| `buy_elg_vol` | 特大单买入量。 |
| `buy_elg_amount` | 特大单买入金额。 |
| `sell_elg_vol` | 特大单卖出量。 |
| `sell_elg_amount` | 特大单卖出金额。 |
| `net_mf_vol` | 净流入量。 |
| `net_mf_amount` | 净流入金额。 |
| `industry_<category>` | 原始 `industry` 分类的 one-hot 动态列。 |
| `L1_<category>` | 一级行业 one-hot 动态列。 |
| `L2_<category>` | 二级行业 one-hot 动态列。 |
| `L3_<category>` | 三级行业 one-hot 动态列。 |

动态行业 one-hot 列说明：

- 动态列不是代码写死的固定清单，而是由输入行业表中实际出现的类别决定。
- 当前已生成数据中，宽表总列数为 293，其中固定列 43 个，行业 one-hot 动态列 250 个。
- 如需获得当前环境的完整动态列名，可以运行：

```python
import pandas as pd

df = pd.read_parquet(
    "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/industry/industry_onehot.parquet"
)
print(df.columns.tolist())
```

---

## 4. `features_generator`：因子生成类

### 4.1 price_volume_feature_generator.py

#### 4.1.1 `PriceVolumeFeatureGenerateSummary`

- **作用**：`PriceVolumeFeatureGenerator.process()` 的统计返回值。
- **字段**：输入/输出目录、读取/写出文件数和行数、日期范围、因子列名、`zero_close_rows`、`zero_volume_mean_rows`、`duplicate_key_rows`。

#### 4.1.2 `PriceVolumeFeatureGenerator`

- **作用**：从清洗后的 daily bars 生成价格量技术因子。
- **默认输入**：`data/cleaned_data/daily_bars`
- **默认输出**：`data/features_data/price_volume_factors`
- **注意**：为了计算 rolling/shift，若指定 `start_date`，会额外读取最多 60 个历史交易日。

输出样例：

| trade_date | ts_code | open | high | low | close | pct_chg | vol | amount | ret_5 | ret_10 | ret_20 | ret_60 | ma5_bias | ma10_bias | ma20_bias | ma60_bias | volatility_5 | volatility_20 | volatility_60 | amplitude | amplitude_5 | amplitude_20 | vol_ratio_5 | vol_ratio_20 | corr_price_vol_5 | corr_price_vol_20 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20240102 | 000001.SZ | 9.39 | 9.48 | 9.20 | 9.30 | -0.9585 | 412345.67 | 385000.12 | 0.0250 | 0.0310 | 0.0520 | 0.1200 | -0.0120 | -0.0180 | 0.0060 | 0.0330 | 1.53 | 1.92 | 2.31 | 0.0301 | 0.0280 | 0.0350 | 1.20 | 1.05 | 0.35 | 0.22 |

字段含义：

| 字段 | 含义 |
|---|---|
| `trade_date` | 交易日期。 |
| `ts_code` | 股票代码。 |
| `open` | 当日开盘价。 |
| `high` | 当日最高价。 |
| `low` | 当日最低价。 |
| `close` | 当日收盘价。 |
| `pct_chg` | 当日涨跌幅百分数。 |
| `vol` | 当日成交量。 |
| `amount` | 当日成交额。 |
| `ret_5` | 当前收盘价相对 5 个交易日前收盘价的收益率：`close / close.shift(5) - 1`。 |
| `ret_10` | 当前收盘价相对 10 个交易日前收盘价的收益率。 |
| `ret_20` | 当前收盘价相对 20 个交易日前收盘价的收益率。 |
| `ret_60` | 当前收盘价相对 60 个交易日前收盘价的收益率。 |
| `ma5_bias` | 收盘价相对 5 日均线偏离：`close / MA_5(close) - 1`。 |
| `ma10_bias` | 收盘价相对 10 日均线偏离。 |
| `ma20_bias` | 收盘价相对 20 日均线偏离。 |
| `ma60_bias` | 收盘价相对 60 日均线偏离。 |
| `volatility_5` | `pct_chg` 的 5 日滚动标准差。 |
| `volatility_20` | `pct_chg` 的 20 日滚动标准差。 |
| `volatility_60` | `pct_chg` 的 60 日滚动标准差。 |
| `amplitude` | 当日振幅：`(high - low) / close`。 |
| `amplitude_5` | `amplitude` 的 5 日滚动均值。 |
| `amplitude_20` | `amplitude` 的 20 日滚动均值。 |
| `vol_ratio_5` | 当前成交量 / 5 日成交量均值。 |
| `vol_ratio_20` | 当前成交量 / 20 日成交量均值。 |
| `corr_price_vol_5` | `close` 与 `vol` 的 5 日滚动相关系数。 |
| `corr_price_vol_20` | `close` 与 `vol` 的 20 日滚动相关系数。 |

使用示例：

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python -m utils.features_generator.price_volume_feature_generator \
  --start-date 20240101 --end-date 20240131
```

### 4.2 moneyflow_feature_generator.py

#### 4.2.1 `MoneyFlowFeatureGenerateSummary`

- **作用**：`MoneyFlowFeatureGenerator.process()` 的统计返回值。
- **字段**：资金流输入文件、daily bars 输入目录、输出目录、读取/写出统计、日期范围、因子列名、重复键行数、缺失/零成交额行数。

#### 4.2.2 `MoneyFlowFeatureGenerator`

- **作用**：从清洗后的 moneyflow 和 daily bars `amount` 生成资金流因子。
- **默认输入**：`data/cleaned_data/moneyflow/moneyflow.parquet` 与 `data/cleaned_data/daily_bars`
- **默认输出**：`data/features_data/moneyflow_factors`
- **单位处理**：`daily_bars.amount` 通常为千元，moneyflow 金额通常为万元；`amount_unit_scale` 默认 `0.1`，即将千元转换为万元后计算占比。
- **注意**：为计算 20 日滚动窗口，会额外读取最多 20 个历史资金流交易日。

输出样例：

| trade_date | ts_code | amount | buy_sm_amount | sell_sm_amount | buy_md_amount | sell_md_amount | buy_lg_amount | sell_lg_amount | buy_elg_amount | sell_elg_amount | net_mf_amount | lg_net_inflow | elg_net_inflow | main_net_inflow | main_net_ratio | retail_net_inflow | retail_ratio | main_retail_diff | main_net_5 | main_net_10 | main_net_20 | mf_ma5 | mf_ma20 | mf_acceleration |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20240102 | 000001.SZ | 385000.12 | 1200.5 | 1100.2 | 980.0 | 910.0 | 5000.0 | 4300.0 | 2200.0 | 1800.0 | 1500.3 | 700.0 | 400.0 | 1100.0 | 0.0286 | 100.3 | 0.0026 | 0.0260 | 5200.0 | 9800.0 | 18200.0 | 1040.0 | 910.0 | -13000.0 |

字段含义：

| 字段 | 含义 |
|---|---|
| `trade_date` | 交易日期。 |
| `ts_code` | 股票代码。 |
| `amount` | daily bars 成交额；用于分母计算，默认乘以 `amount_unit_scale=0.1` 后与 moneyflow 金额单位对齐。 |
| `buy_sm_amount` | 小单买入金额。 |
| `sell_sm_amount` | 小单卖出金额。 |
| `buy_md_amount` | 中单买入金额。 |
| `sell_md_amount` | 中单卖出金额。 |
| `buy_lg_amount` | 大单买入金额。 |
| `sell_lg_amount` | 大单卖出金额。 |
| `buy_elg_amount` | 特大单买入金额。 |
| `sell_elg_amount` | 特大单卖出金额。 |
| `net_mf_amount` | 原始净流入金额。 |
| `lg_net_inflow` | 大单净流入：`buy_lg_amount - sell_lg_amount`。 |
| `elg_net_inflow` | 特大单净流入：`buy_elg_amount - sell_elg_amount`。 |
| `main_net_inflow` | 主力净流入：`lg_net_inflow + elg_net_inflow`。 |
| `main_net_ratio` | 主力净流入 / 经单位转换后的 `amount`。 |
| `retail_net_inflow` | 小单净流入：`buy_sm_amount - sell_sm_amount`。 |
| `retail_ratio` | 小单净流入 / 经单位转换后的 `amount`。 |
| `main_retail_diff` | 主力与散户资金占比差：`main_net_ratio - retail_ratio`。 |
| `main_net_5` | 单股票 `main_net_inflow` 5 日滚动和。 |
| `main_net_10` | 单股票 `main_net_inflow` 10 日滚动和。 |
| `main_net_20` | 单股票 `main_net_inflow` 20 日滚动和。 |
| `mf_ma5` | 单股票 `main_net_inflow` 5 日滚动均值。 |
| `mf_ma20` | 单股票 `main_net_inflow` 20 日滚动均值。 |
| `mf_acceleration` | 资金流加速度：`main_net_5 - main_net_20`。 |

### 4.3 fundamental_feature_generator.py

#### 4.3.1 `FundamentalFeatureGenerateSummary`

- **作用**：`FundamentalFeatureGenerator.process()` 的统计返回值。
- **字段**：财务输入文件、daily bars 输入目录、输出目录、读取/写出统计、日期范围、因子列名、重复日频键行数、无可用财务数据的行数。

#### 4.3.2 `FundamentalFeatureGenerator`

- **作用**：将公告日维度的财务数据转换为日频财务因子，并生成截面排名。
- **默认输入**：`data/cleaned_data/fundamentals/fundamentals.parquet` 与 `data/cleaned_data/daily_bars`
- **默认输出**：`data/features_data/fundamental_factors`
- **点时逻辑**：每个交易日、每只股票只使用 `ann_date <= trade_date` 的最新财报；若同一公告日有多条记录，使用更晚的 `end_date`。

输出样例：

| trade_date | ts_code | ann_date | end_date | roe | roa | revenue_yoy | gross_margin | debt_ratio | eps | bps | roe_roa_gap | roe_rank | roa_rank | revenue_yoy_rank |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20240102 | 000001.SZ | 20231030 | 20230930 | 3.20 | 0.25 | 6.10 | 52.3 | 91.2 | 0.45 | 21.8 | 2.95 | 0.72 | 0.66 | 0.58 |

字段含义：

| 字段 | 含义 |
|---|---|
| `trade_date` | 交易日期。 |
| `ts_code` | 股票代码。 |
| `ann_date` | 该日可用的最新财报公告日，满足 `ann_date <= trade_date`。 |
| `end_date` | 该日可用的最新财报报告期截止日。 |
| `roe` | 净资产收益率。 |
| `roa` | 总资产收益率。 |
| `revenue_yoy` | 营业收入同比增速。 |
| `gross_margin` | 毛利率。 |
| `debt_ratio` | 资产负债率。 |
| `eps` | 每股收益。 |
| `bps` | 每股净资产。 |
| `roe_roa_gap` | 盈利质量差：`roe - roa`。 |
| `roe_rank` | 当日 `roe` 截面百分位排名，源值越大排名越高。 |
| `roa_rank` | 当日 `roa` 截面百分位排名。 |
| `revenue_yoy_rank` | 当日 `revenue_yoy` 截面百分位排名。 |

### 4.4 industry_feature_generator.py

#### 4.4.1 `IndustryFeatureGenerateSummary`

- **作用**：`IndustryFeatureGenerator.process()` 的统计返回值。
- **字段**：行业输入文件、价格量因子输入目录、财务因子输入目录、输出目录、读取/写出统计、缺失财务文件数、因子列名、L1 one-hot 列名、缺失 L1 行数、输入重复键行数。

#### 4.4.2 `IndustryFeatureGenerator`

- **作用**：基于一级行业、价格量因子和财务因子生成行业收益、相对强弱和行业中性化财务因子。
- **默认输入**：
  - 行业映射：`data/cleaned_data/industry/industry.parquet`
  - 价格量因子：`data/features_data/price_volume_factors`
  - 财务因子：`data/features_data/fundamental_factors`
- **默认输出**：`data/features_data/industry_factors`
- **规则**：
  1. 只使用 `L1_industry_name` 做行业归属。
  2. `industry_ret_1` 使用 price-volume 中的 `ret_1`；若没有则把 `pct_chg / 100` 作为日收益。
  3. 行业收益是同日同 L1 行业股票收益的截面均值。
  4. 行业中性化财务因子 = 个股财务因子 - 所属 L1 行业均值。

输出样例：

| trade_date | ts_code | L1_industry_name | L1_交通运输 | L1_公用事业 | L1_商贸零售 | L1_建筑材料 | L1_建筑装饰 | L1_房地产 | L1_煤炭 | L1_环保 | L1_电力设备 | L1_电子 | L1_石油石化 | L1_纺织服饰 | L1_综合 | L1_美容护理 | L1_通信 | L1_钢铁 | L1_非银金融 | L1_食品饮料 | industry_ret_1 | industry_ret_5 | industry_ret_20 | relative_strength_5 | relative_strength_20 | roe_ind_neutral | roa_ind_neutral | revenue_yoy_ind_neutral |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20240102 | 000001.SZ | 非银金融 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | -0.003 | 0.015 | 0.040 | 0.010 | 0.020 | 0.20 | 0.03 | 1.20 |

字段含义：

| 字段 | 含义 |
|---|---|
| `trade_date` | 交易日期。 |
| `ts_code` | 股票代码。 |
| `L1_industry_name` | 一级行业名称，来自清洗后的行业映射。 |
| `L1_交通运输` | 一级行业“交通运输”的 one-hot，属于该行业为 1，否则 0。 |
| `L1_公用事业` | 一级行业“公用事业”的 one-hot。 |
| `L1_商贸零售` | 一级行业“商贸零售”的 one-hot。 |
| `L1_建筑材料` | 一级行业“建筑材料”的 one-hot。 |
| `L1_建筑装饰` | 一级行业“建筑装饰”的 one-hot。 |
| `L1_房地产` | 一级行业“房地产”的 one-hot。 |
| `L1_煤炭` | 一级行业“煤炭”的 one-hot。 |
| `L1_环保` | 一级行业“环保”的 one-hot。 |
| `L1_电力设备` | 一级行业“电力设备”的 one-hot。 |
| `L1_电子` | 一级行业“电子”的 one-hot。 |
| `L1_石油石化` | 一级行业“石油石化”的 one-hot。 |
| `L1_纺织服饰` | 一级行业“纺织服饰”的 one-hot。 |
| `L1_综合` | 一级行业“综合”的 one-hot。 |
| `L1_美容护理` | 一级行业“美容护理”的 one-hot。 |
| `L1_通信` | 一级行业“通信”的 one-hot。 |
| `L1_钢铁` | 一级行业“钢铁”的 one-hot。 |
| `L1_非银金融` | 一级行业“非银金融”的 one-hot。 |
| `L1_食品饮料` | 一级行业“食品饮料”的 one-hot。 |
| `industry_ret_1` | 当日同 L1 行业内 `ret_1` 截面均值。 |
| `industry_ret_5` | 当日同 L1 行业内 `ret_5` 截面均值。 |
| `industry_ret_20` | 当日同 L1 行业内 `ret_20` 截面均值。 |
| `relative_strength_5` | 个股 `ret_5 - industry_ret_5`。 |
| `relative_strength_20` | 个股 `ret_20 - industry_ret_20`。 |
| `roe_ind_neutral` | 个股 `roe` 减所属 L1 行业当日 `roe` 均值。 |
| `roa_ind_neutral` | 个股 `roa` 减所属 L1 行业当日 `roa` 均值。 |
| `revenue_yoy_ind_neutral` | 个股 `revenue_yoy` 减所属 L1 行业当日 `revenue_yoy` 均值。 |

---

## 5. `label_generator`：监督学习标签生成类

本目录用于从已生成的日频宽表构造监督学习 label。当前实现会读取 `data/processd_data/wide_table_daily_bars` 中的 `close` 与 `adj_factor`，先为每只股票构造连续复权收盘价，再计算多个未来交易日收益率及同日截面百分位排名。上游宽表只读，输出独立写入 `data/generated_label/daily_labels/year=YYYY/month=MM/YYYYMMDD.parquet`。

### 5.1 daily_label_generator.py

#### 5.1.1 `DailyLabelGenerateSummary`

- **所在文件**：`src/utils/label_generator/daily_label_generator.py`
- **作用**：`DailyLabelGenerator.process()` 的聚合统计返回值。
- **主要字段**：
  - `input_dir` / `output_dir`：输入宽表目录与输出 label 目录。
  - `files_read` / `files_written`：读取与写出的 parquet 文件数。注意为计算未来收益，每个输出批次会额外读取最多 20 个未来交易日文件，因此 `files_read` 可能大于最终输出文件数。
  - `rows_read` / `rows_written`：读取与写出的总行数。
  - `start_date` / `end_date`：本次输出日期范围；为空时表示全量输出。
  - `label_columns`：未来收益 label 列，当前为 `label_1d, label_2d, label_3d, label_5d, label_10d, label_20d`。
  - `rank_columns`：同日截面百分位排名列，当前为 `label_rank_1d, label_rank_2d, label_rank_3d, label_rank_5d, label_rank_10d, label_rank_20d`。
  - `missing_source_rows`：输入 `close` 或 `adj_factor` 缺失/非数值的行数。
  - `zero_adj_close_rows`：复权收盘价为 0 的行数，对应 label/rank 会保留为 NaN。
  - `duplicate_key_rows`：输入中重复 `trade_date + ts_code` 键的行数。

#### 5.1.2 `DailyLabelGenerator`

- **作用**：生成日频未来收益 label 和同日截面 rank label。
- **默认输入**：`/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/wide_table_daily_bars`
- **默认输出**：`/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/generated_label/daily_labels`
- **默认日志**：`/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/label_generate/daily_labels.log`
- **必需输入列**：`trade_date, ts_code, close, adj_factor`。
- **默认预测窗口**：`1, 2, 3, 5, 10, 20` 个交易日。
- **核心规则**：
  1. 全量扫描输入宽表，按股票取得最新有效复权因子 `latest_factor`。
  2. 对每只股票构造统一口径复权收盘价：`adj_close(t) = close(t) * adj_factor(t) / latest_factor`。
  3. 在单股票时间序列内按 `trade_date` 排序，计算未来 `h` 个交易日收益：`label_hd = adj_close(t+h) / adj_close(t) - 1`。
  4. 在同一个 `trade_date` 的股票截面内，对 `label_hd` 做百分位排名：`label_rank_hd = rank(label_hd, pct=True)`。
  5. 输出只保留键、`adj_close`、收益 label 和 rank label；不会把上游宽表全部复制到 label 结果中。
  6. 最后不足 `h` 个未来交易日的样本、`close/adj_factor` 缺失样本、`latest_factor=0` 或 `adj_close=0` 的样本，对应 label/rank 会保留为 NaN。

使用示例：

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src" \
python -m utils.label_generator.daily_label_generator \
  --input-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/wide_table_daily_bars" \
  --output-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/generated_label/daily_labels" \
  --start-date 20240101 \
  --end-date 20240131 \
  --output-batch-size 120 \
  --log-file "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log/label_generate/daily_labels.log"
```

Python 直接调用：

```python
from utils.label_generator.daily_label_generator import DailyLabelGenerator

summary = DailyLabelGenerator(start_date=20240101, end_date=20240131).process()
print(summary.files_written, summary.rows_written, summary.label_columns)
```

输出样例：

| trade_date | ts_code | adj_close | label_1d | label_2d | label_3d | label_5d | label_10d | label_20d | label_rank_1d | label_rank_2d | label_rank_3d | label_rank_5d | label_rank_10d | label_rank_20d |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 20000104 | 000003.SZ | 5.74000 | -0.010453 | 0.019164 | 0.066202 | 0.026132 | 0.006969 | 0.036585 | 0.161765 | 0.213235 | 0.426471 | 0.625000 | 0.470588 | 0.176471 |
| 20000104 | 000005.SZ | 2.98467 | 0.001603 | 0.036859 | 0.083333 | 0.036859 | 0.165064 | 0.506410 | 0.411765 | 0.500000 | 0.617647 | 0.639706 | 0.904412 | 0.860294 |

字段含义：

| 字段 | 含义 |
|---|---|
| `trade_date` | label 对应的当前交易日期，`YYYYMMDD`。 |
| `ts_code` | 股票代码。 |
| `adj_close` | 使用全历史最新复权因子统一归一后的复权收盘价，作为未来收益 label 的计算基准。 |
| `label_1d` | 当前交易日到 1 个未来交易日后的复权收益率。 |
| `label_2d` | 当前交易日到 2 个未来交易日后的复权收益率。 |
| `label_3d` | 当前交易日到 3 个未来交易日后的复权收益率。 |
| `label_5d` | 当前交易日到 5 个未来交易日后的复权收益率。 |
| `label_10d` | 当前交易日到 10 个未来交易日后的复权收益率。 |
| `label_20d` | 当前交易日到 20 个未来交易日后的复权收益率。 |
| `label_rank_1d` | 当日 `label_1d` 的截面百分位排名，取值通常为 0~1，越大表示未来 1 日收益在当日股票截面中越靠前。 |
| `label_rank_2d` | 当日 `label_2d` 的截面百分位排名。 |
| `label_rank_3d` | 当日 `label_3d` 的截面百分位排名。 |
| `label_rank_5d` | 当日 `label_5d` 的截面百分位排名。 |
| `label_rank_10d` | 当日 `label_10d` 的截面百分位排名。 |
| `label_rank_20d` | 当日 `label_20d` 的截面百分位排名。 |

> 注意：rank label 只在同一 `trade_date` 的可用 label 截面内排名，不跨日期比较；若原始 label 为 NaN，rank label 也为 NaN。

---

## 6. `cross_sectional_processor`：横截面处理类

本目录用于对已经生成的宽表或因子表做日内横截面处理：每个输入 parquet 只读，按 `trade_date` 分组，对指定因子先做 1%/99% 分位缩尾（winsorize），再做 z-score 标准化，并在原表基础上追加 `{factor}_cc_processed` 列，输出到 `data/cross_sectional_processd_data/<dataset>/year=YYYY/month=MM/YYYYMMDD.parquet`。

### 6.1 data_winsorize_util.py

#### 6.1.1 `winsorize`

- **作用**：对一个横截面 `pd.Series` 按 1% 和 99% 分位数裁剪极端值。
- **输入/输出**：输入单个因子序列，输出同索引的缩尾后序列。

#### 6.1.2 `winsorize_by_trade_date`

- **作用**：按 `trade_date` 分组，对某个因子列逐日调用 `winsorize`。
- **默认日期列**：`trade_date`。

### 6.2 data_z_score_util.py

#### 6.2.1 `zscore`

- **作用**：对一个横截面 `pd.Series` 做 z-score 标准化：`(x - mean) / std`。
- **特殊情况**：若横截面标准差为 0 或 NaN，则退化为 `x - mean`，避免除零。

#### 6.2.2 `zscore_by_trade_date`

- **作用**：按 `trade_date` 分组，对某个因子列逐日调用 `zscore`。
- **默认日期列**：`trade_date`。

### 6.3 price_volume_factors_cross_sectional_processor.py

#### 6.3.1 `PriceVolumeFactorsCrossSectionalSummary`

- **作用**：`PriceVolumeFactorsCrossSectionalProcessor.process()` 的统计返回值。
- **字段**：输入/输出目录、发现/处理文件数、读取/写出行数、目标因子列、缺失必需列文件、失败文件。
- **便捷属性**：`processed_columns` 返回追加列名；`has_errors` 标记是否存在跳过或失败文件。

#### 6.3.2 `PriceVolumeFactorsCrossSectionalProcessor`

- **作用**：对价格量因子做逐日横截面缩尾 + z-score 标准化。
- **默认输入**：`data/features_data/price_volume_factors`
- **默认输出**：`data/cross_sectional_processd_data/price_volume_factors`
- **默认处理因子**：`ret_5, ret_10, ret_20, ret_60, ma5_bias, ma10_bias, ma20_bias, ma60_bias, volatility_5, volatility_20, volatility_60, amplitude, amplitude_5, amplitude_20, vol_ratio_5, vol_ratio_20, corr_price_vol_5, corr_price_vol_20`。
- **输出规则**：保留全部原始列，并追加对应的 `{factor}_cc_processed` 列。

### 6.4 moneyflow_factors_cross_sectional_processor.py

#### 6.4.1 `MoneyflowFactorsCrossSectionalSummary`

- **作用**：`MoneyflowFactorsCrossSectionalProcessor.process()` 的统计返回值。
- **字段**：输入/输出目录、发现/处理文件数、读取/写出行数、目标因子列、缺失必需列文件、失败文件。

#### 6.4.2 `MoneyflowFactorsCrossSectionalProcessor`

- **作用**：对资金流因子做逐日横截面缩尾 + z-score 标准化。
- **默认输入**：`data/features_data/moneyflow_factors`
- **默认输出**：`data/cross_sectional_processd_data/moneyflow_factors`
- **默认处理因子**：`lg_net_inflow, elg_net_inflow, main_net_inflow, main_net_ratio, retail_net_inflow, retail_ratio, main_retail_diff, main_net_5, main_net_10, main_net_20, mf_ma5, mf_ma20, mf_acceleration`。
- **输出规则**：保留全部原始列，并追加对应的 `{factor}_cc_processed` 列。

### 6.5 fundamental_factors_cross_sectional_processor.py

#### 6.5.1 `FundamentalFactorsCrossSectionalSummary`

- **作用**：`FundamentalFactorsCrossSectionalProcessor.process()` 的统计返回值。
- **字段**：输入/输出目录、发现/处理文件数、读取/写出行数、目标因子列、缺失必需列文件、失败文件。

#### 6.5.2 `FundamentalFactorsCrossSectionalProcessor`

- **作用**：对财务因子做逐日横截面缩尾 + z-score 标准化。
- **默认输入**：`data/features_data/fundamental_factors`
- **默认输出**：`data/cross_sectional_processd_data/fundamental_factors`
- **默认处理因子**：`roe_roa_gap`。
- **输出规则**：保留全部原始列，并追加 `roe_roa_gap_cc_processed`。

### 6.6 industry_factors_cross_sectional_processor.py

#### 6.6.1 `IndustryFactorsCrossSectionalSummary`

- **作用**：`IndustryFactorsCrossSectionalProcessor.process()` 的统计返回值。
- **字段**：输入/输出目录、发现/处理文件数、读取/写出行数、目标因子列、缺失必需列文件、失败文件。

#### 6.6.2 `IndustryFactorsCrossSectionalProcessor`

- **作用**：对行业收益、相对强弱、行业中性财务因子做逐日横截面缩尾 + z-score 标准化。
- **默认输入**：`data/features_data/industry_factors`
- **默认输出**：`data/cross_sectional_processd_data/industry_factors`
- **默认处理因子**：`industry_ret_1, industry_ret_5, industry_ret_20, relative_strength_5, relative_strength_20, roe_ind_neutral, roa_ind_neutral, revenue_yoy_ind_neutral`。
- **输出规则**：保留全部原始列，并追加对应的 `{factor}_cc_processed` 列。

### 6.7 wide_table_daily_bars_cross_sectional_processor.py

#### 6.7.1 `WideTableDailyBarsCrossSectionalSummary`

- **作用**：`WideTableDailyBarsCrossSectionalProcessor.process()` 的统计返回值。
- **字段**：输入/输出目录、发现/处理文件数、读取/写出行数、目标因子列、缺失必需列文件、失败文件。

#### 6.7.2 `WideTableDailyBarsCrossSectionalProcessor`

- **作用**：对日频宽表中的财务字段做逐日横截面缩尾 + z-score 标准化，便于直接基于宽表建模时使用统一处理后的财务因子。
- **默认输入**：`data/processd_data/wide_table_daily_bars`
- **默认输出**：`data/cross_sectional_processd_data/wide_table_daily_bars`
- **默认处理因子**：`roe, roa, revenue_yoy, debt_ratio, gross_margin, eps, bps`。
- **输出规则**：保留宽表原始 293 列，并追加 7 个 `{factor}_cc_processed` 列。

#### 6.7.3 `MissingRequiredColumnsError`

- **作用**：各横截面 processor 文件中同名定义的异常类；当输入 parquet 缺少 `trade_date`、`ts_code` 或目标因子列时抛出。
- **处理方式**：默认逐文件记录并跳过；传入 `--fail-fast` 时遇到首个异常即终止。

使用示例：

```bash
PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab" \
python -m src.utils.cross_sectional_processor.price_volume_factors_cross_sectional_processor \
  --input-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/features_data/price_volume_factors" \
  --output-dir "/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cross_sectional_processd_data/price_volume_factors"
```

---

## 7. 当前环境完整输出列清单

本节从当前已生成 parquet 的 schema 中读取列名，列出多列/动态列输出的完整列顺序。若上游行业类别发生变化，`industry_*`、`L1_*`、`L2_*`、`L3_*` 这类动态 one-hot 列会随输入数据变化；应以实际 parquet schema 为准。

### `generated_label` 数据目录说明

`/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/generated_label` 是监督学习 label 输出目录。当前已生成的 label 数据位于：

```text
data/generated_label/daily_labels/year=YYYY/month=MM/YYYYMMDD.parquet
```

该目录下每个 parquet 对应一个交易日，保留 `trade_date + ts_code` 键、统一口径 `adj_close`，以及 6 个未来收益 label 和 6 个同日截面 rank label。当前环境中 `daily_labels` 的统计如下：

| 子目录 | 上游来源 | 日期范围 | 文件数 | 行数 | 总列数 | 收益 label 列数 | rank label 列数 |
|---|---|---:|---:|---:|---:|---:|---:|
| `daily_labels` | `data/processd_data/wide_table_daily_bars` | 20000104 ~ 20260526 | 6385 | 16802319 | 15 | 6 | 6 |

使用建议：

- `label_*d` 适合作为回归目标，表示未来 `h` 个交易日复权收益率。
- `label_rank_*d` 适合作为截面排序/学习排序目标，取值为当日可用股票截面的百分位排名。
- 末尾日期由于未来窗口不足会出现 NaN；历史早期或个别股票的 NaN 通常来自上游 `close/adj_factor` 缺失或无法构造有效 `adj_close`。

### `cross_sectional_processd_data` 数据目录说明

`/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cross_sectional_processd_data` 是横截面处理后的建模输入目录。该目录下每个子目录都对应一个上游数据集，分区结构与上游保持一致：

```text
data/cross_sectional_processd_data/<dataset>/year=YYYY/month=MM/YYYYMMDD.parquet
```

处理后的 parquet **保留所有上游原始列**，并在末尾追加若干 `{factor}_cc_processed` 列。追加列的含义统一为：在同一个 `trade_date` 的股票截面内，对原始因子先做 1%/99% 分位缩尾，再做 z-score 标准化。它适合作为截面模型、线性模型或需要同日可比尺度的模型输入；若需要原始数值，仍可直接读取未加后缀的原始列。

当前环境中该目录包含以下数据：

| 子目录 | 上游来源 | 日期范围 | 文件数 | 行数 | 总列数 | 追加横截面列数 | 主要追加列 |
|---|---|---:|---:|---:|---:|---:|---|
| `price_volume_factors` | `data/features_data/price_volume_factors` | 20000104 ~ 20260526 | 6385 | 16802319 | 45 | 18 | `ret_*_cc_processed`、`ma*_bias_cc_processed`、`volatility_*_cc_processed`、`amplitude*_cc_processed`、`vol_ratio_*_cc_processed`、`corr_price_vol_*_cc_processed` |
| `moneyflow_factors` | `data/features_data/moneyflow_factors` | 20070104 ~ 20260526 | 4692 | 14178504 | 38 | 13 | `main_net_*_cc_processed`、`retail_*_cc_processed`、`mf_*_cc_processed`、`lg/elg_net_inflow_cc_processed` |
| `fundamental_factors` | `data/features_data/fundamental_factors` | 20000104 ~ 20260526 | 6385 | 16802319 | 16 | 1 | `roe_roa_gap_cc_processed` |
| `industry_factors` | `data/features_data/industry_factors` | 20000104 ~ 20260526 | 6385 | 16802319 | 37 | 8 | `industry_ret_*_cc_processed`、`relative_strength_*_cc_processed`、`*_ind_neutral_cc_processed` |
| `wide_table_daily_bars` | `data/processd_data/wide_table_daily_bars` | 20000104 ~ 20260526 | 6385 | 16802319 | 300 | 7 | `roe_cc_processed`、`roa_cc_processed`、`revenue_yoy_cc_processed`、`debt_ratio_cc_processed`、`gross_margin_cc_processed`、`eps_cc_processed`、`bps_cc_processed` |

各子目录的使用建议：

- `price_volume_factors`：用于技术面/量价截面模型，追加列消除了不同因子量纲差异和极端值影响。
- `moneyflow_factors`：用于资金流截面模型；由于资金流原始数据从 2007 年开始，日期范围晚于其他主题因子。
- `fundamental_factors`：用于财务质量截面模型，目前只对 `roe_roa_gap` 做横截面处理。
- `industry_factors`：用于行业收益、个股相对行业强弱、行业中性财务因子的截面建模。
- `wide_table_daily_bars`：用于直接基于宽表建模，仅追加宽表中基础财务字段的横截面处理结果；原始行情、资金流和行业 one-hot 列保持不变。

> 注意：`*_cc_processed` 列中的 NaN 通常来自原始因子缺失、上市早期滚动窗口不足，或同日有效样本不足；横截面处理不会主动填充这些 NaN。

### IndustryOneHotProcessor 输出 `industry_onehot.parquet`（251 列）

- schema 示例文件：`/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/industry/industry_onehot.parquet`

```text
001. ts_code
002. industry_IT设备
003. industry_专用机械
004. industry_中成药
005. industry_乳制品
006. industry_互联网
007. industry_仓储物流
008. industry_供气供热
009. industry_保险
010. industry_元器件
011. industry_全国地产
012. industry_公共交通
013. industry_公路
014. industry_其他商业
015. industry_其他建材
016. industry_农业综合
017. industry_农用机械
018. industry_农药化肥
019. industry_出版业
020. industry_化学制药
021. industry_化工原料
022. industry_化工机械
023. industry_化纤
024. industry_区域地产
025. industry_医疗保健
026. industry_医药商业
027. industry_半导体
028. industry_商品城
029. industry_商贸代理
030. industry_啤酒
031. industry_园区开发
032. industry_塑料
033. industry_多元金融
034. industry_家居用品
035. industry_家用电器
036. industry_小金属
037. industry_工程机械
038. industry_广告包装
039. industry_建筑工程
040. industry_影视音像
041. industry_房产服务
042. industry_批发业
043. industry_摩托车
044. industry_文教休闲
045. industry_新型电力
046. industry_旅游景点
047. industry_旅游服务
048. industry_日用化工
049. industry_普钢
050. industry_服饰
051. industry_机场
052. industry_机床制造
053. industry_机械基件
054. industry_林业
055. industry_染料涂料
056. industry_橡胶
057. industry_水力发电
058. industry_水务
059. industry_水泥
060. industry_水运
061. industry_汽车整车
062. industry_汽车服务
063. industry_汽车配件
064. industry_渔业
065. industry_港口
066. industry_火力发电
067. industry_焦炭加工
068. industry_煤炭开采
069. industry_特种钢
070. industry_环境保护
071. industry_玻璃
072. industry_生物制药
073. industry_电信运营
074. industry_电器仪表
075. industry_电器连锁
076. industry_电气设备
077. industry_白酒
078. industry_百货
079. industry_石油加工
080. industry_石油开采
081. industry_石油贸易
082. industry_矿物制品
083. industry_种植业
084. industry_空运
085. industry_红黄酒
086. industry_纺织
087. industry_纺织机械
088. industry_综合类
089. industry_航空
090. industry_船舶
091. industry_装修装饰
092. industry_证券
093. industry_超市连锁
094. industry_路桥
095. industry_软件服务
096. industry_软饮料
097. industry_轻工机械
098. industry_运输设备
099. industry_通信设备
100. industry_造纸
101. industry_酒店餐饮
102. industry_钢加工
103. industry_铁路
104. industry_铅锌
105. industry_铜
106. industry_铝
107. industry_银行
108. industry_陶瓷
109. industry_食品
110. industry_饲料
111. industry_黄金
112. L1_交通运输
113. L1_公用事业
114. L1_商贸零售
115. L1_建筑材料
116. L1_建筑装饰
117. L1_房地产
118. L1_煤炭
119. L1_环保
120. L1_电力设备
121. L1_电子
122. L1_石油石化
123. L1_纺织服饰
124. L1_综合
125. L1_美容护理
126. L1_通信
127. L1_钢铁
128. L1_非银金融
129. L1_食品饮料
130. L2_一般零售
131. L2_专业工程
132. L2_专业连锁Ⅱ
133. L2_乘用车
134. L2_元件
135. L2_光伏设备
136. L2_光学光电子
137. L2_养殖业
138. L2_农产品加工
139. L2_冶钢原料
140. L2_包装印刷
141. L2_化学制品
142. L2_化学原料
143. L2_厨卫电器
144. L2_商用车
145. L2_国有大型银行Ⅱ
146. L2_基础建设
147. L2_塑料
148. L2_家电零部件Ⅱ
149. L2_小金属
150. L2_工业金属
151. L2_广告营销
152. L2_影视院线
153. L2_房地产开发
154. L2_普钢
155. L2_橡胶
156. L2_油服工程
157. L2_消费电子
158. L2_渔业
159. L2_游戏Ⅱ
160. L2_煤炭开采
161. L2_照明设备Ⅱ
162. L2_燃气Ⅱ
163. L2_特钢Ⅱ
164. L2_环保设备Ⅱ
165. L2_环境治理
166. L2_玻璃玻纤
167. L2_电力
168. L2_电池
169. L2_电网设备
170. L2_种植业
171. L2_综合Ⅱ
172. L2_自动化设备
173. L2_航天装备Ⅱ
174. L2_航海装备Ⅱ
175. L2_航运港口
176. L2_装修装饰Ⅱ
177. L2_贸易Ⅱ
178. L2_软件开发
179. L2_通信设备
180. L2_通用设备
181. L2_造纸
182. L2_铁路公路
183. L2_非金属材料Ⅱ
184. L2_食品加工
185. L2_饮料乳品
186. L2_黑色家电
187. L3_中间产品及消费品供应链服务
188. L3_产业地产
189. L3_人工景区
190. L3_仓储物流
191. L3_住宅开发
192. L3_保险Ⅲ
193. L3_公交
194. L3_公路货运
195. L3_其他农产品加工
196. L3_其他塑料制品
197. L3_其他建材
198. L3_其他石化
199. L3_制冷空调设备
200. L3_动物保健Ⅲ
201. L3_印制电路板
202. L3_印刷包装机械
203. L3_原材料供应链服务
204. L3_商业地产
205. L3_商业物业经营
206. L3_复合肥
207. L3_多业态零售
208. L3_工程机械器件
209. L3_快递
210. L3_房屋建设Ⅲ
211. L3_数字芯片设计
212. L3_旅游综合
213. L3_有机硅
214. L3_机床工具
215. L3_果蔬加工
216. L3_楼宇设备
217. L3_模拟芯片设计
218. L3_汽车经销商
219. L3_消费电子零部件及组装
220. L3_涤纶
221. L3_港口
222. L3_游戏Ⅲ
223. L3_激光设备
224. L3_煤化工
225. L3_燃气Ⅲ
226. L3_物业管理
227. L3_电商服务
228. L3_电机Ⅲ
229. L3_畜禽饲料
230. L3_百货
231. L3_种子
232. L3_粮油加工
233. L3_纺织服装设备
234. L3_能源及重型设备
235. L3_自然景区
236. L3_航空运输
237. L3_航运
238. L3_装修装饰Ⅲ
239. L3_证券Ⅲ
240. L3_资产管理
241. L3_超市
242. L3_跨境物流
243. L3_跨境电商
244. L3_酒店
245. L3_金属制品
246. L3_金融控股
247. L3_钛白粉
248. L3_铁路运输
249. L3_铜
250. L3_食品及饲料添加剂
251. L3_高速公路
```

### WideTableDailyBarsBuilder 输出 `wide_table_daily_bars`（293 列）

- schema 示例文件：`/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/processd_data/wide_table_daily_bars/year=2000/month=01/20000104.parquet`

```text
001. ts_code
002. trade_date
003. open
004. high
005. low
006. close
007. pre_close
008. change
009. pct_chg
010. vol
011. amount
012. no_st_stock
013. st_stock
014. star_st_stock
015. is_suspect
016. adj_factor
017. end_date
018. ann_date
019. roe
020. roa
021. revenue_yoy
022. debt_ratio
023. gross_margin
024. eps
025. bps
026. buy_sm_vol
027. buy_sm_amount
028. sell_sm_vol
029. sell_sm_amount
030. buy_md_vol
031. buy_md_amount
032. sell_md_vol
033. sell_md_amount
034. buy_lg_vol
035. buy_lg_amount
036. sell_lg_vol
037. sell_lg_amount
038. buy_elg_vol
039. buy_elg_amount
040. sell_elg_vol
041. sell_elg_amount
042. net_mf_vol
043. net_mf_amount
044. industry_IT设备
045. industry_专用机械
046. industry_中成药
047. industry_乳制品
048. industry_互联网
049. industry_仓储物流
050. industry_供气供热
051. industry_保险
052. industry_元器件
053. industry_全国地产
054. industry_公共交通
055. industry_公路
056. industry_其他商业
057. industry_其他建材
058. industry_农业综合
059. industry_农用机械
060. industry_农药化肥
061. industry_出版业
062. industry_化学制药
063. industry_化工原料
064. industry_化工机械
065. industry_化纤
066. industry_区域地产
067. industry_医疗保健
068. industry_医药商业
069. industry_半导体
070. industry_商品城
071. industry_商贸代理
072. industry_啤酒
073. industry_园区开发
074. industry_塑料
075. industry_多元金融
076. industry_家居用品
077. industry_家用电器
078. industry_小金属
079. industry_工程机械
080. industry_广告包装
081. industry_建筑工程
082. industry_影视音像
083. industry_房产服务
084. industry_批发业
085. industry_摩托车
086. industry_文教休闲
087. industry_新型电力
088. industry_旅游景点
089. industry_旅游服务
090. industry_日用化工
091. industry_普钢
092. industry_服饰
093. industry_机场
094. industry_机床制造
095. industry_机械基件
096. industry_林业
097. industry_染料涂料
098. industry_橡胶
099. industry_水力发电
100. industry_水务
101. industry_水泥
102. industry_水运
103. industry_汽车整车
104. industry_汽车服务
105. industry_汽车配件
106. industry_渔业
107. industry_港口
108. industry_火力发电
109. industry_焦炭加工
110. industry_煤炭开采
111. industry_特种钢
112. industry_环境保护
113. industry_玻璃
114. industry_生物制药
115. industry_电信运营
116. industry_电器仪表
117. industry_电器连锁
118. industry_电气设备
119. industry_白酒
120. industry_百货
121. industry_石油加工
122. industry_石油开采
123. industry_石油贸易
124. industry_矿物制品
125. industry_种植业
126. industry_空运
127. industry_红黄酒
128. industry_纺织
129. industry_纺织机械
130. industry_综合类
131. industry_航空
132. industry_船舶
133. industry_装修装饰
134. industry_证券
135. industry_超市连锁
136. industry_路桥
137. industry_软件服务
138. industry_软饮料
139. industry_轻工机械
140. industry_运输设备
141. industry_通信设备
142. industry_造纸
143. industry_酒店餐饮
144. industry_钢加工
145. industry_铁路
146. industry_铅锌
147. industry_铜
148. industry_铝
149. industry_银行
150. industry_陶瓷
151. industry_食品
152. industry_饲料
153. industry_黄金
154. L1_交通运输
155. L1_公用事业
156. L1_商贸零售
157. L1_建筑材料
158. L1_建筑装饰
159. L1_房地产
160. L1_煤炭
161. L1_环保
162. L1_电力设备
163. L1_电子
164. L1_石油石化
165. L1_纺织服饰
166. L1_综合
167. L1_美容护理
168. L1_通信
169. L1_钢铁
170. L1_非银金融
171. L1_食品饮料
172. L2_一般零售
173. L2_专业工程
174. L2_专业连锁Ⅱ
175. L2_乘用车
176. L2_元件
177. L2_光伏设备
178. L2_光学光电子
179. L2_养殖业
180. L2_农产品加工
181. L2_冶钢原料
182. L2_包装印刷
183. L2_化学制品
184. L2_化学原料
185. L2_厨卫电器
186. L2_商用车
187. L2_国有大型银行Ⅱ
188. L2_基础建设
189. L2_塑料
190. L2_家电零部件Ⅱ
191. L2_小金属
192. L2_工业金属
193. L2_广告营销
194. L2_影视院线
195. L2_房地产开发
196. L2_普钢
197. L2_橡胶
198. L2_油服工程
199. L2_消费电子
200. L2_渔业
201. L2_游戏Ⅱ
202. L2_煤炭开采
203. L2_照明设备Ⅱ
204. L2_燃气Ⅱ
205. L2_特钢Ⅱ
206. L2_环保设备Ⅱ
207. L2_环境治理
208. L2_玻璃玻纤
209. L2_电力
210. L2_电池
211. L2_电网设备
212. L2_种植业
213. L2_综合Ⅱ
214. L2_自动化设备
215. L2_航天装备Ⅱ
216. L2_航海装备Ⅱ
217. L2_航运港口
218. L2_装修装饰Ⅱ
219. L2_贸易Ⅱ
220. L2_软件开发
221. L2_通信设备
222. L2_通用设备
223. L2_造纸
224. L2_铁路公路
225. L2_非金属材料Ⅱ
226. L2_食品加工
227. L2_饮料乳品
228. L2_黑色家电
229. L3_中间产品及消费品供应链服务
230. L3_产业地产
231. L3_人工景区
232. L3_仓储物流
233. L3_住宅开发
234. L3_保险Ⅲ
235. L3_公交
236. L3_公路货运
237. L3_其他农产品加工
238. L3_其他塑料制品
239. L3_其他建材
240. L3_其他石化
241. L3_制冷空调设备
242. L3_动物保健Ⅲ
243. L3_印制电路板
244. L3_印刷包装机械
245. L3_原材料供应链服务
246. L3_商业地产
247. L3_商业物业经营
248. L3_复合肥
249. L3_多业态零售
250. L3_工程机械器件
251. L3_快递
252. L3_房屋建设Ⅲ
253. L3_数字芯片设计
254. L3_旅游综合
255. L3_有机硅
256. L3_机床工具
257. L3_果蔬加工
258. L3_楼宇设备
259. L3_模拟芯片设计
260. L3_汽车经销商
261. L3_消费电子零部件及组装
262. L3_涤纶
263. L3_港口
264. L3_游戏Ⅲ
265. L3_激光设备
266. L3_煤化工
267. L3_燃气Ⅲ
268. L3_物业管理
269. L3_电商服务
270. L3_电机Ⅲ
271. L3_畜禽饲料
272. L3_百货
273. L3_种子
274. L3_粮油加工
275. L3_纺织服装设备
276. L3_能源及重型设备
277. L3_自然景区
278. L3_航空运输
279. L3_航运
280. L3_装修装饰Ⅲ
281. L3_证券Ⅲ
282. L3_资产管理
283. L3_超市
284. L3_跨境物流
285. L3_跨境电商
286. L3_酒店
287. L3_金属制品
288. L3_金融控股
289. L3_钛白粉
290. L3_铁路运输
291. L3_铜
292. L3_食品及饲料添加剂
293. L3_高速公路
```

### PriceVolumeFeatureGenerator 输出 `price_volume_factors`（27 列）

- schema 示例文件：`/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/features_data/price_volume_factors/year=2000/month=01/20000104.parquet`

```text
001. trade_date
002. ts_code
003. open
004. high
005. low
006. close
007. pct_chg
008. vol
009. amount
010. ret_5
011. ret_10
012. ret_20
013. ret_60
014. ma5_bias
015. ma10_bias
016. ma20_bias
017. ma60_bias
018. volatility_5
019. volatility_20
020. volatility_60
021. amplitude
022. amplitude_5
023. amplitude_20
024. vol_ratio_5
025. vol_ratio_20
026. corr_price_vol_5
027. corr_price_vol_20
```

### MoneyFlowFeatureGenerator 输出 `moneyflow_factors`（25 列）

- schema 示例文件：`/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/features_data/moneyflow_factors/year=2007/month=01/20070104.parquet`

```text
001. trade_date
002. ts_code
003. amount
004. buy_sm_amount
005. sell_sm_amount
006. buy_md_amount
007. sell_md_amount
008. buy_lg_amount
009. sell_lg_amount
010. buy_elg_amount
011. sell_elg_amount
012. net_mf_amount
013. lg_net_inflow
014. elg_net_inflow
015. main_net_inflow
016. main_net_ratio
017. retail_net_inflow
018. retail_ratio
019. main_retail_diff
020. main_net_5
021. main_net_10
022. main_net_20
023. mf_ma5
024. mf_ma20
025. mf_acceleration
```

### FundamentalFeatureGenerator 输出 `fundamental_factors`（15 列）

- schema 示例文件：`/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/features_data/fundamental_factors/year=2000/month=01/20000104.parquet`

```text
001. trade_date
002. ts_code
003. ann_date
004. end_date
005. roe
006. roa
007. revenue_yoy
008. gross_margin
009. debt_ratio
010. eps
011. bps
012. roe_roa_gap
013. roe_rank
014. roa_rank
015. revenue_yoy_rank
```

### IndustryFeatureGenerator 输出 `industry_factors`（29 列）

- schema 示例文件：`/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/features_data/industry_factors/year=2000/month=01/20000104.parquet`

```text
001. trade_date
002. ts_code
003. L1_industry_name
004. L1_交通运输
005. L1_公用事业
006. L1_商贸零售
007. L1_建筑材料
008. L1_建筑装饰
009. L1_房地产
010. L1_煤炭
011. L1_环保
012. L1_电力设备
013. L1_电子
014. L1_石油石化
015. L1_纺织服饰
016. L1_综合
017. L1_美容护理
018. L1_通信
019. L1_钢铁
020. L1_非银金融
021. L1_食品饮料
022. industry_ret_1
023. industry_ret_5
024. industry_ret_20
025. relative_strength_5
026. relative_strength_20
027. roe_ind_neutral
028. roa_ind_neutral
029. revenue_yoy_ind_neutral
```

### DailyLabelGenerator 输出 `generated_label/daily_labels`（15 列）

- schema 示例文件：`/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/generated_label/daily_labels/year=2000/month=01/20000104.parquet`
- 结果说明：前 3 列为键和统一复权收盘价，后 12 列为 6 个未来收益 label 与 6 个同日截面 rank label。

```text
001. trade_date
002. ts_code
003. adj_close
004. label_1d
005. label_2d
006. label_3d
007. label_5d
008. label_10d
009. label_20d
010. label_rank_1d
011. label_rank_2d
012. label_rank_3d
013. label_rank_5d
014. label_rank_10d
015. label_rank_20d
```

### PriceVolumeFactorsCrossSectionalProcessor 输出 `cross_sectional_processd_data/price_volume_factors`（45 列）

- schema 示例文件：`/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cross_sectional_processd_data/price_volume_factors/year=2000/month=01/20000104.parquet`
- 结果说明：前 27 列与 `price_volume_factors` 原始输出一致，追加 18 个横截面处理列。

```text
001. trade_date
002. ts_code
003. open
004. high
005. low
006. close
007. pct_chg
008. vol
009. amount
010. ret_5
011. ret_10
012. ret_20
013. ret_60
014. ma5_bias
015. ma10_bias
016. ma20_bias
017. ma60_bias
018. volatility_5
019. volatility_20
020. volatility_60
021. amplitude
022. amplitude_5
023. amplitude_20
024. vol_ratio_5
025. vol_ratio_20
026. corr_price_vol_5
027. corr_price_vol_20
028. ret_5_cc_processed
029. ret_10_cc_processed
030. ret_20_cc_processed
031. ret_60_cc_processed
032. ma5_bias_cc_processed
033. ma10_bias_cc_processed
034. ma20_bias_cc_processed
035. ma60_bias_cc_processed
036. volatility_5_cc_processed
037. volatility_20_cc_processed
038. volatility_60_cc_processed
039. amplitude_cc_processed
040. amplitude_5_cc_processed
041. amplitude_20_cc_processed
042. vol_ratio_5_cc_processed
043. vol_ratio_20_cc_processed
044. corr_price_vol_5_cc_processed
045. corr_price_vol_20_cc_processed
```

### MoneyflowFactorsCrossSectionalProcessor 输出 `cross_sectional_processd_data/moneyflow_factors`（38 列）

- schema 示例文件：`/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cross_sectional_processd_data/moneyflow_factors/year=2007/month=01/20070104.parquet`
- 结果说明：前 25 列与 `moneyflow_factors` 原始输出一致，追加 13 个横截面处理列。

```text
001. trade_date
002. ts_code
003. amount
004. buy_sm_amount
005. sell_sm_amount
006. buy_md_amount
007. sell_md_amount
008. buy_lg_amount
009. sell_lg_amount
010. buy_elg_amount
011. sell_elg_amount
012. net_mf_amount
013. lg_net_inflow
014. elg_net_inflow
015. main_net_inflow
016. main_net_ratio
017. retail_net_inflow
018. retail_ratio
019. main_retail_diff
020. main_net_5
021. main_net_10
022. main_net_20
023. mf_ma5
024. mf_ma20
025. mf_acceleration
026. lg_net_inflow_cc_processed
027. elg_net_inflow_cc_processed
028. main_net_inflow_cc_processed
029. main_net_ratio_cc_processed
030. retail_net_inflow_cc_processed
031. retail_ratio_cc_processed
032. main_retail_diff_cc_processed
033. main_net_5_cc_processed
034. main_net_10_cc_processed
035. main_net_20_cc_processed
036. mf_ma5_cc_processed
037. mf_ma20_cc_processed
038. mf_acceleration_cc_processed
```

### FundamentalFactorsCrossSectionalProcessor 输出 `cross_sectional_processd_data/fundamental_factors`（16 列）

- schema 示例文件：`/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cross_sectional_processd_data/fundamental_factors/year=2000/month=01/20000104.parquet`
- 结果说明：前 15 列与 `fundamental_factors` 原始输出一致，追加 1 个横截面处理列。

```text
001. trade_date
002. ts_code
003. ann_date
004. end_date
005. roe
006. roa
007. revenue_yoy
008. gross_margin
009. debt_ratio
010. eps
011. bps
012. roe_roa_gap
013. roe_rank
014. roa_rank
015. revenue_yoy_rank
016. roe_roa_gap_cc_processed
```

### IndustryFactorsCrossSectionalProcessor 输出 `cross_sectional_processd_data/industry_factors`（37 列）

- schema 示例文件：`/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cross_sectional_processd_data/industry_factors/year=2000/month=01/20000104.parquet`
- 结果说明：前 29 列与 `industry_factors` 原始输出一致，追加 8 个横截面处理列。

```text
001. trade_date
002. ts_code
003. L1_industry_name
004. L1_交通运输
005. L1_公用事业
006. L1_商贸零售
007. L1_建筑材料
008. L1_建筑装饰
009. L1_房地产
010. L1_煤炭
011. L1_环保
012. L1_电力设备
013. L1_电子
014. L1_石油石化
015. L1_纺织服饰
016. L1_综合
017. L1_美容护理
018. L1_通信
019. L1_钢铁
020. L1_非银金融
021. L1_食品饮料
022. industry_ret_1
023. industry_ret_5
024. industry_ret_20
025. relative_strength_5
026. relative_strength_20
027. roe_ind_neutral
028. roa_ind_neutral
029. revenue_yoy_ind_neutral
030. industry_ret_1_cc_processed
031. industry_ret_5_cc_processed
032. industry_ret_20_cc_processed
033. relative_strength_5_cc_processed
034. relative_strength_20_cc_processed
035. roe_ind_neutral_cc_processed
036. roa_ind_neutral_cc_processed
037. revenue_yoy_ind_neutral_cc_processed
```

### WideTableDailyBarsCrossSectionalProcessor 输出 `cross_sectional_processd_data/wide_table_daily_bars`（300 列）

- schema 示例文件：`/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cross_sectional_processd_data/wide_table_daily_bars/year=2000/month=01/20000104.parquet`
- 结果说明：前 293 列与 `wide_table_daily_bars` 原始输出一致，追加 7 个横截面处理列。

```text
001. ts_code
002. trade_date
003. open
004. high
005. low
006. close
007. pre_close
008. change
009. pct_chg
010. vol
011. amount
012. no_st_stock
013. st_stock
014. star_st_stock
015. is_suspect
016. adj_factor
017. end_date
018. ann_date
019. roe
020. roa
021. revenue_yoy
022. debt_ratio
023. gross_margin
024. eps
025. bps
026. buy_sm_vol
027. buy_sm_amount
028. sell_sm_vol
029. sell_sm_amount
030. buy_md_vol
031. buy_md_amount
032. sell_md_vol
033. sell_md_amount
034. buy_lg_vol
035. buy_lg_amount
036. sell_lg_vol
037. sell_lg_amount
038. buy_elg_vol
039. buy_elg_amount
040. sell_elg_vol
041. sell_elg_amount
042. net_mf_vol
043. net_mf_amount
044. industry_IT设备
045. industry_专用机械
046. industry_中成药
047. industry_乳制品
048. industry_互联网
049. industry_仓储物流
050. industry_供气供热
051. industry_保险
052. industry_元器件
053. industry_全国地产
054. industry_公共交通
055. industry_公路
056. industry_其他商业
057. industry_其他建材
058. industry_农业综合
059. industry_农用机械
060. industry_农药化肥
061. industry_出版业
062. industry_化学制药
063. industry_化工原料
064. industry_化工机械
065. industry_化纤
066. industry_区域地产
067. industry_医疗保健
068. industry_医药商业
069. industry_半导体
070. industry_商品城
071. industry_商贸代理
072. industry_啤酒
073. industry_园区开发
074. industry_塑料
075. industry_多元金融
076. industry_家居用品
077. industry_家用电器
078. industry_小金属
079. industry_工程机械
080. industry_广告包装
081. industry_建筑工程
082. industry_影视音像
083. industry_房产服务
084. industry_批发业
085. industry_摩托车
086. industry_文教休闲
087. industry_新型电力
088. industry_旅游景点
089. industry_旅游服务
090. industry_日用化工
091. industry_普钢
092. industry_服饰
093. industry_机场
094. industry_机床制造
095. industry_机械基件
096. industry_林业
097. industry_染料涂料
098. industry_橡胶
099. industry_水力发电
100. industry_水务
101. industry_水泥
102. industry_水运
103. industry_汽车整车
104. industry_汽车服务
105. industry_汽车配件
106. industry_渔业
107. industry_港口
108. industry_火力发电
109. industry_焦炭加工
110. industry_煤炭开采
111. industry_特种钢
112. industry_环境保护
113. industry_玻璃
114. industry_生物制药
115. industry_电信运营
116. industry_电器仪表
117. industry_电器连锁
118. industry_电气设备
119. industry_白酒
120. industry_百货
121. industry_石油加工
122. industry_石油开采
123. industry_石油贸易
124. industry_矿物制品
125. industry_种植业
126. industry_空运
127. industry_红黄酒
128. industry_纺织
129. industry_纺织机械
130. industry_综合类
131. industry_航空
132. industry_船舶
133. industry_装修装饰
134. industry_证券
135. industry_超市连锁
136. industry_路桥
137. industry_软件服务
138. industry_软饮料
139. industry_轻工机械
140. industry_运输设备
141. industry_通信设备
142. industry_造纸
143. industry_酒店餐饮
144. industry_钢加工
145. industry_铁路
146. industry_铅锌
147. industry_铜
148. industry_铝
149. industry_银行
150. industry_陶瓷
151. industry_食品
152. industry_饲料
153. industry_黄金
154. L1_交通运输
155. L1_公用事业
156. L1_商贸零售
157. L1_建筑材料
158. L1_建筑装饰
159. L1_房地产
160. L1_煤炭
161. L1_环保
162. L1_电力设备
163. L1_电子
164. L1_石油石化
165. L1_纺织服饰
166. L1_综合
167. L1_美容护理
168. L1_通信
169. L1_钢铁
170. L1_非银金融
171. L1_食品饮料
172. L2_一般零售
173. L2_专业工程
174. L2_专业连锁Ⅱ
175. L2_乘用车
176. L2_元件
177. L2_光伏设备
178. L2_光学光电子
179. L2_养殖业
180. L2_农产品加工
181. L2_冶钢原料
182. L2_包装印刷
183. L2_化学制品
184. L2_化学原料
185. L2_厨卫电器
186. L2_商用车
187. L2_国有大型银行Ⅱ
188. L2_基础建设
189. L2_塑料
190. L2_家电零部件Ⅱ
191. L2_小金属
192. L2_工业金属
193. L2_广告营销
194. L2_影视院线
195. L2_房地产开发
196. L2_普钢
197. L2_橡胶
198. L2_油服工程
199. L2_消费电子
200. L2_渔业
201. L2_游戏Ⅱ
202. L2_煤炭开采
203. L2_照明设备Ⅱ
204. L2_燃气Ⅱ
205. L2_特钢Ⅱ
206. L2_环保设备Ⅱ
207. L2_环境治理
208. L2_玻璃玻纤
209. L2_电力
210. L2_电池
211. L2_电网设备
212. L2_种植业
213. L2_综合Ⅱ
214. L2_自动化设备
215. L2_航天装备Ⅱ
216. L2_航海装备Ⅱ
217. L2_航运港口
218. L2_装修装饰Ⅱ
219. L2_贸易Ⅱ
220. L2_软件开发
221. L2_通信设备
222. L2_通用设备
223. L2_造纸
224. L2_铁路公路
225. L2_非金属材料Ⅱ
226. L2_食品加工
227. L2_饮料乳品
228. L2_黑色家电
229. L3_中间产品及消费品供应链服务
230. L3_产业地产
231. L3_人工景区
232. L3_仓储物流
233. L3_住宅开发
234. L3_保险Ⅲ
235. L3_公交
236. L3_公路货运
237. L3_其他农产品加工
238. L3_其他塑料制品
239. L3_其他建材
240. L3_其他石化
241. L3_制冷空调设备
242. L3_动物保健Ⅲ
243. L3_印制电路板
244. L3_印刷包装机械
245. L3_原材料供应链服务
246. L3_商业地产
247. L3_商业物业经营
248. L3_复合肥
249. L3_多业态零售
250. L3_工程机械器件
251. L3_快递
252. L3_房屋建设Ⅲ
253. L3_数字芯片设计
254. L3_旅游综合
255. L3_有机硅
256. L3_机床工具
257. L3_果蔬加工
258. L3_楼宇设备
259. L3_模拟芯片设计
260. L3_汽车经销商
261. L3_消费电子零部件及组装
262. L3_涤纶
263. L3_港口
264. L3_游戏Ⅲ
265. L3_激光设备
266. L3_煤化工
267. L3_燃气Ⅲ
268. L3_物业管理
269. L3_电商服务
270. L3_电机Ⅲ
271. L3_畜禽饲料
272. L3_百货
273. L3_种子
274. L3_粮油加工
275. L3_纺织服装设备
276. L3_能源及重型设备
277. L3_自然景区
278. L3_航空运输
279. L3_航运
280. L3_装修装饰Ⅲ
281. L3_证券Ⅲ
282. L3_资产管理
283. L3_超市
284. L3_跨境物流
285. L3_跨境电商
286. L3_酒店
287. L3_金属制品
288. L3_金融控股
289. L3_钛白粉
290. L3_铁路运输
291. L3_铜
292. L3_食品及饲料添加剂
293. L3_高速公路
294. roe_cc_processed
295. roa_cc_processed
296. revenue_yoy_cc_processed
297. debt_ratio_cc_processed
298. gross_margin_cc_processed
299. eps_cc_processed
300. bps_cc_processed
```

> 横截面处理列统一命名为 `{factor}_cc_processed`。每个值均在同一个 `trade_date` 的股票截面内，先按 1%/99% 分位缩尾，再做 z-score 标准化；原始列仍保留，便于对比处理前后差异。

## 8. 推荐执行顺序

如果从原始数据开始重建全部清洗表、宽表、因子与 label，建议按以下顺序执行：

```bash
export PYTHONPATH="/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src"

# 1) 清洗基础数据
python -m utils.dataset_cleaner.daily_bars_cleaner
python -m utils.dataset_cleaner.adj_factors_cleaner
python -m utils.dataset_cleaner.fundamentals_cleaner
python -m utils.dataset_cleaner.industry_cleaner
python -m utils.dataset_cleaner.moneyflow_cleaner

# 2) 生成中间加工表
python -m utils.dataset_processor.namechange_st_processor
python -m utils.dataset_processor.suspend_d_processor
python -m utils.dataset_processor.industry_onehot_processor

# 3) 构建日频大宽表（可选，供直接建模或检查使用）
python -m utils.dataset_processor.wide_table_daily_bars_builder

# 4) 生成分主题因子
python -m utils.features_generator.price_volume_feature_generator
python -m utils.features_generator.moneyflow_feature_generator
python -m utils.features_generator.fundamental_feature_generator
python -m utils.features_generator.industry_feature_generator

# 5) 生成监督学习 label（依赖 wide_table_daily_bars）
python -m utils.label_generator.daily_label_generator

# 6) 横截面处理（可选，供截面模型或标准化建模输入使用）
python -m utils.cross_sectional_processor.price_volume_factors_cross_sectional_processor
python -m utils.cross_sectional_processor.moneyflow_factors_cross_sectional_processor
python -m utils.cross_sectional_processor.fundamental_factors_cross_sectional_processor
python -m utils.cross_sectional_processor.industry_factors_cross_sectional_processor
python -m utils.cross_sectional_processor.wide_table_daily_bars_cross_sectional_processor
```

指定日期区间时，大部分 generator / builder 支持：

```bash
--start-date 20240101 --end-date 20240131
```

---

## 9. 常见注意事项

1. **避免未来函数**：财务相关合并和 `FundamentalFeatureGenerator` 都基于 `ann_date <= trade_date`，不要改成按 `end_date` 直接合并。
2. **历史/未来窗口读取**：价格量因子需要最多 60 个历史交易日；资金流因子需要最多 20 个历史资金流交易日；`DailyLabelGenerator` 为计算未来收益会读取最多 20 个未来交易日。指定 `start_date` / `end_date` 时，因子实际读取范围可能向前扩展，label 实际读取范围可能向后扩展，但输出仍限制在指定日期区间。
3. **缺失值含义**：
   - rolling 类因子前 N-1 个有效历史通常为 NaN，是正常现象。
   - 尚未公告财报的交易日，财务因子为 NaN，是 point-in-time 约束的正常结果。
   - 行业缺失会导致行业收益/中性化因子为 NaN，行业 one-hot 全 0。
   - label 末尾日期未来窗口不足、`close/adj_factor` 缺失或无法构造有效 `adj_close` 时，`label_*d` 与 `label_rank_*d` 会保留为 NaN。
4. **主键约束**：大多数日频输出以 `trade_date + ts_code` 唯一；行业 one-hot 以 `ts_code` 唯一。
5. **日志位置**：CLI 默认会写入 `log/`、`log/features_generate/`、`log/label_generate/` 或 `log/cross_sectional_process/` 下对应日志文件。建议排查数据覆盖率时优先查看日志中的 match/missing/nan 统计。
6. **金额单位**：`MoneyFlowFeatureGenerator` 默认用 `amount_unit_scale=0.1` 将 daily bars 的千元成交额转换为万元；若上游已统一单位，应显式传 `--amount-unit-scale 1.0`。
7. **分区格式**：多数日频输出均使用 `year=YYYY/month=MM/YYYYMMDD.parquet`，便于按日期增量处理和下游扫描。
8. **横截面处理语义**：`*_cc_processed` 只表示同日截面内的缩尾标准化结果，不改变时间序列窗口逻辑，也不会覆盖原始因子列。
