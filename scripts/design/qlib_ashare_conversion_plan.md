# A 股原始数据转换为 Qlib 可用格式方案

## 1. 目标

基于 `/opt/tiger/qyd/quant_llm/A_stocks_all_data` 下已有的 A 股数据，编写一个统一转换脚本，将多源 parquet 数据整理成 **Qlib 可直接消费** 的数据集。

本方案的目标不是只做单一的 OHLCV 转换，而是同时兼顾后续量化研究常用的数据层：

1. **日频行情层**：用于构建基础价格、收益率、量价因子。
2. **复权层**：保证价格序列可用于回测与特征计算。
3. **日频扩展特征层**：如 moneyflow。
4. **低频基本面层**：按公告日对齐，避免未来函数。
5. **行业层**：用于行业中性化或行业哑变量编码。
6. **指数层**：用于基准收益、市场状态特征或选股池约束。

最终产物优先选择 **Qlib 标准二进制目录（bin provider）**，因为它兼容性最好、性能最高，也方便后续直接接入 `qlib.init(provider_uri=...)`。

---

## 2. 已确认的数据现状

从当前样本检查结果看，数据已经具备搭建 Qlib 数据集的核心条件：

### 2.1 日线行情 `daily_bars`

样例文件：`/opt/tiger/qyd/quant_llm/A_stocks_all_data/daily_bars/year=2026/month=01/20260113.parquet`

```
ts_code trade_date   open   high    low  close  pre_close  change  pct_chg      vol      amount
0   000003.SZ   20000104   5.48   5.85   5.40   5.74       5.47    0.27     4.94  19073.0  10787.1201
1   000005.SZ   20000104   6.10   6.27   6.00   6.24       6.04    0.20     3.31   8365.0   5132.4196
2   000007.SZ   20000104   8.20   8.28   7.88   8.28       8.00    0.28     3.50   3368.0   2739.0356
```

字段包括：

- `ts_code`
- `trade_date`
- `open`
- `high`
- `low`
- `close`
- `pre_close`
- `change`
- `pct_chg`
- `vol`
- `amount`

这部分可以作为 Qlib 的主行情输入。

### 2.2 复权因子 `adj_factors`

样例文件：`/opt/tiger/qyd/quant_llm/A_stocks_all_data/adj_factors/adj_factors.parquet`

字段包括：

- `ts_code`
- `trade_date`
- `adj_factor`

可用于生成前复权/后复权价格，以及 Qlib 常用的 `factor` 字段。

### 2.3 基本面 `fundamentals`

样例文件：`/opt/tiger/qyd/quant_llm/A_stocks_all_data/fundamentals/fundamentals.parquet`

字段包括：

- `ts_code`
- `end_date`
- `ann_date`
- `roe`
- `roa`
- `revenue_yoy`
- `debt_ratio`
- `gross_margin`
- `eps`
- `bps`

这部分适合做公告日对齐后的低频特征扩展。

### 2.4 资金流 `moneyflow`

样例文件：`/opt/tiger/qyd/quant_llm/A_stocks_all_data/moneyflow/moneyflow.parquet`

字段包括：

- `ts_code`
- `trade_date`
- `buy_sm_vol`
- `buy_sm_amount`
- `sell_sm_vol`
- `sell_sm_amount`
- `buy_md_vol`
- `buy_md_amount`
- `sell_md_vol`
- `sell_md_amount`
- `buy_lg_vol`
- `buy_lg_amount`
- `sell_lg_vol`
- `sell_lg_amount`
- `buy_elg_vol`
- `buy_elg_amount`
- `sell_elg_vol`
- `sell_elg_amount`
- `net_mf_vol`
- `net_mf_amount`

这部分可直接扩展成日频 alpha 特征。

### 2.5 行业 `industry`

已确认存在：

- `industry.parquet`
- `stock_basic.parquet`
- `sw_industry.parquet`
- `sw_industry_all.parquet`
- `sw_industry_pivoted.parquet`

其中：

- `stock_basic.parquet` 提供上市/退市等静态信息；
- `sw_industry_all.parquet` 提供更完整的行业进出区间；
- `sw_industry_pivoted.parquet` 提供 L1/L2/L3 行业编码与名称。

### 2.6 指数 `index_data`

已确认存在：

- `000001_SH.parquet`
- `000300_SH.parquet`
- `000905_SH.parquet`

可作为上证综指、沪深 300、中证 500 等基准或市场状态特征来源。

---

## 3. Qlib 目标输出形式

建议目标输出为两层：

### 3.1 中间层：标准化宽表/长表 parquet（便于调试）

先生成一份中间层目录，例如：

```text
./qlib_stage/
  calendar/day.txt
  instruments/all.txt
  features_raw/
    sh600000.parquet
    sz000001.parquet
    ...
```

这层用于检查 merge 逻辑是否正确。

### 3.2 最终层：Qlib bin provider

最终输出：

```text
./qlib_cn_data/
  calendars/day.txt
  instruments/all.txt
  features/sh600000/*.bin
  features/sz000001/*.bin
  ...
```

后续可直接：

```python
import qlib
qlib.init(provider_uri="./qlib_cn_data", region="cn")
```

---

## 4. 总体实现思路

建议采用 **两阶段转换**：

### 阶段 A：统一标准化

把多源 parquet 统一整理成“按股票、按交易日”的日频主表，并把低频数据转换成按日对齐后的扩展特征。

### 阶段 B：落地为 Qlib 格式

把标准化后的结果输出成：

1. `calendar`
2. `instruments`
3. `per-symbol feature files`

然后调用 Qlib 官方 `dump_bin` 流程，或者在脚本中直接复用同等逻辑生成 `.bin`。

为了降低出错率，**第一版建议直接复用 Qlib 的 dump_bin 入口**，不要一开始就自己手写底层 bin writer。

---

## 5. 字段映射设计

## 5.1 证券代码规范化

原始数据代码形如：

- `000001.SZ`
- `600000.SH`

Qlib 侧建议统一为：

- `sz000001`
- `sh600000`

设计一个统一函数：

```python
normalize_symbol("000001.SZ") -> "sz000001"
normalize_symbol("600000.SH") -> "sh600000"
```

原因：

1. 更贴近 Qlib CN 社区习惯；
2. 文件名天然稳定；
3. 后续 instruments 与 feature 路径统一。

---

## 5.2 主行情字段映射

建议把 `daily_bars` 转成如下主字段：

| 原字段 | 目标字段 | 说明 |
|---|---|---|
| `trade_date` | `date` | 转成 `YYYY-MM-DD` |
| `ts_code` | `symbol` | 转为 `sh600000/sz000001` |
| `open` | `open` | 原始价格 |
| `high` | `high` | 原始价格 |
| `low` | `low` | 原始价格 |
| `close` | `close` | 原始价格 |
| `pre_close` | `pre_close` | 原始价格 |
| `vol` | `volume` | 建议统一成“股”或“手”，但需全流程固定 |
| `amount` | `amount` | 成交额 |
| `change` | `change` | 原始涨跌额 |
| `pct_chg` | `pct_chg` | 原始涨跌幅 |

另外扩展生成：

- `vwap = amount / volume`（注意量纲）
- `paused = 1/0`（若后续接入 `suspend_d`）
- `factor`（由复权因子生成）

---

## 5.3 复权设计

Qlib 里最关键的是让价格与因子字段自洽。

建议第一版采用：

1. 保留原始未复权的 `open/high/low/close/pre_close`
2. 额外加入 `factor`
3. 再衍生一套前复权价字段，便于研究与排错：
   - `open_adj`
   - `high_adj`
   - `low_adj`
   - `close_adj`
   - `pre_close_adj`

推荐公式：

```text
adj_price = raw_price * adj_factor / latest_adj_factor_of_symbol
```

这样生成的是“相对最新日归一化”的前复权序列，适合研究使用。

同时保留：

```text
factor = adj_factor / latest_adj_factor_of_symbol
```

注意事项：

- 同一股票必须按时间升序处理；
- 若某日缺少 `adj_factor`，应优先向前/向后排查，不要直接静默丢弃；
- 若上市早期部分日期没有复权因子，需要记录到日志。

---

## 5.4 基本面按公告日对齐

`fundamentals` 不能直接按 `end_date` 合并，否则会产生未来函数。

正确做法：

1. 使用 `ann_date` 作为生效日期；
2. 对每个股票，将每一条财报记录从 `ann_date` 起向后生效；
3. 在下一个财报 `ann_date` 之前，对日频数据 forward fill；
4. 若 `ann_date` 缺失，则该条财报默认不进入第一版主流程，而是记录异常。

最终可生成日频基本面字段：

- `roe_ttm_like`（若当前数据仅单点值，则先按原值日对齐）
- `roa`
- `revenue_yoy`
- `debt_ratio`
- `gross_margin`
- `eps`
- `bps`

第一版不强行做复杂 TTM 重算，先保证 **无未来函数 + 可稳定落地**。

---

## 5.5 资金流映射

`moneyflow` 可直接按 `ts_code + trade_date` 左连接到主表。

建议保留原字段，并新增几个常用比率特征：

- `net_mf_amount_ratio = net_mf_amount / amount`
- `buy_elg_amount_ratio = buy_elg_amount / amount`
- `sell_elg_amount_ratio = sell_elg_amount / amount`
- `net_mf_vol_ratio = net_mf_vol / volume`

这样后续在 Qlib 表达式里更好直接使用。

---

## 5.6 行业映射

行业数据建议分两层：

### 静态层

来自 `sw_industry_pivoted.parquet`：

- `sw_l1_code`
- `sw_l2_code`
- `sw_l3_code`
- `sw_l1_name`
- `sw_l2_name`
- `sw_l3_name`

### 动态层

来自 `sw_industry_all.parquet`：

- 使用 `in_date/out_date` 还原区间归属；
- 生成每个交易日对应的行业归属；
- 若某股票行业发生变更，应按区间切换，而不是用静态表覆盖全历史。

第一版建议：

1. 主流程先落地 **L1 行业日频字段**；
2. L2/L3 可选；
3. 若某天无行业映射，允许为空，但要输出缺失统计。

---

## 5.7 指数数据处理

`index_data` 不强制并入每只股票的主表，但建议在脚本中顺手标准化输出，供后续模型直接使用：

- `sh000001`
- `sh000300`
- `sh000905`

可以有两种方式：

1. 作为额外 instrument 一起进入 Qlib；
2. 单独存成 `benchmarks/`，供训练或回测阶段引用。

第一版更推荐 **单独输出**，避免股票 universe 与指数 instrument 混在一起。

---

## 6. 交易日历与股票池设计

## 6.1 交易日历 `calendar/day.txt`

优先从 `daily_bars` 汇总全部唯一 `trade_date`，排序后写出。

补充原则：

- 如个别交易日股票数据缺失，但指数数据存在，也应纳入交易日历；
- 输出格式统一为 `YYYY-MM-DD`。

---

## 6.2 股票池 `instruments/all.txt`

建议综合以下来源确定股票生命周期：

1. `stock_basic.list_date`
2. `stock_basic.delist_date`
3. `daily_bars` 中实际出现的最早/最晚交易日

生成规则：

```text
symbol\tstart_date\tend_date
```

例如：

```text
sz000001	1991-04-03	2099-12-31
```

建议：

- 若 `delist_date` 为空，则用全局最大交易日或 `2099-12-31`；
- 若 `list_date` 晚于行情首日，优先使用 `list_date`；
- 若静态信息缺失，则退回用行情首尾日期。

---

## 7. 数据清洗规则

脚本里建议显式实现以下清洗规则，而不是依赖人工判断：

### 7.1 主键去重

- `daily_bars`：按 `ts_code + trade_date` 去重；
- `adj_factors`：按 `ts_code + trade_date` 去重；
- `moneyflow`：按 `ts_code + trade_date` 去重；
- `fundamentals`：按 `ts_code + ann_date + end_date` 去重；
- `industry`：按区间字段去重。

### 7.2 日期规范化

- `20260113 -> 2026-01-13`
- `None/空字符串` 统一为缺失值；
- 所有日期字段最终都转成 pandas datetime。

### 7.3 数值异常

- `volume <= 0` 时，`vwap` 不计算；
- 价格字段小于等于 0 时记为异常；
- `adj_factor <= 0` 记为异常；
- 极端值先不裁剪，但输出分布统计。

### 7.4 停牌处理

若后续接入 `suspend_d`：

- 为停牌日生成 `paused=1`；
- 若某停牌日没有行情记录，可选择是否补齐空行；
- 第一版可以先不补齐停牌空日，只打标已有记录。

### 7.5 ST/名称变更

利用：

- `namechange`
- `stock_basic.name`

可扩展生成：

- `is_st`
- `name`

但这部分不作为第一版阻塞项。

---

## 8. 脚本设计建议

建议最终落地一个主脚本：

```text
./convert_ashare_to_qlib.py
```

### 8.1 推荐命令行参数

```bash
python convert_ashare_to_qlib.py \
  --source-dir /opt/tiger/qyd/quant_llm/A_stocks_all_data \
  --stage-dir ./qlib_stage \
  --qlib-dir ./qlib_cn_data \
  --region cn \
  --freq day \
  --universe all \
  --use-adjusted-price true \
  --include-moneyflow true \
  --include-fundamentals true \
  --include-industry true \
  --dump-bin true
```

### 8.2 模块拆分建议

建议即使只有一个脚本，也按函数模块组织：

1. `load_daily_bars()`
2. `load_adj_factors()`
3. `load_moneyflow()`
4. `load_fundamentals()`
5. `load_industry()`
6. `build_calendar()`
7. `build_instruments()`
8. `merge_daily_features()`
9. `align_fundamentals_by_ann_date()`
10. `write_stage_files()`
11. `dump_to_qlib_bin()`
12. `validate_output()`

---

## 9. 推荐实现流程

### Step 1：读取并规范化所有源数据

- 统一列名
- 统一日期格式
- 统一 symbol 格式

### Step 2：建立全市场交易日历

- 汇总 `daily_bars.trade_date`
- 补充指数交易日
- 输出 `calendar/day.txt`

### Step 3：建立 instruments 生命周期

- 用 `stock_basic` + 实际行情首尾日
- 输出 `instruments/all.txt`

### Step 4：生成日频主表

- 以 `daily_bars` 为 base
- 左连接 `adj_factors`
- 左连接 `moneyflow`
- 计算 `factor/vwap/adj_price`

### Step 5：把基本面对齐到日频

- 基于 `ann_date` 生效
- forward fill 到下一次公告前
- 合并到主表

### Step 6：把行业映射到日频

- 优先落地 SW L1
- 若有区间切换则按区间展开

### Step 7：按股票拆分输出 stage 文件

- 每只股票一个 parquet/csv
- 同时写 calendar/instruments

### Step 8：调用 Qlib dump_bin 生成最终数据

- 若环境中存在 Qlib，则直接调用；
- 否则先保留 stage 数据，并提示用户执行 Qlib dump。

### Step 9：做转换后校验

至少校验：

1. 股票数是否匹配
2. 交易日数量是否匹配
3. 每只股票是否按日期升序
4. 是否存在重复日期
5. `close_adj` 是否存在大面积缺失
6. `factor` 是否存在非正值

---

## 10. 性能与实现策略

由于数据量较大，建议脚本不要一次性把全市场所有年份全读入内存。

推荐策略：

### 方案 A：按股票分组流式处理（优先推荐）

适合：

- `fundamentals_raw`
- `moneyflow_raw`
- `industry` 静态表

优点：

- 内存占用低；
- 易于按 symbol 输出。

### 方案 B：按年份读取、再按股票切分

适合：

- `daily_bars/year=YYYY/...`
- `adj_factors_raw/year=YYYY/...`

建议组合策略：

1. `daily_bars` 和 `adj_factors_raw` 按年扫描；
2. 中间结果累积到每只股票；
3. 单股票落盘后释放内存。

如果环境允许，也可以直接使用 `pyarrow.dataset` 做懒加载扫描。

---

## 11. 第一版范围建议（MVP）

为了尽快得到可用的 Qlib 数据，第一版建议范围如下：

### 必做

1. `daily_bars` -> 主行情
2. `adj_factors` -> `factor` + 前复权价
3. `stock_basic` -> instruments 生命周期
4. `moneyflow` -> 日频扩展特征
5. `calendar/instruments/features` 输出
6. `dump_bin` 生成 Qlib 最终目录

### 第二优先级

1. `fundamentals` 按公告日对齐
2. `sw_industry_all` 行业日频映射

### 可后续再做

1. `suspend_d` 停牌空日补齐
2. `namechange` / `is_st`
3. 指数成分股权重与动态股票池
4. 更复杂的 TTM/YoY 财务重建

---

## 12. 验收标准

脚本完成后，验收建议如下：

### 功能验收

1. 能从给定 `source-dir` 成功生成 `qlib_cn_data`
2. `qlib.init(provider_uri=..., region="cn")` 可正常初始化
3. 能成功读取样例股票的 `$close/$volume/factor`
4. 能读取自定义字段，如 `net_mf_amount`

### 数据验收

1. `calendar/day.txt` 日期连续且有序
2. `instruments/all.txt` 股票数量合理
3. 样例股票复权价格序列平滑，无明显除权跳点
4. 基本面字段在公告日之后才出现
5. 行业字段在历史切换点前后正确变化

### 性能验收

1. 全量转换可在可接受时间内完成
2. 不因一次性读全量数据导致 OOM
3. 支持断点日志或进度输出

---

## 13. 风险点

1. **量纲风险**：`vol` 和 `amount` 的单位需要核对，决定 `vwap` 是否需要乘除 100。
2. **复权对齐风险**：若 `adj_factor` 与 `daily_bars` 某些日期对不上，会导致复权价缺失。
3. **基本面未来函数风险**：若误用 `end_date` 对齐，会污染训练集。
4. **行业历史漂移风险**：若只用静态行业表，会覆盖历史行业变更。
5. **Qlib 字段兼容风险**：自定义字段虽然可存，但最终使用时需确认 loader / expression 是否按预期读取。

---

## 14. 最终建议

我建议下一步实现时采用以下落地策略：

1. **先做 MVP 脚本**：行情 + 复权 + moneyflow + instruments + calendar + dump_bin。
2. **保留 stage 层**：先输出 parquet/csv 中间层，方便核查数据质量。
3. **基本面与行业做成可选开关**：避免第一版就被低频对齐复杂度拖慢。
4. **所有转换写日志和统计摘要**：缺失率、重复率、异常值数量必须可见。

如果按这个方案实现，后续得到的数据将能同时支持：

- Qlib 数据加载
- Alpha 因子研究
- 回测
- 行业中性化
- 多源特征扩展

---

## 15. 下一步实现建议

如果你认可这个方案，下一步我会直接在当前工作目录实现：

1. `convert_ashare_to_qlib.py`
2. 必要的参数入口
3. stage 输出
4. Qlib bin 输出
5. 基础校验逻辑

第一版会优先保证：**能跑通、能落地、能被 Qlib 读取**。
