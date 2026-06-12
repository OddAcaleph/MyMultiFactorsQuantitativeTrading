# A 股数据路径样例说明

数据根目录：`/opt/tiger/qyd/quant_llm/A_stocks_all_data`

本文档按你的要求，逐个说明这些路径下：

1. 有什么样子的文件
2. 代表性文件的字段有哪些
3. 给出 1 到 3 行代表性数据

> 说明：除 `industry` 目录外，其它目录各选了 1 个代表性文件；`industry` 目录下把多个 parquet 文件都列出来了。

---

## 1. `daily_bars`

### 路径下的文件样式

- 目录结构是按年月分区的 parquet 文件：`year=YYYY/month=MM/*.parquet`
- 代表性文件：`/opt/tiger/qyd/quant_llm/A_stocks_all_data/daily_bars/year=2000/month=01/20000104.parquet`

### 字段列表

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

### 代表性数据（前 3 行）

```text
     ts_code trade_date  open  high  low  close  pre_close  change  pct_chg      vol      amount
0  000003.SZ   20000104  5.48  5.85  5.4   5.74       5.47    0.27     4.94  19073.0  10787.1201
1  000005.SZ   20000104  6.10  6.27  6.0   6.24       6.04    0.20     3.31   8365.0   5132.4196
2  000007.SZ   20000104  8.20  8.28  7.88  8.28       8.00    0.28     3.50   3368.0   2739.0356
```

---

## 2. `adj_factors`

### 路径下的文件样式

- 目录下是聚合后的单个 parquet 文件
- 代表性文件：`/opt/tiger/qyd/quant_llm/A_stocks_all_data/adj_factors/adj_factors.parquet`

### 字段列表

- `ts_code`
- `trade_date`
- `adj_factor`

### 代表性数据（前 3 行）

```text
     ts_code trade_date  adj_factor
0  000001.SZ   20100126      35.906
1  000002.SZ   20100126     110.804
2  000004.SZ   20100126       4.064
```

---

## 3. `adj_factors_raw`

### 路径下的文件样式

- 目录结构是按年月分区的 parquet 文件：`year=YYYY/month=MM/*.parquet`
- 代表性文件：`/opt/tiger/qyd/quant_llm/A_stocks_all_data/adj_factors_raw/year=2000/month=01/20000104.parquet`

### 字段列表

- `ts_code`
- `trade_date`
- `adj_factor`

### 代表性数据（前 3 行）

```text
     ts_code trade_date  adj_factor
0  000003.SZ   20000104       3.833
1  000005.SZ   20000104       4.433
2  000013.SZ   20000104       1.939
```

---

## 4. `fundamentals`

### 路径下的文件样式

- 目录下是聚合后的单个 parquet 文件
- 代表性文件：`/opt/tiger/qyd/quant_llm/A_stocks_all_data/fundamentals/fundamentals.parquet`

### 字段列表

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

### 代表性数据（前 3 行）

```text
     ts_code  end_date  ann_date      roe      roa  revenue_yoy  debt_ratio  gross_margin     eps     bps
0  600734.SH  20260331  20260429  -4.9649  -0.6028      12.9689     83.4345       11.5481 -0.0055  0.1052
1  600734.SH  20251231  20260429 -44.7979 -10.0170     -71.4472     77.7290       24.5547 -0.0669  0.1158
2  600734.SH  20250930  20251028 -24.7285  -7.8560     111.9670     72.3975        1.9384 -0.0402  0.1425
```

---

## 5. `fundamentals_raw`

### 路径下的文件样式

- 目录下是按股票存储的 parquet 文件：`000001.SZ.parquet`、`600000.SH.parquet` 这类形式
- 代表性文件：`/opt/tiger/qyd/quant_llm/A_stocks_all_data/fundamentals_raw/000001.SZ.parquet`

### 字段列表

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

### 代表性数据（前 3 行）

```text
     ts_code  end_date  ann_date     roe   roa  revenue_yoy  debt_ratio gross_margin   eps      bps
0  000001.SZ  20260331  20260425  2.6520  None       4.6516     90.9830         None  0.67  23.9145
1  000001.SZ  20251231  20260321  8.1514  None     -10.3978     90.6985         None  2.07  23.2522
2  000001.SZ  20250930  20251025  7.5711  None      -9.7811     91.0187         None  1.87  23.0846
```

---

## 对于这部分数据做以下几个数据清理动作：



1. 首先检查是否有 TS code 和 trade date 的缺失

2. 如果出现了 TS code 和 trade date 这个主键的相同的数据行，那就去重，只保留最新的一行

你需要在/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/src/utils/dataset_cleaner路径下生成一个专门用于这部分数据检查的工具类。该类应具备明确的日志功能，能够告知我是否存在唯一性或缺失值的问题。

日志保存到/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/log

禁止修改原始数据，清洗后的数据保存到/opt/tiger/qyd/qlib_quant_scripts/my_qlib_lab/data/cleaned_data的对应路径下
---

## 7. `moneyflow_raw`

### 路径下的文件样式

- 目录结构是按年月分区的 parquet 文件：`year=YYYY/month=MM/*.parquet`
- 代表性文件：`/opt/tiger/qyd/quant_llm/A_stocks_all_data/moneyflow_raw/year=2007/month=01/20070104.parquet`

### 字段列表

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

### 代表性数据（前 3 行）

```text
     ts_code trade_date  buy_sm_vol  buy_sm_amount  sell_sm_vol  sell_sm_amount  buy_md_vol  buy_md_amount  sell_md_vol  sell_md_amount  buy_lg_vol  buy_lg_amount  sell_lg_vol  sell_lg_amount  buy_elg_vol  buy_elg_amount  sell_elg_vol  sell_elg_amount  net_mf_vol  net_mf_amount
0  600000.SH   20070104     34736.0        7730.74        29985         6742.49     77016.0       17175.68      68638.0        15418.10      145180       32523.42     154662.0        34661.77       251840        56625.75      255487.0         57233.23     26857.0        6617.10
1  600001.SH   20070104    254907.0       11810.32       207085         9633.40    325088.0       15079.94     231673.0        10755.21      227439       10574.51     301098.0        13980.58        59456         2765.37      127033.0          5860.94     38247.0        1899.06
2  600004.SH   20070104     66468.0        5180.62        89059         6965.49     93710.0        7316.08      97758.0         7640.14       61698        4830.09      45299.0         3537.83        17289         1368.31        7048.0           551.64     21673.0        1715.58
```

---

## 8. `index_data`

### 路径下的文件样式

- 目录下是按指数代码命名的 parquet 文件，例如：`000001_SH.parquet`、`000300_SH.parquet`、`000905_SH.parquet`
- 代表性文件：`/opt/tiger/qyd/quant_llm/A_stocks_all_data/index_data/000001_SH.parquet`

### 字段列表

- `ts_code`
- `trade_date`
- `close`
- `open`
- `high`
- `low`
- `pre_close`
- `change`
- `pct_chg`
- `vol`
- `amount`

### 代表性数据（前 3 行）

```text
     ts_code trade_date   close    open     high      low  pre_close  change  pct_chg       vol       amount
0  000001.SH   19930702  986.98   995.7  1002.87   986.98     997.53  -10.55  -1.0576  264050.0   372539.438
1  000001.SH   19930701  997.53  1000.8  1009.89   980.05    1007.05   -9.52  -0.9453  399043.0   636201.668
2  000001.SH   19930630 1007.05 1037.89  1043.05  1003.86    1028.44  -21.39  -2.0798  580651.0  1099510.219
```

---

## 9. `industry`

`industry` 目录下有多个不同的 parquet 文件，这里分别列出。

### 9.1 `industry.parquet`

#### 路径下的文件样式

- 文件路径：`/opt/tiger/qyd/quant_llm/A_stocks_all_data/industry/industry.parquet`

#### 字段列表

- `ts_code`
- `industry`
- `name`
- `area`
- `L1_index_code`
- `L2_index_code`
- `L3_index_code`
- `L1_industry_name`
- `L2_industry_name`
- `L3_industry_name`

#### 代表性数据（前 3 行）

```text
     ts_code industry   name area L1_index_code L2_index_code L3_index_code L1_industry_name L2_industry_name L3_industry_name
0  000001.SZ       银行   平安银行   深圳          None          None          None             None             None             None
1  000002.SZ     全国地产    万科Ａ   深圳     801180.SI     801181.SI     851811.SI              房地产            房地产开发             住宅开发
2  000004.SZ     软件服务  *ST国华   深圳          None     801104.SI          None             None             软件开发             None
```

### 9.2 `stock_basic.parquet`

#### 路径下的文件样式

- 文件路径：`/opt/tiger/qyd/quant_llm/A_stocks_all_data/industry/stock_basic.parquet`

#### 字段列表

- `ts_code`
- `symbol`
- `name`
- `area`
- `industry`
- `fullname`
- `enname`
- `cnspell`
- `market`
- `exchange`
- `curr_type`
- `list_status`
- `list_date`
- `delist_date`
- `is_hs`
- `act_name`
- `act_ent_type`

#### 代表性数据（前 3 行）

```text
     ts_code  symbol   name area industry         fullname                                                 enname cnspell market exchange curr_type list_status list_date delist_date is_hs act_name act_ent_type
0  000001.SZ  000001   平安银行   深圳       银行       平安银行股份有限公司                                 Ping An Bank Co., Ltd.    PAYH     主板     SZSE       CNY           L  19910403        None     S  无实际控制人           其他
1  000002.SZ  000002    万科Ａ   深圳     全国地产       万科企业股份有限公司                                   China Vanke Co.,Ltd.     WKA     主板     SZSE       CNY           L  19910129        None     S  无实际控制人           其他
2  000004.SZ  000004  *ST国华   深圳     软件服务   深圳国华网安科技股份有限公司  Shenzhen GuoHua Network Security Technology Co., Ltd.   *STGH     主板     SZSE       CNY           L  19901201        None     N     李映彤         民营企业
```

### 9.3 `sw_industry.parquet`

#### 路径下的文件样式

- 文件路径：`/opt/tiger/qyd/quant_llm/A_stocks_all_data/industry/sw_industry.parquet`

#### 字段列表

- `index_code`
- `ts_code`
- `in_date`
- `out_date`
- `is_new`
- `industry`
- `industry_code`

#### 代表性数据（前 3 行）

```text
  index_code    ts_code   in_date  out_date is_new industry industry_code
0  801120.SI  000019.SZ  19921012  20190705      N     食品饮料     801120.SI
1  801120.SI  000019.SZ  19921012  20190705      N     食品饮料     801120.SI
2  801120.SI  000019.SZ  19921012  20190705      N     食品饮料     801120.SI
```

### 9.4 `sw_industry_all.parquet`

#### 路径下的文件样式

- 文件路径：`/opt/tiger/qyd/quant_llm/A_stocks_all_data/industry/sw_industry_all.parquet`

#### 字段列表

- `index_code`
- `ts_code`
- `in_date`
- `out_date`
- `is_new`
- `industry_level`
- `industry_name`

#### 代表性数据（前 3 行）

```text
  index_code    ts_code   in_date  out_date is_new industry_level industry_name
0  801010.SI  000019.SZ  20211213      None      Y             L1          农林牧渔
1  801010.SI  000034.SZ  20070703  20090601      N             L1          农林牧渔
2  801010.SI  000034.SZ  20150701  20170630      N             L1          农林牧渔
```

### 9.5 `sw_industry_pivoted.parquet`

#### 路径下的文件样式

- 文件路径：`/opt/tiger/qyd/quant_llm/A_stocks_all_data/industry/sw_industry_pivoted.parquet`

#### 字段列表

- `ts_code`
- `L1_index_code`
- `L2_index_code`
- `L3_index_code`
- `L1_industry_name`
- `L2_industry_name`
- `L3_industry_name`

#### 代表性数据（前 3 行）

```text
     ts_code L1_index_code L2_index_code L3_index_code L1_industry_name L2_industry_name L3_industry_name
0  000001.SZ     801780.SI     801783.SI     857831.SI               银行           股份制银行Ⅱ           股份制银行Ⅲ
1  000002.SZ     801180.SI     801181.SI     851811.SI              房地产            房地产开发             住宅开发
2  000004.SZ     801750.SI     801104.SI     851042.SI              计算机             软件开发           横向通用软件
```
