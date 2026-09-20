# Risk Model 开发方案

**项目**：MyMultiFactorsQuantitativeTrading  
**阶段**：Alpha Model V1 → Risk Model V1  
**制定日期**：2026-08-07  
**目标**：为现有 XGBoost Alpha 模型建立可用于 Portfolio Optimizer 的股票风险模型

---

## 1. 项目现状与开发结论

当前项目已经具备完整的 A 股多因子投研流水线，包括数据获取、清洗、特征工程、横截面处理、XGBoost 训练、Walk-Forward 训练以及带 A 股交易约束的回测。README 中明确将风险模型列为后续扩展方向。fileciteturn0file0L39-L68

当前系统已经具备较好的 Risk Model 数据基础：

- 日线 OHLCV、成交额等行情数据；
- Point-in-Time 基本面数据；
- 资金流数据；
- 行业 One-Hot 数据；
- ST、停牌、次新股等状态信息；
- 已有 Momentum、Value、Growth、Volatility、Liquidity、Leverage 等大量可用于风险暴露描述的变量；
- 已有横截面 Winsorize + Z-Score 处理流程；
- 回测已经具备行业权重、个股权重、流动性等组合约束。

README 中的数据处理流程已经包含 Point-in-Time 财务数据和行业 One-Hot 编码，这可以直接作为 Risk Model 的数据基础。fileciteturn0file0L397-L406

因此，当前阶段**不建议继续优先扩充 Alpha 因子**，而应该把研发重点转向：

```text
Alpha Model
    ↓
Risk Model
    ↓
Portfolio Optimizer
    ↓
Backtest
```

Risk Model V1 的目标不是预测股票收益，而是估计：

1. 股票对系统性风险因子的暴露；
2. 各风险因子的共同波动；
3. 股票自身的特质风险；
4. 最终得到可供 Optimizer 使用的股票协方差矩阵。

核心关系：

\[
\Sigma = XFX^T + D
\]

其中：

- \(X\)：股票 × 风险因子的暴露矩阵；
- \(F\)：风险因子收益的协方差矩阵；
- \(D\)：特质风险协方差矩阵，V1 中先采用对角矩阵；
- \(\Sigma\)：最终股票协方差矩阵。

---

# 2. Risk Model V1 的整体架构

建议新增独立的 `risk` 模块，不要把 Risk Model 混入现有 Alpha Feature Generator。

```text
src/
├── trainer/
├── backtester/
├── utils/
│   ├── dataset_fetcher/
│   ├── dataset_processor/
│   ├── features_generator/
│   ├── cross_sectional_processor/
│   └── ic_validator/
│
└── risk/
    ├── __init__.py
    ├── risk_exposure.py
    ├── factor_return.py
    ├── factor_covariance.py
    ├── specific_risk.py
    ├── covariance_builder.py
    ├── risk_model.py
    └── validators/
        ├── exposure_validator.py
        ├── covariance_validator.py
        └── risk_forecast_validator.py
```

配置文件：

```text
conf/
└── risk_model/
    ├── risk_model_v1.json
    ├── risk_factors_v1.json
    └── risk_model_validation_v1.json
```

输出：

```text
outputs/
└── risk_model/
    └── v1/
        ├── exposures/
        ├── factor_returns/
        ├── factor_covariance/
        ├── specific_risk/
        ├── covariance/
        ├── diagnostics/
        └── reports/
```

---

# 3. V1 Risk Factor 设计

## 3.1 设计原则

Risk Factor 与 Alpha Factor 的目的不同。

Alpha Factor：

> 寻找未来收益。

Risk Factor：

> 描述组合的系统性风险来源。

因此不要直接把 86 个 Alpha 特征全部放进 Risk Model。

V1 控制在 **8 个风格因子 + 行业因子** 左右。

---

## 3.2 风格风险因子

| Risk Factor | 推荐计算方式 | 当前数据情况 | V1 |
|---|---|---|---|
| Size | `log(market_cap)` | 需要确认/补充市值 | 必须 |
| Beta | 252 日股票收益对市场收益滚动回归 | 可由日线计算 | 必须 |
| Momentum | 12M Skip-1M 动量 | 已有 | 必须 |
| Value | PB/PE/BPS 等综合 | 已有基础数据 | 必须 |
| Growth | Revenue YoY + ROE 等 | 已有 | 必须 |
| Volatility | 60D/20D 波动率 | 已有 | 必须 |
| Liquidity | Amihud 20D | 已有 | 必须 |
| Leverage | Debt Ratio | 已有 | 必须 |

建议 V1 风格因子：

```text
SIZE
BETA
MOMENTUM
VALUE
GROWTH
VOLATILITY
LIQUIDITY
LEVERAGE
```

---

# 4. 风格因子的具体计算

## 4.1 Size

使用：

\[
Size_i = \log(MarketCap_i)
\]

然后每日横截面：

1. Winsorize；
2. Z-Score。

注意：

**Size 是 Risk Exposure，不是 Alpha Feature。**

如果当前数据中没有 Point-in-Time 市值，需要从现有行情和股本数据构建；不能使用未来市值。

---

## 4.2 Beta

使用过去 252 个交易日：

\[
r_{i,t} = \alpha_i + \beta_i r_{m,t} + \epsilon_{i,t}
\]

其中：

- \(r_{i,t}\)：股票日收益；
- \(r_{m,t}\)：市场基准收益；
- \(\beta_i\)：股票 Beta。

市场基准 V1 建议：

```text
000300.SH
```

如果现有 benchmark 数据不能完整覆盖 2005-2025，应使用项目已有可靠市场收益序列重新构建。

Beta 不建议简单使用相关系数。

计算完成后：

```text
Beta
→ Winsorize
→ Z-Score
```

---

## 4.3 Momentum

优先使用：

```text
momentum_12m_skip1m
```

即：

\[
Momentum_i =
\frac{P_{t-21}}{P_{t-252}}-1
\]

实际实现应根据现有交易日历计算，不直接使用自然日。

目的：

避免把最近一个月的短期反转效应混入长期动量。

---

## 4.4 Value

V1 可以组合：

```text
log(BPS / Price)
-EPS / Price
-PB
-PE
```

由于不同 Value 指标尺度不同，建议：

```text
单因子横截面标准化
        ↓
等权平均
        ↓
再次横截面标准化
```

得到：

```text
VALUE
```

注意：

所有基本面数据必须遵守现有 Point-in-Time 规则。

---

## 4.5 Growth

V1 使用：

```text
revenue_yoy
roe
gross_margin_change
revenue_yoy_acceleration
```

推荐：

```text
各自横截面标准化
        ↓
等权平均
        ↓
横截面标准化
```

得到：

```text
GROWTH
```

不要直接把 86 个 Alpha 因子都塞入 Growth。

Risk Model 需要稳定、可解释的风险暴露。

---

## 4.6 Volatility

推荐：

```text
volatility_20
volatility_60
downside_vol_20
```

组合：

```text
VOLATILITY
=
mean(
    z(volatility_20),
    z(volatility_60),
    z(downside_vol_20)
)
```

再进行一次横截面 Z-Score。

---

## 4.7 Liquidity

直接使用：

```text
amihud_20
```

计算：

\[
Amihud_{20}
=
mean\left(
\frac{|r_t|}{Amount_t}
\right)
\]

然后：

```text
log(1 + scaled_amihud)
→ Winsorize
→ Z-Score
```

原因是 Amihud 通常高度右偏。

---

## 4.8 Leverage

直接使用：

```text
debt_ratio
```

Point-in-Time 对齐后：

```text
Winsorize
→ Z-Score
```

---

# 5. 行业风险因子

行业风险因子与行业 Alpha 因子必须区分。

当前项目已经在数据处理阶段生成行业 One-Hot 编码。fileciteturn0file0L399-L406

Risk Model 中使用：

```text
申万一级行业
```

每只股票：

```text
所属行业 = 1
其他行业 = 0
```

例如：

```text
        银行  食品饮料  医药  电子
股票A    1      0       0     0
股票B    0      1       0     0
股票C    0      0       0     1
```

---

## 5.1 行业因子是否需要全部保留？

需要。

但每日横截面回归存在：

```text
截距 + 全行业Dummy
```

完全共线的问题。

V1 采用：

```text
截距
+
N-1个行业Dummy
```

或者：

```text
去掉一个基准行业
```

即可。

---

# 6. Exposure Matrix

每天生成：

```text
X_t
```

结构：

```text
             SIZE BETA MOM VALUE GROWTH VOL LIQ LEV BANK 食品 ...
000001.SZ      ...
000002.SZ      ...
...
```

其中：

- 风格因子：连续暴露；
- 行业因子：0/1。

建议输出：

```text
trade_date
ts_code
SIZE
BETA
MOMENTUM
VALUE
GROWTH
VOLATILITY
LIQUIDITY
LEVERAGE
industry_*
```

---

# 7. Factor Return 估计

这是 Risk Model 的核心步骤之一。

每日使用股票未来/当日收益做横截面回归：

\[
r_{i,t}
=
\alpha_t
+
X_{i,t}f_t
+
\epsilon_{i,t}
\]

其中：

- \(r_{i,t}\)：股票当日收益；
- \(X_{i,t}\)：风险暴露；
- \(f_t\)：当天风险因子收益；
- \(\epsilon_{i,t}\)：特质收益。

---

## 7.1 使用 WLS，而不是普通 OLS

建议 V1：

```text
Weighted Least Squares
```

权重可以使用：

```text
w_i = sqrt(Amount_20d)
```

或者更保守：

```text
w_i = sqrt(min(Amount_20d, cap))
```

原因：

成交额极低的小盘股噪声非常大。

同时又不能让超大市值股票完全主导回归。

---

## 7.2 每日回归流程

```text
当天股票池
    ↓
剔除无收益股票
    ↓
剔除无风险暴露股票
    ↓
构造 X
    ↓
构造 r
    ↓
WLS
    ↓
factor_returns_t
    ↓
residual_t
```

输出：

```text
trade_date
SIZE
BETA
MOMENTUM
VALUE
GROWTH
VOLATILITY
LIQUIDITY
LEVERAGE
industry_1
industry_2
...
```

---

# 8. Factor Covariance

得到每日：

```text
factor_returns
```

之后估计：

\[
F = Cov(f_t)
\]

---

## 8.1 V1 推荐参数

```text
lookback = 252 trading days
```

即：

```text
过去一年
```

但不直接使用简单样本协方差。

推荐：

```text
Ledoit-Wolf Shrinkage
```

得到：

```text
F_t
```

原因：

Risk Factor 数量约 30+：

```text
8 style
+
20~30 industry
```

如果只有 252 个样本，普通样本协方差容易不稳定。

---

## 8.2 后续再增加 EWMA

V1：

```text
Ledoit-Wolf
```

V1.1：

```text
EWMA
```

V2：

```text
EWMA + Shrinkage
```

不要一开始同时实现多个方法。

---

# 9. Specific Risk

每日横截面回归得到：

\[
\epsilon_{i,t}
=
r_{i,t}
-
X_{i,t}f_t
\]

即股票特质收益。

---

## 9.1 计算方法

过去 252 个交易日：

\[
\sigma_{i,specific}
=
Std(\epsilon_{i,t})
\]

得到：

```text
specific_vol
```

然后：

```text
specific_variance
=
specific_vol²
```

构造：

\[
D = diag(\sigma_1^2,\sigma_2^2,\ldots)
\]

---

## 9.2 必须处理小样本股票

例如：

```text
上市不足252日
长期停牌
数据缺失
```

不要直接计算。

建议：

```text
有效残差样本 >= 120
```

否则使用：

```text
行业/市值分组中位数
```

进行 Shrinkage。

---

# 10. 最终 Covariance Matrix

对于股票集合：

```text
N = 当前组合候选股票数量
K = Risk Factors
```

：

\[
\Sigma = XFX^T+D
\]

得到：

```text
N × N
```

的股票协方差矩阵。

例如：

```text
1000 stocks

↓

1000 × 1000 covariance
```

---

# 11. Risk Model 的时间点要求

这是本项目最重要的防泄露要求之一。

Risk Model 必须遵循：

```text
T日收盘后
计算Risk Model

↓

只能服务于T+1交易
```

因此：

```text
Exposure(T)
Factor Return(T)
Covariance(T)
Specific Risk(T)
```

都不能使用：

```text
T+1及以后
```

的信息。

---

## 11.1 Walk-Forward Risk Model

不能：

```text
2005-2025全部数据
        ↓
计算一个Risk Model
        ↓
回测2005-2025
```

这会产生严重的未来信息污染。

应该：

```text
2005
  ↓
只使用此前数据

2006
  ↓
只使用此前数据

……

2025
  ↓
只使用此前数据
```

---

# 12. 与当前 Walk-Forward Alpha 对接

当前 Alpha：

```text
5年训练
1年测试
滚动1年
```

Risk Model 建议与 Alpha 保持相同的 OOS 时间逻辑。

例如：

```text
Alpha WF Window

Train:
2000-2004

Test:
2005
```

2005 年某个交易日：

```text
Alpha Prediction
        +
Risk Model
        ↓
Optimizer
        ↓
Portfolio
```

Risk Model 的参数只能使用该日期之前的数据。

---

# 13. Risk Model V1 的验证体系

不能只看“协方差矩阵算出来了”。

必须做四层验证。

---

## 13.1 Exposure 验证

检查：

```text
缺失率
异常值
均值
标准差
最大值
最小值
```

例如：

```text
SIZE mean ≈ 0
SIZE std ≈ 1
```

对于标准化后的风格因子应该基本满足。

---

## 13.2 Covariance 数学验证

必须满足：

```text
Σ = Σ.T
```

并且：

```text
eigenvalues >= 0
```

允许极小数值误差：

```text
eigenvalue >= -1e-8
```

否则不能交给 Optimizer。

同时检查：

```text
NaN
Inf
condition number
diagonal > 0
```

---

# 14. Risk Forecast 验证

这是最重要的验证。

Risk Model 预测：

```text
未来风险
```

因此要做：

```text
预测风险
vs
实际风险
```

---

## 14.1 Portfolio Risk Backtest

随机生成或者使用当前 TopK 组合：

```text
w
```

模型预测：

\[
\sigma_{pred}
=
\sqrt{w^T\Sigma w}
\]

然后观察未来：

```text
1d realized vol
5d realized vol
20d realized vol
```

比较：

```text
predicted risk
vs
realized risk
```

---

## 14.2 重点指标

至少计算：

```text
Predicted Vol
Realized Vol
Vol Bias
RMSE
Correlation
Rank Correlation
```

如果：

```text
Predicted Vol
```

与：

```text
Realized Vol
```

完全没有关系，

那么 Risk Model 即使数学上完美，也没有实际价值。

---

# 15. Risk Model 与现有 Alpha Model 的边界

必须严格区分：

```text
Alpha Model
```

负责：

```text
谁值得买
```

Risk Model：

```text
这些股票的风险是什么
```

Optimizer：

```text
每只股票买多少
```

不要让 Risk Model 偷偷使用：

```text
XGBoost prediction
label
future return
```

Risk Model 应该是独立于 Alpha 的。

---

# 16. 第一版不要做的事情

V1 明确禁止：

### 16.1 不把86个Alpha特征全部作为Risk Factor

原因：

```text
Risk Factor ≠ Alpha Factor
```

---

### 16.2 不做深度学习Risk Model

暂时没有必要。

---

### 16.3 不做复杂GARCH

V1 没必要。

---

### 16.4 不直接优化收益

Risk Model 不负责预测收益。

---

### 16.5 不直接做Black-Litterman

Black-Litterman 属于 Portfolio Construction 层。

先把：

```text
X
F
D
Σ
```

做好。

---

# 17. 建议的开发顺序

严格按照以下顺序开发。

```text
Phase 1
Risk Exposure
        ↓
Phase 2
Daily Factor Return
        ↓
Phase 3
Factor Covariance
        ↓
Phase 4
Specific Risk
        ↓
Phase 5
Stock Covariance
        ↓
Phase 6
Risk Validation
        ↓
Phase 7
Portfolio Optimizer
```

不要并行开发 Optimizer。

---

# 18. Phase 1：Risk Exposure

实现：

```text
src/risk/risk_exposure.py
```

功能：

```python
build_style_exposure()
build_industry_exposure()
build_exposure_matrix()
```

输入：

```text
daily bars
fundamentals
industry
```

输出：

```text
trade_date × stock × risk_factor
```

验收标准：

- 无未来数据；
- 风格因子横截面标准化；
- 行业暴露正确；
- 缺失值处理明确；
- 输出可以按日期读取。

---

# 19. Phase 2：Factor Return

实现：

```text
src/risk/factor_return.py
```

核心：

```python
estimate_daily_factor_returns(
    exposures,
    returns,
    weights
)
```

使用 WLS。

输出：

```text
factor_returns.parquet
residual_returns.parquet
```

验收标准：

- 每个交易日都有回归结果；
- 回归残差可重构股票收益；
- \(R^2\) 合理；
- 无明显异常因子收益；
- 行业因子没有完全共线。

---

# 20. Phase 3：Factor Covariance

实现：

```text
src/risk/factor_covariance.py
```

V1：

```text
lookback = 252
estimator = LedoitWolf
```

输出：

```text
factor_covariance/
    20050104.npy
    ...
```

验收：

```text
F == F.T
eigenvalue(F) >= 0
```

---

# 21. Phase 4：Specific Risk

实现：

```text
src/risk/specific_risk.py
```

输出：

```text
trade_date
ts_code
specific_vol
specific_variance
sample_count
```

需要处理：

```text
停牌
上市不足
长期缺失
极端残差
```

---

# 22. Phase 5：Covariance Builder

实现：

```text
src/risk/covariance_builder.py
```

核心：

```python
Sigma = X @ F @ X.T + D
```

同时加入：

```text
PSD检查
对称化
数值稳定处理
```

如果出现非常小的负特征值：

```text
Sigma = (Sigma + Sigma.T) / 2
```

必要时做最小特征值修正。

但**不要默认暴力加很大的 diagonal jitter**，否则会扭曲 Risk Model。

---

# 23. Phase 6：Risk Validation

建立：

```text
src/risk/validators/
```

至少包含：

```text
ExposureValidator
CovarianceValidator
RiskForecastValidator
```

最终生成：

```text
risk_model_validation_report.md
```

报告必须回答：

1. 风格因子是否稳定？
2. 行业因子是否合理？
3. Factor Return 是否稳定？
4. Factor Covariance 是否 PSD？
5. Specific Risk 是否合理？
6. Predicted Vol 是否能解释 Realized Vol？
7. Risk Model 是否存在明显结构性偏差？

---

# 24. 建议的配置文件

`conf/risk_model/risk_model_v1.json`

```json
{
  "version": "risk_model_v1",
  "style_factors": [
    "size",
    "beta",
    "momentum",
    "value",
    "growth",
    "volatility",
    "liquidity",
    "leverage"
  ],
  "industry_factor": {
    "enabled": true,
    "classification": "sw_l1"
  },
  "cross_sectional": {
    "winsorize": true,
    "winsorize_method": "mad",
    "winsorize_n": 5,
    "zscore": true
  },
  "factor_return": {
    "method": "wls",
    "weight": "sqrt_amount_20",
    "min_stocks": 100
  },
  "factor_covariance": {
    "method": "ledoit_wolf",
    "lookback": 252
  },
  "specific_risk": {
    "lookback": 252,
    "min_observations": 120,
    "method": "rolling_std"
  },
  "output": {
    "exposure": "outputs/risk_model/v1/exposures",
    "factor_return": "outputs/risk_model/v1/factor_returns",
    "factor_covariance": "outputs/risk_model/v1/factor_covariance",
    "specific_risk": "outputs/risk_model/v1/specific_risk",
    "covariance": "outputs/risk_model/v1/covariance"
  }
}
```

---

# 25. 第一版 Risk Model 的验收标准

只有满足以下条件，才能进入 Optimizer。

## 数据层

- [ ] Point-in-Time 数据；
- [ ] 没有未来数据；
- [ ] 风险暴露每日可生成；
- [ ] 行业分类无明显错误。

## 因子层

- [ ] 8 个风格因子；
- [ ] 申万一级行业因子；
- [ ] 每日因子收益可估计；
- [ ] 因子收益没有明显异常。

## Covariance

- [ ] \(F\) PSD；
- [ ] \(D\) 对角元素 > 0；
- [ ] \(\Sigma\) PSD；
- [ ] 无 NaN/Inf；
- [ ] 数值条件合理。

## Forecast

- [ ] Predicted Vol 与 Realized Vol 有正相关；
- [ ] 风险排序具有预测能力；
- [ ] 不存在系统性严重低估风险。

## Walk-Forward

- [ ] 不能使用未来数据；
- [ ] 每个 OOS 日期独立生成 Risk Model；
- [ ] 与 Alpha WF 时间轴兼容。

---

# 26. 与现有项目的最终集成架构

完成 Risk Model 后，系统将从：

```text
Features
   ↓
XGBoost
   ↓
Prediction
   ↓
TopK
   ↓
Equal Weight
   ↓
Backtest
```

升级为：

```text
                         ┌───────────────┐
                         │   XGBoost     │
                         │ Alpha Model   │
                         └───────┬───────┘
                                 │
                            expected return
                                 │
                                 ▼
                         ┌───────────────┐
                         │   Optimizer   │
                         └───────┬───────┘
                                 ▲
                                 │
                         covariance Σ
                                 │
                         ┌───────┴───────┐
                         │  Risk Model   │
                         └───────┬───────┘
                                 │
                  ┌──────────────┼──────────────┐
                  │              │              │
              Exposure       Factor Cov      Specific
                  X              F             D
                  │              │              │
                  └──────────────┴──────────────┘

                                 ↓
                              Weights
                                 ↓
                         Backtest Engine
                                 ↓
                       Performance Attribution
```

---

# 27. V1 → V2 演进路线

## Risk Model V1

```text
8 Style Factors
+
Industry Factors
+
WLS
+
Ledoit-Wolf
+
Specific Risk
```

目标：

> 建立稳定、可解释、可用于 Optimizer 的基础风险矩阵。

---

## Risk Model V1.1

加入：

```text
EWMA covariance
```

比较：

```text
Sample Cov
vs
Ledoit-Wolf
vs
EWMA
```

使用 OOS Realized Vol 判断。

---

## Risk Model V2

加入：

```text
更细的 Size
Beta 非线性
Residual Correlation
Risk Factor Hierarchy
Volatility Regime
```

再考虑：

```text
Barra-style hierarchical risk model
```

---

# 28. Risk Model 开发完成后的 Optimizer

Risk Model V1 验证通过后，再开发：

```text
Portfolio Optimizer V1
```

第一版只需要：

\[
\max_w
\quad
\alpha^T w
-
\lambda w^T\Sigma w
-
\gamma TC(w,w_{prev})
\]

约束：

```text
sum(w) = 1

0 <= w_i <= 3%

industry_weight <= 20%

industry_stock_count <= 10

turnover <= threshold

liquidity_constraint

beta constraint
```

这会自然接入当前已有的回测约束体系。README 已经包含行业约束、个股权重上限、流动性限制以及波动率加权等组合管理能力，因此 Optimizer 不需要重新设计这些底层交易规则。fileciteturn0file0L505-L515

---

# 29. 本阶段最终目标

Risk Model V1 **不是为了让回测收益更高**。

正确的目标应该是：

```text
Alpha 不变

↓

Risk Model
↓

Optimizer

↓

在相近 Alpha 下：

降低 Volatility
降低 Max Drawdown
降低行业集中度
降低风格集中度
提高 Risk-adjusted Return
```

因此最终比较必须至少包含：

| 指标 | 当前 TopK 等权 | Risk + Optimizer |
|---|---:|---:|
| 年化收益 | 基准 | 对比 |
| 年化波动 | 基准 | 对比 |
| Sharpe | 基准 | 对比 |
| Max Drawdown | 基准 | 对比 |
| Calmar | 基准 | 对比 |
| Beta | 基准 | 对比 |
| Turnover | 基准 | 对比 |
| Transaction Cost | 基准 | 对比 |
| 行业最大暴露 | 基准 | 对比 |
| 风格最大暴露 | 基准 | 对比 |

**最重要的判断标准不是“Optimizer 后收益有没有暴涨”，而是：在 Alpha 信号基本不变的情况下，是否能够用更低的风险获得更好的 Sharpe / Calmar，并且在不同 Walk-Forward 窗口中保持稳定。**

---

# 30. 最终开发任务清单

### Phase 1 — Exposure

- [ ] 确认 Point-in-Time 市值数据
- [ ] 实现 Size
- [ ] 实现 Beta
- [ ] 整理 Momentum
- [ ] 整理 Value
- [ ] 整理 Growth
- [ ] 整理 Volatility
- [ ] 整理 Liquidity
- [ ] 整理 Leverage
- [ ] 接入行业 One-Hot
- [ ] Exposure Validator

### Phase 2 — Factor Return

- [ ] WLS
- [ ] 每日横截面回归
- [ ] Factor Return
- [ ] Residual Return
- [ ] 回归质量分析

### Phase 3 — Factor Covariance

- [ ] 252D rolling window
- [ ] Ledoit-Wolf
- [ ] PSD检查
- [ ] Factor Covariance 可视化/诊断

### Phase 4 — Specific Risk

- [ ] Residual volatility
- [ ] 252D rolling
- [ ] Min observations
- [ ] Small sample shrinkage
- [ ] Specific Risk Validator

### Phase 5 — Stock Covariance

- [ ] \(XFX^T+D\)
- [ ] PSD
- [ ] Numerical stability
- [ ] 输出每日 Sigma

### Phase 6 — OOS Validation

- [ ] Predicted Vol
- [ ] Realized Vol
- [ ] Vol correlation
- [ ] Vol bias
- [ ] Risk ranking
- [ ] Walk-Forward 验证

### Phase 7 — Optimizer

**Risk Model V1 完成并验证通过后再开始。**

---

## 结论

当前项目已经不适合继续以“增加 Alpha 因子 → 看 IC → 回测收益”的方式作为主要研发路径。

下一阶段应该正式进入：

```text
Alpha Research
       ↓
Risk Modeling        ← 当前阶段
       ↓
Portfolio Optimization
       ↓
Portfolio Attribution
       ↓
Execution Modeling
       ↓
Paper Trading
```

**Risk Model V1 的核心交付物只有四个：**

```text
X = Exposure Matrix

F = Factor Covariance

D = Specific Risk

Σ = Stock Covariance
```

只要这四个对象能够在严格 Walk-Forward / OOS 条件下稳定生成，并且 `Predicted Risk` 对 `Realized Risk` 有实际预测能力，就可以进入 Portfolio Optimizer 开发。

