# Portfolio Optimizer V1 技术设计文档

**版本**: V1.0  
**设计日期**: 2026-08-10  
**状态**: Design  
**适用项目**: A股多因子 + XGBoost Alpha + Barra Risk Model 量化交易系统

---

# 1. 文档目标

本文档定义 Portfolio Optimizer V1 的完整技术方案。

Optimizer 位于当前量化系统的：

```text
Data
  ↓
Factor Engineering
  ↓
XGBoost Alpha Model
  ↓
Risk Model V1
  ↓
Portfolio Optimizer V1
  ↓
Backtest / Execution
```

当前系统已经具备：

1. 基于 86 个特征的 XGBoost Alpha Model
2. Walk-Forward 训练与预测
3. 21 年 OOS 回测
4. Barra 风格 Risk Model V1
5. 风险暴露 \(X\)
6. 因子协方差 \(F\)
7. 特异性风险 \(D\)
8. 风险排序验证
9. 全约束 Backtest

因此 Portfolio Optimizer V1 的目标不是重新产生 Alpha，而是：

> **在尽可能保留 Alpha 的前提下，对组合风险、风格暴露、行业暴露、换手率和交易成本进行系统控制。**

核心思想：

\[
\boxed{
Alpha
\rightarrow
Risk\ Adjusted\ Portfolio
}
\]

---

# 2. 当前系统背景

## 2.1 Alpha Model

当前 Alpha Model：

- 模型：XGBoost
- 特征：86
- Label：5 日远期收益排名 `label_rank_5d`
- Walk-Forward：5 年训练 + 1 年测试
- OOS：2005-2025
- 平均 IC：约 0.1594
- 平均 RankIC：约 0.1417
- IC IR：约 5.80

当前最佳回测组合：

```text
topk = 30
n_drop = 10
rebalance = 5d
weight_cap = 3%
```

结果：

```text
Annual Return       41.89%
Sharpe               1.373
Max Drawdown       -58.36%
Volatility           28.45%
Beta                  0.721
Annual Alpha         22.08%
```

---

# 3. Risk Model V1

Risk Model 当前采用：

\[
\Sigma = XFX^T+D
\]

其中：

- \(X\)：股票 × 风险因子暴露矩阵
- \(F\)：因子收益协方差矩阵
- \(D\)：股票特异性风险矩阵

风险因子：

```text
INTERCEPT

SIZE
BETA
MOMENTUM
VALUE
GROWTH
VOLATILITY
LIQUIDITY
LEVERAGE

26 × 申万一级行业
```

共 35 个风险因子。

Risk Model V1 已完成：

- WLS 截面因子收益估计
- Ledoit-Wolf 因子协方差
- 特异性风险估计
- PSD 验证
- 风险横截面排序验证
- 风险分组单调性验证

2013-2025 补充验证：

```text
平均 Pearson       0.38
平均 Spearman      0.44
正相关天数          100%
5组单调通过率        100%
Q5/Q1 波动率比       1.84x
PSD 通过率           100%
```

因此 Risk Model V1 可以作为 Optimizer V1 的正式风险输入。

---

# 4. Optimizer V1 的设计目标

Optimizer V1 需要解决以下问题。

## 4.1 Alpha 集中导致的风险暴露

XGBoost 可能天然偏好某些风格：

```text
Momentum
Liquidity
Size
Growth
...
```

最终组合可能出现：

```text
Momentum exposure = +1.8
Liquidity exposure = +1.5
Size exposure = +1.3
```

Optimizer 负责控制这些暴露。

---

## 4.2 行业集中

Alpha Model 可能集中选择某些行业。

例如：

```text
电子     35%
计算机   22%
机械     15%
```

Optimizer 负责限制：

```text
单行业绝对权重
行业相对基准偏离
```

---

## 4.3 个股集中

控制：

\[
w_i \leq 3\%
\]

---

## 4.4 组合整体风险

控制：

\[
w^T\Sigma w
\]

---

## 4.5 换手率

避免 Optimizer 每次调仓产生大量交易：

\[
\|w-w_{prev}\|_1
\]

---

## 4.6 交易成本

避免理论 Alpha 被交易成本吞噬。

---

# 5. Optimizer V1 核心设计

Optimizer V1 采用：

> **Long-only Quadratic Programming**

即：

\[
\boxed{
\text{Quadratic Programming}
}
\]

而不是：

- Black-Litterman
- CVaR
- Risk Parity
- GARCH
- 多期优化
- 非线性优化

这些全部放到 V2/V3。

---

# 6. 数学定义

设：

\[
N = \text{可投资股票数量}
\]

\[
K = 35
\]

---

## 6.1 决策变量

Optimizer 的唯一核心决策变量：

\[
\boxed{
w\in R^N
}
\]

其中：

\[
w_i
\]

表示股票 \(i\) 的目标组合权重。

---

# 7. Alpha 输入

Alpha Model 输出：

\[
p_i
\]

表示股票 \(i\) 的预测收益 / Alpha Score。

但是：

> **不能直接把 XGBoost prediction 原值作为优化器 Alpha。**

因为不同 Walk-Forward 窗口、不同日期的 prediction scale 可能不同。

建议在 Optimizer 前进行：

### 横截面标准化

\[
\tilde p_i
=
\frac{
p_i-\mu_p
}{
\sigma_p
}
\]

然后作为 Optimizer 的 Alpha 输入。

---

# 8. Alpha Scaling

建议：

```text
raw prediction
      ↓
cross-sectional winsorize
      ↓
cross-sectional z-score
      ↓
alpha_score
```

即：

\[
\boxed{
\alpha = ZScore(Winsorize(prediction))
}
\]

注意：

> Alpha 标准化必须按交易日进行，不能跨整个历史区间标准化。

---

# 9. Risk Model 输入

每个调仓日需要：

```text
X_t
F_t
D_t
```

其中：

```text
X_t : N × K
F_t : K × K
D_t : N × N diagonal
```

但是：

> **Optimizer 不需要实际构造 N×N Sigma。**

---

# 10. Sigma 的高效计算

理论：

\[
\Sigma=XFX^T+D
\]

组合方差：

\[
\sigma_p^2=w^T\Sigma w
\]

展开：

\[
w^TXFX^Tw+w^TDw
\]

令：

\[
z=X^Tw
\]

则：

\[
\boxed{
\sigma_p^2
=
z^TFz
+
\sum_iD_iw_i^2
}
\]

因此：

```text
Portfolio Risk
    ↓
X.T @ w
    ↓
factor exposure
    ↓
factor covariance
    ↓
specific risk
```

复杂度从：

\[
O(N^2)
\]

降低到：

\[
O(NK+K^2)
\]

由于：

\[
K=35
\]

而：

\[
N\approx5000
\]

这个方式非常适合当前项目。

---

# 11. Optimizer 核心目标函数

V1 使用：

\[
\boxed{
\max_w
\left[
\alpha^Tw
-\lambda w^T\Sigma w
-\gamma\|w-w_{prev}\|_1
-C(w,w_{prev})
\right]
}
\]

其中：

### 第一项

\[
\alpha^Tw
\]

最大化组合 Alpha。

---

### 第二项

\[
\lambda w^T\Sigma w
\]

控制组合风险。

---

### 第三项

\[
\gamma\|w-w_{prev}\|_1
\]

控制换手。

---

### 第四项

\[
C(w,w_{prev})
\]

交易成本。

---

# 12. V1 推荐简化形式

第一版建议先采用：

\[
\boxed{
\max_w
\alpha^Tw
-\lambda w^T\Sigma w
-\gamma\|w-w_{prev}\|_1
}
\]

交易成本暂时作为：

> 独立诊断指标

而不是第一版直接加入复杂的非线性成本模型。

原因：

当前最重要的是验证：

> Risk Model 是否真的能够改善 Alpha Portfolio。

不要同时引入太多变量。

---

# 13. 为什么使用 L1 Turnover Penalty

定义：

\[
Turnover=
\frac12
\sum_i|w_i-w_{i,prev}|
\]

如果使用：

\[
\gamma\|w-w_{prev}\|_1
\]

可以直接惩罚调仓。

其作用：

```text
Alpha
  ↓
Optimizer
  ↓
如果新增 Alpha 不足以覆盖 Risk + Turnover
  ↓
不调仓
```

这比单纯设置一个最大换手率更平滑。

---

# 14. 基础约束

## 14.1 满仓

\[
\boxed{
\sum_iw_i=1
}
\]

---

## 14.2 Long-only

\[
\boxed{
w_i\ge0
}
\]

V1 不支持做空。

---

## 14.3 个股权重

\[
\boxed{
w_i\le w_{max}
}
\]

默认：

\[
w_{max}=3\%
\]

---

# 15. 行业约束

设：

\[
G_{ij}
=
\begin{cases}
1 & stock_i\in industry_j\\
0 & otherwise
\end{cases}
\]

行业权重：

\[
I_j=G_j^Tw
\]

---

## 15.1 绝对行业上限

V1 默认：

\[
\boxed{
I_j\le20\%
}
\]

---

## 15.2 行业最小权重

默认：

\[
I_j\ge0
\]

不强制行业配置。

原因：

> Alpha Model 本身应该拥有行业选择能力。

Optimizer 主要负责防止过度集中。

---

# 16. 风格因子约束

对于 8 个风格因子：

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

组合因子暴露：

\[
E_k=X_k^Tw
\]

---

# 17. 风格暴露约束

V1 支持：

\[
L_k\le X_k^Tw\le U_k
\]

例如：

```text
SIZE        [-0.5, +0.5]
BETA        [ 0.7, +1.2]
MOMENTUM    [-0.5, +0.8]
VALUE       [-0.5, +0.5]
GROWTH      [-0.5, +0.8]
VOLATILITY  [-0.5, +0.5]
LIQUIDITY   [-0.5, +0.5]
LEVERAGE    [-0.5, +0.5]
```

但这些不是最终参数。

---

# 18. 风格约束参数的确定方法

不能拍脑袋设置。

应该先运行：

```text
Raw Alpha Portfolio
```

统计 2005-2025：

```text
SIZE exposure
BETA exposure
MOMENTUM exposure
...
```

得到：

```text
P5
P25
P50
P75
P95
```

然后根据历史 Alpha Portfolio 的自然暴露分布确定限制。

例如：

```text
MOMENTUM

Raw Alpha:
P5   = -0.8
P50  = +0.9
P95  = +1.8
```

可以设计：

\[
-0.5\le Momentum\le1.2
\]

即：

> 不完全消灭 Alpha 的 Momentum 暴露，只限制极端暴露。

---

# 19. Beta 约束

Beta 与其他风格因子略有不同。

建议 V1：

\[
\boxed{
0.7\le Beta\le1.2
}
\]

原因：

你的当前 Alpha Portfolio：

\[
Beta\approx0.72
\]

所以不能直接把 Beta 限制在：

\[
0.9\sim1.1
\]

否则会人为破坏现有 Alpha 策略。

V1 应该允许：

```text
Low Beta
Normal Beta
Moderate High Beta
```

而不是强制 Beta = 1。

---

# 20. 基准相对行业约束

V1 同时支持 Benchmark-relative 模式。

设：

\[
b_j
\]

为基准行业权重。

则：

\[
Active_j=I_j-b_j
\]

约束：

\[
\boxed{
-A_j\le I_j-b_j\le A_j
}
\]

例如：

```text
电子     ±5%
银行     ±3%
计算机   ±5%
医药     ±5%
...
```

但默认：

> **V1 首先使用绝对行业约束，Benchmark-relative 作为实验模式。**

因为你的策略本质上不是指数增强。

---

# 21. 风险约束

除了目标函数中的风险惩罚，V1 建议支持硬风险约束。

定义：

\[
\sigma_p
=
\sqrt{
w^T\Sigma w
}
\]

可以约束：

\[
\boxed{
\sigma_p\le\sigma_{max}
}
\]

例如：

```text
annualized volatility <= 25%
```

但是：

> V1 不建议默认启用硬波动率约束。

原因：

风险模型本身存在预测误差，硬约束容易导致：

```text
Optimization infeasible
```

因此默认使用：

\[
\lambda Risk
\]

作为软约束。

---

# 22. Risk Budget

Optimizer V1 必须提供风险分解。

总风险：

\[
\sigma_p^2=w^T\Sigma w
\]

因子风险：

\[
\sigma_F^2
=
(X^Tw)^TF(X^Tw)
\]

特异性风险：

\[
\sigma_D^2
=
\sum_iD_iw_i^2
\]

因此：

\[
\boxed{
\sigma_p^2
=
\sigma_F^2+\sigma_D^2
}
\]

---

# 23. 因子风险贡献

设：

\[
z=X^Tw
\]

因子风险：

\[
\sigma_F^2=z^TFz
\]

可以进一步计算：

\[
RC_k
=
\frac{
z_k(Fz)_k
}{
\sigma_p^2
}
\]

从而得到：

```text
SIZE risk contribution
BETA risk contribution
MOMENTUM risk contribution
...
```

---

# 24. 特异性风险贡献

股票 \(i\) 的特异性风险贡献：

\[
RC_i^{specific}
=
\frac{
w_i^2D_i
}{
\sigma_p^2
}
\]

用于诊断：

> 哪些股票虽然 Alpha 很高，但承担了大量 idiosyncratic risk？

---

# 25. Optimizer 的参数

V1 配置：

```json
{
    "objective": {
        "risk_aversion": 1.0,
        "turnover_penalty": 0.01
    },

    "portfolio": {
        "long_only": true,
        "fully_invested": true,
        "max_weight": 0.03
    },

    "industry": {
        "max_weight": 0.20
    },

    "style": {
        "SIZE": [-0.5, 0.5],
        "BETA": [0.7, 1.2],
        "MOMENTUM": [-0.5, 0.8],
        "VALUE": [-0.5, 0.5],
        "GROWTH": [-0.5, 0.8],
        "VOLATILITY": [-0.5, 0.5],
        "LIQUIDITY": [-0.5, 0.5],
        "LEVERAGE": [-0.5, 0.5]
    }
}
```

注意：

> 这里的参数只是初始实验值，不代表最终生产参数。

---

# 26. Risk Aversion 的确定

核心参数：

\[
\lambda
\]

决定：

```text
Alpha vs Risk
```

的权衡。

建议进行：

```text
lambda ∈
[0,
 0.01,
 0.03,
 0.05,
 0.1,
 0.2,
 0.5,
 1.0]
```

网格实验。

观察：

```text
Annual Return
Sharpe
Max Drawdown
Volatility
Turnover
Alpha retention
```

---

# 27. Alpha Retention

这是 Optimizer V1 非常重要的指标。

定义：

\[
AlphaRetention
=
\frac{
\alpha^Tw_{opt}
}{
\alpha^Tw_{raw}
}
\]

例如：

```text
Raw Alpha = 1.00
Optimizer Alpha = 0.91
```

则：

\[
AlphaRetention=91\%
\]

一个好的 Optimizer 应该：

> 用较小 Alpha 损失换取明显风险下降。

---

# 28. Optimizer 的核心评价指标

不能只看收益。

必须同时看：

### Return

```text
Annual Return
```

### Risk

```text
Annual Volatility
Maximum Drawdown
VaR
```

### Risk-adjusted Return

```text
Sharpe
Sortino
Calmar
```

### Alpha

```text
Alpha Retention
IC
```

### Risk Model

```text
Factor Exposure
Factor Risk
Specific Risk
```

### Trading

```text
Turnover
Transaction Cost
```

---

# 29. Optimizer 与当前 TopK 策略的关系

当前策略：

```text
XGBoost
 ↓
Rank
 ↓
Top 30
 ↓
n_drop=10
 ↓
Equal Weight
```

Optimizer 不应该立即替换整个流程。

第一阶段采用：

```text
XGBoost
   ↓
Rank
   ↓
Candidate Pool
   ↓
Optimizer
   ↓
Target Weight
   ↓
Backtest
```

---

# 30. Candidate Pool

V1 不建议让 Optimizer 直接对全 A 股 5000+ 股票优化。

建议：

\[
N_{candidate}=100\sim300
\]

例如：

```text
Top 100 Alpha stocks
```

进入 Optimizer。

原因：

1. 降低优化规模
2. 降低噪声
3. 保留 Alpha
4. 加快求解
5. 更接近真实交易

---

# 31. Candidate Pool 的推荐实验

至少测试：

```text
Top 30
Top 50
Top 100
Top 200
Top 300
```

然后比较：

```text
Optimization quality
Alpha retention
Turnover
Sharpe
```

我预计：

> Top 100 或 Top 200 会是比较合理的范围。

但必须用 OOS 实验确定。

---

# 32. 为什么不直接优化 5000 股票

理论上可以。

但是：

```text
N = 5500
K = 35
```

同时存在：

```text
个股约束
行业约束
风格约束
Turnover
交易成本
```

会显著增加 Solver 压力。

更重要的是：

> Alpha Model 本身对全市场的预测并不意味着所有股票都值得进入 Optimizer。

Candidate Pool 本身就是一个 Alpha Filter。

---

# 33. Optimizer V1 推荐流程

每个调仓日：

```text
T
│
├── Alpha Prediction
│
├── 股票池过滤
│   ├── ST
│   ├── 停牌
│   ├── 新股
│   ├── 流动性
│   └── 涨跌停
│
├── Alpha Ranking
│
├── Candidate Selection
│   └── Top 100
│
├── Risk Model
│   ├── X
│   ├── F
│   └── D
│
├── Previous Portfolio
│
├── Optimizer
│
└── Target Portfolio
        │
        ↓
T+1 Execution
```

---

# 34. 时间对齐

这是 Optimizer 最重要的防泄漏要求之一。

如果：

```text
T日收盘
```

产生 Alpha。

那么：

```text
T日 Risk Model
```

只能使用 T 日收盘时已经知道的信息。

最终：

```text
T signal
    ↓
Optimizer
    ↓
T+1 open
```

不能使用：

```text
T+1 data
```

---

# 35. Risk Model 时间对齐

建议：

\[
X_t,F_t,D_t
\]

全部对应：

> **T 日收盘之后可获得的风险状态。**

然后：

\[
w_{t+1}
=
Optimizer(
Alpha_t,
Risk_t,
w_t
)
\]

最终：

\[
Trade_{t+1}
\]

---

# 36. Optimizer 与交易约束的边界

建议将：

### Optimizer 负责

```text
Alpha
Risk
Style
Industry
Weight
Turnover
```

### Backtest / Execution 负责

```text
涨停
跌停
停牌
最小交易单位
成交量
滑点
佣金
印花税
开盘价成交
```

原因：

> Optimizer 产生“目标组合”，Execution 决定“能不能真正成交”。

---

# 37. 不可交易股票的处理

Optimizer Candidate Pool 中直接过滤：

```text
ST
停牌
上市不足60日
成交额不足
涨停无法买入
```

对于已有持仓：

> 即使当前不可卖，也不能让 Optimizer 假设它已经消失。

应该保留：

```text
current_weight
```

然后由 Execution Layer 判断：

```text
target_weight < current_weight
```

但实际：

```text
cannot sell
```

形成：

```text
actual_weight
```

---

# 38. Target Weight 与 Actual Weight

必须区分：

```text
target_weight
actual_weight
```

例如：

```text
股票A

current = 3%
target  = 0%

但是跌停无法卖出

actual = 3%
```

下一次 Optimizer 必须使用：

\[
w_{prev}=actual
\]

而不是：

\[
target
\]

否则会产生虚假的 Turnover。

---

# 39. Solver 选择

V1 推荐：

### 首选

**OSQP**

原因：

- QP
- 支持 sparse
- 支持线性约束
- Python 接口成熟
- 适合大量约束
- 可以 warm start

---

### 备选

```text
cvxpy
```

作为建模层。

推荐架构：

```text
CVXPY
   ↓
OSQP
```

而不是业务代码直接绑定 OSQP API。

这样后续可以替换：

```text
ECOS
CLARABEL
MOSEK
```

---

# 40. Optimizer 软件架构

建议新增：

```text
src/optimizer/
├── __init__.py
├── optimizer.py
├── objective.py
├── constraints.py
├── risk.py
├── alpha.py
├── candidate.py
├── solver.py
├── turnover.py
├── portfolio.py
└── diagnostics.py
```

---

# 41. optimizer.py

统一入口：

```python
class PortfolioOptimizer:

    def optimize(
        self,
        alpha,
        risk_model,
        current_weights,
        market_data,
        constraints
    ):
        ...
```

返回：

```python
OptimizationResult
```

---

# 42. OptimizationResult

建议：

```python
@dataclass
class OptimizationResult:

    weights: pd.Series

    expected_alpha: float

    portfolio_variance: float

    portfolio_volatility: float

    turnover: float

    factor_exposure: pd.Series

    factor_risk_contribution: pd.Series

    specific_risk: float

    solver_status: str

    objective_value: float

    diagnostics: dict
```

---

# 43. risk.py

提供：

```python
portfolio_variance(
    weights,
    exposures,
    factor_covariance,
    specific_risk
)
```

内部：

\[
z=X^Tw
\]

\[
risk=z^TFz+w^TDw
\]

---

# 44. constraints.py

负责：

```text
FullyInvestedConstraint
LongOnlyConstraint
WeightConstraint
IndustryConstraint
StyleConstraint
BetaConstraint
TurnoverConstraint
```

---

# 45. alpha.py

负责：

```text
raw prediction
    ↓
winsorize
    ↓
zscore
    ↓
alpha score
```

---

# 46. candidate.py

负责：

```text
Universe Filter
    ↓
Tradability Filter
    ↓
Alpha Ranking
    ↓
Top N
```

---

# 47. diagnostics.py

必须记录：

```text
solver_status
objective_value
alpha_before
alpha_after
risk_before
risk_after
turnover
constraint violations
factor exposure
industry exposure
```

---

# 48. 输出文件设计

建议：

```text
outputs/optimizer/v1/
├── portfolios/
│   └── portfolio.parquet
│
├── diagnostics/
│   └── optimizer_diagnostics.parquet
│
├── factor_exposure/
│   └── factor_exposure.parquet
│
├── risk_attribution/
│   └── risk_attribution.parquet
│
├── industry_exposure/
│   └── industry_exposure.parquet
│
└── summary.json
```

---

# 49. portfolio.parquet

建议：

```text
date
code
alpha
rank
current_weight
target_weight
actual_weight
trade_weight
```

---

# 50. optimizer_diagnostics.parquet

建议：

```text
date
solver_status
objective_value
expected_alpha
portfolio_volatility
turnover
alpha_retention
factor_risk
specific_risk
total_risk
```

---

# 51. factor_exposure.parquet

格式：

```text
date
SIZE
BETA
MOMENTUM
VALUE
GROWTH
VOLATILITY
LIQUIDITY
LEVERAGE
```

以及：

```text
industry_*
```

---

# 52. risk_attribution.parquet

建议：

```text
date
total_risk
factor_risk
specific_risk

SIZE
BETA
MOMENTUM
VALUE
GROWTH
VOLATILITY
LIQUIDITY
LEVERAGE
```

每个值表示对应风险贡献。

---

# 53. V1 必须实现的三种 Portfolio

为了验证 Optimizer，不应该只跑一个组合。

至少跑：

## Portfolio A：Raw Alpha

当前：

```text
Top30
Equal Weight
n_drop=10
5d
```

作为 baseline。

---

## Portfolio B：Risk Optimized

```text
Top100
Optimizer
Risk penalty
```

---

## Portfolio C：Risk + Turnover Optimized

```text
Top100
Optimizer
Risk penalty
Turnover penalty
```

最终比较：

```text
A vs B vs C
```

---

# 54. 第一阶段实验

首先不要改 Backtest。

只研究：

> Optimizer 是否真的改变了风险结构？

对每个调仓日：

```text
Raw Alpha Portfolio
        vs
Optimized Portfolio
```

比较：

```text
Beta
SIZE
MOMENTUM
VALUE
GROWTH
VOLATILITY
LIQUIDITY
LEVERAGE

Industry concentration

Predicted volatility
Factor risk
Specific risk
```

---

# 55. 第二阶段实验

加入 Backtest。

比较：

| 指标 | Raw Alpha | Optimized |
|---|---:|---:|
| Annual Return | | |
| Volatility | | |
| Sharpe | | |
| Sortino | | |
| Max Drawdown | | |
| Calmar | | |
| Turnover | | |
| Transaction Cost | | |
| Beta | | |
| Alpha | | |

---

# 56. 第三阶段实验：Risk Aversion Grid

测试：

```text
lambda =
0
0.01
0.03
0.05
0.1
0.2
0.5
1.0
```

观察：

```text
lambda ↑
    ↓
Risk ↓
    ↓
Alpha ↓
    ↓
Turnover ?
```

最终画出：

\[
Risk-Return\ Frontier
\]

---

# 57. 第四阶段实验：Style Constraint

逐步加入：

```text
Baseline
↓
Industry Constraint
↓
Style Constraint
↓
Beta Constraint
↓
Turnover Constraint
```

观察每增加一个约束：

```text
Return
Sharpe
Risk
Turnover
Alpha Retention
```

发生什么变化。

---

# 58. 第五阶段实验：Candidate Pool

测试：

```text
Top30
Top50
Top100
Top200
Top300
```

目标：

找到：

\[
N_{candidate}
\]

使得：

```text
Alpha Retention 高
+
Optimizer 稳定
+
Turnover 可控
+
Sharpe 提升
```

---

# 59. 第六阶段实验：交易成本

最终加入：

```text
Commission
Stamp Tax
Slippage
Market Impact
```

重点观察：

\[
NetReturn
\]

而不是：

\[
GrossReturn
\]

---

# 60. Optimizer 成功标准

V1 不要求：

> 年化收益一定超过 41.89%。

这是错误目标。

真正成功标准：

### 第一层

Risk：

```text
Predicted Volatility ↓
Factor Risk ↓
Style Concentration ↓
Industry Concentration ↓
```

---

### 第二层

Performance：

```text
Sharpe ↑
Max Drawdown ↓
Calmar ↑
```

---

### 第三层

Alpha：

```text
Alpha Retention >= 80~90%
```

作为初始参考。

---

### 第四层

Trading：

```text
Turnover 不明显增加
Transaction Cost 可控
```

---

# 61. 最重要的验证：Optimizer 是否真的创造价值

最终必须回答：

> **如果 Risk Model + Optimizer 加进去，Sharpe 是否比原始 Alpha Portfolio 更好？**

即：

\[
Sharpe_{optimized}
>
Sharpe_{raw}
\]

同时：

\[
MDD_{optimized}
<
MDD_{raw}
\]

理想情况下：

\[
Return_{optimized}
\approx
Return_{raw}
\]

也就是说：

> 用较少的收益换明显更低的风险。

---

# 62. 一个重要原则：不能用 Optimizer 调参污染 OOS

你的 Alpha Model 已经采用：

```text
Walk Forward
```

Optimizer 必须继续遵守同样原则。

不能：

```text
2005-2025 全历史
       ↓
寻找最佳 lambda
       ↓
得到 lambda=0.17
       ↓
用于整个回测
```

这是未来信息泄露。

---

# 63. 正确的 Walk-Forward Optimizer

例如：

```text
2000-2004
Training / Calibration
      ↓
2005
OOS

2001-2005
Calibration
      ↓
2006
OOS

...
```

Optimizer 参数：

```text
lambda
gamma
style limits
industry limits
candidate size
```

都应该只利用历史数据确定。

---

# 64. 但是 V1 可以先固定参数做研究

开发阶段允许：

```text
lambda = 0.1
gamma = 0.01
```

先跑：

```text
2005-2025
```

目的：

> 验证 Optimizer 机制本身。

但是一旦进入最终策略评估：

> 必须 Walk-Forward Calibration。

---

# 65. Optimizer 与现有 Backtest 的接口

建议：

```python
target_weights = optimizer.optimize(...)
```

然后：

```python
backtest_engine.execute(
    target_weights=target_weights,
    market_data=market_data
)
```

Optimizer 不负责：

```text
开盘成交
滑点
印花税
佣金
涨停成交失败
跌停卖不出
```

这些仍由 Backtest Engine 处理。

---

# 66. 实际交易流程

最终：

```text
T日收盘
│
├── 生成 Alpha
│
├── Risk Model
│
├── Candidate Pool
│
├── Optimizer
│
└── Target Portfolio
        │
        ▼
     T+1开盘
        │
        ├── 检查涨停
        ├── 检查停牌
        ├── 检查成交量
        ├── 计算实际成交
        └── 得到 Actual Portfolio
```

---

# 67. V1 暂不支持的功能

以下全部放到 V2/V3。

## V2

```text
EWMA Factor Covariance
动态风险权重
风险预算
交易成本显式建模
Market Impact
Benchmark-relative Optimization
```

## V3

```text
CVaR
Black-Litterman
Multi-period Optimization
GARCH
非线性市场冲击模型
```

---

# 68. 推荐开发顺序

不要一次全部实现。

按照下面顺序：

```text
Step 1
Alpha → Candidate Pool
        ↓
Step 2
X/F/D → Portfolio Risk
        ↓
Step 3
QP Optimizer
        ↓
Step 4
Weight Constraint
        ↓
Step 5
Industry Constraint
        ↓
Step 6
Style Constraint
        ↓
Step 7
Turnover Penalty
        ↓
Step 8
Risk Attribution
        ↓
Step 9
Backtest Integration
        ↓
Step 10
Walk-Forward Validation
```

---

# 69. 第一阶段 MVP

第一版 MVP 只实现：

```text
Input:
    alpha
    X
    F
    D
    previous_weights

Objective:
    alpha
    -
    lambda * risk

Constraints:
    sum(w) = 1
    w >= 0
    w <= 3%

Output:
    weights
    risk
    alpha
```

先不要加入：

```text
Industry
Style
Turnover
Transaction Cost
```

原因：

> 先证明 QP Optimizer 能正确工作。

---

# 70. MVP 第二阶段

加入：

```text
Industry <= 20%
```

验证：

> Optimizer 是否能够降低行业集中度，同时保持 Alpha。

---

# 71. MVP 第三阶段

加入：

```text
8 Style Exposure Constraints
```

然后观察：

```text
Factor Exposure
```

是否明显下降。

---

# 72. MVP 第四阶段

加入：

```text
Turnover Penalty
```

目标：

\[
\gamma\|w-w_{prev}\|_1
\]

验证：

```text
Turnover ↓
Transaction Cost ↓
Sharpe 是否 ↑
```

---

# 73. 最终 V1

最终形成：

\[
\boxed{
\max_w
\alpha^Tw
-\lambda w^T\Sigma w
-\gamma\|w-w_{prev}\|_1
}
\]

subject to：

\[
\sum_iw_i=1
\]

\[
0\le w_i\le3\%
\]

\[
Industry_j\le20\%
\]

\[
L_k\le X_k^Tw\le U_k
\]

最终：

```text
Alpha
+
Risk
+
Industry
+
Style
+
Turnover
```

全部纳入统一 Portfolio Construction。

---

# 74. 最终系统架构

```text
                         ┌─────────────────────┐
                         │   Market Data       │
                         └──────────┬──────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  │                                   │
                  ▼                                   ▼
          ┌───────────────┐                   ┌───────────────┐
          │ Alpha Model   │                   │  Risk Model   │
          │               │                   │               │
          │ XGBoost       │                   │ X             │
          │ Prediction    │                   │ F             │
          │               │                   │ D             │
          └───────┬───────┘                   └───────┬───────┘
                  │                                   │
                  │ Alpha                             │ Risk
                  ▼                                   ▼
          ┌────────────────────────────────────────────────┐
          │             Portfolio Optimizer                │
          │                                                │
          │  Objective:                                    │
          │  Alpha - λRisk - γTurnover                    │
          │                                                │
          │  Constraints:                                  │
          │  • Long-only                                  │
          │  • Fully invested                             │
          │  • Stock weight                               │
          │  • Industry                                   │
          │  • Style                                      │
          │  • Beta                                       │
          └──────────────────────┬─────────────────────────┘
                                 │
                                 ▼
                       ┌──────────────────┐
                       │ Target Portfolio │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Execution Layer  │
                       │                  │
                       │ Limit Up/Down    │
                       │ Suspension       │
                       │ Slippage         │
                       │ Commission       │
                       │ Stamp Tax        │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Actual Portfolio │
                       └────────┬─────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │ Backtest / Eval  │
                       └──────────────────┘
```

---

# 75. 最终结论

Portfolio Optimizer V1 的核心不是：

> “让 XGBoost 赚更多钱”。

而是：

\[
\boxed{
\text{把 XGBoost 的预测结果转换成风险可控的可交易组合}
}
\]

你的系统现在已经具备了非常适合进入这一阶段的条件：

```text
Alpha Model
    ↓
IC ≈ 0.16

Risk Model
    ↓
Pearson ≈ 0.38
Spearman ≈ 0.44
PSD = 100%
Q5/Q1 ≈ 1.84x

Portfolio Optimizer
    ↓
Alpha
+ Risk
+ Style
+ Industry
+ Turnover
    ↓
Target Portfolio
```

**第一阶段不要追求复杂。**

建议严格按照：

```text
MVP
 ↓
Alpha + Risk
 ↓
Weight Constraints
 ↓
Industry
 ↓
Style
 ↓
Turnover
 ↓
Risk Attribution
 ↓
Backtest
 ↓
Walk-Forward
```

逐层增加。

最终 Optimizer V1 的核心判断标准也只有一个：

> **在 Alpha Retention 可接受的情况下，是否能够显著降低组合风险和回撤，并提高 Sharpe/Calmar。**

如果这个结果成立，那么你的整个系统就从：

```text
单纯的 Alpha 选股模型
```

正式升级成：

```text
Alpha Model
+
Risk Model
+
Portfolio Construction
+
Execution
```

完整的量化投资组合系统。