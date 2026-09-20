# Portfolio Optimizer V1 代码设计文档

**版本**: V1.0
**设计日期**: 2026-08-10
**状态**: Implementation Design
**基于**: `Portfolio_Optimizer_v1_design.md` (High-Level Design)

---

## 1. 文档目标

本文档是 High-Level Design 的落地实现设计，定义：
- 模块划分与文件结构
- 每个类的接口、输入输出、核心逻辑
- 与现有 Alpha / Risk / Backtest 模块的对接方式
- 配置文件结构
- 数据格式约定
- 开发顺序与验收标准

---

## 2. 模块结构

```
src/optimizer/
├── __init__.py                  # 导出公共接口
├── portfolio_optimizer.py       # 统一入口门面类
├── alpha_processor.py           # Alpha 标准化（winsorize + z-score）
├── candidate_pool.py            # 候选股票池构建
├── risk_interface.py            # Risk Model 接口适配 + 组合风险计算
├── objective_builder.py         # QP 目标函数构建
├── constraint_builder.py        # QP 约束构建
├── qp_solver.py                 # QP 求解器封装（CVXPY + OSQP）
├── optimization_result.py       # 优化结果数据类
├── risk_attribution.py          # 风险归因计算
├── diagnostics.py               # 诊断信息收集
└── run_optimizer_backtest.py    # 回测入口脚本
```

**依赖方向**：
```
portfolio_optimizer.py
    ├── alpha_processor.py
    ├── candidate_pool.py
    ├── risk_interface.py
    ├── objective_builder.py
    ├── constraint_builder.py
    ├── qp_solver.py
    ├── risk_attribution.py
    └── diagnostics.py
```

所有模块只依赖 `optimization_result.py` 中的数据类做数据交换，不互相直接依赖内部实现。

---

## 3. 核心数据结构

### 3.1 OptimizationResult

文件：`src/optimizer/optimization_result.py`

```python
from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class OptimizationResult:
    """单次优化结果。"""

    # 核心输出
    weights: pd.Series          # 目标权重，index=ts_code, value=float
    expected_alpha: float       # α^T w
    portfolio_variance: float   # w^T Σ w
    portfolio_volatility: float # sqrt(w^T Σ w)  日度

    # 交易
    turnover: float             # 0.5 * Σ|w - w_prev|
    trade_list: pd.DataFrame    # ts_code, current_weight, target_weight, trade_weight

    # 风险暴露
    factor_exposure: pd.Series  # 风格+行业因子暴露，index=因子名
    style_exposure: pd.Series   # 仅风格因子
    industry_exposure: pd.Series # 仅行业因子

    # 风险归因
    factor_risk_total: float    # 因子风险（方差）
    specific_risk_total: float  # 特质风险（方差）
    factor_risk_contribution: pd.Series  # 各因子风险贡献占比
    specific_risk_contribution: float    # 特质风险贡献占比

    # 求解器状态
    solver_status: str          # optimal / infeasible / ...
    objective_value: float      # 目标函数值
    solve_time_ms: float        # 求解耗时

    # 诊断
    alpha_retention: float      # optimized_alpha / raw_alpha
    raw_alpha_portfolio: float  # 原始 Alpha 组合的 α^T w
    n_candidates: int           # 候选股票数
    n_holdings: int             # 持仓数（权重>0）
```

---

## 4. Alpha Processor

文件：`src/optimizer/alpha_processor.py`

### 4.1 类定义

```python
class AlphaProcessor:
    """Alpha 预测值的横截面标准化。"""

    def __init__(self, config: dict):
        self.winsorize_quantile = config.get("winsorize_quantile", 0.01)
        self.method = config.get("method", "zscore")  # zscore / rank

    def process(self, alpha: pd.Series) -> pd.Series:
        """
        对单日 Alpha 做横截面标准化。

        Parameters
        ----------
        alpha : pd.Series
            原始预测值，index=ts_code

        Returns
        -------
        pd.Series
            标准化后的 alpha score
        """
```

### 4.2 处理流程

```
raw alpha (pd.Series)
    ↓
drop NaN
    ↓
winsorize (1% 双侧)
    ↓
z-score 标准化
    ↓
alpha_score (mean≈0, std≈1)
```

### 4.3 配置

```json
{
  "alpha": {
    "method": "zscore",
    "winsorize": true,
    "winsorize_quantile": 0.01
  }
}
```

---

## 5. Candidate Pool

文件：`src/optimizer/candidate_pool.py`

### 5.1 类定义

```python
class CandidatePoolBuilder:
    """构建优化候选股票池。"""

    def __init__(self, config: dict):
        self.pool_size = config.get("pool_size", 100)
        self.min_avg_amount_20d = config.get("min_avg_amount_20d", 5000)  # 万元
        self.filter_st = config.get("filter_st", True)
        self.filter_suspend = config.get("filter_suspend", True)
        self.filter_new_stock_days = config.get("filter_new_stock_days", 60)

    def build(
        self,
        alpha: pd.Series,
        market_data: pd.DataFrame,   # 当日股票状态（ST/停牌/成交额等）
        current_holdings: pd.Series | None = None,
    ) -> pd.Index:
        """
        返回候选股票代码列表。

        流程：
        1. 基础过滤（ST/停牌/新股/流动性）
        2. 保留当前持仓（即使不在topN，也允许继续持有）
        3. 按 Alpha 排名取 top N
        """
```

### 5.2 关键设计决策

**当前持仓必须保留在候选池中**，否则 Optimizer 会以为这些股票不存在，导致无法计算正确的风险和换手。

### 5.3 配置

```json
{
  "candidate_pool": {
    "pool_size": 100,
    "min_avg_amount_20d": 5000,
    "filter_st": true,
    "filter_suspend": true,
    "filter_new_stock_days": 60
  }
}
```

---

## 6. Risk Interface

文件：`src/optimizer/risk_interface.py`

### 6.1 职责

对接 Risk Model V1，为 Optimizer 提供：
1. 当日因子暴露矩阵 X
2. 当日因子协方差矩阵 F
3. 当日特质风险 D（对角线）
4. 组合风险计算工具函数

### 6.2 类定义

```python
class RiskInterface:
    """Risk Model 接口，为 Optimizer 提供风险数据。"""

    def __init__(self, risk_model_config: dict):
        # 预加载因子名称列表等静态信息
        self.style_factors = [...]
        self.industry_factors = [...]
        self.all_factor_names = [...]

    def get_day_risk_data(
        self,
        trade_date: int,
        stock_codes: list[str],
    ) -> DayRiskData:
        """
        获取指定日期、指定股票的风险数据。

        Returns
        -------
        DayRiskData
            exposures: np.array (N, K)
            factor_cov: np.array (K, K)
            specific_variance: np.array (N,)
            factor_names: list[str]
        """

    @staticmethod
    def portfolio_variance(
        weights: np.ndarray,
        exposures: np.ndarray,
        factor_cov: np.ndarray,
        specific_variance: np.ndarray,
    ) -> float:
        """
        高效计算组合方差：z^T F z + Σ(w_i^2 D_i)
        其中 z = X^T w
        O(NK + K^2)，不需要构建 N×N Σ
        """

    @staticmethod
    def factor_exposure(
        weights: np.ndarray,
        exposures: np.ndarray,
        factor_names: list[str],
    ) -> pd.Series:
        """计算组合因子暴露：X^T w"""
```

### 6.3 DayRiskData 数据类

```python
@dataclass
class DayRiskData:
    trade_date: int
    stock_codes: list[str]
    exposures: np.ndarray          # (N, K)
    factor_cov: np.ndarray         # (K, K)
    specific_variance: np.ndarray  # (N,)
    factor_names: list[str]        # K
    style_factor_idx: list[int]    # 风格因子在factor_names中的索引
    industry_factor_idx: list[int] # 行业因子索引
```

### 6.4 数据加载策略

Risk Model 输出按年存放在 `outputs/risk_model/v1/year=YYYY/`。
-  exposures 和 specific_risk 用 parquet，按天切片
-  factor_cov 用 .npy，按文件名加载

为了性能，内部做缓存：
-  exposures 按年加载到内存
-  factor_cov 按需加载，LRU 缓存最近 252 天

---

## 7. QP 问题构建与求解

### 7.1 数学形式

标准 QP 形式：

```
minimize    (1/2) x^T P x + q^T x
subject to  G x ≤ h
            A x = b
            lb ≤ x ≤ ub
```

我们的问题：

```
maximize    α^T w - λ w^T Σ w - γ ||w - w_prev||_1
≡ minimize  λ w^T Σ w + γ ||w - w_prev||_1 - α^T w
```

Turnover penalty 的 L1 范数需要引入辅助变量 t：

```
t_i ≥ w_i - w_prev_i
t_i ≥ w_prev_i - w_i
```

目标变为：

```
minimize  λ w^T Σ w + γ Σ t_i - α^T w
```

### 7.2 Objective Builder

文件：`src/optimizer/objective_builder.py`

```python
class ObjectiveBuilder:
    """构建 QP 目标函数的 P, q 矩阵。"""

    def build(
        self,
        alpha: np.ndarray,          # (N,)
        exposures: np.ndarray,      # (N, K)
        factor_cov: np.ndarray,     # (K, K)
        specific_var: np.ndarray,   # (N,)
        risk_aversion: float,       # λ
        turnover_penalty: float,    # γ
        prev_weights: np.ndarray,   # (N,)
    ) -> QPObjective:
        """
        构建扩展变量 x = [w; t] 的目标函数。

        Returns
        -------
        QPObjective
            P: (N+N) x (N+N)  二次项矩阵
            q: (N+N,)        线性项
            n_variables: N + N  （w + t）
        """
```

**P 矩阵结构**：
- 左上 N×N：`2λ * (X F X^T + diag(D))` — 风险二次项
- 其他位置：0（turnover penalty 是线性的）

**q 向量结构**：
- 前 N 个：`-α` — Alpha 线性项
- 后 N 个：`γ` — turnover penalty

### 7.3 Constraint Builder

文件：`src/optimizer/constraint_builder.py`

```python
class ConstraintBuilder:
    """构建 QP 约束矩阵 G, h, A, b, lb, ub。"""

    def build(
        self,
        n_stocks: int,
        exposures: np.ndarray,
        industry_matrix: np.ndarray,  # (N, n_industries)  0/1
        prev_weights: np.ndarray,
        config: dict,
    ) -> QPConstraints:
        """
        Returns
        -------
        QPConstraints
            G: (m, 2N)  不等式约束矩阵
            h: (m,)
            A: (p, 2N)  等式约束矩阵
            b: (p,)
            lb: (2N,)   变量下界
            ub: (2N,)   变量上界
        """
```

**约束清单**：

| 约束 | 类型 | 维度 | 说明 |
|------|------|------|------|
| sum(w) = 1 | 等式 | 1 | 满仓 |
| w_i ≥ 0 | 边界 | N | long-only |
| w_i ≤ w_max | 边界 | N | 个股权重上限 |
| t_i ≥ w_i - w_prev_i | 不等式 | N | turnover 辅助变量 |
| t_i ≥ w_prev_i - w_i | 不等式 | N | turnover 辅助变量 |
| t_i ≥ 0 | 边界 | N | turnover 非负 |
| G_ind^T w ≤ ind_max | 不等式 | n_industries | 行业上限 |
| style_lb ≤ X_style^T w ≤ style_ub | 不等式 | 2 × n_styles | 风格暴露上下限 |

### 7.4 QP Solver

文件：`src/optimizer/qp_solver.py`

```python
class QPSolver:
    """QP 求解器封装。"""

    def __init__(self, config: dict):
        self.solver = config.get("solver", "osqp")
        self.max_iter = config.get("max_iter", 4000)
        self.eps_abs = config.get("eps_abs", 1e-6)
        self.eps_rel = config.get("eps_rel", 1e-6)
        self.warm_start = config.get("warm_start", True)

    def solve(self, objective: QPObjective, constraints: QPConstraints) -> QPSolution:
        """
        求解 QP 问题。

        使用 CVXPY 建模层 + OSQP 求解器。
        这样后续可以无缝切换求解器。
        """
```

**为什么用 CVXPY 而不是直接用 OSQP**：
1. 约束构建更清晰，不易出错
2. 自动处理矩阵维度对齐
3. 切换求解器只需改一行
4. 支持 DCP 规则检查，提前发现问题

**性能考虑**：
- N=100~300 只股票，K=35 因子
- 变量数 = 2N ≈ 200~600
- 约束数 ≈ 2N + n_industries + 2×n_styles + 1 ≈ 250~700
- OSQP 求解时间应 < 100ms/次

---

## 8. Portfolio Optimizer 门面类

文件：`src/optimizer/portfolio_optimizer.py`

### 8.1 类定义

```python
class PortfolioOptimizer:
    """组合优化器统一入口。"""

    def __init__(self, config: dict, risk_interface: RiskInterface):
        self.config = config
        self.risk_interface = risk_interface
        self.alpha_processor = AlphaProcessor(config.get("alpha", {}))
        self.candidate_builder = CandidatePoolBuilder(config.get("candidate_pool", {}))
        self.objective_builder = ObjectiveBuilder()
        self.constraint_builder = ConstraintBuilder()
        self.solver = QPSolver(config.get("solver", {}))
        self.attribution = RiskAttribution()

    def optimize(
        self,
        trade_date: int,
        alpha_raw: pd.Series,         # 原始 Alpha 预测
        current_weights: pd.Series,   # 当前持仓权重
        market_data: pd.DataFrame,    # 当日市场数据（ST/停牌/成交额等）
    ) -> OptimizationResult:
        """
        执行一次完整优化。

        流程：
        1. Alpha 标准化
        2. 构建候选池
        3. 获取风险数据（X, F, D）
        4. 构建 QP 问题
        5. 求解
        6. 计算风险归因
        7. 收集诊断信息
        """
```

### 8.2 主流程伪代码

```python
def optimize(self, trade_date, alpha_raw, current_weights, market_data):
    # Step 1: Alpha 标准化
    alpha_std = self.alpha_processor.process(alpha_raw)

    # Step 2: 候选池
    candidates = self.candidate_builder.build(alpha_std, market_data, current_weights)
    alpha = alpha_std.loc[candidates]
    w_prev = current_weights.reindex(candidates, fill_value=0.0).values

    # Step 3: 风险数据
    risk_data = self.risk_interface.get_day_risk_data(trade_date, candidates.tolist())
    X = risk_data.exposures
    F = risk_data.factor_cov
    D = risk_data.specific_variance

    # Step 4: 构建 QP
    obj = self.objective_builder.build(
        alpha=alpha.values, exposures=X, factor_cov=F,
        specific_var=D,
        risk_aversion=self.config["objective"]["risk_aversion"],
        turnover_penalty=self.config["objective"]["turnover_penalty"],
        prev_weights=w_prev,
    )
    cons = self.constraint_builder.build(
        n_stocks=len(candidates), exposures=X,
        industry_matrix=X[:, industry_idx],
        prev_weights=w_prev,
        config=self.config["constraints"],
    )

    # Step 5: 求解
    sol = self.solver.solve(obj, cons)
    w_opt = sol.x[:len(candidates)]
    weights = pd.Series(w_opt, index=candidates, name="weight")

    # Step 6: 风险归因
    attribution = self.attribution.compute(weights, X, F, D, risk_data.factor_names)

    # Step 7: 组装结果
    return OptimizationResult(
        weights=weights,
        expected_alpha=float(alpha.values @ w_opt),
        portfolio_variance=float(w_opt @ (X @ F @ X.T + np.diag(D)) @ w_opt),
        portfolio_volatility=float(np.sqrt(max(...))),
        turnover=float(0.5 * np.sum(np.abs(w_opt - w_prev)))),
        ...
    )
```

---

## 9. Risk Attribution

文件：`src/optimizer/risk_attribution.py`

```python
class RiskAttribution:
    """风险归因计算。"""

    def compute(
        self,
        weights: np.ndarray,
        exposures: np.ndarray,
        factor_cov: np.ndarray,
        specific_var: np.ndarray,
        factor_names: list[str],
    ) -> RiskAttributionResult:
        """
        计算：
        - 总风险（方差）
        - 因子风险 vs 特质风险
        - 各因子风险贡献（z_k * (Fz)_k / total_var）
        - 特质风险贡献（Σ w_i^2 D_i / total_var）
        """
```

输出：
```python
@dataclass
class RiskAttributionResult:
    total_variance: float
    factor_variance: float
    specific_variance: float
    factor_contribution_pct: pd.Series  # 每个因子的风险贡献%
    specific_contribution_pct: float
```

---

## 10. Diagnostics

文件：`src/optimizer/diagnostics.py`

```python
class OptimizationDiagnostics:
    """收集每次优化的诊断信息。"""

    def record(self, trade_date: int, result: OptimizationResult) -> None:
        ...

    def to_dataframe(self) -> pd.DataFrame:
        """导出为 DataFrame，方便回测分析。"""
```

记录字段（每日一行）：
- trade_date
- solver_status
- objective_value
- solve_time_ms
- expected_alpha
- portfolio_volatility
- turnover
- alpha_retention
- n_holdings
- n_candidates
- factor_risk_pct
- specific_risk_pct
- 各风格因子暴露
- 最大行业权重
- 最大个股权重

---

## 11. 配置文件

文件：`conf/optimizer/optimizer_v1.json`

```json
{
  "version": "optimizer_v1",

  "alpha": {
    "method": "zscore",
    "winsorize": true,
    "winsorize_quantile": 0.01
  },

  "candidate_pool": {
    "pool_size": 100,
    "min_avg_amount_20d": 5000,
    "filter_st": true,
    "filter_suspend": true,
    "filter_new_stock_days": 60
  },

  "objective": {
    "risk_aversion": 0.1,
    "turnover_penalty": 0.01
  },

  "constraints": {
    "long_only": true,
    "fully_invested": true,
    "max_weight": 0.03,

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
  },

  "solver": {
    "backend": "cvxpy",
    "solver": "OSQP",
    "max_iter": 4000,
    "eps_abs": 1e-6,
    "eps_rel": 1e-6,
    "warm_start": true
  },

  "risk_model": {
    "config_path": "${PROJECT_ROOT}/conf/risk_model/risk_model_v1.json",
    "output_dir": "${PROJECT_ROOT}/outputs/risk_model/v1"
  },

  "output": {
    "base_dir": "${PROJECT_ROOT}/outputs/optimizer/v1"
  }
}
```

---

## 12. 回测入口脚本

文件：`src/optimizer/run_optimizer_backtest.py`

### 12.1 职责

将 Optimizer 接入回测流程，跑完整的 Walk-Forward 回测。

### 12.2 输入

- Alpha 预测结果（walk-forward stitched predictions）
- Risk Model 输出（X/F/D，按年存放）
- 回测配置（调仓频率、成本等）
- Optimizer 配置

### 12.3 流程

```
加载预测数据 + Risk Model 数据
    ↓
初始化持仓 = 空
    ↓
遍历每个调仓日：
    ├── 获取当日 Alpha
    ├── 获取当日市场状态（ST/停牌/涨跌停）
    ├── Optimizer.optimize() → target_weights
    ├── 执行层：处理涨跌停/停牌 → actual_weights
    ├── 记录组合净值
    ├── 记录诊断信息
    └── current_weights = actual_weights
    ↓
输出回测结果 + 诊断报告
```

### 12.4 与现有 Backtester 的关系

**V1 先不改造 SimpleBacktester**，而是写一个独立的 `run_optimizer_backtest.py` 脚本。

原因：
1. SimpleBacktester 是 topk + n_drop 的固定逻辑，和 Optimizer 范式不同
2. 独立脚本可以快速迭代
3. 验证有效后再考虑是否重构统一

执行层的逻辑（涨跌停、停牌、滑点、成本）复用 `simple_backtester.py` 中的工具函数。

---

## 13. 输出文件结构

```
outputs/optimizer/v1/{experiment_name}/
├── portfolio.parquet              # 每日持仓
│   # trade_date, ts_code, alpha, rank, current_weight,
│   # target_weight, actual_weight, trade_weight
├── diagnostics.parquet            # 每日优化诊断
│   # trade_date, solver_status, objective_value, expected_alpha,
│   # portfolio_vol, turnover, alpha_retention, n_holdings, ...
├── factor_exposure.parquet        # 每日因子暴露
│   # trade_date, SIZE, BETA, ..., industry_*, ...
├── risk_attribution.parquet       # 每日风险归因
│   # trade_date, total_risk, factor_risk, specific_risk,
│   # SIZE_pct, BETA_pct, ...
├── backtest_results.json          # 回测绩效指标
└── config.json                    # 使用的配置快照
```

---

## 14. 依赖清单

### 新增依赖

| 包 | 用途 | 版本建议 |
|----|------|---------|
| cvxpy | QP 建模 | >= 1.4 |
| osqp | QP 求解器 | >= 0.6 |

CVXPY 会自动拉取 OSQP 作为依赖。

### 验证

```bash
pip install cvxpy
python -c "import cvxpy; print(cvxpy.installed_solvers())"
```

应包含 `OSQP`。

---

## 15. 开发顺序与验收标准

### Phase 1：MVP — Alpha + Risk + 基本约束

**实现文件**：
- `optimization_result.py`
- `alpha_processor.py`
- `candidate_pool.py`
- `risk_interface.py`
- `objective_builder.py`
- `constraint_builder.py`（仅 sum=1, long-only, weight cap）
- `qp_solver.py`
- `portfolio_optimizer.py`

**验收**：
- [ ] 单元测试：给定简单 X/F/D，手算验证最优解
- [ ] 约束满足：sum(w)=1, 0≤w≤3%
- [ ] 风险计算正确：手算验证 portfolio variance
- [ ] 求解时间 < 100ms（N=100）

### Phase 2：行业约束 + 风格约束

**实现文件**：
- `constraint_builder.py` 增加行业和风格约束
- 配置文件增加约束参数

**验收**：
- [ ] 行业权重 ≤ 20%（误差 < 1e-4）
- [ ] 风格暴露在设定范围内
- [ ] 约束不可行时有明确报错

### Phase 3：Turnover Penalty

**实现文件**：
- `objective_builder.py` 增加 L1 turnover
- `constraint_builder.py` 增加辅助变量约束

**验收**：
- [ ] 增大 γ → turnover 下降
- [ ] 手算验证 turnover 计算正确

### Phase 4：风险归因 + 诊断

**实现文件**：
- `risk_attribution.py`
- `diagnostics.py`

**验收**：
- [ ] 因子风险贡献 + 特质风险贡献 = 100%（误差 < 0.1%）
- [ ] 诊断信息完整

### Phase 5：回测集成

**实现文件**：
- `run_optimizer_backtest.py`

**验收**：
- [ ] 2013-2025 完整回测跑通
- [ ] 输出所有结果文件
- [ ] 与 Raw Alpha Top30 对比有意义

### Phase 6：网格搜索 + 实验

**实现**：
- 网格搜索脚本
- 结果对比

**验收**：
- [ ] λ 网格：风险-收益前沿曲线
- [ ] candidate size 网格：Alpha retention 曲线
- [ ] 风格约束消融实验

---

## 16. 测试计划

### 单元测试

文件：`test/test_optimizer.py`

| 测试用例 | 验证内容 |
|---------|---------|
| test_alpha_processor_zscore | 标准化后均值≈0，std≈1 |
| test_candidate_pool_filter | ST/停牌/流动性过滤正确 |
| test_candidate_pool_keep_holdings | 当前持仓保留在候选池 |
| test_portfolio_variance | 手算验证组合方差公式 |
| test_objective_builder_p_matrix | P 矩阵维度和对称性 |
| test_constraint_builder_sum_one | sum(w)=1 约束正确 |
| test_constraint_builder_long_only | w≥0 约束正确 |
| test_constraint_builder_weight_cap | w≤3% 约束正确 |
| test_constraint_builder_industry | 行业约束正确 |
| test_constraint_builder_style | 风格约束正确 |
| test_qp_solver_simple | 简单问题求解正确 |
| test_qp_solver_infeasible | 不可行问题正确识别 |
| test_optimizer_end_to_end | 端到端优化结果合理 |
| test_risk_attribution_sum | 风险贡献加总=100% |
| test_turnover_calculation | 换手率计算正确 |

### 集成测试

- 用 2023 年 100 只股票的数据跑完整优化流程
- 验证约束全部满足
- 验证风险归因合理

---

## 17. 关键设计决策记录

### 17.1 为什么用 CVXPY + OSQP 而不是直接 OSQP

- CVXPY 是建模层，OSQP 是求解器，解耦
- 约束构建更不容易出错
- 切换求解器（如 MOSEK、CLARABEL）只改一行
- 代价是一点点 overhead，但 N=100~300 的规模完全可以忽略

### 17.2 为什么用 Candidate Pool 而不是全市场优化

- Alpha Model 对尾部股票的预测质量低，引入噪声
- 全市场 5000+ 只股票优化会更慢
- 真实交易中也不会考虑全市场
- Candidate Pool 本身就是 Alpha Filter

### 17.3 为什么 Turnover 用 L1 Penalty 而不是硬约束

- L1 penalty 更平滑，优化更稳定
- 硬约束可能导致角点解和数值不稳定
- γ 参数可以连续调节，更容易找最优

### 17.4 Optimizer 与 Execution 的边界

Optimizer 输出 **target weights**，不考虑涨跌停/停牌/滑点等执行问题。
Execution Layer（Backtester）负责将 target 转换成 actual。
下一次优化的输入是 actual weights，而不是 target weights。

---

## 18. 与现有模块的对接点

### 对接 Alpha Model

- 输入：`output/walk_forward_optimized/predictions/*.parquet`
- 格式：trade_date, ts_code, pred_score
- 对接方式：按日期读取，取当日所有股票的预测值

### 对接 Risk Model

- 输入：`outputs/risk_model/v1/year=YYYY/`
- exposures.parquet → X
- factor_cov/YYYYMMDD.npy → F
- specific_risk.parquet → D
- 对接方式：`RiskInterface` 类封装加载逻辑

### 对接 Backtester

- 输出：target_weights（每日）
- 对接方式：`run_optimizer_backtest.py` 调用 SimpleBacktester 的执行层工具函数
- 执行层复用：涨跌停过滤、停牌过滤、滑点、佣金、最小交易单位

### 对接 Config 系统

- 新增 `load_optimizer_config()` 到 `src/utils/config.py`
- 支持 `${PROJECT_ROOT}` 占位符
- 支持 deep_merge CLI 覆盖

---

## 19. 性能预估

| 操作 | 规模 | 预估时间 |
|------|------|---------|
| 单次优化（N=100） | 200 变量, ~250 约束 | < 50ms |
| 单次优化（N=300） | 600 变量, ~700 约束 | < 200ms |
| 1年回测（5天调仓 × 50次） | - | < 1分钟 |
| 13年回测（2005-2025） | ~500 次优化 | < 10分钟 |

内存：
- Risk Model 数据按年加载：~200MB/年
- Optimizer 本身：< 100MB
- 总计：< 500MB

---

## 20. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| OSQP 求解不稳定 | 优化失败或结果异常 | 1. warm start 2. 增加 eps 3. 备用求解器 ECOS |
| 约束不可行 | 无法得到解 | 1. 检测 infeasible 状态 2. 自动放松约束重试 3. fallback 到等权 |
| Risk Model 数据缺失 | 无法优化 | 1. 检查数据完整性 2. 缺失日期跳过并告警 |
| 因子共线性导致 F 接近奇异 | 优化数值不稳定 | 1. Ledoit-Wolf 收缩已处理 2. 加微小 diagonal jitter |
| Alpha 尺度变化影响 λ | 不同窗口效果不一致 | Alpha 每日横截面标准化 |
