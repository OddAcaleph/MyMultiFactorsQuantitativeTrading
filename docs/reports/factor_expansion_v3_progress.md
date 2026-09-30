# 因子扩展 v3 + 风险平价策略优化进展报告

**日期**: 2026-09-11
**项目**: MyMultiFactorsQuantitativeTrading
**版本**: v0.3.0 (因子扩展 v3 + 风险平价)
**状态**: 进行中 — Walk-forward 训练 9/21 窗口完成

---

## 一、目标

提升 A 股多因子量化策略至 **Sharpe ≥ 2.0、年化收益 ≥ 20%**（2005-2025 全周期），采用风险模型 + 风险平价优化器框架。

**基线**（110 因子，风险平价 pool25 tilt0.7）：
- 年化收益：36.9%
- 夏普比率：1.70
- 最大回撤：-48.8%
- 卡玛比率：0.76

---

## 二、因子扩展方案

### 2.1 新增因子（9 个）

**价值因子（5 个）**：
- `ep_ratio` — 市盈率倒数（E/P）
- `bp_ratio` — 市净率倒数（B/P）
- `sp_ratio` — 市销率倒数（S/P）
- `cfp_ratio` — 市现率倒数（CF/P）
- `dividend_yield_approx` — 近似股息率

**Piotroski F-score（4 个）**：
- `piotroski_f_score` — 综合 F-score（9 分项）
- `f_profitability` — 盈利能力分
- `f_leverage_liquidity` — 杠杆与流动性分
- `f_efficiency` — 运营效率分

### 2.2 剔除因子（7 个）

基于 2019H2 IC 分析，剔除 ICIR < 0.05 的弱因子：
- `main_net_momentum_5d`
- `main_net_ema_slope_20`
- `main_net_consecutive`
- `max_drawdown_60`
- `ret_kurt_60`
- `momentum_12m_skip1m`
- `momentum_52w_high_break`

### 2.3 最终因子构成

| 类别 | 数量 |
|------|------|
| 动量类 | ~75 |
| 波动率类 | ~15 |
| 价值类 | 5（新增） |
| 质量/基本面 | ~8（含 Piotroski 4 个新增） |
| 量价/资金流 | ~5 |
| 行业类 | 5 |
| **合计** | **112** |

---

## 三、Walk-Forward 训练进度

### 3.1 配置
- 窗口数：21（滚动 5 年训练，1 年步进）
- 模型：XGBoost，300 棵树，max_depth=8，lr=0.03
- 标签：`label_rank_5d`（5 日排名收益）
- 因子数：112

### 3.2 进度
- ✅ wf00 (2005) — IC=0.1545, RankIC=0.1441
- ✅ wf01 (2006) — IC=0.1361, RankIC=0.1288
- ✅ wf02 (2007)
- ✅ wf03 (2008)
- ✅ wf04 (2009)
- ✅ wf05 (2010)
- ✅ wf06 (2011)
- ✅ wf07 (2012)
- ✅ wf08 (2013) — IC=0.1756, RankIC=0.1628
- 🔄 wf09-wf20 (2014-2025) — 进行中

---

## 四、初步对比（2005-2010 年，6 窗口）

### 4.1 IC 对比

| 指标 | v3 pruned (112f) | baseline (110f) | 变化 |
|------|-------------------|-----------------|------|
| 平均 RankIC | 0.1597 | 0.1615 | -0.0018 |
| ICIR | 1.71 | 1.73 | -0.02 |
| 正 IC 占比 | 95.7% | 96.1% | -0.4% |

### 4.2 风险平价回测对比（pool25, tilt0.7, MA50）

| 指标 | v3 pruned (112f) | baseline (110f) | 变化 |
|------|-------------------|-----------------|------|
| 年化收益 | 34.7% | 41.8% | -7.1% |
| 夏普比率 | 1.29 | 1.56 | -0.27 |
| 最大回撤 | -42.4% | -38.0% | -4.4% |
| 平均持仓 | 24.9 | 24.9 | 0 |
| 换手率 | 42.0% | 42.1% | -0.1% |

### 4.3 初步分析

**早期（2005-2010）v3 因子表现略差于 baseline**，可能原因：

1. **数据覆盖率问题**：早期上市公司少，价值因子和 Piotroski F-score 依赖的财务数据覆盖率低
2. **市场风格差异**：2005-2010 年 A 股以炒作为主，价值因子效果弱
3. **剔除因子的早期有效性**：被剔除的 7 个因子（如 52 周新高、动量 12m）在早期市场可能还有效

**关键要看全周期表现**——价值因子在 2016 年后的 A 股市场表现更强。等全部 21 个窗口跑完后做完整对比。

---

## 五、待完成工作

1. ⏳ 等待剩余 12 个窗口训练完成（预计 1-1.5 小时）
2. ⏳ 拼接全周期预测结果
3. ⏳ 运行风险平价回测（基准配置：pool25, tilt0.7, MA50）
4. ⏳ 与 baseline 全周期对比
5. ⏳ 参数优化（alpha_tilt、pool_size、MA 周期等），冲击 Sharpe 2.0

---

## 六、文件变更

### 新增
- `src/utils/features_generator/enhanced_alpha_feature_generator.py` — 新增价值和 Piotroski 因子生成方法
- `src/utils/cross_sectional_processor/enhanced_alpha_factors_cross_sectional_processor.py` — 新增 9 个因子的横截面处理
- `conf/experiments/parquet_loader_config_v3_pruned.json` — v3 pruned 数据加载配置（112 因子）
- `conf/experiments/walk_forward_v3_pruned_config.json` — v3 pruned 训练配置
- `scripts/run_v3_pruned_analysis.py` — 训练后分析脚本
- `data/features_data/enhanced_alpha_factors_v3/` — v3 原始因子数据
- `data/cross_sectional_processd_data/enhanced_alpha_factors_v3/` — v3 横截面处理后数据
- `output/walk_forward_v3_pruned/` — v3 pruned 训练输出

### 修改
- `src/utils/dataset_generator/parquet_loader.py` — 新增 v3 因子列定义

---

## 七、技术说明

### 训练问题排查
Walk-forward 训练最初使用 `nohup` 启动时静默失败，原因排查：
1. `--cpu` 参数不存在（正确方式：不指定 `--gpu` 即默认 CPU）
2. `spawn` 子进程模式在某些环境下可能有问题
3. 改用**进程内训练**（不使用子进程隔离）后稳定运行

### 内存使用
- 单窗口训练峰值：~6-10GB（后期窗口数据更多）
- 预测阶段：~1-2GB
- 系统内存：503GB（充足）
