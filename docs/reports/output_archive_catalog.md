# 项目输出归档目录

> 生成日期：2026-09-20
> 总占用：output/ ≈ 90G，outputs/ 较小
> 说明：本文件记录 `output/` 和 `outputs/` 下所有实验目录的用途，便于后续清理或回溯。

---

## 一、网格搜索（Grid Search）输出

### 1.1 XGBoost 单模型网格搜索

| 目录 | 大小 | 说明 |
|------|------|------|
| `output/xgboost_grid_search_outputs/` | 99M | 初始网格搜索（phase0），108 组参数，label_rank_1d/3d/5d/10d |
| `output/xgboost_grid_search_outputs_phase2/` | 161M | Phase2 网格搜索，200 组参数，label_rank_10d，大模型（n_estimators 600-800） |
| `output/xgboost_grid_search_outputs_phase3/` | 241M | Phase3 随机搜索，200 组参数，label_rank_3d/5d，小模型（n_estimators 20-100） |
| `output/xgboost_grid_models_phase2/` | 30G | Phase2 网格搜索的完整模型文件（每个 grid 一个 .json 模型） |
| `output/xgboost_grid_models_phase3/` | 7.7G | Phase3 网格搜索的完整模型文件 |

### 1.2 Walk-Forward 网格搜索

| 目录 | 大小 | 说明 |
|------|------|------|
| `output/wf_grid_search/` | 38G | 滚动窗口网格搜索总目录 |
| `output/wf_grid_search/full/` | — | 全特征集 WF 网格搜索 |
| `output/wf_grid_search/industry_v2/` | — | 行业 v2 版本 WF 网格搜索 |
| `output/wf_grid_search/industry_v2_27model/` | — | 27 模型集成 WF 网格搜索 |
| `output/wf_grid_search/industry_v2_ensemble/` | — | 集成 WF 网格搜索 |
| `output/wf_grid_search/logs/` | — | 网格搜索日志 |

### 1.3 流动性网格搜索

| 目录 | 大小 | 说明 |
|------|------|------|
| `output/liquidity_grid_search/` | 2.7G | 流动性约束网格搜索（models + predictions + backtests） |
| `output/liquidity_grid_search_fixed/` | 小 | 流动性网格搜索修复版（仅结果汇总） |

### 1.4 验证汇总

| 目录 | 大小 | 说明 |
|------|------|------|
| `output/2026_validation/` | 小 | 2026 年验证结果汇总（200 组 grid 的 IC/回测指标） |
| `output/2026_validation_phase1/` | 小 | Phase1 验证汇总（108 组） |
| `output/2026_validation_phase3/` | 小 | Phase3 验证汇总（200 组） |
| `output/2026_backtest_grid0045/` | 小 | grid0045 的回测结果（最佳参数之一） |
| `output/2026_inference_grid0045/` | 小 | grid0045 的推理预测结果 |

---

## 二、Walk-Forward 训练输出

### 2.1 基础版本

| 目录 | 大小 | 说明 |
|------|------|------|
| `output/walk_forward/` | 622M | 基础 WF 训练（基准版本） |
| `output/walk_forward_expanded/` | 579M | 扩展特征集 WF 训练 |
| `output/walk_forward_v3_pruned/` | 448M | v3 剪枝特征 WF 训练 |
| `output/walk_forward_v4_expanded/` | 201M | v4 扩展 WF 训练 |
| `output/walk_forward_weighted/` | 446M | 加权 WF 训练 |
| `output/walk_forward_quantile/` | 432M | 分位数 WF 训练 |

### 2.2 约束/过滤变体

| 目录 | 大小 | 说明 |
|------|------|------|
| `output/walk_forward_no_st/` | 323M | 剔除 ST 股票的 WF 训练 |
| `output/walk_forward_top16/` | 318M | Top16 因子 WF 训练 |
| `output/walk_forward_topk_noleak/` | 316M | TopK 无泄露 WF 训练 |
| `output/walk_forward_industry_v2/` | 313M | 行业 v2 版本 WF 训练 |
| `output/walk_forward_liquidity/` | 299M | 流动性约束 WF 训练 |

### 2.3 增强 & 集成

| 目录 | 大小 | 说明 |
|------|------|------|
| `output/walk_forward_enhanced_alpha/` | 2.7G | 增强 alpha 因子 WF 训练（含 baseline 和 enhanced 对比） |
| `output/walk_forward_ensembles/` | 1.6G | 多模型集成 WF 预测 |
| `output/walk_forward_ensemble_10d20d/` | 143M | 10d+20d 双周期集成 |
| `output/walk_forward_ensemble_3h/` | 139M | 3 周期集成 |
| `output/walk_forward_multi_horizon/` | 82M | 多周期 WF 训练 |

### 2.4 优化后版本

| 目录 | 大小 | 说明 |
|------|------|------|
| `output/walk_forward_optimized/` | 769M | 优化后 WF 训练（含 grid_search 和 best 回测） |

---

## 三、投资组合优化器（Optimizer）输出

### 3.1 optimizer_v 系列（均值方差优化器）

命名规则：`optimizer_v{版本}_{pool大小}_{风险厌恶ra}_{平滑}_{跟踪误差tp}`

| 目录 | 大小 | 说明 |
|------|------|------|
| `output/optimizer_v1/` | 小 | v1 初版（baseline + 多组参数子目录） |
| `output/optimizer_v1_baseline/` | 小 | v1 基线 |
| `output/optimizer_v1_debug_nocost/` | 小 | v1 无成本调试 |
| `output/optimizer_v1_low_liquidity/` | 小 | v1 低流动性 |
| `output/optimizer_v1_low_ra/` | 小 | v1 低风险厌恶 |
| `output/optimizer_v1_ultra_low_ra/` | 小 | v1 极低风险厌恶 |
| `output/optimizer_v2_20day_rebal/` | 小 | v2 20 日调仓 |
| `output/optimizer_v3_indneutral_highpenalty/` | 小 | v3 行业中性高惩罚 |
| `output/optimizer_v4_best_params/` | 小 | v4 最佳参数 |
| `output/optimizer_v5_ra10/` | 小 | v5 ra=10 |
| `output/optimizer_v5_ra100/` | 小 | v5 ra=100（未完成，仅 config） |
| `output/optimizer_v5_ra500/` | 小 | v5 ra=500（未完成，仅 config） |
| `output/optimizer_v6_ra0.01/` | 小 | v6 ra=0.01 |
| `output/optimizer_v7_pool30_ra0.001/` | 小 | v7 pool=30, ra=0.001 |
| `output/optimizer_v8_pool30_ra0.001_smooth5_tp5/` | 小 | v8 + 平滑5 + 跟踪误差5 |
| `output/optimizer_v9_pool30_ra0.0005_smooth10_tp10/` | 小 | v9 ra=0.0005, smooth=10, tp=10 |
| `output/optimizer_v10_pool20_ra0.0003_smooth5_tp3/` | 小 | v10 pool=20 |
| `output/optimizer_v11_pool25_ra0.0001_smooth3_tp1/` | 小 | v11 pool=25, ra=0.0001 |
| `output/optimizer_v12_pool25_ra0.0001_notp_smooth3/` | 小 | v12 无跟踪误差约束 |
| `output/optimizer_v13_pool50_ra10_smooth5_tp5/` | 小 | v13 pool=50, ra=10 |
| `output/optimizer_v14_pool30_ra50_smooth3_tp2/` | 小 | v14 ra=50 |
| `output/optimizer_v15_pool30_ra0.001_smooth3_tp0.5/` | 小 | v15 tp=0.5 |
| `output/optimizer_v16_expanded_pool30_ra0.001_smooth3_tp0.5/` | 小 | v16 扩展特征 + pool30 |
| `output/optimizer_v17_expanded_pool25_ra0.0003_smooth5_tp2/` | 小 | v17 扩展特征 + pool25 |
| `output/optimizer_v18_expanded_pool25_ra0_tp0/` | 小 | v18 无风险厌恶无跟踪误差 |

### 3.2 rp_v 系列（风险平价 / Risk Parity）

命名规则：`rp_v{版本}_pool{大小}_tilt{alpha倾斜}_{其他约束}`

| 目录 | 大小 | 说明 |
|------|------|------|
| `output/rp_v1_pool30_tilt05_ind/` | 小 | RP v1，pool=30，alpha倾斜0.5，行业约束 |
| `output/rp_v2_pool50_tilt03_ind/` | 小 | RP v2，pool=50，倾斜0.3 |
| `output/rp_v3_pool25_tilt07_ind/` | 小 | RP v3，pool=25，倾斜0.7 |
| `output/rp_v4_pool20_tilt08_ind/` | 小 | RP v4，pool=20，倾斜0.8 |
| `output/rp_v5_pool30_tilt06_reb10/` | 小 | RP v5，倾斜0.6，10日调仓 |
| `output/rp_v6_pool25_tilt07_liq/` | 小 | RP v6，流动性约束 |
| `output/rp_v7_pool30_tilt05_tv15/` | 小 | RP v7，目标波动15% |
| `output/rp_v8_pool30_tilt05_rank/` | 小 | RP v8，rank 加权 |
| `output/rp_v9_pool15_tilt09_ind/` | 小 | RP v9，pool=15，高倾斜0.9 |
| `output/rp_v10_pool30_tilt05_ma20/` | 小 | RP v10，MA20 熊市过滤 |
| `output/rp_v11_pool25_tilt06_ma30/` | 小 | RP v11，MA30 过滤 |
| `output/rp_v12_pool30_tilt05_nosmooth/` | 小 | RP v12，无平滑 |
| `output/rp_v13_pool30_tilt05_tv12/` | 小 | RP v13，目标波动12% |
| `output/rp_v14_pool25_tilt065_ma40/` | 小 | RP v14，MA40 过滤 |
| `output/rp_v15_pool30_tilt05_volscaled/` | 小 | RP v15，波动率缩放 |
| `output/rp_v16_pool25_tilt06_noma/` | 小 | RP v16，无 MA 过滤 |
| `output/rp_v17_pool30_tilt05_reb20/` | 小 | RP v17，20日调仓 |
| `output/rp_v18_pool30_tilt05_liq5k/` | 小 | RP v18，5kw 流动性门槛 |
| `output/rp_v19_pool30_tilt05_style_neutral/` | 小 | RP v19，风格中性（未完成，仅 config） |
| `output/rp_v20_pool30_tilt05_cap015/` | 小 | RP v20，个股权重上限15%（未完成） |

### 3.3 优化器回测 & Sweep

| 目录 | 大小 | 说明 |
|------|------|------|
| `output/optimizer_backtest/` | 785M | 优化器回测结果（27 模型集成 alpha，多组参数） |
| `output/optimizer_sweep/` | 55M | 优化器参数 sweep（industry_v2_L32） |
| `output/portfolio_optimization_test/` | 小 | 投资组合优化测试 |

---

## 四、回测（Backtest）输出

### 4.1 Simple Backtest

| 目录 | 大小 | 说明 |
|------|------|------|
| `output/simple_backtest/` | 24M | 基础简单回测（多组参数子目录） |
| `output/simple_backtest_baseline_top25/` | 小 | baseline top25 回测 |
| `output/simple_backtest_baseline_top25_full/` | 小 | baseline top25 全周期回测 |
| `output/simple_backtest_sweep/` | 小 | 简单回测参数 sweep |
| `output/simple_backtest_top100_ma50_vol/` | 21M | top100 + MA50 + 波动率加权 |
| `output/simple_backtest_top25_ma50_industry/` | 小 | top25 + MA50 + 行业约束 |

### 4.2 最终优化版本

| 目录 | 大小 | 说明 |
|------|------|------|
| `output/final_optimized/` | 小 | 最终优化 v1（趋势策略） |
| `output/final_optimized_v2/` | 19M | 最终优化 v2（行业中性） |
| `output/ind_neutral_maxind8_slip01/` | 小 | 行业中性 + 最大行业权重8% + 滑点0.1% |
| `output/industry_constraint_backtest/` | 小 | 行业约束回测对比 |
| `output/preparatory_backtest_full_baseline/` | 小 | 预备回测全量基线 |
| `output/test_debug/` | 小 | 调试用回测 |

### 4.3 Phase2 优化

| 目录 | 大小 | 说明 |
|------|------|------|
| `output/phase2_optimization/` | 小 | Phase2 优化回测（baseline + industry_neutral） |

---

## 五、IC 分析 & 因子研究

| 目录 | 大小 | 说明 |
|------|------|------|
| `output/factor_ic_decay_analysis/` | 小 | 因子 IC 衰减分析 |
| `output/enhanced_alpha_ic_2023_2025/` | 小 | 增强 alpha 因子 IC（2023-2025） |
| `output/enhanced_alpha_ic_2023_2025_v3/` | 小 | 增强 alpha IC v3 |
| `output/ic_cross_universe_grid0188/` | 小 | 跨股票池 IC 分析（沪深300/500/800等） |
| `output/ic_decile_analysis_grid0188/` | 小 | 十分位 IC 分析（按收益/波动率分组） |
| `output/cost_ratio_analysis/` | 小 | 成本比率分析 |
| `output/model_diagnostic_grid0188/` | 小 | grid0188 模型深度验证（行业中性IC / 市场分层 / 多头诊断） |

---

## 六、风险模型（Risk Model）

### 6.1 output/risk_model

| 目录 | 大小 | 说明 |
|------|------|------|
| `output/risk_model/v1/` | 1.2G | 风险模型 v1（含 exposures, factor_returns, specific_risk 等） |

### 6.2 outputs/risk_model（旧版/对比版）

| 目录 | 大小 | 说明 |
|------|------|------|
| `outputs/risk_model/v1/` | 小 | 风险模型 v1 验证汇总（yearly_stats, supplementary_validation） |
| `outputs/risk_model/v1_patched_industry/` | 小 | v1 + 行业修补 |
| `outputs/risk_model/v2_stockbasic_industry/` | 小 | v2 股票基本信息行业分类 |
| `outputs/risk_model/v3_sw2021_industry/` | 小 | v3 申万2021行业分类 |
| `outputs/risk_model/v4_sw2021_dynamic/` | 小 | v4 申万2021动态行业 |
| `outputs/risk_model/industry_compare_tdx_vs_sw.csv` | 小 | 通达信 vs 申万行业对比 |

---

## 七、其他实验输出（outputs/）

| 目录/文件 | 大小 | 说明 |
|-----------|------|------|
| `outputs/factor_ic_analysis_2015_2020.csv` | 小 | 因子 IC 分析（2015-2020） |
| `outputs/factor_ic_2019H2.csv` | 小 | 因子 IC（2019下半年） |
| `outputs/ic_recomputation_grid0188.csv` | 小 | grid0188 IC 重算结果 |
| `outputs/xgb_cross_sectional_dataset/pred_test.parquet` | 小 | XGB 横截面数据集预测 |
| `outputs/work_summary_260722-1749.md` | 小 | 工作总结（2026-07-22） |

### 7.1 行业反弹分析

| 目录 | 大小 | 说明 |
|------|------|------|
| `outputs/industry_bounce_analysis/` | 小 | 行业反弹策略分析（crash事件、反弹统计等） |

### 7.2 最差表现替换策略

| 目录 | 大小 | 说明 |
|------|------|------|
| `outputs/worst_performer_replacement/` ~ `worst_performer_replacement8/` | 小（共8组） | 最差表现替换策略回测（8 组参数实验） |

### 7.3 趋势策略回测

| 目录 | 大小 | 说明 |
|------|------|------|
| `outputs/trend_strategy_backtest/` | 小 | 趋势票量化策略回测（baseline + combo30 + grid search） |

### 7.4 空目录

| 目录 | 说明 |
|------|------|
| `outputs/walk_forward_expanded/` | 空目录 |
| `outputs/risk_model/v2/` | 空目录（仅子目录结构） |

---

## 八、未跟踪源码/配置文件（git ??）

以下文件不在 .gitignore 中，也未被 git 跟踪。需要决定是提交还是删除。

### 8.1 配置文件

| 文件 | 说明 | 建议 |
|------|------|------|
| `conf/parquet_loader_config_base.json` | ParquetLoader 基础配置（基线特征集） | 可提交 |
| `conf/risk_model/risk_model_v2.json` | 风险模型 v2 配置（申万 L1 行业 + 8 风格因子） | 可提交 |
| `conf/walk_forward_industry_v2_config.json` | 行业 v2 版本 WF 训练配置 | 可提交 |

### 8.2 脚本文件

| 文件 | 说明 | 建议 |
|------|------|------|
| `scripts/build_27model_ensemble.py` | 构建 27 模型 IC-IR 加权集成（逐年内存高效版） | 可提交 |
| `scripts/run_cross_universe_ic.py` | 跨股票池 IC 分析（沪深300/500/800/市值分层） | 可提交 |
| `scripts/run_decile_ic_analysis.py` | 十分位 IC 分析（按收益/波动率分组） | 可提交 |
| `scripts/run_longshort_backtest.py` | 多空对冲回测（50%多/50%空） | 可提交 |
| `scripts/run_optimizer_L32_sweep.py` | 优化器 L32 正交 sweep 运行器 | 可提交 |

### 8.3 源码文件

| 文件 | 说明 | 建议 |
|------|------|------|
| `src/utils/dataset_fetcher/index_constituent_fetcher.py` | 指数成分股获取（Tushare index_weight，前向填充） | 可提交 |
| `src/utils/dataset_fetcher/index_daily_fetcher.py` | 指数日线数据获取（Tushare index_daily） | 可提交 |

### 8.4 文档文件

| 文件 | 说明 | 建议 |
|------|------|------|
| `docs/model_ic_validation_report_20260920.md` | 模型 IC 验证报告（2026-09-20） | 可提交 |
| `docs/optimizer_industry_v2_L32_sweep_plan.md` | 优化器行业 v2 L32 sweep 计划 | 可提交 |
| `docs/optimizer_industry_v2_L32_sweep_report.md` | 优化器行业 v2 L32 sweep 报告 | 可提交 |

---

## 九、清理优先级建议

### 高价值（建议保留）
- `output/walk_forward_optimized/` — 最终优化 WF 模型
- `output/walk_forward_enhanced_alpha/` — 增强 alpha WF
- `output/walk_forward_ensembles/` — 集成预测
- `output/optimizer_backtest/` — 优化器回测（最新）
- `output/final_optimized_v2/` — 最终优化版本
- `output/risk_model/v1/` — 风险模型（最新完整版本）
- `output/2026_validation*/` — 验证汇总（仅汇总文件小）

### 中价值（按需保留）
- 各 walk_forward 变体（top16, no_st, liquidity 等）
- optimizer_v15~v18, rp_v10~v18（较新版本）
- IC 分析目录

### 低价值（可清理，释放 ~75G）
- `output/wf_grid_search/`（38G）— 网格搜索中间产物，汇总已在 validation 中
- `output/xgboost_grid_models_phase2/`（30G）— phase2 模型文件
- `output/xgboost_grid_models_phase3/`（7.7G）— phase3 模型文件
- `output/liquidity_grid_search/`（2.7G）— 流动性网格搜索
- `output/xgboost_grid_search_outputs_phase2/`（161M）
- `output/xgboost_grid_search_outputs_phase3/`（241M）
- `outputs/worst_performer_replacement1~8/` — 旧实验
- `outputs/trend_strategy_backtest/` — 旧趋势策略实验
- `outputs/industry_bounce_analysis/` — 旧行业反弹分析
- 旧版 optimizer_v1~v14、rp_v1~v9（早期探索版本）
- 旧版 walk_forward 变体（v3_pruned, v4_expanded, weighted, quantile 等）
