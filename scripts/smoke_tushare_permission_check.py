"""Smoke test: 验证 Tushare 各特色数据接口权限。

逐个调用目标接口，判断当前 token 是否有权限访问。
不做大量数据拉取，仅验证接口可达性和权限。

用法:
    python scripts/smoke_tushare_permission_check.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import tushare as ts


# ---------------------------------------------------------------------------
# Tushare 初始化（与项目内 moneyflow_fetcher 一致）
# ---------------------------------------------------------------------------
DEFAULT_HTTP_URL = (
    os.environ.get("DEFAULT_HTTP_URL")
    or os.environ.get("TUSHARE_HTTP_URL")
    or "http://jiaoch.site"
)


def _load_token() -> str:
    token = os.environ.get("TUSHARE_TOKEN")
    if token:
        return token.strip()
    token_file = Path.home() / ".tushare_token"
    if token_file.exists():
        token = token_file.read_text(encoding="utf-8").strip()
        if token:
            return token
    raise ValueError("请通过环境变量 TUSHARE_TOKEN 或 ~/.tushare_token 提供 Tushare Token")


def init_pro():
    token = _load_token()
    ts.set_token(token)
    pro = ts.pro_api(token)
    if DEFAULT_HTTP_URL:
        pro._DataApi__http_url = DEFAULT_HTTP_URL
    pro._DataApi__token = token
    return pro


# ---------------------------------------------------------------------------
# 单接口测试
# ---------------------------------------------------------------------------
def check_interface(name: str, api_call, retries: int = 2, sleep_retry: float = 1.0) -> dict:
    """调用单个接口，返回权限检查结果。支持重试。"""
    result = {
        "name": name,
        "ok": False,
        "rows": 0,
        "error": None,
    }
    last_err = None
    for attempt in range(retries + 1):
        try:
            df = api_call()
            result["ok"] = True
            result["rows"] = len(df) if df is not None else 0
            result["columns"] = list(df.columns) if df is not None else []
            return result
        except Exception as e:
            last_err = str(e)[:300]
            if attempt < retries:
                time.sleep(sleep_retry)
    result["error"] = last_err
    return result


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Tushare 接口权限 Smoke 验证")
    print("=" * 70)

    pro = init_pro()
    print(f"HTTP URL: {DEFAULT_HTTP_URL}")
    print()

    # 选一个近期的交易日做测试（取最近的交易日）
    test_date = "20250630"
    test_ts_code = "000001.SZ"

    checks = [
        # --- 1. 业绩预告 ---
        {
            "category": "业绩预告",
            "tests": [
                ("forecast (单股)", lambda: pro.forecast(ts_code=test_ts_code, period="20241231")),
            ],
        },
        # --- 2. 分析师一致预期 / 研报覆盖 ---
        {
            "category": "分析师盈利预测 / 研报",
            "tests": [
                ("report_rc (卖方盈利预测)", lambda: pro.report_rc(ts_code=test_ts_code, start_date="20250101", end_date="20250630")),
            ],
        },
        # --- 3. 机构调研 ---
        # Tushare 无专门接口，跳过（仅做说明）
        # --- 4. 龙虎榜 ---
        {
            "category": "龙虎榜",
            "tests": [
                ("top_list (每日明细)", lambda: pro.top_list(trade_date=test_date)),
                ("top_inst (机构明细)", lambda: pro.top_inst(trade_date=test_date)),
            ],
        },
        # --- 5. 大宗交易 ---
        {
            "category": "大宗交易",
            "tests": [
                ("block_trade", lambda: pro.block_trade(trade_date=test_date)),
            ],
        },
        # --- 6. 回购 / 增减持 ---
        {
            "category": "回购 & 增减持",
            "tests": [
                ("repurchase (股票回购)", lambda: pro.repurchase(ann_date=test_date)),
                ("stk_holdertrade (股东增减持)", lambda: pro.stk_holdertrade(ann_date=test_date)),
            ],
        },
        # --- 7. 概念/主题板块（政策主题相关性） ---
        {
            "category": "概念/主题板块",
            "tests": [
                ("ths_index (同花顺概念指数)", lambda: pro.ths_index(type="N")),
                ("dc_index (东方财富概念板块)", lambda: pro.dc_index(trade_date=test_date, idx_type="概念板块")),
                ("dc_member (板块成分股)", lambda: pro.dc_member(ts_code="BK1184.DC", trade_date="20250102")),
                ("ths_hot (THS热榜)", lambda: pro.ths_hot(trade_date="20250315", market="热股", is_new="Y")),
            ],
        },
        # --- 8. 基础对照（确保 token 有效） ---
        {
            "category": "基础接口（对照）",
            "tests": [
                ("daily (日线行情)", lambda: pro.daily(ts_code=test_ts_code, start_date=test_date, end_date=test_date)),
                ("daily_basic (每日指标)", lambda: pro.daily_basic(ts_code=test_ts_code, start_date=test_date, end_date=test_date)),
                ("moneyflow (资金流向)", lambda: pro.moneyflow(ts_code=test_ts_code, start_date=test_date, end_date=test_date)),
            ],
        },
    ]

    summary = []
    for group in checks:
        category = group["category"]
        print(f"\n{'─' * 60}")
        print(f"【{category}】")
        print(f"{'─' * 60}")
        for name, fn in group["tests"]:
            print(f"  测试 {name} ...", end=" ", flush=True)
            r = check_interface(name, fn)
            if r["ok"]:
                print(f"✅ OK  (返回 {r['rows']} 行)")
                summary.append((category, name, True, r["rows"], None))
            else:
                print(f"❌ FAIL")
                print(f"     错误: {r['error']}")
                summary.append((category, name, False, 0, r["error"]))
            time.sleep(0.3)  # 避免触发限流

    # 汇总
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    total = len(summary)
    passed = sum(1 for _, _, ok, _, _ in summary if ok)
    failed = total - passed
    print(f"总计: {total} 个接口  |  ✅ 通过: {passed}  |  ❌ 失败: {failed}")
    print()

    if failed > 0:
        print("失败接口列表:")
        for cat, name, ok, rows, err in summary:
            if not ok:
                print(f"  [{cat}] {name}")
                print(f"      原因: {err}")
        print()

    # 机构调研说明
    print("ℹ️  机构调研：Tushare 无原生接口，需通过其他数据源（Wind/Choice/同花顺）补充。")
    print()


if __name__ == "__main__":
    main()
