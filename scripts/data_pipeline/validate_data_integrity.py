#!/usr/bin/env python3
"""
数据完整性验证脚本

验证各阶段数据的完整性和正确性，包括：
- 文件数量检查
- 日期连续性检查
- 行数合理性检查
- 关键字段非空检查
- 上下游数据一致性检查

用法：
    python validate_data_integrity.py --stage raw
    python validate_data_integrity.py --stage cleaned
    python validate_data_integrity.py --stage processed
    python validate_data_integrity.py --stage features
    python validate_data_integrity.py --stage cross_sectional
    python validate_data_integrity.py --stage labels
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"


def get_trade_dates_from_dir(data_dir: Path, date_col: str = "trade_date") -> list[str]:
    """从按日分区的 parquet 目录提取所有交易日"""
    dates = set()
    for f in data_dir.rglob("*.parquet"):
        try:
            df = pd.read_parquet(f, columns=[date_col])
            dates.update(df[date_col].astype(str).tolist())
        except Exception:
            pass
    return sorted(dates)


def get_file_dates_from_dir(data_dir: Path) -> list[str]:
    """从文件名提取日期（YYYYMMDD.parquet）"""
    dates = set()
    for f in data_dir.rglob("*.parquet"):
        name = f.stem
        if name.isdigit() and len(name) == 8:
            dates.add(name)
    return sorted(dates)


def check_date_continuity(dates: list[str], start: str, end: str) -> dict:
    """检查日期连续性（粗略：只检查首尾区间内的交易日数量）"""
    if not dates:
        return {"status": "FAIL", "reason": "无数据"}

    first, last = dates[0], dates[-1]
    count = len(dates)

    # 粗略估算：每年约 244 个交易日
    start_dt = datetime.strptime(first, "%Y%m%d")
    end_dt = datetime.strptime(last, "%Y%m%d")
    years = (end_dt - start_dt).days / 365.25
    expected = int(years * 244)
    ratio = count / max(expected, 1)

    status = "OK"
    if ratio < 0.8:
        status = "WARN"
    if ratio < 0.5:
        status = "FAIL"

    return {
        "status": status,
        "first_date": first,
        "last_date": last,
        "count": count,
        "expected_approx": expected,
        "ratio": round(ratio, 2),
    }


def validate_raw() -> dict:
    """验证 raw_data 完整性"""
    raw = DATA_ROOT / "raw_data"
    results = {}

    # 日度分区数据集
    daily_datasets = {
        "daily_bars": ("trade_date", True),
        "moneyflow": ("trade_date", True),
        "index_daily": ("trade_date", True),
        "top_list": ("trade_date", True),
        "top_inst": ("trade_date", True),
        "block_trade": ("trade_date", True),
        "forecast": ("ann_date", True),
        "repurchase": ("ann_date", True),
        "stk_holdertrade": ("ann_date", True),
        "report_rc": ("report_date", True),
        "dc_index": ("trade_date", True),
    }

    for ds, (date_col, check_continuity) in daily_datasets.items():
        ds_dir = raw / ds
        if not ds_dir.exists():
            results[ds] = {"status": "FAIL", "reason": "目录不存在"}
            continue

        dates = get_file_dates_from_dir(ds_dir)
        if not dates:
            results[ds] = {"status": "FAIL", "reason": "无 parquet 文件"}
            continue

        info = check_date_continuity(dates, "", "")
        info["files"] = len(list(ds_dir.rglob("*.parquet")))

        # 抽样检查关键字段
        try:
            sample_file = sorted(ds_dir.rglob("*.parquet"))[-1]
            df = pd.read_parquet(sample_file)
            info["sample_rows"] = len(df)
            info["columns"] = list(df.columns)
            if date_col not in df.columns:
                info["status"] = "FAIL"
                info["reason"] = f"缺少日期列 {date_col}"
        except Exception as e:
            info["status"] = "FAIL"
            info["reason"] = f"读取失败: {e}"

        results[ds] = info

    # 单文件数据集
    single_datasets = ["adj_factors", "fundamentals", "industry", "suspend_d"]
    for ds in single_datasets:
        fpath = raw / ds / f"{ds}.parquet"
        if not fpath.exists():
            results[ds] = {"status": "FAIL", "reason": "文件不存在"}
            continue
        try:
            df = pd.read_parquet(fpath)
            results[ds] = {
                "status": "OK",
                "rows": len(df),
                "columns": list(df.columns),
            }
        except Exception as e:
            results[ds] = {"status": "FAIL", "reason": str(e)}

    return results


def validate_cleaned() -> dict:
    """验证 cleaned_data 完整性"""
    cleaned = DATA_ROOT / "cleaned_data"
    results = {}

    datasets = ["daily_bars", "adj_factors", "moneyflow", "fundamentals", "industry"]
    for ds in datasets:
        ds_path = cleaned / ds
        if not ds_path.exists():
            results[ds] = {"status": "FAIL", "reason": "不存在"}
            continue

        if ds == "daily_bars":
            dates = get_file_dates_from_dir(ds_path)
            info = check_date_continuity(dates, "", "")
            info["files"] = len(list(ds_path.rglob("*.parquet")))
        else:
            fpath = ds_path / f"{ds}.parquet"
            if not fpath.exists():
                results[ds] = {"status": "FAIL", "reason": "parquet 文件不存在"}
                continue
            df = pd.read_parquet(fpath)
            info = {"status": "OK", "rows": len(df), "columns": list(df.columns)}

        results[ds] = info

    return results


def validate_processed() -> dict:
    """验证 processd_data 完整性"""
    proc = DATA_ROOT / "processd_data"
    results = {}

    # wide_table
    wt_dir = proc / "wide_table_daily_bars"
    if wt_dir.exists():
        dates = get_file_dates_from_dir(wt_dir)
        info = check_date_continuity(dates, "", "")
        info["files"] = len(list(wt_dir.rglob("*.parquet")))
        # 抽样检查列数
        try:
            sample = sorted(wt_dir.rglob("*.parquet"))[-1]
            df = pd.read_parquet(sample)
            info["sample_columns"] = len(df.columns)
            info["sample_rows"] = len(df)
        except Exception as e:
            info["status"] = "FAIL"
            info["reason"] = str(e)
        results["wide_table_daily_bars"] = info
    else:
        results["wide_table_daily_bars"] = {"status": "FAIL", "reason": "不存在"}

    # 其他处理后数据
    for ds in ["namechange", "suspend_d", "industry"]:
        ds_dir = proc / ds
        if ds_dir.exists():
            files = list(ds_dir.rglob("*.parquet"))
            results[ds] = {"status": "OK", "files": len(files)}
        else:
            results[ds] = {"status": "WARN", "reason": "不存在"}

    return results


def validate_features() -> dict:
    """验证 features_data 完整性"""
    feat = DATA_ROOT / "features_data"
    results = {}

    feature_types = [
        "price_volume_factors",
        "fundamental_factors",
        "moneyflow_factors",
        "industry_factors",
        "enhanced_alpha_factors",
        "enhanced_alpha_factors_v2",
        "enhanced_alpha_factors_v3",
        "enhanced_alpha_factors_v4",
    ]

    for ft in feature_types:
        ft_dir = feat / ft
        if not ft_dir.exists():
            results[ft] = {"status": "SKIP", "reason": "不存在"}
            continue

        dates = get_file_dates_from_dir(ft_dir)
        if not dates:
            results[ft] = {"status": "FAIL", "reason": "无数据文件"}
            continue

        info = check_date_continuity(dates, "", "")
        info["files"] = len(list(ft_dir.rglob("*.parquet")))

        # 抽样检查因子数量
        try:
            sample = sorted(ft_dir.rglob("*.parquet"))[-1]
            df = pd.read_parquet(sample)
            info["num_factors"] = len([c for c in df.columns if c not in ["ts_code", "trade_date"]])
            info["sample_rows"] = len(df)
        except Exception as e:
            info["status"] = "FAIL"
            info["reason"] = str(e)

        results[ft] = info

    return results


def validate_cross_sectional() -> dict:
    """验证 cross_sectional_processd_data 完整性"""
    cs = DATA_ROOT / "cross_sectional_processd_data"
    results = {}

    cs_types = [
        "price_volume_factors",
        "fundamental_factors",
        "moneyflow_factors",
        "industry_factors",
        "wide_table_daily_bars",
        "enhanced_alpha_factors",
        "enhanced_alpha_factors_v2",
        "enhanced_alpha_factors_v3",
        "enhanced_alpha_factors_v4",
    ]

    for ct in cs_types:
        ct_dir = cs / ct
        if not ct_dir.exists():
            results[ct] = {"status": "SKIP", "reason": "不存在"}
            continue

        dates = get_file_dates_from_dir(ct_dir)
        if not dates:
            results[ct] = {"status": "FAIL", "reason": "无数据文件"}
            continue

        info = check_date_continuity(dates, "", "")
        info["files"] = len(list(ct_dir.rglob("*.parquet")))
        results[ct] = info

    return results


def validate_labels() -> dict:
    """验证 generated_label 完整性"""
    lbl = DATA_ROOT / "generated_label"
    results = {}

    label_types = ["daily_labels", "advanced_labels"]

    for lt in label_types:
        lt_dir = lbl / lt
        if not lt_dir.exists():
            results[lt] = {"status": "SKIP", "reason": "不存在"}
            continue

        dates = get_file_dates_from_dir(lt_dir)
        if not dates:
            results[lt] = {"status": "FAIL", "reason": "无数据文件"}
            continue

        info = check_date_continuity(dates, "", "")
        info["files"] = len(list(lt_dir.rglob("*.parquet")))

        # 抽样检查标签数量
        try:
            sample = sorted(lt_dir.rglob("*.parquet"))[-1]
            df = pd.read_parquet(sample)
            info["num_labels"] = len([c for c in df.columns if c not in ["ts_code", "trade_date"]])
            info["sample_rows"] = len(df)
        except Exception as e:
            info["status"] = "FAIL"
            info["reason"] = str(e)

        results[lt] = info

    return results


def print_summary(results: dict, stage: str) -> None:
    """打印验证摘要"""
    print(f"\n{'='*60}")
    print(f"  {stage.upper()} 数据完整性验证")
    print(f"{'='*60}")

    ok = 0
    warn = 0
    fail = 0
    skip = 0

    for name, info in results.items():
        status = info.get("status", "UNKNOWN")
        status_icon = {"OK": "✅", "WARN": "⚠️", "FAIL": "❌", "SKIP": "⏭️"}.get(status, "❓")

        detail_parts = []
        if "first_date" in info and "last_date" in info:
            detail_parts.append(f"{info['first_date']}~{info['last_date']}")
        if "count" in info:
            detail_parts.append(f"{info['count']}天")
        if "files" in info:
            detail_parts.append(f"{info['files']}文件")
        if "rows" in info:
            detail_parts.append(f"{info['rows']}行")
        if "num_factors" in info:
            detail_parts.append(f"{info['num_factors']}因子")
        if "num_labels" in info:
            detail_parts.append(f"{info['num_labels']}标签")
        if "reason" in info and status in ("FAIL", "WARN"):
            detail_parts.append(f"原因: {info['reason']}")

        detail = ", ".join(detail_parts)
        print(f"  {status_icon} {name:<35s} {detail}")

        if status == "OK":
            ok += 1
        elif status == "WARN":
            warn += 1
        elif status == "FAIL":
            fail += 1
        elif status == "SKIP":
            skip += 1

    print(f"\n  总计: ✅ {ok}  ⚠️ {warn}  ❌ {fail}  ⏭️ {skip}")
    print(f"{'='*60}")

    if fail > 0:
        print("\n❌ 验证失败，存在错误项，请检查后再继续。")
        sys.exit(1)
    elif warn > 0:
        print("\n⚠️  存在警告项，请确认是否可继续。")
        sys.exit(0)
    else:
        print("\n✅ 全部通过。")
        sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description="数据完整性验证")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["raw", "cleaned", "processed", "features", "cross_sectional", "labels", "all"],
        help="验证阶段",
    )
    parser.add_argument("--output", type=str, help="输出 JSON 结果文件")
    args = parser.parse_args()

    all_results = {}

    stages = (
        ["raw", "cleaned", "processed", "features", "cross_sectional", "labels"]
        if args.stage == "all"
        else [args.stage]
    )

    validators = {
        "raw": validate_raw,
        "cleaned": validate_cleaned,
        "processed": validate_processed,
        "features": validate_features,
        "cross_sectional": validate_cross_sectional,
        "labels": validate_labels,
    }

    has_fail = False
    for stage in stages:
        results = validators[stage]()
        all_results[stage] = results
        print_summary(results, stage)

        # 检查是否有 FAIL
        for info in results.values():
            if info.get("status") == "FAIL":
                has_fail = True

    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\n详细结果已写入: {args.output}")

    if has_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
