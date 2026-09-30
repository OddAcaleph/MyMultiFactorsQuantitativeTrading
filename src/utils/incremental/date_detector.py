"""增量更新日期检测工具。

检测各数据层的最新日期，计算增量更新需要的日期范围。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def get_latest_trade_date(data_dir: str | Path) -> Optional[str]:
    """获取按日期分区存储的数据集中最新的交易日期。

    数据集格式：year=YYYY/month=MM/YYYYMMDD.parquet
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        return None

    dates = []
    for f in data_path.rglob("*.parquet"):
        name = f.stem
        if len(name) == 8 and name.isdigit():
            dates.append(name)

    if not dates:
        return None

    return max(dates)


def get_trade_calendar(start_date: str, end_date: str, data_dir: str | Path) -> list[str]:
    """从 daily_bars 数据中获取交易日历。"""
    data_path = Path(data_dir)
    if not data_path.exists():
        return []

    dates = []
    for f in data_path.rglob("*.parquet"):
        name = f.stem
        if len(name) == 8 and name.isdigit() and start_date <= name <= end_date:
            dates.append(name)

    return sorted(dates)


def shift_trade_date(date_str: str, shift: int, data_dir: str | Path) -> Optional[str]:
    """在交易日历中向前/向后偏移 N 个交易日。"""
    data_path = Path(data_dir)
    all_dates = []
    for f in data_path.rglob("*.parquet"):
        name = f.stem
        if len(name) == 8 and name.isdigit():
            all_dates.append(name)
    all_dates = sorted(all_dates)

    if date_str not in all_dates:
        all_dates.append(date_str)
        all_dates = sorted(set(all_dates))

    idx = all_dates.index(date_str)
    new_idx = idx + shift

    if 0 <= new_idx < len(all_dates):
        return all_dates[new_idx]
    return None


def compute_incremental_range(
    data_dir: str | Path,
    target_end_date: str,
    lookback_days: int = 0,
    reference_daily_bars: str | Path | None = None,
) -> tuple[Optional[str], Optional[str]]:
    """计算增量更新的日期范围。

    Returns
    -------
    tuple[Optional[str], Optional[str]]
        (start_date, end_date)，如果已经是最新则返回 (None, None)
    """
    latest = get_latest_trade_date(data_dir)

    if latest is None:
        return None, None

    if latest >= target_end_date:
        return None, None

    # 确定交易日历来源
    daily_bars_dir = None
    if reference_daily_bars:
        daily_bars_dir = Path(reference_daily_bars)
    else:
        data_path = Path(data_dir)
        candidates = [
            data_path.parent / "daily_bars",
            data_path.parent.parent / "raw_data" / "daily_bars",
            data_path.parent.parent / "cleaned_data" / "daily_bars",
        ]
        for p in candidates:
            if p.exists():
                daily_bars_dir = p
                break

    start_date = latest
    if lookback_days > 0 and daily_bars_dir and daily_bars_dir.exists():
        shifted = shift_trade_date(latest, -lookback_days, daily_bars_dir)
        if shifted:
            start_date = shifted

    return start_date, target_end_date


def check_dataset_completeness(
    data_dir: str | Path,
    reference_dir: str | Path,
    start_date: str = "20000101",
    end_date: str = "20991231",
) -> dict:
    """检查数据集的完整性（与参照数据集对比）。"""
    ref_dates = set(get_trade_calendar(start_date, end_date, reference_dir))
    data_dates = set(get_trade_calendar(start_date, end_date, data_dir))

    missing = sorted(ref_dates - data_dates)
    extra = sorted(data_dates - ref_dates)

    return {
        "reference_count": len(ref_dates),
        "data_count": len(data_dates),
        "missing_count": len(missing),
        "extra_count": len(extra),
        "missing_dates": missing[:20],
        "extra_dates": extra[:20],
        "latest_date": max(data_dates) if data_dates else None,
        "completeness": len(data_dates) / len(ref_dates) if ref_dates else 0,
    }
