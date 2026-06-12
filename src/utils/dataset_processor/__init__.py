"""Dataset processors for derived feature generation."""

from __future__ import annotations


def __getattr__(name: str):
    if name in {"NamechangeStProcessSummary", "NamechangeStProcessor"}:
        from .namechange_st_processor import NamechangeStProcessSummary, NamechangeStProcessor

        return {
            "NamechangeStProcessSummary": NamechangeStProcessSummary,
            "NamechangeStProcessor": NamechangeStProcessor,
        }[name]
    if name in {"IndustryOneHotProcessSummary", "IndustryOneHotProcessor"}:
        from .industry_onehot_processor import IndustryOneHotProcessSummary, IndustryOneHotProcessor

        return {
            "IndustryOneHotProcessSummary": IndustryOneHotProcessSummary,
            "IndustryOneHotProcessor": IndustryOneHotProcessor,
        }[name]
    if name in {"SuspendDProcessSummary", "SuspendDProcessor"}:
        from .suspend_d_processor import SuspendDProcessSummary, SuspendDProcessor

        return {
            "SuspendDProcessSummary": SuspendDProcessSummary,
            "SuspendDProcessor": SuspendDProcessor,
        }[name]
    if name in {"WideTableDailyBarsBuildSummary", "WideTableDailyBarsBuilder"}:
        from .wide_table_daily_bars_builder import WideTableDailyBarsBuildSummary, WideTableDailyBarsBuilder

        return {
            "WideTableDailyBarsBuildSummary": WideTableDailyBarsBuildSummary,
            "WideTableDailyBarsBuilder": WideTableDailyBarsBuilder,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "NamechangeStProcessSummary",
    "NamechangeStProcessor",
    "IndustryOneHotProcessSummary",
    "IndustryOneHotProcessor",
    "SuspendDProcessSummary",
    "SuspendDProcessor",
    "WideTableDailyBarsBuildSummary",
    "WideTableDailyBarsBuilder",
]
