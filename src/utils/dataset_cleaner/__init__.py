"""Dataset cleaners for raw market-data quality checks and cleaning."""

from __future__ import annotations


def __getattr__(name: str):
    if name in {"IndustryProcessSummary", "IndustryCleaner"}:
        from .industry_cleaner import IndustryProcessSummary, IndustryCleaner

        return {
            "IndustryProcessSummary": IndustryProcessSummary,
            "IndustryCleaner": IndustryCleaner,
        }[name]
    if name in {"FundamentalsProcessSummary", "FundamentalsCleaner"}:
        from .fundamentals_cleaner import FundamentalsProcessSummary, FundamentalsCleaner

        return {
            "FundamentalsProcessSummary": FundamentalsProcessSummary,
            "FundamentalsCleaner": FundamentalsCleaner,
        }[name]
    if name in {"DailyBarsProcessSummary", "DailyBarsCleaner"}:
        from .daily_bars_cleaner import DailyBarsProcessSummary, DailyBarsCleaner

        return {
            "DailyBarsProcessSummary": DailyBarsProcessSummary,
            "DailyBarsCleaner": DailyBarsCleaner,
        }[name]
    if name in {"AdjFactorsProcessSummary", "AdjFactorsCleaner"}:
        from .adj_factors_cleaner import AdjFactorsProcessSummary, AdjFactorsCleaner

        return {
            "AdjFactorsProcessSummary": AdjFactorsProcessSummary,
            "AdjFactorsCleaner": AdjFactorsCleaner,
        }[name]
    if name in {"MoneyflowProcessSummary", "MoneyflowCleaner"}:
        from .moneyflow_cleaner import MoneyflowProcessSummary, MoneyflowCleaner

        return {
            "MoneyflowProcessSummary": MoneyflowProcessSummary,
            "MoneyflowCleaner": MoneyflowCleaner,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AdjFactorsProcessSummary",
    "AdjFactorsCleaner",
    "DailyBarsProcessSummary",
    "DailyBarsCleaner",
    "FundamentalsProcessSummary",
    "FundamentalsCleaner",
    "IndustryProcessSummary",
    "IndustryCleaner",
    "MoneyflowProcessSummary",
    "MoneyflowCleaner",
]
