"""Feature generation utilities for A-share datasets."""

from __future__ import annotations

from typing import Any

__all__ = [
    "FundamentalFeatureGenerateSummary",
    "FundamentalFeatureGenerator",
    "IndustryFeatureGenerateSummary",
    "IndustryFeatureGenerator",
    "MoneyFlowFeatureGenerateSummary",
    "MoneyFlowFeatureGenerator",
    "PriceVolumeFeatureGenerateSummary",
    "PriceVolumeFeatureGenerator",
    "configure_logging",
    "main",
    "parse_args",
]


def __getattr__(name: str) -> Any:
    """Lazily expose feature generators without eager module import."""

    if name in {"FundamentalFeatureGenerateSummary", "FundamentalFeatureGenerator"}:
        from . import fundamental_feature_generator as module

        return getattr(module, name)
    if name in {"IndustryFeatureGenerateSummary", "IndustryFeatureGenerator"}:
        from . import industry_feature_generator as module

        return getattr(module, name)
    if name in {"MoneyFlowFeatureGenerateSummary", "MoneyFlowFeatureGenerator"}:
        from . import moneyflow_feature_generator as module

        return getattr(module, name)
    if name in {"PriceVolumeFeatureGenerateSummary", "PriceVolumeFeatureGenerator", "configure_logging", "main", "parse_args"}:
        from . import price_volume_feature_generator as module

        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
