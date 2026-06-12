"""Label generation utilities for A-share datasets."""

from __future__ import annotations

from typing import Any

__all__ = [
    "DailyLabelGenerateSummary",
    "DailyLabelGenerator",
    "configure_logging",
    "main",
    "parse_args",
]


def __getattr__(name: str) -> Any:
    """Lazily expose label generators without eager module import."""

    if name in {"DailyLabelGenerateSummary", "DailyLabelGenerator", "configure_logging", "main", "parse_args"}:
        from . import daily_label_generator as module

        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
