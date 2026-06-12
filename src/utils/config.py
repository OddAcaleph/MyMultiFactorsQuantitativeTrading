"""Helpers for loading project-level JSON config files."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONF_DIR = PROJECT_ROOT / "conf"
DEFAULT_LOADER_CONFIG_PATH = CONF_DIR / "parquet_loader_config.json"
DEFAULT_TRAINER_CONFIG_PATH = CONF_DIR / "xgboost_trainer_config.json"
DEFAULT_BACKTESTER_CONFIG_PATH = CONF_DIR / "preparatory_backtester_config.json"
DEFAULT_SIMPLE_BACKTESTER_CONFIG_PATH = CONF_DIR / "simple_backtester_config.json"
DEFAULT_SIMPLE_BACKTEST_GRID_SEARCH_CONFIG_PATH = CONF_DIR / "simple_backtest_grid_search_config.json"
DEFAULT_INFERENCER_CONFIG_PATH = CONF_DIR / "xgboost_inferencer_config.json"
DEFAULT_XGBOOST_TRAIN_BACKTEST_GRID_SEARCH_CONFIG_PATH = CONF_DIR / "xgboost_train_backtest_grid_search_config.json"
DEFAULT_CONFIG_PATH = DEFAULT_TRAINER_CONFIG_PATH


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load JSON config from disk.

    Parameters
    ----------
    config_path
        Config file path.
    """

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_loader_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the default or user-specified ParquetLoader config."""

    return load_config(config_path or DEFAULT_LOADER_CONFIG_PATH)


def load_trainer_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the default or user-specified XGBoostTrainer config."""

    return load_config(config_path or DEFAULT_TRAINER_CONFIG_PATH)


def load_backtester_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the default or user-specified PreparatoryBacktester config."""

    return load_config(config_path or DEFAULT_BACKTESTER_CONFIG_PATH)


def load_simple_backtester_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the default or user-specified SimpleBacktester config."""

    return load_config(config_path or DEFAULT_SIMPLE_BACKTESTER_CONFIG_PATH)


def load_simple_backtest_grid_search_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the default or user-specified SimpleBacktester grid-search config."""

    return load_config(config_path or DEFAULT_SIMPLE_BACKTEST_GRID_SEARCH_CONFIG_PATH)


def load_inferencer_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the default or user-specified XGBoostInferencer config."""

    return load_config(config_path or DEFAULT_INFERENCER_CONFIG_PATH)


def load_xgboost_train_backtest_grid_search_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the default or user-specified XGBoost train+backtest grid-search config."""

    return load_config(config_path or DEFAULT_XGBOOST_TRAIN_BACKTEST_GRID_SEARCH_CONFIG_PATH)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a recursive merge of two mappings without mutating inputs."""

    result = deepcopy(dict(base))
    if not override:
        return result

    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


__all__ = [
    "CONF_DIR",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_BACKTESTER_CONFIG_PATH",
    "DEFAULT_LOADER_CONFIG_PATH",
    "DEFAULT_INFERENCER_CONFIG_PATH",
    "DEFAULT_SIMPLE_BACKTESTER_CONFIG_PATH",
    "DEFAULT_SIMPLE_BACKTEST_GRID_SEARCH_CONFIG_PATH",
    "DEFAULT_TRAINER_CONFIG_PATH",
    "DEFAULT_XGBOOST_TRAIN_BACKTEST_GRID_SEARCH_CONFIG_PATH",
    "PROJECT_ROOT",
    "deep_merge",
    "load_config",
    "load_backtester_config",
    "load_loader_config",
    "load_inferencer_config",
    "load_simple_backtester_config",
    "load_simple_backtest_grid_search_config",
    "load_trainer_config",
    "load_xgboost_train_backtest_grid_search_config",
]
