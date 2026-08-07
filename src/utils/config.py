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
DEFAULT_WALK_FORWARD_CONFIG_PATH = CONF_DIR / "walk_forward_config.json"
DEFAULT_CONFIG_PATH = DEFAULT_TRAINER_CONFIG_PATH

_PROJECT_ROOT_VAR = "${PROJECT_ROOT}"


def resolve_path(value: str, base: Path = PROJECT_ROOT) -> Path:
    """Resolve a config path value.

    Supports ``${PROJECT_ROOT}`` placeholder and relative paths (resolved
    against *base*, defaulting to the project root). Absolute paths are
    returned unchanged.
    """
    expanded = value.replace(_PROJECT_ROOT_VAR, str(PROJECT_ROOT))
    p = Path(expanded)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def _expand_project_root(obj: Any) -> Any:
    """Recursively expand ``${PROJECT_ROOT}`` in all string values."""
    if isinstance(obj, Mapping):
        return {k: _expand_project_root(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_project_root(item) for item in obj]
    if isinstance(obj, str) and _PROJECT_ROOT_VAR in obj:
        return obj.replace(_PROJECT_ROOT_VAR, str(PROJECT_ROOT))
    return obj


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load JSON config from disk and expand ``${PROJECT_ROOT}`` placeholders.

    Parameters
    ----------
    config_path
        Config file path.
    """

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    return _expand_project_root(config)


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


def load_walk_forward_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load the default or user-specified walk-forward config."""

    return load_config(config_path or DEFAULT_WALK_FORWARD_CONFIG_PATH)


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
    "resolve_path",
]
