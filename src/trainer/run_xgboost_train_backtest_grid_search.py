"""Asynchronous XGBoost training-grid and SimpleBacktester grid-search runner.

This script trains one XGBoost model for each model-parameter grid point.  As
soon as one model and its test prediction parquet are saved, the model is put
into a backtest queue.  A second worker thread consumes that queue and runs the
existing ``run_simple_backtest_grid_search.py`` logic with its default grid
parameters, only overriding ``prediction_path`` and per-model output root.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import itertools
import json
import logging
import queue
import re
import sys
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from utils import (  # noqa: E402
    DEFAULT_SIMPLE_BACKTESTER_CONFIG_PATH,
    DEFAULT_TRAINER_CONFIG_PATH,
    DEFAULT_XGBOOST_TRAIN_BACKTEST_GRID_SEARCH_CONFIG_PATH,
    PROJECT_ROOT,
    load_xgboost_train_backtest_grid_search_config,
)

try:  # noqa: E402 - supports both package and direct script execution.
    from .xgboost_trainer import XGBoostTrainer
except ImportError:  # pragma: no cover
    from trainer.xgboost_trainer import XGBoostTrainer

from backtester.run_simple_backtest_grid_search import (  # noqa: E402
    BacktestPlotter,
    build_base_overrides,
    build_grid as build_backtest_grid,
    evaluate_one as evaluate_one_backtest,
    flatten_record as flatten_backtest_record,
    json_safe,
    preload_data as preload_backtest_data,
    resolve_jobs,
    run_one as run_one_backtest,
)
from backtester.simple_backtester import SimpleBacktester  # noqa: E402

TRAINER_ONLY_GRID_KEYS = {"label_name"}


def expand_project_path(value: str) -> str:
    """Expand project-root placeholders used by grid-search config files."""

    return value.replace("${PROJECT_ROOT}", str(PROJECT_ROOT))


def load_grid_search_defaults(config_path: str | Path | None = None) -> dict[str, Any]:
    defaults = load_xgboost_train_backtest_grid_search_config(config_path)
    for key in ("model_root", "log_path", "backtest_output_root"):
        if key in defaults and defaults[key] is not None:
            defaults[key] = expand_project_path(str(defaults[key]))
    return defaults


_DEFAULT_GRID_SEARCH_CONFIG = load_grid_search_defaults()
LABEL_NAME = str(_DEFAULT_GRID_SEARCH_CONFIG["label_name_grid"])
N_ESTIMATORS = str(_DEFAULT_GRID_SEARCH_CONFIG["n_estimators_grid"])
MAX_DEPTH = str(_DEFAULT_GRID_SEARCH_CONFIG["max_depth_grid"])
LEARNING_RATE = str(_DEFAULT_GRID_SEARCH_CONFIG["learning_rate_grid"])
DEFAULT_FREQ_GRID = str(_DEFAULT_GRID_SEARCH_CONFIG["freq_grid"])
DEFAULT_TOPK_GRID = str(_DEFAULT_GRID_SEARCH_CONFIG["topk_grid"])
DEFAULT_N_DROP_GRID = str(_DEFAULT_GRID_SEARCH_CONFIG["n_drop_grid"])
DEFAULT_ACCOUNT_GRID = str(_DEFAULT_GRID_SEARCH_CONFIG["account_grid"])
DEFAULT_MODEL_ROOT = Path(_DEFAULT_GRID_SEARCH_CONFIG["model_root"])
DEFAULT_LOG_PATH = Path(_DEFAULT_GRID_SEARCH_CONFIG["log_path"])
DEFAULT_BACKTEST_OUTPUT_ROOT = Path(_DEFAULT_GRID_SEARCH_CONFIG["backtest_output_root"])
DEFAULT_JOBS = int(_DEFAULT_GRID_SEARCH_CONFIG.get("jobs", 4))
SENTINEL = object()


class TeeStream:
    """Write all stdout/stderr text to multiple streams."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams
        self._lock = threading.Lock()

    def write(self, data: str) -> int:
        with self._lock:
            for stream in self.streams:
                stream.write(data)
                stream.flush()
        return len(data)

    def flush(self) -> None:
        with self._lock:
            for stream in self.streams:
                stream.flush()


@dataclass(frozen=True)
class TrainedModelTask:
    idx: int
    total: int
    run_id: str
    params: dict[str, Any]
    model_params: dict[str, Any]
    label_name: str
    model_path: Path
    prediction_path: Path
    train_output_dir: Path
    train_metrics: dict[str, Any]
    device_used: str | None


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--grid-search-config",
        default=str(DEFAULT_XGBOOST_TRAIN_BACKTEST_GRID_SEARCH_CONFIG_PATH),
        help="Path to XGBoost train+backtest grid-search defaults JSON.",
    )
    pre_args, _ = pre_parser.parse_known_args()
    grid_defaults = load_grid_search_defaults(pre_args.grid_search_config)

    parser = argparse.ArgumentParser(
        description="Train XGBoost model grid asynchronously with SimpleBacktester grid search.",
        parents=[pre_parser],
    )

    # Trainer settings.
    parser.add_argument("--trainer-config", default=str(DEFAULT_TRAINER_CONFIG_PATH), help="XGBoost trainer config JSON.")
    parser.add_argument("--loader-config", default=None, help="Optional parquet loader config JSON.")
    parser.add_argument("--model-root", default=str(grid_defaults["model_root"]), help="Root directory for trained grid models.")
    parser.add_argument(
        "--train-output-root",
        default=None,
        help="Root directory for training predictions/metrics. Defaults to <model-root>/<run-id>/train_outputs.",
    )
    parser.add_argument("--verbose", default=str(grid_defaults.get("verbose", "50")), help="XGBoost fit verbose: false/0, true/1, or integer interval.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU training by setting prefer_gpu=False.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first training/backtest failure.")
    resume_default = bool(grid_defaults.get("resume", True))
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument("--resume", dest="resume", action="store_true", default=resume_default, help="Resume completed training/backtest grid points when artifacts already exist.")
    resume_group.add_argument("--no-resume", dest="resume", action="store_false", help="Disable resume and rerun all grid points.")

    # XGBoost parameter grid.  Extra parameters default to the current config
    # values unless explicitly supplied here, so the default grid is useful but
    # not accidentally explosive.
    parser.add_argument("--n-estimators-grid", default=grid_defaults.get("n_estimators_grid"), help="Comma-separated n_estimators grid.")
    parser.add_argument("--max-depth-grid", default=grid_defaults.get("max_depth_grid"), help="Comma-separated max_depth grid.")
    parser.add_argument("--learning-rate-grid", default=grid_defaults.get("learning_rate_grid"), help="Comma-separated learning_rate grid.")
    parser.add_argument(
        "--label-name-grid",
        default=grid_defaults.get("label_name_grid"),
        help="Optional comma-separated label_name grid, e.g. label_1d,label_5d,label_10d. Defaults to trainer config label_name.",
    )
    parser.add_argument("--subsample-grid", default=grid_defaults.get("subsample_grid"), help="Optional comma-separated subsample grid.")
    parser.add_argument("--colsample-bytree-grid", default=grid_defaults.get("colsample_bytree_grid"), help="Optional comma-separated colsample_bytree grid.")
    parser.add_argument("--min-child-weight-grid", default=grid_defaults.get("min_child_weight_grid"), help="Optional comma-separated min_child_weight grid.")
    parser.add_argument("--gamma-grid", default=grid_defaults.get("gamma_grid"), help="Optional comma-separated gamma grid.")
    parser.add_argument("--reg-alpha-grid", default=grid_defaults.get("reg_alpha_grid"), help="Optional comma-separated reg_alpha grid.")
    parser.add_argument("--reg-lambda-grid", default=grid_defaults.get("reg_lambda_grid"), help="Optional comma-separated reg_lambda grid.")

    # Backtest settings. Defaults mirror run_simple_backtest_grid_search.py.
    parser.add_argument(
        "--backtest-config",
        default=str(DEFAULT_SIMPLE_BACKTESTER_CONFIG_PATH),
        help="SimpleBacktester config JSON. Defaults to conf/simple_backtester_config.json.",
    )
    parser.add_argument("--backtest-output-root", default=str(grid_defaults["backtest_output_root"]), help="Backtest output root.")
    parser.add_argument("--freq-grid", default=grid_defaults.get("freq_grid"), help="Backtest freq grid.")
    parser.add_argument("--topk-grid", default=grid_defaults.get("topk_grid"), help="Backtest topk grid.")
    parser.add_argument("--n-drop-grid", default=grid_defaults.get("n_drop_grid"), help="Backtest n_drop grid.")
    parser.add_argument("--account-grid", default=grid_defaults.get("account_grid"), help="Backtest account grid.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=int(grid_defaults.get("jobs", DEFAULT_JOBS)),
        help=f"Parallel backtest grid-search workers per trained model. Defaults to {grid_defaults.get('jobs', DEFAULT_JOBS)}. Use 0 to auto-select CPU count.",
    )

    # Optional pass-through overrides for the backtester; all default to None so
    # the SimpleBacktester config/default script behavior is preserved.
    parser.add_argument("--price-path", default=None, help="Optional daily wide-table bar parquet root.")
    parser.add_argument("--benchmark-path", default=None, help="Optional qlib features directory for benchmark data.")
    parser.add_argument("--start-time", default=None, help="Optional backtest start date.")
    parser.add_argument("--end-time", default=None, help="Optional backtest end date.")
    parser.add_argument("--score-col", default=None, help="Optional prediction score column.")
    parser.add_argument("--benchmark", default=None, help="Optional benchmark index.")
    parser.add_argument("--deal-price", choices=["open", "close"], default=None, help="Optional deal price column.")
    parser.add_argument("--open-cost", type=float, default=None, help="Optional buy-side commission rate.")
    parser.add_argument("--close-cost", type=float, default=None, help="Optional sell-side commission/stamp-duty rate.")
    parser.add_argument("--min-cost", type=float, default=None, help="Optional minimum fee per trade.")
    parser.add_argument("--slippage", type=float, default=None, help="Optional one-way slippage rate.")
    parser.add_argument("--limit-threshold", type=float, default=None, help="Optional limit-up/down threshold.")
    parser.add_argument("--lot-size", type=int, default=None, help="Optional board lot size.")
    parser.add_argument("--allow-buy-limit-up", action="store_true", help="Allow buying limit-up stocks.")
    parser.add_argument("--allow-buy-limit-down", action="store_true", help="Allow buying limit-down stocks.")
    parser.add_argument("--forbid-sell-limit-down", action="store_true", help="Forbid selling limit-down stocks.")
    parser.add_argument("--no-positions", action="store_true", help="Do not save positions parquet in final best backtest runs.")

    parser.add_argument("--log-path", default=str(grid_defaults["log_path"]), help="Log directory or concrete log file path.")
    return parser.parse_args()


def parse_verbose(value: str) -> bool | int:
    normalized = value.strip().lower()
    if normalized in {"false", "f", "no", "n", "0", "none"}:
        return False
    if normalized in {"true", "t", "yes", "y", "1"}:
        return True
    return int(value)


def parse_grid(value: str | None, cast: type, name: str) -> list[Any] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} grid is empty")
    parsed = [cast(item) for item in items]
    if any(item <= 0 for item in parsed):
        raise ValueError(f"{name} grid values must be positive: {parsed}")
    return parsed


def parse_str_grid(value: str | None, name: str) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError(f"{name} grid is empty")
    return items


def build_model_grid(args: argparse.Namespace) -> list[dict[str, Any]]:
    grid_spec: list[tuple[str, list[Any] | None]] = [
        ("n_estimators", parse_grid(args.n_estimators_grid, int, "n_estimators")),
        ("max_depth", parse_grid(args.max_depth_grid, int, "max_depth")),
        ("learning_rate", parse_grid(args.learning_rate_grid, float, "learning_rate")),
        ("label_name", parse_str_grid(args.label_name_grid, "label_name")),
        ("subsample", parse_grid(args.subsample_grid, float, "subsample")),
        ("colsample_bytree", parse_grid(args.colsample_bytree_grid, float, "colsample_bytree")),
        ("min_child_weight", parse_grid(args.min_child_weight_grid, float, "min_child_weight")),
        ("gamma", parse_grid(args.gamma_grid, float, "gamma")),
        ("reg_alpha", parse_grid(args.reg_alpha_grid, float, "reg_alpha")),
        ("reg_lambda", parse_grid(args.reg_lambda_grid, float, "reg_lambda")),
    ]
    active = [(name, values) for name, values in grid_spec if values is not None]
    if not active:
        raise ValueError("At least one model parameter grid must be non-empty")

    names = [name for name, _ in active]
    values_list = [values for _, values in active]
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*values_list)]


def split_grid_params(params: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    """Split XGBoost model params from trainer-only grid params."""

    model_params = {key: value for key, value in params.items() if key not in TRAINER_ONLY_GRID_KEYS}
    label_name = params.get("label_name")
    return model_params, str(label_name) if label_name is not None else None


def grid_params_key(params: dict[str, Any]) -> str:
    """Stable identity for model-grid parameters across process restarts."""

    return json.dumps(json_safe(params), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def backtest_params_key(params: dict[str, Any]) -> str:
    """Stable identity for SimpleBacktester grid parameters across CSV reloads."""

    return "|".join(
        [
            str(params["freq"]),
            str(int(params["topk"])),
            str(int(params["n_drop"])),
            f"{float(params['account']):.12g}",
        ]
    )


def read_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def is_nonempty_file(path: Path) -> bool:
    return path.exists() and path.is_file() and path.stat().st_size > 0


def make_run_id(idx: int, params: dict[str, Any]) -> str:
    params_key = grid_params_key(params)
    digest = hashlib.sha1(params_key.encode("utf-8")).hexdigest()[:10]
    compact_params = "_".join(f"{key}-{value}" for key, value in params.items())
    compact_params = re.sub(r"[^A-Za-z0-9_.-]+", "-", compact_params).strip("-")[:120]
    return f"grid{idx:04d}_{digest}_{compact_params}"


def build_trained_model_task_from_metadata(metadata_path: Path, idx: int, total: int, params: dict[str, Any]) -> TrainedModelTask | None:
    try:
        metadata = read_json_file(metadata_path)
    except Exception as exc:  # noqa: BLE001
        logging.warning("读取训练元信息失败，不能用于恢复: %s error=%s", metadata_path, exc)
        return None

    if grid_params_key(metadata.get("grid_params", {})) != grid_params_key(params):
        return None

    model_path = Path(metadata.get("model_path", ""))
    prediction_path = Path(metadata.get("prediction_path", ""))
    train_output_dir = Path(metadata.get("train_output_dir", prediction_path.parent if str(prediction_path) else ""))
    if not is_nonempty_file(model_path) or not is_nonempty_file(prediction_path):
        return None

    metrics = metadata.get("metrics", {})
    metrics_path = train_output_dir / "metrics.json"
    if is_nonempty_file(metrics_path):
        try:
            metrics = read_json_file(metrics_path)
        except Exception as exc:  # noqa: BLE001
            logging.warning("读取已存在 metrics 失败，使用 train_metadata 中的 metrics: %s error=%s", metrics_path, exc)

    model_params, label_name = split_grid_params(params)
    return TrainedModelTask(
        idx=idx,
        total=total,
        run_id=str(metadata.get("run_id") or metadata_path.parent.name),
        params=dict(params),
        model_params=dict(metadata.get("model_params") or model_params),
        label_name=str(metadata.get("label_name") or label_name or ""),
        model_path=model_path,
        prediction_path=prediction_path,
        train_output_dir=train_output_dir,
        train_metrics=dict(metrics or {}),
        device_used=metadata.get("device_used"),
    )


def find_completed_training_task(args: argparse.Namespace, idx: int, total: int, params: dict[str, Any]) -> TrainedModelTask | None:
    """Find reusable training outputs for a model-grid point.

    The deterministic run_id path handles new runs.  The recursive metadata scan
    keeps compatibility with older interrupted runs whose run_id contained a
    timestamp.
    """

    model_root = Path(args.model_root)
    expected_metadata = model_root / make_run_id(idx, params) / "train_metadata.json"
    candidates = [expected_metadata]
    if model_root.exists():
        candidates.extend(
            sorted(
                (path for path in model_root.glob("*/train_metadata.json") if path != expected_metadata),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )

    for metadata_path in candidates:
        if not metadata_path.exists():
            continue
        task = build_trained_model_task_from_metadata(metadata_path, idx=idx, total=total, params=params)
        if task is not None:
            return task
    return None


def load_completed_backtest_summary(summary_path: Path) -> dict[str, Any] | None:
    if not is_nonempty_file(summary_path):
        return None
    try:
        summary = read_json_file(summary_path)
    except Exception as exc:  # noqa: BLE001
        logging.warning("读取已有回测摘要失败，将继续尝试恢复回测明细: %s error=%s", summary_path, exc)
        return None

    if summary.get("status") != "success":
        return None
    final_output_dir = Path(summary.get("final_backtest_output_dir", ""))
    results_csv = Path(summary.get("backtest_results_csv", ""))
    if final_output_dir.exists() and is_nonempty_file(results_csv):
        return summary
    return None


def load_backtest_progress(search_results_path: Path, grid: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], set[str], dict[str, Any] | None]:
    if not is_nonempty_file(search_results_path):
        return [], set(), None

    try:
        existing_df = pd.read_csv(search_results_path)
    except Exception as exc:  # noqa: BLE001
        logging.warning("读取已有回测搜索结果失败，将重新执行回测网格: %s error=%s", search_results_path, exc)
        return [], set(), None

    required_columns = {"freq", "topk", "n_drop", "account", "status"}
    if not required_columns.issubset(existing_df.columns):
        logging.warning("已有回测搜索结果缺少必要列，将重新执行回测网格: %s", search_results_path)
        return [], set(), None

    grid_key_to_params = {backtest_params_key(params): params for params in grid}
    records: list[dict[str, Any]] = []
    completed_keys: set[str] = set()
    best: dict[str, Any] | None = None

    for row in existing_df.to_dict(orient="records"):
        try:
            key = backtest_params_key(row)
        except Exception:  # noqa: BLE001
            continue
        if key not in grid_key_to_params:
            continue
        record = dict(row)
        records.append(record)
        completed_keys.add(key)
        total_return = record.get("total_return")
        if record.get("status") == "success" and pd.notna(total_return):
            record["total_return"] = float(total_return)
            if best is None or record["total_return"] > best["record"]["total_return"]:
                best = {"params": dict(grid_key_to_params[key]), "record": record}

    return records, completed_keys, best


def build_backtest_args(args: argparse.Namespace, prediction_path: Path, output_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        config=args.backtest_config,
        freq_grid=args.freq_grid,
        topk_grid=args.topk_grid,
        n_drop_grid=args.n_drop_grid,
        account_grid=args.account_grid,
        output_root=str(output_root),
        trainer_config=args.trainer_config,
        prediction_path=str(prediction_path),
        price_path=args.price_path,
        benchmark_path=args.benchmark_path,
        start_time=args.start_time,
        end_time=args.end_time,
        score_col=args.score_col,
        benchmark=args.benchmark,
        deal_price=args.deal_price,
        open_cost=args.open_cost,
        close_cost=args.close_cost,
        min_cost=args.min_cost,
        slippage=args.slippage,
        limit_threshold=args.limit_threshold,
        lot_size=args.lot_size,
        allow_buy_limit_up=args.allow_buy_limit_up,
        allow_buy_limit_down=args.allow_buy_limit_down,
        forbid_sell_limit_down=args.forbid_sell_limit_down,
        no_positions=args.no_positions,
        fail_fast=args.fail_fast,
        jobs=args.jobs,
    )


def train_worker(args: argparse.Namespace, model_grid: list[dict[str, Any]], task_queue: queue.Queue[Any]) -> None:
    model_root = Path(args.model_root)
    train_output_root = Path(args.train_output_root) if args.train_output_root else None
    verbose = parse_verbose(args.verbose)

    try:
        logging.info("训练线程启动，共 %s 个 XGBoost 参数网格点。", len(model_grid))
        for idx, params in enumerate(model_grid, start=1):
            model_params, label_name = split_grid_params(params)
            run_id = make_run_id(idx, params)
            model_dir = model_root / run_id
            model_path = model_dir / "xgboost_model.json"
            output_dir = (train_output_root / run_id) if train_output_root else (model_dir / "train_outputs")

            if args.resume:
                resumed_task = find_completed_training_task(args, idx=idx, total=len(model_grid), params=params)
                if resumed_task is not None:
                    logging.info(
                        "[%s/%s] 训练已完成，断点续传跳过训练并进入回测队列: run_id=%s model=%s prediction=%s",
                        idx,
                        len(model_grid),
                        resumed_task.run_id,
                        resumed_task.model_path,
                        resumed_task.prediction_path,
                    )
                    task_queue.put(resumed_task)
                    continue

            logging.info("[%s/%s] 开始训练模型: run_id=%s params=%s", idx, len(model_grid), run_id, params)
            try:
                trainer = XGBoostTrainer(
                    config_path=args.trainer_config,
                    loader_config_path=args.loader_config,
                    output_dir=output_dir,
                    model_path=model_path,
                    label_name=label_name,
                    model_params=model_params,
                    prefer_gpu=False if args.cpu else None,
                )
                result = trainer.run(verbose=verbose, save=True)
                prediction_path = output_dir / "pred_test.parquet"
                if not model_path.exists():
                    raise FileNotFoundError(f"训练完成但模型文件不存在: {model_path}")
                if not prediction_path.exists():
                    raise FileNotFoundError(f"训练完成但预测文件不存在: {prediction_path}")

                meta = {
                    "run_id": run_id,
                    "idx": idx,
                    "total": len(model_grid),
                    "grid_params": params,
                    "model_params": model_params,
                    "label_name": trainer.label_name,
                    "model_path": str(model_path),
                    "prediction_path": str(prediction_path),
                    "train_output_dir": str(output_dir),
                    "device_used": trainer.device_used,
                    "metrics": result.get("metrics", {}),
                }
                model_dir.mkdir(parents=True, exist_ok=True)
                with (model_dir / "train_metadata.json").open("w", encoding="utf-8") as f:
                    json.dump(json_safe(meta), f, ensure_ascii=False, indent=2, allow_nan=True)

                logging.info("[%s/%s] 训练完成: model=%s prediction=%s metrics=%s", idx, len(model_grid), model_path, prediction_path, result.get("metrics", {}))
                task_queue.put(
                    TrainedModelTask(
                        idx=idx,
                        total=len(model_grid),
                        run_id=run_id,
                        params=dict(params),
                        model_params=dict(model_params),
                        label_name=trainer.label_name,
                        model_path=model_path,
                        prediction_path=prediction_path,
                        train_output_dir=output_dir,
                        train_metrics=dict(result.get("metrics", {})),
                        device_used=trainer.device_used,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logging.exception("[%s/%s] 训练失败: params=%s error=%s", idx, len(model_grid), params, exc)
                if args.fail_fast:
                    raise
    finally:
        task_queue.put(SENTINEL)
        logging.info("训练线程结束，已发送回测结束信号。")


def run_backtest_grid_for_model(args: argparse.Namespace, task: TrainedModelTask) -> dict[str, Any]:
    output_root = Path(args.backtest_output_root) / task.run_id
    summary_path = output_root / "model_backtest_summary.json"
    search_results_path = output_root / "grid_search_results.csv"
    if args.resume:
        completed_summary = load_completed_backtest_summary(summary_path)
        if completed_summary is not None:
            logging.info(
                "[%s/%s] 回测已完成，断点续传跳过回测: run_id=%s best_total_return=%s",
                task.idx,
                task.total,
                task.run_id,
                completed_summary.get("best_total_return"),
            )
            return completed_summary

    bt_args = build_backtest_args(args, prediction_path=task.prediction_path, output_root=output_root)
    base_overrides = build_base_overrides(bt_args)
    grid = build_backtest_grid(bt_args)
    jobs = resolve_jobs(bt_args.jobs, len(grid))
    preloaded_data = preload_backtest_data(bt_args.config, base_overrides)
    records: list[dict[str, Any]] = []
    completed_keys: set[str] = set()
    best: dict[str, Any] | None = None
    if args.resume:
        records, completed_keys, best = load_backtest_progress(search_results_path, grid)
        if completed_keys:
            logging.info(
                "[%s/%s] 已恢复回测搜索进度: run_id=%s completed=%s/%s",
                task.idx,
                task.total,
                task.run_id,
                len(completed_keys),
                len(grid),
            )

    pending_grid = [params for params in grid if backtest_params_key(params) not in completed_keys]

    logging.info(
        "[%s/%s] 开始回测网格搜索: run_id=%s grid_size=%s pending=%s workers=%s",
        task.idx,
        task.total,
        task.run_id,
        len(grid),
        len(pending_grid),
        jobs,
    )

    def persist_backtest_progress() -> None:
        output_root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(search_results_path, index=False)

    def consume_evaluation(bt_idx: int, evaluation: dict[str, Any]) -> None:
        nonlocal best
        bt_params = evaluation["params"]
        record = evaluation["record"]
        records.append(record)
        persist_backtest_progress()
        if evaluation["error"] is None:
            logging.info(
                "[%s/%s][BT %s/%s] 回测完成: params=%s total_return=%s sharpe=%s",
                task.idx,
                task.total,
                bt_idx,
                len(pending_grid),
                bt_params,
                record.get("total_return"),
                record.get("sharpe"),
            )
            if record["total_return"] is not None and pd.notna(record["total_return"]) and (
                best is None or record["total_return"] > best["record"]["total_return"]
            ):
                best = {"params": dict(bt_params), "record": record}
        else:
            logging.error("[%s/%s][BT %s/%s] 回测失败: params=%s error=%s", task.idx, task.total, bt_idx, len(pending_grid), bt_params, evaluation["error"])
            if args.fail_fast:
                raise RuntimeError(evaluation["error"])

    if not pending_grid:
        logging.info("[%s/%s] 回测网格明细已全部完成，直接使用已有搜索结果选择最优参数: run_id=%s", task.idx, task.total, task.run_id)
    elif jobs == 1:
        for bt_idx, bt_params in enumerate(pending_grid, start=1):
            logging.info("[%s/%s][BT %s/%s] 回测参数: %s", task.idx, task.total, bt_idx, len(pending_grid), bt_params)
            evaluation = evaluate_one_backtest(bt_args.config, base_overrides, bt_params, preloaded_data)
            consume_evaluation(bt_idx, evaluation)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
            futures = {
                executor.submit(evaluate_one_backtest, bt_args.config, base_overrides, bt_params, preloaded_data): bt_params
                for bt_params in pending_grid
            }
            for bt_idx, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                consume_evaluation(bt_idx, future.result())

    output_root.mkdir(parents=True, exist_ok=True)
    search_df = pd.DataFrame(records)
    search_df.to_csv(search_results_path, index=False)

    if best is None:
        summary = {
            "run_id": task.run_id,
            "status": "failed",
            "error": "All backtest combinations failed or produced no total_return",
            "model_path": str(task.model_path),
            "prediction_path": str(task.prediction_path),
            "grid_params": task.params,
            "model_params": task.model_params,
            "label_name": task.label_name,
            "train_metrics": task.train_metrics,
            "backtest_grid_size": len(grid),
            "backtest_jobs": jobs,
            "backtest_results_csv": str(search_results_path),
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(json_safe(summary), f, ensure_ascii=False, indent=2, allow_nan=True)
        logging.warning("[%s/%s] 回测网格全部失败或无 total_return: run_id=%s", task.idx, task.total, task.run_id)
        return summary

    final_output_dir = output_root / "best_saved_run"
    final_output_dir.mkdir(parents=True, exist_ok=True)
    logging.info("[%s/%s] 回测最优参数: %s total_return=%s", task.idx, task.total, best["params"], best["record"].get("total_return"))
    final_run = run_one_backtest(bt_args.config, base_overrides, best["params"], save=True, output_dir=final_output_dir, preloaded_data=preloaded_data)
    backtester: SimpleBacktester = final_run["backtester"]
    final_result = final_run["result"]
    plot_path = final_output_dir / "backtest_plots.png"
    plot_info = BacktestPlotter(report_path=final_output_dir / "backtest_report.parquet", output_path=plot_path).plot(
        title=f"XGBoost Grid {task.idx} Best SimpleBacktester Performance"
    )

    summary = {
        "run_id": task.run_id,
        "status": "success",
        "model_path": str(task.model_path),
        "prediction_path": str(task.prediction_path),
        "train_output_dir": str(task.train_output_dir),
        "device_used": task.device_used,
        "grid_params": task.params,
        "model_params": task.model_params,
        "label_name": task.label_name,
        "train_metrics": task.train_metrics,
        "best_backtest_params": best["params"],
        "best_backtest_record": best["record"],
        "best_total_return": best["record"].get("total_return"),
        "backtest_grid_size": len(grid),
        "backtest_jobs": jobs,
        "successful_backtests": int((search_df["status"] == "success").sum()) if not search_df.empty else 0,
        "failed_backtests": int((search_df["status"] == "failed").sum()) if not search_df.empty else 0,
        "backtest_output_root": str(output_root),
        "final_backtest_output_dir": str(final_output_dir),
        "backtest_results_csv": str(search_results_path),
        "plot": plot_info,
        "final_analysis_summary": final_result["analysis"].get("summary", {}),
        "final_return_metrics": final_result["analysis"].get("return", {}),
        "final_risk_metrics": final_result["analysis"].get("risk", {}),
        "backtest_report_path": str(Path(backtester.output_dir) / "backtest_report.parquet"),
        "trades_path": str(Path(backtester.output_dir) / "trades.parquet"),
        "analysis_path": str(Path(backtester.output_dir) / "analysis.json"),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, indent=2, allow_nan=True)
    logging.info("[%s/%s] 回测完成: run_id=%s best_total_return=%s", task.idx, task.total, task.run_id, summary["best_total_return"])
    return summary


def backtest_worker(args: argparse.Namespace, task_queue: queue.Queue[Any], results: list[dict[str, Any]]) -> None:
    logging.info("回测线程启动。")
    while True:
        task = task_queue.get()
        try:
            if task is SENTINEL:
                logging.info("回测线程收到结束信号。")
                return
            assert isinstance(task, TrainedModelTask)
            try:
                summary = run_backtest_grid_for_model(args, task)
                results.append(summary)
            except Exception as exc:  # noqa: BLE001
                logging.exception("[%s/%s] 模型回测流程失败: run_id=%s error=%s", task.idx, task.total, task.run_id, exc)
                results.append(
                    {
                        "run_id": task.run_id,
                        "status": "failed",
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                        "model_path": str(task.model_path),
                        "prediction_path": str(task.prediction_path),
                        "grid_params": task.params,
                        "model_params": task.model_params,
                        "label_name": task.label_name,
                        "train_metrics": task.train_metrics,
                    }
                )
                if args.fail_fast:
                    raise
        finally:
            task_queue.task_done()


def resolve_log_paths(log_path_arg: str | Path) -> tuple[Path, Path]:
    """Resolve log directory and concrete log file path.

    ``--log-path`` is intentionally compatible with both a directory and a
    concrete file path.  The project default is a directory named ``xgb_train``;
    each run writes one timestamped log file under it.
    """

    raw_path = Path(log_path_arg)
    if raw_path.exists() and raw_path.is_dir():
        log_dir = raw_path
        log_file_path = log_dir / f"xgb_grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    elif raw_path.exists() and raw_path.is_file():
        log_dir = raw_path.parent
        log_file_path = raw_path
    elif raw_path.suffix:
        log_dir = raw_path.parent
        log_file_path = raw_path
    else:
        log_dir = raw_path
        log_file_path = log_dir / f"xgb_grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    return log_dir, log_file_path


def save_final_outputs(args: argparse.Namespace, started_at: str, model_grid: list[dict[str, Any]], results: list[dict[str, Any]]) -> dict[str, Any]:
    fallback_log_dir, fallback_log_path = resolve_log_paths(args.log_path)
    log_dir = Path(getattr(args, "log_dir", fallback_log_dir))
    log_path = Path(getattr(args, "log_file_path", fallback_log_path))
    summary_dir = log_dir
    summary_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    mapping_records: list[dict[str, Any]] = []
    for item in results:
        best_record = item.get("best_backtest_record") or {}
        mapping_records.append(
            {
                "run_id": item.get("run_id"),
                "status": item.get("status"),
                "model_path": item.get("model_path"),
                "prediction_path": item.get("prediction_path"),
                "label_name": item.get("label_name"),
                "grid_params": json.dumps(json_safe(item.get("grid_params", {})), ensure_ascii=False),
                "model_params": json.dumps(json_safe(item.get("model_params", {})), ensure_ascii=False),
                "train_metrics": json.dumps(json_safe(item.get("train_metrics", {})), ensure_ascii=False, allow_nan=True),
                "best_backtest_params": json.dumps(json_safe(item.get("best_backtest_params", {})), ensure_ascii=False),
                "best_total_return": item.get("best_total_return"),
                "backtest_jobs": item.get("backtest_jobs"),
                "best_backtest_record": json.dumps(json_safe(best_record), ensure_ascii=False, allow_nan=True),
                "backtest_results_csv": item.get("backtest_results_csv"),
                "final_backtest_output_dir": item.get("final_backtest_output_dir"),
                "error": item.get("error"),
            }
        )

    mapping_df = pd.DataFrame(mapping_records)
    mapping_csv = summary_dir / f"xgb_model_backtest_mapping_{timestamp}.csv"
    mapping_json = summary_dir / f"xgb_model_backtest_mapping_{timestamp}.json"
    summary_json = summary_dir / f"xgb_grid_search_summary_{timestamp}.json"
    mapping_df.to_csv(mapping_csv, index=False)
    with mapping_json.open("w", encoding="utf-8") as f:
        json.dump(json_safe(results), f, ensure_ascii=False, indent=2, allow_nan=True)

    successful = [item for item in results if item.get("status") == "success" and item.get("best_total_return") is not None]
    best_model = max(successful, key=lambda item: item["best_total_return"]) if successful else None
    final_summary = {
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "model_grid_size": len(model_grid),
        "trained_and_backtested_models": len(results),
        "successful_models": len(successful),
        "failed_models": len(results) - len(successful),
        "resume": bool(getattr(args, "resume", False)),
        "max_total_return": best_model.get("best_total_return") if best_model else None,
        "best_model": best_model,
        "mapping_csv": str(mapping_csv),
        "mapping_json": str(mapping_json),
        "log_path": str(log_path),
        "log_dir": str(log_dir),
    }
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(json_safe(final_summary), f, ensure_ascii=False, indent=2, allow_nan=True)

    logging.info("模型-回测结果映射 CSV: %s", mapping_csv)
    logging.info("模型-回测结果映射 JSON: %s", mapping_json)
    logging.info("总摘要 JSON: %s", summary_json)
    if best_model:
        logging.info("最大 total_return=%s，对应模型=%s，回测参数=%s", best_model.get("best_total_return"), best_model.get("model_path"), best_model.get("best_backtest_params"))
    else:
        logging.warning("没有成功产出 total_return 的模型。")
    return final_summary


def setup_logging(log_path: Path) -> tuple[TextIO, TextIO, TextIO]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)  # type: ignore[assignment]
    sys.stderr = TeeStream(original_stderr, log_file)  # type: ignore[assignment]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(threadName)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    return log_file, original_stdout, original_stderr


def main() -> dict[str, Any]:
    args = parse_args()
    log_dir, log_file_path = resolve_log_paths(args.log_path)
    args.log_dir = str(log_dir)
    args.log_file_path = str(log_file_path)
    log_file, original_stdout, original_stderr = setup_logging(log_file_path)
    started_at = datetime.now().isoformat(timespec="seconds")
    model_grid = build_model_grid(args)
    task_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
    results: list[dict[str, Any]] = []

    try:
        logging.info("XGBoost 训练 + 回测异步网格搜索启动。")
        logging.info("断点续传: %s", "启用" if args.resume else "禁用")
        logging.info("模型输出根目录: %s", Path(args.model_root))
        logging.info("日志目录: %s", log_dir)
        logging.info("日志文件: %s", log_file_path)
        logging.info("XGBoost 网格大小: %s", len(model_grid))
        logging.info("XGBoost 网格明细: %s", model_grid)

        trainer_thread = threading.Thread(target=train_worker, args=(args, model_grid, task_queue), name="xgb-train-worker")
        tester_thread = threading.Thread(target=backtest_worker, args=(args, task_queue, results), name="backtest-worker")
        tester_thread.start()
        trainer_thread.start()
        trainer_thread.join()
        tester_thread.join()

        final_summary = save_final_outputs(args, started_at, model_grid, results)
        print("\n========== XGBoost 模型网格搜索 + 回测网格搜索完成 ==========")
        print(f"模型-回测映射 CSV: {final_summary['mapping_csv']}")
        print(f"模型-回测映射 JSON: {final_summary['mapping_json']}")
        print(f"日志文件: {final_summary['log_path']}")
        print(f"最大 total_return: {final_summary['max_total_return']}")
        if final_summary.get("best_model"):
            best = final_summary["best_model"]
            print(f"最佳模型: {best.get('model_path')}")
            print(f"最佳 label_name: {best.get('label_name')}")
            print(f"最佳模型参数: {best.get('model_params')}")
            print(f"最佳回测参数: {best.get('best_backtest_params')}")
            print(f"最佳回测输出目录: {best.get('final_backtest_output_dir')}")
        return final_summary
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    main()
