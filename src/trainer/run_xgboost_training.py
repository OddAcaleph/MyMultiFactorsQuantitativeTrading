"""Command-line entry point for running XGBoost training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from utils import DEFAULT_TRAINER_CONFIG_PATH

try:
    from .xgboost_trainer import XGBoostTrainer
except ImportError:  # pragma: no cover - supports running this file directly.
    from trainer.xgboost_trainer import XGBoostTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XGBoost training with project config.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_TRAINER_CONFIG_PATH),
        help="Path to xgboost trainer config JSON.",
    )
    parser.add_argument(
        "--loader-config",
        default=None,
        help="Optional path to parquet loader config JSON. Overrides trainer config loader_config_path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for predictions and metrics.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional model output path. Overrides model_dir/model_filename in trainer config.",
    )
    parser.add_argument(
        "--verbose",
        default="50",
        help="XGBoost fit verbose setting. Use false/0 to disable, true/1 to enable, or an integer interval.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run training/evaluation without saving model, predictions and metrics.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU training by setting prefer_gpu=False.",
    )
    return parser.parse_args()


def parse_verbose(value: str) -> bool | int:
    normalized = value.strip().lower()
    if normalized in {"false", "f", "no", "n", "0", "none"}:
        return False
    if normalized in {"true", "t", "yes", "y", "1"}:
        return True
    return int(value)


def json_safe_metrics(metrics: dict[str, Any]) -> str:
    return json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=True)


def main() -> dict[str, Any]:
    args = parse_args()

    trainer = XGBoostTrainer(
        config_path=args.config,
        loader_config_path=args.loader_config,
        output_dir=args.output_dir,
        model_path=args.model_path,
        prefer_gpu=False if args.cpu else None,
    )
    result = trainer.run(verbose=parse_verbose(args.verbose), save=not args.no_save)

    print("训练完成")
    print(f"使用设备: {trainer.device_used}")
    print(f"模型路径: {trainer.model_path}")
    if trainer.output_dir is not None:
        print(f"输出目录: {trainer.output_dir}")
        print(f"测试集预测: {Path(trainer.output_dir) / 'pred_test.parquet'}")
        print(f"评估指标文件: {Path(trainer.output_dir) / 'metrics.json'}")
    print("评估指标:")
    print(json_safe_metrics(result["metrics"]))
    return result


if __name__ == "__main__":
    main()
