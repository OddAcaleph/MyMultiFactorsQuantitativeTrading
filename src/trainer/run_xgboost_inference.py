"""Command-line entry point for running XGBoost inference."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils import DEFAULT_INFERENCER_CONFIG_PATH

try:
    from .xgboost_inferencer import XGBoostInferencer
except ImportError:  # pragma: no cover - supports running this file directly.
    from trainer.xgboost_inferencer import XGBoostInferencer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run XGBoost inference with project config.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_INFERENCER_CONFIG_PATH),
        help="Path to xgboost inferencer config JSON.",
    )
    parser.add_argument(
        "--trainer-config",
        default=None,
        help="Optional path to xgboost trainer config JSON. Overrides inferencer config trainer_config_path.",
    )
    parser.add_argument(
        "--loader-config",
        default=None,
        help="Optional path to parquet loader config JSON. Overrides inferencer/trainer config loader_config_path.",
    )
    parser.add_argument(
        "--model-path",
        default=None,
        help="Optional model path. Overrides inferencer config model_path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory for prediction parquet.",
    )
    parser.add_argument(
        "--segment",
        default=None,
        help="Dataset segment name to run inference on. Default comes from config.",
    )
    parser.add_argument(
        "--start-time",
        default=None,
        help="Optional inference start date, e.g. 2023-01-01.",
    )
    parser.add_argument(
        "--end-time",
        default=None,
        help="Optional inference end date, e.g. 2025-12-31.",
    )
    parser.add_argument(
        "--no-label",
        action="store_true",
        help="Do not load label column during inference.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU prediction by setting prefer_gpu=False.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    inferencer = XGBoostInferencer(
        config_path=args.config,
        trainer_config_path=args.trainer_config,
        loader_config_path=args.loader_config,
        model_path=args.model_path,
        output_dir=args.output_dir,
        segment=args.segment,
        start_time=args.start_time,
        end_time=args.end_time,
        prefer_gpu=False if args.cpu else None,
    )
    if args.no_label:
        inferencer.include_label = False

    pred_df = inferencer.predict(save=True)
    prediction_path = Path(inferencer.output_dir) / inferencer.config.get("prediction_filename", "pred_test.parquet")

    print("推理完成")
    print(f"模型路径: {inferencer.model_path}")
    print(f"输出目录: {inferencer.output_dir}")
    print(f"预测结果: {prediction_path}")
    print("预测结果预览:")
    print(pred_df.head())
    return pred_df


if __name__ == "__main__":
    main()
