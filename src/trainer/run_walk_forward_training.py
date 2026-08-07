"""CLI entry point for WalkForward XGBoost training.

Usage::

    python src/trainer/run_walk_forward_training.py \
        --config conf/walk_forward_config.json \
        --verbose 50

    # Resume from existing models/predictions
    python src/trainer/run_walk_forward_training.py --resume

    # Only generate windows and show summary (dry-run)
    python src/trainer/run_walk_forward_training.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from trainer.walk_forward_trainer import WalkForwardTrainer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-Forward XGBoost Training")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to walk-forward config JSON (default: conf/walk_forward_config.json)",
    )
    parser.add_argument(
        "--loader-config",
        type=str,
        default=None,
        help="Override path to ParquetLoader config",
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=50,
        help="XGBoost verbosity (0=silent, 50=every 50 rounds)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save models or predictions to disk",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from existing models/predictions",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only generate windows and print summary, do not train",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Prefer GPU training (falls back to CPU if unavailable)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    trainer = WalkForwardTrainer(
        config_path=args.config,
        loader_config_path=args.loader_config,
        prefer_gpu=args.gpu if args.gpu else None,
    )

    windows = trainer.generate_windows()

    print(f"Walk-Forward Configuration:")
    print(f"  Mode:            {trainer.mode}")
    print(f"  Train window:    {trainer.train_window_years} years")
    print(f"  Step:            {trainer.step_years} years")
    print(f"  Valid ratio:     {trainer.valid_ratio}")
    print(f"  Train range:     {trainer.train_start} ~ {trainer.train_end}")
    print(f"  Test range:      {trainer.test_start} ~ {trainer.test_end}")
    print(f"  Label:           {trainer.label_name}")
    print(f"  Feature count:   {len(trainer.feature_cols)}")
    print(f"  Windows:         {len(windows)}")
    print()

    for w in windows:
        print(f"  {w.name}: train={w.train_start}~{w.train_end}, "
              f"valid={w.valid_start}~{w.valid_end}, "
              f"test={w.test_start}~{w.test_end}")

    if args.dry_run:
        print("\nDry-run complete. No training performed.")
        return

    print()
    results = trainer.run(
        verbose=args.verbose,
        save=not args.no_save,
        resume=not args.no_resume,
    )

    print("\n" + "=" * 60)
    print("Walk-Forward Summary")
    print("=" * 60)
    print(f"{'Window':<35} {'IC':>8} {'RankIC':>8} {'RMSE':>10}")
    print("-" * 60)
    for r in results:
        ic = r.metrics.get("ic", float("nan"))
        rank_ic = r.metrics.get("rank_ic", float("nan"))
        rmse = r.metrics.get("rmse", float("nan"))
        print(f"{r.window.name:<35} {ic:8.4f} {rank_ic:8.4f} {rmse:10.6f}")

    ics = [r.metrics["ic"] for r in results if "ic" in r.metrics and not (r.metrics["ic"] != r.metrics["ic"])]
    rank_ics = [r.metrics["rank_ic"] for r in results if "rank_ic" in r.metrics and not (r.metrics["rank_ic"] != r.metrics["rank_ic"])]

    if ics:
        print("-" * 60)
        print(f"{'Average':<35} {sum(ics)/len(ics):8.4f} {sum(rank_ics)/len(rank_ics):8.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
