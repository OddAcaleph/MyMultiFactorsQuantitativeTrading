"""Post-training analysis for v3 pruned walk-forward results.

Stitches predictions, runs RP backtest, and compares with baseline.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
for p in (str(PROJECT_ROOT), str(SRC_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd


def stitch_predictions(pred_dir: Path, output_path: Path) -> pd.DataFrame:
    """Stitch all window predictions into a single DataFrame."""
    all_preds = []
    for f in sorted(pred_dir.glob("wf*_pred.parquet")):
        df = pd.read_parquet(f)
        all_preds.append(df)
        print(f"  {f.name}: {df.shape[0]:,} rows, "
              f"{df.index.get_level_values(0).min().date()} ~ "
              f"{df.index.get_level_values(0).max().date()}")

    combined = pd.concat(all_preds).sort_index()
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.to_parquet(output_path)
    print(f"\nStitched: {combined.shape[0]:,} rows, "
          f"{combined.index.get_level_values(0).min().date()} ~ "
          f"{combined.index.get_level_values(0).max().date()}")
    print(f"Saved to: {output_path}")
    return combined


def compute_ic_stats(pred: pd.DataFrame, label_col: str = "label_rank_5d") -> dict:
    """Compute IC statistics."""
    pred = pred.dropna(subset=["pred", label_col])
    daily_ic = pred.groupby(level="datetime").apply(
        lambda x: x["pred"].corr(x[label_col], method="spearman")
    )
    return {
        "avg_rank_ic": daily_ic.mean(),
        "icir": daily_ic.mean() / daily_ic.std(),
        "positive_ic_pct": (daily_ic > 0).mean(),
        "n_days": len(daily_ic),
    }


def main() -> None:
    v3_pred_dir = PROJECT_ROOT / "output/walk_forward_v3_pruned/predictions"
    v3_output = v3_pred_dir / "all_predictions_2005_2025.parquet"

    print("=" * 60)
    print("Step 1: Stitch v3 pruned predictions")
    print("=" * 60)
    v3_pred = stitch_predictions(v3_pred_dir, v3_output)

    print("\n" + "=" * 60)
    print("Step 2: IC comparison")
    print("=" * 60)

    baseline_path = PROJECT_ROOT / "output/walk_forward_expanded/predictions/all_predictions_2005_2025.parquet"
    if baseline_path.exists():
        baseline_pred = pd.read_parquet(baseline_path)
        v3_ic = compute_ic_stats(v3_pred)
        base_ic = compute_ic_stats(baseline_pred)
        print(f"{'Metric':<25} {'v3 pruned (112f)':>15} {'baseline (110f)':>15} {'Diff':>10}")
        print("-" * 65)
        for k in ["avg_rank_ic", "icir", "positive_ic_pct", "n_days"]:
            v = v3_ic[k]
            b = base_ic[k]
            d = v - b
            if k == "positive_ic_pct":
                print(f"{k:<25} {v:>14.2%} {b:>14.2%} {d:>+9.2%}")
            elif k == "n_days":
                print(f"{k:<25} {v:>15.0f} {b:>15.0f} {d:>+10.0f}")
            else:
                print(f"{k:<25} {v:>15.4f} {b:>15.4f} {d:>+10.4f}")
    else:
        v3_ic = compute_ic_stats(v3_pred)
        print(f"v3 pruned RankIC: {v3_ic['avg_rank_ic']:.4f}")
        print(f"v3 pruned ICIR:   {v3_ic['icir']:.4f}")
        print(f"(baseline not found at {baseline_path})")

    print("\n" + "=" * 60)
    print("Step 3: Per-window IC")
    print("=" * 60)
    for f in sorted(v3_pred_dir.glob("wf*_pred.parquet")):
        df = pd.read_parquet(f)
        df = df.dropna(subset=["pred", "label_rank_5d"])
        if len(df) == 0:
            continue
        daily_ic = df.groupby(level="datetime").apply(
            lambda x: x["pred"].corr(x["label_rank_5d"], method="spearman")
        )
        year = f.name.split("_")[-1].replace("_pred.parquet", "")
        print(f"  {year}: RankIC={daily_ic.mean():.4f}, ICIR={daily_ic.mean()/daily_ic.std():.2f}, "
              f"days={len(daily_ic)}")


if __name__ == "__main__":
    main()
