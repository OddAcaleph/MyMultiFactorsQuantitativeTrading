"""L32 orthogonal sweep runner for optimizer backtests.

Usage:
    PYTHONPATH=src python scripts/run_optimizer_L32_sweep.py --n-workers 4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
SRC = BASE / "src"

PRED_PATH = BASE / "output/walk_forward_industry_v2/predictions/walk_forward_full_pred.parquet"
RISK_DIR = BASE / "outputs/risk_model/v1"
BARS_DIR = BASE / "data/cleaned_data/daily_bars"
OUT_BASE = BASE / "output/optimizer_sweep/industry_v2_L32"

# L32 orthogonal array: 8 factors x 4 levels + 1 factor x 2 levels
# Column mapping: A=col1, B=col2, C=col3, D=col4, E=col5, F=col6, G=col7, H=col8, I=col9
L32 = [
    #  A  B  C  D  E  F  G  H  I
    (0, 0, 0, 0, 0, 0, 0, 0, 0),  # 1
    (0, 0, 0, 0, 1, 1, 1, 1, 1),  # 2
    (0, 1, 1, 1, 0, 1, 1, 2, 0),  # 3
    (0, 1, 1, 1, 1, 0, 0, 3, 1),  # 4
    (0, 2, 2, 2, 2, 2, 2, 0, 0),  # 5
    (0, 2, 2, 2, 3, 3, 3, 1, 1),  # 6
    (0, 3, 3, 3, 2, 3, 3, 2, 0),  # 7
    (0, 3, 3, 3, 3, 2, 2, 3, 1),  # 8
    (1, 0, 1, 2, 2, 1, 2, 3, 0),  # 9
    (1, 0, 1, 2, 3, 2, 3, 2, 1),  # 10
    (1, 1, 0, 0, 2, 0, 3, 1, 0),  # 11
    (1, 1, 0, 0, 3, 1, 2, 0, 1),  # 12
    (1, 2, 3, 3, 0, 3, 0, 3, 0),  # 13
    (1, 2, 3, 3, 1, 2, 1, 2, 1),  # 14
    (1, 3, 2, 1, 0, 0, 1, 1, 0),  # 15
    (1, 3, 2, 1, 1, 1, 0, 0, 1),  # 16
    (2, 0, 2, 3, 1, 3, 1, 3, 0),  # 17
    (2, 0, 2, 3, 2, 1, 0, 2, 1),  # 18
    (2, 1, 3, 2, 1, 2, 0, 1, 0),  # 19
    (2, 1, 3, 2, 2, 3, 1, 0, 1),  # 20
    (2, 2, 0, 1, 3, 0, 3, 3, 0),  # 21
    (2, 2, 0, 1, 0, 1, 2, 2, 1),  # 22
    (2, 3, 1, 0, 3, 2, 2, 1, 0),  # 23
    (2, 3, 1, 0, 0, 0, 3, 0, 1),  # 24
    (3, 0, 3, 1, 3, 0, 3, 2, 0),  # 25
    (3, 0, 3, 1, 0, 3, 2, 3, 1),  # 26
    (3, 1, 2, 0, 3, 1, 0, 0, 0),  # 27
    (3, 1, 2, 0, 0, 0, 1, 3, 1),  # 28
    (3, 2, 1, 3, 1, 2, 1, 2, 0),  # 29
    (3, 2, 1, 3, 2, 0, 0, 3, 1),  # 30
    (3, 3, 0, 2, 1, 3, 3, 0, 0),  # 31
    (3, 3, 0, 2, 2, 1, 2, 1, 1),  # 32
]

# Factor levels — all ranges verified feasible across 2010-2024
# Feasibility constraint: N_industries * ind_max + pool*max_weight must exceed 100%
# With 26 SW L1 industries and ~30-40% stocks having no industry exposure:
#   pool=30 + ind_max=2% + max_w=5% → 83% feasible (borderline)
#   pool=50 + ind_max=2% + max_w=5% → 100% feasible
FACTORS = {
    "A": {"name": "pool_size", "levels": [30, 50, 80, 120], "flag": "--pool-size"},
    "B": {"name": "risk_aversion", "levels": [0.0001, 0.0005, 0.002, 0.01], "flag": "--risk-aversion"},
    "C": {"name": "turnover_penalty", "levels": [0.05, 0.15, 0.3, 0.6], "flag": "--turnover-penalty"},
    "D": {"name": "rebalance_freq", "levels": [2, 5, 10, 20], "flag": "--rebalance-freq"},
    "E": {"name": "industry_max_weight", "levels": [0.02, 0.05, 0.10, 0.20], "flag": "--industry-max-weight"},
    "F": {"name": "max_weight", "levels": [0.05, 0.10, 0.15, 0.20], "flag": "--max-weight"},
    "G": {"name": "alpha_smooth_span", "levels": [3, 5, 10, 20], "flag": "--alpha-smooth-span"},
    "H": {"name": "ft_alpha_mom_min_scale", "levels": [0.3, 0.5, 0.7, 0.9], "flag": "--ft-alpha-mom-min-scale"},
    "I": {"name": "trend_filter", "levels": ["off", "on"], "flag": "--trend-ma-days"},
}


def exp_name(idx: int, levels: tuple) -> str:
    codes = []
    for i, (key, fac) in enumerate(FACTORS.items()):
        val = fac["levels"][levels[i]]
        if key == "A":
            codes.append(f"pool{val}")
        elif key == "B":
            codes.append(f"ra{val:.4f}")
        elif key == "C":
            codes.append(f"tp{val:.2f}")
        elif key == "D":
            codes.append(f"reb{val}d")
        elif key == "E":
            codes.append(f"ind{val*100:.0f}pct")
        elif key == "F":
            codes.append(f"mw{val*100:.0f}pct")
        elif key == "G":
            codes.append(f"sm{val}")
        elif key == "H":
            codes.append(f"ftmin{val:.2f}")
        elif key == "I":
            codes.append(f"trend{val}")
    return f"exp{idx:03d}_{'_'.join(codes)}"


def build_cmd(idx: int, levels: tuple) -> list[str]:
    name = exp_name(idx, levels)
    out_dir = OUT_BASE / "experiments" / name

    cmd = [
        sys.executable, str(SRC / "optimizer" / "run_optimizer_backtest.py"),
        "--pred-path", str(PRED_PATH),
        "--risk-dir", str(RISK_DIR),
        "--bars-dir", str(BARS_DIR),
        "--output-dir", str(out_dir),
        "--filter-st",
        "--filter-suspend",
        "--filter-new-stock-days", "60",
        "--signal-delay", "1",
        "--board-lot", "100",
        "--slippage", "0.003",
        "--end-date", "2025-12-31",
        "--factor-timing",
        "--ft-alpha-momentum",
        "--ft-alpha-mom-lookback", "20",
        "--ft-alpha-mom-max-scale", "1.4",
    ]

    for i, (key, fac) in enumerate(FACTORS.items()):
        val = fac["levels"][levels[i]]
        if key == "I":
            if val == "on":
                cmd += ["--trend-ma-days", "20", "--trend-ma-bear-position", "0.45", "--trend-ma-smooth"]
            # off: don't add trend flags
        else:
            cmd += [fac["flag"], str(val)]

    return cmd


def run_experiment(args: tuple[int, tuple]) -> dict:
    idx, levels = args
    name = exp_name(idx, levels)
    out_dir = OUT_BASE / "experiments" / name
    metrics_file = out_dir / "metrics.json"

    if metrics_file.exists():
        with open(metrics_file) as f:
            m = json.load(f)
        return {"idx": idx, "name": name, "status": "cached", "sharpe": m.get("sharpe_ratio", 0)}

    cmd = build_cmd(idx, levels)
    t0 = time.time()
    env = {"PYTHONPATH": str(SRC), "PATH": __import__("os").environ.get("PATH", "")}
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(BASE))
    elapsed = time.time() - t0

    if result.returncode != 0:
        # save stderr
        err_file = out_dir / "error.log"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(err_file, "w") as f:
            f.write(result.stderr[-5000:])
        return {"idx": idx, "name": name, "status": "failed", "error": result.stderr[-500:], "elapsed": elapsed}

    if metrics_file.exists():
        with open(metrics_file) as f:
            m = json.load(f)
        return {"idx": idx, "name": name, "status": "done", "sharpe": m.get("sharpe_ratio", 0),
                "annual_return": m.get("annual_return", 0), "max_drawdown": m.get("max_drawdown", 0),
                "sortino": m.get("sortino_ratio", 0), "calmar": m.get("calmar_ratio", 0),
                "turnover": m.get("avg_turnover", 0), "holdings": m.get("avg_holdings", 0),
                "elapsed": elapsed}

    return {"idx": idx, "name": name, "status": "no_metrics", "elapsed": elapsed}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    experiments = [(i + 1, levels) for i, levels in enumerate(L32)]

    if args.dry_run:
        print(f"L32 orthogonal sweep: {len(experiments)} experiments")
        print()
        for idx, levels in experiments[:5]:
            cmd = build_cmd(idx, levels)
            print(f"Exp {idx}: {exp_name(idx, levels)}")
            print(f"  {' '.join(cmd[2:8])} ...")
        print(f"... ({len(experiments) - 5} more)")
        return

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    # Save experiment table
    table = []
    for idx, levels in experiments:
        row = {"exp": idx, "name": exp_name(idx, levels)}
        for i, (key, fac) in enumerate(FACTORS.items()):
            row[fac["name"]] = fac["levels"][levels[i]]
        table.append(row)
    with open(OUT_BASE / "experiment_table.json", "w") as f:
        json.dump(table, f, indent=2, default=str)

    print(f"Starting L32 sweep: {len(experiments)} experiments, {args.n_workers} workers")
    print(f"Output: {OUT_BASE}")
    print()

    results = []
    done = 0
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {executor.submit(run_experiment, exp): exp for exp in experiments}
        for future in as_completed(futures):
            exp = futures[future]
            try:
                result = future.result()
                results.append(result)
                done += 1
                status_icon = {"done": "✓", "cached": "⊙", "failed": "✗", "no_metrics": "?"}.get(result["status"], "?")
                sharpe_str = f" sharpe={result['sharpe']:.3f}" if result["status"] == "done" else ""
                elapsed_str = f" ({result['elapsed']:.0f}s)" if "elapsed" in result else ""
                print(f"  [{done:2d}/32] {status_icon} {result['name']}{sharpe_str}{elapsed_str}", flush=True)
            except Exception as e:
                print(f"  [{done:2d}/32] ✗ exp{exp[0]:03d} ERROR: {e}", flush=True)

    # Sort by index
    results.sort(key=lambda r: r["idx"])

    # Save summary
    summary = []
    for r in results:
        row = {"exp": r["idx"], "name": r["name"], "status": r["status"]}
        for k in ["sharpe", "annual_return", "max_drawdown", "sortino", "calmar", "turnover", "holdings", "elapsed"]:
            row[k] = r.get(k, None)
        summary.append(row)

    with open(OUT_BASE / "results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save CSV
    import csv
    csv_path = OUT_BASE / "results_summary.csv"
    if summary:
        keys = list(summary[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(summary)

    total_time = time.time() - t_start
    n_success = sum(1 for r in results if r["status"] == "done")
    best = max((r for r in results if r["status"] == "done"), key=lambda r: r["sharpe"], default=None)

    print()
    print(f"=" * 60)
    print(f"Sweep complete: {n_success}/{len(results)} succeeded in {total_time:.0f}s")
    if best:
        print(f"Best sharpe: {best['sharpe']:.3f} ({best['name']})")
    print(f"Results saved to {OUT_BASE}")
    print(f"=" * 60)


if __name__ == "__main__":
    main()
