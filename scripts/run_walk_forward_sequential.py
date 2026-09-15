"""Sequential walk-forward training driver.

Runs each window in a subprocess sequentially. More reliable than nohup
when the parent process environment is fragile.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
WF_SCRIPT = PROJECT_ROOT / "scripts" / "run_walk_forward.py"

NUM_WINDOWS = 21


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--start-window", type=int, default=0)
    parser.add_argument("--end-window", type=int, default=NUM_WINDOWS)
    parser.add_argument("--verbose", default="0")
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(SRC_ROOT) + ":" + env.get("PYTHONPATH", "")

    t0 = time.time()
    for i in range(args.start_window, args.end_window):
        print(f"\n[{i+1}/{NUM_WINDOWS}] Training window {i}...", flush=True)
        cmd = [
            sys.executable, str(WF_SCRIPT),
            "--config", args.config,
            "--train-window", str(i),
            "--cpu",
            "--verbose", args.verbose,
        ]
        result = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, env=env,
        )
        # Print non-gym stdout lines
        for line in result.stdout.strip().split("\n"):
            if line.strip() and "Gym" not in line and "gymnasium" not in line and "migration" not in line:
                print(f"  {line.strip()}", flush=True)
        if result.returncode != 0:
            print(f"  FAILED with code {result.returncode}", flush=True)
            print(f"  stderr: {result.stderr[-1000:]}", flush=True)
            sys.exit(1)

    print(f"\nAll {NUM_WINDOWS} windows done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
