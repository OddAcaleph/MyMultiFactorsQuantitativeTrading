"""Update ST columns in wide_table daily bars with corrected ST status.

Fixes退市股 (e.g. "东通退") not being marked as ST/*ST.
Only updates st_stock, star_st_stock, no_st_stock columns.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def update_wide_table_st(
    wide_table_dir: str | Path,
    st_file: str | Path,
    start_year: int = 2020,
) -> dict:
    wide_table_dir = Path(wide_table_dir)
    st_file = Path(st_file)

    print(f"Loading ST data from {st_file}...")
    st_df = pd.read_parquet(st_file)
    st_df["trade_date"] = st_df["trade_date"].astype(int)
    st_df = st_df.set_index(["trade_date", "ts_code"])
    st_df = st_df[["no_st_stock", "st_stock", "star_st_stock"]].astype("int8")
    print(f"  ST data loaded: {st_df.shape[0]:,} rows")

    # Find all parquet files
    files = sorted(wide_table_dir.rglob("*.parquet"))
    print(f"Found {len(files)} parquet files in {wide_table_dir}")

    # Filter by year if needed
    if start_year:
        files = [f for f in files if any(f"year={y}" in str(f) for y in range(start_year, 2027))]
        print(f"  Processing {len(files)} files from year={start_year}+")

    updated = 0
    total_rows = 0
    changed_rows = 0
    t0 = time.time()

    for i, fpath in enumerate(files):
        df = pd.read_parquet(fpath)

        # Check if ST columns exist
        if "st_stock" not in df.columns:
            continue

        # Build index for merge
        df_indexed = df.set_index(["trade_date", "ts_code"])
        orig_st = df_indexed[["no_st_stock", "st_stock", "star_st_stock"]].copy()

        # Join new ST values
        merged = df_indexed.join(st_df, how="left", rsuffix="_new")

        # Count changes
        changed_mask = (
            (merged["st_stock"] != merged["st_stock_new"].fillna(merged["st_stock"])) |
            (merged["star_st_stock"] != merged["star_st_stock_new"].fillna(merged["star_st_stock"])) |
            (merged["no_st_stock"] != merged["no_st_stock_new"].fillna(merged["no_st_stock"]))
        )
        n_changed = int(changed_mask.sum())

        if n_changed == 0:
            continue

        # Update ST columns
        df_indexed["no_st_stock"] = merged["no_st_stock_new"].fillna(merged["no_st_stock"]).astype("int8")
        df_indexed["st_stock"] = merged["st_stock_new"].fillna(merged["st_stock"]).astype("int8")
        df_indexed["star_st_stock"] = merged["star_st_stock_new"].fillna(merged["star_st_stock"]).astype("int8")

        # Reset index back
        df_updated = df_indexed.reset_index()

        # Ensure column order matches original
        df_updated = df_updated[df.columns]

        # Write back
        df_updated.to_parquet(fpath, index=False)

        updated += 1
        total_rows += len(df)
        changed_rows += n_changed

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  Processed {i+1}/{len(files)} files, {updated} updated, {changed_rows:,} rows changed, {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nDone. Updated {updated}/{len(files)} files, {changed_rows:,} rows changed in {elapsed:.1f}s")

    return {
        "files_total": len(files),
        "files_updated": updated,
        "rows_changed": changed_rows,
        "elapsed_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Update ST columns in wide table daily bars.")
    parser.add_argument("--wide-table-dir", required=True, help="Wide table daily bars directory.")
    parser.add_argument("--st-file", default="data/processd_data/namechange/namechange_st_daily.parquet",
                        help="Processed ST daily parquet file.")
    parser.add_argument("--start-year", type=int, default=2020, help="Only update files from this year onward.")
    args = parser.parse_args()

    update_wide_table_st(
        wide_table_dir=args.wide_table_dir,
        st_file=args.st_file,
        start_year=args.start_year,
    )


if __name__ == "__main__":
    main()
