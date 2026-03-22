#!/usr/bin/env python3
"""
Usage:
  python3 sample_global_data.py --path data --n 1000
"""
from __future__ import annotations
import os
import csv
import argparse
import random
from pathlib import Path
from typing import Iterator, Dict, List, Tuple, Set

# use station ids by default for pairing
START_COL = "start_station_id"
END_COL = "end_station_id"


def iter_csv_files(path: str) -> Iterator[Path]:
    p = Path(path)
    if p.is_file():
        yield p
        return
    for root, _, files in os.walk(p):
        for fn in files:
            if fn.lower().endswith(".csv"):
                yield Path(root) / fn


def sample_and_count(path: str, n: int) -> Tuple[List[Dict[str, str]], int, Set[Tuple[str, str]]]:
    reservoir: List[Dict[str, str]] = []
    total_rows = 0
    pairs: Set[Tuple[str, str]] = set()

    for fp in iter_csv_files(path):
        try:
            with fp.open("r", newline="", encoding="utf-8", errors="replace") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    total_rows += 1

                    # update unordered pairs (only when both endpoints present)
                    a = (row.get(START_COL) or "").strip()
                    b = (row.get(END_COL) or "").strip()
                    if a and b:
                        pair = (a, b) if a <= b else (b, a)
                        pairs.add(pair)

                    # reservoir sampling over all rows
                    if len(reservoir) < n:
                        reservoir.append(row.copy())
                    else:
                        # replace with probability n/total_rows
                        r = random.randrange(total_rows)
                        if r < n:
                            reservoir[r] = row.copy()
        except Exception:
            # quick-and-dirty: skip unreadable files
            continue

    return reservoir, total_rows, pairs


def write_sample(out_path: str, rows: List[Dict[str, str]]) -> None:
    if not rows:
        print("No rows sampled; output file not written.")
        return
    fieldnames = list(rows[0].keys())
    out_p = Path(out_path)
    with out_p.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample N rows from CSVs under a folder and count unique unordered OD pairs.")
    p.add_argument("--path", default="./data", help="File or directory to scan (default: data)")
    p.add_argument("--n", type=int, required=True, help="Number of rows to sample")
    p.add_argument("--out", default="citibike_sampled_data.csv", help="Output CSV path (default: citibike_sampled_data.csv)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Sampling {args.n} rows from CSVs under: {args.path}")
    reservoir, total_rows, pairs = sample_and_count(args.path, args.n)

    write_sample(args.out, reservoir)

    # compute unique pairs in the sampled set
    sample_pairs: Set[Tuple[str, str]] = set()
    for row in reservoir:
        a = (row.get(START_COL) or "").strip()
        b = (row.get(END_COL) or "").strip()
        if a and b:
            pair = (a, b) if a <= b else (b, a)
            sample_pairs.add(pair)

    print(f"Total rows scanned: {total_rows}")
    print(f"Sampled rows written: {len(reservoir)} -> {args.out}")
    print(f"Unique unordered origin/destination pairs (full set): {len(pairs)}")
    print(f"Unique unordered origin/destination pairs (sample): {len(sample_pairs)}")


if __name__ == "__main__":
    main()
