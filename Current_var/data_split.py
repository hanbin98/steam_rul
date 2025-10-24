# -*- coding: utf-8 -*-
"""
Created on Mon Aug  4 13:01:45 2025

@author: user
"""

#!/usr/bin/env python
# split_dataset.py
"""
Split the CSV files in  data/all  into train / test folders
(80 % / 20 %) using a fixed random seed.

Run:
    python split_dataset.py --seed 42          # default seed = 42
    python split_dataset.py --seed 123 --ratio 0.75
"""

import argparse, random, shutil
from pathlib import Path

# ────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data",
                   help="Root data directory (contains all/ train/ test/)")
    p.add_argument("--ratio", type=float, default=0.8,
                   help="Train-set ratio (e.g. 0.8 → 80 % train, 20 % test)")
    p.add_argument("--seed", type=int, default=71,
                   help="Random seed for reproducible split")
    return p.parse_args()

# ────────────────────────────────
def main():
    args = parse_args()
    root   = Path(args.data_dir)
    all_dir   = root / "all"
    train_dir = root / "train"
    test_dir  = root / "test"

    # gather *.csv
    csv_files = sorted(all_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {all_dir.resolve()}")

    # reproducible shuffle & split
    random.seed(args.seed)
    random.shuffle(csv_files)
    split_idx = int(len(csv_files) * args.ratio)
    train_files = csv_files[:split_idx]
    test_files  = csv_files[split_idx:]

    # (re-)create output folders — empty them first if they exist
    for d in (train_dir, test_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # copy files
    for src in train_files:
        shutil.copy2(src, train_dir / src.name)
    for src in test_files:
        shutil.copy2(src, test_dir / src.name)

    print(f"✔  Copied {len(train_files)} files to {train_dir}")
    print(f"✔  Copied {len(test_files)}  files to {test_dir}")
    print("Done.")

# ────────────────────────────────
if __name__ == "__main__":
    main()
