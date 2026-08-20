"""Create and audit the fixed fold assignment shared by final experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from kaggle_workflow.cv import group_overlap_by_fold, make_fold_assignments, save_folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("input/train.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/folds_group_5_seed42.csv"))
    parser.add_argument(
        "--strategy", choices=["stratified", "stratified_group"], default="stratified_group"
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = pd.read_csv(args.input)
    folds = make_fold_assignments(
        train,
        strategy=args.strategy,
        n_splits=args.n_splits,
        seed=args.seed,
    )
    save_folds(train, folds, args.output)
    print(f"saved: {args.output}")
    print(f"rows per fold: {pd.Series(folds).value_counts().sort_index().to_dict()}")
    print(f"group overlap per fold: {group_overlap_by_fold(train, folds)}")


if __name__ == "__main__":
    main()
