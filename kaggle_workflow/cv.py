"""Cross-validation helpers with explicit group-leakage checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def passenger_groups(passenger_ids: pd.Series) -> pd.Series:
    return passenger_ids.astype("string").str.split("_").str[0]


def make_fold_assignments(
    train: pd.DataFrame,
    *,
    target: str = "Transported",
    strategy: str = "stratified_group",
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Return one validation-fold integer per row."""
    y = train[target].astype(int).to_numpy()
    folds = np.full(len(train), -1, dtype=int)
    if strategy == "stratified":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iterator = splitter.split(train, y)
    elif strategy == "stratified_group":
        groups = passenger_groups(train["PassengerId"])
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iterator = splitter.split(train, y, groups)
    else:
        raise ValueError(f"Unknown CV strategy: {strategy}")

    for fold, (_, valid_index) in enumerate(split_iterator):
        folds[valid_index] = fold
    if (folds < 0).any():
        raise RuntimeError("Some rows were not assigned to a validation fold")
    return folds


def group_overlap_by_fold(train: pd.DataFrame, folds: np.ndarray) -> list[int]:
    """Count passenger groups appearing in both train and validation for each fold."""
    groups = passenger_groups(train["PassengerId"])
    overlaps: list[int] = []
    for fold in sorted(np.unique(folds)):
        train_groups = set(groups[folds != fold])
        valid_groups = set(groups[folds == fold])
        overlaps.append(len(train_groups & valid_groups))
    return overlaps


def save_folds(train: pd.DataFrame, folds: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"PassengerId": train["PassengerId"], "fold": folds}).to_csv(path, index=False)


def load_folds(train: pd.DataFrame, path: str | Path) -> np.ndarray:
    stored = pd.read_csv(path)
    expected = train["PassengerId"].astype(str).tolist()
    actual = stored["PassengerId"].astype(str).tolist()
    if actual != expected:
        raise ValueError("Saved fold PassengerId order does not match input/train.csv")
    return stored["fold"].to_numpy(dtype=int)
