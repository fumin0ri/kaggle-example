"""Validation helpers shared by the notebooks."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def add_stratified_folds(
    train: pd.DataFrame,
    target: str,
    n_splits: int = 5,
    random_state: int = 42,
    fold_column: str = "fold",
) -> pd.DataFrame:
    """Return a copy of ``train`` with deterministic stratified fold IDs.

    Keeping fold IDs in a file and reusing them for every model is essential:
    OOF scores and ensemble predictions are comparable only when each model
    predicts the same validation rows.
    """
    if target not in train:
        raise KeyError(f"target column is missing: {target}")
    if train[target].isna().any():
        raise ValueError(f"target column contains missing values: {target}")
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2")

    result = train.copy()
    result[fold_column] = -1
    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    fold_position = result.columns.get_loc(fold_column)
    for fold, (_, valid_idx) in enumerate(splitter.split(result, result[target])):
        result.iloc[valid_idx, fold_position] = fold

    if (result[fold_column] < 0).any():
        raise RuntimeError("some rows were not assigned to a fold")
    return result
