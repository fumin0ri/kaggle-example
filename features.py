"""Feature schema and leakage-safe nested target encoding."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


PAIR_SEPARATOR = "__PAIR__"
MISSING_TOKEN = "__MISSING__"


def get_base_features(
    train: pd.DataFrame,
    target: str,
    id_column: str,
    min_nunique: int = 5,
) -> tuple[list[str], list[str], list[str]]:
    """Split predictors into numeric and categorical model inputs.

    Low-cardinality integer columns are treated as categorical. In this data,
    for example, ``SeniorCitizen`` is encoded as 0/1 but represents a category.
    """
    excluded = {target, id_column, "fold"}
    features = [column for column in train.columns if column not in excluded]
    numeric = [
        column
        for column in features
        if pd.api.types.is_numeric_dtype(train[column])
        and train[column].nunique(dropna=True) >= min_nunique
    ]
    categorical = [column for column in features if column not in numeric]
    return features, numeric, categorical


def get_categorical_pairs(categorical: list[str]) -> list[tuple[str, str]]:
    """Return every unique two-column categorical interaction."""
    return list(combinations(categorical, 2))


def _make_pair_key(frame: pd.DataFrame, left: str, right: str) -> pd.Series:
    left_values = frame[left].astype("string").fillna(MISSING_TOKEN)
    right_values = frame[right].astype("string").fillna(MISSING_TOKEN)
    return left_values + PAIR_SEPARATOR + right_values


def _fit_target_rate_map(
    keys: pd.Series,
    target: pd.Series,
    smoothing: float,
) -> tuple[pd.Series, np.float32]:
    prior = np.float32(target.mean())
    statistics = (
        pd.DataFrame(
            {
                "key": keys.to_numpy(),
                "target": target.to_numpy(dtype=np.float32),
            }
        )
        .groupby("key", sort=False)["target"]
        .agg(["sum", "count"])
    )
    mapping = (
        (statistics["sum"] + smoothing * prior)
        / (statistics["count"] + smoothing)
    ).astype("float32")
    return mapping, prior


def _apply_target_rate_map(
    keys: pd.Series,
    mapping: pd.Series,
    prior: np.float32,
) -> np.ndarray:
    return keys.map(mapping).fillna(float(prior)).to_numpy(dtype=np.float32)


def build_nested_pair_te_for_outer_fold(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str,
    categorical_pairs: list[tuple[str, str]],
    outer_fold: int,
    id_column: str = "id",
    fold_column: str = "fold",
    n_inner_splits: int = 5,
    smoothing: float = 20.0,
    random_state: int = 42,
    verbose_every: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create leakage-safe pair target encodings for one outer fold.

    Outer-training rows receive inner out-of-fold encodings. Outer-valid and
    test rows receive mappings fitted only on the complete outer-training
    partition. An outer validation label can therefore never affect its own
    feature value.
    """
    train_mask = train[fold_column].ne(outer_fold)
    valid_mask = train[fold_column].eq(outer_fold)
    outer_train = train.loc[train_mask]
    outer_valid = train.loc[valid_mask]
    y_outer = outer_train[target].astype("float32")

    inner_splits = min(n_inner_splits, int(y_outer.value_counts().min()))
    if inner_splits < 2:
        raise ValueError(f"outer_fold={outer_fold}: not enough rows for inner CV")

    inner_fold_ids = np.full(len(outer_train), -1, dtype=np.int16)
    inner_splitter = StratifiedKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=random_state,
    )
    for inner_fold, (_, inner_valid_idx) in enumerate(
        inner_splitter.split(outer_train, y_outer)
    ):
        inner_fold_ids[inner_valid_idx] = inner_fold

    train_features: dict[str, np.ndarray] = {
        id_column: outer_train[id_column].to_numpy()
    }
    valid_features: dict[str, np.ndarray] = {
        id_column: outer_valid[id_column].to_numpy()
    }
    test_features: dict[str, np.ndarray] = {id_column: test[id_column].to_numpy()}

    for pair_number, (left, right) in enumerate(categorical_pairs, start=1):
        feature_name = f"TE__{left}__{right}"
        train_keys = _make_pair_key(outer_train, left, right)
        valid_keys = _make_pair_key(outer_valid, left, right)
        test_keys = _make_pair_key(test, left, right)

        encoded_train = np.empty(len(outer_train), dtype=np.float32)
        for inner_fold in range(inner_splits):
            inner_valid_mask = inner_fold_ids == inner_fold
            inner_fit_mask = ~inner_valid_mask
            mapping, prior = _fit_target_rate_map(
                train_keys.iloc[inner_fit_mask],
                y_outer.iloc[inner_fit_mask],
                smoothing,
            )
            encoded_train[inner_valid_mask] = _apply_target_rate_map(
                train_keys.iloc[inner_valid_mask], mapping, prior
            )

        outer_mapping, outer_prior = _fit_target_rate_map(
            train_keys, y_outer, smoothing
        )
        train_features[feature_name] = encoded_train
        valid_features[feature_name] = _apply_target_rate_map(
            valid_keys, outer_mapping, outer_prior
        )
        test_features[feature_name] = _apply_target_rate_map(
            test_keys, outer_mapping, outer_prior
        )

        if (
            pair_number == 1
            or pair_number % verbose_every == 0
            or pair_number == len(categorical_pairs)
        ):
            print(
                f"outer_fold={outer_fold} | "
                f"{pair_number:>4}/{len(categorical_pairs)} pairs"
            )

    return (
        pd.DataFrame(train_features),
        pd.DataFrame(valid_features),
        pd.DataFrame(test_features),
    )


def load_nested_pair_te(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    id_column: str,
    fold_column: str,
    outer_fold: int,
    train_mask: pd.Series,
    valid_mask: pd.Series,
    feature_dir: str | Path,
    selected_te_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and align one outer fold's saved target-encoding features."""
    del fold_column  # Kept in the loader interface used by run_cv.
    feature_dir = Path(feature_dir)
    fold_tag = str(outer_fold).replace("/", "_")

    train_te = pd.read_parquet(feature_dir / f"outer_fold_{fold_tag}_train.parquet")
    valid_te = pd.read_parquet(feature_dir / f"outer_fold_{fold_tag}_valid.parquet")
    test_te = pd.read_parquet(feature_dir / f"outer_fold_{fold_tag}_test.parquet")

    available_te_features = [
        column for column in train_te.columns if column != id_column
    ]
    expected_columns = {id_column, *available_te_features}
    if set(valid_te.columns) != expected_columns or set(test_te.columns) != expected_columns:
        raise ValueError(f"outer_fold={outer_fold}: inconsistent TE columns")

    te_features = (
        available_te_features
        if selected_te_features is None
        else list(selected_te_features)
    )
    missing_features = set(te_features) - set(available_te_features)
    if missing_features:
        raise ValueError(
            f"outer_fold={outer_fold}: requested TE features are missing: "
            f"{sorted(missing_features)}"
        )

    train_te = train_te[[id_column, *te_features]]
    valid_te = valid_te[[id_column, *te_features]]
    test_te = test_te[[id_column, *te_features]]

    duplicated = set(features) & set(te_features)
    if duplicated:
        raise ValueError(f"base and TE features overlap: {sorted(duplicated)}")

    def align(ids: pd.Series, encoded: pd.DataFrame, split_name: str) -> pd.DataFrame:
        aligned = ids.reset_index(drop=True).to_frame(name=id_column).merge(
            encoded,
            on=id_column,
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        missing = aligned["_merge"].ne("both")
        if missing.any():
            examples = aligned.loc[missing, id_column].head(10).tolist()
            raise ValueError(
                f"outer_fold={outer_fold}: {split_name} IDs missing from TE: {examples}"
            )
        result = aligned.drop(columns=[id_column, "_merge"]).reset_index(drop=True)
        if not np.isfinite(result.to_numpy()).all():
            raise ValueError(f"outer_fold={outer_fold}: non-finite {split_name} TE")
        return result

    base_train = train.loc[train_mask, features].reset_index(drop=True).copy()
    base_valid = train.loc[valid_mask, features].reset_index(drop=True).copy()
    base_test = test[features].reset_index(drop=True).copy()
    aligned_train = align(train.loc[train_mask, id_column], train_te, "train")
    aligned_valid = align(train.loc[valid_mask, id_column], valid_te, "valid")
    aligned_test = align(test[id_column], test_te, "test")

    return (
        pd.concat([base_train, aligned_train], axis=1),
        pd.concat([base_valid, aligned_valid], axis=1),
        pd.concat([base_test, aligned_test], axis=1),
    )
