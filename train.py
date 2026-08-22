"""Reusable preprocessing, model factories, and cross-validation runner."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


MODEL_ORDER = [
    "CatBoost",
    "XGBoost",
    "ExtraTrees",
    "LogisticRegression",
    "MLP",
]


def make_onehot_encoder() -> OneHotEncoder:
    """Create a dense encoder across old and new scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), numeric_features),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", make_onehot_encoder()),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def make_model(
    model_name: str,
    numeric_features: list[str],
    categorical_features: list[str],
    seed: int = 42,
    use_gpu: bool = False,
    model_params: dict[str, object] | None = None,
):
    """Build the shared baseline definition used in comparison notebooks."""
    if model_name == "CatBoost":
        parameters = {
            "iterations": 500,
            "depth": 5,
            "learning_rate": 0.03,
            "l2_leaf_reg": 5.0,
            "loss_function": "Logloss",
            "random_seed": seed,
            "verbose": False,
            "allow_writing_files": False,
            "cat_features": categorical_features,
        }
        if model_params:
            parameters.update(model_params)
        if use_gpu:
            parameters.update(task_type="GPU", devices="0")
        return CatBoostClassifier(**parameters)

    if model_name == "XGBoost":
        parameters = {
            "n_estimators": 500,
            "max_depth": 4,
            "learning_rate": 0.03,
            "min_child_weight": 2,
            "subsample": 0.90,
            "colsample_bytree": 0.90,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "random_state": seed,
            "n_jobs": -1,
        }
        if model_params:
            parameters.update(model_params)
        if use_gpu:
            parameters["device"] = "cuda"
        estimator = XGBClassifier(**parameters)
        scale_numeric = False
    elif model_name == "ExtraTrees":
        parameters = {
            "n_estimators": 500,
            "min_samples_leaf": 2,
            "max_features": "sqrt",
            "random_state": seed,
            "n_jobs": -1,
        }
        if model_params:
            parameters.update(model_params)
        estimator = ExtraTreesClassifier(**parameters)
        scale_numeric = False
    elif model_name == "LogisticRegression":
        parameters = {
            "C": 1.0,
            "max_iter": 3000,
            "solver": "lbfgs",
            "random_state": seed,
        }
        if model_params:
            parameters.update(model_params)
        estimator = LogisticRegression(**parameters)
        scale_numeric = True
    elif model_name == "MLP":
        parameters = {
            "hidden_layer_sizes": (32, 16),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-3,
            "batch_size": 64,
            "learning_rate_init": 1e-3,
            "max_iter": 1000,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "n_iter_no_change": 30,
            "random_state": seed,
        }
        if model_params:
            parameters.update(model_params)
        estimator = MLPClassifier(**parameters)
        scale_numeric = True
    else:
        raise ValueError(f"unknown model: {model_name}")

    return Pipeline(
        [
            (
                "preprocess",
                make_preprocessor(
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    scale_numeric=scale_numeric,
                ),
            ),
            ("model", estimator),
        ]
    )


def prepare_catboost_frame(
    frame: pd.DataFrame,
    features: list[str],
    categorical_features: list[str],
) -> pd.DataFrame:
    """Select model columns and make CatBoost categories explicit strings."""
    result = frame[features].copy()
    for column in categorical_features:
        result[column] = (
            result[column].astype("string").fillna("__MISSING__").astype(str)
        )
    return result


FoldFeatureLoader = Callable[..., tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]


def run_cv(
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    target: str,
    id_column: str,
    fold_column: str = "fold",
    label: str = "cv",
    save_prefix: str | None = None,
    output_dir: str | Path | None = None,
    fold_feature_loader: FoldFeatureLoader | None = None,
    return_models: bool = False,
    predict_test: bool = True,
) -> dict[str, object]:
    """Run fixed-fold CV and optionally save aligned OOF/test predictions."""
    required = {target, id_column, fold_column}
    missing = required - set(train.columns)
    if missing:
        raise KeyError(f"train is missing required columns: {sorted(missing)}")
    if train[fold_column].isna().any():
        raise ValueError("fold column contains missing values")
    if train[id_column].duplicated().any() or test[id_column].duplicated().any():
        raise ValueError("ID column must be unique in train and test")

    y = train[target]
    folds = sorted(train[fold_column].unique())
    if len(folds) < 2:
        raise ValueError("at least two folds are required")

    if fold_feature_loader is None:
        all_train_features = train[features]
        all_test_features = test[features]

    oof = np.full(len(train), np.nan, dtype=float)
    test_pred = np.zeros(len(test), dtype=float) if predict_test else None
    fold_rows: list[dict[str, object]] = []
    models = []

    for fold in folds:
        train_mask = train[fold_column].ne(fold)
        valid_mask = train[fold_column].eq(fold)
        y_train = y.loc[train_mask]
        y_valid = y.loc[valid_mask]

        if fold_feature_loader is None:
            x_train = all_train_features.loc[train_mask]
            x_valid = all_train_features.loc[valid_mask]
            x_test = all_test_features
        else:
            x_train, x_valid, x_test = fold_feature_loader(
                train=train,
                test=test,
                features=features,
                id_column=id_column,
                fold_column=fold_column,
                outer_fold=fold,
                train_mask=train_mask,
                valid_mask=valid_mask,
            )

        expected_rows = (int(train_mask.sum()), int(valid_mask.sum()), len(test))
        actual_rows = (len(x_train), len(x_valid), len(x_test))
        if actual_rows != expected_rows:
            raise ValueError(
                f"fold={fold}: feature row counts {actual_rows} != {expected_rows}"
            )

        try:
            fitted_model = clone(model)
        except RuntimeError:
            fitted_model = deepcopy(model)
        fitted_model.fit(x_train, y_train)

        valid_pred = fitted_model.predict_proba(x_valid)[:, 1]
        oof[valid_mask.to_numpy()] = valid_pred
        if predict_test:
            fold_test_pred = fitted_model.predict_proba(x_test)[:, 1]
            test_pred += fold_test_pred / len(folds)

        fold_auc = roc_auc_score(y_valid, valid_pred)
        fold_rows.append(
            {
                "cv": label,
                "fold": fold,
                "n_train": len(x_train),
                "n_valid": len(x_valid),
                "train_target_rate": y_train.mean(),
                "valid_target_rate": y_valid.mean(),
                "auc": fold_auc,
            }
        )
        if return_models:
            models.append(fitted_model)
        print(f"[{label}] fold={fold} AUC={fold_auc:.5f}")

    if np.isnan(oof).any():
        raise RuntimeError("OOF predictions are incomplete")

    fold_df = pd.DataFrame(fold_rows)
    oof_auc = roc_auc_score(y, oof)
    print(f"\n[{label}] OOF AUC={oof_auc:.5f}")
    print(
        f"fold mean={fold_df['auc'].mean():.5f}, "
        f"std={fold_df['auc'].std(ddof=0):.5f}"
    )
    print(fold_df)

    if save_prefix is not None:
        if not predict_test:
            raise ValueError("cannot save test predictions when predict_test=False")
        destination = Path(output_dir or "artifacts")
        destination.mkdir(parents=True, exist_ok=True)
        np.save(destination / f"{save_prefix}_oof.npy", oof)
        np.save(destination / f"{save_prefix}_test.npy", test_pred)
        pd.DataFrame(
            {
                id_column: train[id_column].to_numpy(),
                target: y.to_numpy(),
                fold_column: train[fold_column].to_numpy(),
                "oof_prob": oof,
            }
        ).to_csv(destination / f"{save_prefix}_oof.csv", index=False)
        pd.DataFrame(
            {id_column: test[id_column].to_numpy(), "pred_prob": test_pred}
        ).to_csv(destination / f"{save_prefix}_test.csv", index=False)
        fold_df.to_csv(destination / f"{save_prefix}_fold_scores.csv", index=False)
        pd.DataFrame(
            [
                {
                    "cv": label,
                    "oof_auc": oof_auc,
                    "fold_auc_mean": fold_df["auc"].mean(),
                    "fold_auc_std": fold_df["auc"].std(ddof=0),
                    "n_splits": len(folds),
                }
            ]
        ).to_csv(destination / f"{save_prefix}_summary.csv", index=False)
        print(f"saved to: {destination.resolve()}")

    return {
        "label": label,
        "oof": oof,
        "test_pred": test_pred,
        "fold_df": fold_df,
        "oof_auc": oof_auc,
        "models": models,
    }
