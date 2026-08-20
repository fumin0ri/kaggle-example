"""Model factories and fold-local preprocessing for diverse baselines."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def categorical_columns(frame: pd.DataFrame) -> list[str]:
    return frame.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()


def prepare_catboost_frame(frame: pd.DataFrame, cat_columns: list[str]) -> pd.DataFrame:
    prepared = frame.copy()
    for column in cat_columns:
        prepared[column] = prepared[column].astype("string").fillna("__MISSING__").astype(str)
    return prepared


def _normalize_categorical(frame):
    """Convert pandas nullable scalars to ordinary object/NaN for sklearn imputers."""
    if hasattr(frame, "astype"):
        return frame.astype(object).where(pd.notna(frame), np.nan)
    return frame


def _preprocessor(frame: pd.DataFrame, *, dense: bool, scale: bool) -> ColumnTransformer:
    cat_columns = categorical_columns(frame)
    numeric_columns = [column for column in frame.columns if column not in cat_columns]
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical_pipeline = Pipeline(
        [
            (
                "normalize_nullable",
                FunctionTransformer(_normalize_categorical, feature_names_out="one-to-one"),
            ),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=2,
                    sparse_output=not dense,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), numeric_columns),
            ("categorical", categorical_pipeline, cat_columns),
        ],
        sparse_threshold=0.0 if dense else 0.3,
        verbose_feature_names_out=False,
    )


def build_model(model_name: str, params: dict[str, Any], seed: int, frame: pd.DataFrame):
    """Build a native CatBoost model or a leakage-safe sklearn pipeline."""
    params = dict(params)
    if model_name == "catboost":
        from catboost import CatBoostClassifier

        params.setdefault("loss_function", "Logloss")
        params.setdefault("eval_metric", "Accuracy")
        params.setdefault("random_seed", seed)
        params.setdefault("allow_writing_files", False)
        params.setdefault("thread_count", -1)
        return CatBoostClassifier(**params)

    if model_name == "xgboost":
        from xgboost import XGBClassifier

        params.setdefault("random_state", seed)
        params.setdefault("n_jobs", -1)
        params.setdefault("eval_metric", "logloss")
        estimator = XGBClassifier(**params)
        return Pipeline(
            [("preprocess", _preprocessor(frame, dense=False, scale=False)), ("model", estimator)]
        )

    if model_name == "extratrees":
        params.setdefault("random_state", seed)
        params.setdefault("n_jobs", -1)
        params.setdefault("class_weight", "balanced")
        estimator = ExtraTreesClassifier(**params)
        return Pipeline(
            [("preprocess", _preprocessor(frame, dense=False, scale=False)), ("model", estimator)]
        )

    if model_name == "logistic":
        params.setdefault("random_state", seed)
        params.setdefault("max_iter", 2000)
        estimator = LogisticRegression(**params)
        return Pipeline(
            [("preprocess", _preprocessor(frame, dense=False, scale=True)), ("model", estimator)]
        )

    if model_name == "mlp":
        params.setdefault("random_state", seed)
        params.setdefault("max_iter", 300)
        params.setdefault("early_stopping", True)
        estimator = MLPClassifier(**params)
        return Pipeline(
            [("preprocess", _preprocessor(frame, dense=True, scale=True)), ("model", estimator)]
        )

    raise ValueError(f"Unknown model: {model_name}")


def fit_model(
    model,
    model_name: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    *,
    verbose: int | bool = 100,
    early_stopping_rounds: int | None = 100,
) -> None:
    if model_name == "catboost":
        cats = categorical_columns(x_train)
        model.fit(
            prepare_catboost_frame(x_train, cats),
            y_train,
            eval_set=(prepare_catboost_frame(x_valid, cats), y_valid),
            cat_features=cats,
            verbose=verbose,
            early_stopping_rounds=early_stopping_rounds,
        )
    else:
        model.fit(x_train, y_train)


def predict_probability(model, model_name: str, frame: pd.DataFrame) -> np.ndarray:
    if model_name == "catboost":
        frame = prepare_catboost_frame(frame, categorical_columns(frame))
    return np.asarray(model.predict_proba(frame)[:, 1], dtype=float)


def feature_importance(model, model_name: str, original_columns: list[str]) -> pd.DataFrame:
    """Return best-effort importance; models without native importance use absolute coefficients."""
    if model_name == "catboost":
        values = np.asarray(model.feature_importances_, dtype=float)
        names = original_columns
    else:
        preprocess = model.named_steps["preprocess"]
        estimator = model.named_steps["model"]
        names = preprocess.get_feature_names_out().tolist()
        if hasattr(estimator, "feature_importances_"):
            values = np.asarray(estimator.feature_importances_, dtype=float)
        elif hasattr(estimator, "coef_"):
            values = np.abs(np.asarray(estimator.coef_)[0])
        else:
            return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame({"feature": names, "importance": values}).sort_values(
        "importance", ascending=False
    )
