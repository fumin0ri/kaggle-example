"""Optuna search spaces for the three post-selection models."""

from __future__ import annotations

import json
from pathlib import Path


HPO_MODELS = ["XGBoost", "LogisticRegression", "MLP"]


def model_slug(model_name: str) -> str:
    return model_name.lower().replace(" ", "_")


def suggest_model_params(trial, model_name: str) -> dict[str, object]:
    """Return model parameters while keeping preprocessing fixed."""
    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 300, 1200, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.01, 0.10, log=True
            ),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 12),
            "subsample": trial.suggest_float("subsample", 0.70, 1.00),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.60, 1.00
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        }

    if model_name == "LogisticRegression":
        class_weight = trial.suggest_categorical(
            "class_weight", ["none", "balanced"]
        )
        return {
            "C": trial.suggest_float("C", 1e-3, 30.0, log=True),
            "class_weight": None if class_weight == "none" else class_weight,
        }

    if model_name == "MLP":
        architecture_name = trial.suggest_categorical(
            "architecture", ["32", "32_16", "64_32", "128_64", "64_32_16"]
        )
        architectures = {
            "32": (32,),
            "32_16": (32, 16),
            "64_32": (64, 32),
            "128_64": (128, 64),
            "64_32_16": (64, 32, 16),
        }
        return {
            "hidden_layer_sizes": architectures[architecture_name],
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "batch_size": trial.suggest_categorical(
                "batch_size", [64, 128, 256, 512]
            ),
            "learning_rate_init": trial.suggest_float(
                "learning_rate_init", 1e-4, 5e-3, log=True
            ),
        }

    raise ValueError(f"unsupported HPO model: {model_name}")


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
