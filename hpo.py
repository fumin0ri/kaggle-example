"""Reproducible Optuna HPO with a dedicated grouped tuning CV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import yaml

from kaggle_workflow.cv import load_folds, make_fold_assignments, save_folds
from kaggle_workflow.features import create_features
from kaggle_workflow.metrics import binary_metrics
from kaggle_workflow.modeling import build_model, fit_model, predict_probability

EXPERIMENT_NUMBERS = {
    "catboost": 301,
    "xgboost": 302,
    "extratrees": 303,
    "logistic": 304,
    "mlp": 305,
}

INITIAL_PARAMETERS: dict[str, dict[str, Any]] = {
    "catboost": {
        "iterations": 650,
        "learning_rate": 0.045,
        "depth": 7,
        "l2_leaf_reg": 5.0,
        "random_strength": 0.4,
        "bagging_temperature": 1.0,
    },
    "xgboost": {
        "n_estimators": 650,
        "learning_rate": 0.035,
        "max_depth": 5,
        "min_child_weight": 4,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.05,
        "reg_lambda": 2.5,
    },
    "extratrees": {
        "n_estimators": 550,
        "max_features": 0.8,
        "min_samples_leaf": 2,
        "max_depth": None,
    },
    "logistic": {"C": 0.12, "class_weight": None},
    "mlp": {
        "hidden_layers": "64,32",
        "activation": "relu",
        "alpha": 0.001,
        "batch_size": 128,
        "learning_rate_init": 0.001,
    },
}


def suggest_parameters(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    if model_name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 350, 750, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
            "depth": trial.suggest_int("depth", 5, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 12.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.05, 1.5, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 3.0),
        }
    if model_name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 400, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.015, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.65, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 1.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 8.0, log=True),
        }
    if model_name == "extratrees":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 350, 700, step=50),
            "max_features": trial.suggest_float("max_features", 0.4, 1.0),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
            "max_depth": trial.suggest_categorical("max_depth", [None, 12, 18, 24]),
        }
    if model_name == "logistic":
        return {
            "C": trial.suggest_float("C", 1e-3, 10.0, log=True),
            "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
        }
    if model_name == "mlp":
        return {
            "hidden_layers": trial.suggest_categorical(
                "hidden_layers", ["32", "64", "64,32", "128,64", "128,64,32"]
            ),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 2e-4, 5e-3, log=True),
        }
    raise ValueError(f"Unknown model: {model_name}")


def materialize_parameters(model_name: str, sampled: dict[str, Any]) -> dict[str, Any]:
    params = dict(sampled)
    if model_name == "catboost":
        params.update({"allow_writing_files": False})
    elif model_name == "xgboost":
        params.update({"tree_method": "hist"})
    elif model_name == "extratrees":
        params.update({"class_weight": "balanced"})
    elif model_name == "logistic":
        params.update({"solver": "liblinear", "max_iter": 2000})
    elif model_name == "mlp":
        layers = params.pop("hidden_layers")
        params["hidden_layer_sizes"] = [int(value) for value in layers.split(",")]
        params.update(
            {
                "early_stopping": True,
                "validation_fraction": 0.15,
                "n_iter_no_change": 20,
                "max_iter": 180,
            }
        )
    return params


def write_best_config(
    path: Path,
    *,
    model_name: str,
    seed: int,
    params: dict[str, Any],
    trial_number: int,
    tuning_score: float,
) -> None:
    payload = {
        "model": model_name,
        "target": "Transported",
        "seed": seed,
        "feature_set": "selected",
        "feature_blocks_file": "artifacts/selected_blocks.yaml",
        "threshold": 0.5,
        "memo": (
            f"HPO best trial {trial_number}; dedicated grouped tuning CV accuracy "
            f"{tuning_score:.6f}"
        ),
        "cv": {
            "strategy": "stratified_group",
            "n_splits": 5,
            "fold_path": "artifacts/folds_group_5_seed42.csv",
        },
        "verbose": 0,
        "early_stopping_rounds": 80,
        "params": params,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(EXPERIMENT_NUMBERS), required=True)
    parser.add_argument("--n-trials", type=int, default=30, help="Total completed-trial budget")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--train", type=Path, default=Path("input/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("input/test.csv"))
    parser.add_argument(
        "--fold-path", type=Path, default=Path("artifacts/folds_hpo_3_seed2025.csv")
    )
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/hpo"))
    parser.add_argument("--output-config", type=Path)
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    if args.fold_path.exists():
        folds = load_folds(train, args.fold_path)
    else:
        folds = make_fold_assignments(
            train,
            strategy="stratified_group",
            n_splits=args.n_splits,
            seed=args.seed,
        )
        save_folds(train, folds, args.fold_path)
    if len(np.unique(folds)) != args.n_splits:
        raise ValueError("Existing HPO fold file does not match --n-splits")

    with Path("artifacts/selected_blocks.yaml").open(encoding="utf-8") as file:
        blocks = yaml.safe_load(file)["selected_blocks"]
    reference = pd.concat([train.drop(columns="Transported"), test], ignore_index=True)
    features = create_features(
        train.drop(columns="Transported"), reference=reference, blocks=blocks
    )
    target = train["Transported"].astype(int).to_numpy()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    storage_path = (args.artifact_dir / f"{args.model}.db").resolve().as_posix()
    study = optuna.create_study(
        study_name=f"spaceship_{args.model}_seed{args.seed}",
        storage=f"sqlite:///{storage_path}",
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        load_if_exists=True,
    )
    if not study.trials:
        study.enqueue_trial(INITIAL_PARAMETERS[args.model])

    def objective(trial: optuna.Trial) -> float:
        sampled = suggest_parameters(trial, args.model)
        params = materialize_parameters(args.model, sampled)
        oof = np.zeros(len(train), dtype=float)
        evaluated = np.zeros(len(train), dtype=bool)
        fold_accuracies = []
        for fold in sorted(np.unique(folds)):
            fit_index = np.flatnonzero(folds != fold)
            valid_index = np.flatnonzero(folds == fold)
            model = build_model(args.model, params, args.seed + int(fold), features)
            fit_model(
                model,
                args.model,
                features.iloc[fit_index],
                target[fit_index],
                features.iloc[valid_index],
                target[valid_index],
                verbose=0,
                early_stopping_rounds=60,
            )
            predictions = predict_probability(model, args.model, features.iloc[valid_index])
            oof[valid_index] = predictions
            evaluated[valid_index] = True
            fold_accuracy = binary_metrics(target[valid_index], predictions, 0.5)["accuracy"]
            fold_accuracies.append(fold_accuracy)
            trial.report(float(np.mean(fold_accuracies)), step=int(fold))
            if trial.should_prune():
                raise optuna.TrialPruned()
        metrics = binary_metrics(target[evaluated], oof[evaluated], 0.5)
        trial.set_user_attr("oof_log_loss", metrics["log_loss"])
        trial.set_user_attr("oof_auc", metrics["roc_auc"])
        trial.set_user_attr("fold_accuracy_std", float(np.std(fold_accuracies)))
        return metrics["accuracy"]

    completed = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
    remaining = max(0, args.n_trials - completed)
    if remaining:
        study.optimize(objective, n_trials=remaining, n_jobs=1, gc_after_trial=True)

    trials_path = args.artifact_dir / f"{args.model}_trials.csv"
    study.trials_dataframe().to_csv(trials_path, index=False)
    best = study.best_trial
    best_params = materialize_parameters(args.model, best.params)
    best_payload = {
        "model": args.model,
        "best_trial": best.number,
        "tuning_cv_accuracy": best.value,
        "tuning_cv_strategy": f"StratifiedGroupKFold({args.n_splits})",
        "tuning_seed": args.seed,
        "params": best_params,
    }
    with (args.artifact_dir / f"{args.model}_best.json").open("w", encoding="utf-8") as file:
        json.dump(best_payload, file, indent=2)

    output_config = args.output_config or Path(
        f"configs/exp{EXPERIMENT_NUMBERS[args.model]}_{args.model}_hpo.yaml"
    )
    write_best_config(
        output_config,
        model_name=args.model,
        seed=42,
        params=best_params,
        trial_number=best.number,
        tuning_score=float(best.value),
    )
    print(f"model={args.model}; completed_trials={completed + remaining}")
    print(f"best_trial={best.number}; tuning_cv_accuracy={best.value:.6f}")
    print(f"saved: {trials_path}, {output_config}")


if __name__ == "__main__":
    main()
