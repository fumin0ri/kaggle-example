"""Run one reproducible Spaceship Titanic experiment from one YAML config."""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from kaggle_workflow.cv import (
    group_overlap_by_fold,
    load_folds,
    make_fold_assignments,
    save_folds,
)
from kaggle_workflow.features import create_features, resolve_blocks
from kaggle_workflow.metrics import best_accuracy_threshold, binary_metrics
from kaggle_workflow.modeling import (
    build_model,
    feature_importance,
    fit_model,
    predict_probability,
)


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    required = {"model", "target", "seed", "cv", "params"}
    if not isinstance(config, dict) or required - config.keys():
        raise ValueError(f"Config must contain {sorted(required)}: {path}")
    return config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def get_folds(train: pd.DataFrame, config: dict[str, Any]) -> np.ndarray:
    cv = config["cv"]
    path = Path(cv["fold_path"]) if cv.get("fold_path") else None
    if path and path.exists():
        folds = load_folds(train, path)
    else:
        folds = make_fold_assignments(
            train,
            target=config["target"],
            strategy=cv["strategy"],
            n_splits=int(cv["n_splits"]),
            seed=int(config["seed"]),
        )
        if path:
            save_folds(train, folds, path)
    return folds


def update_results(path: Path, row: dict[str, str]) -> None:
    fields = ["exp", "model", "feature_set", "cv_strategy", "cv_accuracy", "cv_auc", "memo"]
    rows: list[dict[str, str]] = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    rows = [old for old in rows if old.get("exp") != row["exp"]]
    rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    exp_name = config_path.stem
    output_dir = Path("output") / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    seed = int(config["seed"])
    seed_everything(seed)
    train = pd.read_csv(config.get("train_path", "input/train.csv"))
    test = pd.read_csv(config.get("test_path", "input/test.csv"))
    target = config["target"]
    y = train[target].astype(int).to_numpy()

    raw_reference = pd.concat([train.drop(columns=[target]), test], ignore_index=True, sort=False)
    blocks = config.get("feature_blocks")
    if config.get("feature_blocks_file"):
        with Path(config["feature_blocks_file"]).open(encoding="utf-8") as file:
            selection = yaml.safe_load(file)
        blocks = selection["selected_blocks"]
    resolved_blocks = resolve_blocks(config.get("feature_set", "full"), blocks)
    x = create_features(
        train.drop(columns=[target]),
        reference=raw_reference,
        feature_set=config.get("feature_set", "full"),
        blocks=blocks,
    )
    x_test = create_features(
        test,
        reference=raw_reference,
        feature_set=config.get("feature_set", "full"),
        blocks=blocks,
    )
    if x.columns.tolist() != x_test.columns.tolist():
        raise ValueError("Train and test feature columns differ")

    folds = get_folds(train, config)
    overlap = group_overlap_by_fold(train, folds)
    model_name = config["model"]
    unique_folds = sorted(np.unique(folds).tolist())
    print(f"experiment: {exp_name}")
    print(f"model/features: {model_name} / {config.get('feature_set', 'full')}")
    print(f"train/test/features: {train.shape} / {test.shape} / {x.shape[1]}")
    print(f"cv: {config['cv']['strategy']}; group overlap per fold: {overlap}")

    oof = np.zeros(len(train), dtype=float)
    test_predictions = np.zeros(len(test), dtype=float)
    fold_metrics: list[dict[str, float]] = []
    importances: list[pd.DataFrame] = []

    for fold in unique_folds:
        train_index = np.flatnonzero(folds != fold)
        valid_index = np.flatnonzero(folds == fold)
        model = build_model(model_name, config["params"], seed + fold, x)
        fit_model(
            model,
            model_name,
            x.iloc[train_index],
            y[train_index],
            x.iloc[valid_index],
            y[valid_index],
            verbose=config.get("verbose", 100),
            early_stopping_rounds=config.get("early_stopping_rounds", 100),
        )
        valid_prediction = predict_probability(model, model_name, x.iloc[valid_index])
        oof[valid_index] = valid_prediction
        test_predictions += predict_probability(model, model_name, x_test) / len(unique_folds)
        metrics = binary_metrics(y[valid_index], valid_prediction, 0.5)
        metrics["fold"] = int(fold)
        fold_metrics.append(metrics)
        importance = feature_importance(model, model_name, x.columns.tolist())
        if not importance.empty:
            importance["fold"] = fold
            importances.append(importance)
        print(f"fold {fold}: accuracy={metrics['accuracy']:.6f}, auc={metrics['roc_auc']:.6f}")

    configured_threshold = config.get("threshold", 0.5)
    if configured_threshold == "oof":
        threshold, _ = best_accuracy_threshold(y, oof)
    else:
        threshold = float(configured_threshold)
    overall = binary_metrics(y, oof, threshold)
    metrics_payload: dict[str, Any] = {
        **overall,
        "experiment": exp_name,
        "model": model_name,
        "feature_set": config.get("feature_set", "full"),
        "feature_blocks": list(resolved_blocks),
        "cv_strategy": config["cv"]["strategy"],
        "n_splits": len(unique_folds),
        "seed": seed,
        "group_overlap_by_fold": overlap,
        "fold_metrics": fold_metrics,
    }

    pd.DataFrame(
        {
            "PassengerId": train["PassengerId"],
            "target": y,
            "prediction": oof,
            "fold": folds,
        }
    ).to_csv(output_dir / "oof.csv", index=False)
    pd.DataFrame({"PassengerId": test["PassengerId"], "prediction": test_predictions}).to_csv(
        output_dir / "test_predictions.csv", index=False
    )
    pd.DataFrame(
        {
            "PassengerId": test["PassengerId"],
            "Transported": (test_predictions >= threshold).astype(bool),
        }
    ).to_csv(output_dir / "submission.csv", index=False)
    if importances:
        (
            pd.concat(importances, ignore_index=True)
            .groupby("feature", as_index=False)["importance"]
            .agg(["mean", "std"])
            .sort_values("mean", ascending=False)
            .reset_index()
            .to_csv(output_dir / "feature_importance.csv", index=False)
        )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics_payload, file, ensure_ascii=False, indent=2)

    update_results(
        Path("results.csv"),
        {
            "exp": exp_name,
            "model": model_name,
            "feature_set": config.get("feature_set", "full"),
            "cv_strategy": config["cv"]["strategy"],
            "cv_accuracy": f"{overall['accuracy']:.6f}",
            "cv_auc": f"{overall['roc_auc']:.6f}",
            "memo": config.get("memo", ""),
        },
    )
    print(
        f"OOF: accuracy={overall['accuracy']:.6f}, auc={overall['roc_auc']:.6f}, "
        f"threshold={threshold:.4f}"
    )
    print(f"saved: {output_dir}")
    return metrics_payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
