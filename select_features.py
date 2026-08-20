"""Greedy feature-block selection using only fixed group-CV OOF accuracy."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from kaggle_workflow.cv import load_folds, make_fold_assignments, save_folds
from kaggle_workflow.features import FEATURE_BLOCKS, create_features
from kaggle_workflow.metrics import binary_metrics
from kaggle_workflow.modeling import build_model, predict_probability

DEFAULT_BASE = ["cabin", "group", "age", "spend_total", "missing"]


def evaluate_blocks(
    train: pd.DataFrame,
    test: pd.DataFrame,
    folds: np.ndarray,
    blocks: list[str],
    seed: int,
    n_estimators: int,
) -> float:
    target = train["Transported"].astype(int).to_numpy()
    reference = pd.concat([train.drop(columns="Transported"), test], ignore_index=True)
    features = create_features(
        train.drop(columns="Transported"), reference=reference, blocks=blocks
    )
    oof = np.zeros(len(train), dtype=float)
    for fold in sorted(np.unique(folds)):
        fit_index = np.flatnonzero(folds != fold)
        valid_index = np.flatnonzero(folds == fold)
        model = build_model(
            "extratrees",
            {
                "n_estimators": n_estimators,
                "max_features": 0.8,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
            },
            seed,
            features,
        )
        model.fit(features.iloc[fit_index], target[fit_index])
        oof[valid_index] = predict_probability(model, "extratrees", features.iloc[valid_index])
    return binary_metrics(target, oof, 0.5)["accuracy"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("input/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("input/test.csv"))
    parser.add_argument(
        "--fold-path", type=Path, default=Path("artifacts/folds_group_5_seed42.csv")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/feature_selection.csv"))
    parser.add_argument(
        "--selected-output", type=Path, default=Path("artifacts/selected_blocks.yaml")
    )
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--min-improvement", type=float, default=0.0001)
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    if args.fold_path.exists():
        folds = load_folds(train, args.fold_path)
    else:
        folds = make_fold_assignments(train, strategy="stratified_group", seed=args.seed)
        save_folds(train, folds, args.fold_path)

    selected = list(DEFAULT_BASE)
    remaining = [block for block in FEATURE_BLOCKS if block not in selected]
    best_score = evaluate_blocks(train, test, folds, selected, args.seed, args.n_estimators)
    trials: list[dict[str, object]] = [
        {
            "round": 0,
            "candidate": "BASE",
            "blocks": ",".join(selected),
            "cv_accuracy": best_score,
            "accepted": True,
        }
    ]
    print(f"base {selected}: {best_score:.6f}")

    for round_number in range(1, args.max_rounds + 1):
        round_trials = []
        for candidate in remaining:
            blocks = [*selected, candidate]
            score = evaluate_blocks(train, test, folds, blocks, args.seed, args.n_estimators)
            row = {
                "round": round_number,
                "candidate": candidate,
                "blocks": ",".join(blocks),
                "cv_accuracy": score,
                "accepted": False,
            }
            trials.append(row)
            round_trials.append(row)
            print(f"round {round_number} +{candidate}: {score:.6f}")
        winner = max(round_trials, key=lambda row: float(row["cv_accuracy"]))
        if float(winner["cv_accuracy"]) < best_score + args.min_improvement:
            break
        winner["accepted"] = True
        selected.append(str(winner["candidate"]))
        remaining.remove(str(winner["candidate"]))
        best_score = float(winner["cv_accuracy"])
        if not remaining:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trials).to_csv(args.output, index=False)
    payload = {
        "selected_blocks": selected,
        "selection_model": "ExtraTreesClassifier",
        "cv_strategy": "StratifiedGroupKFold",
        "cv_accuracy": float(best_score),
        "n_estimators": args.n_estimators,
        "seed": args.seed,
    }
    with args.selected_output.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)
    print(f"selected: {selected} ({best_score:.6f})")
    print(f"saved: {args.output}, {args.selected_output}")


if __name__ == "__main__":
    main()
