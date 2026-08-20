"""Hill-climb a convex ensemble using OOF predictions only—never Public LB scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from kaggle_workflow.metrics import best_accuracy_threshold, binary_metrics


def load_predictions(experiment_dirs: list[Path]):
    names: list[str] = []
    oof_predictions: list[np.ndarray] = []
    test_predictions: list[np.ndarray] = []
    reference_oof: pd.DataFrame | None = None
    reference_test: pd.DataFrame | None = None
    for directory in experiment_dirs:
        oof = pd.read_csv(directory / "oof.csv")
        test = pd.read_csv(directory / "test_predictions.csv")
        if reference_oof is None:
            reference_oof = oof[["PassengerId", "target", "fold"]].copy()
            reference_test = test[["PassengerId"]].copy()
        elif not oof[["PassengerId", "target", "fold"]].equals(reference_oof):
            raise ValueError(f"OOF identity/target/fold mismatch: {directory}")
        elif not test[["PassengerId"]].equals(reference_test):
            raise ValueError(f"Test PassengerId mismatch: {directory}")
        names.append(directory.name)
        oof_predictions.append(oof["prediction"].to_numpy(dtype=float))
        test_predictions.append(test["prediction"].to_numpy(dtype=float))
    if reference_oof is None or reference_test is None:
        raise ValueError("At least one experiment directory is required")
    return (
        names,
        np.vstack(oof_predictions),
        np.vstack(test_predictions),
        reference_oof,
        reference_test,
    )


def hill_climb(
    y: np.ndarray,
    prediction_matrix: np.ndarray,
    *,
    max_steps: int,
    min_improvement: float,
    selection_metric: str = "accuracy_then_log_loss",
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    if selection_metric not in {"log_loss", "accuracy_then_log_loss"}:
        raise ValueError(f"Unknown selection metric: {selection_metric}")

    def score(predictions: np.ndarray) -> tuple[float, float, float]:
        threshold, accuracy = best_accuracy_threshold(y, predictions)
        logloss = binary_metrics(y, predictions, threshold)["log_loss"]
        if selection_metric == "log_loss":
            return -logloss, accuracy, threshold
        return accuracy, -logloss, threshold

    single_scores = [score(row) for row in prediction_matrix]
    first = max(range(len(single_scores)), key=lambda index: single_scores[index][:2])
    weights = np.zeros(len(prediction_matrix), dtype=float)
    weights[first] = 1.0
    ensemble = prediction_matrix[first].copy()
    current_primary, current_secondary, current_threshold = single_scores[first]
    history: list[dict[str, float | int]] = [
        {
            "step": 1,
            "selected_model_index": first,
            "selection_primary": current_primary,
            "selection_secondary": current_secondary,
            "oof_threshold": current_threshold,
        }
    ]
    alpha_grid = np.array([0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
    for step in range(2, max_steps + 1):
        trials = []
        for model_index, model_prediction in enumerate(prediction_matrix):
            for alpha in alpha_grid:
                candidate = (1 - alpha) * ensemble + alpha * model_prediction
                trials.append((score(candidate), model_index, float(alpha), candidate))
        winner_score, winner, winner_alpha, winner_prediction = max(
            trials, key=lambda trial: trial[0][:2]
        )
        winner_primary, winner_secondary, winner_threshold = winner_score
        primary_improved = winner_primary >= current_primary + min_improvement
        primary_tied = abs(winner_primary - current_primary) < min_improvement
        secondary_improved = winner_secondary >= current_secondary + min_improvement
        if not (primary_improved or (primary_tied and secondary_improved)):
            break
        ensemble = winner_prediction
        weights *= 1 - winner_alpha
        weights[winner] += winner_alpha
        current_primary = winner_primary
        current_secondary = winner_secondary
        current_threshold = winner_threshold
        history.append(
            {
                "step": step,
                "selected_model_index": winner,
                "alpha": winner_alpha,
                "selection_primary": current_primary,
                "selection_secondary": current_secondary,
                "oof_threshold": current_threshold,
            }
        )
    return weights, ensemble, history


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output/hill_climb_ensemble"))
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--min-improvement", type=float, default=0.000001)
    parser.add_argument(
        "--selection-metric",
        choices=["accuracy_then_log_loss", "log_loss"],
        default="accuracy_then_log_loss",
    )
    args = parser.parse_args()

    names, oof_matrix, test_matrix, oof_identity, test_identity = load_predictions(
        args.experiment_dirs
    )
    y = oof_identity["target"].to_numpy(dtype=int)
    weights, ensemble_oof, history = hill_climb(
        y,
        oof_matrix,
        max_steps=args.max_steps,
        min_improvement=args.min_improvement,
        selection_metric=args.selection_metric,
    )
    ensemble_test = np.average(test_matrix, axis=0, weights=weights)
    threshold, _ = best_accuracy_threshold(y, ensemble_oof)
    metrics = binary_metrics(y, ensemble_oof, threshold)
    weight_rows = [
        {
            "experiment": name,
            "selections": sum(int(row["selected_model_index"] == index) for row in history),
            "weight": weights[index],
        }
        for index, name in enumerate(names)
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(weight_rows).to_csv(args.output_dir / "weights.csv", index=False)
    pd.DataFrame(history).to_csv(args.output_dir / "history.csv", index=False)
    pd.DataFrame(oof_matrix.T, columns=names).corr().to_csv(
        args.output_dir / "oof_correlations.csv"
    )
    oof_identity.assign(prediction=ensemble_oof).to_csv(args.output_dir / "oof.csv", index=False)
    pd.DataFrame(
        {
            "PassengerId": test_identity["PassengerId"],
            "prediction": ensemble_test,
        }
    ).to_csv(args.output_dir / "test_predictions.csv", index=False)
    pd.DataFrame(
        {
            "PassengerId": test_identity["PassengerId"],
            "Transported": (ensemble_test >= threshold).astype(bool),
        }
    ).to_csv(args.output_dir / "submission.csv", index=False)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                **metrics,
                "weights": weight_rows,
                "selection_metric": f"OOF {args.selection_metric}",
            },
            file,
            ensure_ascii=False,
            indent=2,
        )
    results_path = Path("results.csv")
    if results_path.exists():
        results = pd.read_csv(results_path)
        results = results[results["exp"] != "hill_climb_ensemble"]
        ensemble_row = pd.DataFrame(
            [
                {
                    "exp": "hill_climb_ensemble",
                    "model": "hill_climb",
                    "feature_set": "oof_selected",
                    "cv_strategy": "stratified_group",
                    "cv_accuracy": metrics["accuracy"],
                    "cv_auc": metrics["roc_auc"],
                    "memo": "Weights and threshold selected only from shared-fold OOF",
                }
            ]
        )
        pd.concat([results, ensemble_row], ignore_index=True).to_csv(results_path, index=False)
    print(pd.DataFrame(weight_rows).to_string(index=False))
    print(f"OOF accuracy={metrics['accuracy']:.6f}; threshold={threshold:.4f}")
    print(f"saved: {args.output_dir}")


if __name__ == "__main__":
    main()
