"""Cross-fitted OOF hill-climbing ensemble utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def load_aligned_predictions(
    artifact_dir: str | Path,
    prefixes: dict[str, str],
    id_column: str,
    target: str,
    fold_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Load candidates and fail fast if their OOF/test rows differ."""
    artifact_dir = Path(artifact_dir)
    oof_frames = {
        name: pd.read_csv(artifact_dir / f"{prefix}_oof.csv")
        for name, prefix in prefixes.items()
    }
    test_frames = {
        name: pd.read_csv(artifact_dir / f"{prefix}_test.csv")
        for name, prefix in prefixes.items()
    }

    names = list(prefixes)
    reference_oof = oof_frames[names[0]]
    reference_test = test_frames[names[0]]
    for name in names[1:]:
        for column in [id_column, target, fold_column]:
            if not np.array_equal(
                reference_oof[column].to_numpy(),
                oof_frames[name][column].to_numpy(),
            ):
                raise ValueError(f"OOF {column} mismatch: {name}")
        if not np.array_equal(
            reference_test[id_column].to_numpy(),
            test_frames[name][id_column].to_numpy(),
        ):
            raise ValueError(f"test ID mismatch: {name}")

    oof_matrix = np.column_stack(
        [oof_frames[name]["oof_prob"].to_numpy() for name in names]
    )
    test_matrix = np.column_stack(
        [test_frames[name]["pred_prob"].to_numpy() for name in names]
    )
    if not np.isfinite(oof_matrix).all() or not np.isfinite(test_matrix).all():
        raise ValueError("prediction matrix contains non-finite values")
    return reference_oof, reference_test, oof_matrix, test_matrix


def hill_climb_weights(
    predictions: np.ndarray,
    target: np.ndarray,
    step_sizes: tuple[float, ...] = (0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50),
    max_rounds: int = 100,
    minimum_improvement: float = 1e-7,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Greedily move the blend toward the candidate with the best AUC gain."""
    n_models = predictions.shape[1]
    single_scores = [
        roc_auc_score(target, predictions[:, index])
        for index in range(n_models)
    ]
    best_single = int(np.argmax(single_scores))
    weights = np.zeros(n_models, dtype=float)
    weights[best_single] = 1.0
    current_score = float(single_scores[best_single])
    history = [
        {
            "round": 0,
            "selected_model_index": best_single,
            "step_size": 1.0,
            "auc": current_score,
        }
    ]

    for round_number in range(1, max_rounds + 1):
        best_score = current_score
        best_weights = None
        best_model = None
        best_step = None

        for model_index in range(n_models):
            for step_size in step_sizes:
                candidate_weights = weights * (1.0 - step_size)
                candidate_weights[model_index] += step_size
                score = roc_auc_score(target, predictions @ candidate_weights)
                if score > best_score:
                    best_score = float(score)
                    best_weights = candidate_weights
                    best_model = model_index
                    best_step = step_size

        if best_weights is None or best_score <= current_score + minimum_improvement:
            break
        weights = best_weights
        current_score = best_score
        history.append(
            {
                "round": round_number,
                "selected_model_index": best_model,
                "step_size": best_step,
                "auc": current_score,
            }
        )

    return weights, pd.DataFrame(history)


def cross_fitted_hill_climb(
    oof_matrix: np.ndarray,
    test_matrix: np.ndarray,
    target: np.ndarray,
    fold_ids: np.ndarray,
    **hill_climb_kwargs,
) -> dict[str, object]:
    """Fit blend weights outside each fold and predict that held-out fold."""
    cross_fitted_oof = np.full(len(target), np.nan, dtype=float)
    test_predictions = []
    fold_weights = []
    histories = []

    for fold in sorted(np.unique(fold_ids)):
        fit_mask = fold_ids != fold
        valid_mask = fold_ids == fold
        weights, history = hill_climb_weights(
            oof_matrix[fit_mask], target[fit_mask], **hill_climb_kwargs
        )
        cross_fitted_oof[valid_mask] = oof_matrix[valid_mask] @ weights
        test_predictions.append(test_matrix @ weights)
        fold_weights.append(weights)
        history = history.copy()
        history.insert(0, "held_out_fold", fold)
        histories.append(history)

    if np.isnan(cross_fitted_oof).any():
        raise RuntimeError("cross-fitted ensemble OOF is incomplete")
    return {
        "oof": cross_fitted_oof,
        "test_pred": np.mean(test_predictions, axis=0),
        "fold_weights": np.vstack(fold_weights),
        "history": pd.concat(histories, ignore_index=True),
        "oof_auc": roc_auc_score(target, cross_fitted_oof),
    }
