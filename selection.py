"""Utilities for screening and pruning leakage-safe TE features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def load_oof_te_matrix(
    train: pd.DataFrame,
    feature_dir: str | Path,
    id_column: str = "id",
    fold_column: str = "fold",
) -> pd.DataFrame:
    """Assemble outer-validation TE files into one ID-aligned OOF matrix."""
    feature_dir = Path(feature_dir)
    validation_parts = []
    expected_features: list[str] | None = None

    for fold in sorted(train[fold_column].unique()):
        fold_tag = str(fold).replace("/", "_")
        part = pd.read_parquet(feature_dir / f"outer_fold_{fold_tag}_valid.parquet")
        features = [column for column in part.columns if column != id_column]
        if expected_features is None:
            expected_features = features
        elif features != expected_features:
            raise ValueError(f"fold={fold}: TE feature order differs")
        if part[id_column].duplicated().any():
            raise ValueError(f"fold={fold}: duplicate IDs in validation TE")
        validation_parts.append(part)

    encoded = pd.concat(validation_parts, ignore_index=True)
    if encoded[id_column].duplicated().any():
        raise ValueError("an ID appears in more than one validation fold")

    aligned = train[[id_column]].merge(
        encoded,
        on=id_column,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if aligned["_merge"].ne("both").any():
        raise ValueError("some train IDs are missing from OOF TE")
    result = aligned.drop(columns=[id_column, "_merge"])
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError("OOF TE contains non-finite values")
    return result


def rank_te_features(
    oof_features: pd.DataFrame,
    target: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Rank candidates by leakage-safe univariate OOF AUC."""
    y = np.asarray(target)
    rows = []
    for feature in oof_features.columns:
        values = oof_features[feature].to_numpy()
        auc = roc_auc_score(y, values)
        rows.append(
            {
                "feature": feature,
                "oof_auc": auc,
                "signal_auc": max(auc, 1.0 - auc),
                "std": float(np.std(values)),
                "nunique": int(pd.Series(values).nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["signal_auc", "feature"], ascending=[False, True], ignore_index=True
    )


def prune_correlated_features(
    oof_features: pd.DataFrame,
    ranking: pd.DataFrame,
    threshold: float = 0.995,
    sample_size: int = 100_000,
    random_state: int = 42,
) -> tuple[list[str], pd.DataFrame]:
    """Greedily keep strong features while dropping near-duplicates.

    Correlations are estimated on a deterministic sample for speed. Candidates
    are considered in univariate OOF-AUC order, so the stronger member of a
    highly correlated pair is retained first.
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")

    ranked_features = ranking["feature"].tolist()
    sample = oof_features[ranked_features]
    if len(sample) > sample_size:
        sample = sample.sample(sample_size, random_state=random_state)
    correlation = sample.corr().abs()

    selected: list[str] = []
    decisions = []
    for feature in ranked_features:
        if not selected:
            max_correlation = 0.0
            keep = True
        else:
            correlations = correlation.loc[feature, selected]
            max_correlation = float(correlations.max())
            keep = max_correlation < threshold
        decisions.append(
            {
                "feature": feature,
                "keep_after_correlation_pruning": keep,
                "max_abs_correlation_with_kept": max_correlation,
            }
        )
        if keep:
            selected.append(feature)

    audit = ranking.merge(pd.DataFrame(decisions), on="feature", validate="one_to_one")
    return selected, audit


def candidate_feature_counts(n_available: int) -> list[int]:
    """Return a compact top-k search grid including no-TE and all candidates."""
    proposed = [0, 5, 10, 20, 40, 80, n_available]
    return sorted({min(value, n_available) for value in proposed})


def select_conservative_candidate(
    fold_scores: pd.DataFrame,
    minimum_improvement: float = 5e-5,
) -> pd.Series:
    """Choose top-k using paired fold deltas and a one-standard-error penalty.

    ``fold_scores`` must contain ``n_te_features``, ``fold``, and ``auc`` for
    one model. The zero-feature row is the paired baseline.
    """
    baseline = (
        fold_scores.loc[fold_scores["n_te_features"].eq(0), ["fold", "auc"]]
        .rename(columns={"auc": "baseline_auc"})
    )
    if baseline.empty:
        raise ValueError("zero-feature baseline is required")

    rows = []
    for n_features, group in fold_scores.groupby("n_te_features"):
        paired = group[["fold", "auc"]].merge(
            baseline, on="fold", validate="one_to_one"
        )
        deltas = paired["auc"] - paired["baseline_auc"]
        standard_error = float(deltas.std(ddof=1) / np.sqrt(len(deltas)))
        if np.isnan(standard_error):
            standard_error = 0.0
        rows.append(
            {
                "n_te_features": int(n_features),
                "mean_auc": float(paired["auc"].mean()),
                "mean_delta": float(deltas.mean()),
                "delta_standard_error": standard_error,
                "conservative_delta": float(deltas.mean() - standard_error),
            }
        )

    summary = pd.DataFrame(rows)
    eligible = summary.loc[
        summary["conservative_delta"].ge(minimum_improvement)
    ]
    if eligible.empty:
        return summary.loc[summary["n_te_features"].eq(0)].iloc[0]
    return eligible.sort_values(
        ["conservative_delta", "n_te_features"], ascending=[False, True]
    ).iloc[0]
