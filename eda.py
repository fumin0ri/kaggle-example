"""Generate simple or detailed EDA, including CV leakage and adversarial validation."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from kaggle_workflow.cv import group_overlap_by_fold, make_fold_assignments, passenger_groups
from kaggle_workflow.features import create_features
from kaggle_workflow.modeling import build_model, feature_importance, predict_probability


def _markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
    display = frame.head(max_rows).copy().fillna("")
    headers = [str(column) for column in display.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in display.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def simple_eda(train: pd.DataFrame, test: pd.DataFrame, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    target_counts = (
        train["Transported"]
        .value_counts(dropna=False)
        .rename_axis("Transported")
        .reset_index(name="count")
    )
    target_counts["rate"] = target_counts["count"] / len(train)
    missing = pd.DataFrame(
        {
            "column": train.columns,
            "dtype": train.dtypes.astype(str).values,
            "train_missing": train.isna().sum().values,
            "train_missing_rate": train.isna().mean().values,
            "test_missing": [
                test[column].isna().sum() if column in test else np.nan for column in train.columns
            ],
        }
    )
    missing.to_csv(report_dir / "simple_missingness.csv", index=False)
    duplicate_ids = (
        f"{train['PassengerId'].duplicated().sum()} / {test['PassengerId'].duplicated().sum()}"
    )
    text = f"""# 01 Simple EDA

## Shape

- train: {train.shape[0]:,} rows x {train.shape[1]} columns
- test: {test.shape[0]:,} rows x {test.shape[1]} columns
- duplicated PassengerId in train/test: {duplicate_ids}

## Target distribution

{_markdown_table(target_counts)}

## Missing values and dtypes

{_markdown_table(missing)}

## First baseline decision

The target is nearly balanced, so the first reproducible baseline uses accuracy, a fixed 0.5
threshold, ordinary shuffled StratifiedKFold, one-hot preprocessing, and XGBoost. This is a
deliberately simple checkpoint; group leakage is audited before trusting later CV scores.
"""
    (report_dir / "01_simple_eda.md").write_text(text, encoding="utf-8")


def _target_rate_table(train: pd.DataFrame, column: str) -> pd.DataFrame:
    return (
        train.assign(**{column: train[column].astype("string").fillna("__MISSING__")})
        .groupby(column, dropna=False)["Transported"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("count", ascending=False)
    )


def adversarial_validation(
    train: pd.DataFrame, test: pd.DataFrame, report_dir: Path, seed: int
) -> tuple[float, pd.DataFrame]:
    train_x = train.drop(columns=["Transported"])
    reference = pd.concat([train_x, test], ignore_index=True)
    combined = create_features(reference, reference=reference, feature_set="full")
    domain = np.r_[np.zeros(len(train), dtype=int), np.ones(len(test), dtype=int)]
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros(len(combined), dtype=float)
    importances: list[pd.DataFrame] = []
    for fold, (fit_index, valid_index) in enumerate(splitter.split(combined, domain)):
        model = build_model(
            "logistic",
            {"C": 0.08, "solver": "liblinear", "max_iter": 2000},
            seed + fold,
            combined,
        )
        model.fit(combined.iloc[fit_index], domain[fit_index])
        oof[valid_index] = predict_probability(model, "logistic", combined.iloc[valid_index])
        importance = feature_importance(model, "logistic", combined.columns.tolist())
        importance["fold"] = fold
        importances.append(importance)
    auc = float(roc_auc_score(domain, oof))
    importance = (
        pd.concat(importances, ignore_index=True)
        .groupby("feature", as_index=False)["importance"]
        .mean()
        .sort_values("importance", ascending=False)
    )
    pd.DataFrame({"domain_is_test": domain, "oof_prediction": oof}).to_csv(
        report_dir / "adversarial_oof.csv", index=False
    )
    importance.to_csv(report_dir / "adversarial_importance.csv", index=False)
    return auc, importance


def detailed_eda(train: pd.DataFrame, test: pd.DataFrame, report_dir: Path, seed: int) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    target_tables = []
    for column in ["HomePlanet", "CryoSleep", "Destination", "VIP"]:
        table = _target_rate_table(train, column)
        table.insert(0, "feature", column)
        table = table.rename(columns={column: "value", "mean": "target_rate"})
        target_tables.append(table)
    target_rates = pd.concat(target_tables, ignore_index=True)
    target_rates.to_csv(report_dir / "target_rates.csv", index=False)

    train_groups = passenger_groups(train["PassengerId"])
    group_stats = (
        train.assign(GroupId=train_groups)
        .groupby("GroupId")["Transported"]
        .agg(group_size="size", target_rate="mean", target_nunique="nunique")
        .reset_index()
    )
    group_summary = (
        group_stats.groupby("group_size")["target_rate"].agg(["mean", "std", "count"]).reset_index()
    )
    group_summary.to_csv(report_dir / "group_target_summary.csv", index=False)

    stratified_folds = make_fold_assignments(train, strategy="stratified", seed=seed)
    group_folds = make_fold_assignments(train, strategy="stratified_group", seed=seed)
    stratified_overlap = group_overlap_by_fold(train, stratified_folds)
    group_overlap = group_overlap_by_fold(train, group_folds)
    fold_balance = pd.DataFrame(
        {
            "fold": range(5),
            "stratified_target_rate": [
                train.loc[stratified_folds == fold, "Transported"].mean() for fold in range(5)
            ],
            "group_target_rate": [
                train.loc[group_folds == fold, "Transported"].mean() for fold in range(5)
            ],
            "stratified_group_overlap": stratified_overlap,
            "group_group_overlap": group_overlap,
        }
    )
    fold_balance.to_csv(report_dir / "cv_audit.csv", index=False)

    train_missing = train.drop(columns="Transported").isna().mean()
    test_missing = test.isna().mean()
    drift = pd.DataFrame(
        {
            "column": train_missing.index,
            "train_missing_rate": train_missing.values,
            "test_missing_rate": test_missing.reindex(train_missing.index).values,
        }
    )
    drift["absolute_gap"] = (drift["train_missing_rate"] - drift["test_missing_rate"]).abs()
    drift.sort_values("absolute_gap", ascending=False).to_csv(
        report_dir / "missingness_drift.csv", index=False
    )

    adversarial_auc, adversarial_importance = adversarial_validation(train, test, report_dir, seed)

    plt.figure(figsize=(9, 4.5))
    plot_data = drift.sort_values("absolute_gap", ascending=False)
    positions = np.arange(len(plot_data))
    plt.bar(positions - 0.2, plot_data["train_missing_rate"], width=0.4, label="train")
    plt.bar(positions + 0.2, plot_data["test_missing_rate"], width=0.4, label="test")
    plt.xticks(positions, plot_data["column"], rotation=55, ha="right")
    plt.ylabel("Missing rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_dir / "missingness_train_vs_test.png", dpi=140)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    top = adversarial_importance.head(15).sort_values("importance")
    plt.barh(top["feature"], top["importance"])
    plt.xlabel("Mean absolute standardized coefficient")
    plt.title(f"Adversarial validation (OOF AUC={adversarial_auc:.3f})")
    plt.tight_layout()
    plt.savefig(report_dir / "adversarial_importance.png", dpi=140)
    plt.close()

    homogeneous_groups = int((group_stats["target_nunique"] == 1).sum())
    group_finding = (
        f"Passenger groups: {group_stats.shape[0]:,}; groups whose observed members all share "
        f"one target: {homogeneous_groups:,}."
    )
    text = f"""# 02 Detailed EDA and CV audit

## Main findings

- {group_finding}
- Ordinary StratifiedKFold group overlap per fold: {stratified_overlap}.
- StratifiedGroupKFold group overlap per fold: {group_overlap}.
- Adversarial-validation OOF ROC AUC: **{adversarial_auc:.4f}**.
  (0.5 means no detectable train/test drift.)

The ordinary split is useful only as the requested first checkpoint. Since members of the same
PassengerId group often share outcomes, its train/validation overlap makes it optimistic. Every
later experiment therefore loads the exact same persisted StratifiedGroupKFold assignment.

Adversarial validation is diagnostic, not a Public-LB tuning signal. A high AUC means that some
features expose the chronological/train-test construction split; inspect
`adversarial_importance.csv` and require improvements to hold under group CV.

## Fold audit

{_markdown_table(fold_balance)}

## Target rates for important categorical variables

{_markdown_table(target_rates)}

## Group-size behavior

{_markdown_table(group_summary)}
"""
    (report_dir / "02_detailed_eda.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", choices=["simple", "detailed", "all"], default="all")
    parser.add_argument("--train", type=Path, default=Path("input/train.csv"))
    parser.add_argument("--test", type=Path, default=Path("input/test.csv"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train = pd.read_csv(args.train)
    test = pd.read_csv(args.test)
    if args.level in {"simple", "all"}:
        simple_eda(train, test, args.report_dir)
    if args.level in {"detailed", "all"}:
        detailed_eda(train, test, args.report_dir, args.seed)
    print(f"saved EDA reports: {args.report_dir}")


if __name__ == "__main__":
    main()
