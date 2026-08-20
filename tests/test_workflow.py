from __future__ import annotations

from pathlib import Path

import nbformat
import numpy as np
import pandas as pd

from ensemble import hill_climb
from hpo import materialize_parameters
from kaggle_workflow.cv import group_overlap_by_fold, make_fold_assignments
from kaggle_workflow.features import create_features
from kaggle_workflow.metrics import best_accuracy_threshold

ROOT = Path(__file__).resolve().parents[1]


def tiny_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": ["0001_01", "0001_02", "0002_01", "0003_01", "0003_02", "0004_01"],
            "HomePlanet": ["Earth", "Earth", "Mars", "Europa", "Europa", "Earth"],
            "CryoSleep": [False, False, True, True, True, False],
            "Cabin": ["F/1/P", "F/1/P", "E/2/S", "B/3/P", "B/3/P", None],
            "Destination": ["A", "A", "B", "A", "A", "B"],
            "Age": [20, 18, 40, 10, 8, np.nan],
            "VIP": [False, False, True, False, False, False],
            "RoomService": [1, 0, 0, 0, 0, np.nan],
            "FoodCourt": [2, 0, 0, 0, 0, 1],
            "ShoppingMall": [0, 0, 0, 0, 0, 0],
            "Spa": [0, 0, 0, 0, 0, 0],
            "VRDeck": [0, 0, 0, 0, 0, 0],
            "Name": ["A One", "B One", "C Two", "D Three", "E Three", None],
            "Transported": [False, False, True, True, True, False],
        }
    )


def test_features_are_target_free_and_keep_schema() -> None:
    train = tiny_frame()
    test = train.drop(columns="Transported").iloc[:2].copy()
    reference = pd.concat([train.drop(columns="Transported"), test], ignore_index=True)
    train_features = create_features(train, reference=reference, feature_set="full")
    test_features = create_features(test, reference=reference, feature_set="full")
    assert train_features.columns.tolist() == test_features.columns.tolist()
    assert not {"PassengerId", "Name", "Transported"} & set(train_features.columns)
    assert train_features.loc[0, "GroupSize"] == 4


def test_group_folds_have_no_overlap() -> None:
    base = tiny_frame()
    train = pd.concat(
        [
            base.assign(PassengerId=lambda x: x["PassengerId"].str.replace("000", f"{i:03}"))
            for i in range(1, 7)
        ],
        ignore_index=True,
    )
    folds = make_fold_assignments(train, n_splits=3, strategy="stratified_group", seed=42)
    assert group_overlap_by_fold(train, folds) == [0, 0, 0]


def test_threshold_and_hill_climb_are_deterministic() -> None:
    y = np.array([0, 0, 1, 1])
    predictions = np.array([[0.1, 0.4, 0.6, 0.9], [0.2, 0.7, 0.8, 0.6]])
    threshold, score = best_accuracy_threshold(y, predictions[0])
    weights, ensemble, history = hill_climb(y, predictions, max_steps=5, min_improvement=1e-6)
    assert score == 1.0
    assert abs(threshold - 0.5) < 1e-9
    assert np.isclose(weights.sum(), 1.0)
    assert len(history) >= 1
    assert ensemble.shape == y.shape


def test_hpo_parameters_are_materialized_for_each_estimator() -> None:
    mlp = materialize_parameters(
        "mlp",
        {
            "hidden_layers": "64,32",
            "alpha": 0.001,
            "learning_rate_init": 0.002,
            "batch_size": 64,
        },
    )
    xgboost = materialize_parameters(
        "xgboost",
        {
            "n_estimators": 500,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "reg_alpha": 0.1,
            "reg_lambda": 2.0,
        },
    )
    assert mlp["hidden_layer_sizes"] == [64, 32]
    assert "hidden_layers" not in mlp
    assert xgboost["tree_method"] == "hist"


def test_analysis_notebooks_are_valid_and_executed() -> None:
    notebook_paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    assert [path.name for path in notebook_paths] == [
        "01_simple_eda.ipynb",
        "02_detailed_eda_and_cv.ipynb",
        "03_hpo_and_ensemble_analysis.ipynb",
    ]
    for path in notebook_paths:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
        assert code_cells
        assert any(cell.get("outputs") for cell in code_cells)


def test_final_submission_matches_competition_schema() -> None:
    submission = pd.read_csv(ROOT / "submissions" / "submission_hill_climb.csv")
    assert submission.columns.tolist() == ["PassengerId", "Transported"]
    assert len(submission) == 4_277
    assert set(submission["Transported"].unique()) <= {True, False}
