from __future__ import annotations

import numpy as np
import pandas as pd

from ensemble import hill_climb
from kaggle_workflow.cv import group_overlap_by_fold, make_fold_assignments
from kaggle_workflow.features import create_features
from kaggle_workflow.metrics import best_accuracy_threshold


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
