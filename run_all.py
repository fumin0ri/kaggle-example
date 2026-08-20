"""Run the complete tutorial in the intended order."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

BASELINES = [
    "exp101_catboost_baseline",
    "exp102_xgboost_baseline",
    "exp103_extratrees_baseline",
    "exp104_logistic_baseline",
    "exp105_mlp_baseline",
]
SELECTED = [
    "exp201_catboost_selected",
    "exp202_xgboost_selected",
    "exp203_extratrees_selected",
    "exp204_logistic_selected",
    "exp205_mlp_selected",
]


def run(*args: str) -> None:
    print(f"\n$ {' '.join(args)}", flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    required = [Path("input/train.csv"), Path("input/test.csv")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing competition data: {missing}; see README.md")

    python = sys.executable
    run(python, "eda.py", "--level", "simple")
    run(python, "train.py", "--config", "configs/exp001_xgb_initial.yaml")
    run(python, "eda.py", "--level", "detailed")
    run(python, "make_folds.py")

    for experiment in BASELINES:
        run(python, "train.py", "--config", f"configs/{experiment}.yaml")

    run(python, "select_features.py")
    for experiment in SELECTED:
        run(python, "train.py", "--config", f"configs/{experiment}.yaml")

    experiment_dirs = [f"output/{experiment}" for experiment in [*BASELINES, *SELECTED]]
    run(python, "ensemble.py", *experiment_dirs)

    submission_dir = Path("submissions")
    submission_dir.mkdir(exist_ok=True)
    shutil.copy2(
        "output/exp001_xgb_initial/submission.csv",
        submission_dir / "submission_initial_xgb.csv",
    )
    shutil.copy2(
        "output/hill_climb_ensemble/submission.csv",
        submission_dir / "submission_hill_climb.csv",
    )
    print("\nComplete. Submit submissions/submission_hill_climb.csv to Kaggle.")


if __name__ == "__main__":
    main()
