"""Download official competition files through the authenticated Kaggle CLI."""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("input"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "spaceship-titanic.zip"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "kaggle",
            "competitions",
            "download",
            "-c",
            "spaceship-titanic",
            "-p",
            str(args.output),
            "--force",
        ],
        check=True,
    )
    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(args.output)
    expected = [args.output / name for name in ["train.csv", "test.csv", "sample_submission.csv"]]
    missing = [str(path) for path in expected if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Kaggle archive did not contain expected files: {missing}")
    print(f"downloaded: {', '.join(str(path) for path in expected)}")


if __name__ == "__main__":
    main()
