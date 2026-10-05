from __future__ import annotations

import subprocess
from contextlib import contextmanager
from typing import Any, Iterator

import mlflow
import pandas as pd

EXPERIMENT_NAME = "repeat-purchase-propensity"


def git_commit() -> str:
    """Current commit, or "unknown" if git isn't available."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


@contextmanager
def track_run(run_name: str, params: dict[str, Any]) -> Iterator[Any]:
    """Starts an MLflow run tagged with the git commit, logs params, yields log_metrics."""
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("git_commit", git_commit())
        mlflow.log_params(params)
        yield mlflow.log_metrics


def log_table(data: pd.DataFrame, artifact_file: str) -> None:
    """Logs a DataFrame as a table artifact on the current run."""
    mlflow.log_table(data=data, artifact_file=artifact_file)
