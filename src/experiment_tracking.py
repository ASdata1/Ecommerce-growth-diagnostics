"""
Thin MLflow wrapper used by every script in this project that trains or
evaluates a model.

The traceability goal (see README "Where this is going"): given any metric
quoted anywhere - README, CV, interview - it should be possible to say
exactly which git commit, feature set, and config produced it. A tracker
bolted on after a project is "done" can't do that retroactively; this is
meant to be in place *before* the next model change (new geolocation/seller
features, any tuning pass), not after.

Kept deliberately small - one context manager, no framework. If a script
someday needs a different tracker (W&B, etc.) this is the one place to swap it.
"""

from __future__ import annotations

import subprocess
from contextlib import contextmanager
from typing import Any, Iterator

import mlflow

EXPERIMENT_NAME = "repeat-purchase-propensity"


def _git_commit() -> str:
    """The commit the run actually executed on. Falls back to "unknown" rather
    than raising, so a missing git binary never takes down an analysis run -
    tracking is meant to help, not become a new point of failure."""
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
    """
    with track_run("cv-base-features", params={"interaction_terms": False, ...}) as log:
        val_pr_auc = cross_validate_pr_auc(...)
        log({"cv_pr_auc": val_pr_auc})

    Tags the run with the git commit it ran on, logs every param up front (so
    the config is visible even if the run later errors), and hands back
    mlflow.log_metrics so the caller can log results as they're computed.
    """
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("git_commit", _git_commit())
        mlflow.log_params(params)
        yield mlflow.log_metrics
