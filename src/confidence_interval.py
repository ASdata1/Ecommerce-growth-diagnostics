"""
Confidence intervals for the repeat-purchase model: Wald CIs on the logistic
regression's log-odds coefficients (compute_confidence_intervals()), and
bootstrap CIs on the held-out test set's headline metrics
(bootstrap_metric_cis()). Two different methods because they answer two
different questions with two different kinds of quantity - a fitted
coefficient (closed-form standard error, if the model is unregularized) vs. a
metric computed from predictions on one sample (no closed form; resampling
is the standard way to get its sampling distribution).

odds_ratio_table() in repeat_purchase_analysis.py reports point-estimate
log-odds from the DEPLOYED sklearn pipeline - L2-regularized (LogisticRegression's
default penalty) and fit with class_weight="balanced" to optimize predictive
performance under class imbalance. Regularized, weighted-loss coefficients don't
have a clean closed-form standard error, so they can't get a Wald confidence
interval directly off that model.

This module instead fits a second, unregularized, unweighted statsmodels Logit
on the exact same preprocessed design matrix - the fitted pipeline's own
"preprocess" step (imputer/scaler/one-hot encoder) is reused as-is, not
reimplemented - purely for inference. This mirrors the split
likelihood_ratio_test() already draws in this project between the sklearn model
(tuned for predictive performance) and a statsmodels model (used only for
hypothesis testing), extended here from the numeric-only LR test to every
feature in the model. Class imbalance biases predicted probabilities and
decision thresholds, not the consistency of unweighted MLE log-odds estimates,
so leaving weighting out of the inference model is a deliberate choice, not an
oversight.

Because of that split, expect the two point estimates (statsmodels' unregularized
vs. sklearn's regularized) to be close but not identical - that gap is reported
alongside the CI, not hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline


def _clean_feature_names(names) -> list[str]:
    return [str(n).replace("num__", "").replace("cat__", "") for n in names]


def _categorical_reference_levels_to_drop(preprocessor) -> set[str]:
    """The ColumnTransformer's OneHotEncoder keeps every category (no dropped
    reference level) - fine for the deployed, L2-regularized sklearn model, which
    tolerates the redundancy, but fatal for an unregularized fit: intercept +
    every dummy level for a feature is the classic dummy-variable trap (the
    dummies for that feature sum to 1 on every row, exactly reproducing the
    intercept column), which makes the Hessian singular and standard errors NaN.
    Returns one prefixed output column name per categorical feature to exclude,
    equivalent to what OneHotEncoder(drop="first") would have produced."""
    _, cat_pipeline, cat_columns = next(t for t in preprocessor.transformers_ if t[0] == "cat")
    ohe = cat_pipeline.named_steps["encode"]
    output_names = ohe.get_feature_names_out(cat_columns)

    drop_names = set()
    for col in cat_columns:
        first_level = next(name for name in output_names if name.startswith(f"{col}_"))
        drop_names.add(f"cat__{first_level}")
    return drop_names


def compute_confidence_intervals(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, alpha: float = 0.05
) -> pd.DataFrame:
    """
    `pipeline` must already be fitted, with "preprocess" and "model" steps (as
    build_pipeline() + .fit(X, y) in repeat_purchase_analysis.py produces), and
    `X`/`y` should be whatever it was fit on - so the CI companion model's point
    estimate is directly comparable to odds_ratio_table()'s point estimate for
    the same run, not from a different split.

    Returns one row per feature (plus "intercept"): the statsmodels point
    estimate and standard error, the alpha-level Wald CI on the log-odds
    coefficient and on the odds ratio, the p-value, whether the CI excludes
    zero, and the sklearn pipeline's own (regularized) coefficient for
    comparison. Sorted by |statsmodels coefficient| descending.
    """
    preprocessor = pipeline.named_steps["preprocess"]
    sklearn_model = pipeline.named_steps["model"]

    design = preprocessor.transform(X)
    if hasattr(design, "toarray"):  # sparse ColumnTransformer output
        design = design.toarray()
    design = np.asarray(design)

    raw_names = preprocessor.get_feature_names_out()
    drop_raw = _categorical_reference_levels_to_drop(preprocessor)
    keep = np.array([name not in drop_raw for name in raw_names])
    design = design[:, keep]
    feature_names = _clean_feature_names(raw_names[keep])
    names = ["intercept"] + feature_names

    design_with_const = sm.add_constant(design, has_constant="add")
    # lbfgs rather than Logit's default Newton step - more robust to the
    # collinearity one-hot-encoded categoricals introduce; conf_int()/bse still
    # come from the Hessian at the optimum regardless of which optimizer found it.
    sm_fit = sm.Logit(y.to_numpy(), design_with_const).fit(method="lbfgs", disp=0, maxiter=500)

    ci = sm_fit.conf_int(alpha=alpha)
    result = pd.DataFrame({
        "feature": names,
        "coefficient_statsmodels": sm_fit.params,
        "std_error": sm_fit.bse,
        "ci_lower": ci[:, 0],
        "ci_upper": ci[:, 1],
        "p_value": sm_fit.pvalues,
    })
    result["odds_ratio_statsmodels"] = np.exp(result["coefficient_statsmodels"])
    result["odds_ratio_ci_lower"] = np.exp(result["ci_lower"])
    result["odds_ratio_ci_upper"] = np.exp(result["ci_upper"])
    result["significant_at_alpha"] = (result["ci_lower"] > 0) | (result["ci_upper"] < 0)

    # sklearn's own coef_ still has ALL one-hot columns (it never drops a
    # reference level - only this inference copy does), so build its comparison
    # series from the full, unfiltered name list and let the merge below narrow
    # it down to the features `result` actually has rows for.
    sklearn_all_names = ["intercept"] + _clean_feature_names(raw_names)
    sklearn_coefs = pd.Series(
        np.concatenate([sklearn_model.intercept_, sklearn_model.coef_[0]]),
        index=sklearn_all_names,
        name="coefficient_sklearn_deployed",
    )
    result = result.merge(sklearn_coefs.rename_axis("feature").reset_index(), on="feature")

    result = (
        result.assign(abs_coef=lambda d: d["coefficient_statsmodels"].abs())
        .sort_values("abs_coef", ascending=False)
        .drop(columns="abs_coef")
        .reset_index(drop=True)
    )

    n_significant = int(result["significant_at_alpha"].sum())
    print(
        f"\n--- {alpha:.0%}-level Wald confidence intervals on log-odds coefficients "
        f"({len(result)} features incl. intercept) ---"
    )
    print(f"{n_significant}/{len(result)} have a CI that excludes zero (95% CI on the odds ratio excludes 1)")
    print(
        result.loc[~result["feature"].eq("intercept")]
        .head(5)[["feature", "coefficient_statsmodels", "ci_lower", "ci_upper", "p_value"]]
        .round(4)
        .to_string(index=False)
    )

    return result


def _test_set_metrics(y_true: np.ndarray, y_proba: np.ndarray, top_pcts: tuple[float, ...]) -> dict[str, float]:
    row = {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }
    for pct in top_pcts:
        k = int(len(y_true) * pct)
        top_idx = np.argsort(-y_proba)[:k]
        row[f"top_{int(pct * 100)}pct_capture_rate"] = 100 * y_true[top_idx].sum() / max(y_true.sum(), 1)
    return row


def bootstrap_metric_cis(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    top_pcts: tuple[float, ...] = (0.1, 0.2),
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Percentile bootstrap CIs on the held-out test set's headline metrics
    (ROC-AUC, PR-AUC, top-k% decile capture rate).

    The Wald approach above only works for a fitted model's coefficients -
    these are metrics computed from predictions on one fixed test set, with no
    closed-form standard error, so instead this resamples (y_true, y_proba)
    pairs WITH REPLACEMENT `n_boot` times, recomputes every metric on each
    resample, and reports the alpha-level percentile interval of that
    distribution. A resample where every row (or no row) is a repeat purchaser
    makes ROC-AUC/PR-AUC undefined and is skipped - rare at this class
    imbalance's sample size, but not impossible.

    This is what actually answers "is 16.8% top-decile capture a reliable
    number, or could it just as easily have been 12% on a different sample of
    the same customers?" - a question the point estimate alone can't answer.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    n = len(y_true)
    rng = np.random.default_rng(random_state)

    point = _test_set_metrics(y_true, y_proba, top_pcts)

    boot_rows = []
    while len(boot_rows) < n_boot:
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_proba[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        boot_rows.append(_test_set_metrics(yt, yp, top_pcts))
    boot_df = pd.DataFrame(boot_rows)

    result = pd.DataFrame({
        "metric": list(point.keys()),
        "point_estimate": list(point.values()),
        "ci_lower": [boot_df[m].quantile(alpha / 2) for m in point],
        "ci_upper": [boot_df[m].quantile(1 - alpha / 2) for m in point],
    })

    print(f"\n--- {alpha:.0%}-level bootstrap confidence intervals on test-set metrics ({n_boot} resamples) ---")
    print(result.round(4).to_string(index=False))

    return result
