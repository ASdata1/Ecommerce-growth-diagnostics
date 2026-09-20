"""
Wald confidence intervals for the logistic regression's log-odds coefficients.

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
    return {f"cat__{next(name for name in output_names if name.startswith(f'{col}_'))}" for col in cat_columns}


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
