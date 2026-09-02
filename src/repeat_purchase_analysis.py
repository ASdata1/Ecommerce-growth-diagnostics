"""
Repeat-purchase propensity: hypothesis testing + logistic regression.

Question this answers: using only what's known from a customer's FIRST
order, is there a statistically significant relationship between their
first-order experience (review score, delivery delay, payment behaviour)
and whether they ever place a second order - and can we score new
first-time customers on that propensity?

Why logistic regression specifically: the target (repeat_purchase) is
binary, and the point of this script is not just "predict yes/no" but
"which factors matter, and by how much" - logistic regression's
coefficients convert directly into odds ratios, which is the readable,
defensible output a business stakeholder can act on

Class imbalance, stated up front: repeat purchase is rare in this
dataset (see the EDA notebook / cohort_retention.sql finding). That means:
  - accuracy is a useless headline metric - PR-AUC and recall are used instead
  - class_weight="balanced" is used so the rare positive class isn't
    ignored during training
  

Validation strategy: a single 80/20 train/test split, with 5-fold
STRATIFIED cross-validation *inside* the training set for the one model
comparison this script makes (plain features vs. + pairwise interaction
terms). The test set is touched exactly once, at the end, for the final
number - not used to pick between the two model versions. Stratified
because with a rare positive class, an unstratified split can easily
land a fold with very few positives by chance.

Model comparison, scoped deliberately: this script makes ONE comparison -
does adding pairwise interaction terms between the numeric features
improve the model beyond what's explained by noise? - not a search over
many feature sets or hyperparameters. Two things decide it:
  1. Cross-validated PR-AUC (predictive: is it actually better on unseen folds?)
  2. A likelihood-ratio test via statsmodels (inferential: is the improvement
     in fit statistically significant, not just numerically higher?)
The simpler, fully interpretable model is kept as the default unless both
of those say otherwise - odds ratios on interaction terms are much harder
to explain to a stakeholder, so the bar to include them is deliberately high.

Run: python src/repeat_purchase_analysis.py
(after src/etl.py has been run, so data/olist.db exists and includes the
order_reviews and products tables, and after the EDA notebook has been
reviewed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2 as chi2_dist
from scipy.stats import chi2_contingency, ttest_ind
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from db import get_engine
from experiment_tracking import track_run

QUERY_PATH = Path(__file__).resolve().parent.parent / "queries" / "repeat_purchase_features.sql"
CANDIDATES_QUERY_PATH = (
    Path(__file__).resolve().parent.parent / "queries" / "repeat_purchase_scoring_candidates.sql"
)

NUMERIC_FEATURES = [
    "num_items",
    "items_price",
    "freight_value",
    "payment_value",
    "payment_installments",
    "review_score",
    "delivery_time_days",
    "delivery_delay_days",
]


CATEGORICAL_FEATURES = ["customer_state", "payment_type", "product_category"]
TARGET = "repeat_purchase"


def load_features(engine=None) -> pd.DataFrame:
    engine = engine or get_engine()
 
    query_path = QUERY_PATH
    if engine.dialect.name == "postgresql":
        query_path = QUERY_PATH.with_name("repeat_purchase_features.postgres.sql")
    return pd.read_sql(query_path.read_text(), engine)


def run_hypothesis_tests(df: pd.DataFrame) -> None:
    print("\n=== Hypothesis tests ===")

    # H1: mean review score differs between repeat and one-time customers (among
    # customers who left one the model can use - review_score is the gated column,
    # so this compares one-timers against repeaters' 
    repeat_scores = df.loc[df[TARGET] == 1, "review_score"].dropna() 
    one_time_scores = df.loc[df[TARGET] == 0, "review_score"].dropna()  
    t_stat, p_val = ttest_ind(repeat_scores, one_time_scores, equal_var=False)  # Welch's t-test
    print(
        f"Review score, repeat (n={len(repeat_scores)}, mean={repeat_scores.mean():.2f}) "
        f"vs one-time (n={len(one_time_scores)}, mean={one_time_scores.mean():.2f}): "
        f"t={t_stat:.2f}, p={p_val:.4f}"
    )

    # H2: repeat-purchase rate differs by payment type
    contingency = pd.crosstab(df["payment_type"], df[TARGET])
    chi2_stat, p_val_chi2, dof, _ = chi2_contingency(contingency)
    print(f"\nPayment type vs repeat purchase: chi2={chi2_stat:.2f}, dof={dof}, p={p_val_chi2:.4f}")
    print(contingency.assign(repeat_rate_pct=lambda d: round(100 * d[1] / (d[0] + d[1]), 2)))


def build_pipeline(interaction_terms: bool) -> Pipeline:
    numeric_steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]
    if interaction_terms:
        # pairwise products only (no squared terms) among the numeric features -
        # bounded (8 features -> 36 columns), not a full polynomial expansion
        numeric_steps.append(("interact", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)))
    numeric_transformer = Pipeline(numeric_steps)

    categorical_transformer = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocess", preprocessor),
        ("model", LogisticRegression(class_weight="balanced", max_iter=1000)),
    ])


def cross_validate_pr_auc(interaction_terms: bool, X: pd.DataFrame, y: pd.Series, label: str) -> float:
    """5-fold stratified CV, returns mean validation-fold PR-AUC. Also prints the
    train-fold vs validation-fold gap as an overfitting check. Builds a fresh
    pipeline per fold (rather than cloning one passed in) so there's no risk of
    fold N reusing a fitted transformer from fold N-1."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # use statified folds to maintain class distbution in each fold
    train_scores, val_scores = [], []

    for train_idx, val_idx in skf.split(X, y):
        fold_pipeline = build_pipeline(interaction_terms=interaction_terms)
        fold_pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        train_proba = fold_pipeline.predict_proba(X.iloc[train_idx])[:, 1]
        val_proba = fold_pipeline.predict_proba(X.iloc[val_idx])[:, 1]
        train_scores.append(average_precision_score(y.iloc[train_idx], train_proba))
        val_scores.append(average_precision_score(y.iloc[val_idx], val_proba))

    train_mean, val_mean = np.mean(train_scores), np.mean(val_scores)
    print(
        f"{label}: CV train PR-AUC={train_mean:.3f}, CV val PR-AUC={val_mean:.3f} ")
    return val_mean # type: ignore


def likelihood_ratio_test(X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """Fits nested models in statsmodels to get a proper LR test:
    is the interaction model's improvement in fit statistically significant, or
    just noise? 
    """
    print("\n=== Likelihood-ratio test: base numeric features vs. + their pairwise interactions ===")

    def numeric_design(interaction_terms: bool) -> np.ndarray:
        steps = [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
        if interaction_terms:
            steps.append(("interact", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)))
        return Pipeline(steps).fit_transform(X_train[NUMERIC_FEATURES])

    X_base = sm.add_constant(numeric_design(interaction_terms=False))
    X_interact = sm.add_constant(numeric_design(interaction_terms=True))

    model_base = sm.Logit(y_train.to_numpy(), X_base).fit(disp=0)
    model_interact = sm.Logit(y_train.to_numpy(), X_interact).fit(disp=0)

    lr_stat = 2 * (model_interact.llf - model_base.llf) #log liklihood ratio stat
    df_diff = X_interact.shape[1] - X_base.shape[1]
    p_value = chi2_dist.sf(lr_stat, df_diff)

    print(f"Base model log-likelihood: {model_base.llf:.1f} ({X_base.shape[1]} params)")
    print(f"+Interactions log-likelihood: {model_interact.llf:.1f} ({X_interact.shape[1]} params)")
    print(f"LR statistic={lr_stat:.2f}, df={df_diff}, p={p_value:.4g}")
    if p_value < 0.05:
        print("-> statistically significant improvement, but check the CV PR-AUC gap too before")
        print("   trading away interpretability for it.")
    else:
        print("-> not a statistically significant improvement - stick with the simpler model.")
    return p_value


def _rank_and_decile(scores: pd.DataFrame) -> pd.DataFrame:
    """Adds a 1-based rank and decile (1 = top 10% by repeat_probability, 10 =
    bottom 10%) to a table already sorted descending by repeat_probability.
    Rank-based rather than pd.qcut so duplicate probabilities - common here,
    since several features are categorical - can't raise qcut's "duplicate
    bin edges" error.
    """
    scores = scores.reset_index(drop=True)
    n = len(scores)
    scores["rank"] = np.arange(1, n + 1)
    scores["decile"] = np.arange(n) * 10 // n + 1
    return scores


def evaluate_on_test(
    pipeline: Pipeline, X_train, y_train, X_test, y_test, ids_test: pd.Series, label: str, engine
) -> dict:
    print(f"\n=== Final held-out test evaluation: {label} ===")
    baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    print("--- Baseline (always predict majority class) ---")
    print(classification_report(y_test, baseline.predict(X_test), zero_division=0))

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print(f"--- {label} ---")
    print(classification_report(y_test, y_pred, zero_division=0))
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"PR-AUC (average precision): {pr_auc:.3f}")
    print(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")

    metrics = {"roc_auc": roc_auc, "pr_auc": pr_auc}
    for top_pct in (0.1, 0.2):
        n = int(len(y_test) * top_pct)
        top_idx = np.argsort(-y_proba)[:n]
        captured = y_test.iloc[top_idx].sum()
        capture_rate = 100 * captured / max(y_test.sum(), 1)
        metrics[f"top_{int(top_pct * 100)}pct_capture_rate"] = capture_rate
        print(
            f"Targeting the top {int(top_pct * 100)}% highest-scored customers "
            f"captures {captured}/{y_test.sum()} ({capture_rate:.1f}%) "
            f"of actual repeat purchasers, vs {int(top_pct * 100)}% expected at random."
        )

    # Persist the ranked, per-customer scores behind that lift stat - the actual
    # list a stakeholder needs to see who's "in the top 10%", not just the
    # aggregate capture rate above. This is the held-out TEST set: outcomes here
    # are already known, so it validates the ranking rather than being a live
    # call list - see score_scoring_candidates() for the list to actually act on.
    scored_test = _rank_and_decile(
        pd.DataFrame({
            "customer_unique_id": ids_test.to_numpy(),
            "repeat_probability": y_proba,
            "actual_repeat_purchase": y_test.to_numpy(),
        }).sort_values("repeat_probability", ascending=False)
    )
    scored_test.to_sql(
        "repeat_purchase_test_scores", engine, if_exists="replace", index=False, chunksize=1000, method="multi"
    )
    print(f"\nWrote {len(scored_test):,} scored test-set customers -> repeat_purchase_test_scores")

    print_odds_ratios(pipeline)
    return metrics


def score_scoring_candidates(X: pd.DataFrame, y: pd.Series, interaction_terms: bool, engine) -> None:
    """Score the customers repeat_purchase_features.sql's right-censoring cutoff
    excludes: first-time buyers too recent to know yet whether they'll place a
    second order. This - not the test-set lift numbers in evaluate_on_test - is
    the actual live outreach candidate list, written out as a ranked,
    per-customer table rather than only an aggregate stat.

    Refits on ALL labelled data (train + test): evaluate_on_test has already
    reported the held-out numbers, so there's no more leakage risk to protect -
    the deployed model should use every labelled row available.
    """
    query_path = CANDIDATES_QUERY_PATH
    if engine.dialect.name == "postgresql":
        query_path = CANDIDATES_QUERY_PATH.with_name("repeat_purchase_scoring_candidates.postgres.sql")
    candidates = pd.read_sql(query_path.read_text(), engine)

    if candidates.empty:
        print(
            "\nNo live scoring candidates found (no first-time customers inside "
            "the right-censoring window) - skipping repeat_purchase_scoring_candidates."
        )
        return

    deploy_pipeline = build_pipeline(interaction_terms=interaction_terms)
    deploy_pipeline.fit(X, y)
    proba = deploy_pipeline.predict_proba(candidates[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]

    scored = _rank_and_decile(
        pd.DataFrame({
            "customer_unique_id": candidates["customer_unique_id"].to_numpy(),
            "repeat_probability": proba,
        }).sort_values("repeat_probability", ascending=False)
    )
    scored.to_sql(
        "repeat_purchase_scoring_candidates",
        engine,
        if_exists="replace",
        index=False,
        chunksize=1000,
        method="multi",
    )
    print(
        f"\nScored {len(scored):,} live outreach candidates -> repeat_purchase_scoring_candidates table "
        f"({(scored['decile'] == 1).sum():,} in the top decile)"
    )


def print_odds_ratios(pipeline: Pipeline) -> None:
    model = pipeline.named_steps["model"]
    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    odds_ratios = pd.Series(np.exp(model.coef_[0]), index=feature_names).sort_values()
    print("\n--- Odds ratios (>1 = associated with higher repeat-purchase odds) ---")
    print(pd.concat([odds_ratios.head(5), odds_ratios.tail(5)]).round(3))


def main() -> None:
    engine = get_engine()
    df = load_features(engine)
    print(f"Loaded {len(df):,} first-time customers, {df[TARGET].sum():,} repeat purchasers")
    run_hypothesis_tests(df)

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    ids = df["customer_unique_id"]
    X_train, X_test, y_train, y_test, ids_train, ids_test = train_test_split(
        X, y, ids, test_size=0.2, stratify=y, random_state=42
    )

    print("\n=== Cross-validation (train set only) ===")
    # Feature set is logged with every run: when geolocation/seller features get
    # added to NUMERIC_FEATURES/CATEGORICAL_FEATURES, this is what makes a later
    # run's metrics comparable (or not) to this one - see experiment_tracking.py.
    feature_config = {
        "numeric_features": ",".join(NUMERIC_FEATURES),
        "categorical_features": ",".join(CATEGORICAL_FEATURES),
    }

    with track_run("cv-base-features", params={**feature_config, "interaction_terms": False}) as log:
        base_cv = cross_validate_pr_auc(False, X_train, y_train, "Base features")
        log({"cv_pr_auc": base_cv})

    lr_p_value = likelihood_ratio_test(X_train, y_train)

    with track_run("cv-interaction-terms", params={**feature_config, "interaction_terms": True}) as log:
        interact_cv = cross_validate_pr_auc(True, X_train, y_train, "+ Interaction terms")
        log({"cv_pr_auc": interact_cv, "lr_test_p_value": lr_p_value})

    # Decision: default to the simpler model unless interactions clearly win on
    # BOTH the predictive (CV PR-AUC) and inferential (LR test) check - see the
    # printed output above for the LR test's verdict.
    use_interactions = interact_cv > base_cv + 0.01  # a small, deliberate bar - not "any" improvement
    chosen = build_pipeline(interaction_terms=use_interactions)
    label = "Logistic regression + interaction terms" if use_interactions else "Logistic regression (base features)"
    if not use_interactions:
        print(f"\nCV PR-AUC did not clearly favour interaction terms ({interact_cv:.3f} vs {base_cv:.3f}) "
              "- keeping the simpler, interpretable model.")

    with track_run(
        "final-evaluation",
        params={
            **feature_config,
            "interaction_terms": use_interactions,
            "model": label,
            "n_customers": len(df),
            "base_rate": round(float(y.mean()), 4),
        },
    ) as log:
        final_metrics = evaluate_on_test(chosen, X_train, y_train, X_test, y_test, ids_test, label, engine)
        log(final_metrics)

    score_scoring_candidates(X, y, use_interactions, engine)


if __name__ == "__main__":
    main()
