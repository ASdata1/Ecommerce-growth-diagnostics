# Ecommerce Growth Diagnostics

## Questions this project answers

1. **Funnel** — of all orders placed, what % make it through each status (purchased → approved →
   shipped → delivered), and which single handoff loses the most orders?
2. **Cohort retention** — grouping customers by the month of their first order, what % of each
   cohort comes back and places a second order in month 1, month 2?
3. **Payment / segment cut** — how does average order value and payment installments vary by
   customer region, and where's the volume vs. value mismatch?
4. **Repeat-purchase drivers** — using only what's known from a customer's first order (review
   score, delivery experience, payment behaviour, what and where they bought), what predicts
   whether they come back for a second? See
   [`notebooks/repeat_purchase_eda.ipynb`](notebooks/repeat_purchase_eda.ipynb) for the
   exploratory pass and [`src/repeat_purchase_analysis.py`](src/repeat_purchase_analysis.py) for
   the hypothesis tests and the logistic-regression model.

Questions 1–3 are each answered with a SQL query, a chart, and a short written finding in
[`notebooks/analysis.ipynb`](notebooks/analysis.ipynb). Question 4 has its own EDA notebook and
analysis script (see *How it's built* below).

## Result

| Metric | Model | Baseline |
|---|---|---|
| ROC-AUC | 0.59 | 0.50 (random) |
| PR-AUC | 0.052 | 0.033 (base rate) |
| Top-10% decile capture | 16.8% of repeaters (1.7x lift) | 10% (random) |
| Top-20% decile capture | 30.5% of repeaters | 20% (random) |

Repeat purchase is rare (3.3% of 83,644 first-time customers) and only
weakly predictable from first-order data — not a reliable classifier,
but useful for ranking/targeting.

## The data

[Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
— ~100k real orders placed on a Brazilian e-commerce marketplace, 2016-2018. All 8 tables
are used:

- `olist_orders_dataset.csv` — order status + timestamps
- `olist_order_items_dataset.csv` — line items, price, freight
- `olist_order_payments_dataset.csv` — payment type, installments, value
- `olist_customers_dataset.csv` — customer id + state
- `olist_order_reviews_dataset.csv` — review score (added for the repeat-purchase model)
- `olist_products_dataset.csv` — product category (added for the repeat-purchase model)
- `olist_geolocation_dataset.csv` — zip-prefix lat/lng (added for the repeat-purchase model's geo features)
- `olist_sellers_dataset.csv` — seller id + state (added for the repeat-purchase model's geo features)

## How it's built

1. [`src/etl.py`](src/etl.py) — reads the raw CSVs, does light cleaning (nulls, dtypes, dedup),
   loads them into a local SQLite database at `data/olist.db`, and indexes the join columns every
   downstream query relies on. Deliberately does **not** impute or otherwise transform values —
   that decision belongs downstream, informed by EDA, not baked silently into the raw data.
2. [`queries/`](queries) — one SQL file per question above, plus `step_funnel.sql` for the
   stage-over-stage conversion view and `repeat_purchase_features.sql` for the model's feature set.
   Every feature is built from the customer's **first order only** (pulling from later orders would
   leak the target); the query applies a **right-censoring cutoff** (first orders in the last 3
   months of the data are dropped — those customers haven't had time to come back yet); and it
   marks `review_score` as "known" only when the review arrived within **30 days of delivery**.
   That window depends only on delivery and review dates, not on whether or when a second order
   happened, so `review_score`'s null-ness is not a function of the target. Late or missing
   reviews leave it null and are median-imputed by the model; `review_delay_days` is emitted as an
   EDA diagnostic only. See the query's header comments for the full reasoning.
3. [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb) — runs each query against
   `data/olist.db`, charts the result, and writes up the finding.
4. [`notebooks/repeat_purchase_eda.ipynb`](notebooks/repeat_purchase_eda.ipynb) — EDA on the raw
   feature-query output, run **before** any imputation or modeling decision: class balance,
   missingness per column and how to handle the missing `review_score` values, a VIF check on the
   collinear delivery/payment features, whether the recency cutoff actually removed the
   right-censoring problem, and per-feature distributions split by target.
5. [`src/repeat_purchase_analysis.py`](src/repeat_purchase_analysis.py) — hypothesis tests
   (Welch's t-test, chi-square) and a logistic regression predicting repeat purchase: 5-fold
   stratified cross-validation on the training set, a likelihood-ratio test (via `statsmodels`)
   comparing plain features against added interaction terms, and odds ratios for interpretability
   on the held-out test set. See the module docstring for the full reasoning. Every CV comparison
   and the final held-out evaluation are logged to MLflow via
   [`src/experiment_tracking.py`](src/experiment_tracking.py) — each run is tagged with the git
   commit it ran on and the exact feature set used, so a metric quoted anywhere is traceable back
   to what produced it; past runs are browsable in the MLflow UI. The run also writes its
   headline results — odds ratios, hypothesis-test p-values, and the top-line metrics — back to
   the database as four small tables (`repeat_purchase_odds_ratios`,
   `repeat_purchase_confidence_intervals`, `repeat_purchase_hypothesis_tests`,
   `repeat_purchase_model_metrics`), each row stamped with `run_at` and the git commit, and
   mirrored to `Dashboard/exports/*.csv` for the dashboard.
   **Decision: why the confidence intervals come from a second, unregularized model.**
   [`src/confidence_interval.py`](src/confidence_interval.py) attaches a 95% Wald CI to each
   feature's log-odds coefficient — but the *deployed* model is L2-regularized (sklearn's default
   penalty) and fit with `class_weight="balanced"`, and regularized, reweighted coefficients don't
   have a clean closed-form standard error, so a Wald CI can't legitimately be put on them
   directly. Rather than approximate that, the module fits a **second, unregularized, unweighted
   `statsmodels.Logit`** on the exact same preprocessed design matrix (the deployed pipeline's own
   fitted preprocessing step — imputer, scaler, one-hot encoder — is reused as-is, not
   reimplemented), purely for inference. This is the same split the project already draws for the
   interaction-terms likelihood-ratio test: one model tuned for predictive performance, a separate
   one used only for hypothesis testing/inference, because a class-imbalance correction changes
   predicted probabilities and decision thresholds, not the consistency of an unweighted MLE
   estimate. The two models' point estimates are reported side by side (`coefficient_statsmodels`
   vs. `coefficient_sklearn_deployed`) rather than treated as interchangeable, since they can be
   close but aren't identical.
6. **Power BI dashboard** — built from the CSV exports in `Dashboard/exports/`: the funnel, the
   cohort-retention heatmap, the regional value/volume cut, and a repeat-purchase drivers page
   from the odds-ratio and hypothesis-test tables above. Not yet assembled — see *Where this is
   going*.
   **Temporary fix:** until the Power BI report is built, the same numbers are viewable as a
   self-contained HTML dashboard — [Olist Growth Dashboard](https://claude.ai/artifact/ULMkaLkcgrCmd6pFdqADDu)
   (growth overview, funnel + cohort retention, repeat-purchase drivers) — built with Claude
   directly from the `Dashboard/exports/*.csv` files, so the figures match what's in the database.
7. [`tests/`](tests) — data-quality checks on the ETL output (`test_etl.py`) and sanity checks
   on the model feature query (`test_repeat_purchase_features.py`), including the right-censoring
   cutoff and an independent recompute of the `review_score` 30-day timing gate.

## Headline findings

- The funnel is healthy end-to-end (97% of orders reach `delivered`), but the single biggest
  stage-to-stage drop is **approved → shipped**, not payment approval or last-mile delivery.
- Repeat purchase is the real growth problem: month-1 cohort retention sits under 1% almost
  everywhere in the dataset, with no improving trend over time.
- Order value and order volume are inversely related by region — the highest-volume region has
  the *lowest* average order value, while low-volume, farther-out regions pay the most, most
  likely reflecting freight cost baked into payment value.
- **Repeat purchase is rare and barely predictable from the first order.** Of 83,644 first-time
  customers (after the right-censoring cutoff), only 3.3% ever order again. A logistic regression
  on everything knowable at first-order time — review score, delivery speed and lateness, payment
  value, installments, product category, state — reaches only ROC-AUC 0.59 / PR-AUC 0.052 (base
  rate 0.033). It stays useful for *ranking*, though: targeting the top 10% of scored customers
  captures 16.8% of those who actually return (1.7x lift over random), the top 20% captures 30.5%.
- **The measurable drivers are "what" and "where", not "how the first order went".** Review score
  differs by a significant-but-trivial 0.06 points between repeaters and one-timers (Welch t-test
  p=0.039, Cohen's d=0.04); payment type shows no relationship (chi-square p=0.13); delivery
  time, lateness, and order value all have |Cohen's d| < 0.08. The largest odds ratios are all
  product category and region — fashion-accessory, bed/bath, and furniture/decor first orders
  carry roughly 1.5–2.3x the repeat odds of electronics and "cool stuff", and customers in Rio
  de Janeiro repeat more than those in Ceará. These are `class_weight="balanced"` point estimates
  from the deployed, L2-regularized model; [`src/confidence_interval.py`](src/confidence_interval.py)
  fits an unregularized companion model on the same design matrix to attach a 95% Wald confidence
  interval to each one (see the *decision* note under `src/repeat_purchase_analysis.py` above for
  why), logged to both MLflow (as a table artifact on every `final-evaluation` run) and the
  `repeat_purchase_confidence_intervals` table / `Dashboard/exports/*.csv`. The product-category
  and region drivers above are among the ones whose interval excludes zero;
  `customer_seller_distance_km` does too (see next point) — `same_state` and
  `seller_state_seller_count` don't clear that bar.
- **Interaction terms didn't earn their place** — adding pairwise numeric interactions improved
  model fit significantly (likelihood-ratio test p=0.012) but added no cross-validated PR-AUC, so
  the simpler, interpretable model is the one reported.
- **Geo features (distance, same-state, seller density) are in the model, but don't move the
  headline metric.** `customer_seller_distance_km`, `same_state`, and `seller_state_seller_count`
  (from `olist_geolocation_dataset.csv` and `olist_sellers_dataset.csv`) were added to
  `NUMERIC_FEATURES` — see [`notebooks/geolocation_analysis.ipynb`](notebooks/geolocation_analysis.ipynb)
  for the EDA case. Distance has a real, if tiny, effect (Welch's t-test p<0.0001, Cohen's d=-0.08;
  its 95% CI in the fitted model excludes zero), but adding all three left ROC-AUC and PR-AUC
  essentially flat and slightly *lowered* top-decile/quintile capture versus the pre-geo run —
  comparing the two runs' odds ratios, the `customer_state` coefficients shifted the most
  (up to ±0.26 in log-odds), suggesting the geo features are mostly reshuffling signal
  `customer_state` already carried (distance is, after all, a function of where the customer
  and seller are) rather than adding new information.

## Where this is going

The analysis so far is diagnostic, not causal, and nothing is wired into a live workflow yet.
Planned next steps:

- **Build the Power BI dashboard.** `src/repeat_purchase_analysis.py` and `notebooks/analysis.ipynb`
  write their outputs to `Dashboard/exports/*.csv`. Next step is to import those CSVs into
  Power BI and build the report pages (growth overview, funnel, cohort retention, repeat-purchase
  drivers). Until then, the [HTML dashboard](https://claude.ai/artifact/ULMkaLkcgrCmd6pFdqADDu)
  linked above stands in as a temporary fix, covering the same pages from the same CSV exports.

- **Show the confidence intervals on the dashboard.** They're now in the
  `repeat_purchase_confidence_intervals` table / `Dashboard/exports/*.csv`; the dashboard's
  repeat-purchase-drivers page still needs to join them onto the odds-ratio bar chart to show
  uncertainty next to each point estimate.

- **A written experiment design (not yet run)** for evaluating a retention campaign:
  - **The causal question the model can't answer.** The odds ratios say fashion-accessory first
    orders repeat more; they don't say nudging customers toward that category would raise repeat
    purchase. Separating the two needs a randomised intervention, or an observational design that
    names and adjusts for the likely confounders (product price, region-level income,
    seasonality).
  - **Targeting.** Who gets the intervention — e.g. the top deciles of the ranking model among
    customers still inside the censoring window — and the rationale for that cut.
  - **Primary metric.** Repeat-purchase rate within 90 days of the first order, treated vs. control.
  - **Guardrail metrics.** The campaign must not degrade customers *outside* it: overall
    purchasing rate, mean review score, and existing-customer retention rate are monitored against
    pre-set tolerances, and the campaign is pulled if any is breached.

- **Candidate interventions to test.** Post-first-purchase discount, personalised recommendations,
  free shipping, priority delivery — each with a different cost profile and a different plausible
  mechanism.

Everything above is in service of three questions:

1. **What drives repeat purchase, and how sure are we?** — drivers with uncertainty attached, not
   bare point estimates.
2. **Which customers should we target, and why?** — a defensible ranking with a stated rationale.
3. **Do the interventions actually work?** — measured against a control, with guardrails, not
   inferred from correlation.
