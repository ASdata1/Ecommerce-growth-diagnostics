# Ecommerce Growth Diagnostics

A small end-to-end analytics project: take messy raw order data, turn it into a clean queryable
database, and use SQL to answer the kind of funnel, retention, and segmentation questions a
product/growth analyst gets asked at any company with a marketplace or checkout flow.

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
| Top-10% decile capture | 18% of repeaters (1.8x lift) | 10% (random) |
| Top-20% decile capture | 31% of repeaters | 20% (random) |

Repeat purchase is rare (3.3% of 83,644 first-time customers) and only
weakly predictable from first-order data — not a reliable classifier,
but useful for ranking/targeting.

## The data

[Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
— ~100k real orders placed on a Brazilian e-commerce marketplace, 2016-2018. 6 of the 8 tables
are used:

- `olist_orders_dataset.csv` — order status + timestamps
- `olist_order_items_dataset.csv` — line items, price, freight
- `olist_order_payments_dataset.csv` — payment type, installments, value
- `olist_customers_dataset.csv` — customer id + state
- `olist_order_reviews_dataset.csv` — review score (added for the repeat-purchase model)
- `olist_products_dataset.csv` — product category (added for the repeat-purchase model)

## How it's built

1. [`src/etl.py`](src/etl.py) — reads the raw CSVs, does light cleaning (nulls, dtypes, dedup),
   loads them into a local SQLite database at `data/olist.db`, and indexes the join columns every
   downstream query relies on. Deliberately does **not** impute or otherwise transform values —
   that decision belongs downstream, informed by EDA, not baked silently into the raw data.
2. [`src/migrate_to_cloud.py`](src/migrate_to_cloud.py) — pushes the local SQLite tables to a
   cloud Postgres database (free-tier Supabase) so the dashboard and model run against a real
   cloud data source instead of a local file.
3. [`queries/`](queries) — one SQL file per question above, plus `step_funnel.sql` for the
   stage-over-stage conversion view and `repeat_purchase_features.sql` for the model's feature set.
   Every feature is built from the customer's **first order only** (pulling from later orders would
   leak the target); the query applies a **right-censoring cutoff** (first orders in the last 3
   months of the data are dropped — those customers haven't had time to come back yet); and it
   marks `review_score` as "known" only when the review arrived within **30 days of delivery**.
   That window depends only on delivery and review dates, not on whether or when a second order
   happened, so `review_score`'s null-ness is not a function of the target. Late or missing
   reviews leave it null and are median-imputed by the model; `review_delay_days` is emitted as an
   EDA diagnostic only. See the query's header comments for the full reasoning.
4. [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb) — runs each query against
   `data/olist.db`, charts the result, and writes up the finding.
5. [`notebooks/repeat_purchase_eda.ipynb`](notebooks/repeat_purchase_eda.ipynb) — EDA on the raw
   feature-query output, run **before** any imputation or modeling decision: class balance,
   missingness per column and how to handle the missing `review_score` values, a VIF check on the
   collinear delivery/payment features, whether the recency cutoff actually removed the
   right-censoring problem, and per-feature distributions split by target.
6. [`src/repeat_purchase_analysis.py`](src/repeat_purchase_analysis.py) — hypothesis tests
   (Welch's t-test, chi-square) and a logistic regression predicting repeat purchase: 5-fold
   stratified cross-validation on the training set, a likelihood-ratio test (via `statsmodels`)
   comparing plain features against added interaction terms, and odds ratios for interpretability
   on the held-out test set. See the module docstring for the full reasoning. Every CV comparison
   and the final held-out evaluation are logged to MLflow via
   [`src/experiment_tracking.py`](src/experiment_tracking.py) — each run is tagged with the git
   commit it ran on and the exact feature set used, so a metric quoted anywhere is traceable back
   to what produced it. Run `mlflow ui` from the project root to browse past runs.
7. [`power_bi/`](power_bi) — dashboard connected to the cloud database. See
   [`power_bi/README.md`](power_bi/README.md) for setup and page layout.
8. [`tests/`](tests) — data-quality checks on the ETL output (`test_etl.py`) and sanity checks
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
  captures 18% of those who actually return (1.8x lift over random), the top 20% captures 31%.
- **The measurable drivers are "what" and "where", not "how the first order went".** Review score
  differs by a significant-but-trivial 0.06 points between repeaters and one-timers (Welch t-test
  p=0.039, Cohen's d=0.04); payment type shows no relationship (chi-square p=0.13); delivery
  time, lateness, and order value all have |Cohen's d| < 0.08. The largest odds ratios are all
  product category and region — fashion-accessory, bed/bath, and furniture/decor first orders
  carry roughly 1.5–2.3x the repeat odds of electronics and "cool stuff", and customers in Rio
  de Janeiro repeat more than those in Ceará. Treat these as directional: they're
  `class_weight="balanced"` point estimates with no confidence intervals.
- **Interaction terms didn't earn their place** — adding pairwise numeric interactions improved
  model fit significantly (likelihood-ratio test p=0.008) but added no cross-validated PR-AUC, so
  the simpler, interpretable model is the one reported.

## Where this is going

The analysis so far is diagnostic, not causal, and nothing is wired into a live workflow yet.
Planned next steps:

- **~~Experiment tracking with MLflow~~ — in place.** Every CV run and the final held-out
  evaluation log their feature set, config, and metrics to MLflow, tagged with the git commit
  they ran on (see `src/experiment_tracking.py`). Set up now, ahead of the geolocation/seller
  feature work below, so that comparison is a real run-over-run diff instead of "from memory."

- **More features from the geolocation and seller data.** Fold `olist_geolocation_dataset.csv` and
  `olist_sellers_dataset.csv` into the feature query — customer↔seller distance, seller state,
  delivery-region density — to test whether *how far the order travelled* and *who sold it* carry
  signal the current "what / where" features miss.

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
