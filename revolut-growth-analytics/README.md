# Revolut Growth Analytics Speedrun

A weekend-scoped SQL + Python project that mirrors what a Revolut Graduate Data Scientist/Analyst
actually does in week one: turn messy raw data into a clean, queryable table, then answer real
product questions — where are we losing customers in the funnel, and which cohorts stick around.

Built as a portfolio piece for my application to Revolut's Graduate Programme (see
`applications/Revolut_Graduate-Data-Scientist-Analyst/` in this job-hunt folder for the full
gap analysis this project came out of).

## The data

[Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
— ~100k real orders, 2016-2018. Free, requires a (free) Kaggle account to download.

Only 4 of the 8 tables are used, to keep this a weekend project:
- `olist_orders_dataset.csv` — order status + timestamps
- `olist_order_items_dataset.csv` — line items, price, freight
- `olist_order_payments_dataset.csv` — payment type, installments, value
- `olist_customers_dataset.csv` — customer id + state

Download the zip, unzip it, and drop those 4 CSVs into `data/raw/` (already gitignored — don't
commit raw data).

## The 3 questions this project answers

1. **Funnel** — of all orders placed, what % make it through each status
   (created → approved → invoiced → shipped → delivered)? Where's the biggest drop-off?
2. **Cohort retention** — group customers by the month of their first order. What % of each
   cohort places a second order in month 1, 2, 3...?
3. **Payment/segment cut** — how does average order value and payment installments vary by
   customer state? (a rough stand-in for "which segment is most valuable")

## How it's built

1. `src/etl.py` — reads the raw CSVs, does light cleaning (nulls, dtypes, dedup), loads them
   into a local SQLite database at `data/olist.db`. This part is fully working — run it first.
2. `queries/*.sql` — one SQL file per question above. **These are intentionally left as
   skeletons with TODOs** — writing them is the actual skill-building part of this project.
   See "Skills you'll need" below.
3. `notebooks/analysis.ipynb` — runs each query against `data/olist.db`, charts the result,
   and has a markdown cell for you to write the finding in plain English.
4. `tests/test_etl.py` — a couple of sanity checks on the ETL output (row counts, no null keys).

## Setup

```bash
pip install -r requirements.txt
# download data/raw/*.csv from the Kaggle link above first
python src/etl.py
pytest tests/
jupyter notebook notebooks/analysis.ipynb
```

## Skills you'll need to learn to finish this (and where)

| File | What's missing | Skill to learn |
| :-- | :-- | :-- |
| `queries/funnel.sql` | The funnel query itself | SQL `CASE WHEN`, `GROUP BY`, computing % of a total (often via a CTE or subquery for the denominator) |
| `queries/cohort_retention.sql` | The cohort query | SQL window functions (`MIN() OVER`), date truncation (`strftime` in SQLite), self-joins or `GROUP BY` to build a cohort table |
| `queries/payment_by_state.sql` | The aggregation query | `GROUP BY` + `AVG()`/`COUNT()`, optionally `HAVING` to filter groups |
| `tests/test_etl.py` | A couple of assertions | Basic `pytest` — one test per data-quality check |
| `notebooks/analysis.ipynb` | The charts | `matplotlib` or `plotly` basics — a bar chart for the funnel, a heatmap for cohort retention |

I can walk through any of these with you when you're ready — window functions and the cohort
query are the two genuinely new skills here (everything else is a small step up from what you've
already done in your SQL/pandas courses).

## Why this project (and where else it's useful)

This was scoped specifically to close the gap flagged for the Revolut posting: your Python/SQL
are course-only, and Revolut wants "impressive" applied skill in exactly this kind of funnel/
cohort/ad-hoc analysis. But the pattern is not fintech-specific:

- **Any product/growth analyst role** (your target sectors — Sports, Healthcare, Tech, Cyber,
  Production, Research) has a signup funnel and a retention question. Same repo, same queries,
  a different story in the README about who the "customer" is.
- **ML Engineer roles** care about the ETL half specifically — most ML engineering work is
  building and testing clean data pipelines *before* any model gets trained. The `src/etl.py` +
  `tests/test_etl.py` pair is exactly that pattern in miniature.
- **The repo structure itself** (`src/`, `tests/`, `requirements.txt`) is a stated gap in your
  profile (SWE best practices, rated 2/5) — this closes it regardless of which company you send
  it to.
- SQLite here is a stand-in for Snowflake/BigQuery/Postgres — the SQL you write transfers
  directly; only the connection string changes.

If you want, once this is done we can re-skin the same repo (new dataset, new README framing)
for a sports- or healthcare-flavoured application without redoing the SQL skills from scratch.
