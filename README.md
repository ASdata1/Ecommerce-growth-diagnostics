# Product Growth Analytics

A small end-to-end analytics project: take messy raw order data, turn it into a clean queryable
database, and use SQL to answer the kind of funnel, retention, and segmentation questions a
product/growth analyst gets asked in week one at any company with a marketplace or checkout flow.

## Questions this project answers

1. **Funnel** — of all orders placed, what % make it through each status (purchased → approved →
   shipped → delivered), and which single handoff loses the most orders?
2. **Cohort retention** — grouping customers by the month of their first order, what % of each
   cohort comes back and places a second order in month 1, month 2?
3. **Payment / segment cut** — how does average order value and payment installments vary by
   customer region, and where's the volume vs. value mismatch?

Each question is answered with a SQL query, a chart, and a short written finding in
[`notebooks/analysis.ipynb`](notebooks/analysis.ipynb).

## The data

[Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
— ~100k real orders placed on a Brazilian e-commerce marketplace, 2016-2018. Free, requires a
(free) Kaggle account to download. Only 4 of the 8 tables are used:

- `olist_orders_dataset.csv` — order status + timestamps
- `olist_order_items_dataset.csv` — line items, price, freight
- `olist_order_payments_dataset.csv` — payment type, installments, value
- `olist_customers_dataset.csv` — customer id + state

## How it's built

1. [`src/etl.py`](src/etl.py) — reads the raw CSVs, does light cleaning (nulls, dtypes, dedup),
   loads them into a local SQLite database at `data/olist.db`, and indexes the join columns every
   downstream query relies on.
2. [`queries/`](queries) — one SQL file per question above, plus `step_funnel.sql` for the
   stage-over-stage conversion view.
3. [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb) — runs each query against
   `data/olist.db`, charts the result, and writes up the finding.
4. [`tests/test_etl.py`](tests/test_etl.py) — data-quality checks on the ETL output: tables aren't
   empty, key columns have no nulls, and referential integrity holds between orders/customers/items.

## Skills demonstrated

- **SQL**: CTEs, window-function-style cohort logic (`MIN()` per customer, month-offset math),
  date truncation (`strftime`), self-joins, `GROUP BY`/`HAVING` aggregation, and indexing to keep
  joins fast on ~100k-row tables.
- **Python / pandas**: CSV ingestion, null handling, dtype coercion, de-duplication, and loading
  into SQLite.
- **Data visualisation**: `matplotlib` bar charts and a cohort-retention heatmap.
- **Testing**: `pytest` fixtures and data-quality/referential-integrity assertions against a real
  database, not mocks.
- **Project structure**: a conventional `src/` + `tests/` + `queries/` layout instead of one
  notebook doing everything.

## How to run it

```bash
git clone <this-repo-url>
cd <cloned-directory>
pip install -r requirements.txt

# download the 4 CSVs listed above from the Kaggle link and place them in data/raw/
python src/etl.py        # builds data/olist.db
pytest tests/            # runs the data-quality checks
jupyter notebook notebooks/analysis.ipynb   # or open it in VS Code / JupyterLab
```

Run all cells top to bottom in the notebook — it reads `../data/olist.db` and `../queries/*.sql`
relative to its own folder, so it works from a fresh clone as long as `src/etl.py` has been run
first.

## Headline findings

- The funnel is healthy end-to-end (97% of orders reach `delivered`), but the single biggest
  stage-to-stage drop is **approved → shipped**, not payment approval or last-mile delivery.
- Repeat purchase is the real growth problem: month-1 cohort retention sits under 1% almost
  everywhere in the dataset, with no improving trend over time.
- Order value and order volume are inversely related by region — the highest-volume region has
  the *lowest* average order value, while low-volume, farther-out regions pay the most, most
  likely reflecting freight cost baked into payment value.
