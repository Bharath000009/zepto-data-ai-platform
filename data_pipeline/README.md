# Data Pipeline — Zepto Module 1

Scrapes books.toscrape.com, cleans fields, converts GBP to INR using a fixed project rate, loads into a normalized SQLite database, and runs 6 SQL queries against it.

## Install

From the project root:

    pip install -r requirements.txt

## Run

    # 1. Scrape + clean + load into SQLite
    python data_pipeline/scrape.py

    # 2. Run 6 SQL queries + verify pd.read_sql vs pd.merge equivalence
    python data_pipeline/queries.py

## Outputs

- **books_clean.csv** — Cleaned scraped data (163 books across 5 categories)
- **zepto_books.db** — SQLite DB with categories and books tables
- **queries_output.txt** — Captured output of all 6 queries and pandas equivalence check

## Design Decisions

### Scope

Scraped the first 5 categories from the sidebar (Travel, Mystery, Historical Fiction, Sequential Art, Classics) with pagination. Final dataset: 163 books across 5 categories — above the required minimum of 60 books across 3 categories.

### Cleaning

- **Price**: stripped the pound symbol with regex, then cast to float → price_gbp.
- **Rating**: mapped text ratings ("One" to "Five") to integers 1-5. Unparseable values are median-imputed — safer than dropping since they are rare.
- **Availability**: converted free-text into boolean in_stock via substring match on "In stock".
- **Parse failures**: rows with unparseable price or availability would be dropped. No rows required dropping in this run.

### Currency Conversion

price_inr = price_gbp × 105.50

The rate **1 GBP = 105.50 INR** is a fixed, project-defined constant. No API call, no date reference, no network access. This is the required graded baseline.

### Database Schema

Two tables with PK/FK relationship:

- **categories** (category_id INTEGER PRIMARY KEY AUTOINCREMENT, category_name TEXT UNIQUE NOT NULL)
- **books** (book_id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, price_gbp REAL NOT NULL, price_inr REAL NOT NULL, rating INTEGER NOT NULL, in_stock INTEGER NOT NULL, category_id INTEGER NOT NULL REFERENCES categories(category_id))

The foreign key books.category_id to categories.category_id enforces referential integrity.

### SQL Query Coverage

- Query 1 (1_select_where): SELECT + WHERE
- Query 2 (2_order_by_limit): ORDER BY + LIMIT
- Query 3 (3_distinct): DISTINCT
- Query 4 (4_in_clause): IN
- Query 5 (5_between): BETWEEN
- Query 6 (6_join_top_rated_per_category): JOIN between books and categories

Full SQL text and outputs are in queries_output.txt.

### pandas Equivalence Check

Query 6 (the JOIN) was executed two ways:

- **pd.read_sql** — direct SQL JOIN executed against SQLite.
- **pd.merge** — pure pandas merge on category_id, no SQL used.

Final printed line: **Equivalent outputs: True** — both approaches produce identical results.

## Reproducibility

Running python data_pipeline/scrape.py from a clean checkout regenerates books_clean.csv and zepto_books.db from scratch. The SQLite file is committed for grading, but it can always be regenerated.
