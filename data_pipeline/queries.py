"""
Zepto Data Pipeline - SQL Queries
Runs 6 SQL queries against zepto_books.db covering SELECT/WHERE, ORDER BY,
LIMIT, DISTINCT, IN, BETWEEN, and JOIN. Also demonstrates pd.read_sql vs
pd.merge equivalence for the JOIN query.
"""

import sqlite3
import os
import pandas as pd

DB_PATH = os.path.join("data_pipeline", "zepto_books.db")

QUERIES = {
    "1_select_where": """
        SELECT title, price_gbp, rating
        FROM books
        WHERE rating >= 4
        ORDER BY price_gbp DESC
        LIMIT 10
    """,

    "2_order_by_limit": """
        SELECT title, price_inr
        FROM books
        ORDER BY price_inr DESC
        LIMIT 5
    """,

    "3_distinct": """
        SELECT DISTINCT category_name
        FROM categories
        ORDER BY category_name
    """,

    "4_in_clause": """
        SELECT title, rating
        FROM books
        WHERE rating IN (4, 5)
        ORDER BY rating DESC
        LIMIT 10
    """,

    "5_between": """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 20 AND 40
        ORDER BY price_gbp
    """,

    "6_join_top_rated_per_category": """
        SELECT c.category_name, b.title, b.rating, b.price_gbp
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        WHERE b.rating >= 4
        ORDER BY c.category_name, b.rating DESC, b.price_gbp DESC
        LIMIT 15
    """,
}


def main():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"{DB_PATH} not found. Run `python data_pipeline/scrape.py` first."
        )

    conn = sqlite3.connect(DB_PATH)

    print("=" * 70)
    print("SQL QUERIES AND OUTPUTS")
    print("=" * 70)
    for name, sql in QUERIES.items():
        print(f"\n--- {name} ---")
        print("SQL:")
        print(sql.strip())
        print("\nOutput:")
        result = pd.read_sql(sql, conn)
        print(result.to_string(index=False))

    print("\n" + "=" * 70)
    print("pd.read_sql vs pd.merge EQUIVALENCE (JOIN query)")
    print("=" * 70)

    # Approach 1: SQL JOIN via pandas.read_sql
    sql_join_df = pd.read_sql(QUERIES["6_join_top_rated_per_category"], conn)
    print("\n-- pd.read_sql (SQL JOIN) --")
    print(sql_join_df.to_string(index=False))

    # Approach 2: pure pandas merge (no SQL for the join itself)
    books_df = pd.read_sql("SELECT * FROM books", conn)
    cats_df = pd.read_sql("SELECT * FROM categories", conn)

    merge_df = (
        books_df
        .merge(cats_df, on="category_id", how="inner")
        .query("rating >= 4")
        .sort_values(
            by=["category_name", "rating", "price_gbp"],
            ascending=[True, False, False],
        )
        .head(15)
        [["category_name", "title", "rating", "price_gbp"]]
        .reset_index(drop=True)
    )
    print("\n-- pd.merge (no SQL for JOIN) --")
    print(merge_df.to_string(index=False))

    # Compare — reset index so equality check aligns
    sql_reset = sql_join_df.reset_index(drop=True)
    print("\nEquivalent outputs:", sql_reset.equals(merge_df))

    conn.close()


if __name__ == "__main__":
    main()
