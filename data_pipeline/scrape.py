"""
Zepto Data Pipeline - Scraper
Scrapes books.toscrape.com, cleans fields, converts GBP->INR at fixed 105.50,
and loads into a normalized SQLite database (categories + books).
"""

import os
import sqlite3
import requests
import pandas as pd
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://books.toscrape.com/"
RATING_MAP = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
# fixed project-defined conversion rate (1 GBP = 105.50 INR)
GBP_TO_INR = 105.50

DB_PATH = os.path.join("data_pipeline", "zepto_books.db")
CSV_PATH = os.path.join("data_pipeline", "books_clean.csv")


def get_category_links(num_categories=5):
    """Fetch the first N category links from the site's sidebar."""
    resp = requests.get(BASE_URL, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")
    links = soup.select("div.side_categories ul li ul li a")
    categories = []
    for a in links[:num_categories]:
        name = a.text.strip()
        url = urljoin(BASE_URL, a["href"])
        categories.append((name, url))
    return categories


def scrape_category(category_url, category_name):
    """Scrape all books from a category, following pagination."""
    books = []
    url = category_url
    while url:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        for article in soup.select("article.product_pod"):
            title = article.h3.a["title"]
            price_text = article.select_one("p.price_color").text.strip()
            rating_text = article.select_one("p.star-rating")["class"][1]
            availability_text = article.select_one(
                "p.instock.availability").text.strip()

            books.append({
                "title": title,
                "price_raw": price_text,
                "star_rating_text": rating_text,
                "availability_text": availability_text,
                "category": category_name,
            })

        next_btn = soup.select_one("li.next > a")
        url = urljoin(url, next_btn["href"]) if next_btn else None

    return books


def clean_dataframe(df):
    """Clean scraped fields and add derived columns."""
    df["price_gbp"] = (
        df["price_raw"]
        .astype(str)
        .str.replace("\u00a3", "", regex=False)
        .str.replace(r"[^0-9.]", "", regex=True)
        .astype(float)
    )

    df["rating"] = df["star_rating_text"].map(RATING_MAP)
    median_rating = df["rating"].median()
    df["rating"] = df["rating"].fillna(median_rating).astype(int)

    df["in_stock"] = df["availability_text"].str.contains(
        "In stock", case=False, na=False)

    df["price_inr"] = df["price_gbp"] * GBP_TO_INR

    return df[["title", "price_gbp", "price_inr", "rating", "in_stock", "category"]]


def load_to_sqlite(df, db_path=DB_PATH):
    """Create schema and load the cleaned dataframe."""
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
        CREATE TABLE categories (
            category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        );
        CREATE TABLE books (
            book_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            title       TEXT NOT NULL,
            price_gbp   REAL NOT NULL,
            price_inr   REAL NOT NULL,
            rating      INTEGER NOT NULL,
            in_stock    INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories(category_id)
        );
    """)

    for cat_name in sorted(df["category"].unique()):
        cur.execute(
            "INSERT INTO categories (category_name) VALUES (?)", (cat_name,))

    cur.execute("SELECT category_id, category_name FROM categories")
    cat_map = {name: cid for cid, name in cur.fetchall()}

    for _, row in df.iterrows():
        cur.execute(
            """INSERT INTO books
               (title, price_gbp, price_inr, rating, in_stock, category_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                row["title"],
                float(row["price_gbp"]),
                float(row["price_inr"]),
                int(row["rating"]),
                int(bool(row["in_stock"])),
                cat_map[row["category"]],
            ),
        )

    conn.commit()
    return conn


def main():
    print("Fetching category links...")
    categories = get_category_links(num_categories=5)
    print(f"Found {len(categories)} categories: {[c[0] for c in categories]}")

    all_books = []
    for name, url in categories:
        print(f"Scraping category: {name}")
        all_books.extend(scrape_category(url, name))

    df = pd.DataFrame(all_books)
    print(f"Total scraped rows: {len(df)}")

    df_clean = clean_dataframe(df)
    df_clean.to_csv(CSV_PATH, index=False)
    print(f"Saved cleaned CSV to {CSV_PATH}")

    conn = load_to_sqlite(df_clean)
    row_count = conn.execute("SELECT COUNT(*) FROM books").fetchone()[0]
    cat_count = conn.execute("SELECT COUNT(*) FROM categories").fetchone()[0]
    conn.close()
    print(
        f"Loaded {row_count} books across {cat_count} categories into SQLite at {DB_PATH}")


if __name__ == "__main__":
    main()
