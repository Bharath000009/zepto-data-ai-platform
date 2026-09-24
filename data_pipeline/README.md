# Data Pipeline — Zepto Module 1

Scrapes books.toscrape.com, cleans fields, converts GBP→INR, loads into a normalized SQLite database, and runs 6 SQL queries against it.

## Install

From the project root (all deps listed in the root `requirements.txt`):

```bash
pip install -r requirements.txt