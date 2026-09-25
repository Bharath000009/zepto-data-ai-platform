# Zepto Data & AI Platform

One connected repository with three modules:
- `/data_pipeline` — data engineering (scrape → clean → SQLite → SQL queries)
- `/analytics` — analytics and modeling (EDA → 3 classifiers → tuning → regression)
- `/support_assistant` — grounded GenAI service (RAG with LangGraph + FastAPI)

## Setup

From the project root:

    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate

    pip install -r requirements.txt

All Python dependencies live in the single root `requirements.txt`.

## Run Each Module

### 1. Data Pipeline (`/data_pipeline`)

    python data_pipeline/scrape.py     # scrape + clean + load SQLite (produces 163 books)
    python data_pipeline/queries.py    # run 6 SQL queries + pandas equivalence check

Outputs: `books_clean.csv`, `zepto_books.db`, `queries_output.txt`.

Full design decisions: see `data_pipeline/README.md`.

### 2. Analytics (`/analytics`)

    python analytics/01_eda.py         # load Titanic once, clean, save titanic.csv, run EDA
    python analytics/02_modeling.py    # train 3 classifiers, tune RF, regress fare, save joblib

Outputs: `titanic.csv`, `titanic_clean.csv`, `best_pipeline.joblib`, plots in `analytics/plots/`.

Full written interpretations: see `analytics/README.md`.

### 3. Support Assistant (`/support_assistant`)

Runs fully offline by default — no API key required.

    # quick local test:
    python support_assistant/app.py

    # FastAPI server:
    uvicorn support_assistant.app:app --host 127.0.0.1 --port 8000

    # Docker:
    docker build -t zepto-assistant .
    docker run -p 7860:7860 zepto-assistant

Full pipeline architecture: see `support_assistant/README.md`.

## Design Decisions (per module)

### Data Pipeline

Scrapes 5 categories from `books.toscrape.com` with pagination (163 books
across 5 categories — above the 60-book / 3-category minimum). Cleans price,
rating, and availability into proper typed columns. Converts GBP to INR at
the fixed project rate **1 GBP = 105.50 INR** (a project-defined constant —
no live FX lookup, no date reference, no network call). Loads into a
normalized SQLite schema with two tables (`categories`, `books`) sharing a
PK/FK relationship. Runs 6 SQL queries covering SELECT/WHERE, ORDER BY,
LIMIT, DISTINCT, IN, BETWEEN, and JOIN; verifies the JOIN result matches
`pd.merge` via `pd.read_sql`.

### Analytics

Loads the Titanic dataset exactly once via `sns.load_dataset('titanic')` and
immediately saves `titanic.csv` as an offline fallback. Missing values are
handled per the threshold rule — `deck` (>30%) is dropped as a column, `age`
(5–30%) is median-imputed, `embarked`/`embark_town` (<5%) drops the 2 rows.
Six-column correlation matrix on the specified columns only (`survived`,
`pclass`, `age`, `sibsp`, `parch`, `fare`) — `adult_male` and `alone` are
excluded as derived flags. Stratified train/test split performed first, with
all preprocessing steps wrapped in a `ColumnTransformer` inside a
`Pipeline` — fit on train only, no leakage. Three classifiers trained and
compared (Logistic Regression, Decision Tree, Random Forest). Imbalance
comparison across baseline, `class_weight='balanced'`, and SMOTE applied
only to the training fold. `GridSearchCV` over Random Forest hyperparameters
with `oob_score=True` to report the OOB score. Multivariate linear regression
side-task predicting `fare` with residual plot and heteroscedasticity
conclusion. Best full pipeline (preprocessing + estimator) saved via
`joblib.dump`, reloadable end-to-end on raw input.

### Support Assistant

Eight Zepto policy documents embedded with `all-MiniLM-L6-v2` in a ChromaDB
persistent collection. LangGraph StateGraph with 3 nodes (`classify_intent`,
`retrieve_and_answer`, `direct_answer`) and a conditional edge that routes
based on intent classification. Retrieval always runs for real (embedding +
ChromaDB need no API key); only the generation step branches on `MOCK_LLM`.
Pydantic response schema (`answer`, `sources`, `confidence`) enforced on
every response. FastAPI `POST /ask` endpoint wrapped in a Dockerfile that
serves the app on port 7860. Default mode is fully offline — every LLM call
is gated behind `MOCK_LLM`, and the graded baseline uses only the
deterministic mock path.

## Git Workflow

Three feature branches, each with at least two commits and merged into
`main`:

- `feature/data-pipeline` — 3 commits (scraping + SQL, expanded README, final README)
- `feature/analytics` — 3 commits (EDA + modeling, clarifying note)
- `feature/support-assistant` — 1 commit (module + Dockerfile + README)

Verify with:

    git log --oneline --graph --all

## Notes

- No paid services are required anywhere in this project.
- The fixed currency rate 1 GBP = 105.50 INR is a project-defined constant —
  no live FX lookup is performed.
- The Support Assistant's mock mode (`MOCK_LLM` unset or `=1`) runs fully
  offline with no API key. `MOCK_LLM=0` is an optional, ungraded extension.
- All required deliverables are committed to this repository as text
  (`.py`, `.md`, `.txt`, `.csv`, `.sqlite`, `.joblib`) with chart PNGs
  included as supporting artifacts.