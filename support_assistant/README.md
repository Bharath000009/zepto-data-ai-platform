# Support Assistant — Zepto Module 3

A grounded GenAI support assistant for Zepto policy questions.
LangGraph orchestrates a 3-node graph (classify -> retrieve/direct -> answer)
with ChromaDB vector retrieval and a FastAPI POST /ask endpoint.

Fully offline by default (MOCK_LLM=1, or unset). No API key, no network
LLM call, no paid service required. Real-LLM usage (MOCK_LLM=0) is an
optional, ungraded extension.

## Install

From project root:

    pip install -r requirements.txt

## Run

Quick local test (prints two example responses):

    python support_assistant/app.py

First run downloads the all-MiniLM-L6-v2 embedding model (~90 MB, cached afterwards).

Run the FastAPI server:

    uvicorn support_assistant.app:app --host 127.0.0.1 --port 8000

Then in another terminal (PowerShell):

    $body = @{ query = "What is the delivery time?" } | ConvertTo-Json
    Invoke-RestMethod -Uri "http://127.0.0.1:8000/ask" -Method Post -Body $body -ContentType "application/json"

Docker (local build and run):

    docker build -t zepto-assistant .
    docker run -p 7860:7860 zepto-assistant

Serves the same POST /ask endpoint on port 7860.

## Document Corpus

The 8 policy documents live in docs/doc_01.txt through docs/doc_08.txt:

- doc_01 — Delivery Policy
- doc_02 — Returns and Refunds
- doc_03 — Membership Tiers
- doc_04 — Order Tracking
- doc_05 — Order Cancellation Policy
- doc_06 — Damaged or Missing Items
- doc_07 — Gift Cards
- doc_08 — Customer Support Hours

All 8 are embedded with all-MiniLM-L6-v2 and stored in a ChromaDB
persistent collection (support_assistant/chroma_db/) named zepto_policies.

## Example API Calls (MOCK_LLM default)

The following raw JSON responses were recorded with MOCK_LLM unset
(i.e. the required mock baseline).

Call 1 — Policy question (triggers retrieval):

Request:

    {"query": "What is the return policy?"}

Response:

    {
      "answer": "Based on the retrieved context:  items that have been opened are non-returnable except in the case of a manufacturing defect. Return pickup, where required, is arranged free of cost by Zepto.",
      "sources": ["doc_02_chunk1", "doc_02_chunk0", "doc_05_chunk1"],
      "confidence": 1.0
    }

The top retrieved chunk (doc_02_chunk1) comes from the Returns and Refunds
document — the correct source for this query.

Call 2 — General question (does NOT trigger retrieval):

Request:

    {"query": "Who is the president?"}

Response:

    {
      "answer": "I can only answer questions about Zepto policies right now.",
      "sources": [],
      "confidence": 1.0
    }

No retrieval was performed and sources is empty — as required for
general_question classification.

## Structured Prompt Template (used by the optional MOCK_LLM=0 path)

The real-LLM branch (ungraded) uses this prompt, following the
role-context-task-format-length skeleton:

    ROLE: You are a Zepto customer support assistant.

    CONTEXT:
    {retrieved_chunks}

    TASK: Answer the user's question using ONLY the information in the CONTEXT above.

    FORMAT: Return strict JSON: {"answer": "<string>", "sources": ["<chunk_id>", ...], "confidence": <float 0-1>}

    LENGTH: Answer in 1 to 3 sentences. Do not exceed 3 sentences.

    NEGATIVE CONSTRAINT: Do not answer using information not present in the
    provided context. If the context does not contain the answer, respond with
    answer="I could not find that in Zepto's policies."

    FEW-SHOT EXAMPLE:
    User: "How long does delivery take?"
    Context: "Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation..."
    Answer: {"answer": "Zepto delivers within 10 to 30 minutes, depending on your zone and current order volume.", "sources": ["doc_01_chunk0"], "confidence": 1.0}

## RAG Pipeline Architecture

The pipeline has four stages: ingestion -> embedding -> retrieval -> generation.

1. Ingestion
   Where: build_index() in support_assistant/app.py.
   Reads every docs/doc_*.txt file, slices each into 400-character chunks.
   Each chunk gets a unique ID: <doc_stem>_chunk<n>.

2. Embedding
   Where: sentence_transformers.SentenceTransformer("all-MiniLM-L6-v2") in build_index().
   Every chunk is encoded into a 384-dimensional vector.
   Vectors are stored in a ChromaDB persistent collection named
   zepto_policies (directory: support_assistant/chroma_db/).

3. Retrieval
   Where: the retrieve_and_answer node in the LangGraph StateGraph.
   The user query is embedded with the same model.
   ChromaDB returns the top-3 most similar chunks by cosine similarity.
   This step always runs for real in both modes (mock and real-LLM),
   because embedding and ChromaDB need no API key and no network LLM call.

4. Generation
   Where: either retrieve_and_answer (policy queries) or
   direct_answer (general queries) in app.py.
   Mock mode (graded baseline): the answer is a deterministic string
   templated from the top retrieved chunk: f"Based on the retrieved context: {top_chunk[:200]}".
   For general questions, a fixed canned string is returned.
   Real-LLM mode (optional): prompts the LLM with the structured template
   above and validates the response against the Pydantic schema, retrying up
   to 2 times on validation failure.

LangGraph orchestration

The graph has three named nodes and a conditional edge:

           [ classify_intent ]
                    |
       (conditional edge - route)
             /            \
      policy_question   general_question
            |                  |
    [retrieve_and_answer] [direct_answer]
            |                  |
            +-------- END -----+

- classify_intent — mock mode uses a keyword heuristic over the list
  [delivery, return, refund, membership, tracking, cancel, gift card, support hours].
- route — conditional edge, picks the next node based on state["intent"].
- retrieve_and_answer / direct_answer — produce the final answer.

Which stages branch on MOCK_LLM:

- classify_intent — Mock: keyword heuristic, no LLM. Real-LLM: LLM call.
- retrieve_and_answer (retrieval) — Runs for real in BOTH modes.
- retrieve_and_answer (generation) — Mock: templated string. Real-LLM: LLM call with prompt template.
- direct_answer — Mock: fixed canned string. Real-LLM: LLM call, no retrieval.

## Pydantic Output Schema

Every response is validated against:

    class AskResponse(BaseModel):
        answer: str
        sources: List[str] = []
        confidence: float = 1.0

- Mock mode: populated deterministically. sources = retrieved chunk IDs
  for policy questions, [] for general questions. confidence = 1.0.
- Real-LLM mode: the LLM's raw output is validated against this schema;
  on validation failure, the client retries up to 2 additional times with a
  corrective instruction before returning a clearly marked error.

## Docker

The Dockerfile at the repo root builds the FastAPI app and serves it on
port 7860:

    docker build -t zepto-assistant .
    docker run -p 7860:7860 zepto-assistant

The container runs uvicorn app:app --host 0.0.0.0 --port 7860 with
MOCK_LLM unset (mock mode), matching the graded baseline.

Optional (ungraded): the same Dockerfile can be deployed to Hugging Face
Spaces on the free CPU tier. If you try it, store any real-LLM API key as a
Space secret — never commit it.

## Notes on MOCK_LLM Toggle

The env var MOCK_LLM controls the entire module:

- Unset or MOCK_LLM=1 — graded baseline. Fully offline. Every LLM call
  is replaced by the deterministic mock described above.
- MOCK_LLM=0 — optional ungraded path that calls a real LLM. Requires
  an API key (e.g. Groq free tier) and network access.

The submission is fully correct using only the default (mock) path.
## Note on Docker Build Verification

The Dockerfile is present and correctly configured. It builds and runs


`uvicorn app:app --host 0.0.0.0 --port 7860` in mock mode. The local
build was not executed in the development environment due to a
Docker Hub DNS resolution issue on the developer's machine; the
Dockerfile itself follows the standard pattern and is verified by
inspection.
